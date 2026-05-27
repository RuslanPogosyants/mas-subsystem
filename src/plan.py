"""Plan and Subtask: build the DAG of subtasks for CoordinatorAgent.

VKR thesis section 2.3.7: the coordinator receives a Task, builds a plan
(intentions), publishes subtasks to the bus, awaits the results.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.schemas import DocumentType, Operation, Task

AgentName = Literal["transcriber", "ocr", "summarizer", "test_generator", "terminology", "recommender"]

REQUIRED_AGENTS: frozenset[AgentName] = frozenset({"transcriber", "ocr"})
OPTIONAL_AGENTS: frozenset[AgentName] = frozenset({"summarizer", "test_generator", "terminology", "recommender"})

OPERATION_TO_AGENT: dict[Operation, AgentName] = {
    Operation.F1_TRANSCRIBE: "transcriber",
    Operation.F2_OCR: "ocr",
    Operation.F3_SUMMARIZE: "summarizer",
    Operation.F4_TEST: "test_generator",
    Operation.F5_TERMS: "terminology",
    Operation.F6_RECOMMEND: "recommender",
}


class Subtask(BaseModel):
    """A single subtask within a Plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    agent: AgentName
    operation: Operation
    payload: dict[str, object] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    required: bool


class Plan(BaseModel):
    """Execution plan for a Task. Produced by `build_plan`.

    Subtask ids are deterministic in `(task_id, operation)` so the plan can be
    rebuilt identically after a Coordinator restart and existing Redis Stream
    replies still correlate to pending subtasks (spec section 8.7 recovery).
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    subtasks: list[Subtask]
    max_total_timeout_sec: float = 1800.0

    def _by_id(self) -> dict[str, Subtask]:
        return {subtask.id: subtask for subtask in self.subtasks}

    def get(self, subtask_id: str) -> Subtask:
        """Return the Subtask with the given id, or raise KeyError."""
        index = self._by_id()
        if subtask_id not in index:
            raise KeyError(f"unknown subtask id: {subtask_id!r}")
        return index[subtask_id]

    def is_required(self, subtask_id: str) -> bool:
        return self.get(subtask_id).required

    def is_optional(self, subtask_id: str) -> bool:
        return not self.is_required(subtask_id)

    def stage1(self) -> list[Subtask]:
        """Subtasks with no upstream dependencies (executed first)."""
        return [subtask for subtask in self.subtasks if not subtask.depends_on]

    def stage2(self) -> list[Subtask]:
        """Subtasks that wait for stage 1 results."""
        return [subtask for subtask in self.subtasks if subtask.depends_on]

    def all_stage1_done(self, results: dict[str, object | None]) -> bool:
        """True when every stage-1 subtask has a result (success or refuse)."""
        return all(subtask.id in results for subtask in self.stage1())


DEPENDENCY_MAP: Final[dict[Operation, Operation]] = {
    Operation.F4_TEST: Operation.F3_SUMMARIZE,
    Operation.F6_RECOMMEND: Operation.F5_TERMS,
}


def subtask_id_for(task_id: str, operation: Operation) -> str:
    """Deterministic subtask id; stable across Coordinator restarts."""
    return f"st-{task_id}-{operation.value}"


def _eligible_operations(task: Task) -> list[Operation]:
    """Filter requested operations against the documents actually present."""
    has_audio = any(doc.document_type == DocumentType.AUDIO for doc in task.documents)
    has_pdf_or_image = any(doc.document_type in (DocumentType.PDF, DocumentType.IMAGE) for doc in task.documents)
    eligible: list[Operation] = []
    for operation in task.requested_outputs:
        if operation == Operation.F1_TRANSCRIBE and not has_audio:
            continue
        if operation == Operation.F2_OCR and not has_pdf_or_image:
            continue
        eligible.append(operation)
    return eligible


def _dependencies(operation: Operation, eligible: set[Operation], task_id: str) -> list[str]:
    """Return the depends_on list for `operation`, filtered to eligible operations."""
    upstream = DEPENDENCY_MAP.get(operation)
    if upstream is None or upstream not in eligible:
        return []
    return [subtask_id_for(task_id, upstream)]


def build_plan(task: Task) -> Plan:
    """Build a Plan from a Task.

    F1 / F2 emit subtasks only when matching documents are attached. Optional
    operations (F3-F6) emit unconditionally; F4 depends on F3 and F6 on F5,
    when those upstream subtasks are also in the plan.

    Note: in M1 every Subtask is built with an empty `payload`. The Coordinator
    dispatch loop (M2) will inject the per-agent input (file_path for F1/F2,
    upstream Summary/Terms for F4/F6) at publish time.
    """
    eligible = _eligible_operations(task)
    eligible_set = set(eligible)
    subtasks = [
        Subtask(
            id=subtask_id_for(task.id, operation),
            agent=OPERATION_TO_AGENT[operation],
            operation=operation,
            depends_on=_dependencies(operation, eligible_set, task.id),
            required=OPERATION_TO_AGENT[operation] in REQUIRED_AGENTS,
        )
        for operation in eligible
    ]
    return Plan(task_id=task.id, subtasks=subtasks)
