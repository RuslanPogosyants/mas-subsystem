"""Plan and Subtask: build the DAG of subtasks for CoordinatorAgent.

VKR thesis section 2.3.7: the coordinator receives a Task, builds a plan
(intentions), publishes subtasks to the bus, awaits the results.
"""

from __future__ import annotations

import uuid
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
    """Execution plan for a Task. Produced by `build_plan`."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    subtasks: list[Subtask]
    max_total_timeout_sec: float = 1800.0

    def is_required(self, subtask_id: str) -> bool:
        return any(subtask.required for subtask in self.subtasks if subtask.id == subtask_id)

    def is_optional(self, subtask_id: str) -> bool:
        return any(not subtask.required for subtask in self.subtasks if subtask.id == subtask_id)

    def get(self, subtask_id: str) -> Subtask | None:
        return next((subtask for subtask in self.subtasks if subtask.id == subtask_id), None)


DEPENDENCY_MAP: Final[dict[Operation, Operation]] = {
    Operation.F4_TEST: Operation.F3_SUMMARIZE,
    Operation.F6_RECOMMEND: Operation.F5_TERMS,
}


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


def build_plan(task: Task) -> Plan:
    """Build a Plan from a Task.

    F1 / F2 emit subtasks only when matching documents are attached. Optional
    operations (F3-F6) emit unconditionally; F4 depends on F3 and F6 on F5,
    when those upstream subtasks are also in the plan.
    """
    eligible = _eligible_operations(task)
    operation_to_id: dict[Operation, str] = {operation: f"st-{uuid.uuid4().hex[:8]}" for operation in eligible}
    subtasks = [
        Subtask(
            id=operation_to_id[operation],
            agent=OPERATION_TO_AGENT[operation],
            operation=operation,
            depends_on=(
                [operation_to_id[DEPENDENCY_MAP[operation]]] if DEPENDENCY_MAP.get(operation) in operation_to_id else []
            ),
            required=OPERATION_TO_AGENT[operation] in REQUIRED_AGENTS,
        )
        for operation in eligible
    ]
    return Plan(task_id=task.id, subtasks=subtasks)
