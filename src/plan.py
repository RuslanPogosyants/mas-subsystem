"""Plan and Subtask: build the DAG of subtasks for CoordinatorAgent.

VKR thesis section 2.3.7: the coordinator receives a Task, builds a plan
(intentions), publishes subtasks to the bus, awaits the results.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.schemas import Operation, Task

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


def build_plan(task: Task) -> Plan:
    """Build a Plan from a Task. Stub for M0; real implementation in M2."""
    raise NotImplementedError("build_plan: implemented in M2")
