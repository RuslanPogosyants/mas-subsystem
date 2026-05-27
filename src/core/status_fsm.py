"""Final task status FSM.

Decides the terminal status based on which subtasks succeeded or failed,
honouring the required vs optional split defined by the plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.schemas import TaskStatus
    from src.plan import Plan


def determine_final_status(plan: Plan, results: dict[str, Any | None]) -> TaskStatus:
    """Determine the terminal status of a task.

    Rules:
        - All required and all optional succeeded -> COMPLETED
        - Any required failed -> FAILED
        - All required succeeded but some optional failed -> PARTIAL_READY
    """
    raise NotImplementedError("determine_final_status: implemented in M1")
