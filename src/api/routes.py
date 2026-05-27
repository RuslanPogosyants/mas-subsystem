"""REST API routes. M0 placeholders; real implementation in M2-M4."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile

router = APIRouter(prefix="/api")


@router.post("/tasks", status_code=202)
async def create_task(files: list[UploadFile], ops: list[str]) -> dict[str, Any]:
    """Accept files and ops list, create a Task. Implemented in M2."""
    raise HTTPException(status_code=501, detail="create_task: implemented in M2")


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str) -> dict[str, Any]:
    """Return task status. Implemented in M2."""
    raise HTTPException(status_code=501, detail="get_task_status: implemented in M2")


@router.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str) -> dict[str, Any]:
    """Return ResultArtifact. Implemented in M3."""
    raise HTTPException(status_code=501, detail="get_task_result: implemented in M3")
