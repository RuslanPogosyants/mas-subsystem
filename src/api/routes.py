"""REST API routes wired to the Coordinator and persistence."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from src.core.schemas import Document, DocumentType, Operation, Task, TaskStatus
from src.db.repos import DocumentRepo, TaskRepo

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

router = APIRouter(prefix="/api")

UPLOAD_ROOT: Final[Path] = Path("data/uploads")
ALLOWED_AUDIO_SUFFIXES: Final[tuple[str, ...]] = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
ALLOWED_IMAGE_SUFFIXES: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
MAX_UPLOAD_BYTES: Final[int] = 200 * 1024 * 1024
UPLOAD_CHUNK_BYTES: Final[int] = 1 << 20


def _detect_document_type(upload: UploadFile) -> DocumentType:
    """Choose DocumentType from MIME type or filename suffix."""
    content_type = (upload.content_type or "").lower()
    name = (upload.filename or "").lower()
    if content_type.startswith("audio/") or name.endswith(ALLOWED_AUDIO_SUFFIXES):
        return DocumentType.AUDIO
    if content_type == "application/pdf" or name.endswith(".pdf"):
        return DocumentType.PDF
    if content_type.startswith("image/") or name.endswith(ALLOWED_IMAGE_SUFFIXES):
        return DocumentType.IMAGE
    return DocumentType.TEXT


def _safe_destination(task_dir: Path, original_name: str | None, index: int) -> Path:
    """Build a destination path inside `task_dir`, rejecting path-traversal attempts."""
    candidate = original_name or f"file-{index}"
    base_name = Path(candidate).name
    if not base_name or base_name in (".", ".."):
        base_name = f"file-{index}"
    destination = task_dir / f"{index:02d}-{base_name}"
    resolved_root = task_dir.resolve()
    if not destination.resolve().is_relative_to(resolved_root):
        raise HTTPException(status_code=400, detail="invalid filename")
    return destination


def _write_chunk(handle: Any, chunk: bytes) -> None:
    handle.write(chunk)


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    """Stream the upload to disk; enforces MAX_UPLOAD_BYTES, file I/O off-loop."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    handle = await asyncio.to_thread(destination.open, "wb")
    try:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="file exceeds size limit")
            await asyncio.to_thread(_write_chunk, handle, chunk)
    finally:
        await asyncio.to_thread(handle.close)


@router.post("/tasks", status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    request: Request,
    files: list[UploadFile] = File(...),
    ops: list[str] = Form(...),
) -> dict[str, str]:
    """Accept files and ops, create a Task, dispatch async, return 202."""
    if not files:
        raise HTTPException(status_code=400, detail="no files provided")
    try:
        operations = [Operation(op) for op in ops]
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"invalid operation: {error}") from error

    # Reject unsupported inputs up front (before any disk/DB write): only audio,
    # pdf, and image documents have an ingestion agent. Plain text has no chunker
    # yet, so accepting it would silently produce an empty artifact.
    document_types = [_detect_document_type(upload) for upload in files]
    unsupported = [
        upload.filename for upload, kind in zip(files, document_types, strict=False) if kind == DocumentType.TEXT
    ]
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported document type(s): {unsupported}; supported: audio, pdf, image",
        )

    task_id = f"task-{uuid.uuid4().hex[:8]}"
    task_dir = UPLOAD_ROOT / task_id

    session_factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    documents: list[Document] = []
    async with session_factory() as session:
        task_repo = TaskRepo(session)
        document_repo = DocumentRepo(session)
        await task_repo.create(task_id=task_id, requested_outputs=operations)
        for index, upload in enumerate(files):
            destination = _safe_destination(task_dir, upload.filename, index)
            await _save_upload(upload, destination)
            document_type = document_types[index]
            document_id = f"doc-{task_id}-{index}"
            await document_repo.create(
                document_id=document_id,
                task_id=task_id,
                document_type=document_type,
                file_path=str(destination),
                original_name=upload.filename,
            )
            documents.append(
                Document(
                    id=document_id,
                    task_id=task_id,
                    document_type=document_type,
                    file_path=str(destination),
                    original_name=upload.filename,
                )
            )
        await session.commit()

    task = Task(
        id=task_id,
        requested_outputs=operations,
        conversation_id=f"conv-{task_id}",
        documents=documents,
    )
    await request.app.state.coordinator.submit(task)
    return {"task_id": task_id, "status": TaskStatus.PLANNING.value}


@router.get("/tasks/{task_id}")
async def get_task_status(request: Request, task_id: str) -> dict[str, str]:
    session_factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with session_factory() as session:
        task_repo = TaskRepo(session)
        row = await task_repo.get(task_id)
        if row is None:
            raise HTTPException(status_code=404, detail="task not found")
        return {"task_id": task_id, "status": row.status}


@router.get("/tasks/{task_id}/result")
async def get_task_result(request: Request, task_id: str) -> dict[str, Any]:
    session_factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with session_factory() as session:
        task_repo = TaskRepo(session)
        row = await task_repo.get(task_id)
        if row is None:
            raise HTTPException(status_code=404, detail="task not found")
        if row.status in (TaskStatus.PLANNING.value, TaskStatus.RUNNING.value):
            raise HTTPException(status_code=425, detail="result not ready")
        if row.final_artifact is None:
            raise HTTPException(status_code=425, detail="result not ready")
        return row.final_artifact
