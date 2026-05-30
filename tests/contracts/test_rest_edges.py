"""Additional REST edge cases not covered by the happy-path contract."""

from __future__ import annotations

import io

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from src.api.routes import _detect_document_type, _safe_destination
from src.core.schemas import DocumentType
from src.main import app


@pytest.mark.e2e
def test_post_invalid_operation_returns_400() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/tasks",
            files=[("files", ("a.mp3", b"x", "audio/mpeg"))],
            data={"ops": ["F9"]},
        )
        assert response.status_code == 400
        assert "invalid operation" in response.json()["detail"].lower()


@pytest.mark.e2e
def test_get_status_404_for_unknown_task() -> None:
    with TestClient(app) as client:
        response = client.get("/api/tasks/task-missing")
        assert response.status_code == 404


def test_detect_document_type_image_by_suffix() -> None:
    upload = UploadFile(filename="scan.PNG", file=io.BytesIO(b""))
    assert _detect_document_type(upload) == DocumentType.IMAGE


def test_safe_destination_falls_back_on_empty_name(tmp_path) -> None:
    destination = _safe_destination(tmp_path, "", 3)
    assert destination.name == "03-file-3"


def test_safe_destination_strips_traversal_to_basename(tmp_path) -> None:
    # _safe_destination uses Path(name).name, stripping all directory components.
    # A traversal like "../../etc/passwd" becomes "passwd", so the result stays
    # inside task_dir and no exception is raised.
    destination = _safe_destination(tmp_path, "../../etc/passwd", 0)
    assert destination.resolve().is_relative_to(tmp_path.resolve())
