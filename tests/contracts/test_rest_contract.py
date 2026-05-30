"""Behavioral contract for REST endpoints, with lifespan-initialised app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.mark.e2e
class TestPostTasksContract:
    def test_post_tasks_returns_202_with_task_id(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/api/tasks",
                files=[("files", ("lecture.mp3", b"audio-bytes", "audio/mpeg"))],
                data={"ops": ["F1"]},
            )
            assert response.status_code == 202
            body = response.json()
            assert "task_id" in body
            assert body["task_id"].startswith("task-")
            assert body["status"] == "planning"


@pytest.mark.e2e
class TestGetResultContract:
    def test_get_result_returns_425_if_not_ready(self) -> None:
        with TestClient(app) as client:
            post = client.post(
                "/api/tasks",
                files=[("files", ("lecture.mp3", b"audio-bytes", "audio/mpeg"))],
                data={"ops": ["F1"]},
            )
            task_id = post.json()["task_id"]
            response = client.get(f"/api/tasks/{task_id}/result")
            # Immediately after POST the task is planning/running, hence 425.
            assert response.status_code == 425

    def test_get_result_returns_404_for_unknown_task(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/tasks/task-nonexistent/result")
            assert response.status_code == 404


@pytest.mark.e2e
class TestUnsupportedDocumentTypeContract:
    def test_post_unsupported_type_returns_400(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/api/tasks",
                files=[("files", ("notes.txt", b"some lecture text", "text/plain"))],
                data={"ops": ["F3"]},
            )
            assert response.status_code == 400
            assert "unsupported" in response.json()["detail"].lower()
