"""E2E acceptance tests for S8 graceful degradation."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from src.main import app


async def _wait_until_complete(client: AsyncClient, task_id: str, timeout: float = 60.0) -> dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        response = await client.get(f"/api/tasks/{task_id}")
        if response.status_code == 200 and response.json()["status"] in (
            "completed",
            "partial_ready",
            "failed",
        ):
            return response.json()
        await asyncio.sleep(0.5)
    raise TimeoutError(f"task {task_id} did not complete in {timeout}s")


@pytest.mark.e2e
class TestS8aForceRefuseF6:
    async def test_force_refuse_f6_returns_partial(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORCE_REFUSE", "F6")

        async with (
            LifespanManager(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        ):
            response = await client.post(
                "/api/tasks",
                files=[
                    ("files", ("lecture.mp3", b"FAKE", "audio/mpeg")),
                    ("files", ("paper.pdf", b"FAKE", "application/pdf")),
                ],
                data={"ops": ["F1", "F2", "F3", "F4", "F5", "F6"]},
            )
            assert response.status_code == 202
            task_id = response.json()["task_id"]

            await _wait_until_complete(client, task_id)
            result = (await client.get(f"/api/tasks/{task_id}/result")).json()

            assert result["status"] == "partial_ready"
            assert "F6" not in result["operations"]
            assert result["degraded"] == ["F6"]
            assert result["result"]["citations"] == []
            assert len(result["stats"]["failed_operations"]) == 1

            failed = result["stats"]["failed_operations"][0]
            assert failed["op"] == "F6"
            assert failed["agent"] == "RecommenderAgent"
            assert "refuse" in failed["reason"].lower()
            assert failed["retries"] == 2


@pytest.mark.e2e
class TestS8bHangTimeoutF6:
    async def test_hang_f6_returns_partial_after_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HANG_AGENT", "F6")
        monkeypatch.setenv("COORD_TIMEOUT_RECOMMENDER", "2")

        async with (
            LifespanManager(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        ):
            response = await client.post(
                "/api/tasks",
                files=[("files", ("lecture.mp3", b"FAKE", "audio/mpeg"))],
                data={"ops": ["F1", "F3", "F5", "F6"]},
            )
            assert response.status_code == 202
            task_id = response.json()["task_id"]

            result_meta = await _wait_until_complete(client, task_id, timeout=30)
            assert result_meta["status"] == "partial_ready"

            result = (await client.get(f"/api/tasks/{task_id}/result")).json()
            assert result["status"] == "partial_ready"
            assert result["degraded"] == ["F6"]

            failed = result["stats"]["failed_operations"][0]
            assert "timeout" in failed["reason"].lower()
            assert failed["retries"] == 2
            assert 3.0 <= failed["elapsed_sec"] <= 20.0
