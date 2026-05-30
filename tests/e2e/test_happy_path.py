"""E2E acceptance: full pipeline happy path."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from src.main import app


@pytest.mark.e2e
class TestHappyPath:
    async def _wait_until_complete(self, client: AsyncClient, task_id: str, timeout: float = 30.0) -> dict[str, Any]:
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

    async def test_full_pipeline_returns_complete_artifact(self) -> None:
        async with (
            LifespanManager(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        ):
            response = await client.post(
                "/api/tasks",
                files=[
                    ("files", ("lecture.mp3", b"FAKE_AUDIO", "audio/mpeg")),
                    ("files", ("paper1.pdf", b"FAKE_PDF", "application/pdf")),
                    ("files", ("paper2.pdf", b"FAKE_PDF", "application/pdf")),
                ],
                data={"ops": ["F1", "F2", "F3", "F4", "F5", "F6"]},
            )
            assert response.status_code == 202
            task_id = response.json()["task_id"]

            final = await self._wait_until_complete(client, task_id)
            assert final["status"] == "completed"

            result_resp = await client.get(f"/api/tasks/{task_id}/result")
            assert result_resp.status_code == 200
            artifact = result_resp.json()

            assert artifact["status"] == "completed"
            assert set(artifact["operations"]) == {
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
            }
            assert artifact["result"]["summary"] is not None
            assert len(artifact["result"]["summary"]["sections"]) >= 1
            assert len(artifact["result"]["terms"]) > 0
            assert len(artifact["result"]["quiz"]) > 0
            assert len(artifact["result"]["citations"]) > 0
            assert artifact["degraded"] == []
            assert artifact["stats"]["agents_called"] == 7
            assert len(artifact["stats"]["failed_operations"]) == 0
