"""Contract tests: FastAPI OpenAPI specification matches design section 5."""

from __future__ import annotations

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestOpenAPIEndpoints:
    def test_openapi_json_available(self) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.json()["openapi"].startswith("3.")

    def test_post_tasks_endpoint_declared(self) -> None:
        spec = client.get("/openapi.json").json()
        assert "/api/tasks" in spec["paths"]
        assert "post" in spec["paths"]["/api/tasks"]

    def test_get_task_status_endpoint_declared(self) -> None:
        spec = client.get("/openapi.json").json()
        assert "/api/tasks/{task_id}" in spec["paths"]
        assert "get" in spec["paths"]["/api/tasks/{task_id}"]

    def test_get_task_result_endpoint_declared(self) -> None:
        spec = client.get("/openapi.json").json()
        assert "/api/tasks/{task_id}/result" in spec["paths"]
        assert "get" in spec["paths"]["/api/tasks/{task_id}/result"]

    def test_root_health_check(self) -> None:
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "mas-research"
        assert body["status"] == "ok"
