from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_liveness_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "SchemaLens"


def test_service_health() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "SchemaLens"
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data
