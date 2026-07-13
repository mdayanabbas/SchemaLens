from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.api.exception_handlers import register_exception_handlers
from app.core.exceptions import NotFoundError
from app.core.middleware import RequestIDMiddleware

app = FastAPI()
app.add_middleware(RequestIDMiddleware)
register_exception_handlers(app)


class DummyModel(BaseModel):
    value: int


@app.get("/error/app")
async def error_app() -> None:
    raise NotFoundError("Custom not found message")


@app.post("/error/validation")
async def error_validation(data: DummyModel) -> None:
    pass


@app.get("/error/unhandled")
async def error_unhandled() -> None:
    raise RuntimeError("Internal secret error")


client = TestClient(app)


def test_app_error() -> None:
    response = client.get("/error/app")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "NOT_FOUND"
    assert data["error"]["message"] == "Custom not found message"
    assert "request_id" in data


def test_validation_error() -> None:
    response = client.post("/error/validation", json={"value": "not-an-int"})
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert "request_id" in data
    errors = data["error"]["details"]["errors"]
    assert len(errors) > 0
    assert "input" not in errors[0]


def test_unhandled_error() -> None:
    response = client.get("/error/unhandled")
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert data["error"]["message"] == "An unexpected internal error occurred."
    assert "request_id" in data
    assert "Internal secret error" not in response.text
