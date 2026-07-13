from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.constants import REQUEST_ID_HEADER
from app.core.middleware import RequestIDMiddleware
from app.core.request_context import get_request_id

app = FastAPI()
app.add_middleware(RequestIDMiddleware)


@app.get("/test")
async def test_endpoint() -> dict[str, str | None]:
    return {"request_id": get_request_id()}


client = TestClient(app)


def test_request_id_generated() -> None:
    response = client.get("/test")
    assert response.status_code == 200
    req_id = response.headers.get(REQUEST_ID_HEADER)
    assert req_id is not None
    assert len(req_id) > 0
    assert response.json() == {"request_id": req_id}


def test_request_id_preserved() -> None:
    response = client.get("/test", headers={REQUEST_ID_HEADER: "my-safe-id"})
    assert response.status_code == 200
    assert response.headers.get(REQUEST_ID_HEADER) == "my-safe-id"
    assert response.json() == {"request_id": "my-safe-id"}


def test_empty_request_id_replaced() -> None:
    response = client.get("/test", headers={REQUEST_ID_HEADER: "   "})
    assert response.status_code == 200
    req_id = response.headers.get(REQUEST_ID_HEADER)
    assert req_id is not None
    assert req_id != ""
    assert req_id != "   "


def test_excessively_long_request_id_replaced() -> None:
    long_id = "a" * 150
    response = client.get("/test", headers={REQUEST_ID_HEADER: long_id})
    assert response.status_code == 200
    req_id = response.headers.get(REQUEST_ID_HEADER)
    assert req_id is not None
    assert req_id != long_id
