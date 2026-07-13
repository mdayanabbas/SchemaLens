import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.constants import REQUEST_ID_HEADER
from app.core.request_context import clear_request_id, set_request_id

MAX_REQUEST_ID_LENGTH = 128


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to inject or generate a request ID."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER, "").strip()
        if not request_id or len(request_id) > MAX_REQUEST_ID_LENGTH:
            request_id = str(uuid.uuid4())

        token = set_request_id(request_id)
        request.state.request_id = request_id

        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            clear_request_id(token)
