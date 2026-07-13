import contextvars

_request_id_ctx_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def set_request_id(request_id: str) -> contextvars.Token[str | None]:
    """Set the request ID for the current context."""
    return _request_id_ctx_var.set(request_id)


def get_request_id() -> str | None:
    """Get the request ID from the current context."""
    return _request_id_ctx_var.get()


def clear_request_id(token: contextvars.Token[str | None]) -> None:
    """Clear the request ID using the token returned by set_request_id."""
    _request_id_ctx_var.reset(token)
