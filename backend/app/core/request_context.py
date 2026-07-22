import contextvars
import uuid

from app.models.enums import OrganizationRole


_request_id_ctx_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
_user_id_ctx_var: contextvars.ContextVar[uuid.UUID | None] = contextvars.ContextVar(
    "user_id", default=None
)
_organization_id_ctx_var: contextvars.ContextVar[uuid.UUID | None] = contextvars.ContextVar(
    "organization_id", default=None
)
_membership_id_ctx_var: contextvars.ContextVar[uuid.UUID | None] = contextvars.ContextVar(
    "membership_id", default=None
)
_organization_role_ctx_var: contextvars.ContextVar[OrganizationRole | None] = contextvars.ContextVar(
    "organization_role", default=None
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


def set_user_context(user_id: uuid.UUID) -> contextvars.Token[uuid.UUID | None]:
    return _user_id_ctx_var.set(user_id)


def set_organization_context(
    organization_id: uuid.UUID,
    membership_id: uuid.UUID | None,
    role: OrganizationRole | None,
) -> tuple[
    contextvars.Token[uuid.UUID | None],
    contextvars.Token[uuid.UUID | None],
    contextvars.Token[OrganizationRole | None],
]:
    t1 = _organization_id_ctx_var.set(organization_id)
    t2 = _membership_id_ctx_var.set(membership_id)
    t3 = _organization_role_ctx_var.set(role)
    return t1, t2, t3


def get_current_user_id() -> uuid.UUID | None:
    return _user_id_ctx_var.get()


def get_current_organization_id() -> uuid.UUID | None:
    return _organization_id_ctx_var.get()


def get_current_membership_id() -> uuid.UUID | None:
    return _membership_id_ctx_var.get()


def get_current_organization_role() -> OrganizationRole | None:
    return _organization_role_ctx_var.get()


def clear_authorization_context() -> None:
    """Clear the authorization context."""
    _user_id_ctx_var.set(None)
    _organization_id_ctx_var.set(None)
    _membership_id_ctx_var.set(None)
    _organization_role_ctx_var.set(None)
