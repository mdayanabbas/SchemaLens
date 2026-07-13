from typing import Any


class AppError(Exception):
    """Base application exception."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class NotFoundError(AppError):
    """Resource not found."""

    def __init__(
        self, message: str = "The requested resource was not found.", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            code="NOT_FOUND",
            message=message,
            status_code=404,
            details=details,
        )


class ConflictError(AppError):
    """Resource conflict."""

    def __init__(
        self, message: str = "A conflict occurred with the resource.", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            code="CONFLICT",
            message=message,
            status_code=409,
            details=details,
        )


class AuthenticationError(AppError):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="UNAUTHENTICATED",
            message=message,
            status_code=401,
            details=details,
        )


class AuthorizationError(AppError):
    """Authorization failed."""

    def __init__(
        self, message: str = "You do not have permission to access this resource.", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            code="UNAUTHORIZED",
            message=message,
            status_code=403,
            details=details,
        )


class ValidationAppError(AppError):
    """Validation failed."""

    def __init__(self, message: str = "Validation failed.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="VALIDATION_ERROR",
            message=message,
            status_code=422,
            details=details,
        )


class ExternalServiceError(AppError):
    """External service error."""

    def __init__(self, message: str = "An external service error occurred.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="EXTERNAL_SERVICE_ERROR",
            message=message,
            status_code=502,
            details=details,
        )


class PolicyViolationError(AppError):
    """Policy violation."""

    def __init__(self, message: str = "A policy violation occurred.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="POLICY_VIOLATION",
            message=message,
            status_code=403,
            details=details,
        )


class WorkflowStateError(AppError):
    """Workflow state error."""

    def __init__(
        self, message: str = "The action is invalid for the current workflow state.", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            code="WORKFLOW_STATE_ERROR",
            message=message,
            status_code=409,
            details=details,
        )
