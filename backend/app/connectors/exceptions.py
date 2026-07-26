from typing import Any

from app.core.exceptions import AppError


class ConnectorError(AppError):
    """Base class for all connector exceptions."""

    def __init__(
        self,
        code: str = "DATABASE_CONNECTION_TEST_FAILED",
        message: str = "A database connection error occurred.",
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            status_code=status_code,
            details=details,
        )


class ConnectorConfigurationError(ConnectorError):
    def __init__(self, message: str = "Invalid connector configuration.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="CONNECTOR_CONFIGURATION_ERROR",
            message=message,
            status_code=422,
            details=details,
        )


class ConnectorUnavailableError(ConnectorError):
    def __init__(self, message: str = "The target database is unreachable.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="DATABASE_UNREACHABLE",
            message=message,
            status_code=502,
            details=details,
        )


class ConnectorAuthenticationError(ConnectorError):
    def __init__(self, message: str = "Database authentication failed.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="DATABASE_AUTHENTICATION_FAILED",
            message=message,
            status_code=401,
            details=details,
        )


class ConnectorAuthorizationError(ConnectorError):
    def __init__(self, message: str = "Database access denied.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="DATABASE_ACCESS_DENIED",
            message=message,
            status_code=403,
            details=details,
        )


class ConnectorTimeoutError(ConnectorError):
    def __init__(self, message: str = "Database connection timed out.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="DATABASE_CONNECTION_TIMEOUT",
            message=message,
            status_code=504,
            details=details,
        )


class ConnectorSSLError(ConnectorError):
    def __init__(self, message: str = "SSL connection failed.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="DATABASE_SSL_FAILED",
            message=message,
            status_code=502,
            details=details,
        )


class ConnectorDatabaseNotFoundError(ConnectorError):
    def __init__(self, message: str = "Target database not found.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="DATABASE_NOT_FOUND",
            message=message,
            status_code=404,
            details=details,
        )


class ConnectorSchemaNotFoundError(ConnectorError):
    def __init__(self, message: str = "Approved schema not found.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="APPROVED_SCHEMA_NOT_FOUND",
            message=message,
            status_code=404,
            details=details,
        )


class ConnectorReadOnlyVerificationError(ConnectorError):
    def __init__(self, message: str = "Could not verify read-only role.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="READ_ONLY_ROLE_NOT_VERIFIED",
            message=message,
            status_code=403,
            details=details,
        )


class ConnectorUnsupportedError(ConnectorError):
    def __init__(self, message: str = "Connector dialect not supported.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="CONNECTOR_NOT_SUPPORTED",
            message=message,
            status_code=400,
            details=details,
        )
