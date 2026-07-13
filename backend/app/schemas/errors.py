from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed information about an error."""

    code: str = Field(..., description="A constant error code")
    message: str = Field(..., description="A human-readable error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional context about the error")


class ErrorResponse(BaseModel):
    """Standardized API error response format."""

    error: ErrorDetail
    request_id: str | None = Field(None, description="The request ID associated with the error")
