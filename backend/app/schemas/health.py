from datetime import datetime

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    """Liveness health check response."""

    status: str = Field(..., description="The status of the service")
    service: str = Field(..., description="The name of the service")


class ServiceHealthResponse(BaseModel):
    """Detailed service health check response."""

    status: str = Field(..., description="The status of the service")
    service: str = Field(..., description="The name of the service")
    version: str = Field(..., description="The version of the service")
    environment: str = Field(..., description="The current environment")
    timestamp: datetime = Field(..., description="The current UTC time")
