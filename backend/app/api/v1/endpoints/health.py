from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.constants import SERVICE_NAME
from app.schemas.health import ServiceHealthResponse

router = APIRouter()


@router.get("/health", response_model=ServiceHealthResponse)
async def service_health() -> ServiceHealthResponse:
    """Get detailed service health status."""
    settings = get_settings()
    return ServiceHealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=settings.app_version,
        environment=settings.app_environment,
        timestamp=datetime.now(tz=timezone.utc),
    )
