from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.exception_handlers import register_exception_handlers
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.constants import SERVICE_NAME
from app.core.logging import configure_logging
from app.core.middleware import RequestIDMiddleware
from app.schemas.health import LivenessResponse


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    configure_logging(settings)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.app_debug,
    )

    application.add_middleware(RequestIDMiddleware)

    if settings.backend_cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.backend_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )

    register_exception_handlers(application)

    @application.get("/health", response_model=LivenessResponse, tags=["health"])
    async def liveness_probe() -> LivenessResponse:
        """Process-level liveness probe."""
        return LivenessResponse(status="ok", service=SERVICE_NAME)

    application.include_router(api_router, prefix=settings.api_v1_prefix)

    return application


app = create_application()
