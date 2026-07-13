import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError
from app.core.request_context import get_request_id
from app.schemas.errors import ErrorDetail, ErrorResponse

logger = structlog.get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers for the application."""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        response_model = ErrorResponse(
            error=ErrorDetail(
                code=exc.code,
                message=exc.message,
                details=exc.details,
            ),
            request_id=get_request_id(),
        )
        return JSONResponse(status_code=exc.status_code, content=response_model.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        safe_details = []
        for error in errors:
            safe_details.append({
                "loc": error.get("loc", []),
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
            })

        response_model = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Validation failed.",
                details={"errors": safe_details},
            ),
            request_id=get_request_id(),
        )
        return JSONResponse(status_code=422, content=response_model.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception occurred")
        response_model = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected internal error occurred.",
                details={},
            ),
            request_id=get_request_id(),
        )
        return JSONResponse(status_code=500, content=response_model.model_dump())
