import asyncio
import time

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


class DatabaseHealthResult(BaseModel):
    """Database health check result."""

    status: str = Field(..., description="ok or error")
    latency_ms: float = Field(..., description="Latency in milliseconds")
    error_code: str | None = Field(None, description="Safe error code if failed")


async def check_database_health(engine: AsyncEngine) -> DatabaseHealthResult:
    """Check database health by executing a simple SELECT 1 ping."""
    start_time = time.perf_counter()
    status = "ok"
    error_code = None

    try:
        async with asyncio.timeout(3.0):
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
    except TimeoutError:
        status = "error"
        error_code = "TIMEOUT"
    except Exception:
        status = "error"
        error_code = "CONNECTION_FAILED"

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return DatabaseHealthResult(
        status=status,
        latency_ms=round(latency_ms, 2),
        error_code=error_code,
    )
