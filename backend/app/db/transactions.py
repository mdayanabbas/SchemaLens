from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppError


@asynccontextmanager
async def transactional(
    session: AsyncSession,
) -> AsyncIterator[AsyncSession]:
    """Execute a block within a database transaction."""
    if session.in_transaction():
        raise AppError(
            code="TRANSACTION_ERROR",
            message="A transaction is already active. Nested transactions are not allowed.",
            status_code=500,
        )

    async with session.begin():
        yield session
