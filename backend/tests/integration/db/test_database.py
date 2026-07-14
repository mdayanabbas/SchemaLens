import os
import pytest
from sqlalchemy import text

from app.core.config import Settings
from app.db.engine import create_database_engine
from app.db.session import async_session_factory
from app.db.transactions import transactional

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("TEST_DATABASE_URL"),
        reason="TEST_DATABASE_URL is not set",
    )
]


@pytest.fixture
async def integration_engine():
    settings = Settings(database_url=os.getenv("TEST_DATABASE_URL", ""))
    engine = create_database_engine(settings)
    yield engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_database_select_1(integration_engine) -> None:
    async with integration_engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        row = result.scalar()
        assert row == 1


@pytest.mark.asyncio
async def test_session_commit_and_rollback(integration_engine) -> None:
    async_session_factory.configure(bind=integration_engine)
    
    async with async_session_factory() as session:
        async with transactional(session):
            await session.execute(text("SELECT 1"))
            
    assert session.is_active is False
