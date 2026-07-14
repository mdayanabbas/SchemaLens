from unittest.mock import patch

from app.core.config import Settings
from app.db.engine import create_database_engine


@patch("app.db.engine.create_async_engine")
def test_create_database_engine(mock_create_async_engine) -> None:
    settings = Settings(
        database_url="postgresql+asyncpg://test:test@localhost/test",
        database_echo=True,
        database_pool_size=15,
        database_max_overflow=25,
        database_pool_timeout_seconds=45,
        database_pool_recycle_seconds=3600,
    )
    
    _ = create_database_engine(settings)
    
    mock_create_async_engine.assert_called_once_with(
        url="postgresql+asyncpg://test:test@localhost/test",
        echo=True,
        pool_size=15,
        max_overflow=25,
        pool_timeout=45,
        pool_recycle=3600,
        pool_pre_ping=True,
    )
