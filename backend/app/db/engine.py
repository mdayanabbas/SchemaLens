from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import Settings, get_settings


def create_database_engine(settings: Settings) -> AsyncEngine:
    """Create the SQLAlchemy async engine based on application settings."""
    return create_async_engine(
        url=settings.database_url,
        echo=settings.database_echo,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout_seconds,
        pool_recycle=settings.database_pool_recycle_seconds,
        pool_pre_ping=True,
    )


engine = create_database_engine(get_settings())
