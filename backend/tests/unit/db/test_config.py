import os
import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_database_defaults_load_correctly() -> None:
    get_settings.cache_clear()
    settings = Settings()
    assert settings.database_url == "postgresql+asyncpg://schemalens:schemalens@localhost:5432/schemalens"
    assert settings.database_echo is False
    assert settings.database_pool_size == 10
    assert settings.database_max_overflow == 20
    assert settings.database_pool_timeout_seconds == 30
    assert settings.database_pool_recycle_seconds == 1800


def test_database_environment_override() -> None:
    get_settings.cache_clear()
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test"
    os.environ["DATABASE_ECHO"] = "true"
    settings = Settings()
    assert settings.database_url == "postgresql+asyncpg://test:test@localhost:5432/test"
    assert settings.database_echo is True
    os.environ.pop("DATABASE_URL")
    os.environ.pop("DATABASE_ECHO")


def test_empty_database_url_rejected() -> None:
    get_settings.cache_clear()
    os.environ["DATABASE_URL"] = "   "
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    
    error_msg = str(exc_info.value)
    assert "database_url must be non-empty" in error_msg
    # Ensure no passwords or anything weird leaked
    assert "schemalens" not in error_msg
    
    os.environ.pop("DATABASE_URL")
