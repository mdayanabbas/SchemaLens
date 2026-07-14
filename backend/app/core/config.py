import functools
import json
from typing import Any

from pydantic import BeforeValidator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated


def parse_cors_origins(value: Any) -> list[str]:
    """Parse CORS origins safely from a comma-separated string or a JSON-style list."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        return [str(item) for item in value]
    return []


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "SchemaLens"
    app_version: str = "0.1.0"
    app_environment: str = "local"
    app_debug: bool = False
    api_v1_prefix: str = "/api/v1"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_cors_origins: Annotated[list[str], BeforeValidator(parse_cors_origins)] = ["http://localhost:3000"]
    log_level: str = "INFO"

    # Database settings
    database_url: str = "postgresql+asyncpg://schemalens:schemalens@localhost:5432/schemalens"
    database_echo: bool = False
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout_seconds: int = 30
    database_pool_recycle_seconds: int = 1800

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("database_url must be non-empty")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@functools.lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
