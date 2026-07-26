import functools
import json
from typing import Any

from pydantic import BeforeValidator, Field, field_validator
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
    database_pool_timeout_seconds: int = Field(default=30)
    database_pool_recycle_seconds: int = Field(default=1800)

    # Authentication
    jwt_secret_key: str
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=15)
    refresh_token_expire_days: int = Field(default=7)
    refresh_token_pepper: str
    password_min_length: int = Field(default=12)
    password_max_length: int = Field(default=128)
    authentication_issuer: str = Field(default="schemalens-api")
    authentication_audience: str = Field(default="schemalens-client")

    # Secrets
    local_secret_master_key: str | None = Field(default=None)
    local_secret_key_version: str = Field(default="v1")
    secret_value_max_bytes: int = Field(default=16384)
    secret_reference_max_length: int = Field(default=512)
    
    # AWS Secrets Manager
    aws_region: str | None = Field(default=None)
    aws_secrets_manager_endpoint_url: str | None = Field(default=None)
    aws_secrets_manager_timeout_seconds: int = Field(default=5)

    # Connector settings
    connector_pool_size: int = Field(default=3, ge=1, le=20)
    connector_max_overflow: int = Field(default=2, ge=0, le=20)
    connector_pool_timeout_seconds: int = Field(default=10, ge=1, le=60)
    connector_pool_recycle_seconds: int = Field(default=900, ge=60, le=3600)
    connector_connect_timeout_seconds: int = Field(default=10, ge=1, le=60)
    connector_test_statement_timeout_ms: int = Field(default=5000, ge=1000, le=30000)
    connector_test_lock_timeout_ms: int = Field(default=2000, ge=500, le=10000)
    connector_max_registered_pools: int = Field(default=50, ge=1, le=500)
    connector_pool_idle_ttl_seconds: int = Field(default=900, ge=60, le=3600)
    connector_application_name_prefix: str = Field(default="schemalens", pattern=r"^[a-zA-Z0-9_-]+$")

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
