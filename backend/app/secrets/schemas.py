import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from app.core.exceptions import ValidationError


def _reject_connection_url(value: str) -> str:
    if "://" in value:
        raise ValueError("Must not be a connection URL")
    return value

def _reject_control_chars(value: str) -> str:
    if re.search(r"[\x00-\x1F\x7F]", value):
        raise ValueError("Must not contain control characters")
    return value


class DatabaseCredentialPayload(BaseModel):
    """
    Internal validated schema for resolved database credentials.
    """
    username: str = Field(..., max_length=255)
    password: SecretStr
    database: str | None = Field(None, max_length=100)
    host: str | None = Field(None, max_length=255)
    port: int | None = Field(None, ge=1, le=65535)
    ssl_ca: SecretStr | None = Field(None)
    ssl_cert: SecretStr | None = Field(None)
    ssl_key: SecretStr | None = Field(None)
    expires_at: datetime | None = None

    model_config = ConfigDict(
        extra="forbid",
        hide_input_in_errors=True
    )

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        v = _reject_control_chars(v)
        v = _reject_connection_url(v)
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value():
            raise ValueError("Password must not be empty")
        return v

    @field_validator("database")
    @classmethod
    def validate_database(cls, v: str | None) -> str | None:
        if v is not None:
            v = _reject_connection_url(v)
        return v

    @field_validator("ssl_ca", "ssl_cert", "ssl_key", mode="before")
    @classmethod
    def validate_cert_size(cls, v: Any) -> Any:
        if isinstance(v, str) and len(v) > 32768:
            raise ValueError("Certificate or key too large")
        return v
