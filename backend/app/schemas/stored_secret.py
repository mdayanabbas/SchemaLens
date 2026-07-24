import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from app.secrets.enums import SecretStatus
from app.secrets.schemas import DatabaseCredentialPayload, _reject_control_chars


class StoredSecretCreate(BaseModel):
    name: str = Field(..., max_length=150)
    username: str = Field(..., max_length=255)
    password: SecretStr
    database: str | None = Field(None, max_length=100)
    host: str | None = Field(None, max_length=255)
    port: int | None = Field(None, ge=1, le=65535)
    ssl_ca: SecretStr | None = None
    ssl_cert: SecretStr | None = None
    ssl_key: SecretStr | None = None

    model_config = ConfigDict(hide_input_in_errors=True)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        v = _reject_control_chars(v)
        if not v:
            raise ValueError("Name must not be empty")
        return v
        
    def to_credential_payload(self) -> DatabaseCredentialPayload:
        return DatabaseCredentialPayload(
            username=self.username,
            password=self.password,
            database=self.database,
            host=self.host,
            port=self.port,
            ssl_ca=self.ssl_ca,
            ssl_cert=self.ssl_cert,
            ssl_key=self.ssl_key,
        )


class StoredSecretRotate(BaseModel):
    username: str = Field(..., max_length=255)
    password: SecretStr
    database: str | None = Field(None, max_length=100)
    host: str | None = Field(None, max_length=255)
    port: int | None = Field(None, ge=1, le=65535)
    ssl_ca: SecretStr | None = None
    ssl_cert: SecretStr | None = None
    ssl_key: SecretStr | None = None
    
    update_connection_ids: list[uuid.UUID] | None = Field(None, max_length=100)

    model_config = ConfigDict(hide_input_in_errors=True)

    def to_credential_payload(self) -> DatabaseCredentialPayload:
        return DatabaseCredentialPayload(
            username=self.username,
            password=self.password,
            database=self.database,
            host=self.host,
            port=self.port,
            ssl_ca=self.ssl_ca,
            ssl_cert=self.ssl_cert,
            ssl_key=self.ssl_key,
        )


class StoredSecretRead(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    status: SecretStatus
    provider: str
    reference: str
    key_version: str
    payload_version: int
    fields_present: list[str]
    created_by_user_id: uuid.UUID
    updated_by_user_id: uuid.UUID
    rotated_from_secret_id: uuid.UUID | None
    last_resolved_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class StoredSecretPage(BaseModel):
    items: list[StoredSecretRead]
    offset: int
    limit: int
    total: int
    has_more: bool
