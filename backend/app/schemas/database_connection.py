import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.connection_enums import (
    ConnectionEnvironment,
    ConnectionStatus,
    ConnectionTestStatus,
    DatabaseDialect,
    SecretProviderType,
    SSLMode,
)
from app.services.connection_validation import (
    validate_connection_name,
    validate_database_name,
    validate_host,
    validate_secret_reference,
)


class DatabaseConnectionCreate(BaseModel):
    name: str = Field(..., max_length=150)
    description: str | None = Field(None, max_length=1000)
    environment: ConnectionEnvironment
    dialect: DatabaseDialect
    host: str = Field(..., max_length=255)
    port: int = Field(5432, ge=1, le=65535)
    database_name: str = Field(..., max_length=100)
    default_catalog: str | None = Field(None, max_length=100)
    ssl_mode: SSLMode = SSLMode.REQUIRE
    secret_provider: SecretProviderType
    secret_reference: str = Field(..., max_length=500)

    @model_validator(mode="after")
    def validate_fields(self):
        self.name = validate_connection_name(self.name)
        self.host = validate_host(self.host)
        self.database_name = validate_database_name(self.database_name)
        self.secret_reference = validate_secret_reference(self.secret_reference)
        # Note: Do not silently convert dialects. The enum takes care of ensuring it's valid.
        return self


class DatabaseConnectionUpdate(BaseModel):
    name: str | None = Field(None, max_length=150)
    description: str | None = Field(None, max_length=1000)
    environment: ConnectionEnvironment | None = None
    host: str | None = Field(None, max_length=255)
    port: int | None = Field(None, ge=1, le=65535)
    database_name: str | None = Field(None, max_length=100)
    default_catalog: str | None = Field(None, max_length=100)
    ssl_mode: SSLMode | None = None
    secret_provider: SecretProviderType | None = None
    secret_reference: str | None = Field(None, max_length=500)
    status: ConnectionStatus | None = None
    
    # Dialect change is not permitted.

    @model_validator(mode="after")
    def validate_fields(self):
        if self.name is not None:
            self.name = validate_connection_name(self.name)
        if self.host is not None:
            self.host = validate_host(self.host)
        if self.database_name is not None:
            self.database_name = validate_database_name(self.database_name)
        if self.secret_reference is not None:
            self.secret_reference = validate_secret_reference(self.secret_reference)
        return self


class DatabaseConnectionRead(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    description: str | None
    environment: ConnectionEnvironment
    dialect: DatabaseDialect
    host: str
    port: int
    database_name: str
    default_catalog: str | None
    ssl_mode: SSLMode
    secret_provider: SecretProviderType
    redacted_secret_reference: str
    status: ConnectionStatus
    last_tested_at: datetime | None
    last_test_status: ConnectionTestStatus
    last_test_error_code: str | None
    created_by_user_id: uuid.UUID
    updated_by_user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatabaseConnectionSummaryRead(BaseModel):
    id: uuid.UUID
    name: str
    environment: ConnectionEnvironment
    dialect: DatabaseDialect
    host: str
    port: int
    database_name: str
    status: ConnectionStatus
    last_test_status: ConnectionTestStatus
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
