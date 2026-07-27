import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.audit.enums import AuditActorType
from app.core.config import get_settings
from app.models.schema_scan_enums import (
    SchemaScanFailureStage,
    SchemaScanStatus,
    SchemaScanTrigger,
)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1F\x7F]")


class SchemaScanCreate(BaseModel):
    requested_schemas: list[str] | None = Field(default=None)
    force: bool = Field(default=False)

    @field_validator("requested_schemas")
    @classmethod
    def validate_requested_schemas(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
            
        settings = get_settings()
        
        normalized: list[str] = []
        seen: set[str] = set()
        
        for schema_name in v:
            trimmed = schema_name.strip()
            if not trimmed:
                raise ValueError("Schema names cannot be empty")
            if _CONTROL_CHAR_RE.search(trimmed):
                raise ValueError("Schema names cannot contain control characters")
                
            if trimmed not in seen:
                seen.add(trimmed)
                normalized.append(trimmed)
                
        if len(normalized) > settings.schema_scan_max_requested_schemas:
            raise ValueError(f"Requested schemas cannot exceed {settings.schema_scan_max_requested_schemas}")
            
        return normalized


class SchemaScanRead(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    connection_id: uuid.UUID
    requested_by_user_id: uuid.UUID
    trigger: SchemaScanTrigger
    status: SchemaScanStatus
    requested_schemas: list[str] = Field(alias="requested_schemas_json")
    attempt_count: int
    max_attempts: int
    progress_phase: str | None
    progress_current: int
    progress_total: int
    discovered_object_count: int
    successful_object_count: int
    failed_object_count: int
    warning_count: int
    cancellation_requested_at: datetime | None
    started_at: datetime | None
    heartbeat_at: datetime | None
    completed_at: datetime | None
    failure_stage: SchemaScanFailureStage | None
    safe_error_code: str | None
    safe_error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class SchemaScanSummaryRead(BaseModel):
    id: uuid.UUID
    connection_id: uuid.UUID
    status: SchemaScanStatus
    progress_phase: str | None
    requested_schemas: list[str] = Field(alias="requested_schemas_json")
    started_at: datetime | None
    completed_at: datetime | None
    safe_error_code: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class SchemaScanPage(BaseModel):
    items: list[SchemaScanSummaryRead]
    offset: int
    limit: int
    total: int
    has_more: bool


class SchemaScanCancelResponse(BaseModel):
    id: uuid.UUID
    status: SchemaScanStatus
    cancellation_requested_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class SchemaScanTransitionRead(BaseModel):
    id: uuid.UUID
    from_status: SchemaScanStatus | None
    to_status: SchemaScanStatus
    actor_type: AuditActorType | None
    actor_user_id: uuid.UUID | None
    reason_code: str
    safe_metadata: dict[str, Any] | None = Field(alias="safe_metadata_json")
    occurred_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
