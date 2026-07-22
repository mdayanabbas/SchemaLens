import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.audit.enums import (
    AuditAction,
    AuditActorType,
    AuditEventSource,
    AuditOutcome,
    AuditResourceType,
)


class AuditEventCreate(BaseModel):
    """Internal schema for creating an audit event."""
    organization_id: uuid.UUID | None = None
    actor_user_id: uuid.UUID | None = None
    actor_type: AuditActorType
    action: AuditAction
    outcome: AuditOutcome
    resource_type: AuditResourceType
    resource_id: uuid.UUID | None = None
    request_id: str | None = Field(default=None, max_length=255)
    workflow_id: uuid.UUID | None = None
    source: AuditEventSource = AuditEventSource.API
    ip_hash: str | None = None
    user_agent_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime | None = None


class AuditEventRead(BaseModel):
    """Public schema for an audit event response."""
    id: uuid.UUID
    organization_id: uuid.UUID | None
    actor_user_id: uuid.UUID | None
    actor_type: AuditActorType
    action: AuditAction
    outcome: AuditOutcome
    resource_type: AuditResourceType
    resource_id: uuid.UUID | None
    request_id: str | None
    workflow_id: uuid.UUID | None
    event_version: int
    source: AuditEventSource
    metadata_json: dict[str, Any]
    occurred_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class AuditEventPage(BaseModel):
    items: list[AuditEventRead]
    offset: int
    limit: int
    total: int
    has_more: bool
