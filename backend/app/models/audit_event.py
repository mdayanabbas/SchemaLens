import uuid
from datetime import datetime, UTC

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.audit.enums import (
    AuditAction,
    AuditActorType,
    AuditEventSource,
    AuditOutcome,
    AuditResourceType,
)
from app.db.base import Base
from app.db.mixins import UUIDPrimaryKeyMixin


class AuditEvent(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "audit_events"

    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"),
        index=True,
        nullable=True,
    )
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        index=True,
        nullable=True,
    )
    actor_type: Mapped[AuditActorType] = mapped_column(String, index=True, nullable=False)
    action: Mapped[AuditAction] = mapped_column(String, index=True, nullable=False)
    outcome: Mapped[AuditOutcome] = mapped_column(String, index=True, nullable=False)
    resource_type: Mapped[AuditResourceType] = mapped_column(String, index=True, nullable=False)
    resource_id: Mapped[uuid.UUID | None] = mapped_column(index=True, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    workflow_id: Mapped[uuid.UUID | None] = mapped_column(index=True, nullable=True)
    event_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    source: Mapped[AuditEventSource] = mapped_column(String, nullable=False)
    ip_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    user_agent_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("ix_audit_events_org_occurred", "organization_id", "occurred_at"),
        Index("ix_audit_events_actor_occurred", "actor_user_id", "occurred_at"),
        Index("ix_audit_events_action_occurred", "action", "occurred_at"),
        Index("ix_audit_events_resource_occurred", "resource_type", "resource_id", "occurred_at"),
        Index("ix_audit_events_workflow_occurred", "workflow_id", "occurred_at"),
    )
