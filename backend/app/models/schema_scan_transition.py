from datetime import datetime, timezone
from typing import Any
import uuid

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.audit.enums import AuditActorType
from app.db.base_class import Base
from app.db.mixins import UUIDPrimaryKeyMixin
from app.models.schema_scan_enums import SchemaScanStatus


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class SchemaScanTransition(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "schema_scan_transitions"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), index=True, nullable=False
    )
    schema_scan_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_scans.id", ondelete="RESTRICT"), index=True, nullable=False
    )

    from_status: Mapped[SchemaScanStatus | None] = mapped_column(String(50), nullable=True)
    to_status: Mapped[SchemaScanStatus] = mapped_column(String(50), nullable=False)

    actor_type: Mapped[AuditActorType | None] = mapped_column(String(50), nullable=True)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=True
    )

    reason_code: Mapped[str] = mapped_column(String(100), nullable=False)
    safe_metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now_utc, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now_utc, nullable=False
    )

    __table_args__ = (
        Index("ix_scan_transitions_scan_occurred", "schema_scan_id", "occurred_at"),
        Index("ix_scan_transitions_org_occurred", "organization_id", "occurred_at"),
    )
