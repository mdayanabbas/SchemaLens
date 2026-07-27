from datetime import datetime
from typing import Any
import uuid

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base_class import Base
from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin
from app.models.schema_scan_enums import (
    SchemaScanFailureStage,
    SchemaScanStatus,
    SchemaScanTrigger,
)


class SchemaScan(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "schema_scans"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), index=True, nullable=False
    )
    connection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("database_connections.id", ondelete="RESTRICT"), index=True, nullable=False
    )
    requested_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )

    trigger: Mapped[SchemaScanTrigger] = mapped_column(String(50), nullable=False)
    status: Mapped[SchemaScanStatus] = mapped_column(
        String(50), default=SchemaScanStatus.QUEUED, index=True, nullable=False
    )
    requested_schemas_json: Mapped[list[str]] = mapped_column(
        JSONB, default=list, nullable=False
    )

    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    
    worker_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    progress_phase: Mapped[str | None] = mapped_column(String(100), nullable=True)
    progress_current: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    discovered_object_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    successful_object_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_object_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    warning_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    cancellation_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancellation_requested_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=True
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    failure_stage: Mapped[SchemaScanFailureStage | None] = mapped_column(String(50), nullable=True)
    safe_error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    safe_error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_schema_scans_org_status", "organization_id", "status"),
        Index("ix_schema_scans_conn_status", "connection_id", "status"),
        Index("ix_schema_scans_conn_created", "connection_id", "created_at"),
        Index("ix_schema_scans_org_created", "organization_id", "created_at"),
        Index("ix_schema_scans_status_heartbeat", "status", "heartbeat_at"),
        Index(
            "ix_schema_scans_worker_task_id", 
            "worker_task_id", 
            postgresql_where=worker_task_id.isnot(None)
        ),
        # Partial unique index for active scans
        Index(
            "uq_schema_scans_active_connection",
            "connection_id",
            unique=True,
            postgresql_where=status.in_([
                SchemaScanStatus.QUEUED,
                SchemaScanStatus.RUNNING,
                SchemaScanStatus.CANCELLATION_REQUESTED
            ])
        ),
        CheckConstraint("attempt_count >= 0", name="chk_schema_scans_attempt_count"),
        CheckConstraint("attempt_count <= max_attempts", name="chk_schema_scans_attempt_max"),
        CheckConstraint("max_attempts > 0", name="chk_schema_scans_max_attempts_pos"),
        CheckConstraint("progress_current >= 0", name="chk_schema_scans_prog_curr_pos"),
        CheckConstraint("progress_total >= 0", name="chk_schema_scans_prog_tot_pos"),
        CheckConstraint("discovered_object_count >= 0", name="chk_schema_scans_doc_pos"),
        CheckConstraint("successful_object_count >= 0", name="chk_schema_scans_soc_pos"),
        CheckConstraint("failed_object_count >= 0", name="chk_schema_scans_foc_pos"),
        CheckConstraint("warning_count >= 0", name="chk_schema_scans_warn_pos"),
    )
