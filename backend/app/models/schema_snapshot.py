import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base_class import Base
from app.models.mixins import UUIDPrimaryKeyMixin
from app.models.schema_snapshot_enums import SchemaSnapshotStatus

if TYPE_CHECKING:
    from app.models.connection_schema_state import ConnectionSchemaState
    from app.models.database_connection import DatabaseConnection
    from app.models.organization import Organization
    from app.models.schema_namespace import SchemaNamespace
    from app.models.schema_scan import SchemaScan
    from app.models.user import User


class SchemaSnapshot(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "schema_snapshots"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    connection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("database_connections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    schema_scan_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_scans.id", ondelete="RESTRICT"), nullable=False, unique=True
    )
    
    status: Mapped[SchemaSnapshotStatus] = mapped_column(nullable=False, default=SchemaSnapshotStatus.BUILDING)
    snapshot_version: Mapped[int] = mapped_column(nullable=False)
    
    fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fingerprint_algorithm: Mapped[str] = mapped_column(String(32), nullable=False, default="sha256")
    fingerprint_input_version: Mapped[int] = mapped_column(nullable=False, default=1)
    
    server_version: Mapped[str] = mapped_column(String(128), nullable=False)
    database_name: Mapped[str] = mapped_column(String(128), nullable=False)
    
    selected_schemas_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    
    namespace_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    relation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    constraint_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    index_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    routine_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    warning_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    metadata_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    safe_invalid_reason_code: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="schema_snapshots")
    connection: Mapped["DatabaseConnection"] = relationship()
    schema_scan: Mapped["SchemaScan"] = relationship(back_populates="schema_snapshot")
    creator: Mapped["User"] = relationship()
    namespaces: Mapped[list["SchemaNamespace"]] = relationship(back_populates="snapshot", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("connection_id", "snapshot_version", name="uq_schema_snapshots_connection_version"),
        Index("ix_schema_snapshots_connection_status", "connection_id", "status"),
        Index("ix_schema_snapshots_connection_created", "connection_id", "created_at"),
        Index("ix_schema_snapshots_org_created", "organization_id", "created_at"),
        Index("ix_schema_snapshots_fingerprint", "fingerprint", postgresql_where=func.char_length("fingerprint") > 0),
        CheckConstraint("namespace_count >= 0", name="chk_schema_snapshots_namespace_count"),
        CheckConstraint("relation_count >= 0", name="chk_schema_snapshots_relation_count"),
        CheckConstraint("column_count >= 0", name="chk_schema_snapshots_column_count"),
        CheckConstraint("constraint_count >= 0", name="chk_schema_snapshots_constraint_count"),
        CheckConstraint("index_count >= 0", name="chk_schema_snapshots_index_count"),
        CheckConstraint("routine_count >= 0", name="chk_schema_snapshots_routine_count"),
        CheckConstraint("warning_count >= 0", name="chk_schema_snapshots_warning_count"),
        CheckConstraint("metadata_size_bytes >= 0", name="chk_schema_snapshots_metadata_size"),
    )
