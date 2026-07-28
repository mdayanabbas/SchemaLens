import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base_class import Base
from app.models.mixins import UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.database_connection import DatabaseConnection
    from app.models.organization import Organization
    from app.models.schema_scan import SchemaScan
    from app.models.schema_snapshot import SchemaSnapshot


class ConnectionSchemaState(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "connection_schema_states"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    connection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("database_connections.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    
    current_snapshot_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("schema_snapshots.id", ondelete="SET NULL"), nullable=True
    )
    previous_snapshot_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("schema_snapshots.id", ondelete="SET NULL"), nullable=True
    )
    latest_scan_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("schema_scans.id", ondelete="SET NULL"), nullable=True
    )
    
    current_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship()
    connection: Mapped["DatabaseConnection"] = relationship()
    current_snapshot: Mapped["SchemaSnapshot"] = relationship(foreign_keys=[current_snapshot_id])
    previous_snapshot: Mapped["SchemaSnapshot"] = relationship(foreign_keys=[previous_snapshot_id])
    latest_scan: Mapped["SchemaScan"] = relationship(foreign_keys=[latest_scan_id])
