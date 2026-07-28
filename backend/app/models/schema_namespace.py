import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base_class import Base
from app.models.mixins import UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.schema_snapshot import SchemaSnapshot


class SchemaNamespace(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "schema_namespaces"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_snapshots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    comment: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    normalized_identifier: Mapped[str] = mapped_column(String(255), nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship()
    snapshot: Mapped["SchemaSnapshot"] = relationship(back_populates="namespaces")

    __table_args__ = (
        UniqueConstraint("snapshot_id", "name", name="uq_schema_namespaces_snapshot_name"),
        Index("ix_schema_namespaces_snapshot_name", "snapshot_id", "name"),
    )
