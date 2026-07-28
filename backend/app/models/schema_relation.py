import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base_class import Base
from app.models.mixins import UUIDPrimaryKeyMixin
from app.models.schema_snapshot_enums import SchemaRelationKind

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.schema_namespace import SchemaNamespace
    from app.models.schema_snapshot import SchemaSnapshot


class SchemaRelation(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "schema_relations"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_snapshots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    schema_name: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_identifier: Mapped[str] = mapped_column(String(255), nullable=False)
    qualified_name: Mapped[str] = mapped_column(String(512), nullable=False)
    
    kind: Mapped[SchemaRelationKind] = mapped_column(nullable=False)
    comment: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    estimated_rows: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    is_partition: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    parent_schema_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    parent_relation_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship()
    snapshot: Mapped["SchemaSnapshot"] = relationship()
    namespace: Mapped["SchemaNamespace"] = relationship()

    __table_args__ = (
        UniqueConstraint("snapshot_id", "schema_name", "name", name="uq_schema_relations_snapshot_schema_name"),
        Index("ix_schema_relations_snapshot_schema_name", "snapshot_id", "schema_name", "name"),
        Index("ix_schema_relations_snapshot_kind", "snapshot_id", "kind"),
    )
