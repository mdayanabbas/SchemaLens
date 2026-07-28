import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base_class import Base
from app.models.mixins import UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.schema_index_column import SchemaIndexColumn
    from app.models.schema_relation import SchemaRelation
    from app.models.schema_snapshot import SchemaSnapshot


class SchemaIndex(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "schema_indexes"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_snapshots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_relations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    is_unique: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_ready: Mapped[bool] = mapped_column(Boolean, nullable=False)
    
    access_method: Mapped[str] = mapped_column(String(64), nullable=False)
    
    predicate_present: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    predicate_expression: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    predicate_expression_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    predicate_expression_truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    expression_index: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    estimated_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship()
    snapshot: Mapped["SchemaSnapshot"] = relationship()
    relation: Mapped["SchemaRelation"] = relationship()
    columns: Mapped[list["SchemaIndexColumn"]] = relationship(back_populates="index")

    __table_args__ = (
        UniqueConstraint("snapshot_id", "relation_id", "name", name="uq_schema_indexes_snapshot_rel_name"),
    )
