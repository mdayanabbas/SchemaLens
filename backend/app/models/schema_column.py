import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base_class import Base
from app.models.mixins import UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.schema_relation import SchemaRelation
    from app.models.schema_snapshot import SchemaSnapshot


class SchemaColumn(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "schema_columns"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_snapshots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_relations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    schema_name: Mapped[str] = mapped_column(String(255), nullable=False)
    relation_name: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_identifier: Mapped[str] = mapped_column(String(255), nullable=False)
    
    ordinal_position: Mapped[int] = mapped_column(Integer, nullable=False)
    
    formatted_data_type: Mapped[str] = mapped_column(String(255), nullable=False)
    base_data_type: Mapped[str] = mapped_column(String(255), nullable=False)
    character_maximum_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    numeric_precision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    numeric_scale: Mapped[int | None] = mapped_column(Integer, nullable=True)
    datetime_precision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    is_nullable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    
    has_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    default_expression: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    default_expression_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    default_expression_truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    is_identity: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    identity_generation: Mapped[str | None] = mapped_column(String(64), nullable=True)
    
    is_generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    generation_expression_present: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    collation: Mapped[str | None] = mapped_column(String(255), nullable=True)
    comment: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship()
    snapshot: Mapped["SchemaSnapshot"] = relationship()
    relation: Mapped["SchemaRelation"] = relationship()

    __table_args__ = (
        UniqueConstraint("relation_id", "ordinal_position", name="uq_schema_columns_relation_ordinal"),
        UniqueConstraint("relation_id", "name", name="uq_schema_columns_relation_name"),
        Index("ix_schema_columns_schema_relation_name", "snapshot_id", "schema_name", "relation_name", "name"),
    )
