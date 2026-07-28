import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base_class import Base
from app.models.mixins import UUIDPrimaryKeyMixin
from app.models.schema_snapshot_enums import NullsOrder, SortDirection

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.schema_column import SchemaColumn
    from app.models.schema_index import SchemaIndex
    from app.models.schema_snapshot import SchemaSnapshot


class SchemaIndexColumn(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "schema_index_columns"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_snapshots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    index_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_indexes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    column_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("schema_columns.id", ondelete="CASCADE"), nullable=True, index=True
    )
    
    ordinal_position: Mapped[int] = mapped_column(Integer, nullable=False)
    
    expression: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    expression_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expression_truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    included: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    sort_direction: Mapped[SortDirection | None] = mapped_column(nullable=True)
    nulls_order: Mapped[NullsOrder | None] = mapped_column(nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship()
    snapshot: Mapped["SchemaSnapshot"] = relationship()
    index: Mapped["SchemaIndex"] = relationship(back_populates="columns")
    column: Mapped["SchemaColumn"] = relationship()

    __table_args__ = (
        UniqueConstraint("index_id", "ordinal_position", name="uq_schema_index_cols_ordinal"),
    )
