import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base_class import Base
from app.models.mixins import UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.schema_column import SchemaColumn
    from app.models.schema_constraint import SchemaConstraint
    from app.models.schema_snapshot import SchemaSnapshot


class SchemaConstraintColumn(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "schema_constraint_columns"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_snapshots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    constraint_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_constraints.id", ondelete="CASCADE"), nullable=False, index=True
    )
    column_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("schema_columns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    ordinal_position: Mapped[int] = mapped_column(Integer, nullable=False)
    
    referenced_column_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("schema_columns.id", ondelete="SET NULL"), nullable=True, index=True
    )
    referenced_column_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship()
    snapshot: Mapped["SchemaSnapshot"] = relationship()
    constraint: Mapped["SchemaConstraint"] = relationship(back_populates="columns")
    column: Mapped["SchemaColumn"] = relationship(foreign_keys=[column_id])
    referenced_column: Mapped["SchemaColumn"] = relationship(foreign_keys=[referenced_column_id])

    __table_args__ = (
        UniqueConstraint("constraint_id", "ordinal_position", name="uq_schema_constraint_cols_ordinal"),
    )
