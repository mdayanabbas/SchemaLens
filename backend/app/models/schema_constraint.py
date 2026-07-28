import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base_class import Base
from app.models.mixins import UUIDPrimaryKeyMixin
from app.models.schema_snapshot_enums import MatchType, ReferentialAction, SchemaConstraintKind

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.schema_constraint_column import SchemaConstraintColumn
    from app.models.schema_relation import SchemaRelation
    from app.models.schema_snapshot import SchemaSnapshot


class SchemaConstraint(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "schema_constraints"

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
    kind: Mapped[SchemaConstraintKind] = mapped_column(nullable=False)
    
    is_deferrable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    initially_deferred: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    
    check_expression: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    check_expression_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    check_expression_truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    referenced_schema_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    referenced_relation_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    referenced_relation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("schema_relations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    
    update_action: Mapped[ReferentialAction | None] = mapped_column(nullable=True)
    delete_action: Mapped[ReferentialAction | None] = mapped_column(nullable=True)
    match_type: Mapped[MatchType | None] = mapped_column(nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship()
    snapshot: Mapped["SchemaSnapshot"] = relationship()
    relation: Mapped["SchemaRelation"] = relationship(foreign_keys=[relation_id])
    referenced_relation: Mapped["SchemaRelation"] = relationship(foreign_keys=[referenced_relation_id])
    columns: Mapped[list["SchemaConstraintColumn"]] = relationship(back_populates="constraint")

    __table_args__ = (
        UniqueConstraint("snapshot_id", "relation_id", "name", name="uq_schema_constraints_snapshot_rel_name"),
        Index("ix_schema_constraints_kind", "snapshot_id", "kind"),
    )
