import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base_class import Base
from app.models.mixins import UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.schema_namespace import SchemaNamespace
    from app.models.schema_snapshot import SchemaSnapshot


class SchemaRoutine(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "schema_routines"

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
    
    identity_arguments: Mapped[str] = mapped_column(String(2000), nullable=False)
    result_type: Mapped[str] = mapped_column(String(2000), nullable=False)
    
    routine_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    volatility: Mapped[str] = mapped_column(String(64), nullable=False)
    parallel_safety: Mapped[str] = mapped_column(String(64), nullable=False)
    security_definer: Mapped[bool] = mapped_column(Boolean, nullable=False)
    language: Mapped[str] = mapped_column(String(64), nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship()
    snapshot: Mapped["SchemaSnapshot"] = relationship()
    namespace: Mapped["SchemaNamespace"] = relationship()

    __table_args__ = (
        UniqueConstraint("snapshot_id", "schema_name", "name", "identity_arguments", name="uq_schema_routines_signature"),
    )
