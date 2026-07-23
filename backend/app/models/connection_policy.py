import uuid
from typing import Any

from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin
from app.models.connection_enums import ApprovalMode


class ConnectionPolicy(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "connection_policies"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    connection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("database_connections.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
        index=True,
    )

    approved_schemas_json: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(JSONB), nullable=False, default=list
    )
    blocked_schemas_json: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(JSONB), nullable=False, default=lambda: ["pg_catalog", "information_schema"]
    )
    
    allow_schema_scanning: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    allow_query_generation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    allow_query_execution: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    approval_mode: Mapped[ApprovalMode] = mapped_column(
        String(50), nullable=False, default=ApprovalMode.ALWAYS
    )

    max_statement_timeout_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=30000)
    max_lock_timeout_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=5000)
    max_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=1000)
    max_response_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=5242880) # 5 MB
    max_estimated_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=100000)
    max_estimated_cost: Mapped[float] = mapped_column(Float, nullable=False, default=10000.0)
    max_joined_tables: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    max_subquery_depth: Mapped[int] = mapped_column(Integer, nullable=False, default=5)

    allow_system_catalogs: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    allow_cross_joins: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    require_fully_qualified_tables: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    updated_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )

    # Relationships
    connection: Mapped["DatabaseConnection"] = relationship(
        "DatabaseConnection", back_populates="policy"
    )
    
    __table_args__ = (
        CheckConstraint("max_statement_timeout_ms > 0", name="chk_max_statement_timeout"),
        CheckConstraint("max_lock_timeout_ms > 0", name="chk_max_lock_timeout"),
        CheckConstraint("max_rows > 0", name="chk_max_rows"),
        CheckConstraint("max_response_bytes > 0", name="chk_max_response_bytes"),
        CheckConstraint("max_estimated_rows > 0", name="chk_max_estimated_rows"),
        CheckConstraint("max_estimated_cost > 0", name="chk_max_estimated_cost"),
        CheckConstraint("max_joined_tables > 0", name="chk_max_joined_tables"),
        CheckConstraint("max_subquery_depth > 0", name="chk_max_subquery_depth"),
    )
