import uuid
from datetime import datetime

from sqlalchemy import String, Integer, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin
from app.models.connection_enums import (
    ConnectionEnvironment,
    ConnectionStatus,
    ConnectionTestStatus,
    DatabaseDialect,
    SecretProviderType,
    SSLMode,
)


class DatabaseConnection(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "database_connections"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"),
        nullable=False,
        index=True
    )
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    environment: Mapped[ConnectionEnvironment] = mapped_column(String(50), nullable=False, index=True)
    dialect: Mapped[DatabaseDialect] = mapped_column(String(50), nullable=False, index=True)
    
    host: Mapped[str] = mapped_column(String(255), nullable=False)
    port: Mapped[int] = mapped_column(Integer, nullable=False)
    database_name: Mapped[str] = mapped_column(String(100), nullable=False)
    default_catalog: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    ssl_mode: Mapped[SSLMode] = mapped_column(String(50), nullable=False, default=SSLMode.REQUIRE)
    secret_provider: Mapped[SecretProviderType] = mapped_column(String(50), nullable=False)
    secret_reference: Mapped[str] = mapped_column(String(500), nullable=False)
    
    status: Mapped[ConnectionStatus] = mapped_column(
        String(50), nullable=False, default=ConnectionStatus.DRAFT, index=True
    )
    
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_test_status: Mapped[ConnectionTestStatus] = mapped_column(
        String(50), nullable=False, default=ConnectionTestStatus.NEVER_TESTED, index=True
    )
    last_test_error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    updated_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )

    organization: Mapped["Organization"] = relationship("Organization")
    creator: Mapped["User"] = relationship("User", foreign_keys=[created_by_user_id])
    updater: Mapped["User"] = relationship("User", foreign_keys=[updated_by_user_id])
    
    policy: Mapped["ConnectionPolicy"] = relationship(
        "ConnectionPolicy",
        back_populates="connection",
        uselist=False,
    )
