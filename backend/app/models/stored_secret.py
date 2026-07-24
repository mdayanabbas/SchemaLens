import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class StoredSecret(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "stored_secrets"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False, default="local_encrypted")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")
    
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    nonce: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    encryption_algorithm: Mapped[str] = mapped_column(String(50), nullable=False, default="AES-256-GCM")
    key_version: Mapped[str] = mapped_column(String(50), nullable=False)
    payload_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    updated_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    
    rotated_from_secret_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("stored_secrets.id", ondelete="RESTRICT"), nullable=True
    )
    last_resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_stored_secrets_organization_id_name"),
        Index("ix_stored_secrets_org_status", "organization_id", "status"),
        Index("ix_stored_secrets_org_created_at", "organization_id", "created_at"),
    )
