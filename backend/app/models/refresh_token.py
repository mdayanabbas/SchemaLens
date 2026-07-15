import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import RefreshTokenStatus


class RefreshToken(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "refresh_tokens"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        index=True,
        nullable=False,
    )
    family_id: Mapped[uuid.UUID] = mapped_column(index=True, nullable=False)
    token_hash: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    status: Mapped[RefreshTokenStatus] = mapped_column(
        String, index=True, nullable=False, default=RefreshTokenStatus.ACTIVE
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    
    rotated_to_token_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("refresh_tokens.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_ip_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    created_user_agent_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    last_used_ip_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    last_used_user_agent_hash: Mapped[str | None] = mapped_column(String, nullable=True)

    user: Mapped["User"] = relationship("User")
    rotated_to_token: Mapped["RefreshToken"] = relationship(
        "RefreshToken", remote_side="RefreshToken.id"
    )
