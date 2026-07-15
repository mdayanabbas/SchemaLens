import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import RefreshTokenStatus
from app.models.refresh_token import RefreshToken
from app.repositories.base import BaseRepository


class RefreshTokenRepository(BaseRepository[RefreshToken, uuid.UUID]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, RefreshToken)

    async def get_by_hash(self, token_hash: str, lock: bool = False) -> RefreshToken | None:
        """Get a refresh token by its keyed hash, optionally locking the row."""
        stmt = select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        if lock:
            stmt = stmt.with_for_update()
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_by_hash(self, token_hash: str, lock: bool = False) -> RefreshToken | None:
        """Get a refresh token by its hash, ensuring it is active."""
        stmt = select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.status == RefreshTokenStatus.ACTIVE,
        )
        if lock:
            stmt = stmt.with_for_update()
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_active_for_user(self, user_id: uuid.UUID, limit: int = 100, offset: int = 0) -> list[RefreshToken]:
        """List active refresh tokens for a user."""
        stmt = (
            select(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.status == RefreshTokenStatus.ACTIVE,
            )
            .order_by(RefreshToken.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_family(self, family_id: uuid.UUID, limit: int = 100, offset: int = 0) -> list[RefreshToken]:
        """List all refresh tokens for a specific family."""
        stmt = (
            select(RefreshToken)
            .where(RefreshToken.family_id == family_id)
            .order_by(RefreshToken.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def mark_rotated(self, token_id: uuid.UUID, new_token_id: uuid.UUID) -> None:
        """Mark a token as successfully rotated to a new token."""
        stmt = (
            update(RefreshToken)
            .where(RefreshToken.id == token_id)
            .values(
                status=RefreshTokenStatus.ROTATED,
                rotated_to_token_id=new_token_id,
            )
        )
        await self.session.execute(stmt)

    async def mark_revoked(self, token_id: uuid.UUID) -> None:
        """Mark a token as revoked."""
        stmt = (
            update(RefreshToken)
            .where(RefreshToken.id == token_id)
            .values(status=RefreshTokenStatus.REVOKED)
        )
        await self.session.execute(stmt)

    async def mark_compromised(self, token_id: uuid.UUID) -> None:
        """Mark a token as compromised."""
        stmt = (
            update(RefreshToken)
            .where(RefreshToken.id == token_id)
            .values(status=RefreshTokenStatus.COMPROMISED)
        )
        await self.session.execute(stmt)

    async def revoke_family(self, user_id: uuid.UUID, family_id: uuid.UUID) -> None:
        """Revoke all active tokens for a specific user and family."""
        stmt = (
            update(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.family_id == family_id,
                RefreshToken.status == RefreshTokenStatus.ACTIVE,
            )
            .values(status=RefreshTokenStatus.REVOKED)
        )
        await self.session.execute(stmt)

    async def revoke_all_for_user(self, user_id: uuid.UUID) -> None:
        """Revoke all active tokens across all families for a user."""
        stmt = (
            update(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.status == RefreshTokenStatus.ACTIVE,
            )
            .values(status=RefreshTokenStatus.REVOKED)
        )
        await self.session.execute(stmt)
