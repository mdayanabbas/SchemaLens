import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User, uuid.UUID]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, User)

    async def get_by_email(self, email: str) -> User | None:
        """Get a user by their unique email address."""
        normalized_email = email.lower().strip()
        stmt = select(User).where(User.email == normalized_email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def email_exists(self, email: str) -> bool:
        """Check if an email address is already in use."""
        normalized_email = email.lower().strip()
        stmt = select(User.id).where(User.email == normalized_email)
        result = await self.session.execute(stmt)
        return result.first() is not None

    async def get_active_by_email(self, email: str) -> User | None:
        normalized_email = email.lower().strip()
        stmt = select(User).where(
            User.email == normalized_email,
            User.status == "active"
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_by_id(self, user_id: uuid.UUID) -> User | None:
        stmt = select(User).where(
            User.id == user_id,
            User.status == "active"
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def set_password_hash(self, user_id: uuid.UUID, password_hash: str) -> None:
        from sqlalchemy import update
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(password_hash=password_hash)
        )
        await self.session.execute(stmt)

    async def update_last_login(self, user_id: uuid.UUID, login_time: __import__('datetime').datetime) -> None:
        from sqlalchemy import update
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(last_login_at=login_time)
        )
        await self.session.execute(stmt)
