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
