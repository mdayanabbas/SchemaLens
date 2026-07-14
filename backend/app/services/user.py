import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.db.transactions import transactional
from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.user import UserCreate, UserRead, UserUpdate


class UserService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = UserRepository(session)

    async def create_user(self, user_in: UserCreate) -> UserRead:
        """Create a new user without a password."""
        async with transactional(self.session):
            exists = await self.repository.email_exists(user_in.email)
            if exists:
                raise ConflictError(
                    message="A user with this email already exists.",
                    code="USER_EMAIL_CONFLICT",
                )

            user = User(
                email=user_in.email,
                display_name=user_in.display_name,
            )
            self.repository.add(user)
            await self.repository.flush()
            
            return UserRead.model_validate(user)

    async def get_user(self, user_id: uuid.UUID) -> UserRead:
        """Retrieve a user by ID."""
        user = await self.repository.get_by_id(user_id)
        if not user:
            raise NotFoundError(
                message="User not found.",
                code="USER_NOT_FOUND",
            )
        return UserRead.model_validate(user)

    async def update_user(self, user_id: uuid.UUID, update_in: UserUpdate) -> UserRead:
        """Update a user's basic profile."""
        async with transactional(self.session):
            user = await self.repository.get_by_id(user_id)
            if not user:
                raise NotFoundError(
                    message="User not found.",
                    code="USER_NOT_FOUND",
                )
            
            if update_in.display_name is not None:
                user.display_name = update_in.display_name
            if update_in.status is not None:
                user.status = update_in.status
                
            await self.repository.flush()
            return UserRead.model_validate(user)
