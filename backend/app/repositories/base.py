from typing import Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

ModelT = TypeVar("ModelT")
IdentifierT = TypeVar("IdentifierT")


class BaseRepository(Generic[ModelT, IdentifierT]):
    """
    Base generic repository for SQLAlchemy models.
    NOTE: Tenant-owned repositories MUST override lookups to enforce organization_id.
    """

    def __init__(self, session: AsyncSession, model_type: type[ModelT]) -> None:
        self.session = session
        self.model_type = model_type

    async def get_by_id(self, id_: IdentifierT) -> ModelT | None:
        """Get a single record by its primary key ID."""
        return await self.session.get(self.model_type, id_)

    def add(self, obj: ModelT) -> None:
        """Add an object to the session. Does not commit."""
        self.session.add(obj)

    async def delete(self, obj: ModelT) -> None:
        """Delete an object from the session. Does not commit."""
        await self.session.delete(obj)

    async def flush(self) -> None:
        """Flush pending changes to the database. Does not commit."""
        await self.session.flush()

    async def refresh(self, obj: ModelT) -> None:
        """Refresh the attributes of an object from the database."""
        await self.session.refresh(obj)
