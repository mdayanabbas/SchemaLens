from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.organization import Organization
from app.repositories.base import BaseRepository


class OrganizationRepository(BaseRepository[Organization, str]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, Organization)

    async def get_by_slug(self, slug: str) -> Organization | None:
        """Get an organization by its unique slug."""
        normalized_slug = slug.lower().strip()
        stmt = select(Organization).where(Organization.slug == normalized_slug)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def slug_exists(self, slug: str) -> bool:
        """Check if an organization slug already exists."""
        normalized_slug = slug.lower().strip()
        stmt = select(Organization.id).where(Organization.slug == normalized_slug)
        result = await self.session.execute(stmt)
        return result.first() is not None
