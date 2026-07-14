import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.membership import OrganizationMembership
from app.repositories.base import BaseRepository


class MembershipRepository(BaseRepository[OrganizationMembership, uuid.UUID]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, OrganizationMembership)
    
    async def get_by_id_for_organization(
        self, membership_id: uuid.UUID, organization_id: uuid.UUID
    ) -> OrganizationMembership | None:
        """Get a membership ensuring it belongs to the given organization."""
        stmt = select(OrganizationMembership).where(
            OrganizationMembership.id == membership_id,
            OrganizationMembership.organization_id == organization_id,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_for_user_and_organization(
        self, user_id: uuid.UUID, organization_id: uuid.UUID
    ) -> OrganizationMembership | None:
        """Get a specific user's membership for a given organization."""
        stmt = select(OrganizationMembership).where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.organization_id == organization_id,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def membership_exists(self, user_id: uuid.UUID, organization_id: uuid.UUID) -> bool:
        """Check if a user is already a member of an organization."""
        stmt = select(OrganizationMembership.id).where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.organization_id == organization_id,
        )
        result = await self.session.execute(stmt)
        return result.first() is not None

    async def list_for_organization(
        self, organization_id: uuid.UUID, limit: int = 100, offset: int = 0
    ) -> list[OrganizationMembership]:
        """List memberships belonging to an organization with bounds."""
        stmt = (
            select(OrganizationMembership)
            .where(OrganizationMembership.organization_id == organization_id)
            .order_by(OrganizationMembership.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_for_user(
        self, user_id: uuid.UUID, limit: int = 100, offset: int = 0
    ) -> list[OrganizationMembership]:
        """List all organizations a user belongs to."""
        stmt = (
            select(OrganizationMembership)
            .where(OrganizationMembership.user_id == user_id)
            .order_by(OrganizationMembership.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
