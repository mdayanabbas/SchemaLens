import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.db.transactions import transactional
from app.models.membership import OrganizationMembership
from app.repositories.membership import MembershipRepository
from app.repositories.organization import OrganizationRepository
from app.repositories.user import UserRepository
from app.schemas.membership import MembershipCreate, MembershipRead, MembershipUpdate


class MembershipService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.membership_repo = MembershipRepository(session)
        self.org_repo = OrganizationRepository(session)
        self.user_repo = UserRepository(session)

    async def create_membership(
        self, organization_id: uuid.UUID, membership_in: MembershipCreate
    ) -> MembershipRead:
        """Create a new membership linking a user to an organization."""
        async with transactional(self.session):
            org = await self.org_repo.get_by_id(organization_id)
            if not org:
                raise NotFoundError(
                    message="Organization not found.",
                    code="ORGANIZATION_NOT_FOUND",
                )
            
            user = await self.user_repo.get_by_id(membership_in.user_id)
            if not user:
                raise NotFoundError(
                    message="User not found.",
                    code="USER_NOT_FOUND",
                )

            exists = await self.membership_repo.membership_exists(
                user_id=membership_in.user_id,
                organization_id=organization_id,
            )
            if exists:
                raise ConflictError(
                    message="User is already a member of this organization.",
                    code="MEMBERSHIP_CONFLICT",
                )

            membership = OrganizationMembership(
                organization_id=organization_id,
                user_id=membership_in.user_id,
                role=membership_in.role,
                status=membership_in.status,
            )
            self.membership_repo.add(membership)
            await self.membership_repo.flush()

            return MembershipRead.model_validate(membership)

    async def get_membership(
        self, organization_id: uuid.UUID, membership_id: uuid.UUID
    ) -> MembershipRead:
        """Retrieve a specific membership for an organization."""
        membership = await self.membership_repo.get_by_id_for_organization(
            membership_id=membership_id, organization_id=organization_id
        )
        if not membership:
            raise NotFoundError(
                message="Membership not found.",
                code="MEMBERSHIP_NOT_FOUND",
            )
        return MembershipRead.model_validate(membership)

    async def update_membership(
        self, organization_id: uuid.UUID, membership_id: uuid.UUID, update_in: MembershipUpdate
    ) -> MembershipRead:
        """Update a membership's role or status."""
        async with transactional(self.session):
            membership = await self.membership_repo.get_by_id_for_organization(
                membership_id=membership_id, organization_id=organization_id
            )
            if not membership:
                raise NotFoundError(
                    message="Membership not found.",
                    code="MEMBERSHIP_NOT_FOUND",
                )
            
            if update_in.role is not None:
                membership.role = update_in.role
            if update_in.status is not None:
                membership.status = update_in.status
                
            await self.membership_repo.flush()
            return MembershipRead.model_validate(membership)

    async def list_organization_memberships(
        self, organization_id: uuid.UUID, limit: int = 100, offset: int = 0
    ) -> list[MembershipRead]:
        """List memberships for a specific organization."""
        org = await self.org_repo.get_by_id(organization_id)
        if not org:
            raise NotFoundError(
                message="Organization not found.",
                code="ORGANIZATION_NOT_FOUND",
            )

        memberships = await self.membership_repo.list_for_organization(
            organization_id=organization_id, limit=limit, offset=offset
        )
        return [MembershipRead.model_validate(m) for m in memberships]

    async def list_user_memberships(
        self, user_id: uuid.UUID, limit: int = 100, offset: int = 0
    ) -> list[MembershipRead]:
        """List memberships for a specific user across all their organizations."""
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundError(
                message="User not found.",
                code="USER_NOT_FOUND",
            )

        memberships = await self.membership_repo.list_for_user(
            user_id=user_id, limit=limit, offset=offset
        )
        return [MembershipRead.model_validate(m) for m in memberships]
