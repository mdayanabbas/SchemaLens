from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.db.transactions import transactional
from app.models.organization import Organization
from app.repositories.organization import OrganizationRepository
from app.schemas.organization import OrganizationCreate, OrganizationRead, OrganizationUpdate


class OrganizationService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = OrganizationRepository(session)

    async def create_organization(self, org_in: OrganizationCreate) -> OrganizationRead:
        """Create a new organization."""
        async with transactional(self.session):
            exists = await self.repository.slug_exists(org_in.slug)
            if exists:
                raise ConflictError(
                    message="An organization with this slug already exists.",
                    code="ORGANIZATION_SLUG_CONFLICT",
                )

            organization = Organization(
                name=org_in.name,
                slug=org_in.slug,
            )
            self.repository.add(organization)
            await self.repository.flush()
            
            return OrganizationRead.model_validate(organization)

    async def get_organization(self, slug: str) -> OrganizationRead:
        """Retrieve an organization by slug."""
        organization = await self.repository.get_by_slug(slug)
        if not organization:
            raise NotFoundError(
                message="Organization not found.",
                code="ORGANIZATION_NOT_FOUND",
            )
        return OrganizationRead.model_validate(organization)

    async def update_organization(self, slug: str, update_in: OrganizationUpdate) -> OrganizationRead:
        async with transactional(self.session):
            organization = await self.repository.get_by_slug(slug)
            if not organization:
                raise NotFoundError(
                    message="Organization not found.",
                    code="ORGANIZATION_NOT_FOUND",
                )
            
            if update_in.name is not None:
                organization.name = update_in.name
            if update_in.status is not None:
                organization.status = update_in.status
                
            await self.repository.flush()
            return OrganizationRead.model_validate(organization)

    async def list_for_user(
        self,
        user: "User",
        *,
        offset: int,
        limit: int,
        platform_admin_access: bool = False,
    ) -> list[OrganizationSummaryRead]:
        """List organizations for a user."""
        from sqlalchemy import select
        from app.models.membership import OrganizationMembership
        from app.models.enums import MembershipStatus, OrganizationStatus

        if platform_admin_access and user.is_platform_admin:
            # Platform admin explicit access: list all organizations (including suspended)
            stmt = (
                select(Organization)
                .order_by(Organization.name.asc(), Organization.id.asc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(stmt)
            orgs = result.scalars().all()
            
            return [
                OrganizationSummaryRead(
                    id=org.id,
                    name=org.name,
                    slug=org.slug,
                    status=org.status,
                    role=None,
                    is_platform_admin_access=True,
                    created_at=org.created_at,
                )
                for org in orgs
            ]
        else:
            # Normal user: list only organizations with active memberships
            stmt = (
                select(Organization, OrganizationMembership.role)
                .join(OrganizationMembership, Organization.id == OrganizationMembership.organization_id)
                .where(
                    OrganizationMembership.user_id == user.id,
                    OrganizationMembership.status == MembershipStatus.ACTIVE,
                    Organization.status == OrganizationStatus.ACTIVE,
                )
                .order_by(Organization.name.asc(), Organization.id.asc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(stmt)
            rows = result.all()
            
            return [
                OrganizationSummaryRead(
                    id=org.id,
                    name=org.name,
                    slug=org.slug,
                    status=org.status,
                    role=role,
                    is_platform_admin_access=False,
                    created_at=org.created_at,
                )
                for org, role in rows
            ]

    async def update_organization_authorized(
        self,
        context: "AuthorizedOrganizationContext",
        update_in: OrganizationUpdate,
    ) -> OrganizationRead:
        """Update an organization using an authorized context."""
        async with transactional(self.session):
            organization = await self.repository.get_by_id(context.organization_id)
            if not organization:
                raise NotFoundError(
                    message="Organization not found.",
                    code="ORGANIZATION_NOT_FOUND",
                )
            
            if update_in.name is not None:
                organization.name = update_in.name
            if update_in.status is not None:
                organization.status = update_in.status
                
            await self.repository.flush()
            return OrganizationRead.model_validate(organization)
