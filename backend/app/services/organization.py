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
