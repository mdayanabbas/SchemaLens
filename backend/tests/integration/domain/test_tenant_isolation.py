import os
import uuid
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.engine import create_database_engine
from app.models.enums import OrganizationRole
from app.schemas.organization import OrganizationCreate
from app.schemas.user import UserCreate
from app.schemas.membership import MembershipCreate
from app.services.organization import OrganizationService
from app.services.user import UserService
from app.services.membership import MembershipService


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("TEST_DATABASE_URL"),
        reason="TEST_DATABASE_URL is not set",
    )
]

@pytest.fixture
async def integration_session_factory():
    settings = Settings(database_url=os.getenv("TEST_DATABASE_URL", ""))
    engine = create_database_engine(settings)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False, autocommit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_tenant_isolation_flow(integration_session_factory):
    async with integration_session_factory() as session:
        org_service = OrganizationService(session)
        user_service = UserService(session)
        mem_service = MembershipService(session)
        
        org_slug = f"test-org-{uuid.uuid4().hex[:8]}"
        org = await org_service.create_organization(OrganizationCreate(name="Test Org", slug=org_slug))
        assert org.id is not None
        
        user_email = f"test-{uuid.uuid4().hex[:8]}@example.com"
        user = await user_service.create_user(UserCreate(email=user_email, display_name="Test User"))
        assert user.id is not None
        
        membership = await mem_service.create_membership(
            org.id, MembershipCreate(user_id=user.id, role=OrganizationRole.VIEWER)
        )
        assert membership.id is not None
