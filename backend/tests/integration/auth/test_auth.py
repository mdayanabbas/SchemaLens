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
from app.services.authentication import AuthenticationService, AuthenticationException


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("TEST_DATABASE_URL"),
        reason="TEST_DATABASE_URL is not set",
    )
]

@pytest.fixture
async def integration_session_factory():
    settings = Settings(
        database_url=os.getenv("TEST_DATABASE_URL", ""),
        jwt_secret_key="test-secret",
        refresh_token_pepper="test-pepper"
    )
    engine = create_database_engine(settings)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False, autocommit=False)
    yield factory, settings
    await engine.dispose()


@pytest.mark.asyncio
async def test_auth_login_and_refresh_flow(integration_session_factory):
    factory, settings = integration_session_factory
    async with factory() as session:
        org_service = OrganizationService(session)
        user_service = UserService(session)
        mem_service = MembershipService(session)
        auth_service = AuthenticationService(session, settings)
        
        # Setup data
        org_slug = f"auth-org-{uuid.uuid4().hex[:8]}"
        org = await org_service.create_organization(OrganizationCreate(name="Test Org", slug=org_slug))
        
        user_email = f"auth-{uuid.uuid4().hex[:8]}@example.com"
        user = await user_service.create_user(UserCreate(email=user_email, display_name="Auth User"))
        
        await mem_service.create_membership(
            org.id, MembershipCreate(user_id=user.id, role=OrganizationRole.VIEWER)
        )
        
        # 1. Set password
        await auth_service.set_initial_password(user.id, "secure_password_123", "secure_password_123")
        
        # 2. Login
        tokens = await auth_service.login(user_email, "secure_password_123")
        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        
        # 3. Refresh (Token rotation is atomic)
        new_tokens = await auth_service.refresh(tokens.refresh_token)
        assert new_tokens.access_token is not None
        assert new_tokens.refresh_token is not None
        assert new_tokens.refresh_token != tokens.refresh_token
        
        # 4. Reusing old token compromises family
        with pytest.raises(AuthenticationException) as excinfo:
            await auth_service.refresh(tokens.refresh_token)
        assert excinfo.value.code == "TOKEN_REUSE_DETECTED"
        
        # New token should now also be revoked (family compromised)
        with pytest.raises(AuthenticationException) as excinfo:
            await auth_service.refresh(new_tokens.refresh_token)
        assert excinfo.value.code == "SESSION_REVOKED"
        
        # 5. Disabled user cannot use a valid token
        db_user = await user_service.repository.get_by_id(user.id)
        db_user.status = "disabled"
        await session.flush()
        
        with pytest.raises(AuthenticationException) as excinfo:
            await auth_service.authenticate_access_token(new_tokens.access_token)
        assert excinfo.value.code == "USER_DISABLED"
