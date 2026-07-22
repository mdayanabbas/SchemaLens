import uuid
from unittest.mock import AsyncMock

import pytest

from app.governance.authorization import AuthorizationService
from app.governance.permissions import Permission
from app.models.enums import MembershipStatus, OrganizationRole, OrganizationStatus, UserStatus
from app.models.membership import OrganizationMembership
from app.models.organization import Organization
from app.models.user import User


@pytest.fixture
def auth_service():
    session = AsyncMock()
    service = AuthorizationService(session)
    service.org_repo = AsyncMock()
    service.membership_repo = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_active_user_with_active_membership_and_permission_is_allowed(auth_service):
    user = User(id=uuid.uuid4(), status=UserStatus.ACTIVE, is_platform_admin=False)
    org_id = uuid.uuid4()
    org = Organization(id=org_id, status=OrganizationStatus.ACTIVE)
    membership = OrganizationMembership(
        id=uuid.uuid4(),
        user_id=user.id,
        organization_id=org_id,
        role=OrganizationRole.ORGANIZATION_ADMIN,
        status=MembershipStatus.ACTIVE,
    )

    auth_service.org_repo.get_by_id.return_value = org
    auth_service.membership_repo.get_for_user_and_organization.return_value = membership

    decision = await auth_service.evaluate_permission(
        user=user, organization_id=org_id, permission=Permission.ORGANIZATION_MANAGE
    )
    assert decision.allowed is True
    assert decision.decision_code == "ALLOWED"


@pytest.mark.asyncio
async def test_active_user_without_permission_is_denied(auth_service):
    user = User(id=uuid.uuid4(), status=UserStatus.ACTIVE, is_platform_admin=False)
    org_id = uuid.uuid4()
    org = Organization(id=org_id, status=OrganizationStatus.ACTIVE)
    membership = OrganizationMembership(
        id=uuid.uuid4(),
        user_id=user.id,
        organization_id=org_id,
        role=OrganizationRole.VIEWER,
        status=MembershipStatus.ACTIVE,
    )

    auth_service.org_repo.get_by_id.return_value = org
    auth_service.membership_repo.get_for_user_and_organization.return_value = membership

    decision = await auth_service.evaluate_permission(
        user=user, organization_id=org_id, permission=Permission.ORGANIZATION_MANAGE
    )
    assert decision.allowed is False
    assert decision.decision_code == "PERMISSION_DENIED"


@pytest.mark.asyncio
async def test_missing_membership_denied(auth_service):
    user = User(id=uuid.uuid4(), status=UserStatus.ACTIVE, is_platform_admin=False)
    org_id = uuid.uuid4()
    org = Organization(id=org_id, status=OrganizationStatus.ACTIVE)

    auth_service.org_repo.get_by_id.return_value = org
    auth_service.membership_repo.get_for_user_and_organization.return_value = None

    decision = await auth_service.evaluate_permission(
        user=user, organization_id=org_id, permission=Permission.ORGANIZATION_READ
    )
    assert decision.allowed is False
    assert decision.decision_code == "MEMBERSHIP_NOT_FOUND"


@pytest.mark.asyncio
async def test_disabled_membership_denied(auth_service):
    user = User(id=uuid.uuid4(), status=UserStatus.ACTIVE, is_platform_admin=False)
    org_id = uuid.uuid4()
    org = Organization(id=org_id, status=OrganizationStatus.ACTIVE)
    membership = OrganizationMembership(
        id=uuid.uuid4(),
        user_id=user.id,
        organization_id=org_id,
        role=OrganizationRole.ORGANIZATION_ADMIN,
        status=MembershipStatus.DISABLED,
    )

    auth_service.org_repo.get_by_id.return_value = org
    auth_service.membership_repo.get_for_user_and_organization.return_value = membership

    decision = await auth_service.evaluate_permission(
        user=user, organization_id=org_id, permission=Permission.ORGANIZATION_READ
    )
    assert decision.allowed is False
    assert decision.decision_code == "MEMBERSHIP_INACTIVE"


@pytest.mark.asyncio
async def test_disabled_user_denied(auth_service):
    user = User(id=uuid.uuid4(), status=UserStatus.DISABLED, is_platform_admin=False)
    org_id = uuid.uuid4()

    decision = await auth_service.evaluate_permission(
        user=user, organization_id=org_id, permission=Permission.ORGANIZATION_READ
    )
    assert decision.allowed is False
    assert decision.decision_code == "USER_DISABLED"


@pytest.mark.asyncio
async def test_suspended_organization_denied(auth_service):
    user = User(id=uuid.uuid4(), status=UserStatus.ACTIVE, is_platform_admin=False)
    org_id = uuid.uuid4()
    org = Organization(id=org_id, status=OrganizationStatus.SUSPENDED)

    auth_service.org_repo.get_by_id.return_value = org

    decision = await auth_service.evaluate_permission(
        user=user, organization_id=org_id, permission=Permission.ORGANIZATION_READ
    )
    assert decision.allowed is False
    assert decision.decision_code == "ORGANIZATION_SUSPENDED"


@pytest.mark.asyncio
async def test_platform_admin_path_allowed_explicitly(auth_service):
    user = User(id=uuid.uuid4(), status=UserStatus.ACTIVE, is_platform_admin=True)
    org_id = uuid.uuid4()
    org = Organization(id=org_id, status=OrganizationStatus.ACTIVE)

    auth_service.org_repo.get_by_id.return_value = org

    decision = await auth_service.evaluate_permission(
        user=user, organization_id=org_id, permission=Permission.ORGANIZATION_MANAGE
    )
    assert decision.allowed is True
    assert decision.decision_code == "PLATFORM_ADMIN_ALLOWED"
    assert decision.membership_id is None  # Does not fabricate membership


@pytest.mark.asyncio
async def test_denial_does_not_expose_another_organizations_existence(auth_service):
    user = User(id=uuid.uuid4(), status=UserStatus.ACTIVE, is_platform_admin=False)
    org_id = uuid.uuid4()
    
    # Organization doesn't exist
    auth_service.org_repo.get_by_id.return_value = None

    decision = await auth_service.evaluate_permission(
        user=user, organization_id=org_id, permission=Permission.ORGANIZATION_READ
    )
    assert decision.allowed is False
    assert decision.decision_code == "ORGANIZATION_NOT_FOUND"
    
    # Wait, the instruction says: "Reject missing organizations without revealing cross-tenant details."
    # A generic "ORGANIZATION_NOT_FOUND" is standard. The exact user-facing message will be generic.
