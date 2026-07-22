import uuid
from unittest.mock import AsyncMock

import pytest

from app.models.enums import MembershipStatus, OrganizationRole
from app.repositories.membership import MembershipRepository
from app.schemas.membership import MembershipUpdate
from app.services.membership import MembershipService
from app.core.exceptions import ValidationError, NotFoundError


@pytest.fixture
def mock_session():
    return AsyncMock()


@pytest.mark.asyncio
async def test_last_active_organization_admin_cannot_be_disabled(mock_session):
    service = MembershipService(mock_session)
    service.membership_repo = AsyncMock(spec=MembershipRepository)
    
    # Mock context
    context = AsyncMock()
    context.organization_id = uuid.uuid4()
    
    # Mock membership to be admin and active
    mock_membership = AsyncMock()
    mock_membership.role = OrganizationRole.ORGANIZATION_ADMIN
    mock_membership.status = MembershipStatus.ACTIVE
    service.membership_repo.get_by_id_for_organization.return_value = mock_membership
    
    # Mock active admins count to be 1
    service.membership_repo.count_active_admins_for_organization.return_value = 1
    
    update_in = MembershipUpdate(status=MembershipStatus.DISABLED)
    
    with pytest.raises(ValidationError) as exc_info:
        await service.update_member_authorized(context, uuid.uuid4(), update_in)
        
    assert exc_info.value.code == "LAST_ORGANIZATION_ADMIN_REQUIRED"


@pytest.mark.asyncio
async def test_last_active_organization_admin_cannot_be_demoted(mock_session):
    service = MembershipService(mock_session)
    service.membership_repo = AsyncMock(spec=MembershipRepository)
    
    context = AsyncMock()
    context.organization_id = uuid.uuid4()
    
    mock_membership = AsyncMock()
    mock_membership.role = OrganizationRole.ORGANIZATION_ADMIN
    mock_membership.status = MembershipStatus.ACTIVE
    service.membership_repo.get_by_id_for_organization.return_value = mock_membership
    
    service.membership_repo.count_active_admins_for_organization.return_value = 1
    
    update_in = MembershipUpdate(role=OrganizationRole.VIEWER)
    
    with pytest.raises(ValidationError) as exc_info:
        await service.update_member_authorized(context, uuid.uuid4(), update_in)
        
    assert exc_info.value.code == "LAST_ORGANIZATION_ADMIN_REQUIRED"


@pytest.mark.asyncio
async def test_second_administrator_allows_first_to_be_demoted(mock_session):
    service = MembershipService(mock_session)
    service.membership_repo = AsyncMock(spec=MembershipRepository)
    
    context = AsyncMock()
    context.organization_id = uuid.uuid4()
    
    mock_membership = AsyncMock()
    mock_membership.role = OrganizationRole.ORGANIZATION_ADMIN
    mock_membership.status = MembershipStatus.ACTIVE
    service.membership_repo.get_by_id_for_organization.return_value = mock_membership
    
    # Mock active admins count to be 2
    service.membership_repo.count_active_admins_for_organization.return_value = 2
    
    update_in = MembershipUpdate(role=OrganizationRole.VIEWER)
    
    # Should not raise exception
    await service.update_member_authorized(context, uuid.uuid4(), update_in)
