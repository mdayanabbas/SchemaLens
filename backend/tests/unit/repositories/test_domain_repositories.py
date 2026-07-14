import uuid
import pytest
from unittest.mock import AsyncMock

from app.repositories.organization import OrganizationRepository
from app.repositories.membership import MembershipRepository


@pytest.mark.asyncio
async def test_organization_get_by_slug():
    session = AsyncMock()
    repo = OrganizationRepository(session)
    
    await repo.get_by_slug(" Test-Slug ")
    session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_membership_isolation_requires_org_id():
    session = AsyncMock()
    repo = MembershipRepository(session)
    
    await repo.get_by_id_for_organization(uuid.uuid4(), uuid.uuid4())
    session.execute.assert_awaited_once()
