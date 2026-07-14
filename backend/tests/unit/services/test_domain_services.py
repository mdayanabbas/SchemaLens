import pytest
from unittest.mock import AsyncMock

from app.core.exceptions import ConflictError
from app.schemas.organization import OrganizationCreate
from app.services.organization import OrganizationService
from app.services.user import UserService
from app.schemas.user import UserCreate


@pytest.mark.asyncio
async def test_create_organization_conflict():
    session = AsyncMock()
    session.in_transaction.return_value = False
    
    service = OrganizationService(session)
    service.repository.slug_exists = AsyncMock(return_value=True)
    
    with pytest.raises(ConflictError):
        await service.create_organization(OrganizationCreate(name="Test", slug="test"))


@pytest.mark.asyncio
async def test_create_user_conflict():
    session = AsyncMock()
    session.in_transaction.return_value = False
    
    service = UserService(session)
    service.repository.email_exists = AsyncMock(return_value=True)
    
    with pytest.raises(ConflictError):
        await service.create_user(UserCreate(email="test@example.com", display_name="Test"))
