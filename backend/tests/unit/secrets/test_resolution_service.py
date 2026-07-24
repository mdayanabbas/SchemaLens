import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import ForbiddenError
from app.governance.context import AuthorizedOrganizationContext
from app.models.connection_enums import ConnectionStatus, SecretProviderType
from app.models.database_connection import DatabaseConnection
from app.secrets.service import SecretResolutionService


@pytest.fixture
def mock_session():
    return AsyncMock()


@pytest.fixture
def mock_audit_service():
    return AsyncMock()


@pytest.fixture
def service(mock_session, mock_audit_service):
    return SecretResolutionService(mock_session, mock_audit_service)


@pytest.mark.asyncio
async def test_resolve_for_connection_success(service, mock_audit_service, monkeypatch):
    org_id = uuid.uuid4()
    context = AuthorizedOrganizationContext(
        organization_id=org_id,
        user_id=uuid.uuid4(),
        is_platform_admin=True,
        permissions=set()
    )
    
    connection = DatabaseConnection(
        id=uuid.uuid4(),
        organization_id=org_id,
        status=ConnectionStatus.DRAFT,
        secret_provider=SecretProviderType.ENVIRONMENT,
        secret_reference="DB_CREDS"
    )
    
    mock_provider = AsyncMock()
    mock_provider.resolve.return_value = MagicMock(
        username="test_user", 
        password="pwd", 
        database=None, 
        host=None, 
        port=None,
        ssl_ca=None,
        ssl_cert=None,
        ssl_key=None,
        expires_at=None,
        provider_metadata={}
    )
    
    # Mock registry
    monkeypatch.setattr(service.registry, "get", lambda x: mock_provider)
    
    result = await service.resolve_for_connection(context=context, connection=connection)
    
    assert result.username == "test_user"
    mock_provider.resolve.assert_called_once_with(
        organization_id=org_id,
        reference="DB_CREDS"
    )
    mock_audit_service.record_success.assert_called_once()
    

@pytest.mark.asyncio
async def test_resolve_for_connection_fails_wrong_org(service):
    org_id = uuid.uuid4()
    wrong_org_id = uuid.uuid4()
    
    context = AuthorizedOrganizationContext(
        organization_id=org_id,
        user_id=uuid.uuid4(),
        is_platform_admin=True,
        permissions=set()
    )
    
    connection = DatabaseConnection(
        id=uuid.uuid4(),
        organization_id=wrong_org_id,
        status=ConnectionStatus.DRAFT,
        secret_provider=SecretProviderType.ENVIRONMENT,
        secret_reference="DB_CREDS"
    )
    
    with pytest.raises(ForbiddenError) as exc:
        await service.resolve_for_connection(context=context, connection=connection)
    assert "Connection does not belong to the current organization" in str(exc.value)
