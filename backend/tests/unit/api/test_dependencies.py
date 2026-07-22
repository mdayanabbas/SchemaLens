import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request
from starlette.datastructures import Headers

from app.api.dependencies import get_organization_context, require_permission
from app.core.exceptions import AuthorizationError
from app.governance.permissions import Permission
from app.models.user import User


@pytest.fixture
def mock_request():
    scope = {
        "type": "http",
        "headers": Headers({"x-organization-id": str(uuid.uuid4())}).raw,
    }
    return Request(scope)


@pytest.fixture
def mock_request_missing_header():
    scope = {
        "type": "http",
        "headers": [],
    }
    return Request(scope)


@pytest.fixture
def mock_request_invalid_uuid():
    scope = {
        "type": "http",
        "headers": Headers({"x-organization-id": "invalid-uuid"}).raw,
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_missing_organization_id_header(mock_request_missing_header):
    user = User(id=uuid.uuid4())
    session = AsyncMock()

    with pytest.raises(AuthorizationError) as exc_info:
        await get_organization_context(
            request=mock_request_missing_header, user=user, session=session
        )
    assert exc_info.value.code == "ORGANIZATION_CONTEXT_REQUIRED"


@pytest.mark.asyncio
async def test_malformed_uuid_header(mock_request_invalid_uuid):
    user = User(id=uuid.uuid4())
    session = AsyncMock()

    with pytest.raises(AuthorizationError) as exc_info:
        await get_organization_context(
            request=mock_request_invalid_uuid, user=user, session=session
        )
    assert exc_info.value.code == "INVALID_ORGANIZATION_CONTEXT"


@pytest.mark.asyncio
@patch("app.api.dependencies.AuthorizationService")
async def test_valid_organization_header_accepted(mock_auth_service_cls, mock_request):
    user = User(id=uuid.uuid4())
    session = AsyncMock()

    mock_auth_service = AsyncMock()
    mock_auth_service_cls.return_value = mock_auth_service
    
    mock_context = AsyncMock()
    mock_context.membership_id = uuid.uuid4()
    mock_context.role = "viewer"
    mock_auth_service.require_permission.return_value = mock_context

    with patch("app.api.dependencies.set_user_context") as mock_set_user, \
         patch("app.api.dependencies.set_organization_context") as mock_set_org:
        
        context = await get_organization_context(
            request=mock_request, user=user, session=session
        )
        
        assert context == mock_context
        mock_auth_service.require_permission.assert_called_once()
        mock_set_user.assert_called_once_with(user.id)
        mock_set_org.assert_called_once()
