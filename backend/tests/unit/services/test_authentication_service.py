import uuid
import pytest
from unittest.mock import AsyncMock

from app.core.config import Settings
from app.core.exceptions import AppError
from app.services.authentication import AuthenticationService, AuthenticationException
from app.schemas.auth import LoginRequest


@pytest.mark.asyncio
async def test_login_invalid_user():
    session = AsyncMock()
    session.in_transaction.return_value = False
    
    settings = Settings(
        jwt_secret_key="secret",
        refresh_token_pepper="pepper"
    )
    
    service = AuthenticationService(session, settings)
    service.user_repo.get_active_by_email = AsyncMock(return_value=None)
    
    with pytest.raises(AuthenticationException) as excinfo:
        await service.login("test@example.com", "password")
        
    # Generic error
    assert excinfo.value.code == "INVALID_CREDENTIALS"
