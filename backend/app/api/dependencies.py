from typing import AsyncGenerator

from fastapi import Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_database_session
from app.models.user import User


bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    token: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_database_session),
    settings: Settings = Depends(get_settings),
) -> User:
    from app.services.authentication import AuthenticationService, AuthenticationException
    
    if not token:
        raise AuthenticationException("Not authenticated.", code="NOT_AUTHENTICATED")
        
    auth_service = AuthenticationService(session, settings)
    return await auth_service.authenticate_access_token(token.credentials)

__all__ = ["get_database_session", "get_current_user"]
