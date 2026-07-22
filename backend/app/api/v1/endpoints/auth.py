from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_database_session
from app.core.config import Settings, get_settings
from app.models.user import User
from app.schemas.auth import (
    AuthenticatedUserRead,
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    TokenResponse,
)
from app.services.authentication import AuthenticationService

router = APIRouter()


def _get_client_ip(request: Request) -> str | None:
    if request.client:
        return request.client.host
    return None


def _get_user_agent(request: Request) -> str | None:
    ua = request.headers.get("user-agent")
    if ua:
        return ua[:255]
    return None


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    login_in: LoginRequest,
    session: AsyncSession = Depends(get_database_session),
    settings: Settings = Depends(get_settings),
):
    """Authenticate and return access and refresh tokens."""
    auth_service = AuthenticationService(session, settings)
    return await auth_service.login(
        email=login_in.email,
        password=login_in.password,
        ip_address=_get_client_ip(request),
        user_agent=_get_user_agent(request),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    request: Request,
    refresh_in: RefreshRequest,
    session: AsyncSession = Depends(get_database_session),
    settings: Settings = Depends(get_settings),
):
    """Refresh access token using a valid refresh token."""
    auth_service = AuthenticationService(session, settings)
    return await auth_service.refresh(
        refresh_token=refresh_in.refresh_token,
        ip_address=_get_client_ip(request),
        user_agent=_get_user_agent(request),
    )


@router.post("/logout")
async def logout(
    request: Request,
    logout_in: LogoutRequest,
    session: AsyncSession = Depends(get_database_session),
    settings: Settings = Depends(get_settings),
):
    """Log out a session idempotently by revoking the refresh token."""
    auth_service = AuthenticationService(session, settings)
    await auth_service.logout(
        refresh_token=logout_in.refresh_token,
        ip_address=_get_client_ip(request),
        user_agent=_get_user_agent(request),
    )
    return {"status": "logged_out"}


@router.get("/me", response_model=AuthenticatedUserRead)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """Get the currently authenticated user's profile safely."""
    return current_user


@router.get("/me/organizations")
async def get_current_user_organizations(
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_database_session),
):
    """Get the active organization memberships for the current user."""
    from app.services.organization import OrganizationService
    from app.schemas.organization import OrganizationSummaryRead
    
    org_service = OrganizationService(session)
    return await org_service.list_for_user(
        user=current_user,
        offset=offset,
        limit=limit,
    )
