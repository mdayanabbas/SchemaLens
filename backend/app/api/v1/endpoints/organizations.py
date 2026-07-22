from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_database_session, get_organization_context, require_permission
from app.governance.context import AuthorizedOrganizationContext
from app.governance.permissions import Permission
from app.models.user import User
from app.schemas.organization import OrganizationRead, OrganizationSummaryRead, OrganizationUpdate
from app.services.organization import OrganizationService


router = APIRouter()


@router.get("", response_model=list[OrganizationSummaryRead])
async def list_organizations(
    limit: int = 100,
    offset: int = 0,
    platform_admin_access: bool = False,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_database_session),
):
    """List organizations available to the user."""
    org_service = OrganizationService(session)
    return await org_service.list_for_user(
        user=current_user,
        offset=offset,
        limit=limit,
        platform_admin_access=platform_admin_access,
    )


@router.get("/current", response_model=dict)
async def get_current_organization(
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.ORGANIZATION_READ)),
    session: AsyncSession = Depends(get_database_session),
):
    """Get details of the current organization and authorization context."""
    org_service = OrganizationService(session)
    # The get_organization method takes slug, we don't have slug easily here, wait
    # We should add get_organization_by_id in OrganizationService or use repository directly.
    # Let's use repository or add get_organization_by_id in org_service.
    org = await org_service.repository.get_by_id(context.organization_id)
    org_data = OrganizationRead.model_validate(org).model_dump(mode="json")
    
    return {
        "organization": org_data,
        "context": {
            "role": context.role,
            "permissions": list(context.permissions),
            "is_platform_admin": context.is_platform_admin,
        }
    }


@router.patch("/current", response_model=OrganizationRead)
async def update_current_organization(
    update_in: OrganizationUpdate,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.ORGANIZATION_MANAGE)),
    session: AsyncSession = Depends(get_database_session),
):
    """Update the current organization."""
    org_service = OrganizationService(session)
    return await org_service.update_organization_authorized(
        context=context,
        update_in=update_in,
    )
