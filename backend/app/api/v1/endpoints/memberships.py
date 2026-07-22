import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_database_session, require_permission
from app.governance.context import AuthorizedOrganizationContext
from app.governance.permissions import Permission
from app.models.enums import MembershipStatus, OrganizationRole
from app.schemas.membership import MembershipCreate, MembershipDetailedRead, MembershipRead, MembershipUpdate
from app.services.membership import MembershipService


router = APIRouter()


@router.get("", response_model=list[MembershipDetailedRead])
async def list_memberships(
    limit: int = 100,
    offset: int = 0,
    role: OrganizationRole | None = None,
    status: MembershipStatus | None = None,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.MEMBERS_READ)),
    session: AsyncSession = Depends(get_database_session),
):
    """List memberships for the current organization."""
    membership_service = MembershipService(session)
    return await membership_service.list_organization_memberships_authorized(
        context=context,
        limit=limit,
        offset=offset,
        role=role,
        status=status,
    )


@router.post("", response_model=MembershipRead)
async def add_membership(
    membership_in: MembershipCreate,
    exact_email: str | None = None,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.MEMBERS_MANAGE)),
    session: AsyncSession = Depends(get_database_session),
):
    """Add a new member to the current organization."""
    membership_service = MembershipService(session)
    return await membership_service.add_member_authorized(
        context=context,
        membership_in=membership_in,
        exact_email=exact_email,
    )


@router.patch("/{membership_id}", response_model=MembershipRead)
async def update_membership(
    membership_id: uuid.UUID,
    update_in: MembershipUpdate,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.MEMBERS_MANAGE)),
    session: AsyncSession = Depends(get_database_session),
):
    """Update a membership's role or status safely."""
    membership_service = MembershipService(session)
    return await membership_service.update_member_authorized(
        context=context,
        membership_id=membership_id,
        update_in=update_in,
    )


@router.post("/{membership_id}/disable", response_model=MembershipRead)
async def disable_membership(
    membership_id: uuid.UUID,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.MEMBERS_MANAGE)),
    session: AsyncSession = Depends(get_database_session),
):
    """Disable a membership safely."""
    membership_service = MembershipService(session)
    return await membership_service.disable_member_authorized(
        context=context,
        membership_id=membership_id,
    )
