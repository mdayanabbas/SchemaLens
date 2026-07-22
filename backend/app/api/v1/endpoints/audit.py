import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_audit_service, get_database_session, require_permission
from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
from app.audit.schemas import AuditEventCreate, AuditEventPage, AuditEventRead
from app.audit.service import AuditService
from app.core.exceptions import NotFoundError
from app.governance.context import AuthorizedOrganizationContext
from app.governance.permissions import Permission


router = APIRouter()


@router.get("", response_model=AuditEventPage)
async def list_audit_events(
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    actor_user_id: uuid.UUID | None = None,
    action: AuditAction | None = None,
    outcome: AuditOutcome | None = None,
    resource_type: AuditResourceType | None = None,
    resource_id: uuid.UUID | None = None,
    workflow_id: uuid.UUID | None = None,
    occurred_from: datetime | None = None,
    occurred_to: datetime | None = None,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.AUDIT_READ)),
    audit_service: AuditService = Depends(get_audit_service),
):
    """List audit events for the current organization."""
    if occurred_from and occurred_to and occurred_from > occurred_to:
        from app.core.exceptions import ValidationError
        raise ValidationError("occurred_from cannot be later than occurred_to", code="INVALID_DATE_RANGE")
        
    events = await audit_service.repository.list_for_organization(
        organization_id=context.organization_id,
        offset=offset,
        limit=limit,
        actor_user_id=actor_user_id,
        action=action,
        outcome=outcome,
        resource_type=resource_type,
        resource_id=resource_id,
        workflow_id=workflow_id,
        occurred_from=occurred_from,
        occurred_to=occurred_to,
    )
    
    total = await audit_service.repository.count_for_organization(
        organization_id=context.organization_id,
        actor_user_id=actor_user_id,
        action=action,
        outcome=outcome,
        resource_type=resource_type,
        resource_id=resource_id,
        workflow_id=workflow_id,
        occurred_from=occurred_from,
        occurred_to=occurred_to,
    )
    
    # Exclude ip_hash and user_agent_hash from public read schemas implicitly via AuditEventRead ignoring them.
    # Write summary event
    actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER
    filters = {}
    if actor_user_id: filters["actor_user_id"] = True
    if action: filters["action"] = True
    if outcome: filters["outcome"] = True
    if resource_type: filters["resource_type"] = True
    if resource_id: filters["resource_id"] = True
    if workflow_id: filters["workflow_id"] = True
    
    await audit_service.record_success(AuditEventCreate(
        organization_id=context.organization_id,
        actor_user_id=context.user_id,
        actor_type=actor_type,
        action=AuditAction.AUDIT_EVENTS_VIEWED,
        outcome=AuditOutcome.SUCCEEDED,
        resource_type=AuditResourceType.AUDIT_EVENT,
        metadata={"operation": "list", "filters_used": list(filters.keys()), "returned_count": len(events)}
    ))
    
    return AuditEventPage(
        items=[AuditEventRead.model_validate(e) for e in events],
        offset=offset,
        limit=limit,
        total=total,
        has_more=(offset + len(events) < total),
    )


@router.get("/{event_id}", response_model=AuditEventRead)
async def get_audit_event(
    event_id: uuid.UUID,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.AUDIT_READ)),
    audit_service: AuditService = Depends(get_audit_service),
):
    """Get a specific audit event safely scoped to the organization."""
    event = await audit_service.repository.get_by_id_for_organization(
        event_id=event_id,
        organization_id=context.organization_id,
    )
    
    if not event:
        raise NotFoundError("Audit event not found.", code="AUDIT_EVENT_NOT_FOUND")
        
    actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER
    await audit_service.record_success(AuditEventCreate(
        organization_id=context.organization_id,
        actor_user_id=context.user_id,
        actor_type=actor_type,
        action=AuditAction.AUDIT_EVENTS_VIEWED,
        outcome=AuditOutcome.SUCCEEDED,
        resource_type=AuditResourceType.AUDIT_EVENT,
        resource_id=event.id,
        metadata={"operation": "detail"}
    ))
    
    return AuditEventRead.model_validate(event)
