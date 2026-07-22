import uuid
from datetime import datetime, UTC

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
from app.audit.schemas import AuditEventCreate
from app.audit.service import AuditService


@pytest.mark.asyncio
async def test_audit_service_record(db_session: AsyncSession):
    audit_service = AuditService(db_session)
    org_id = uuid.uuid4()
    actor_id = uuid.uuid4()
    
    event_in = AuditEventCreate(
        organization_id=org_id,
        actor_user_id=actor_id,
        actor_type=AuditActorType.USER,
        action=AuditAction.ORGANIZATION_CREATED,
        outcome=AuditOutcome.SUCCEEDED,
        resource_type=AuditResourceType.ORGANIZATION,
        metadata={"key": "value", "password": "super_secret"},
    )
    
    event = await audit_service.record(event_in)
    
    assert event.id is not None
    assert event.organization_id == org_id
    assert event.actor_user_id == actor_id
    assert event.action == AuditAction.ORGANIZATION_CREATED
    assert event.outcome == AuditOutcome.SUCCEEDED
    assert event.metadata_json["key"] == "value"
    assert event.metadata_json["password"] == "[REDACTED]"
    
    # Test repository listing
    events = await audit_service.repository.list_for_organization(
        organization_id=org_id,
        offset=0,
        limit=10,
    )
    assert len(events) == 1
    assert events[0].id == event.id


@pytest.mark.asyncio
async def test_audit_service_record_success_failure_denial(db_session: AsyncSession):
    audit_service = AuditService(db_session)
    org_id = uuid.uuid4()
    
    base_event = AuditEventCreate(
        organization_id=org_id,
        actor_type=AuditActorType.SYSTEM,
        action=AuditAction.API_REQUEST,
        outcome=AuditOutcome.SUCCEEDED, # will be overwritten
        resource_type=AuditResourceType.SYSTEM,
    )
    
    e1 = await audit_service.record_success(base_event.model_copy())
    assert e1.outcome == AuditOutcome.SUCCEEDED
    
    e2 = await audit_service.record_failure(base_event.model_copy())
    assert e2.outcome == AuditOutcome.FAILED
    
    e3 = await audit_service.record_denial(base_event.model_copy())
    assert e3.outcome == AuditOutcome.DENIED
