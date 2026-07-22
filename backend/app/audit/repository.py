import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditAction, AuditOutcome, AuditResourceType
from app.models.audit_event import AuditEvent


class AuditEventRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, event: AuditEvent) -> AuditEvent:
        """Add an audit event without automatically committing."""
        self.session.add(event)
        return event

    async def get_by_id_for_organization(
        self,
        *,
        event_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> AuditEvent | None:
        """Get an audit event for a specific organization."""
        stmt = select(AuditEvent).where(
            AuditEvent.id == event_id,
            AuditEvent.organization_id == organization_id,
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def list_for_organization(
        self,
        *,
        organization_id: uuid.UUID,
        offset: int,
        limit: int,
        actor_user_id: uuid.UUID | None = None,
        action: AuditAction | None = None,
        outcome: AuditOutcome | None = None,
        resource_type: AuditResourceType | None = None,
        resource_id: uuid.UUID | None = None,
        workflow_id: uuid.UUID | None = None,
        occurred_from: datetime | None = None,
        occurred_to: datetime | None = None,
    ) -> list[AuditEvent]:
        """List audit events for a specific organization with deterministic ordering."""
        stmt = select(AuditEvent).where(AuditEvent.organization_id == organization_id)

        if actor_user_id:
            stmt = stmt.where(AuditEvent.actor_user_id == actor_user_id)
        if action:
            stmt = stmt.where(AuditEvent.action == action)
        if outcome:
            stmt = stmt.where(AuditEvent.outcome == outcome)
        if resource_type:
            stmt = stmt.where(AuditEvent.resource_type == resource_type)
        if resource_id:
            stmt = stmt.where(AuditEvent.resource_id == resource_id)
        if workflow_id:
            stmt = stmt.where(AuditEvent.workflow_id == workflow_id)
        if occurred_from:
            stmt = stmt.where(AuditEvent.occurred_at >= occurred_from)
        if occurred_to:
            stmt = stmt.where(AuditEvent.occurred_at <= occurred_to)

        # Deterministic ordering
        stmt = stmt.order_by(AuditEvent.occurred_at.desc(), AuditEvent.id.desc())
        stmt = stmt.offset(offset).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_for_organization(
        self,
        *,
        organization_id: uuid.UUID,
        actor_user_id: uuid.UUID | None = None,
        action: AuditAction | None = None,
        outcome: AuditOutcome | None = None,
        resource_type: AuditResourceType | None = None,
        resource_id: uuid.UUID | None = None,
        workflow_id: uuid.UUID | None = None,
        occurred_from: datetime | None = None,
        occurred_to: datetime | None = None,
    ) -> int:
        """Count audit events for a specific organization."""
        stmt = select(func.count()).select_from(AuditEvent).where(AuditEvent.organization_id == organization_id)

        if actor_user_id:
            stmt = stmt.where(AuditEvent.actor_user_id == actor_user_id)
        if action:
            stmt = stmt.where(AuditEvent.action == action)
        if outcome:
            stmt = stmt.where(AuditEvent.outcome == outcome)
        if resource_type:
            stmt = stmt.where(AuditEvent.resource_type == resource_type)
        if resource_id:
            stmt = stmt.where(AuditEvent.resource_id == resource_id)
        if workflow_id:
            stmt = stmt.where(AuditEvent.workflow_id == workflow_id)
        if occurred_from:
            stmt = stmt.where(AuditEvent.occurred_at >= occurred_from)
        if occurred_to:
            stmt = stmt.where(AuditEvent.occurred_at <= occurred_to)

        result = await self.session.execute(stmt)
        return result.scalar() or 0
