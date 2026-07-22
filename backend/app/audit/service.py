import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditOutcome
from app.audit.repository import AuditEventRepository
from app.audit.sanitizer import AuditMetadataSanitizer
from app.audit.schemas import AuditEventCreate
from app.core.request_context import (
    get_request_id,
    get_request_organization_id,
    get_request_user_id,
)
from app.models.audit_event import AuditEvent

logger = logging.getLogger(__name__)


class AuditService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = AuditEventRepository(session)
        self.sanitizer = AuditMetadataSanitizer()

    async def record(self, event_in: AuditEventCreate) -> AuditEvent:
        """
        Record a sanitized, immutable audit event safely.
        """
        try:
            # Resolve implicit request context if not explicitly provided
            request_id = event_in.request_id or get_request_id()
            if request_id and len(request_id) > 255:
                request_id = request_id[:255]

            actor_user_id = event_in.actor_user_id or get_request_user_id()
            organization_id = event_in.organization_id or get_request_organization_id()

            # Sanitize metadata
            safe_metadata = self.sanitizer.sanitize(event_in.metadata)

            occurred_at = event_in.occurred_at or datetime.now(UTC)
            
            # Avoid naive datetime comparison errors by ensuring UTC timezone awareness
            if occurred_at.tzinfo is None:
                occurred_at = occurred_at.replace(tzinfo=UTC)

            event = AuditEvent(
                organization_id=organization_id,
                actor_user_id=actor_user_id,
                actor_type=event_in.actor_type,
                action=event_in.action,
                outcome=event_in.outcome,
                resource_type=event_in.resource_type,
                resource_id=event_in.resource_id,
                request_id=request_id,
                workflow_id=event_in.workflow_id,
                source=event_in.source,
                ip_hash=event_in.ip_hash,
                user_agent_hash=event_in.user_agent_hash,
                metadata_json=safe_metadata,
                occurred_at=occurred_at,
            )

            await self.repository.add(event)
            await self.session.flush()
            return event
        except Exception as e:
            # We must never crash the underlying business operation because the 
            # audit formatting failed unexpectedly. But we should log it.
            # However, for critical administrative actions, the caller should decide
            # if they want to fail. The instruction states:
            # "Do not expose persistence exceptions to clients."
            logger.error(f"Failed to record audit event: {e}", exc_info=True)
            raise e  # Allow caller to handle or fail closed. 
            # Wait, the instruction says "Do not expose persistence exceptions to clients." 
            # But "For critical administrative changes, failure to persist required audit evidence should fail the action closed."
            # So raising the error from the service is correct. The route or caller can catch and throw a 500 without leaking details.

    async def record_success(self, event_in: AuditEventCreate) -> AuditEvent:
        event_in.outcome = AuditOutcome.SUCCEEDED
        return await self.record(event_in)

    async def record_failure(self, event_in: AuditEventCreate) -> AuditEvent:
        event_in.outcome = AuditOutcome.FAILED
        return await self.record(event_in)

    async def record_denial(self, event_in: AuditEventCreate) -> AuditEvent:
        event_in.outcome = AuditOutcome.DENIED
        return await self.record(event_in)
