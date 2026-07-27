import uuid
from typing import Mapping

from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditActorType
from app.core.exceptions import WorkflowStateError
from app.models.schema_scan import SchemaScan
from app.models.schema_scan_enums import SchemaScanStatus
from app.models.schema_scan_transition import SchemaScanTransition


class SchemaScanStateMachine:
    def __init__(self, session: AsyncSession):
        self.session = session

    _ALLOWED_TRANSITIONS = {
        None: {SchemaScanStatus.QUEUED},
        SchemaScanStatus.QUEUED: {
            SchemaScanStatus.RUNNING,
            SchemaScanStatus.CANCELLATION_REQUESTED,
            SchemaScanStatus.CANCELLED,
            SchemaScanStatus.FAILED,
        },
        SchemaScanStatus.RUNNING: {
            SchemaScanStatus.CANCELLATION_REQUESTED,
            SchemaScanStatus.SUCCEEDED,
            SchemaScanStatus.PARTIALLY_SUCCEEDED,
            SchemaScanStatus.FAILED,
        },
        SchemaScanStatus.CANCELLATION_REQUESTED: {
            SchemaScanStatus.CANCELLED,
            SchemaScanStatus.FAILED,
        },
        SchemaScanStatus.PARTIALLY_SUCCEEDED: set(),
        SchemaScanStatus.SUCCEEDED: set(),
        SchemaScanStatus.FAILED: set(),
        SchemaScanStatus.CANCELLED: set(),
    }

    async def transition(
        self,
        *,
        scan: SchemaScan,
        to_status: SchemaScanStatus,
        actor_type: AuditActorType,
        actor_user_id: uuid.UUID | None,
        reason_code: str,
        metadata: Mapping[str, object] | None = None,
    ) -> SchemaScan:
        """
        Transitions a schema scan to a new status.
        Creates a SchemaScanTransition record in the same session.
        """
        from_status = scan.status if scan.id else None

        # Verify transition is allowed
        allowed_destinations = self._ALLOWED_TRANSITIONS.get(from_status, set())
        if to_status not in allowed_destinations:
            raise WorkflowStateError(
                message=f"Cannot transition schema scan from {from_status} to {to_status}.",
                details={
                    "from_status": from_status,
                    "to_status": to_status,
                },
            )

        # Apply state
        scan.status = to_status

        # Create transition record
        transition_record = SchemaScanTransition(
            id=uuid.uuid4(),
            organization_id=scan.organization_id,
            schema_scan_id=scan.id,
            from_status=from_status,
            to_status=to_status,
            actor_type=actor_type,
            actor_user_id=actor_user_id,
            reason_code=reason_code,
            safe_metadata_json=dict(metadata) if metadata else None,
        )
        self.session.add(transition_record)

        return scan
