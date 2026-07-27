from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
from app.audit.schemas import AuditEventCreate
from app.audit.service import AuditService
from app.core.config import get_settings
from app.core.exceptions import WorkflowStateError
from app.models.schema_scan_enums import SchemaScanFailureStage, SchemaScanStatus
from app.repositories.schema_scan import SchemaScanRepository
from app.workflows.schema_scan_state_machine import SchemaScanStateMachine


@dataclass
class SchemaScanRecoveryResult:
    inspected: int = 0
    failed: int = 0
    cancelled: int = 0


class SchemaScanRecoveryService:
    def __init__(self, session: AsyncSession, audit_service: AuditService):
        self.session = session
        self.audit_service = audit_service
        self.scan_repo = SchemaScanRepository(session)
        self.state_machine = SchemaScanStateMachine(session)
        self.settings = get_settings()

    async def recover_stale_scans(self, *, limit: int) -> SchemaScanRecoveryResult:
        """
        Internal system task to recover scans that have not heartbeated.
        """
        now = self._now()
        threshold = now - timedelta(seconds=self.settings.schema_scan_stale_after_seconds)
        
        # We can't lock immediately in the list query efficiently across all rows without blocking,
        # so we fetch candidates first, then lock one by one.
        candidates = await self.scan_repo.list_stale_running(stale_before=threshold, limit=limit)
        
        result = SchemaScanRecoveryResult()
        
        for candidate in candidates:
            result.inspected += 1
            
            # Lock the row
            scan = await self.scan_repo.get_by_id_for_organization(
                scan_id=candidate.id, 
                organization_id=candidate.organization_id, 
                for_update=True
            )
            
            if not scan:
                continue
                
            # Recheck staleness
            is_stale = False
            if scan.status in {SchemaScanStatus.RUNNING, SchemaScanStatus.CANCELLATION_REQUESTED}:
                if scan.heartbeat_at:
                    if scan.heartbeat_at < threshold:
                        is_stale = True
                else:
                    if scan.created_at < threshold:
                        is_stale = True
            
            if not is_stale:
                # Was updated concurrently, skip
                continue

            scan.completed_at = now
            
            try:
                if scan.status == SchemaScanStatus.RUNNING:
                    scan.failure_stage = SchemaScanFailureStage.STALE_RECOVERY
                    scan.safe_error_code = "SCHEMA_SCAN_WORKER_LOST"
                    scan.safe_error_message = "The worker running this scan stopped responding."
                    
                    await self.state_machine.transition(
                        scan=scan,
                        to_status=SchemaScanStatus.FAILED,
                        actor_type=AuditActorType.SYSTEM,
                        actor_user_id=None,
                        reason_code="STALE_RECOVERY_FAILED",
                    )
                    
                    await self.audit_service.record_success(AuditEventCreate(
                        organization_id=scan.organization_id,
                        actor_user_id=None,
                        actor_type=AuditActorType.SYSTEM,
                        action=AuditAction.SCHEMA_SCAN_STALE_RECOVERED,
                        outcome=AuditOutcome.SUCCEEDED,
                        resource_type=AuditResourceType.SCHEMA_SCAN,
                        resource_id=scan.id,
                        metadata={"recovery_action": "failed"}
                    ))
                    
                    result.failed += 1
                    
                elif scan.status == SchemaScanStatus.CANCELLATION_REQUESTED:
                    scan.failure_stage = SchemaScanFailureStage.CANCELLATION
                    
                    await self.state_machine.transition(
                        scan=scan,
                        to_status=SchemaScanStatus.CANCELLED,
                        actor_type=AuditActorType.SYSTEM,
                        actor_user_id=None,
                        reason_code="STALE_RECOVERY_CANCELLED",
                    )
                    
                    await self.audit_service.record_success(AuditEventCreate(
                        organization_id=scan.organization_id,
                        actor_user_id=None,
                        actor_type=AuditActorType.SYSTEM,
                        action=AuditAction.SCHEMA_SCAN_STALE_RECOVERED,
                        outcome=AuditOutcome.SUCCEEDED,
                        resource_type=AuditResourceType.SCHEMA_SCAN,
                        resource_id=scan.id,
                        metadata={"recovery_action": "cancelled"}
                    ))
                    
                    result.cancelled += 1
                
                await self.scan_repo.flush()
                # Commit after each successful recovery to free locks
                await self.session.commit()
                
            except WorkflowStateError:
                await self.session.rollback()
            except Exception:
                await self.session.rollback()
                raise
                
        return result
        
    def _now(self):
        return datetime.now(timezone.utc)
