import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
from app.audit.schemas import AuditEventCreate
from app.audit.service import AuditService
from app.core.config import get_settings
from app.core.exceptions import AppError, WorkflowStateError
from app.models.connection_policy import ConnectionPolicy
from app.models.database_connection import DatabaseConnection
from app.models.schema_scan import SchemaScan
from app.models.schema_scan_enums import SchemaScanFailureStage, SchemaScanStatus
from app.repositories.connection_policy import ConnectionPolicyRepository
from app.repositories.database_connection import DatabaseConnectionRepository
from app.repositories.schema_scan import SchemaScanRepository
from app.services.schema_scan_validation import SchemaScanValidator
from app.workflows.schema_scan_state_machine import SchemaScanStateMachine


class SchemaScanWorkflow:
    def __init__(self, session: AsyncSession, audit_service: AuditService):
        self.session = session
        self.audit_service = audit_service
        self.scan_repo = SchemaScanRepository(session)
        self.conn_repo = DatabaseConnectionRepository(session)
        self.policy_repo = ConnectionPolicyRepository(session)
        self.state_machine = SchemaScanStateMachine(session)
        self.validator = SchemaScanValidator()
        self.settings = get_settings()

    async def run(
        self,
        *,
        scan_id: uuid.UUID,
        organization_id: uuid.UUID,
        connection_id: uuid.UUID,
    ) -> None:
        """
        Main worker execution flow for a schema scan.
        """
        scan = await self.claim_scan(scan_id=scan_id, organization_id=organization_id)
        if not scan:
            return

        # Double check connection ID
        if scan.connection_id != connection_id:
            await self.fail_scan(
                scan=scan,
                failure_stage=SchemaScanFailureStage.WORKER_START,
                error_code="CONNECTION_MISMATCH",
                error_message="Worker received mismatched connection ID.",
            )
            return

        # Validate eligibility
        eligibility_error = await self.validate_worker_eligibility(scan=scan)
        if eligibility_error:
            await self.fail_scan(
                scan=scan,
                failure_stage=SchemaScanFailureStage.POLICY_VALIDATION,
                error_code=eligibility_error["code"],
                error_message=eligibility_error["message"],
            )
            return

        # Update Phase
        scan.progress_phase = "awaiting_introspection"
        await self.update_heartbeat(scan=scan)

        if await self.check_cancellation(scan=scan):
            return

        # Setup connectors and services
        from app.connectors.registry import get_connector_for_dialect
        from app.services.schema_snapshot_persistence import SchemaSnapshotPersistenceService

        connection = await self.conn_repo.get_by_id_for_organization(connection_id, organization_id)
        policy = await self.policy_repo.get_by_connection_id_for_organization(connection_id, organization_id)
        
        if not connection or not policy:
            # Caught by validate_worker_eligibility already, but just in case
            return
            
        connector = get_connector_for_dialect(connection.dialect)
        
        async def cancel_check():
            if await self.check_cancellation(scan=scan):
                raise asyncio.CancelledError("Scan cancelled during introspection.")
                
        async def progress_cb(phase: str, current: int, total: int):
            scan.progress_phase = f"introspecting_{phase}"
            await self.update_heartbeat(scan=scan)
            
        # 1. Introspection Phase
        try:
            introspection_result = await connector.introspect_schema(
                organization_id=organization_id,
                connection=connection,
                policy=policy,
                schemas=scan.requested_schemas_json,
                cancellation_check=cancel_check,
                progress_callback=progress_cb,
            )
        except asyncio.CancelledError:
            return
        except AppError as e:
            await self.fail_scan(
                scan=scan,
                failure_stage=SchemaScanFailureStage.INTROSPECTION,
                error_code=e.code,
                error_message=e.message,
            )
            return
        except Exception as e:
            await self.fail_scan(
                scan=scan,
                failure_stage=SchemaScanFailureStage.INTROSPECTION,
                error_code="SCHEMA_INTROSPECTION_ERROR",
                error_message=str(e),
            )
            return
            
        # 2. Persistence Phase
        scan.progress_phase = "persisting_snapshot"
        await self.update_heartbeat(scan=scan)
        
        try:
            persistence_service = SchemaSnapshotPersistenceService(self.session, self.settings)
            snapshot = await persistence_service.persist_and_promote(
                organization_id=organization_id,
                connection_id=connection_id,
                scan_id=scan.id,
                introspection_result=introspection_result
            )
        except asyncio.CancelledError:
            return
        except Exception as e:
            await self.fail_scan(
                scan=scan,
                failure_stage=SchemaScanFailureStage.SNAPSHOT_PERSISTENCE,
                error_code="SNAPSHOT_PERSISTENCE_FAILED",
                error_message=str(e),
            )
            return
            
        # 3. Finalize
        scan.progress_phase = "completed"
        scan.completed_at = self._now()
        
        final_status = SchemaScanStatus.PARTIALLY_SUCCEEDED if introspection_result.warnings else SchemaScanStatus.SUCCEEDED
        
        await self.state_machine.transition(
            scan=scan,
            to_status=final_status,
            actor_type=AuditActorType.WORKER,
            actor_user_id=None,
            reason_code="WORKER_SUCCEEDED",
        )
        
        await self.audit_service.record_success(AuditEventCreate(
            organization_id=scan.organization_id,
            actor_user_id=None,
            actor_type=AuditActorType.WORKER,
            action=AuditAction.SCHEMA_SCAN_SUCCEEDED,
            outcome=AuditOutcome.SUCCEEDED,
            resource_type=AuditResourceType.SCHEMA_SCAN,
            resource_id=scan.id,
            metadata={"snapshot_id": str(snapshot.id)}
        ))

    async def claim_scan(
        self, *, scan_id: uuid.UUID, organization_id: uuid.UUID
    ) -> SchemaScan | None:
        """
        Attempts to claim the scan, locking the row and advancing state if possible.
        """
        scan = await self.scan_repo.get_by_id_for_organization(
            scan_id=scan_id, organization_id=organization_id, for_update=True
        )
        if not scan:
            return None

        # Exit idempotently if terminal
        terminal_states = {
            SchemaScanStatus.CANCELLED,
            SchemaScanStatus.SUCCEEDED,
            SchemaScanStatus.PARTIALLY_SUCCEEDED,
            SchemaScanStatus.FAILED,
        }
        if scan.status in terminal_states:
            return None

        if scan.status == SchemaScanStatus.CANCELLATION_REQUESTED:
            # Cancel immediately before doing anything
            await self.cancel_scan(scan=scan, reason="Cancellation requested before start.")
            return None

        if scan.status == SchemaScanStatus.QUEUED:
            try:
                await self.state_machine.transition(
                    scan=scan,
                    to_status=SchemaScanStatus.RUNNING,
                    actor_type=AuditActorType.WORKER,
                    actor_user_id=None,
                    reason_code="WORKER_CLAIMED",
                )
                scan.attempt_count += 1
                
                await self.audit_service.record_success(AuditEventCreate(
                    organization_id=scan.organization_id,
                    actor_user_id=None,
                    actor_type=AuditActorType.WORKER,
                    action=AuditAction.SCHEMA_SCAN_STARTED,
                    outcome=AuditOutcome.SUCCEEDED,
                    resource_type=AuditResourceType.SCHEMA_SCAN,
                    resource_id=scan.id,
                    metadata={"attempt": scan.attempt_count}
                ))
            except WorkflowStateError:
                return None
                
        elif scan.status == SchemaScanStatus.RUNNING:
            # Re-delivery of already running task. Attempt count shouldn't increment
            pass
            
        await self.update_heartbeat(scan, set_started=True)
        return scan

    async def check_cancellation(self, *, scan: SchemaScan) -> bool:
        """Checks if cancellation was requested and applies it."""
        # Refresh the scan to check for external cancellation requests
        current = await self.scan_repo.get_by_id_for_organization(
            scan_id=scan.id, organization_id=scan.organization_id, for_update=True
        )
        if not current:
            return True
            
        if current.status == SchemaScanStatus.CANCELLATION_REQUESTED:
            await self.cancel_scan(scan=current, reason="Cancellation requested during execution.")
            return True
            
        if current.status == SchemaScanStatus.CANCELLED:
            return True
            
        # Write back potential local changes if any
        scan.status = current.status
        return False

    async def update_heartbeat(self, scan: SchemaScan, set_started: bool = False) -> None:
        """Updates heartbeat_at and flushes."""
        now = self._now()
        scan.heartbeat_at = now
        if set_started and not scan.started_at:
            scan.started_at = now
        await self.scan_repo.flush()

    async def validate_worker_eligibility(self, *, scan: SchemaScan) -> dict[str, str] | None:
        """Re-validate connection and policy during worker run."""
        connection = await self.conn_repo.get_by_id_for_organization(
            scan.connection_id, scan.organization_id
        )
        if not connection:
            return {"code": "CONNECTION_NOT_FOUND", "message": "Connection was removed."}
            
        policy = await self.policy_repo.get_by_connection_id_for_organization(
            scan.connection_id, scan.organization_id
        )
        if not policy:
            return {"code": "POLICY_NOT_FOUND", "message": "Policy was removed."}

        try:
            self.validator.validate_scan_eligibility(
                connection=connection,
                policy=policy,
                active_scan=None, # Already claimed by us
                requested_schemas=scan.requested_schemas_json,
            )
            return None
        except AppError as e:
            return {"code": e.code, "message": e.message}

    async def fail_scan(
        self,
        *,
        scan: SchemaScan,
        failure_stage: SchemaScanFailureStage,
        error_code: str,
        error_message: str,
    ) -> None:
        """Fails the scan gracefully."""
        if scan.status in {SchemaScanStatus.FAILED, SchemaScanStatus.CANCELLED, SchemaScanStatus.SUCCEEDED, SchemaScanStatus.PARTIALLY_SUCCEEDED}:
            return
            
        try:
            scan.failure_stage = failure_stage
            scan.safe_error_code = error_code
            scan.safe_error_message = error_message
            scan.completed_at = self._now()
            
            await self.state_machine.transition(
                scan=scan,
                to_status=SchemaScanStatus.FAILED,
                actor_type=AuditActorType.WORKER,
                actor_user_id=None,
                reason_code="WORKER_FAILED",
                metadata={"error_code": error_code, "stage": failure_stage.value}
            )
            
            await self.audit_service.record_success(AuditEventCreate(
                organization_id=scan.organization_id,
                actor_user_id=None,
                actor_type=AuditActorType.WORKER,
                action=AuditAction.SCHEMA_SCAN_FAILED,
                outcome=AuditOutcome.SUCCEEDED,
                resource_type=AuditResourceType.SCHEMA_SCAN,
                resource_id=scan.id,
                metadata={
                    "error_code": error_code,
                    "stage": failure_stage.value
                }
            ))
            await self.scan_repo.flush()
        except WorkflowStateError:
            pass

    async def cancel_scan(self, *, scan: SchemaScan, reason: str) -> None:
        """Cancels a scan during worker execution."""
        if scan.status == SchemaScanStatus.CANCELLED:
            return
            
        try:
            scan.completed_at = self._now()
            scan.failure_stage = SchemaScanFailureStage.CANCELLATION
            
            await self.state_machine.transition(
                scan=scan,
                to_status=SchemaScanStatus.CANCELLED,
                actor_type=AuditActorType.WORKER,
                actor_user_id=None,
                reason_code="WORKER_CANCELLED",
            )
            
            await self.audit_service.record_success(AuditEventCreate(
                organization_id=scan.organization_id,
                actor_user_id=None,
                actor_type=AuditActorType.WORKER,
                action=AuditAction.SCHEMA_SCAN_CANCELLED,
                outcome=AuditOutcome.SUCCEEDED,
                resource_type=AuditResourceType.SCHEMA_SCAN,
                resource_id=scan.id,
            ))
            await self.scan_repo.flush()
        except WorkflowStateError:
            pass
            
    def _now(self):
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)
