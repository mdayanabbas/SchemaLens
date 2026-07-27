import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
from app.audit.schemas import AuditEventCreate
from app.audit.service import AuditService
from app.core.config import get_settings
from app.core.exceptions import ConflictError, ExternalServiceError, NotFoundError
from app.governance.context import AuthorizedOrganizationContext
from app.models.schema_scan import SchemaScan
from app.models.schema_scan_enums import (
    SchemaScanFailureStage,
    SchemaScanStatus,
    SchemaScanTrigger,
)
from app.repositories.connection_policy import ConnectionPolicyRepository
from app.repositories.database_connection import DatabaseConnectionRepository
from app.repositories.schema_scan import SchemaScanRepository
from app.repositories.schema_scan_transition import SchemaScanTransitionRepository
from app.schemas.schema_scan import (
    SchemaScanCancelResponse,
    SchemaScanCreate,
    SchemaScanPage,
    SchemaScanRead,
    SchemaScanSummaryRead,
    SchemaScanTransitionRead,
)
from app.services.schema_scan_validation import SchemaScanValidator
from app.workers.cancellation import TaskCancellationService
from app.workers.dispatcher import TaskDispatcherProtocol
from app.workflows.schema_scan_state_machine import SchemaScanStateMachine


class SchemaScanService:
    def __init__(
        self,
        session: AsyncSession,
        audit_service: AuditService,
        dispatcher: TaskDispatcherProtocol,
        cancellation_service: TaskCancellationService,
    ):
        self.session = session
        self.audit_service = audit_service
        self.dispatcher = dispatcher
        self.cancellation_service = cancellation_service
        
        self.scan_repo = SchemaScanRepository(session)
        self.transition_repo = SchemaScanTransitionRepository(session)
        self.conn_repo = DatabaseConnectionRepository(session)
        self.policy_repo = ConnectionPolicyRepository(session)
        self.state_machine = SchemaScanStateMachine(session)
        self.validator = SchemaScanValidator()
        self.settings = get_settings()

    async def request_scan(
        self,
        *,
        connection_id: uuid.UUID,
        schema: SchemaScanCreate,
        context: AuthorizedOrganizationContext,
    ) -> SchemaScanRead:
        """
        Request a new schema scan.
        """
        context.require_permission("schemas.scan")

        # Load connection and policy
        connection = await self.conn_repo.get_by_id_for_organization(connection_id, context.organization_id)
        if not connection:
            raise NotFoundError("Connection not found.", code="CONNECTION_NOT_FOUND")
            
        policy = await self.policy_repo.get_by_connection_id_for_organization(connection_id, context.organization_id)
        if not policy:
            raise NotFoundError("Connection policy not found.", code="POLICY_NOT_FOUND")

        # Check for active scan (row-level protection in logic, database has unique constraint)
        active_scan = await self.scan_repo.get_active_for_connection(
            connection_id=connection_id, organization_id=context.organization_id, for_update=True
        )

        effective_schemas = self.validator.validate_scan_eligibility(
            connection=connection,
            policy=policy,
            active_scan=active_scan,
            requested_schemas=schema.requested_schemas,
        )

        scan_id = uuid.uuid4()
        scan = SchemaScan(
            id=scan_id,
            organization_id=context.organization_id,
            connection_id=connection_id,
            requested_by_user_id=context.user_id,
            trigger=SchemaScanTrigger.MANUAL,
            status=SchemaScanStatus.QUEUED,
            requested_schemas_json=effective_schemas,
            max_attempts=self.settings.schema_scan_max_attempts,
        )
        self.scan_repo.add(scan)

        # Initial transition
        await self.state_machine.transition(
            scan=scan,
            to_status=SchemaScanStatus.QUEUED,
            actor_type=AuditActorType.USER,
            actor_user_id=context.user_id,
            reason_code="MANUAL_SCAN_REQUESTED",
        )

        actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER
        await self.audit_service.record_success(AuditEventCreate(
            organization_id=context.organization_id,
            actor_user_id=context.user_id,
            actor_type=actor_type,
            action=AuditAction.SCHEMA_SCAN_REQUESTED,
            outcome=AuditOutcome.SUCCEEDED,
            resource_type=AuditResourceType.SCHEMA_SCAN,
            resource_id=scan.id,
            metadata={
                "connection_id": str(connection.id),
                "trigger": SchemaScanTrigger.MANUAL.value,
                "requested_schema_count": len(effective_schemas)
            }
        ))

        # Commit before dispatching so the worker sees the task
        await self.session.commit()

        # Dispatch
        try:
            task_id = await self.dispatcher.dispatch_schema_scan(
                scan_id=scan.id,
                organization_id=context.organization_id,
                connection_id=connection_id,
            )
            # Persist task_id in short transaction
            scan.worker_task_id = task_id
            self.session.add(scan)
            
            await self.audit_service.record_success(AuditEventCreate(
                organization_id=context.organization_id,
                actor_user_id=None,
                actor_type=AuditActorType.SYSTEM,
                action=AuditAction.SCHEMA_SCAN_DISPATCHED,
                outcome=AuditOutcome.SUCCEEDED,
                resource_type=AuditResourceType.SCHEMA_SCAN,
                resource_id=scan.id,
            ))
            await self.session.commit()
            
        except ExternalServiceError as e:
            # Reopen session state, mark failed
            self.session.add(scan)
            scan.failure_stage = SchemaScanFailureStage.DISPATCH
            scan.safe_error_code = e.code
            scan.safe_error_message = e.message
            
            await self.state_machine.transition(
                scan=scan,
                to_status=SchemaScanStatus.FAILED,
                actor_type=AuditActorType.SYSTEM,
                actor_user_id=None,
                reason_code="DISPATCH_FAILED",
            )
            
            await self.audit_service.record_success(AuditEventCreate(
                organization_id=context.organization_id,
                actor_user_id=None,
                actor_type=AuditActorType.SYSTEM,
                action=AuditAction.SCHEMA_SCAN_DISPATCH_FAILED,
                outcome=AuditOutcome.SUCCEEDED,
                resource_type=AuditResourceType.SCHEMA_SCAN,
                resource_id=scan.id,
            ))
            await self.session.commit()
            raise e

        # Refresh for read response
        self.session.add(scan)
        return SchemaScanRead.model_validate(scan)

    async def get_scan(self, *, scan_id: uuid.UUID, context: AuthorizedOrganizationContext) -> SchemaScanRead:
        context.require_permission("schemas.read")
        scan = await self.scan_repo.get_by_id_for_organization(scan_id=scan_id, organization_id=context.organization_id)
        if not scan:
            raise NotFoundError("Schema scan not found.", code="SCHEMA_SCAN_NOT_FOUND")
        return SchemaScanRead.model_validate(scan)

    async def list_scans(
        self,
        *,
        connection_id: uuid.UUID,
        offset: int,
        limit: int,
        status: SchemaScanStatus | None,
        context: AuthorizedOrganizationContext,
    ) -> SchemaScanPage:
        context.require_permission("schemas.read")
        items = await self.scan_repo.list_for_connection(
            connection_id=connection_id,
            organization_id=context.organization_id,
            offset=offset,
            limit=limit,
            status=status,
        )
        total = await self.scan_repo.count_for_connection(
            connection_id=connection_id,
            organization_id=context.organization_id,
            status=status,
        )
        return SchemaScanPage(
            items=[SchemaScanSummaryRead.model_validate(i) for i in items],
            offset=offset,
            limit=limit,
            total=total,
            has_more=(offset + limit) < total
        )

    async def cancel_scan(
        self, *, scan_id: uuid.UUID, context: AuthorizedOrganizationContext
    ) -> SchemaScanCancelResponse:
        context.require_permission("schemas.scan")
        
        scan = await self.scan_repo.get_by_id_for_organization(
            scan_id=scan_id, organization_id=context.organization_id, for_update=True
        )
        if not scan:
            raise NotFoundError("Schema scan not found.", code="SCHEMA_SCAN_NOT_FOUND")

        # Terminal state -> idempotent
        terminal_states = {
            SchemaScanStatus.CANCELLED,
            SchemaScanStatus.SUCCEEDED,
            SchemaScanStatus.PARTIALLY_SUCCEEDED,
            SchemaScanStatus.FAILED,
        }
        
        if scan.status not in terminal_states and scan.status != SchemaScanStatus.CANCELLATION_REQUESTED:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            scan.cancellation_requested_at = now
            scan.cancellation_requested_by_user_id = context.user_id
            
            to_status = SchemaScanStatus.CANCELLATION_REQUESTED
            if scan.status == SchemaScanStatus.QUEUED:
                to_status = SchemaScanStatus.CANCELLED
                scan.completed_at = now
                scan.failure_stage = SchemaScanFailureStage.CANCELLATION
            
            await self.state_machine.transition(
                scan=scan,
                to_status=to_status,
                actor_type=AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER,
                actor_user_id=context.user_id,
                reason_code="MANUAL_CANCELLATION_REQUESTED",
            )
            
            await self.audit_service.record_success(AuditEventCreate(
                organization_id=context.organization_id,
                actor_user_id=context.user_id,
                actor_type=AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER,
                action=AuditAction.SCHEMA_SCAN_CANCELLATION_REQUESTED,
                outcome=AuditOutcome.SUCCEEDED,
                resource_type=AuditResourceType.SCHEMA_SCAN,
                resource_id=scan.id,
            ))

            await self.scan_repo.flush()

            # Attempt Celery Revoke if a worker task ID exists
            if scan.worker_task_id:
                await self.cancellation_service.request_revoke(scan.worker_task_id)

        return SchemaScanCancelResponse.model_validate(scan)

    async def list_scan_transitions(
        self, *, scan_id: uuid.UUID, offset: int, limit: int, context: AuthorizedOrganizationContext
    ) -> list[SchemaScanTransitionRead]:
        context.require_permission("schemas.read")
        
        # Verify scan exists first
        scan = await self.scan_repo.get_by_id_for_organization(scan_id=scan_id, organization_id=context.organization_id)
        if not scan:
            raise NotFoundError("Schema scan not found.", code="SCHEMA_SCAN_NOT_FOUND")
            
        transitions = await self.transition_repo.list_for_scan_and_organization(
            schema_scan_id=scan_id,
            organization_id=context.organization_id,
            offset=offset,
            limit=limit,
        )
        return [SchemaScanTransitionRead.model_validate(t) for t in transitions]
