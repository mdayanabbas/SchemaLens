import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
from app.audit.schemas import AuditEventCreate
from app.audit.service import AuditService
from app.connectors.exceptions import ConnectorError
from app.connectors.registry import ConnectorRegistry
from app.core.exceptions import NotFoundError, ValidationError
from app.governance.context import AuthorizedOrganizationContext
from app.models.connection_enums import ConnectionStatus, ConnectionTestStatus
from app.models.user import User
from app.repositories.database_connection import DatabaseConnectionRepository
from app.schemas.connection_test import ConnectionTestResponse


class ConnectionTestService:
    def __init__(self, session: AsyncSession, audit_service: AuditService, connector_registry: ConnectorRegistry):
        self.session = session
        self.audit_service = audit_service
        self.connector_registry = connector_registry
        self.repository = DatabaseConnectionRepository(session)

    async def test_connection(
        self,
        *,
        context: AuthorizedOrganizationContext,
        acting_user: User,
        connection_id: uuid.UUID,
    ) -> ConnectionTestResponse:
        
        # 1. Read connection and policy
        connection = await self.repository.get_by_id_for_organization(connection_id, context.organization_id)
        if not connection:
            raise NotFoundError("Connection not found.", code="CONNECTION_NOT_FOUND")
            
        if connection.status == ConnectionStatus.DISABLED:
            raise ValidationError("Cannot test a disabled connection.", code="CONNECTION_DISABLED")
            
        policy = connection.policy
        if not policy:
            raise ValidationError("Connection policy is missing.", code="POLICY_MISSING")

        connector = self.connector_registry.get(connection.dialect)
        
        # 2. End the read transaction to avoid holding it open during external network call
        await self.session.commit()
        
        # 3. Perform external test
        test_success = False
        test_result = None
        safe_error_code = None
        
        try:
            test_result = await connector.test_connection(
                organization_id=context.organization_id,
                connection=connection,
                policy=policy
            )
            test_success = True
        except ConnectorError as e:
            safe_error_code = e.code
            logger_msg = e.message
        except Exception as e:
            safe_error_code = "DATABASE_CONNECTION_TEST_FAILED"
            
        # 4. Re-fetch connection (session will implicitly begin a new transaction)
        connection = await self.repository.get_by_id_for_organization(connection_id, context.organization_id)
        if not connection:
            # It was deleted during the test
            raise NotFoundError("Connection not found.", code="CONNECTION_NOT_FOUND")

        actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER
        now = datetime.now(timezone.utc)

        # 5. Persist outcome and audit event
        connection.last_tested_at = now
        connection.updated_by_user_id = acting_user.id
        
        if test_success and test_result:
            connection.last_test_status = ConnectionTestStatus.SUCCEEDED
            connection.last_test_error_code = None
            
            await self.audit_service.record_success(AuditEventCreate(
                organization_id=context.organization_id,
                actor_user_id=acting_user.id,
                actor_type=actor_type,
                action=AuditAction.CONNECTION_TEST_SUCCEEDED,
                outcome=AuditOutcome.SUCCEEDED,
                resource_type=AuditResourceType.DATABASE_CONNECTION,
                resource_id=connection.id,
                metadata={
                    "connection_id": str(connection.id),
                    "dialect": connection.dialect,
                    "environment": connection.environment,
                    "server_version": test_result.server_version,
                    "approved_schemas_found": len(test_result.approved_schemas_found),
                    "approved_schemas_missing": len(test_result.approved_schemas_missing),
                    "warnings_count": len(test_result.warnings),
                    "latency_ms": test_result.latency_ms,
                }
            ))
            
            # Note: the router will commit this transaction
            return ConnectionTestResponse(
                success=True,
                connection_id=connection.id,
                dialect=connection.dialect,
                server_version=test_result.server_version,
                database_name=test_result.database_name,
                approved_schemas_found=test_result.approved_schemas_found,
                approved_schemas_missing=test_result.approved_schemas_missing,
                capabilities=test_result.capabilities,
                warnings=test_result.warnings,
                latency_ms=test_result.latency_ms,
                tested_at=now,
            )
            
        else:
            connection.last_test_status = ConnectionTestStatus.FAILED
            connection.last_test_error_code = safe_error_code
            
            await self.audit_service.record_success(AuditEventCreate(
                organization_id=context.organization_id,
                actor_user_id=acting_user.id,
                actor_type=actor_type,
                action=AuditAction.CONNECTION_TEST_FAILED,
                outcome=AuditOutcome.FAILED,
                resource_type=AuditResourceType.DATABASE_CONNECTION,
                resource_id=connection.id,
                metadata={
                    "connection_id": str(connection.id),
                    "dialect": connection.dialect,
                    "safe_error_code": safe_error_code,
                    "retryable": True, # For now assume network issues are retryable
                }
            ))
            
            # The API endpoint expects us to return an application error if connectivity failed
            # We must still raise an exception so the router turns it into an error response,
            # but wait, the router might roll back if we raise!
            # "A failed external connection test must still persist: failed test status, safe error code, audit event... Do not roll these back merely because the external network operation failed."
            # If we raise an exception, the SQLAlchemy middleware will ROLLBACK the transaction!
            # To persist the status, we MUST commit explicitly before raising!
            
            await self.session.commit()
            
            # Now we can safely raise
            from app.connectors.exceptions import ConnectorError
            # Just re-create a generic connector error to let the Exception handlers do their job
            raise ConnectorError(code=safe_error_code, message="Connection test failed.")
