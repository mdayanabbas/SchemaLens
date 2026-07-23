import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
from app.audit.schemas import AuditEventCreate
from app.audit.service import AuditService
from app.core.exceptions import NotFoundError, ValidationError
from app.governance.context import AuthorizedOrganizationContext
from app.models.connection_enums import ApprovalMode
from app.repositories.connection_policy import ConnectionPolicyRepository
from app.repositories.database_connection import DatabaseConnectionRepository
from app.schemas.connection_policy import ConnectionPolicyRead, ConnectionPolicyUpdate


class ConnectionPolicyService:
    def __init__(self, session: AsyncSession, audit_service: AuditService):
        self.session = session
        self.policy_repository = ConnectionPolicyRepository(session)
        self.connection_repository = DatabaseConnectionRepository(session)
        self.audit_service = audit_service

    async def get_policy(
        self, connection_id: uuid.UUID, context: AuthorizedOrganizationContext
    ) -> ConnectionPolicyRead:
        """Get the policy for a connection."""
        policy = await self.policy_repository.get_for_connection_and_organization(
            connection_id, context.organization_id
        )
        if not policy:
            raise NotFoundError("Policy not found.", code="POLICY_NOT_FOUND")

        return ConnectionPolicyRead.model_validate(policy)

    async def update_policy(
        self,
        connection_id: uuid.UUID,
        schema: ConnectionPolicyUpdate,
        context: AuthorizedOrganizationContext,
    ) -> ConnectionPolicyRead:
        """Update a connection policy."""
        # 1. Fetch connection to verify environment
        connection = await self.connection_repository.get_by_id_for_organization(connection_id, context.organization_id)
        if not connection:
            raise NotFoundError("Connection not found.", code="CONNECTION_NOT_FOUND")
            
        # 2. Lock policy for update
        policy = await self.policy_repository.get_for_connection_and_organization(
            connection_id, context.organization_id, for_update=True
        )
        if not policy:
            raise NotFoundError("Policy not found.", code="POLICY_NOT_FOUND")
            
        update_data = schema.model_dump(exclude_unset=True)
        if not update_data:
            return ConnectionPolicyRead.model_validate(policy)
            
        # 3. Cross-field validations requiring full state
        next_execution = update_data.get("allow_query_execution", policy.allow_query_execution)
        next_generation = update_data.get("allow_query_generation", policy.allow_query_generation)
        next_approval_mode = update_data.get("approval_mode", policy.approval_mode)
        
        if next_execution and not next_generation:
            raise ValidationError("Cannot enable execution while generation is disabled.", code="INVALID_CONNECTION_POLICY")
            
        if connection.environment == "production" and next_execution and next_approval_mode == ApprovalMode.NEVER:
            raise ValidationError("Production execution requires risk_based or always approval mode.", code="PRODUCTION_APPROVAL_REQUIRED")
            
        # 4. Apply changes
        changed_fields = []
        old_numeric_limits = {}
        new_numeric_limits = {}
        old_bool_flags = {}
        new_bool_flags = {}
        
        for field, value in update_data.items():
            old_value = getattr(policy, field)
            if field in ("approved_schemas", "blocked_schemas"):
                # Handle JSON fields separately since schema has _json prefix in model
                db_field = f"{field}_json"
                old_value = getattr(policy, db_field)
                if old_value != value:
                    setattr(policy, db_field, value)
                    changed_fields.append(field)
            else:
                if old_value != value:
                    setattr(policy, field, value)
                    changed_fields.append(field)
                    if isinstance(value, bool):
                        old_bool_flags[field] = old_value
                        new_bool_flags[field] = value
                    elif isinstance(value, (int, float)) and field not in ("created_by_user_id", "updated_by_user_id"):
                        old_numeric_limits[field] = old_value
                        new_numeric_limits[field] = value
                        
        policy.updated_by_user_id = context.user_id
        await self.policy_repository.flush()
        
        # 5. Audit
        actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER
        await self.audit_service.record_success(AuditEventCreate(
            organization_id=context.organization_id,
            actor_user_id=context.user_id,
            actor_type=actor_type,
            action=AuditAction.CONNECTION_POLICY_UPDATED,
            outcome=AuditOutcome.SUCCEEDED,
            resource_type=AuditResourceType.CONNECTION_POLICY,
            resource_id=policy.id,
            metadata={
                "connection_id": str(connection_id),
                "changed_fields": changed_fields,
                "old_boolean_flags": old_bool_flags,
                "new_boolean_flags": new_bool_flags,
                "old_numeric_limits": old_numeric_limits,
                "new_numeric_limits": new_numeric_limits,
                "approved_schema_count": len(policy.approved_schemas_json),
                "blocked_schema_count": len(policy.blocked_schemas_json),
            }
        ))
        
        return ConnectionPolicyRead.model_validate(policy)
