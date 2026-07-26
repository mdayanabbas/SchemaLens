import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
from app.audit.schemas import AuditEventCreate
from app.audit.service import AuditService
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.governance.context import AuthorizedOrganizationContext
from app.models.connection_enums import ConnectionStatus, ConnectionTestStatus
from app.models.connection_policy import ConnectionPolicy
from app.models.database_connection import DatabaseConnection
from app.repositories.database_connection import DatabaseConnectionRepository
from app.schemas.database_connection import (
    DatabaseConnectionCreate,
    DatabaseConnectionRead,
    DatabaseConnectionSummaryRead,
    DatabaseConnectionUpdate,
)
from app.connectors.pool_registry import ConnectionPoolRegistry
from app.services.connection_validation import validate_production_ssl, redact_secret_reference
from app.secrets.service import SecretResolutionService


class DatabaseConnectionService:
    def __init__(
        self,
        session: AsyncSession,
        audit_service: AuditService,
        pool_registry: ConnectionPoolRegistry | None = None
    ):
        self.session = session
        self.repository = DatabaseConnectionRepository(session)
        self.audit_service = audit_service
        self.secret_resolution_service = SecretResolutionService(session, audit_service)
        self.pool_registry = pool_registry

    async def create_connection(
        self,
        schema: DatabaseConnectionCreate,
        context: AuthorizedOrganizationContext,
    ) -> DatabaseConnectionRead:
        """Create a new database connection with a default policy."""
        # 1. Validate production SSL requirement
        validate_production_ssl(schema.environment, schema.ssl_mode)

        # 2. Validate secret reference
        await self.secret_resolution_service.validate_connection_reference(
            context=context,
            provider_type=schema.secret_provider,
            reference=schema.secret_reference,
        )

        # 3. Check name uniqueness
        if await self.repository.name_exists_for_organization(schema.name, context.organization_id):
            raise ConflictError("A connection with this name already exists.", code="CONNECTION_NAME_ALREADY_EXISTS")

        # 4. Create the DatabaseConnection
        connection = DatabaseConnection(
            id=uuid.uuid4(),
            organization_id=context.organization_id,
            name=schema.name,
            description=schema.description,
            environment=schema.environment,
            dialect=schema.dialect,
            host=schema.host,
            port=schema.port,
            database_name=schema.database_name,
            default_catalog=schema.default_catalog,
            ssl_mode=schema.ssl_mode,
            secret_provider=schema.secret_provider,
            secret_reference=schema.secret_reference,
            status=ConnectionStatus.DRAFT,
            last_test_status=ConnectionTestStatus.NEVER_TESTED,
            created_by_user_id=context.user_id,
            updated_by_user_id=context.user_id,
        )
        self.repository.add(connection)

        # 4. Create the secure default ConnectionPolicy
        policy = ConnectionPolicy(
            id=uuid.uuid4(),
            organization_id=context.organization_id,
            connection_id=connection.id,
            created_by_user_id=context.user_id,
            updated_by_user_id=context.user_id,
        )
        self.session.add(policy)

        # 5. Flush and Audit in the same transaction
        await self.repository.flush()

        actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER
        await self.audit_service.record_success(AuditEventCreate(
            organization_id=context.organization_id,
            actor_user_id=context.user_id,
            actor_type=actor_type,
            action=AuditAction.CONNECTION_CREATED,
            outcome=AuditOutcome.SUCCEEDED,
            resource_type=AuditResourceType.DATABASE_CONNECTION,
            resource_id=connection.id,
            metadata={
                "connection_name": connection.name,
                "environment": connection.environment,
                "dialect": connection.dialect,
                "host": connection.host,
                "secret_provider": connection.secret_provider,
            }
        ))

        return DatabaseConnectionRead.model_validate({
            **connection.__dict__,
            "redacted_secret_reference": redact_secret_reference(connection.secret_provider, connection.secret_reference)
        })

    async def get_connection(
        self, connection_id: uuid.UUID, context: AuthorizedOrganizationContext
    ) -> DatabaseConnectionRead:
        """Get connection details."""
        connection = await self.repository.get_by_id_for_organization(connection_id, context.organization_id)
        if not connection:
            raise NotFoundError("Connection not found.", code="CONNECTION_NOT_FOUND")

        return DatabaseConnectionRead.model_validate({
            **connection.__dict__,
            "redacted_secret_reference": redact_secret_reference(connection.secret_provider, connection.secret_reference)
        })

    async def list_connections(
        self,
        context: AuthorizedOrganizationContext,
        offset: int = 0,
        limit: int = 25,
        environment: str | None = None,
        status: str | None = None,
        dialect: str | None = None,
    ) -> tuple[list[DatabaseConnectionSummaryRead], int]:
        """List summary connections."""
        items = await self.repository.list_for_organization(
            context.organization_id,
            offset=offset,
            limit=limit,
            environment=environment,
            status=status,
            dialect=dialect,
        )
        total = await self.repository.count_for_organization(
            context.organization_id,
            environment=environment,
            status=status,
            dialect=dialect,
        )
        
        return [DatabaseConnectionSummaryRead.model_validate(c) for c in items], total

    async def update_connection(
        self,
        connection_id: uuid.UUID,
        schema: DatabaseConnectionUpdate,
        context: AuthorizedOrganizationContext,
    ) -> DatabaseConnectionRead:
        """Update a connection profile."""
        connection = await self.repository.get_by_id_for_organization(connection_id, context.organization_id)
        if not connection:
            raise NotFoundError("Connection not found.", code="CONNECTION_NOT_FOUND")
            
        update_data = schema.model_dump(exclude_unset=True)
        if not update_data:
            return DatabaseConnectionRead.model_validate({
                **connection.__dict__,
                "redacted_secret_reference": redact_secret_reference(connection.secret_provider, connection.secret_reference)
            })
            
        # Ensure production SSL requirement
        env_to_check = update_data.get("environment", connection.environment)
        ssl_to_check = update_data.get("ssl_mode", connection.ssl_mode)
        validate_production_ssl(env_to_check, ssl_to_check)
        
        if "name" in update_data and update_data["name"] != connection.name:
            if await self.repository.name_exists_for_organization(update_data["name"], context.organization_id):
                raise ConflictError("A connection with this name already exists.", code="CONNECTION_NAME_ALREADY_EXISTS")
                
        # Validate secret reference if changed
        if "secret_provider" in update_data or "secret_reference" in update_data:
            new_provider = update_data.get("secret_provider", connection.secret_provider)
            new_reference = update_data.get("secret_reference", connection.secret_reference)
            await self.secret_resolution_service.validate_connection_reference(
                context=context,
                provider_type=new_provider,
                reference=new_reference,
            )
                
        # Check if connectivity fields are changing
        connectivity_fields = {"host", "port", "database_name", "ssl_mode", "secret_provider", "secret_reference"}
        is_connectivity_change = any(f in update_data and update_data[f] != getattr(connection, f) for f in connectivity_fields)
        
        changed_fields = []
        secret_configuration_changed = False
        old_provider = connection.secret_provider
        
        for field, value in update_data.items():
            if getattr(connection, field) != value:
                setattr(connection, field, value)
                changed_fields.append(field)
                if field in ("secret_provider", "secret_reference"):
                    secret_configuration_changed = True
                
        if is_connectivity_change:
            connection.last_tested_at = None
            connection.last_test_status = ConnectionTestStatus.NEVER_TESTED
            connection.last_test_error_code = None
            
        connection.updated_by_user_id = context.user_id
        await self.repository.flush()

        metadata = {"changed_fields": changed_fields, "connection_id": str(connection.id)}
        if secret_configuration_changed:
            metadata["secret_configuration_changed"] = True
            metadata["previous_provider"] = old_provider
            metadata["new_provider"] = connection.secret_provider
            
        actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER
        await self.audit_service.record_success(AuditEventCreate(
            organization_id=context.organization_id,
            actor_user_id=context.user_id,
            actor_type=actor_type,
            action=AuditAction.CONNECTION_UPDATED,
            outcome=AuditOutcome.SUCCEEDED,
            resource_type=AuditResourceType.DATABASE_CONNECTION,
            resource_id=connection.id,
            metadata=metadata
        ))
        
        return DatabaseConnectionRead.model_validate({
            **connection.__dict__,
            "redacted_secret_reference": redact_secret_reference(connection.secret_provider, connection.secret_reference)
        })

    async def disable_connection(
        self, connection_id: uuid.UUID, context: AuthorizedOrganizationContext
    ) -> None:
        """Idempotently disable a connection."""
        connection = await self.repository.get_by_id_for_organization(connection_id, context.organization_id)
        if not connection:
            raise NotFoundError("Connection not found.", code="CONNECTION_NOT_FOUND")
            
        if connection.status != ConnectionStatus.DISABLED:
            connection.status = ConnectionStatus.DISABLED
            connection.updated_by_user_id = context.user_id
            await self.repository.flush()
            
            if self.pool_registry:
                await self.pool_registry.dispose_for_connection(context.organization_id, connection.id)
            
            actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER
            await self.audit_service.record_success(AuditEventCreate(
                organization_id=context.organization_id,
                actor_user_id=context.user_id,
                actor_type=actor_type,
                action=AuditAction.CONNECTION_DISABLED,
                outcome=AuditOutcome.SUCCEEDED,
                resource_type=AuditResourceType.DATABASE_CONNECTION,
                resource_id=connection.id,
                metadata={"connection_id": str(connection.id)}
            ))

