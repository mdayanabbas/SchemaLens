import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
from app.audit.schemas import AuditEventCreate
from app.audit.service import AuditService
from app.core.exceptions import ForbiddenError, ValidationError
from app.governance.context import AuthorizedOrganizationContext
from app.governance.permissions import Permission
from app.models.connection_enums import ConnectionStatus, SecretProviderType
from app.models.database_connection import DatabaseConnection
from app.secrets.registry import SecretProviderRegistry, build_secret_provider_registry
from app.secrets.value import SecretValue


class SecretResolutionService:
    def __init__(self, session: AsyncSession, audit_service: AuditService):
        self.registry = build_secret_provider_registry(session)
        self.audit_service = audit_service

    async def validate_connection_reference(
        self,
        *,
        context: AuthorizedOrganizationContext,
        provider_type: SecretProviderType,
        reference: str,
    ) -> None:
        """
        Validate a connection reference using the appropriate provider.
        Does not resolve the secret value.
        """
        provider = self.registry.get(provider_type)
        await provider.validate_reference(
            organization_id=context.organization_id,
            reference=reference
        )

    async def resolve_for_connection(
        self,
        *,
        context: AuthorizedOrganizationContext,
        connection: DatabaseConnection,
    ) -> SecretValue:
        """
        Resolve the secret for a database connection.
        Requires connections.test permission or trusted internal caller.
        """
        if context.organization_id != connection.organization_id:
            raise ForbiddenError("Connection does not belong to the current organization.")

        if not context.is_platform_admin and Permission.CONNECTIONS_TEST not in context.permissions:
            raise ForbiddenError("Missing permission: connections.test")

        if connection.status == ConnectionStatus.DISABLED:
            raise ValidationError("Cannot resolve credentials for a disabled connection.", code="CONNECTION_DISABLED")

        provider = self.registry.get(connection.secret_provider)
        
        actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER

        try:
            secret_value = await provider.resolve(
                organization_id=context.organization_id,
                reference=connection.secret_reference
            )
        except Exception as e:
            # Audit failure
            await self.audit_service.record_success(AuditEventCreate(
                organization_id=context.organization_id,
                actor_user_id=context.user_id,
                actor_type=actor_type,
                action=AuditAction.SECRET_RESOLUTION_FAILED,
                outcome=AuditOutcome.FAILED,
                resource_type=AuditResourceType.DATABASE_CONNECTION,
                resource_id=connection.id,
                metadata={
                    "connection_id": str(connection.id),
                    "provider_type": connection.secret_provider,
                    "resolution_status": "failed",
                    "safe_error_code": getattr(e, "code", "SECRET_RESOLUTION_FAILED")
                }
            ))
            raise

        # Audit success
        fields_present = []
        if secret_value.username: fields_present.append("username")
        if secret_value.password: fields_present.append("password")
        if secret_value.database: fields_present.append("database")
        if secret_value.host: fields_present.append("host")
        if secret_value.port: fields_present.append("port")
        if secret_value.ssl_ca: fields_present.append("ssl_ca")
        if secret_value.ssl_cert: fields_present.append("ssl_cert")
        if secret_value.ssl_key: fields_present.append("ssl_key")

        metadata: dict[str, Any] = {
            "connection_id": str(connection.id),
            "provider_type": connection.secret_provider,
            "resolution_status": "succeeded",
            "fields_present": fields_present
        }
        
        if secret_value.expires_at:
            metadata["expires_at"] = secret_value.expires_at.isoformat()
            
        metadata.update(secret_value.provider_metadata)

        await self.audit_service.record_success(AuditEventCreate(
            organization_id=context.organization_id,
            actor_user_id=context.user_id,
            actor_type=actor_type,
            action=AuditAction.SECRET_RESOLVED,
            outcome=AuditOutcome.SUCCEEDED,
            resource_type=AuditResourceType.DATABASE_CONNECTION,
            resource_id=connection.id,
            metadata=metadata
        ))

        return secret_value

    async def resolve_secret_for_connector(
        self,
        *,
        organization_id: uuid.UUID,
        provider_type: SecretProviderType,
        reference: str,
    ) -> SecretValue:
        """
        Resolve the secret internally for a connector operation.
        Assumes authorization has already been performed by the caller.
        """
        provider = self.registry.get(provider_type)
        
        try:
            secret_value = await provider.resolve(
                organization_id=organization_id,
                reference=reference
            )
        except Exception as e:
            # Audit failure as SYSTEM
            await self.audit_service.record_success(AuditEventCreate(
                organization_id=organization_id,
                actor_user_id=None,
                actor_type=AuditActorType.SYSTEM,
                action=AuditAction.SECRET_RESOLUTION_FAILED,
                outcome=AuditOutcome.FAILED,
                resource_type=AuditResourceType.DATABASE_CONNECTION,
                resource_id=None,
                metadata={
                    "provider_type": provider_type,
                    "resolution_status": "failed",
                    "safe_error_code": getattr(e, "code", "SECRET_RESOLUTION_FAILED")
                }
            ))
            raise

        # Audit success as SYSTEM
        fields_present = []
        if secret_value.username: fields_present.append("username")
        if secret_value.password: fields_present.append("password")
        if secret_value.database: fields_present.append("database")
        if secret_value.host: fields_present.append("host")
        if secret_value.port: fields_present.append("port")
        if secret_value.ssl_ca: fields_present.append("ssl_ca")
        if secret_value.ssl_cert: fields_present.append("ssl_cert")
        if secret_value.ssl_key: fields_present.append("ssl_key")

        metadata: dict[str, Any] = {
            "provider_type": provider_type,
            "resolution_status": "succeeded",
            "fields_present": fields_present
        }
        
        if secret_value.expires_at:
            metadata["expires_at"] = secret_value.expires_at.isoformat()
            
        metadata.update(secret_value.provider_metadata)

        await self.audit_service.record_success(AuditEventCreate(
            organization_id=organization_id,
            actor_user_id=None,
            actor_type=AuditActorType.SYSTEM,
            action=AuditAction.SECRET_RESOLVED,
            outcome=AuditOutcome.SUCCEEDED,
            resource_type=AuditResourceType.DATABASE_CONNECTION,
            resource_id=None,
            metadata=metadata
        ))

        return secret_value
