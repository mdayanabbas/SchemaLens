import uuid
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
from app.audit.schemas import AuditEventCreate
from app.audit.service import AuditService
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.governance.context import AuthorizedOrganizationContext
from app.models.connection_enums import ConnectionTestStatus, SecretProviderType
from app.models.database_connection import DatabaseConnection
from app.models.stored_secret import StoredSecret
from app.repositories.database_connection import DatabaseConnectionRepository
from app.schemas.stored_secret import StoredSecretCreate, StoredSecretRead, StoredSecretRotate
from app.secrets.crypto import LocalSecretEncryptionService
from app.secrets.enums import SecretStatus
from app.secrets.repository import StoredSecretRepository


class StoredSecretManagementService:
    def __init__(self, session: AsyncSession, audit_service: AuditService):
        self.session = session
        self.repository = StoredSecretRepository(session)
        self.connection_repository = DatabaseConnectionRepository(session)
        self.audit_service = audit_service
        self.encryption_service = LocalSecretEncryptionService()

    def _get_fields_present(self, payload) -> list[str]:
        fields = ["username", "password"]
        if payload.database: fields.append("database")
        if payload.host: fields.append("host")
        if payload.port: fields.append("port")
        if payload.ssl_ca: fields.append("ssl_ca")
        if payload.ssl_cert: fields.append("ssl_cert")
        if payload.ssl_key: fields.append("ssl_key")
        return fields
        
    def _to_read_schema(self, secret: StoredSecret) -> StoredSecretRead:
        fields = secret.metadata_json.get("fields_present", [])
        return StoredSecretRead(
            id=secret.id,
            organization_id=secret.organization_id,
            name=secret.name,
            status=secret.status,
            provider=secret.provider,
            reference=f"{SecretProviderType.LOCAL_ENCRYPTED.value}:{secret.id}",
            key_version=secret.key_version,
            payload_version=secret.payload_version,
            fields_present=fields,
            created_by_user_id=secret.created_by_user_id,
            updated_by_user_id=secret.updated_by_user_id,
            rotated_from_secret_id=secret.rotated_from_secret_id,
            last_resolved_at=secret.last_resolved_at,
            created_at=secret.created_at,
            updated_at=secret.updated_at,
        )

    async def create_local_secret(
        self,
        schema: StoredSecretCreate,
        context: AuthorizedOrganizationContext,
    ) -> dict[str, Any]:
        """
        Create a new local encrypted secret.
        """
        if await self.repository.name_exists_for_organization(name=schema.name, organization_id=context.organization_id):
            raise ConflictError("A secret with this name already exists in the organization.", code="SECRET_NAME_ALREADY_EXISTS")

        secret_id = uuid.uuid4()
        payload = schema.to_credential_payload()
        
        # Encrypt the payload
        encrypted = self.encryption_service.encrypt(
            payload=payload,
            organization_id=context.organization_id,
            secret_id=secret_id
        )
        
        fields_present = self._get_fields_present(schema)
        metadata_json = {
            "fields_present": fields_present,
            "creation_source": "api"
        }

        secret = StoredSecret(
            id=secret_id,
            organization_id=context.organization_id,
            name=schema.name,
            provider=SecretProviderType.LOCAL_ENCRYPTED.value,
            status=SecretStatus.ACTIVE,
            ciphertext=encrypted.ciphertext,
            nonce=encrypted.nonce,
            encryption_algorithm=encrypted.encryption_algorithm,
            key_version=encrypted.key_version,
            payload_version=encrypted.payload_version,
            metadata_json=metadata_json,
            created_by_user_id=context.user_id,
            updated_by_user_id=context.user_id,
        )
        
        self.repository.add(secret)
        await self.repository.flush()
        
        actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER
        await self.audit_service.record_success(AuditEventCreate(
            organization_id=context.organization_id,
            actor_user_id=context.user_id,
            actor_type=actor_type,
            action=AuditAction.SECRET_LOCAL_CREATED,
            outcome=AuditOutcome.SUCCEEDED,
            resource_type=AuditResourceType.STORED_SECRET,
            resource_id=secret.id,
            metadata={
                "secret_name": secret.name,
                "key_version": secret.key_version,
                "payload_version": secret.payload_version,
                "fields_present": fields_present
            }
        ))
        
        read_model = self._to_read_schema(secret)
        return {
            "metadata": read_model,
            "reference": read_model.reference
        }

    async def rotate_local_secret(
        self,
        secret_id: uuid.UUID,
        schema: StoredSecretRotate,
        context: AuthorizedOrganizationContext,
    ) -> dict[str, Any]:
        """
        Rotate a local secret by creating a new version.
        """
        old_secret = await self.repository.get_by_id_for_organization(
            secret_id=secret_id, 
            organization_id=context.organization_id,
            for_update=True
        )
        if not old_secret:
            raise NotFoundError("Secret not found.", code="SECRET_NOT_FOUND")
            
        if old_secret.status != SecretStatus.ACTIVE:
            raise ValidationError(f"Cannot rotate a {old_secret.status} secret.", code="SECRET_DISABLED" if old_secret.status == SecretStatus.DISABLED else "SECRET_ROTATED")
            
        new_secret_id = uuid.uuid4()
        payload = schema.to_credential_payload()
        
        encrypted = self.encryption_service.encrypt(
            payload=payload,
            organization_id=context.organization_id,
            secret_id=new_secret_id
        )
        
        fields_present = self._get_fields_present(schema)
        metadata_json = {
            "fields_present": fields_present,
            "creation_source": "rotation",
            "rotated_from_secret_id": str(old_secret.id)
        }

        # Create new secret with same name
        # Wait, the name is unique per org. If we keep the same name, we must change the old secret's name or remove the unique constraint.
        # Actually, the user requirement says "name: Unique within organization" and "uses disabled or rotated status".
        # If we create a new row with the same name, it violates uniqueness unless the unique index includes status, or we append a suffix.
        # Or maybe the "rotation" just updates the existing record?
        # "Create a new StoredSecret with new ID and fresh nonce. Set new secret's rotated_from_secret_id. Mark old secret rotated."
        # This means the unique constraint MUST NOT be violated.
        # So we should append a rotation suffix to the old secret's name, e.g. " (rotated <timestamp>)" or something, OR rotation creates a new name?
        # The prompt says: "Do not cascade-delete historical rotation records."
        # Let's append `_rotated_<id>` to the old name to free up the name for the new secret.
        
        old_secret.name = f"{old_secret.name}_rotated_{str(old_secret.id)[:8]}"
        old_secret.status = SecretStatus.ROTATED
        old_secret.updated_by_user_id = context.user_id
        
        new_secret = StoredSecret(
            id=new_secret_id,
            organization_id=context.organization_id,
            name=old_secret.name.replace(f"_rotated_{str(old_secret.id)[:8]}", ""), # just to be safe, though it should be the original name
            provider=SecretProviderType.LOCAL_ENCRYPTED.value,
            status=SecretStatus.ACTIVE,
            ciphertext=encrypted.ciphertext,
            nonce=encrypted.nonce,
            encryption_algorithm=encrypted.encryption_algorithm,
            key_version=encrypted.key_version,
            payload_version=encrypted.payload_version,
            metadata_json=metadata_json,
            created_by_user_id=context.user_id,
            updated_by_user_id=context.user_id,
            rotated_from_secret_id=old_secret.id
        )
        
        self.repository.add(new_secret)
        
        updated_connections = []
        if schema.update_connection_ids:
            # Update matching connections
            for conn_id in schema.update_connection_ids:
                conn = await self.connection_repository.get_by_id_for_organization(conn_id, context.organization_id)
                if conn and conn.secret_provider == SecretProviderType.LOCAL_ENCRYPTED.value:
                    # Update reference
                    conn.secret_reference = f"{SecretProviderType.LOCAL_ENCRYPTED.value}:{new_secret_id}"
                    conn.last_tested_at = None
                    conn.last_test_status = ConnectionTestStatus.NEVER_TESTED
                    conn.last_test_error_code = None
                    conn.updated_by_user_id = context.user_id
                    updated_connections.append(str(conn_id))
                    
        await self.repository.flush()
        
        actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER
        await self.audit_service.record_success(AuditEventCreate(
            organization_id=context.organization_id,
            actor_user_id=context.user_id,
            actor_type=actor_type,
            action=AuditAction.SECRET_LOCAL_ROTATED,
            outcome=AuditOutcome.SUCCEEDED,
            resource_type=AuditResourceType.STORED_SECRET,
            resource_id=new_secret.id,
            metadata={
                "secret_name": new_secret.name,
                "old_secret_id": str(old_secret.id),
                "key_version": new_secret.key_version,
                "payload_version": new_secret.payload_version,
                "fields_present": fields_present,
                "updated_connection_ids": updated_connections
            }
        ))
        
        read_model = self._to_read_schema(new_secret)
        return {
            "metadata": read_model,
            "reference": read_model.reference
        }

    async def disable_local_secret(
        self,
        secret_id: uuid.UUID,
        context: AuthorizedOrganizationContext,
    ) -> dict[str, Any]:
        """
        Disable a local encrypted secret.
        """
        secret = await self.repository.get_by_id_for_organization(
            secret_id=secret_id, organization_id=context.organization_id
        )
        if not secret:
            raise NotFoundError("Secret not found.", code="SECRET_NOT_FOUND")
            
        if secret.status != SecretStatus.DISABLED:
            secret.status = SecretStatus.DISABLED
            secret.updated_by_user_id = context.user_id
            await self.repository.flush()
            
            actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER
            await self.audit_service.record_success(AuditEventCreate(
                organization_id=context.organization_id,
                actor_user_id=context.user_id,
                actor_type=actor_type,
                action=AuditAction.SECRET_LOCAL_DISABLED,
                outcome=AuditOutcome.SUCCEEDED,
                resource_type=AuditResourceType.STORED_SECRET,
                resource_id=secret.id,
                metadata={"secret_id": str(secret.id)}
            ))
            
        return {"status": "success", "message": "Secret disabled"}

    async def list_local_secrets(
        self,
        context: AuthorizedOrganizationContext,
        offset: int = 0,
        limit: int = 25,
        status: str | None = None,
    ) -> list[StoredSecretRead]:
        """List metadata for local secrets."""
        
        status_enum = None
        if status:
            try:
                status_enum = SecretStatus(status)
            except ValueError:
                raise ValidationError("Invalid status filter", code="VALIDATION_ERROR")
                
        secrets = await self.repository.list_for_organization(
            organization_id=context.organization_id,
            offset=offset,
            limit=limit,
            status=status_enum
        )
        return [self._to_read_schema(s) for s in secrets]

    async def get_local_secret_metadata(
        self,
        secret_id: uuid.UUID,
        context: AuthorizedOrganizationContext,
    ) -> StoredSecretRead:
        """Get metadata for a specific local secret."""
        secret = await self.repository.get_by_id_for_organization(
            secret_id=secret_id, organization_id=context.organization_id
        )
        if not secret:
            raise NotFoundError("Secret not found.", code="SECRET_NOT_FOUND")
            
        return self._to_read_schema(secret)
