import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, ValidationError
from app.models.connection_enums import SecretProviderType
from app.secrets.base import SecretProvider, SecretProviderHealthResult
from app.secrets.crypto import LocalSecretEncryptionService
from app.secrets.enums import SecretStatus
from app.secrets.repository import StoredSecretRepository
from app.secrets.value import SecretValue


class LocalEncryptedSecretProvider(SecretProvider):
    provider_type = SecretProviderType.LOCAL_ENCRYPTED

    def __init__(self, session: AsyncSession):
        self.repository = StoredSecretRepository(session)
        self.encryption_service = LocalSecretEncryptionService()

    def _parse_reference(self, reference: str) -> uuid.UUID:
        """Parse local-secret:<uuid>"""
        prefix = f"{self.provider_type.value}:"
        if not reference.startswith(prefix):
            raise ValidationError("Invalid local secret reference format", code="INVALID_SECRET_REFERENCE")
            
        uuid_str = reference[len(prefix):]
        try:
            return uuid.UUID(uuid_str)
        except ValueError:
            raise ValidationError("Invalid local secret reference UUID", code="INVALID_SECRET_REFERENCE")

    async def validate_reference(
        self,
        *,
        organization_id: uuid.UUID,
        reference: str,
    ) -> None:
        """
        Validate the syntax and that the secret exists and belongs to the organization.
        """
        secret_id = self._parse_reference(reference)
        
        # Validating existence without decrypting
        secret = await self.repository.get_by_id_for_organization(
            secret_id=secret_id, organization_id=organization_id
        )
        if not secret:
            raise NotFoundError("Local secret not found in this organization.", code="SECRET_NOT_FOUND")

    async def resolve(
        self,
        *,
        organization_id: uuid.UUID,
        reference: str,
    ) -> SecretValue:
        """
        Resolve and decrypt the local stored secret.
        """
        secret_id = self._parse_reference(reference)
        
        secret = await self.repository.get_by_id_for_organization(
            secret_id=secret_id, organization_id=organization_id
        )
        if not secret:
            raise NotFoundError("Local secret not found in this organization.", code="SECRET_NOT_FOUND")
            
        if secret.status == SecretStatus.DISABLED:
            raise ValidationError("Local secret is disabled.", code="SECRET_DISABLED")
            
        if secret.status == SecretStatus.ROTATED:
            raise ValidationError("Local secret has been rotated.", code="SECRET_ROTATED")
            
        if secret.status != SecretStatus.ACTIVE:
            raise ValidationError(f"Local secret has invalid status: {secret.status}", code="SECRET_RESOLUTION_FAILED")
            
        # Decrypt payload
        payload = self.encryption_service.decrypt(
            ciphertext=secret.ciphertext,
            nonce=secret.nonce,
            key_version=secret.key_version,
            payload_version=secret.payload_version,
            organization_id=organization_id,
            secret_id=secret_id,
            encryption_algorithm=secret.encryption_algorithm
        )
        
        # Update last resolved at, but do not flush here (relying on caller's transaction)
        secret.last_resolved_at = datetime.now(timezone.utc)
        
        return SecretValue(
            username=payload.username,
            password=payload.password,
            database=payload.database,
            host=payload.host,
            port=payload.port,
            ssl_ca=payload.ssl_ca,
            ssl_cert=payload.ssl_cert,
            ssl_key=payload.ssl_key,
            expires_at=payload.expires_at,
            provider_metadata={
                "provider": self.provider_type.value,
                "secret_id": str(secret.id),
                "key_version": secret.key_version,
            }
        )

    async def health_check(self) -> SecretProviderHealthResult:
        """
        Verify the local encryption service has a valid master key without reading a real credential.
        """
        try:
            self.encryption_service._get_master_key()
            return SecretProviderHealthResult(
                status="healthy",
                provider=self.provider_type
            )
        except Exception as e:
            return SecretProviderHealthResult(
                status="unhealthy",
                provider=self.provider_type,
                safe_error_code=getattr(e, "code", "SECRET_CONFIGURATION_ERROR")
            )

    def redact_reference(self, reference: str) -> str:
        """
        Redact the reference for logging. (e.g., local-encrypted:...9451)
        """
        if ":" in reference:
            parts = reference.split(":")
            uuid_part = parts[-1]
            if len(uuid_part) > 4:
                return f"{self.provider_type.value}:...{uuid_part[-4:]}"
        return f"{self.provider_type.value}:[REDACTED]"
