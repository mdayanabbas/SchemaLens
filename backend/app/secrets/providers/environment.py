import json
import os
import re
import uuid

from pydantic import ValidationError as PydanticValidationError

from app.core.config import get_settings
from app.core.exceptions import ValidationError
from app.models.connection_enums import SecretProviderType
from app.secrets.base import SecretProvider, SecretProviderHealthResult
from app.secrets.schemas import DatabaseCredentialPayload
from app.secrets.value import SecretValue


class EnvironmentSecretProvider(SecretProvider):
    provider_type = SecretProviderType.ENVIRONMENT

    async def validate_reference(
        self,
        *,
        organization_id: uuid.UUID,
        reference: str,
    ) -> None:
        """
        Validate that the environment variable reference matches the allowed pattern.
        """
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,127}", reference):
            raise ValidationError("Invalid environment variable reference format", code="INVALID_SECRET_REFERENCE")
            
        # We do not require the env var to exist during profile creation unless policy demands it,
        # but this is just syntax validation as requested by the brick.

    async def resolve(
        self,
        *,
        organization_id: uuid.UUID,
        reference: str,
    ) -> SecretValue:
        """
        Resolve the secret from the environment.
        """
        # Validate syntax again for safety
        await self.validate_reference(organization_id=organization_id, reference=reference)

        env_val = os.environ.get(reference)
        if env_val is None:
            raise ValidationError("Secret provider unavailable or reference missing", code="SECRET_NOT_FOUND")

        settings = get_settings()
        if len(env_val) > settings.secret_value_max_bytes:
            raise ValidationError("Secret payload too large", code="SECRET_PAYLOAD_TOO_LARGE")

        try:
            parsed = json.loads(env_val)
        except json.JSONDecodeError:
            raise ValidationError("Failed to parse secret payload", code="SECRET_PAYLOAD_INVALID")

        if not isinstance(parsed, dict):
            raise ValidationError("Secret payload must be a JSON object", code="SECRET_PAYLOAD_INVALID")

        try:
            payload = DatabaseCredentialPayload.model_validate(parsed)
        except PydanticValidationError:
            raise ValidationError("Invalid secret payload structure", code="SECRET_PAYLOAD_INVALID")

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
            provider_metadata={"provider": self.provider_type.value}
        )

    async def health_check(self) -> SecretProviderHealthResult:
        """
        Verify the provider is healthy. Environment is inherently healthy.
        """
        return SecretProviderHealthResult(
            status="healthy",
            provider=self.provider_type
        )

    def redact_reference(self, reference: str) -> str:
        """
        Redact the reference for logging.
        """
        if len(reference) <= 16:
            return f"{self.provider_type.value}:[REDACTED]"
            
        return f"{self.provider_type.value}:{reference[:12]}...{reference[-4:]}"
