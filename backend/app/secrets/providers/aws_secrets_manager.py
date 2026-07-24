import json
import re
import uuid
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.core.config import get_settings
from app.core.exceptions import ValidationError
from app.models.connection_enums import SecretProviderType
from app.secrets.base import SecretProvider, SecretProviderHealthResult
from app.secrets.schemas import DatabaseCredentialPayload
from app.secrets.value import SecretValue

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, BotoCoreError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class AWSSecretsManagerProvider(SecretProvider):
    provider_type = SecretProviderType.AWS_SECRETS_MANAGER

    def __init__(self):
        self.settings = get_settings()

    def _ensure_boto3(self):
        if not BOTO3_AVAILABLE:
            raise ValidationError(
                "AWS Secrets Manager dependency is missing. Install with 'schemalens-backend[aws]'",
                code="SECRET_PROVIDER_UNAVAILABLE"
            )

    def _get_client(self):
        self._ensure_boto3()
        # Use standard AWS credential provider chain
        config = Config(
            connect_timeout=self.settings.aws_secrets_manager_timeout_seconds,
            read_timeout=self.settings.aws_secrets_manager_timeout_seconds,
            retries={"max_attempts": 1} # small bounded retry count
        )
        kwargs: dict[str, Any] = {"config": config}
        if self.settings.aws_region:
            kwargs["region_name"] = self.settings.aws_region
        if self.settings.aws_secrets_manager_endpoint_url:
            kwargs["endpoint_url"] = self.settings.aws_secrets_manager_endpoint_url
            
        return boto3.client("secretsmanager", **kwargs)

    async def validate_reference(
        self,
        *,
        organization_id: uuid.UUID,
        reference: str,
    ) -> None:
        """
        Validate that the reference length and format are acceptable without making a network call.
        """
        if len(reference) > self.settings.secret_reference_max_length:
            raise ValidationError("AWS secret reference is too long", code="INVALID_SECRET_REFERENCE")
            
        if not reference.strip():
            raise ValidationError("AWS secret reference cannot be empty", code="INVALID_SECRET_REFERENCE")
            
        if re.search(r"[\x00-\x1F\x7F]", reference):
            raise ValidationError("AWS secret reference must not contain control characters", code="INVALID_SECRET_REFERENCE")

    async def resolve(
        self,
        *,
        organization_id: uuid.UUID,
        reference: str,
    ) -> SecretValue:
        """
        Resolve the secret from AWS Secrets Manager.
        """
        await self.validate_reference(organization_id=organization_id, reference=reference)
        
        client = self._get_client()
        
        try:
            response = client.get_secret_value(SecretId=reference)
        except Exception as e:
            # Safely handle known botocore exceptions
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                raise ValidationError("AWS secret not found", code="AWS_SECRET_NOT_FOUND")
            elif error_code in ("AccessDeniedException", "UnrecognizedClientException", "InvalidClientTokenId"):
                raise ValidationError("AWS secret access denied", code="AWS_SECRET_ACCESS_DENIED")
            elif "timeout" in str(e).lower() or "read timed out" in str(e).lower():
                raise ValidationError("AWS secret resolution timed out", code="AWS_SECRET_TIMEOUT")
            else:
                raise ValidationError("AWS secret resolution failed", code="SECRET_RESOLUTION_FAILED")
                
        secret_string = response.get("SecretString")
        secret_binary = response.get("SecretBinary")
        
        if secret_string:
            payload_str = secret_string
        elif secret_binary:
            try:
                payload_str = secret_binary.decode("utf-8")
            except UnicodeDecodeError:
                raise ValidationError("AWS secret payload invalid encoding", code="AWS_SECRET_PAYLOAD_INVALID")
        else:
            raise ValidationError("AWS secret payload is empty", code="AWS_SECRET_PAYLOAD_INVALID")

        if len(payload_str) > self.settings.secret_value_max_bytes:
            raise ValidationError("AWS secret payload too large", code="SECRET_PAYLOAD_TOO_LARGE")

        try:
            parsed = json.loads(payload_str)
        except json.JSONDecodeError:
            raise ValidationError("Failed to parse AWS secret payload as JSON", code="AWS_SECRET_PAYLOAD_INVALID")

        if not isinstance(parsed, dict):
            raise ValidationError("AWS secret payload must be a JSON object", code="AWS_SECRET_PAYLOAD_INVALID")

        try:
            payload = DatabaseCredentialPayload.model_validate(parsed)
        except PydanticValidationError:
            raise ValidationError("Invalid AWS secret payload structure", code="AWS_SECRET_PAYLOAD_INVALID")

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
        Verify the provider is healthy.
        Calls list_secrets with max-results=1 just to check connectivity safely,
        but skips network call if boto3 isn't available.
        """
        if not BOTO3_AVAILABLE:
            return SecretProviderHealthResult(
                status="unhealthy",
                provider=self.provider_type,
                safe_error_code="SECRET_PROVIDER_UNAVAILABLE"
            )
            
        try:
            client = self._get_client()
            client.list_secrets(MaxResults=1)
            return SecretProviderHealthResult(
                status="healthy",
                provider=self.provider_type
            )
        except Exception as e:
            return SecretProviderHealthResult(
                status="unhealthy",
                provider=self.provider_type,
                safe_error_code="SECRET_PROVIDER_UNAVAILABLE"
            )

    def redact_reference(self, reference: str) -> str:
        """
        Redact the AWS reference for logging.
        Strips ARN account ID and partial name.
        """
        # Ex: arn:aws:secretsmanager:us-east-1:123456789012:secret:my-db-secret-AbCd
        if reference.startswith("arn:aws:secretsmanager:"):
            parts = reference.split(":")
            if len(parts) >= 7:
                secret_id = parts[6]
                if len(secret_id) > 6:
                    return f"{self.provider_type.value}:arn:aws:secretsmanager:...:...:secret:...{secret_id[-4:]}"
                return f"{self.provider_type.value}:arn:aws:secretsmanager:...:...:secret:[REDACTED]"
                
        # If it's just a name, only reveal last 4
        if len(reference) > 8:
            return f"{self.provider_type.value}:...{reference[-4:]}"
            
        return f"{self.provider_type.value}:[REDACTED]"
