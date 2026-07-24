import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from app.models.connection_enums import SecretProviderType
from app.secrets.value import SecretValue


@dataclass(slots=True)
class SecretProviderHealthResult:
    status: str
    provider: SecretProviderType
    latency_ms: Optional[float] = None
    safe_error_code: Optional[str] = None


class SecretProvider(ABC):
    provider_type: SecretProviderType

    @abstractmethod
    async def validate_reference(
        self,
        *,
        organization_id: uuid.UUID,
        reference: str,
    ) -> None:
        """
        Validate the syntax or safe existence of a reference without resolving the secret payload.
        Throws a safe application exception (e.g., ValidationError, NotFoundError) on failure.
        """
        ...

    @abstractmethod
    async def resolve(
        self,
        *,
        organization_id: uuid.UUID,
        reference: str,
    ) -> SecretValue:
        """
        Resolve the secret into a protected SecretValue.
        Throws a safe application exception on failure.
        """
        ...

    @abstractmethod
    async def health_check(self) -> SecretProviderHealthResult:
        """
        Verify the provider is configured and reachable without reading a real credential.
        """
        ...

    @abstractmethod
    def redact_reference(self, reference: str) -> str:
        """
        Return a safe redacted version of the reference for logging and auditing.
        """
        ...
