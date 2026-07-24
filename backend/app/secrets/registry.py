from typing import Dict

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationError
from app.models.connection_enums import SecretProviderType
from app.secrets.base import SecretProvider
from app.secrets.providers.aws_secrets_manager import BOTO3_AVAILABLE, AWSSecretsManagerProvider
from app.secrets.providers.environment import EnvironmentSecretProvider
from app.secrets.providers.local_encrypted import LocalEncryptedSecretProvider


class SecretProviderRegistry:
    def __init__(self):
        self._providers: Dict[SecretProviderType, SecretProvider] = {}

    def register(self, provider: SecretProvider) -> None:
        """Register a new secret provider."""
        if provider.provider_type in self._providers:
            raise ValueError(f"Provider {provider.provider_type} already registered.")
        self._providers[provider.provider_type] = provider

    def get(self, provider_type: SecretProviderType) -> SecretProvider:
        """Get a configured provider instance."""
        if provider_type not in self._providers:
            raise ValidationError(
                f"Secret provider {provider_type} is not supported or not configured.", 
                code="SECRET_PROVIDER_NOT_SUPPORTED"
            )
        return self._providers[provider_type]

    def supports(self, provider_type: SecretProviderType) -> bool:
        """Check if a provider is supported."""
        return provider_type in self._providers


def build_secret_provider_registry(session: AsyncSession) -> SecretProviderRegistry:
    """
    Factory to build a registry with all available providers for a request.
    """
    registry = SecretProviderRegistry()
    
    # Register environment provider
    registry.register(EnvironmentSecretProvider())
    
    # Register local encrypted provider
    registry.register(LocalEncryptedSecretProvider(session))
    
    # Register AWS provider only if dependency is present
    if BOTO3_AVAILABLE:
        registry.register(AWSSecretsManagerProvider())
        
    return registry
