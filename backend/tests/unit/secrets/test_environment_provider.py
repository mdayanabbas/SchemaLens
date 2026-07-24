import uuid

import pytest

from app.core.exceptions import ValidationError
from app.secrets.providers.environment import EnvironmentSecretProvider


@pytest.fixture
def env_provider():
    return EnvironmentSecretProvider()


@pytest.mark.asyncio
async def test_env_provider_validate_reference_success(env_provider):
    org_id = uuid.uuid4()
    await env_provider.validate_reference(organization_id=org_id, reference="DB_PASSWORD")
    

@pytest.mark.asyncio
async def test_env_provider_validate_reference_fails_format(env_provider):
    org_id = uuid.uuid4()
    with pytest.raises(ValidationError) as exc:
        await env_provider.validate_reference(organization_id=org_id, reference="invalid-format")
    assert "Invalid environment variable reference format" in str(exc.value)


@pytest.mark.asyncio
async def test_env_provider_resolve_success(env_provider, monkeypatch):
    org_id = uuid.uuid4()
    # Mock os.environ.get
    import json
    payload = {
        "username": "user",
        "password": "pwd"
    }
    monkeypatch.setenv("DB_CREDS", json.dumps(payload))
    
    secret_value = await env_provider.resolve(organization_id=org_id, reference="DB_CREDS")
    assert secret_value.username == "user"
    assert secret_value.get_password() == "pwd"


@pytest.mark.asyncio
async def test_env_provider_resolve_fails_not_found(env_provider):
    org_id = uuid.uuid4()
    with pytest.raises(ValidationError) as exc:
        await env_provider.resolve(organization_id=org_id, reference="MISSING_ENV_VAR")
    assert "Secret provider unavailable or reference missing" in str(exc.value)
