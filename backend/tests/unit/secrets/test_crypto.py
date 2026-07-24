import base64
import uuid

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

from app.core.exceptions import ValidationError
from app.secrets.crypto import LocalSecretEncryptionService
from app.secrets.schemas import DatabaseCredentialPayload


@pytest.fixture
def mock_settings(monkeypatch):
    class MockSettings:
        local_secret_master_key = base64.b64encode(b"0" * 32).decode("utf-8")
        local_secret_key_version = "v1"
        secret_value_max_bytes = 16384
        
    monkeypatch.setattr("app.secrets.crypto.get_settings", lambda: MockSettings())
    return MockSettings()


def test_encryption_decryption_roundtrip(mock_settings):
    service = LocalSecretEncryptionService()
    org_id = uuid.uuid4()
    secret_id = uuid.uuid4()
    
    payload = DatabaseCredentialPayload(
        username="db_user",
        password=SecretStr("supersecret"),
        database="prod_db",
        port=5432
    )
    
    encrypted = service.encrypt(
        payload=payload,
        organization_id=org_id,
        secret_id=secret_id
    )
    
    assert encrypted.encryption_algorithm == "AES-256-GCM"
    assert encrypted.key_version == "v1"
    assert encrypted.payload_version == 1
    assert len(encrypted.nonce) == 12
    assert len(encrypted.ciphertext) > 0
    
    decrypted = service.decrypt(
        ciphertext=encrypted.ciphertext,
        nonce=encrypted.nonce,
        key_version=encrypted.key_version,
        payload_version=encrypted.payload_version,
        organization_id=org_id,
        secret_id=secret_id,
        encryption_algorithm=encrypted.encryption_algorithm
    )
    
    assert decrypted.username == "db_user"
    assert decrypted.password.get_secret_value() == "supersecret"
    assert decrypted.database == "prod_db"
    assert decrypted.port == 5432


def test_decryption_fails_with_wrong_organization(mock_settings):
    service = LocalSecretEncryptionService()
    org_id = uuid.uuid4()
    secret_id = uuid.uuid4()
    
    payload = DatabaseCredentialPayload(
        username="db_user",
        password=SecretStr("supersecret")
    )
    
    encrypted = service.encrypt(
        payload=payload,
        organization_id=org_id,
        secret_id=secret_id
    )
    
    wrong_org_id = uuid.uuid4()
    
    with pytest.raises(ValidationError) as exc:
        service.decrypt(
            ciphertext=encrypted.ciphertext,
            nonce=encrypted.nonce,
            key_version=encrypted.key_version,
            payload_version=encrypted.payload_version,
            organization_id=wrong_org_id,
            secret_id=secret_id,
            encryption_algorithm=encrypted.encryption_algorithm
        )
    assert "authentication tag mismatch" in str(exc.value)


def test_decryption_fails_with_tampered_ciphertext(mock_settings):
    service = LocalSecretEncryptionService()
    org_id = uuid.uuid4()
    secret_id = uuid.uuid4()
    
    payload = DatabaseCredentialPayload(
        username="db_user",
        password=SecretStr("supersecret")
    )
    
    encrypted = service.encrypt(
        payload=payload,
        organization_id=org_id,
        secret_id=secret_id
    )
    
    tampered_ciphertext = bytearray(encrypted.ciphertext)
    tampered_ciphertext[0] ^= 0xFF
    
    with pytest.raises(ValidationError):
        service.decrypt(
            ciphertext=bytes(tampered_ciphertext),
            nonce=encrypted.nonce,
            key_version=encrypted.key_version,
            payload_version=encrypted.payload_version,
            organization_id=org_id,
            secret_id=secret_id,
            encryption_algorithm=encrypted.encryption_algorithm
        )
