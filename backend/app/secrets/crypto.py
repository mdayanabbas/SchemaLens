import base64
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import ValidationError as PydanticValidationError

from app.core.config import get_settings
from app.core.exceptions import ValidationError
from app.secrets.schemas import DatabaseCredentialPayload


@dataclass(slots=True)
class EncryptedSecretPayload:
    ciphertext: bytes
    nonce: bytes
    encryption_algorithm: str
    key_version: str
    payload_version: int


class LocalSecretEncryptionService:
    ALGORITHM = "AES-256-GCM"

    def __init__(self):
        self.settings = get_settings()

    def _get_master_key(self) -> bytes:
        """Decode the 32-byte master key only when needed."""
        key_b64 = self.settings.local_secret_master_key
        if not key_b64 or not key_b64.strip():
            raise ValidationError("Missing local master key configuration", code="LOCAL_SECRET_KEY_MISSING")

        try:
            key = base64.b64decode(key_b64.strip())
        except ValueError:
            raise ValidationError("Invalid local master key encoding", code="LOCAL_SECRET_KEY_INVALID")

        if len(key) != 32:
            raise ValidationError("Local master key must be exactly 32 decoded bytes", code="LOCAL_SECRET_KEY_INVALID")

        return key

    def _build_aad(
        self,
        organization_id: uuid.UUID,
        secret_id: uuid.UUID,
        key_version: str,
        payload_version: int
    ) -> bytes:
        """Build Authenticated Additional Data."""
        # Using deterministic JSON for AAD
        aad_dict = {
            "organization_id": str(organization_id),
            "secret_id": str(secret_id),
            "key_version": key_version,
            "payload_version": payload_version,
            "algorithm": self.ALGORITHM,
        }
        return json.dumps(aad_dict, separators=(",", ":"), sort_keys=True).encode("utf-8")

    def encrypt(
        self,
        payload: DatabaseCredentialPayload,
        *,
        organization_id: uuid.UUID,
        secret_id: uuid.UUID,
        payload_version: int = 1
    ) -> EncryptedSecretPayload:
        """
        Encrypt a database credential payload using AES-256-GCM.
        """
        key_version = self.settings.local_secret_key_version
        key = self._get_master_key()
        
        # Serialize payload deterministically
        # Need to use model_dump_json to handle SecretStr properly
        payload_json = payload.model_dump_json(warnings=False, exclude_none=True).encode("utf-8")
        
        if len(payload_json) > self.settings.secret_value_max_bytes:
            raise ValidationError("Secret payload too large", code="SECRET_PAYLOAD_TOO_LARGE")

        aesgcm = AESGCM(key)
        nonce = os.urandom(12)  # 96-bit nonce
        aad = self._build_aad(organization_id, secret_id, key_version, payload_version)

        try:
            ciphertext = aesgcm.encrypt(nonce, payload_json, aad)
        except Exception:
            raise ValidationError("Encryption failed", code="SECRET_ENCRYPTION_FAILED")

        return EncryptedSecretPayload(
            ciphertext=ciphertext,
            nonce=nonce,
            encryption_algorithm=self.ALGORITHM,
            key_version=key_version,
            payload_version=payload_version
        )

    def decrypt(
        self,
        *,
        ciphertext: bytes,
        nonce: bytes,
        key_version: str,
        payload_version: int,
        organization_id: uuid.UUID,
        secret_id: uuid.UUID,
        encryption_algorithm: str
    ) -> DatabaseCredentialPayload:
        """
        Decrypt a database credential payload.
        """
        if encryption_algorithm != self.ALGORITHM:
            raise ValidationError("Unsupported encryption algorithm", code="SECRET_CONFIGURATION_ERROR")
            
        if key_version != self.settings.local_secret_key_version:
            # Note: Future key-ring support would handle multiple versions here
            raise ValidationError("Unsupported key version", code="LOCAL_SECRET_KEY_VERSION_UNSUPPORTED")
            
        if len(ciphertext) > (self.settings.secret_value_max_bytes + 16): # 16 bytes for auth tag
            raise ValidationError("Ciphertext too large", code="SECRET_DECRYPTION_FAILED")

        key = self._get_master_key()
        aesgcm = AESGCM(key)
        aad = self._build_aad(organization_id, secret_id, key_version, payload_version)

        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, aad)
        except InvalidTag:
            raise ValidationError("Secret decryption failed (authentication tag mismatch)", code="SECRET_DECRYPTION_FAILED")
        except Exception:
            raise ValidationError("Secret decryption failed", code="SECRET_DECRYPTION_FAILED")

        try:
            parsed = json.loads(plaintext.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError()
            return DatabaseCredentialPayload.model_validate(parsed)
        except Exception:
            raise ValidationError("Failed to parse decrypted payload", code="SECRET_PAYLOAD_INVALID")
