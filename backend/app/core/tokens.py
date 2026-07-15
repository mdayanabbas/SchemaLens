import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, UTC
from typing import Any

import jwt

from app.core.config import Settings
from app.core.exceptions import AppError


class TokenValidationException(AppError):
    def __init__(self, message: str = "Invalid token."):
        super().__init__(message=message, code="INVALID_ACCESS_TOKEN", status_code=401)


class TokenService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def calculate_access_token_expiry(self) -> datetime:
        return datetime.now(UTC) + timedelta(minutes=self.settings.access_token_expire_minutes)

    def calculate_refresh_token_expiry(self) -> datetime:
        return datetime.now(UTC) + timedelta(days=self.settings.refresh_token_expire_days)

    def create_access_token(self, user_id: uuid.UUID) -> tuple[str, datetime]:
        """Create a JWT access token for the given user."""
        now = datetime.now(UTC)
        expires_at = self.calculate_access_token_expiry()

        claims = {
            "sub": str(user_id),
            "token_type": "access",
            "jti": str(uuid.uuid4()),
            "iat": now,
            "nbf": now,
            "exp": expires_at,
            "iss": self.settings.authentication_issuer,
            "aud": self.settings.authentication_audience,
        }

        token = jwt.encode(
            claims,
            self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm,
        )

        return token, expires_at

    def decode_and_validate_access_token(self, token: str) -> dict[str, Any]:
        """Decode and validate a JWT access token."""
        try:
            payload = jwt.decode(
                token,
                self.settings.jwt_secret_key,
                algorithms=[self.settings.jwt_algorithm],
                issuer=self.settings.authentication_issuer,
                audience=self.settings.authentication_audience,
                options={
                    "require": ["sub", "token_type", "jti", "iat", "nbf", "exp", "iss", "aud"],
                }
            )
        except jwt.ExpiredSignatureError:
            raise TokenValidationException("Access token has expired.")
        except jwt.ImmatureSignatureError:
            raise TokenValidationException("Access token is not yet valid.")
        except jwt.PyJWTError:
            # Generic catch-all to avoid leaking specifics like 'invalid signature' or 'missing claim'
            raise TokenValidationException("Invalid access token.")

        if payload.get("token_type") != "access":
            raise TokenValidationException("Invalid token type.")

        try:
            uuid.UUID(payload["sub"])
        except (ValueError, TypeError):
            raise TokenValidationException("Invalid subject identifier.")

        return payload

    def generate_opaque_refresh_token(self) -> str:
        """Generate a cryptographically secure opaque random token (>= 256 bits of entropy)."""
        # 32 bytes = 256 bits
        return secrets.token_urlsafe(32)

    def hash_refresh_token(self, token: str) -> str:
        """Hash a raw refresh token using HMAC and a configured pepper."""
        key = self.settings.refresh_token_pepper.encode("utf-8")
        message = token.encode("utf-8")
        return hmac.new(key, message, hashlib.sha256).hexdigest()

    def compare_refresh_token_hashes(self, hash1: str, hash2: str) -> bool:
        """Compare two refresh token hashes safely in constant time."""
        return hmac.compare_digest(hash1, hash2)

    def hash_optional_ip_address(self, ip_address: str | None) -> str | None:
        """Securely hash an IP address if present to protect PII."""
        if not ip_address:
            return None
        
        # Use a distinct prefix internally to avoid hash collisions across fingerprint types
        message = f"ip:{ip_address}".encode("utf-8")
        key = self.settings.refresh_token_pepper.encode("utf-8")
        return hmac.new(key, message, hashlib.sha256).hexdigest()

    def hash_optional_user_agent(self, user_agent: str | None) -> str | None:
        """Securely hash a User-Agent if present to protect PII."""
        if not user_agent:
            return None
            
        message = f"ua:{user_agent}".encode("utf-8")
        key = self.settings.refresh_token_pepper.encode("utf-8")
        return hmac.new(key, message, hashlib.sha256).hexdigest()
