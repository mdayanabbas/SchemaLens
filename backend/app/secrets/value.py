import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from pydantic import SecretStr

from app.core.exceptions import ValidationError


@dataclass(slots=True)
class SecretValue:
    """
    A protected container for resolved database credentials.
    
    Warning: Python cannot guarantee immediate memory zeroization.
    This class minimizes accidental exposure through logging or serialization,
    but references should be dropped as soon as they are no longer needed.
    """
    username: str
    password: SecretStr
    database: str | None = None
    host: str | None = None
    port: int | None = None
    ssl_ca: SecretStr | None = None
    ssl_cert: SecretStr | None = None
    ssl_key: SecretStr | None = None
    expires_at: datetime | None = None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.port is not None and not (1 <= self.port <= 65535):
            raise ValidationError("Port must be between 1 and 65535", code="SECRET_PAYLOAD_INVALID")
            
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValidationError("Expires_at must be timezone-aware", code="SECRET_PAYLOAD_INVALID")

    def __repr__(self) -> str:
        fields = []
        fields.append(f"username='{self.username}'")
        fields.append("password='***'")
        if self.database:
            fields.append(f"database='{self.database}'")
        if self.host:
            fields.append(f"host='{self.host}'")
        if self.port:
            fields.append(f"port={self.port}")
        if self.ssl_ca:
            fields.append("ssl_ca='***'")
        if self.ssl_cert:
            fields.append("ssl_cert='***'")
        if self.ssl_key:
            fields.append("ssl_key='***'")
        if self.expires_at:
            fields.append(f"expires_at={self.expires_at.isoformat()}")
        
        # safely encode metadata without exposing potential injected secrets
        safe_meta = json.dumps(self.provider_metadata)
        fields.append(f"provider_metadata={safe_meta}")
            
        return f"SecretValue({', '.join(fields)})"

    def __str__(self) -> str:
        return self.__repr__()

    def get_password(self) -> str:
        """Explicit accessor for password plaintext. Only infrastructure code should call this."""
        return self.password.get_secret_value()
        
    def get_ssl_ca(self) -> str | None:
        """Explicit accessor for ssl_ca plaintext. Only infrastructure code should call this."""
        return self.ssl_ca.get_secret_value() if self.ssl_ca else None
        
    def get_ssl_cert(self) -> str | None:
        """Explicit accessor for ssl_cert plaintext. Only infrastructure code should call this."""
        return self.ssl_cert.get_secret_value() if self.ssl_cert else None
        
    def get_ssl_key(self) -> str | None:
        """Explicit accessor for ssl_key plaintext. Only infrastructure code should call this."""
        return self.ssl_key.get_secret_value() if self.ssl_key else None
