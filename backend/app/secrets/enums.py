from enum import StrEnum


class SecretStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    ROTATED = "rotated"


class SecretField(StrEnum):
    USERNAME = "username"
    PASSWORD = "password"
    DATABASE = "database"
    HOST = "host"
    PORT = "port"
    SSL_CA = "ssl_ca"
    SSL_CERT = "ssl_cert"
    SSL_KEY = "ssl_key"


class SecretResolutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
