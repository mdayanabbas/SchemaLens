from enum import StrEnum


class DatabaseDialect(StrEnum):
    POSTGRESQL = "postgresql"


class ConnectionEnvironment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ConnectionStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


class ConnectionTestStatus(StrEnum):
    NEVER_TESTED = "never_tested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SecretProviderType(StrEnum):
    ENVIRONMENT = "environment"
    LOCAL_ENCRYPTED = "local_encrypted"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"


class SSLMode(StrEnum):
    DISABLE = "disable"
    ALLOW = "allow"
    PREFER = "prefer"
    REQUIRE = "require"
    VERIFY_CA = "verify_ca"
    VERIFY_FULL = "verify_full"


class ApprovalMode(StrEnum):
    NEVER = "never"
    RISK_BASED = "risk_based"
    ALWAYS = "always"
