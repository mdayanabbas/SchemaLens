import pytest

from pydantic import ValidationError

from app.models.connection_enums import ConnectionEnvironment, DatabaseDialect, SecretProviderType, SSLMode, ApprovalMode
from app.schemas.database_connection import DatabaseConnectionCreate
from app.schemas.connection_policy import ConnectionPolicyCreate


def test_database_connection_create_schema_valid():
    schema = DatabaseConnectionCreate(
        name="My DB",
        environment=ConnectionEnvironment.DEVELOPMENT,
        dialect=DatabaseDialect.POSTGRESQL,
        host="localhost",
        port=5432,
        database_name="postgres",
        ssl_mode=SSLMode.REQUIRE,
        secret_provider=SecretProviderType.ENVIRONMENT,
        secret_reference="DB_PASSWORD",
    )
    assert schema.name == "My DB"


def test_database_connection_create_schema_invalid_host():
    with pytest.raises(ValidationError):
        DatabaseConnectionCreate(
            name="My DB",
            environment=ConnectionEnvironment.DEVELOPMENT,
            dialect=DatabaseDialect.POSTGRESQL,
            host="postgresql://user:pass@localhost",
            port=5432,
            database_name="postgres",
            ssl_mode=SSLMode.REQUIRE,
            secret_provider=SecretProviderType.ENVIRONMENT,
            secret_reference="DB_PASSWORD",
        )


def test_connection_policy_create_schema_valid():
    schema = ConnectionPolicyCreate(
        allow_query_generation=True,
        allow_query_execution=True,
    )
    assert schema.allow_query_execution is True


def test_connection_policy_create_schema_invalid_execution():
    with pytest.raises(ValidationError, match="Cannot enable execution while generation is disabled."):
        ConnectionPolicyCreate(
            allow_query_generation=False,
            allow_query_execution=True,
        )


def test_connection_policy_create_schema_invalid_limits():
    with pytest.raises(ValidationError):
        ConnectionPolicyCreate(
            max_rows=0, # Must be > 0
        )
