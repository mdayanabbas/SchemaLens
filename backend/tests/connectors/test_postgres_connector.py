import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.connectors.exceptions import (
    ConnectorAuthenticationError,
    ConnectorConfigurationError,
    ConnectorDatabaseNotFoundError,
)
from app.connectors.pool_key import ConnectorMode
from app.connectors.postgres.connector import PostgreSQLConnector
from app.connectors.postgres.engine_factory import PostgreSQLEngineFactory
from app.connectors.types import WarningSeverity
from app.core.config import Settings
from app.models.connection_enums import DatabaseDialect
from app.models.connection_policy import ConnectionPolicy
from app.models.database_connection import DatabaseConnection
from app.secrets.service import SecretResolutionService


@pytest.fixture
def settings():
    return Settings(
        connector_test_statement_timeout_ms=5000,
        connector_test_lock_timeout_ms=3000,
    )


@pytest.fixture
def mock_secret_service():
    return MagicMock(spec=SecretResolutionService)


@pytest.fixture
def mock_engine_factory():
    return MagicMock(spec=PostgreSQLEngineFactory)


@pytest.fixture
def connector(settings, mock_secret_service, mock_engine_factory):
    return PostgreSQLConnector(
        settings=settings,
        secret_resolution_service=mock_secret_service,
        engine_factory=mock_engine_factory,
    )


@pytest.fixture
def organization_id():
    return uuid.uuid4()


@pytest.fixture
def connection(organization_id):
    return DatabaseConnection(
        id=uuid.uuid4(),
        organization_id=organization_id,
        dialect=DatabaseDialect.POSTGRESQL,
        status="active",
        secret_provider="environment",
        secret_reference="DB_URL"
    )


@pytest.fixture
def policy(organization_id, connection):
    return ConnectionPolicy(
        id=uuid.uuid4(),
        organization_id=organization_id,
        connection_id=connection.id,
        approved_schemas_json=["public", "analytics"]
    )


@pytest.fixture
def mock_engine():
    engine = AsyncMock(spec=AsyncEngine)
    conn = AsyncMock()
    
    # Mock context managers for connect() and begin()
    engine.connect.return_value.__aenter__.return_value = conn
    conn.begin.return_value.__aenter__.return_value = AsyncMock()
    
    # Mock execute results
    async def mock_execute(stmt):
        sql = str(stmt)
        if "current_database" in sql:
            res = MagicMock()
            res.scalar.return_value = "test_db"
            return res
        if "server_version" in sql:
            res = MagicMock()
            res.scalar.return_value = "15.3"
            return res
        if "information_schema.schemata" in sql:
            return [("public",), ("information_schema",), ("pg_catalog",)]
        return MagicMock()
        
    conn.execute.side_effect = mock_execute
    
    return engine


@pytest.mark.asyncio
async def test_test_connection_success(
    connector, mock_secret_service, mock_engine_factory,
    organization_id, connection, policy, mock_engine
):
    mock_engine_factory.create_engine.return_value = mock_engine
    
    result = await connector.test_connection(
        organization_id=organization_id,
        connection=connection,
        policy=policy
    )
    
    assert result.success is True
    assert result.server_version == "15.3"
    assert result.database_name == "test_db"
    assert "public" in result.reachable_schemas
    
    # Analytics was approved but not reachable
    assert "public" in result.approved_schemas_found
    assert "analytics" in result.approved_schemas_missing
    
    # We should have no warnings about version, but maybe missing schemas
    assert any("analytics" in w.message for w in result.warnings if w.code == "ALL_APPROVED_SCHEMAS_MISSING") is False # We have public
    
    mock_engine.dispose.assert_called_once()


@pytest.mark.asyncio
async def test_test_connection_authentication_error(
    connector, mock_secret_service, mock_engine_factory,
    organization_id, connection, policy, mock_engine
):
    mock_engine_factory.create_engine.return_value = mock_engine
    
    # Make connect() fail with auth error
    mock_engine.connect.return_value.__aenter__.side_effect = asyncpg.exceptions.InvalidPasswordError("auth failed")
    
    with pytest.raises(ConnectorAuthenticationError):
        await connector.test_connection(
            organization_id=organization_id,
            connection=connection,
            policy=policy
        )


@pytest.mark.asyncio
async def test_test_connection_invalid_context(
    connector, organization_id, connection, policy
):
    connection.organization_id = uuid.uuid4() # mismatch
    
    with pytest.raises(ConnectorConfigurationError):
        await connector.test_connection(
            organization_id=organization_id,
            connection=connection,
            policy=policy
        )


@pytest.mark.asyncio
async def test_test_connection_empty_schemas_warning(
    connector, mock_secret_service, mock_engine_factory,
    organization_id, connection, policy, mock_engine
):
    policy.approved_schemas_json = []
    mock_engine_factory.create_engine.return_value = mock_engine
    
    result = await connector.test_connection(
        organization_id=organization_id,
        connection=connection,
        policy=policy
    )
    
    assert result.success is True
    assert len(result.warnings) == 1
    assert result.warnings[0].code == "EMPTY_APPROVED_SCHEMAS"
    assert result.warnings[0].severity == WarningSeverity.WARNING


@pytest.mark.asyncio
async def test_test_connection_all_schemas_missing_warning(
    connector, mock_secret_service, mock_engine_factory,
    organization_id, connection, policy, mock_engine
):
    policy.approved_schemas_json = ["unknown_schema"]
    mock_engine_factory.create_engine.return_value = mock_engine
    
    result = await connector.test_connection(
        organization_id=organization_id,
        connection=connection,
        policy=policy
    )
    
    assert result.success is True
    assert len(result.warnings) == 1
    assert result.warnings[0].code == "ALL_APPROVED_SCHEMAS_MISSING"
    assert result.warnings[0].severity == WarningSeverity.CRITICAL


@pytest.mark.asyncio
async def test_list_namespaces(
    connector, mock_secret_service, mock_engine_factory,
    organization_id, connection, policy, mock_engine
):
    mock_engine_factory.create_engine.return_value = mock_engine
    
    namespaces = await connector.list_namespaces(
        organization_id=organization_id,
        connection=connection,
        policy=policy
    )
    
    assert len(namespaces) == 3
    
    # Check public
    public_ns = next(n for n in namespaces if n.name == "public")
    assert public_ns.is_system is False
    assert public_ns.is_approved is True
    
    # Check pg_catalog
    pg_catalog_ns = next(n for n in namespaces if n.name == "pg_catalog")
    assert pg_catalog_ns.is_system is True
    assert pg_catalog_ns.is_approved is False
