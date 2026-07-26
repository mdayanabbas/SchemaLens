import ssl
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine

from app.connectors.exceptions import ConnectorConfigurationError
from app.connectors.pool_key import ConnectorMode
from app.connectors.postgres.engine_factory import PostgreSQLEngineFactory
from app.core.config import Settings
from app.models.connection_enums import SSLMode
from app.models.database_connection import DatabaseConnection
from app.secrets.value import SecretValue


@pytest.fixture
def settings():
    return Settings(
        connector_application_name_prefix="test_app",
        connector_pool_size=5,
        connector_max_overflow=2,
    )


@pytest.fixture
def factory(settings):
    return PostgreSQLEngineFactory(settings)


@pytest.fixture
def connection():
    return DatabaseConnection(
        id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
        host="db.example.com",
        port=5432,
        database_name="test_db",
        ssl_mode=SSLMode.REQUIRE,
    )


@pytest.fixture
def secret():
    return SecretValue(
        username="db_user",
        password=SecretStr("supersecret"),
    )


@pytest.mark.asyncio
async def test_create_engine_test_mode(factory, connection, secret):
    engine = await factory.create_engine(
        connection=connection,
        secret=secret,
        mode=ConnectorMode.TEST,
    )
    
    assert isinstance(engine, AsyncEngine)
    assert engine.url.username == "db_user"
    assert engine.url.password == "supersecret"
    assert engine.url.host == "db.example.com"
    assert engine.url.port == 5432
    assert engine.url.database == "test_db"
    
    assert engine.pool.__class__.__name__ == "NullPool"


@pytest.mark.asyncio
async def test_create_engine_execution_mode(factory, connection, secret):
    engine = await factory.create_engine(
        connection=connection,
        secret=secret,
        mode=ConnectorMode.EXECUTION,
    )
    
    assert engine.pool.__class__.__name__ == "AsyncAdaptedQueuePool"
    assert engine.pool.size() == 5


@pytest.mark.asyncio
async def test_ssl_mode_disable(factory, connection, secret):
    connection.ssl_mode = SSLMode.DISABLE
    engine = await factory.create_engine(
        connection=connection,
        secret=secret,
        mode=ConnectorMode.TEST,
    )
    assert "ssl" not in engine.engine.pool._creator.keywords.get("connect_args", {})


@pytest.mark.asyncio
async def test_ssl_mode_require(factory, connection, secret):
    connection.ssl_mode = SSLMode.REQUIRE
    engine = await factory.create_engine(
        connection=connection,
        secret=secret,
        mode=ConnectorMode.TEST,
    )
    
    # We inspect the sync engine's connect args directly
    assert "ssl" in engine.engine.pool._creator.keywords.get("connect_args", {})


@pytest.mark.asyncio
async def test_ssl_mode_verify_ca(factory, connection, secret):
    connection.ssl_mode = SSLMode.VERIFY_CA
    engine = await factory.create_engine(
        connection=connection,
        secret=secret,
        mode=ConnectorMode.TEST,
    )
    
    ssl_context = engine.engine.pool._creator.keywords["connect_args"]["ssl"]
    assert isinstance(ssl_context, ssl.SSLContext)
    assert not ssl_context.check_hostname


@pytest.mark.asyncio
async def test_ssl_mode_verify_full(factory, connection, secret):
    connection.ssl_mode = SSLMode.VERIFY_FULL
    engine = await factory.create_engine(
        connection=connection,
        secret=secret,
        mode=ConnectorMode.TEST,
    )
    
    ssl_context = engine.engine.pool._creator.keywords["connect_args"]["ssl"]
    assert isinstance(ssl_context, ssl.SSLContext)
    assert ssl_context.check_hostname


@pytest.mark.asyncio
async def test_custom_cert_material_not_supported(factory, connection):
    connection.ssl_mode = SSLMode.REQUIRE
    secret = SecretValue(
        username="user",
        password=SecretStr("pass"),
        ssl_ca=SecretStr("ca_cert_data")
    )
    
    with pytest.raises(ConnectorConfigurationError) as exc:
        await factory.create_engine(
            connection=connection,
            secret=secret,
            mode=ConnectorMode.TEST,
        )
    assert "not supported" in str(exc.value)
