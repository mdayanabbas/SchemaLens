import pytest
from unittest.mock import AsyncMock, MagicMock

from app.connectors.postgres.introspector import PostgreSQLSchemaIntrospector
from app.core.config import Settings
from app.models.connection_policy import ConnectionPolicy

@pytest.mark.asyncio
async def test_introspector_validate_approved_schemas():
    settings = Settings()
    introspector = PostgreSQLSchemaIntrospector(settings)
    
    mock_engine = AsyncMock()
    mock_conn = AsyncMock()
    mock_engine.connect.return_value.__aenter__.return_value = mock_conn
    
    policy = ConnectionPolicy(
        approved_schemas_json=["public", "app_schema"]
    )
    
    # Should not raise exception
    valid_schemas = introspector._validate_approved_schemas(["public"], policy)
    assert valid_schemas == ["public"]
    
    # Should raise error for unapproved schemas
    from app.core.exceptions import AppError
    with pytest.raises(AppError) as exc_info:
        introspector._validate_approved_schemas(["public", "unapproved"], policy)
    assert exc_info.value.code == "SCHEMA_NOT_APPROVED"

@pytest.mark.asyncio
async def test_introspector_fetch_server_info():
    settings = Settings()
    introspector = PostgreSQLSchemaIntrospector(settings)
    
    mock_conn = AsyncMock()
    mock_res_version = MagicMock()
    mock_res_version.scalar.return_value = "15.0"
    
    mock_res_db = MagicMock()
    mock_res_db.scalar.return_value = "test_db"
    
    mock_conn.execute.side_effect = [mock_res_version, mock_res_db]
    
    version, db_name = await introspector._fetch_server_info(mock_conn)
    assert version == "15.0"
    assert db_name == "test_db"
