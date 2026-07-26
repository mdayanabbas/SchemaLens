import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest

from app.connectors.pool_key import ConnectorMode, ConnectorPoolKey
from app.connectors.pool_registry import ConnectionPoolRegistry
from app.core.config import Settings


@pytest.fixture
def settings():
    return Settings(
        connector_max_registered_pools=3,
        connector_pool_idle_ttl_seconds=3600,
    )


@pytest.fixture
def registry(settings):
    return ConnectionPoolRegistry(settings)


@pytest.fixture
def mock_engine():
    engine = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_get_or_create(registry):
    key = ConnectorPoolKey(
        organization_id=uuid.uuid4(),
        connection_id=uuid.uuid4(),
        credential_fingerprint="fingerprint1",
        connector_mode=ConnectorMode.EXECUTION
    )
    
    engine1 = AsyncMock()
    async def factory():
        return engine1
        
    res1 = await registry.get_or_create(key, factory)
    assert res1 is engine1
    
    # Should return existing
    res2 = await registry.get_or_create(key, factory)
    assert res2 is engine1


@pytest.mark.asyncio
async def test_ensure_capacity(registry):
    engines = []
    
    # Fill capacity (3)
    for i in range(3):
        key = ConnectorPoolKey(
            organization_id=uuid.uuid4(),
            connection_id=uuid.uuid4(),
            credential_fingerprint=f"fingerprint{i}",
            connector_mode=ConnectorMode.EXECUTION
        )
        engine = AsyncMock()
        engines.append(engine)
        
        async def factory(e=engine):
            return e
            
        await registry.get_or_create(key, factory)
        # Sleep tiny bit to ensure timestamps differ
        await asyncio.sleep(0.01)
        
    # Should evict oldest (engines[0])
    key = ConnectorPoolKey(
        organization_id=uuid.uuid4(),
        connection_id=uuid.uuid4(),
        credential_fingerprint="fingerprint_new",
        connector_mode=ConnectorMode.EXECUTION
    )
    engine_new = AsyncMock()
    async def factory_new():
        return engine_new
        
    await registry.get_or_create(key, factory_new)
    
    engines[0].dispose.assert_called_once()
    assert len(registry._pools) == 3


@pytest.mark.asyncio
async def test_dispose_all(registry):
    key1 = ConnectorPoolKey(
        organization_id=uuid.uuid4(),
        connection_id=uuid.uuid4(),
        credential_fingerprint="fingerprint1",
        connector_mode=ConnectorMode.EXECUTION
    )
    key2 = ConnectorPoolKey(
        organization_id=uuid.uuid4(),
        connection_id=uuid.uuid4(),
        credential_fingerprint="fingerprint2",
        connector_mode=ConnectorMode.EXECUTION
    )
    
    engine1 = AsyncMock()
    engine2 = AsyncMock()
    
    async def factory1(): return engine1
    async def factory2(): return engine2
    
    await registry.get_or_create(key1, factory1)
    await registry.get_or_create(key2, factory2)
    
    await registry.dispose_all()
    
    engine1.dispose.assert_called_once()
    engine2.dispose.assert_called_once()
    assert len(registry._pools) == 0
