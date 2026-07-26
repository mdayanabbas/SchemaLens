import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable

import structlog
from sqlalchemy.ext.asyncio import AsyncEngine

from app.connectors.pool_key import ConnectorPoolKey
from app.core.config import Settings

logger = structlog.get_logger(__name__)


@dataclass
class RegistryEntry:
    engine: AsyncEngine
    last_used_at: float


class ConnectionPoolRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pools: dict[ConnectorPoolKey, RegistryEntry] = {}
        self._lock = asyncio.Lock()
        # To prevent multiple concurrent creation for the same key
        self._creation_locks: dict[ConnectorPoolKey, asyncio.Lock] = {}

    async def get_or_create(
        self,
        key: ConnectorPoolKey,
        factory_func: Callable[[], Awaitable[AsyncEngine]],
    ) -> AsyncEngine:
        """
        Get an existing engine or create a new one safely.
        """
        async with self._lock:
            entry = self._pools.get(key)
            if entry:
                entry.last_used_at = time.monotonic()
                return entry.engine

            # Get or create a lock for this specific key to avoid creating multiple engines concurrently
            creation_lock = self._creation_locks.setdefault(key, asyncio.Lock())

        async with creation_lock:
            # Check again inside the lock
            async with self._lock:
                entry = self._pools.get(key)
                if entry:
                    entry.last_used_at = time.monotonic()
                    return entry.engine

            # Enforce max pools
            await self._ensure_capacity()

            try:
                engine = await factory_func()
            except Exception:
                async with self._lock:
                    self._creation_locks.pop(key, None)
                raise

            async with self._lock:
                self._pools[key] = RegistryEntry(engine=engine, last_used_at=time.monotonic())
                self._creation_locks.pop(key, None)

            return engine

    async def _ensure_capacity(self) -> None:
        """
        Ensure we don't exceed max registered pools.
        Evict least recently used if necessary.
        Must be called OUTSIDE `_lock` when actually disposing, or carefully within.
        Actually, disposal takes time. Let's do it safely.
        """
        # Find LRU key to evict if we are at capacity
        evict_key = None
        async with self._lock:
            if len(self._pools) >= self.settings.connector_max_registered_pools:
                if not self._pools:
                    return
                # Find the oldest last_used_at
                evict_key = min(self._pools.keys(), key=lambda k: self._pools[k].last_used_at)
        
        if evict_key:
            await self.dispose(evict_key)

    async def dispose(self, key: ConnectorPoolKey) -> None:
        """
        Dispose a specific pool.
        """
        async with self._lock:
            entry = self._pools.pop(key, None)
            
        if entry:
            try:
                await entry.engine.dispose()
            except Exception as e:
                logger.warning(
                    "failed_to_dispose_pool",
                    error=str(e),
                    organization_id=str(key.organization_id),
                    connection_id=str(key.connection_id),
                )

    async def dispose_for_connection(self, organization_id: uuid.UUID, connection_id: uuid.UUID) -> None:
        """
        Dispose all pools associated with a specific connection.
        """
        keys_to_dispose = []
        async with self._lock:
            for key in self._pools.keys():
                if key.organization_id == organization_id and key.connection_id == connection_id:
                    keys_to_dispose.append(key)
        
        for key in keys_to_dispose:
            await self.dispose(key)

    async def dispose_for_organization(self, organization_id: uuid.UUID) -> None:
        """
        Dispose all pools associated with a specific organization.
        """
        keys_to_dispose = []
        async with self._lock:
            for key in self._pools.keys():
                if key.organization_id == organization_id:
                    keys_to_dispose.append(key)
        
        for key in keys_to_dispose:
            await self.dispose(key)

    async def dispose_all(self) -> None:
        """
        Dispose all managed pools.
        """
        keys_to_dispose = []
        async with self._lock:
            keys_to_dispose = list(self._pools.keys())
        
        for key in keys_to_dispose:
            await self.dispose(key)

    async def cleanup_idle(self) -> None:
        """
        Dispose pools that have been idle for too long.
        """
        now = time.monotonic()
        keys_to_dispose = []
        async with self._lock:
            for key, entry in self._pools.items():
                if now - entry.last_used_at > self.settings.connector_pool_idle_ttl_seconds:
                    keys_to_dispose.append(key)
                    
        for key in keys_to_dispose:
            await self.dispose(key)
