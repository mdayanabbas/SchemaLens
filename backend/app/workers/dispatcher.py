import uuid
from typing import Protocol

from celery.exceptions import CeleryError
from kombu.exceptions import KombuError
import redis.exceptions

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError


class TaskDispatcherProtocol(Protocol):
    async def dispatch_schema_scan(
        self,
        *,
        scan_id: uuid.UUID,
        organization_id: uuid.UUID,
        connection_id: uuid.UUID,
    ) -> str:
        ...


class CeleryTaskDispatcher:
    def __init__(self):
        from app.workers.celery_app import celery_app
        self.app = celery_app
        self.settings = get_settings()

    async def dispatch_schema_scan(
        self,
        *,
        scan_id: uuid.UUID,
        organization_id: uuid.UUID,
        connection_id: uuid.UUID,
    ) -> str:
        # Avoid circular imports and load task lazily
        from app.workers.tasks.schema_scan import run_schema_scan
        
        try:
            # We use send_task or delay, but apply_async is standard
            result = run_schema_scan.apply_async(
                kwargs={
                    "schema_scan_id": str(scan_id),
                    "organization_id": str(organization_id),
                    "connection_id": str(connection_id),
                },
                queue=self.settings.schema_scan_queue_name,
                timeout=self.settings.schema_scan_dispatch_timeout_seconds,
            )
            return str(result.id)
        except (CeleryError, KombuError, redis.exceptions.RedisError, TimeoutError, OSError) as e:
            raise ExternalServiceError(
                message="Failed to dispatch schema scan task to broker.",
                code="SCHEMA_SCAN_DISPATCH_FAILED",
                details={"reason": type(e).__name__}
            )
