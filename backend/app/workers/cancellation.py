import structlog

from celery.exceptions import CeleryError
from kombu.exceptions import KombuError
import redis.exceptions


logger = structlog.get_logger(__name__)


class TaskCancellationService:
    def __init__(self):
        from app.workers.celery_app import celery_app
        self.app = celery_app

    async def request_revoke(self, task_id: str) -> None:
        """
        Attempts to revoke a Celery task.
        This is a best-effort secondary signal; the persistent database state is authoritative.
        """
        try:
            # Do not use terminate=True for ordinary cancellation
            self.app.control.revoke(task_id, terminate=False)
            safe_task_id = task_id[:8] + "..." if len(task_id) > 8 else "..."
            logger.info("Task revocation requested", task_id_fragment=safe_task_id)
        except (CeleryError, KombuError, redis.exceptions.RedisError, TimeoutError, OSError) as e:
            # We catch all exceptions because broker failure should not undo the database cancellation state
            logger.warning(
                "Failed to revoke task in broker, relying on database cancellation state.",
                reason=type(e).__name__
            )
