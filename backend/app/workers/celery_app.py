from celery import Celery
from kombu import Exchange, Queue

from app.core.config import get_settings


def create_celery_app() -> Celery:
    settings = get_settings()

    app = Celery("schemalens")

    # Use settings but do not instantiate connections until Celery actually runs
    app.conf.update(
        broker_url=settings.celery_broker_url,
        result_backend=settings.celery_result_backend,
        
        # Serialization - strictly JSON
        accept_content=["json"],
        task_serializer="json",
        result_serializer="json",
        
        # Timezone
        timezone="UTC",
        enable_utc=True,
        
        # Acknowledgements and Prefetch
        task_acks_late=settings.celery_task_acks_late,
        worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
        
        # Eager Mode (Testing/Local)
        task_always_eager=settings.celery_task_always_eager,
        task_eager_propagates=settings.celery_task_eager_propagates,
        
        # Task Tracking
        task_track_started=True,
        task_ignore_result=False,

        # Explicit routing to prevent default queue fallback if possible, though we define it explicitly
        task_default_queue=settings.schema_scan_queue_name,
        task_queues=(
            Queue(
                settings.schema_scan_queue_name, 
                Exchange(settings.schema_scan_queue_name), 
                routing_key=settings.schema_scan_queue_name
            ),
        ),
        
        # Time limits
        task_soft_time_limit=settings.schema_scan_task_soft_time_limit_seconds,
        task_time_limit=settings.schema_scan_task_hard_time_limit_seconds,
    )

    # Autodiscover specific trusted tasks packages only
    app.autodiscover_tasks(["app.workers.tasks"])

    return app


celery_app = create_celery_app()
