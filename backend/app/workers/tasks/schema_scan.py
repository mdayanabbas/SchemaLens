import asyncio
import uuid

import structlog

from app.audit.service import AuditService
from app.db.session import async_session_factory
from app.workers.celery_app import celery_app
from app.workflows.schema_scan_workflow import SchemaScanWorkflow


logger = structlog.get_logger(__name__)


async def _run_schema_scan_async(
    schema_scan_id: str,
    organization_id: str,
    connection_id: str,
) -> None:
    try:
        scan_id_uuid = uuid.UUID(schema_scan_id)
        org_id_uuid = uuid.UUID(organization_id)
        conn_id_uuid = uuid.UUID(connection_id)
    except ValueError:
        logger.error("Invalid UUID provided to schema scan task")
        return

    async with async_session_factory() as session:
        audit_service = AuditService(session)
        workflow = SchemaScanWorkflow(session=session, audit_service=audit_service)
        try:
            await workflow.run(
                scan_id=scan_id_uuid,
                organization_id=org_id_uuid,
                connection_id=conn_id_uuid,
            )
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("Schema scan workflow failed unexpectedly")
            raise


@celery_app.task(bind=True, name="app.workers.tasks.schema_scan.run_schema_scan")
def run_schema_scan(
    self,
    schema_scan_id: str,
    organization_id: str,
    connection_id: str,
) -> None:
    """Celery task wrapper to execute schema scan workflow."""
    # Run the async workflow inside a synchronous Celery task
    asyncio.run(_run_schema_scan_async(
        schema_scan_id=schema_scan_id,
        organization_id=organization_id,
        connection_id=connection_id,
    ))
