import uuid

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import get_organization_context, get_schema_scan_service
from app.governance.context import AuthorizedOrganizationContext
from app.models.schema_scan_enums import SchemaScanStatus
from app.schemas.schema_scan import (
    SchemaScanCancelResponse,
    SchemaScanCreate,
    SchemaScanPage,
    SchemaScanRead,
    SchemaScanTransitionRead,
)
from app.services.schema_scan import SchemaScanService


router = APIRouter()


@router.post(
    "/connections/{connection_id}/scans",
    response_model=SchemaScanRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request a new schema scan",
)
async def request_schema_scan(
    connection_id: uuid.UUID,
    schema: SchemaScanCreate,
    context: AuthorizedOrganizationContext = Depends(get_organization_context),
    scan_service: SchemaScanService = Depends(get_schema_scan_service),
) -> SchemaScanRead:
    """Request a new background schema scan for a connection."""
    return await scan_service.request_scan(
        connection_id=connection_id, schema=schema, context=context
    )


@router.get(
    "/connections/{connection_id}/scans",
    response_model=SchemaScanPage,
    summary="List schema scans for a connection",
)
async def list_schema_scans(
    connection_id: uuid.UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    scan_status: SchemaScanStatus | None = Query(None, alias="status"),
    context: AuthorizedOrganizationContext = Depends(get_organization_context),
    scan_service: SchemaScanService = Depends(get_schema_scan_service),
) -> SchemaScanPage:
    """List schema scans for a specific connection."""
    return await scan_service.list_scans(
        connection_id=connection_id,
        offset=offset,
        limit=limit,
        status=scan_status,
        context=context,
    )


@router.get(
    "/schema-scans/{scan_id}",
    response_model=SchemaScanRead,
    summary="Get a schema scan by ID",
)
async def get_schema_scan(
    scan_id: uuid.UUID,
    context: AuthorizedOrganizationContext = Depends(get_organization_context),
    scan_service: SchemaScanService = Depends(get_schema_scan_service),
) -> SchemaScanRead:
    """Get details of a specific schema scan."""
    return await scan_service.get_scan(scan_id=scan_id, context=context)


@router.post(
    "/schema-scans/{scan_id}/cancel",
    response_model=SchemaScanCancelResponse,
    summary="Cancel a schema scan",
)
async def cancel_schema_scan(
    scan_id: uuid.UUID,
    context: AuthorizedOrganizationContext = Depends(get_organization_context),
    scan_service: SchemaScanService = Depends(get_schema_scan_service),
) -> SchemaScanCancelResponse:
    """Request cancellation of a schema scan."""
    return await scan_service.cancel_scan(scan_id=scan_id, context=context)


@router.get(
    "/schema-scans/{scan_id}/transitions",
    response_model=list[SchemaScanTransitionRead],
    summary="List schema scan transitions",
)
async def list_schema_scan_transitions(
    scan_id: uuid.UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    context: AuthorizedOrganizationContext = Depends(get_organization_context),
    scan_service: SchemaScanService = Depends(get_schema_scan_service),
) -> list[SchemaScanTransitionRead]:
    """List the state transition history for a schema scan."""
    return await scan_service.list_scan_transitions(
        scan_id=scan_id, offset=offset, limit=limit, context=context
    )
