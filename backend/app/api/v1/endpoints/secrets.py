import uuid
from typing import Any

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_audit_service, get_database_session, require_permission
from app.audit.service import AuditService
from app.governance.context import AuthorizedOrganizationContext
from app.governance.permissions import Permission
from app.schemas.stored_secret import StoredSecretCreate, StoredSecretPage, StoredSecretRead, StoredSecretRotate
from app.secrets.management import StoredSecretManagementService


router = APIRouter()


def get_management_service(
    session: AsyncSession = Depends(get_database_session),
    audit_service: AuditService = Depends(get_audit_service),
) -> StoredSecretManagementService:
    return StoredSecretManagementService(session, audit_service)


@router.post("/local", status_code=status.HTTP_201_CREATED)
async def create_local_secret(
    schema: StoredSecretCreate,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.CONNECTIONS_MANAGE)),
    service: StoredSecretManagementService = Depends(get_management_service),
) -> dict[str, Any]:
    """Create a new local encrypted secret."""
    return await service.create_local_secret(schema, context)


@router.get("/local", response_model=StoredSecretPage)
async def list_local_secrets(
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    status: str | None = None,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.CONNECTIONS_MANAGE)),
    service: StoredSecretManagementService = Depends(get_management_service),
):
    """List local encrypted secrets."""
    items = await service.list_local_secrets(
        context,
        offset=offset,
        limit=limit,
        status=status,
    )
    # The repository doesn't have a count method currently, so we'll just return items and estimate has_more
    has_more = len(items) == limit
    return {
        "items": items,
        "offset": offset,
        "limit": limit,
        "total": offset + len(items) + (1 if has_more else 0),
        "has_more": has_more
    }


@router.get("/local/{secret_id}", response_model=StoredSecretRead)
async def get_local_secret(
    secret_id: uuid.UUID,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.CONNECTIONS_MANAGE)),
    service: StoredSecretManagementService = Depends(get_management_service),
):
    """Get metadata for a local encrypted secret."""
    return await service.get_local_secret_metadata(secret_id, context)


@router.post("/local/{secret_id}/rotate")
async def rotate_local_secret(
    secret_id: uuid.UUID,
    schema: StoredSecretRotate,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.CONNECTIONS_MANAGE)),
    service: StoredSecretManagementService = Depends(get_management_service),
) -> dict[str, Any]:
    """Rotate a local encrypted secret."""
    return await service.rotate_local_secret(secret_id, schema, context)


@router.post("/local/{secret_id}/disable")
async def disable_local_secret(
    secret_id: uuid.UUID,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.CONNECTIONS_MANAGE)),
    service: StoredSecretManagementService = Depends(get_management_service),
):
    """Disable a local encrypted secret."""
    return await service.disable_local_secret(secret_id, context)
