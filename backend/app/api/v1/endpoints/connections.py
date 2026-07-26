import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_audit_service,
    get_current_user,
    get_database_session,
    get_connection_test_service,
    require_permission,
)
from app.audit.service import AuditService
from app.governance.context import AuthorizedOrganizationContext
from app.governance.permissions import Permission
from app.models.user import User
from app.schemas.connection_policy import ConnectionPolicyRead, ConnectionPolicyUpdate
from app.schemas.connection_test import ConnectionTestResponse
from app.schemas.database_connection import (
    DatabaseConnectionCreate,
    DatabaseConnectionRead,
    DatabaseConnectionSummaryRead,
    DatabaseConnectionUpdate,
)
from app.services.connection_policy import ConnectionPolicyService
from app.services.connection_test import ConnectionTestService
from app.services.database_connection import DatabaseConnectionService



router = APIRouter()


def get_connection_service(
    session: AsyncSession = Depends(get_database_session),
    audit_service: AuditService = Depends(get_audit_service),
) -> DatabaseConnectionService:
    return DatabaseConnectionService(session, audit_service)


def get_policy_service(
    session: AsyncSession = Depends(get_database_session),
    audit_service: AuditService = Depends(get_audit_service),
) -> ConnectionPolicyService:
    return ConnectionPolicyService(session, audit_service)


@router.post("", response_model=DatabaseConnectionRead, status_code=status.HTTP_201_CREATED)
async def create_connection(
    schema: DatabaseConnectionCreate,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.CONNECTIONS_MANAGE)),
    service: DatabaseConnectionService = Depends(get_connection_service),
):
    """Create a new database connection."""
    return await service.create_connection(schema, context)


@router.get("", response_model=dict)
async def list_connections(
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    environment: str | None = None,
    status: str | None = None,
    dialect: str | None = None,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.CONNECTIONS_READ)),
    service: DatabaseConnectionService = Depends(get_connection_service),
):
    """List connections for the current organization."""
    items, total = await service.list_connections(
        context,
        offset=offset,
        limit=limit,
        environment=environment,
        status=status,
        dialect=dialect,
    )
    return {
        "items": items,
        "offset": offset,
        "limit": limit,
        "total": total,
        "has_more": (offset + len(items) < total),
    }


@router.get("/{connection_id}", response_model=DatabaseConnectionRead)
async def get_connection(
    connection_id: uuid.UUID,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.CONNECTIONS_READ)),
    service: DatabaseConnectionService = Depends(get_connection_service),
):
    """Get a specific database connection."""
    return await service.get_connection(connection_id, context)


@router.patch("/{connection_id}", response_model=DatabaseConnectionRead)
async def update_connection(
    connection_id: uuid.UUID,
    schema: DatabaseConnectionUpdate,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.CONNECTIONS_MANAGE)),
    service: DatabaseConnectionService = Depends(get_connection_service),
):
    """Update a database connection."""
    return await service.update_connection(connection_id, schema, context)


@router.post("/{connection_id}/disable")
async def disable_connection(
    connection_id: uuid.UUID,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.CONNECTIONS_MANAGE)),
    service: DatabaseConnectionService = Depends(get_connection_service),
):
    """Disable a database connection."""
    await service.disable_connection(connection_id, context)
    return {"status": "success", "message": "Connection disabled"}


@router.get("/{connection_id}/policy", response_model=ConnectionPolicyRead)
async def get_connection_policy(
    connection_id: uuid.UUID,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.POLICIES_READ)),
    service: ConnectionPolicyService = Depends(get_policy_service),
):
    """Get the policy for a database connection."""
    return await service.get_policy(connection_id, context)


@router.patch("/{connection_id}/policy", response_model=ConnectionPolicyRead)
async def update_connection_policy(
    connection_id: uuid.UUID,
    schema: ConnectionPolicyUpdate,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.POLICIES_MANAGE)),
    service: ConnectionPolicyService = Depends(get_policy_service),
):
    """Update the policy for a database connection."""
    return await service.update_policy(connection_id, schema, context)


@router.post("/{connection_id}/test", response_model=ConnectionTestResponse)
async def test_connection(
    connection_id: uuid.UUID,
    context: AuthorizedOrganizationContext = Depends(require_permission(Permission.CONNECTIONS_TEST)),
    acting_user: User = Depends(get_current_user),
    service: ConnectionTestService = Depends(get_connection_test_service),
):
    """Test the database connection and configuration."""
    return await service.test_connection(
        context=context,
        acting_user=acting_user,
        connection_id=connection_id,
    )
