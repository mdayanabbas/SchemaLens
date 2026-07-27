from typing import AsyncGenerator

from fastapi import Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_database_session
from app.models.user import User


bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    token: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_database_session),
    settings: Settings = Depends(get_settings),
) -> User:
    from app.services.authentication import AuthenticationService, AuthenticationException
    
    if not token:
        raise AuthenticationException("Not authenticated.", code="NOT_AUTHENTICATED")
        
    auth_service = AuthenticationService(session, settings)
    return await auth_service.authenticate_access_token(token.credentials)


async def get_organization_context(
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_database_session),
) -> "AuthorizedOrganizationContext":
    import uuid
    from app.core.exceptions import AuthorizationError
    from app.governance.authorization import AuthorizationService
    from app.core.request_context import set_organization_context, set_user_context

    org_id_str = request.headers.get("X-Organization-ID")
    if not org_id_str:
        raise AuthorizationError(
            message="X-Organization-ID header is required.",
            code="ORGANIZATION_CONTEXT_REQUIRED",
        )
    
    try:
        org_id = uuid.UUID(org_id_str)
    except ValueError:
        raise AuthorizationError(
            message="Invalid organization ID format.",
            code="INVALID_ORGANIZATION_CONTEXT",
        )

    auth_service = AuthorizationService(session)
    
    # We use a dummy permission to load context. But `get_organization_context` might be
    # used alone without `require_permission` in some cases (e.g., just needing context).
    # The requirement says:
    # 7. Build AuthorizedOrganizationContext.
    # We can just require ORGANIZATION_READ as baseline for having context, or 
    # we can just use AuthorizationService to load it. 
    # Wait, the spec says: "require_permission(permission: Permission)" dependency factory.
    # Let's implement it correctly.
    
    from app.governance.permissions import Permission
    # For baseline context, organization.read is sensible.
    context = await auth_service.require_permission(
        user=user,
        organization_id=org_id,
        permission=Permission.ORGANIZATION_READ,
    )
    
    # Set context variables
    set_user_context(user.id)
    set_organization_context(org_id, context.membership_id, context.role)
    
    return context


from typing import Callable, Awaitable
from app.governance.permissions import Permission
from app.governance.context import AuthorizedOrganizationContext

def require_permission(
    permission: Permission,
) -> Callable[..., Awaitable[AuthorizedOrganizationContext]]:
    async def permission_dependency(
        request: Request,
        user: User = Depends(get_current_user),
        session: AsyncSession = Depends(get_database_session),
    ) -> AuthorizedOrganizationContext:
        import uuid
        from app.core.exceptions import AuthorizationError
        from app.governance.authorization import AuthorizationService
        from app.core.request_context import set_organization_context, set_user_context

        org_id_str = request.headers.get("X-Organization-ID")
        if not org_id_str:
            raise AuthorizationError(
                message="X-Organization-ID header is required.",
                code="ORGANIZATION_CONTEXT_REQUIRED",
            )
        
        try:
            org_id = uuid.UUID(org_id_str)
        except ValueError:
            raise AuthorizationError(
                message="Invalid organization ID format.",
                code="INVALID_ORGANIZATION_CONTEXT",
            )

        auth_service = AuthorizationService(session)
        context = await auth_service.require_permission(
            user=user,
            organization_id=org_id,
            permission=permission,
        )

        set_user_context(user.id)
        set_organization_context(org_id, context.membership_id, context.role)
        
        return context

    return permission_dependency


async def get_audit_service(
    session: AsyncSession = Depends(get_database_session),
) -> "AuditService":
    from app.audit.service import AuditService
    return AuditService(session)


def get_connection_pool_registry(request: Request) -> "ConnectionPoolRegistry":
    from app.connectors.pool_registry import ConnectionPoolRegistry
    return request.app.state.pool_registry


async def get_secret_resolution_service(
    session: AsyncSession = Depends(get_database_session),
    audit_service: "AuditService" = Depends(get_audit_service),
) -> "SecretResolutionService":
    from app.secrets.service import SecretResolutionService
    return SecretResolutionService(session, audit_service)


async def get_connector_registry(
    settings: Settings = Depends(get_settings),
    pool_registry: "ConnectionPoolRegistry" = Depends(get_connection_pool_registry),
    secret_resolution_service: "SecretResolutionService" = Depends(get_secret_resolution_service),
) -> "ConnectorRegistry":
    from app.connectors.factory import build_connector_registry
    from app.connectors.postgres.connector import PostgreSQLConnector
    from app.connectors.postgres.engine_factory import PostgreSQLEngineFactory

    engine_factory = PostgreSQLEngineFactory(settings)
    postgres_connector = PostgreSQLConnector(
        settings=settings,
        secret_resolution_service=secret_resolution_service,
        engine_factory=engine_factory,
        pool_registry=pool_registry,
    )
    return build_connector_registry(postgres_connector)


async def get_database_connection_service(
    session: AsyncSession = Depends(get_database_session),
    audit_service: "AuditService" = Depends(get_audit_service),
    pool_registry: "ConnectionPoolRegistry" = Depends(get_connection_pool_registry),
) -> "DatabaseConnectionService":
    from app.services.database_connection import DatabaseConnectionService
    return DatabaseConnectionService(session, audit_service, pool_registry)


async def get_connection_test_service(
    session: AsyncSession = Depends(get_database_session),
    audit_service: "AuditService" = Depends(get_audit_service),
    connector_registry: "ConnectorRegistry" = Depends(get_connector_registry),
) -> "ConnectionTestService":
    from app.services.connection_test import ConnectionTestService
    return ConnectionTestService(session, audit_service, connector_registry)


async def get_schema_scan_service(
    session: AsyncSession = Depends(get_database_session),
    audit_service: "AuditService" = Depends(get_audit_service),
) -> "SchemaScanService":
    from app.services.schema_scan import SchemaScanService
    from app.workers.dispatcher import CeleryTaskDispatcher
    from app.workers.cancellation import TaskCancellationService
    
    dispatcher = CeleryTaskDispatcher()
    cancellation_service = TaskCancellationService()
    
    return SchemaScanService(
        session=session,
        audit_service=audit_service,
        dispatcher=dispatcher,
        cancellation_service=cancellation_service,
    )


__all__ = [
    "get_database_session",
    "get_current_user",
    "get_organization_context",
    "require_permission",
    "get_audit_service",
    "get_connection_pool_registry",
    "get_connector_registry",
    "get_secret_resolution_service",
    "get_database_connection_service",
    "get_connection_test_service",
    "get_schema_scan_service",
]
