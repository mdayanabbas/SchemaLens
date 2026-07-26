from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.connectors.pool_registry import ConnectionPoolRegistry
from app.connectors.postgres.connector import PostgreSQLConnector
from app.connectors.postgres.engine_factory import PostgreSQLEngineFactory
from app.connectors.registry import ConnectorRegistry
from app.core.config import Settings
from app.secrets.service import SecretResolutionService


def build_connector_registry(
    postgres_connector: PostgreSQLConnector,
) -> ConnectorRegistry:
    """Build and configure the connector registry for a request."""
    registry = ConnectorRegistry()
    registry.register(postgres_connector)
    return registry

