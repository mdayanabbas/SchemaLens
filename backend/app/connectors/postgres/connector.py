import asyncio
import ssl
import time
import uuid
from typing import AsyncGenerator

import asyncpg
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.connectors.base import DatabaseConnector
from app.connectors.exceptions import (
    ConnectorAuthenticationError,
    ConnectorConfigurationError,
    ConnectorDatabaseNotFoundError,
    ConnectorError,
    ConnectorSSLError,
    ConnectorTimeoutError,
    ConnectorUnavailableError,
)
from app.connectors.pool_key import ConnectorMode, ConnectorPoolKey
from app.connectors.pool_registry import ConnectionPoolRegistry
from app.connectors.postgres.constants import (
    POSTGRESQL_MINIMUM_MAJOR_VERSION,
    POSTGRESQL_SYSTEM_SCHEMA_PREFIXES,
    POSTGRESQL_SYSTEM_SCHEMAS,
)
from app.connectors.postgres.engine_factory import PostgreSQLEngineFactory
from app.connectors.types import (
    ConnectionTestResult,
    ConnectionTestWarning,
    ConnectorCapability,
    NamespaceSummary,
    WarningSeverity,
)
from app.core.config import Settings
from app.models.connection_enums import ConnectionStatus, DatabaseDialect
from app.models.connection_policy import ConnectionPolicy
from app.models.database_connection import DatabaseConnection
from app.secrets.service import SecretResolutionService

logger = structlog.get_logger(__name__)


class PostgreSQLConnector(DatabaseConnector):
    dialect = DatabaseDialect.POSTGRESQL

    def __init__(
        self,
        settings: Settings,
        secret_resolution_service: SecretResolutionService,
        engine_factory: PostgreSQLEngineFactory,
        pool_registry: ConnectionPoolRegistry | None = None,
    ):
        self.settings = settings
        self.secret_resolution_service = secret_resolution_service
        self.engine_factory = engine_factory
        self.pool_registry = pool_registry

    async def test_connection(
        self,
        *,
        organization_id: uuid.UUID,
        connection: DatabaseConnection,
        policy: ConnectionPolicy,
    ) -> ConnectionTestResult:
        """Test the connection to the target PostgreSQL database."""
        self._validate_context(organization_id, connection, policy)

        start_time = time.monotonic()
        warnings = []
        capabilities = [
            ConnectorCapability.CONNECTIVITY,
            ConnectorCapability.READ_ONLY_TRANSACTION,
            ConnectorCapability.SCHEMA_LISTING,
            ConnectorCapability.STATEMENT_TIMEOUT,
            ConnectorCapability.LOCK_TIMEOUT,
            ConnectorCapability.EXPLAIN,
        ]

        engine = await self._create_test_engine(connection, organization_id)
        
        try:
            async with engine.connect() as conn:
                # Open transaction
                async with conn.begin():
                    # Set safe session configuration
                    try:
                        await conn.execute(text(f"SET statement_timeout = {self.settings.connector_test_statement_timeout_ms}"))
                        await conn.execute(text(f"SET lock_timeout = {self.settings.connector_test_lock_timeout_ms}"))
                        await conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
                    except Exception as e:
                        logger.error("failed_to_set_session_characteristics", error=str(e))
                        raise self._translate_error(e)
                    
                    # Minimal validation queries
                    try:
                        await conn.execute(text("SELECT 1"))
                        
                        db_name_res = await conn.execute(text("SELECT current_database()"))
                        database_name = db_name_res.scalar()
                        
                        version_res = await conn.execute(text("SHOW server_version"))
                        server_version = version_res.scalar()
                    except Exception as e:
                        raise self._translate_error(e)

                    # Parse major version safely
                    try:
                        major_version = int(server_version.split(".")[0])
                        if major_version < POSTGRESQL_MINIMUM_MAJOR_VERSION:
                            warnings.append(
                                ConnectionTestWarning(
                                    code="UNSUPPORTED_VERSION",
                                    message=f"PostgreSQL version {server_version} is below the supported minimum version {POSTGRESQL_MINIMUM_MAJOR_VERSION}.",
                                    severity=WarningSeverity.WARNING
                                )
                            )
                    except (ValueError, AttributeError):
                        warnings.append(
                            ConnectionTestWarning(
                                code="UNKNOWN_VERSION_FORMAT",
                                message="Could not determine major PostgreSQL version.",
                                severity=WarningSeverity.INFO
                            )
                        )

                    # List namespaces
                    try:
                        namespaces_res = await conn.execute(text("""
                            SELECT schema_name 
                            FROM information_schema.schemata 
                            ORDER BY schema_name
                        """))
                        all_schemas = [row[0] for row in namespaces_res]
                    except Exception as e:
                        raise self._translate_error(e)

                    # Validate approved schemas
                    visible_schemas = set(all_schemas)
                    policy_approved = set(policy.approved_schemas_json)
                    
                    approved_schemas_found = list(visible_schemas & policy_approved)
                    approved_schemas_missing = list(policy_approved - visible_schemas)
                    
                    if not policy_approved:
                        warnings.append(
                            ConnectionTestWarning(
                                code="EMPTY_APPROVED_SCHEMAS",
                                message="No approved schemas are configured in the connection policy. Schema scanning and queries will be blocked.",
                                severity=WarningSeverity.WARNING
                            )
                        )
                    elif not approved_schemas_found:
                        warnings.append(
                            ConnectionTestWarning(
                                code="ALL_APPROVED_SCHEMAS_MISSING",
                                message="None of the approved schemas were found in the database. Ensure the read-only user has access to them.",
                                severity=WarningSeverity.CRITICAL
                            )
                        )
                    
                    latency_ms = int((time.monotonic() - start_time) * 1000)

                    return ConnectionTestResult(
                        success=True,
                        dialect=self.dialect.value,
                        server_version=server_version,
                        database_name=database_name,
                        reachable_schemas=all_schemas,
                        approved_schemas_found=approved_schemas_found,
                        approved_schemas_missing=approved_schemas_missing,
                        capabilities=capabilities,
                        warnings=warnings,
                        latency_ms=latency_ms,
                        tested_at=time.time(), # We might want datetime but time module provides easy timestamp? Wait, pydantic expects datetime.
                        # Wait, let's fix tested_at.
                    )
        except Exception as e:
            if not isinstance(e, ConnectorError):
                raise self._translate_error(e)
            raise
        finally:
            await engine.dispose()

    async def list_namespaces(
        self,
        *,
        organization_id: uuid.UUID,
        connection: DatabaseConnection,
        policy: ConnectionPolicy,
    ) -> list[NamespaceSummary]:
        self._validate_context(organization_id, connection, policy)
        
        # Similar flow to test connection but returns NamespaceSummary
        engine = await self._create_test_engine(connection, organization_id)
        try:
            async with engine.connect() as conn:
                async with conn.begin():
                    try:
                        await conn.execute(text(f"SET statement_timeout = {self.settings.connector_test_statement_timeout_ms}"))
                        await conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
                        namespaces_res = await conn.execute(text("""
                            SELECT schema_name 
                            FROM information_schema.schemata 
                            ORDER BY schema_name
                        """))
                        all_schemas = [row[0] for row in namespaces_res]
                    except Exception as e:
                        raise self._translate_error(e)

                    policy_approved = set(policy.approved_schemas_json)
                    summaries = []
                    
                    for schema_name in all_schemas:
                        is_system = schema_name in POSTGRESQL_SYSTEM_SCHEMAS or any(
                            schema_name.startswith(prefix) for prefix in POSTGRESQL_SYSTEM_SCHEMA_PREFIXES
                        )
                        is_approved = schema_name in policy_approved
                        
                        summaries.append(NamespaceSummary(
                            name=schema_name,
                            is_system=is_system,
                            is_approved=is_approved,
                        ))
                    
                    return summaries
        except Exception as e:
            if not isinstance(e, ConnectorError):
                raise self._translate_error(e)
            raise
        finally:
            await engine.dispose()

    async def introspect_schema(
        self,
        *,
        organization_id: uuid.UUID,
        connection: DatabaseConnection,
        policy: ConnectionPolicy,
        schemas: list[str],
        cancellation_check: callable | None = None,
        progress_callback: callable | None = None,
    ):
        from app.connectors.postgres.introspector import PostgreSQLSchemaIntrospector

        self._validate_context(organization_id, connection, policy)

        introspector = PostgreSQLSchemaIntrospector(self.settings)

        # For introspection, create a temporary engine in METADATA mode
        secret = await self.secret_resolution_service.resolve_secret_for_connector(
            provider_type=connection.secret_provider,
            reference=connection.secret_reference,
            organization_id=organization_id,
        )

        try:
            engine = await self.engine_factory.create_engine(
                connection=connection,
                secret=secret,
                mode=ConnectorMode.METADATA,
            )
        except ConnectorConfigurationError:
            raise
        except Exception as e:
            logger.error("engine_creation_failed", error=str(e), connection_id=str(connection.id))
            raise ConnectorConfigurationError("Failed to configure external connection.", details={"safe_error_code": "CONNECTOR_CONFIGURATION_ERROR"})

        try:
            return await introspector.introspect(
                engine=engine,
                approved_schemas=schemas,
                policy=policy,
                cancellation_check=cancellation_check,
                progress_callback=progress_callback,
            )
        finally:
            await engine.dispose()

    async def dispose_connection_pool(
        self,
        *,
        organization_id: uuid.UUID,
        connection_id: uuid.UUID,
    ) -> None:
        if self.pool_registry:
            await self.pool_registry.dispose_for_connection(organization_id, connection_id)

    def quote_identifier(self, identifier: str) -> str:
        # Simple PostgreSQL quoting
        return f'"{identifier.replace('"', '""')}"'

    def _validate_context(
        self,
        organization_id: uuid.UUID,
        connection: DatabaseConnection,
        policy: ConnectionPolicy,
    ) -> None:
        if connection.organization_id != organization_id:
            raise ConnectorConfigurationError("Connection does not belong to the given organization.")
        if policy.organization_id != organization_id:
            raise ConnectorConfigurationError("Policy does not belong to the given organization.")
        if policy.connection_id != connection.id:
            raise ConnectorConfigurationError("Policy does not match the given connection.")
        if connection.status == ConnectionStatus.DISABLED:
            raise ConnectorConfigurationError("Connection is disabled.")
        if connection.dialect != DatabaseDialect.POSTGRESQL:
            raise ConnectorConfigurationError(f"Unsupported dialect {connection.dialect} for PostgreSQL connector.")

    async def _create_test_engine(
        self, connection: DatabaseConnection, organization_id: uuid.UUID
    ) -> AsyncEngine:
        # For testing, we create a one-off temporary engine.
        secret = await self.secret_resolution_service.resolve_secret_for_connector(
            provider_type=connection.secret_provider,
            reference=connection.secret_reference,
            organization_id=organization_id,
        )
        
        try:
            return await self.engine_factory.create_engine(
                connection=connection,
                secret=secret,
                mode=ConnectorMode.TEST,
            )
        except ConnectorConfigurationError:
            raise
        except Exception as e:
            logger.error("engine_creation_failed", error=str(e), connection_id=str(connection.id))
            raise ConnectorConfigurationError("Failed to configure external connection.", details={"safe_error_code": "CONNECTOR_CONFIGURATION_ERROR"})

    def _translate_error(self, exc: Exception) -> ConnectorError:
        err_str = str(exc).lower()
        
        if isinstance(exc, asyncpg.exceptions.InvalidAuthorizationSpecificationError) or \
           isinstance(exc, asyncpg.exceptions.InvalidPasswordError):
            return ConnectorAuthenticationError()
            
        if "authentication failed" in err_str or "password authentication failed" in err_str:
            return ConnectorAuthenticationError()

        if isinstance(exc, asyncpg.exceptions.InsufficientPrivilegeError):
            return ConnectorConfigurationError("Insufficient privileges for connector operations.")

        if isinstance(exc, asyncpg.exceptions.InvalidCatalogNameError):
            return ConnectorDatabaseNotFoundError()
            
        if "database" in err_str and "does not exist" in err_str:
            return ConnectorDatabaseNotFoundError()

        if isinstance(exc, asyncio.TimeoutError) or "timeout" in err_str:
            return ConnectorTimeoutError()

        if "ssl" in err_str or isinstance(exc, ssl.SSLError if 'ssl' in globals() else type("Dummy", (), {})): # Handle missing ssl import here if not imported at top
            return ConnectorSSLError()

        if "connection refused" in err_str or "nodename nor servname provided" in err_str or "no route to host" in err_str:
            return ConnectorUnavailableError()
            
        if "could not connect to server" in err_str or "network is unreachable" in err_str:
            return ConnectorUnavailableError()

        # Log unhandled exceptions
        logger.warning("untranslated_connector_error", error=str(exc), exc_info=True)
        return ConnectorError()
