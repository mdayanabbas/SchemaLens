import ssl

from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from app.connectors.exceptions import ConnectorConfigurationError
from app.connectors.pool_key import ConnectorMode
from app.connectors.postgres.constants import APPLICATION_NAME_MAX_LENGTH
from app.core.config import Settings
from app.models.connection_enums import SSLMode
from app.models.database_connection import DatabaseConnection
from app.secrets.value import SecretValue


class PostgreSQLEngineFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def create_engine(
        self,
        *,
        connection: DatabaseConnection,
        secret: SecretValue,
        mode: ConnectorMode,
    ) -> AsyncEngine:
        """Create a SQLAlchemy AsyncEngine for an external PostgreSQL database."""
        
        # Profile is authoritative for host, port, db. Ignore secret overrides.
        url = URL.create(
            drivername="postgresql+asyncpg",
            username=secret.username,
            password=secret.get_password(),
            host=connection.host,
            port=connection.port,
            database=connection.database_name,
        )

        connect_args: dict[str, any] = {
            "timeout": self.settings.connector_connect_timeout_seconds,
            "server_settings": {
                "application_name": self._build_application_name(connection, mode),
            },
        }

        # Configure SSL
        ssl_context = self._build_ssl_context(connection.ssl_mode, secret)
        if ssl_context:
            connect_args["ssl"] = ssl_context
        elif connection.ssl_mode != SSLMode.DISABLE:
            # asyncpg accepts ssl strings too, but we translate to standard meaning
            if connection.ssl_mode == SSLMode.REQUIRE:
                connect_args["ssl"] = "require"
            elif connection.ssl_mode == SSLMode.PREFER:
                connect_args["ssl"] = "prefer"
            elif connection.ssl_mode == SSLMode.ALLOW:
                connect_args["ssl"] = "allow"

        engine_args = {
            "echo": False,
            "hide_parameters": True,
            "connect_args": connect_args,
        }

        # Use NullPool for tests/metadata if it is safer, or bounded pool
        if mode == ConnectorMode.TEST:
            engine_args["poolclass"] = NullPool
        else:
            engine_args.update({
                "pool_size": self.settings.connector_pool_size,
                "max_overflow": self.settings.connector_max_overflow,
                "pool_timeout": self.settings.connector_pool_timeout_seconds,
                "pool_recycle": self.settings.connector_pool_recycle_seconds,
                "pool_pre_ping": True,
            })

        return create_async_engine(url, **engine_args)

    def _build_application_name(self, connection: DatabaseConnection, mode: ConnectorMode) -> str:
        prefix = self.settings.connector_application_name_prefix
        short_id = str(connection.id).split("-")[0]
        app_name = f"{prefix}:{mode.value}:{short_id}"
        return app_name[:APPLICATION_NAME_MAX_LENGTH]

    def _build_ssl_context(self, ssl_mode: SSLMode, secret: SecretValue) -> ssl.SSLContext | None:
        if ssl_mode == SSLMode.DISABLE:
            return None

        # Custom cert materials
        ca_cert = secret.get_ssl_ca()
        client_cert = secret.get_ssl_cert()
        client_key = secret.get_ssl_key()

        if ca_cert or client_cert or client_key:
            # This requires writing temporary files safely, as ssl.SSLContext methods load from files.
            # However, for Brick 10, the instruction is:
            # "Support SSL certificate material only if it can be handled securely without writing plaintext temporary files unnecessarily."
            # Given standard python ssl context requires files, we raise ConfigurationError if they provide raw strings
            # and expect us to load them in-memory, unless we implement memory-loading via PyOpenSSL or a custom extension.
            # In asyncpg, you can pass ssl context.
            # We will reject raw cert strings in this brick unless they are file paths (which they usually aren't from secrets manager).
            raise ConnectorConfigurationError(
                "In-memory SSL certificate material is not supported in this version. "
                "Use system certificates with require/verify-full."
            )

        if ssl_mode == SSLMode.VERIFY_CA or ssl_mode == SSLMode.VERIFY_FULL:
            context = ssl.create_default_context()
            if ssl_mode == SSLMode.VERIFY_CA:
                context.check_hostname = False
            return context
            
        return None
