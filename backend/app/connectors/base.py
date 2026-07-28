import uuid
from abc import ABC, abstractmethod

from app.connectors.types import ConnectionTestResult, NamespaceSummary
from app.models.connection_enums import DatabaseDialect
from app.models.connection_policy import ConnectionPolicy
from app.models.database_connection import DatabaseConnection


class DatabaseConnector(ABC):
    dialect: DatabaseDialect

    @abstractmethod
    async def test_connection(
        self,
        *,
        organization_id: uuid.UUID,
        connection: DatabaseConnection,
        policy: ConnectionPolicy,
    ) -> ConnectionTestResult:
        """
        Test the connection to the target database and return a sanitized result.
        """

    @abstractmethod
    async def list_namespaces(
        self,
        *,
        organization_id: uuid.UUID,
        connection: DatabaseConnection,
        policy: ConnectionPolicy,
    ) -> list[NamespaceSummary]:
        """
        List schemas/namespaces available in the target database.
        """

    @abstractmethod
    async def dispose_connection_pool(
        self,
        *,
        organization_id: uuid.UUID,
        connection_id: uuid.UUID,
    ) -> None:
        """
        Dispose all connection pools associated with this connection.
        """

    @abstractmethod
    def quote_identifier(self, identifier: str) -> str:
        """
        Properly quote a SQL identifier for the dialect.
        """

    @abstractmethod
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
        """
        Extract schema metadata from the target database for the specified namespaces.
        """

    # Future methods to be implemented:
    # explain_query
    # create_read_only_session
    # execute_query_stream
    # cancel_query
