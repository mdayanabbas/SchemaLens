from app.connectors.base import DatabaseConnector
from app.connectors.exceptions import ConnectorUnsupportedError
from app.models.connection_enums import DatabaseDialect


class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: dict[DatabaseDialect, DatabaseConnector] = {}

    def register(self, connector: DatabaseConnector) -> None:
        if connector.dialect in self._connectors:
            raise ValueError(f"Connector for dialect {connector.dialect} is already registered.")
        self._connectors[connector.dialect] = connector

    def get(self, dialect: DatabaseDialect) -> DatabaseConnector:
        connector = self._connectors.get(dialect)
        if not connector:
            raise ConnectorUnsupportedError(f"Connector for dialect {dialect} is not supported.")
        return connector

    def supports(self, dialect: DatabaseDialect) -> bool:
        return dialect in self._connectors
