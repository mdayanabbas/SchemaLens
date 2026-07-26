import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.connectors.types import ConnectionTestWarning, ConnectorCapability


class ConnectionTestResponse(BaseModel):
    success: bool
    connection_id: uuid.UUID
    dialect: str
    server_version: Optional[str] = None
    database_name: Optional[str] = None
    approved_schemas_found: list[str] = []
    approved_schemas_missing: list[str] = []
    capabilities: list[ConnectorCapability] = []
    warnings: list[ConnectionTestWarning] = []
    latency_ms: Optional[int] = None
    tested_at: Optional[datetime] = None
    safe_error_code: Optional[str] = None
