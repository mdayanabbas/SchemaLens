from datetime import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel


class ConnectorCapability(StrEnum):
    CONNECTIVITY = "connectivity"
    READ_ONLY_TRANSACTION = "read_only_transaction"
    SCHEMA_LISTING = "schema_listing"
    STATEMENT_TIMEOUT = "statement_timeout"
    LOCK_TIMEOUT = "lock_timeout"
    QUERY_CANCELLATION = "query_cancellation"
    EXPLAIN = "explain"


class WarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ConnectionTestWarning(BaseModel):
    code: str
    message: str
    severity: WarningSeverity


class ConnectionTestResult(BaseModel):
    success: bool
    dialect: str
    server_version: str
    database_name: str
    current_user_name: Optional[str] = None
    reachable_schemas: list[str]
    approved_schemas_found: list[str]
    approved_schemas_missing: list[str]
    capabilities: list[ConnectorCapability]
    warnings: list[ConnectionTestWarning]
    latency_ms: int
    tested_at: datetime


class ConnectorHealthResult(BaseModel):
    status: str
    dialect: str
    latency_ms: Optional[int] = None
    safe_error_code: Optional[str] = None


class NamespaceSummary(BaseModel):
    name: str
    is_system: bool
    is_approved: bool


class ReadOnlySessionConfiguration(BaseModel):
    statement_timeout_ms: int
    lock_timeout_ms: int
    application_name: str
    transaction_read_only: bool


class ConnectorErrorContext(BaseModel):
    operation: str
    safe_code: str
    retryable: bool
    provider: str
    dialect: str
