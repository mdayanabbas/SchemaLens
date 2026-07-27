from enum import StrEnum


class SchemaScanStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    FAILED = "failed"


class SchemaScanTrigger(StrEnum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    SYSTEM = "system"


class SchemaScanFailureStage(StrEnum):
    AUTHORIZATION = "authorization"
    DISPATCH = "dispatch"
    WORKER_START = "worker_start"
    CONNECTION_VALIDATION = "connection_validation"
    POLICY_VALIDATION = "policy_validation"
    INTROSPECTION = "introspection"
    PERSISTENCE = "persistence"
    FINALIZATION = "finalization"
    CANCELLATION = "cancellation"
    STALE_RECOVERY = "stale_recovery"


class SchemaScanWarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
