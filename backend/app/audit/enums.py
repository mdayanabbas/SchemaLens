from enum import StrEnum

class AuditOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"

class AuditActorType(StrEnum):
    USER = "user"
    PLATFORM_ADMIN = "platform_admin"
    SYSTEM = "system"
    WORKER = "worker"
    ANONYMOUS = "anonymous"

class AuditResourceType(StrEnum):
    AUTHENTICATION = "authentication"
    ORGANIZATION = "organization"
    MEMBERSHIP = "membership"
    USER = "user"
    AUTHORIZATION = "authorization"
    AUDIT_EVENT = "audit_event"
    # Future bricks:
    DATABASE_CONNECTION = "database_connection"
    CONNECTION_POLICY = "connection_policy"
    SCHEMA_SCAN = "schema_scan"
    SCHEMA_SNAPSHOT = "schema_snapshot"
    BUSINESS_METADATA = "business_metadata"
    QUERY_REQUEST = "query_request"
    QUERY_PLAN = "query_plan"
    GENERATED_SQL = "generated_sql"
    APPROVAL = "approval"
    QUERY_EXECUTION = "query_execution"
    QUERY_RESULT = "query_result"
    EXPORT = "export"

class AuditAction(StrEnum):
    AUTH_LOGIN_SUCCEEDED = "auth.login_succeeded"
    AUTH_LOGIN_FAILED = "auth.login_failed"
    AUTH_TOKEN_REFRESHED = "auth.token_refreshed"
    AUTH_LOGOUT = "auth.logout"
    AUTH_REFRESH_TOKEN_REUSE_DETECTED = "auth.refresh_token_reuse_detected"
    AUTH_PASSWORD_CHANGED = "auth.password_changed"
    AUTH_SESSIONS_REVOKED = "auth.sessions_revoked"

    ORGANIZATION_CREATED = "organization.created"
    ORGANIZATION_UPDATED = "organization.updated"
    ORGANIZATION_SUSPENDED = "organization.suspended"
    ORGANIZATION_ACCESSED = "organization.accessed"

    MEMBERSHIP_CREATED = "membership.created"
    MEMBERSHIP_UPDATED = "membership.updated"
    MEMBERSHIP_DISABLED = "membership.disabled"
    MEMBERSHIP_ROLE_CHANGED = "membership.role_changed"

    AUTHORIZATION_ALLOWED = "authorization.allowed"
    AUTHORIZATION_DENIED = "authorization.denied"
    AUTHORIZATION_PLATFORM_ADMIN_BYPASS = "authorization.platform_admin_bypass"

    AUDIT_EVENTS_VIEWED = "audit.events_viewed"

    CONNECTION_CREATED = "connection.created"
    CONNECTION_UPDATED = "connection.updated"
    CONNECTION_DISABLED = "connection.disabled"
    CONNECTION_POLICY_UPDATED = "connection.policy_updated"

class AuditEventSource(StrEnum):
    API = "api"
    BOOTSTRAP = "bootstrap"
    WORKER = "worker"
    SYSTEM = "system"
