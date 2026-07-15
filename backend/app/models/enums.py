from enum import StrEnum


class OrganizationStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class UserStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


class MembershipStatus(StrEnum):
    ACTIVE = "active"
    INVITED = "invited"
    DISABLED = "disabled"


class OrganizationRole(StrEnum):
    ORGANIZATION_ADMIN = "organization_admin"
    DATABASE_ADMIN = "database_admin"
    REVIEWER = "reviewer"
    ANALYST = "analyst"
    VIEWER = "viewer"


class RefreshTokenStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"
    ROTATED = "rotated"
    EXPIRED = "expired"
    COMPROMISED = "compromised"


class AuthenticationEventType(StrEnum):
    LOGIN_SUCCEEDED = "login_succeeded"
    LOGIN_FAILED = "login_failed"
    TOKEN_REFRESHED = "token_refreshed"
    LOGOUT_SUCCEEDED = "logout_succeeded"
    REFRESH_TOKEN_REUSE_DETECTED = "refresh_token_reuse_detected"
    PASSWORD_CHANGED = "password_changed"
