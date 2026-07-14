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
