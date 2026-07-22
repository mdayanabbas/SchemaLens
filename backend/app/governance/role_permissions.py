from app.models.enums import OrganizationRole
from app.governance.permissions import Permission


ROLE_PERMISSIONS: dict[OrganizationRole, frozenset[Permission]] = {
    OrganizationRole.ORGANIZATION_ADMIN: frozenset({
        Permission.ORGANIZATION_READ,
        Permission.ORGANIZATION_MANAGE,
        Permission.MEMBERS_READ,
        Permission.MEMBERS_MANAGE,
        Permission.CONNECTIONS_READ,
        Permission.CONNECTIONS_MANAGE,
        Permission.CONNECTIONS_TEST,
        Permission.SCHEMAS_READ,
        Permission.SCHEMAS_SCAN,
        Permission.BUSINESS_METADATA_READ,
        Permission.BUSINESS_METADATA_MANAGE,
        Permission.BUSINESS_METADATA_APPROVE,
        Permission.QUERIES_CREATE,
        Permission.QUERIES_READ,
        Permission.QUERIES_REVIEW,
        Permission.QUERIES_EXECUTE,
        Permission.QUERIES_CANCEL,
        Permission.QUERIES_EXPORT,
        Permission.POLICIES_READ,
        Permission.POLICIES_MANAGE,
        Permission.AUDIT_READ,
    }),
    OrganizationRole.DATABASE_ADMIN: frozenset({
        Permission.ORGANIZATION_READ,
        Permission.MEMBERS_READ,
        Permission.CONNECTIONS_READ,
        Permission.CONNECTIONS_MANAGE,
        Permission.CONNECTIONS_TEST,
        Permission.SCHEMAS_READ,
        Permission.SCHEMAS_SCAN,
        Permission.BUSINESS_METADATA_READ,
        Permission.BUSINESS_METADATA_MANAGE,
        Permission.BUSINESS_METADATA_APPROVE,
        Permission.QUERIES_CREATE,
        Permission.QUERIES_READ,
        Permission.QUERIES_REVIEW,
        Permission.QUERIES_EXECUTE,
        Permission.QUERIES_CANCEL,
        Permission.QUERIES_EXPORT,
        Permission.POLICIES_READ,
        Permission.POLICIES_MANAGE,
        Permission.AUDIT_READ,
    }),
    OrganizationRole.REVIEWER: frozenset({
        Permission.ORGANIZATION_READ,
        Permission.MEMBERS_READ,
        Permission.CONNECTIONS_READ,
        Permission.SCHEMAS_READ,
        Permission.BUSINESS_METADATA_READ,
        Permission.BUSINESS_METADATA_APPROVE,
        Permission.QUERIES_CREATE,
        Permission.QUERIES_READ,
        Permission.QUERIES_REVIEW,
        Permission.QUERIES_EXECUTE,
        Permission.QUERIES_CANCEL,
        Permission.QUERIES_EXPORT,
        Permission.POLICIES_READ,
        Permission.AUDIT_READ,
    }),
    OrganizationRole.ANALYST: frozenset({
        Permission.ORGANIZATION_READ,
        Permission.CONNECTIONS_READ,
        Permission.SCHEMAS_READ,
        Permission.BUSINESS_METADATA_READ,
        Permission.QUERIES_CREATE,
        Permission.QUERIES_READ,
        Permission.QUERIES_EXECUTE,
        Permission.QUERIES_CANCEL,
        Permission.QUERIES_EXPORT,
    }),
    OrganizationRole.VIEWER: frozenset({
        Permission.ORGANIZATION_READ,
        Permission.CONNECTIONS_READ,
        Permission.SCHEMAS_READ,
        Permission.BUSINESS_METADATA_READ,
        Permission.QUERIES_READ,
    }),
}


def permissions_for_role(role: OrganizationRole) -> frozenset[Permission]:
    """Return the frozen set of permissions for a given role, failing closed if unknown."""
    return ROLE_PERMISSIONS.get(role, frozenset())


def role_has_permission(role: OrganizationRole, permission: Permission) -> bool:
    """Check if a role has a specific permission."""
    return permission in permissions_for_role(role)
