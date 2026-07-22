from enum import StrEnum


class Permission(StrEnum):
    ORGANIZATION_READ = "organization.read"
    ORGANIZATION_MANAGE = "organization.manage"

    MEMBERS_READ = "members.read"
    MEMBERS_MANAGE = "members.manage"

    CONNECTIONS_READ = "connections.read"
    CONNECTIONS_MANAGE = "connections.manage"
    CONNECTIONS_TEST = "connections.test"

    SCHEMAS_READ = "schemas.read"
    SCHEMAS_SCAN = "schemas.scan"

    BUSINESS_METADATA_READ = "business_metadata.read"
    BUSINESS_METADATA_MANAGE = "business_metadata.manage"
    BUSINESS_METADATA_APPROVE = "business_metadata.approve"

    QUERIES_CREATE = "queries.create"
    QUERIES_READ = "queries.read"
    QUERIES_REVIEW = "queries.review"
    QUERIES_EXECUTE = "queries.execute"
    QUERIES_CANCEL = "queries.cancel"
    QUERIES_EXPORT = "queries.export"

    POLICIES_READ = "policies.read"
    POLICIES_MANAGE = "policies.manage"

    AUDIT_READ = "audit.read"
