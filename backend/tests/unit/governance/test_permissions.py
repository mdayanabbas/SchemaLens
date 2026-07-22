import pytest

from app.governance.permissions import Permission
from app.governance.role_permissions import permissions_for_role, role_has_permission
from app.models.enums import OrganizationRole


def test_every_role_has_explicit_mapping():
    for role in OrganizationRole:
        # permissions_for_role defaults to empty set if not mapped
        # but we want to ensure it's explicitly in the map and not just empty
        # actually permissions_for_role returns an empty frozenset.
        # We can check if it returns a frozen set.
        perms = permissions_for_role(role)
        assert isinstance(perms, frozenset)
        assert len(perms) > 0  # Every role in our design has at least one permission


def test_organization_admin_permissions():
    perms = permissions_for_role(OrganizationRole.ORGANIZATION_ADMIN)
    assert Permission.ORGANIZATION_MANAGE in perms
    assert Permission.MEMBERS_MANAGE in perms
    assert Permission.CONNECTIONS_TEST in perms
    assert Permission.QUERIES_EXECUTE in perms


def test_database_admin_permissions():
    perms = permissions_for_role(OrganizationRole.DATABASE_ADMIN)
    assert Permission.CONNECTIONS_MANAGE in perms
    assert Permission.SCHEMAS_SCAN in perms
    assert Permission.ORGANIZATION_MANAGE not in perms
    assert Permission.MEMBERS_MANAGE not in perms


def test_reviewer_permissions():
    perms = permissions_for_role(OrganizationRole.REVIEWER)
    assert Permission.QUERIES_REVIEW in perms
    assert Permission.BUSINESS_METADATA_APPROVE in perms
    assert Permission.ORGANIZATION_MANAGE not in perms
    assert Permission.MEMBERS_MANAGE not in perms
    assert Permission.CONNECTIONS_MANAGE not in perms
    assert Permission.SCHEMAS_SCAN not in perms
    assert Permission.POLICIES_MANAGE not in perms


def test_analyst_permissions():
    perms = permissions_for_role(OrganizationRole.ANALYST)
    assert Permission.QUERIES_CREATE in perms
    assert Permission.QUERIES_EXECUTE in perms
    assert Permission.QUERIES_REVIEW not in perms
    assert Permission.ORGANIZATION_MANAGE not in perms
    assert Permission.AUDIT_READ not in perms
    assert Permission.POLICIES_MANAGE not in perms


def test_viewer_permissions():
    perms = permissions_for_role(OrganizationRole.VIEWER)
    assert Permission.ORGANIZATION_READ in perms
    assert Permission.QUERIES_READ in perms
    assert Permission.QUERIES_CREATE not in perms
    assert Permission.QUERIES_EXECUTE not in perms
    assert Permission.QUERIES_REVIEW not in perms


def test_unknown_role_fails_closed():
    # If a non-existent role string is passed
    perms = permissions_for_role("some_unknown_role")
    assert isinstance(perms, frozenset)
    assert len(perms) == 0
    assert role_has_permission("some_unknown_role", Permission.ORGANIZATION_READ) is False


def test_returned_collections_are_immutable():
    perms = permissions_for_role(OrganizationRole.VIEWER)
    with pytest.raises(AttributeError):
        perms.add(Permission.ORGANIZATION_MANAGE)
