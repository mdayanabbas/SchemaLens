from app.models.enums import MembershipStatus, OrganizationRole, OrganizationStatus, UserStatus


def test_enums_values():
    assert OrganizationStatus.ACTIVE.value == "active"
    assert OrganizationStatus.SUSPENDED.value == "suspended"

    assert UserStatus.ACTIVE.value == "active"
    assert UserStatus.DISABLED.value == "disabled"

    assert MembershipStatus.ACTIVE.value == "active"
    assert MembershipStatus.INVITED.value == "invited"
    assert MembershipStatus.DISABLED.value == "disabled"

    assert OrganizationRole.ORGANIZATION_ADMIN.value == "organization_admin"
    assert OrganizationRole.VIEWER.value == "viewer"
