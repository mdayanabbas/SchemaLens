from app.models import Organization, OrganizationMembership, User


def test_organization_metadata():
    assert Organization.__tablename__ == "organizations"
    assert "name" in Organization.__table__.columns
    assert "slug" in Organization.__table__.columns
    assert "status" in Organization.__table__.columns
    assert Organization.slug.unique is True


def test_user_metadata():
    assert User.__tablename__ == "users"
    assert "email" in User.__table__.columns
    assert "password_hash" in User.__table__.columns
    assert User.email.unique is True


def test_membership_metadata():
    assert OrganizationMembership.__tablename__ == "organization_memberships"
    assert "organization_id" in OrganizationMembership.__table__.columns
    assert "user_id" in OrganizationMembership.__table__.columns
    
    unique_constraints = [c for c in OrganizationMembership.__table__.constraints if getattr(c, "name", None) == "uq_organization_memberships_org_user"]
    assert len(unique_constraints) == 1
