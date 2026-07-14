import pytest
from pydantic import ValidationError
from app.schemas.organization import OrganizationCreate
from app.schemas.user import UserCreate


def test_organization_slug_validation():
    org = OrganizationCreate(name="Test Org", slug="test-org-123")
    assert org.slug == "test-org-123"
    
    org2 = OrganizationCreate(name="Test", slug="  TEST-org  ")
    assert org2.slug == "test-org"
    
    with pytest.raises(ValidationError):
        OrganizationCreate(name="T", slug="-test")
    with pytest.raises(ValidationError):
        OrganizationCreate(name="T", slug="test-")
    with pytest.raises(ValidationError):
        OrganizationCreate(name="T", slug="test--org")
    with pytest.raises(ValidationError):
        OrganizationCreate(name="T", slug="test org")


def test_user_email_normalization():
    user = UserCreate(email=" TEST@example.com ", display_name="Test User")
    assert user.email == "test@example.com"
