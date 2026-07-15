"""
Database model registry.
Import all SQLAlchemy declarative models here so Alembic can discover them.
"""

from app.db.base import Base

from app.models import AuthenticationEvent, Organization, OrganizationMembership, RefreshToken, User

target_metadata = Base.metadata
