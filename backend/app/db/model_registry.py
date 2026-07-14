"""
Database model registry.
Import all SQLAlchemy declarative models here so Alembic can discover them.
"""

from app.db.base import Base

# TODO: Import domain models here when they are created
# from app.models.organization import Organization
# ...

target_metadata = Base.metadata
