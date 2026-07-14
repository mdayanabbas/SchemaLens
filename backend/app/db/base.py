from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

from app.db.conventions import naming_convention

metadata = MetaData(naming_convention=naming_convention)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""

    metadata = metadata
