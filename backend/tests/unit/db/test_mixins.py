from sqlalchemy import Column, String
from sqlalchemy.orm import declarative_base

from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin

TestBase = declarative_base()


class DummyModel(TestBase, UUIDPrimaryKeyMixin, TimestampMixin):  # type: ignore
    __tablename__ = "dummy_model"
    name = Column(String)


def test_uuid_primary_key_configuration() -> None:
    pk_col = DummyModel.__table__.columns.get("id")
    assert pk_col is not None
    assert pk_col.primary_key is True
    assert callable(pk_col.default.arg)


def test_timestamp_mixin_configuration() -> None:
    created_at = DummyModel.__table__.columns.get("created_at")
    updated_at = DummyModel.__table__.columns.get("updated_at")
    
    assert created_at is not None
    assert updated_at is not None
    
    assert created_at.type.timezone is True
    assert updated_at.type.timezone is True
    
    assert created_at.server_default is not None
    assert updated_at.server_default is not None
    assert updated_at.onupdate is not None
