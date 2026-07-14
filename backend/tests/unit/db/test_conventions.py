from app.db.base import Base
from app.db.conventions import naming_convention


def test_naming_conventions_exist() -> None:
    assert "ix" in naming_convention
    assert "uq" in naming_convention
    assert "ck" in naming_convention
    assert "fk" in naming_convention
    assert "pk" in naming_convention


def test_base_metadata_uses_conventions() -> None:
    assert Base.metadata.naming_convention == naming_convention
