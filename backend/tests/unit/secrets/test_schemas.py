import pytest
from pydantic import SecretStr, ValidationError

from app.secrets.schemas import DatabaseCredentialPayload


def test_payload_accepts_valid_data():
    payload = DatabaseCredentialPayload(
        username="db_user",
        password=SecretStr("supersecret"),
        database="prod_db",
        port=5432
    )
    assert payload.username == "db_user"
    assert payload.password.get_secret_value() == "supersecret"
    assert payload.database == "prod_db"
    assert payload.port == 5432


def test_payload_rejects_empty_password():
    with pytest.raises(ValidationError) as exc:
        DatabaseCredentialPayload(
            username="db_user",
            password=SecretStr("")
        )
    assert "Password must not be empty" in str(exc.value)


def test_payload_rejects_control_characters_in_username():
    with pytest.raises(ValidationError) as exc:
        DatabaseCredentialPayload(
            username="db_user\n",
            password=SecretStr("secret")
        )
    assert "Must not contain control characters" in str(exc.value)


def test_payload_rejects_connection_urls():
    with pytest.raises(ValidationError) as exc:
        DatabaseCredentialPayload(
            username="db_user",
            password=SecretStr("secret"),
            database="postgresql://localhost:5432/db"
        )
    assert "Must not be a connection URL" in str(exc.value)


def test_payload_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        DatabaseCredentialPayload(
            username="db_user",
            password=SecretStr("secret"),
            unknown_field="injected"
        )
