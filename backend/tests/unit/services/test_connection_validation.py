import pytest

from app.core.exceptions import ValidationError
from app.models.connection_enums import SSLMode
from app.services.connection_validation import (
    normalize_and_deduplicate_schemas,
    redact_secret_reference,
    validate_connection_name,
    validate_database_name,
    validate_host,
    validate_production_ssl,
    validate_schema_lists,
    validate_secret_reference,
)


def test_validate_connection_name():
    assert validate_connection_name(" My Connection ") == "My Connection"
    with pytest.raises(ValidationError):
        validate_connection_name("")
        
def test_validate_host():
    assert validate_host("db.example.com") == "db.example.com"
    assert validate_host("192.168.1.100") == "192.168.1.100"
    
    with pytest.raises(ValidationError):
        validate_host("postgresql://db.example.com")
        
    with pytest.raises(ValidationError):
        validate_host("user:pass@db.example.com")
        
    with pytest.raises(ValidationError):
        validate_host("db.example.com/mydb")

def test_validate_database_name():
    assert validate_database_name("my_db") == "my_db"
    
    with pytest.raises(ValidationError):
        validate_database_name("my/db")
        
    with pytest.raises(ValidationError):
        validate_database_name("my?db")

def test_validate_secret_reference():
    assert validate_secret_reference("arn:aws:secretsmanager:...") == "arn:aws:secretsmanager:..."
    
    with pytest.raises(ValidationError):
        validate_secret_reference("postgresql://user:pass@host/db")

def test_normalize_and_deduplicate_schemas():
    schemas = [" public ", "public", "app"]
    assert normalize_and_deduplicate_schemas(schemas) == ["public", "app"]
    
def test_validate_schema_lists():
    approved, blocked = validate_schema_lists(["app"], ["pg_catalog"])
    assert approved == ["app"]
    assert blocked == ["pg_catalog"]
    
    with pytest.raises(ValidationError):
        validate_schema_lists(["app", "pg_catalog"], ["pg_catalog"])

def test_validate_production_ssl():
    validate_production_ssl("production", SSLMode.REQUIRE)
    
    with pytest.raises(ValidationError):
        validate_production_ssl("production", SSLMode.DISABLE)
        
def test_redact_secret_reference():
    # Long reference
    assert redact_secret_reference("aws_secrets_manager", "arn:aws:secretsmanager:region:account:secret:prod/orders-db-AbCd") == "aws-secrets-manager:...AbCd"
    
    # Short reference
    assert redact_secret_reference("local", "abc") == "local:[REDACTED]"
    
    # Environment variable (will expose up to 12 chars if > 16)
    assert redact_secret_reference("environment", "SCHEMALENS_DB_PROD_PASSWORD_1234") == "environment:SCHEMALENS_D...1234"
