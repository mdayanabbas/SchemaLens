import pytest

from app.connectors.postgres.expression_sanitizer import ExpressionSanitizer
from app.core.config import Settings


def test_sanitize_default_expression():
    settings = Settings(schema_snapshot_expression_hash_salt="test-salt", schema_snapshot_max_expression_length=10)
    sanitizer = ExpressionSanitizer(settings)
    
    # Short expression - no truncation
    result = sanitizer.sanitize_default_expression("123")
    assert result.original == "123"
    assert result.truncated is False
    assert result.hash is not None
    
    # Long expression - truncation
    long_expr = "123456789012345"
    result = sanitizer.sanitize_default_expression(long_expr)
    assert result.original == "1234567890..."
    assert result.truncated is True
    
    # Consistent hash
    res1 = sanitizer.sanitize_default_expression("some_expr")
    res2 = sanitizer.sanitize_default_expression("some_expr")
    assert res1.hash == res2.hash
