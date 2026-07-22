import uuid
from datetime import datetime, UTC

import pytest

from app.audit.sanitizer import AuditMetadataSanitizer


def test_audit_sanitizer_primitives():
    sanitizer = AuditMetadataSanitizer()
    assert sanitizer.sanitize(None) == {}
    
    metadata = {
        "str": "value",
        "int": 42,
        "float": 3.14,
        "bool": True,
        "none": None,
    }
    result = sanitizer.sanitize(metadata)
    assert result == metadata


def test_audit_sanitizer_complex_types():
    sanitizer = AuditMetadataSanitizer()
    now = datetime.now(UTC)
    uid = uuid.uuid4()
    
    class DummyEnum:
        value = "DUMMY"
        
    metadata = {
        "date": now,
        "uuid": uid,
        "enum": DummyEnum(),
        "bytes": b"hello",
    }
    result = sanitizer.sanitize(metadata)
    assert result["date"] == now.isoformat()
    assert result["uuid"] == str(uid)
    assert result["enum"] == "[UNSUPPORTED TYPE: DummyEnum]"  # Not an actual enum class in python test, let's use real Enum
    assert result["bytes"] == "[UNSUPPORTED BYTES]"


def test_audit_sanitizer_enum():
    from enum import Enum
    class Status(Enum):
        ACTIVE = "active"
        
    sanitizer = AuditMetadataSanitizer()
    result = sanitizer.sanitize({"status": Status.ACTIVE})
    assert result["status"] == "active"


def test_audit_sanitizer_redaction():
    sanitizer = AuditMetadataSanitizer()
    metadata = {
        "password": "my_secret_password",
        "token": "12345",
        "SECRET_key": "hidden",
        "database_url": "postgres://user:pass@host/db",
        "normal": "visible",
        "nested": {
            "api_key": "sk-1234",
            "safe": "data",
        }
    }
    result = sanitizer.sanitize(metadata)
    assert result["password"] == "[REDACTED]"
    assert result["token"] == "[REDACTED]"
    assert result["SECRET_key"] == "[REDACTED]"
    assert result["database_url"] == "[REDACTED]"
    assert result["normal"] == "visible"
    assert result["nested"]["api_key"] == "[REDACTED]"
    assert result["nested"]["safe"] == "data"


def test_audit_sanitizer_limits():
    sanitizer = AuditMetadataSanitizer(
        max_depth=2, 
        max_dict_keys=2, 
        max_list_items=2, 
        max_string_length=10
    )
    
    metadata = {
        "long_string": "this is a very long string that should be truncated",
        "list": [1, 2, 3, 4],
        "deep": {
            "deeper": {
                "too_deep": "hidden"
            }
        },
        "many_keys": {
            "k1": 1,
            "k2": 2,
            "k3": 3,
        }
    }
    
    result = sanitizer.sanitize(metadata)
    
    # Check string truncation
    assert result["long_string"] == "this is a ... [TRUNCATED]"
    
    # Check list truncation
    assert len(result["list"]) == 3  # 2 items + truncated marker
    assert result["list"][0] == 1
    assert result["list"][1] == 2
    assert result["list"][2] == "[MAX LIST ITEMS REACHED]"
    
    # Check depth truncation
    assert result["deep"]["deeper"] == {"_truncated": "[MAX DEPTH REACHED]"}
    
    # Check dict keys truncation (top level has 4 keys, max is 2)
    # Wait, the sanitizer truncates during iteration.
    # We passed 4 keys. It processes up to max_dict_keys and then adds "_truncated".
    assert len(result.keys()) == 3  # 2 valid keys + _truncated
    assert "_truncated" in result
