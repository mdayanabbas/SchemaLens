import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Keys containing any of these substrings will be redacted (case-insensitive)
SENSITIVE_KEY_FRAGMENTS = {
    "password",
    "secret",
    "token",
    "authorization",
    "cookie",
    "credential",
    "database_url",
    "connection_url",
    "private_key",
    "api_key",
    "client_secret",
    "sql_parameters",
    "bind_parameters",
    "raw_rows",
    "result_rows",
}

class AuditSanitizationError(Exception):
    """Raised when metadata cannot be safely sanitized."""
    pass


class AuditMetadataSanitizer:
    def __init__(
        self,
        max_depth: int = 5,
        max_dict_keys: int = 50,
        max_list_items: int = 50,
        max_string_length: int = 1000,
        max_serialized_size: int = 65536,  # 64 KB limit
    ):
        self.max_depth = max_depth
        self.max_dict_keys = max_dict_keys
        self.max_list_items = max_list_items
        self.max_string_length = max_string_length
        self.max_serialized_size = max_serialized_size

    def sanitize(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        if metadata is None:
            return {}

        if not isinstance(metadata, Mapping):
            # As a fallback for top-level non-mapping, we wrap it in a dict if it can be sanitized
            try:
                sanitized_value = self._sanitize_value(metadata, depth=0)
                metadata = {"_value": sanitized_value}
            except Exception:
                return {"_error": "Invalid top-level metadata type"}

        try:
            sanitized = self._sanitize_dict(metadata, depth=0)
            
            # Verify serialized size
            serialized = json.dumps(sanitized)
            if len(serialized.encode("utf-8")) > self.max_serialized_size:
                return {"_error": "[TRUNCATED] Exceeded maximum serialized size."}
                
            return sanitized
        except Exception as e:
            logger.warning(f"Failed to sanitize audit metadata: {e}")
            return {"_error": "Failed to sanitize metadata safely"}

    def _is_sensitive_key(self, key: str) -> bool:
        lower_key = str(key).lower()
        return any(frag in lower_key for frag in SENSITIVE_KEY_FRAGMENTS)

    def _sanitize_dict(self, data: Mapping[str, Any], depth: int) -> dict[str, Any]:
        if depth >= self.max_depth:
            return {"_truncated": "[MAX DEPTH REACHED]"}

        result = {}
        for count, (k, v) in enumerate(data.items()):
            if count >= self.max_dict_keys:
                result["_truncated"] = "[MAX DICT KEYS REACHED]"
                break
                
            str_key = str(k)
            if self._is_sensitive_key(str_key):
                result[str_key] = "[REDACTED]"
            else:
                result[str_key] = self._sanitize_value(v, depth + 1)
        return result

    def _sanitize_list(self, data: Sequence[Any], depth: int) -> list[Any]:
        if depth >= self.max_depth:
            return ["[MAX DEPTH REACHED]"]

        result = []
        for count, item in enumerate(data):
            if count >= self.max_list_items:
                result.append("[MAX LIST ITEMS REACHED]")
                break
            result.append(self._sanitize_value(item, depth + 1))
        return result

    def _sanitize_value(self, value: Any, depth: int) -> Any:
        if value is None:
            return None
        elif isinstance(value, bool):
            return value
        elif isinstance(value, int | float):
            import math
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return str(value)
            return value
        elif isinstance(value, str):
            if len(value) > self.max_string_length:
                return value[:self.max_string_length] + "... [TRUNCATED]"
            return value
        elif isinstance(value, uuid.UUID):
            return str(value)
        elif isinstance(value, Enum):
            return str(value.value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Mapping):
            return self._sanitize_dict(value, depth)
        elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
            return self._sanitize_list(value, depth)
        elif isinstance(value, bytes | bytearray):
            return "[UNSUPPORTED BYTES]"
        else:
            # Fallback for unexpected complex objects (e.g., ORM models, exceptions, closures)
            return f"[UNSUPPORTED TYPE: {type(value).__name__}]"
