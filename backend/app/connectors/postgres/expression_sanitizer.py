import hashlib
import re

class ExpressionSanitizer:
    def __init__(self, max_length: int):
        self.max_length = max_length

    def sanitize(self, expression: str | None) -> tuple[str | None, str | None, bool]:
        """
        Sanitizes a database expression.
        Returns:
            - sanitized expression (truncated if necessary)
            - sha256 hash of the original clean expression
            - boolean indicating if it was truncated
        """
        if expression is None:
            return None, None, False

        # Clean string: remove null bytes, normalize line endings
        clean_expr = expression.replace('\x00', '').replace('\r\n', '\n').strip()
        
        if not clean_expr:
            return None, None, False

        # Compute hash of the clean, full expression
        expr_hash = hashlib.sha256(clean_expr.encode('utf-8')).hexdigest()

        # Check length
        truncated = False
        if len(clean_expr) > self.max_length:
            clean_expr = clean_expr[:self.max_length]
            truncated = True

        return clean_expr, expr_hash, truncated
