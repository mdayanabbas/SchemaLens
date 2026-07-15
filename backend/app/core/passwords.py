from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from app.core.exceptions import AppError


password_hash = PasswordHash((Argon2Hasher(),))


class PasswordValidationException(AppError):
    def __init__(self, message: str = "Invalid password format."):
        super().__init__(message=message, code="PASSWORD_VALIDATION_ERROR", status_code=400)


class PasswordService:
    def __init__(self, min_length: int = 12, max_length: int = 128):
        self.min_length = min_length
        self.max_length = max_length

    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        self.validate_password(password)
        return password_hash.hash(password)

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against a hash in constant time."""
        try:
            return password_hash.verify(password, hashed_password)
        except Exception:
            # Safely handle any internal library errors (e.g., malformed hash string)
            # without leaking information or throwing 500s.
            return False

    def validate_password(self, password: str) -> None:
        """
        Validate a plaintext password against security policies.
        Does not trim passwords, rejects null bytes, and enforces length bounds.
        """
        if not password:
            raise PasswordValidationException("Password cannot be empty.")
            
        if len(password) < self.min_length:
            raise PasswordValidationException(f"Password must be at least {self.min_length} characters long.")
            
        if len(password) > self.max_length:
            raise PasswordValidationException(f"Password cannot exceed {self.max_length} characters.")
            
        if "\x00" in password:
            raise PasswordValidationException("Password contains invalid characters.")
            
        if not password.strip():
            raise PasswordValidationException("Password cannot consist only of whitespace.")

    def needs_rehash(self, hashed_password: str) -> bool:
        """Check if the hash format is outdated and needs to be recomputed."""
        try:
            return password_hash.check_needs_rehash(hashed_password)
        except Exception:
            return False
