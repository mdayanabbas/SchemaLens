import pytest

from app.core.passwords import PasswordService, PasswordValidationException


def test_password_hashing():
    service = PasswordService()
    password = "a_strong_password_123"
    
    hashed = service.hash_password(password)
    assert hashed != password
    assert service.verify_password(password, hashed) is True
    assert service.verify_password("wrong_password", hashed) is False


def test_password_validation():
    service = PasswordService(min_length=8, max_length=64)
    
    with pytest.raises(PasswordValidationException):
        service.validate_password("short")
        
    with pytest.raises(PasswordValidationException):
        service.validate_password("a" * 65)
        
    with pytest.raises(PasswordValidationException):
        service.validate_password("has\x00null")
        
    with pytest.raises(PasswordValidationException):
        service.validate_password("   \t  ")
        
    # Should pass
    service.validate_password(" valid password ")


def test_invalid_hash_verification():
    service = PasswordService()
    assert service.verify_password("password", "invalid_hash_format") is False
