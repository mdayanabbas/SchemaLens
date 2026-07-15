import uuid
import pytest
from datetime import datetime, UTC

from app.core.config import Settings
from app.core.tokens import TokenService, TokenValidationException


@pytest.fixture
def token_service():
    settings = Settings(
        jwt_secret_key="test_secret",
        refresh_token_pepper="test_pepper",
        authentication_issuer="test-issuer",
        authentication_audience="test-audience",
    )
    return TokenService(settings)


def test_access_token_creation_and_validation(token_service):
    user_id = uuid.uuid4()
    token, expiry = token_service.create_access_token(user_id)
    
    assert token is not None
    
    payload = token_service.decode_and_validate_access_token(token)
    assert payload["sub"] == str(user_id)
    assert payload["token_type"] == "access"


def test_refresh_token_hashing(token_service):
    raw_token = token_service.generate_opaque_refresh_token()
    assert len(raw_token) >= 32
    
    hash1 = token_service.hash_refresh_token(raw_token)
    hash2 = token_service.hash_refresh_token(raw_token)
    
    assert hash1 == hash2
    assert hash1 != raw_token
    
    # Constant time compare
    assert token_service.compare_refresh_token_hashes(hash1, hash2) is True


def test_token_validation_failures(token_service):
    # Invalid token string
    with pytest.raises(TokenValidationException):
        token_service.decode_and_validate_access_token("not-a-token")
        
    # Valid token structure, wrong secret (would be tested via PyJWT if we manually encode)
    import jwt
    wrong_token = jwt.encode({"sub": str(uuid.uuid4()), "token_type": "access", "jti": "1", "iat": 1, "nbf": 1, "exp": 9999999999, "iss": "test-issuer", "aud": "test-audience"}, "wrong_secret", algorithm="HS256")
    with pytest.raises(TokenValidationException):
        token_service.decode_and_validate_access_token(wrong_token)
