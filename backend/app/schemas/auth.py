import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.models.enums import UserStatus


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., max_length=2048)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.lower().strip()


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class SetPasswordRequest(BaseModel):
    password: str = Field(...)
    password_confirmation: str = Field(...)

    @field_validator("password_confirmation")
    @classmethod
    def passwords_match(cls, v: str, info: ValidationInfo) -> str:
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match.")
        return v
        
    def __repr__(self) -> str:
        return "<SetPasswordRequest>"


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int


class AccessTokenClaims(BaseModel):
    sub: str
    token_type: str
    jti: str
    iat: int
    nbf: int
    exp: int
    iss: str
    aud: str


class AuthenticatedUserRead(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str
    status: UserStatus
    is_platform_admin: bool
    last_login_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
