import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models.enums import UserStatus


class UserCreate(BaseModel):
    email: EmailStr
    display_name: str = Field(..., min_length=1, max_length=150)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.lower().strip()

    @field_validator("display_name")
    @classmethod
    def trim_display_name(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Display name cannot be empty or just whitespace.")
        return trimmed


class UserUpdate(BaseModel):
    display_name: str | None = Field(None, min_length=1, max_length=150)
    status: UserStatus | None = None

    @field_validator("display_name")
    @classmethod
    def trim_display_name(cls, v: str | None) -> str | None:
        if v is not None:
            trimmed = v.strip()
            if not trimmed:
                raise ValueError("Display name cannot be empty or just whitespace.")
            return trimmed
        return v


class UserRead(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str
    status: UserStatus
    is_platform_admin: bool
    last_login_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
