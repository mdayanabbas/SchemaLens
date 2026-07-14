import re
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import OrganizationStatus

SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class OrganizationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    slug: str = Field(..., max_length=100)

    @field_validator("name")
    @classmethod
    def trim_name(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Name cannot be empty or just whitespace.")
        return trimmed

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v: str) -> str:
        normalized = v.lower().strip()
        if not SLUG_PATTERN.match(normalized):
            raise ValueError(
                "Slug must contain only lowercase letters, numbers, and hyphens. "
                "It cannot start or end with a hyphen, nor contain consecutive hyphens."
            )
        return normalized


class OrganizationUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    status: OrganizationStatus | None = None

    @field_validator("name")
    @classmethod
    def trim_name(cls, v: str | None) -> str | None:
        if v is not None:
            trimmed = v.strip()
            if not trimmed:
                raise ValueError("Name cannot be empty or just whitespace.")
            return trimmed
        return v


class OrganizationRead(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    status: OrganizationStatus
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
