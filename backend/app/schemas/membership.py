import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.enums import MembershipStatus, OrganizationRole


class MembershipCreate(BaseModel):
    user_id: uuid.UUID
    role: OrganizationRole
    status: MembershipStatus = MembershipStatus.INVITED


class MembershipUpdate(BaseModel):
    role: OrganizationRole | None = None
    status: MembershipStatus | None = None


class MembershipRead(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    user_id: uuid.UUID
    role: OrganizationRole
    status: MembershipStatus
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MembershipDetailedRead(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    user_id: uuid.UUID
    user_email: str
    user_display_name: str
    role: OrganizationRole
    status: MembershipStatus
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
