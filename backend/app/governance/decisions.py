import uuid
from dataclasses import dataclass
from typing import Literal

from app.governance.permissions import Permission
from app.models.enums import OrganizationRole


DecisionCode = Literal[
    "ALLOWED",
    "PLATFORM_ADMIN_ALLOWED",
    "MEMBERSHIP_NOT_FOUND",
    "MEMBERSHIP_INACTIVE",
    "USER_DISABLED",
    "ORGANIZATION_SUSPENDED",
    "PERMISSION_DENIED",
    "ORGANIZATION_CONTEXT_REQUIRED",
    "ORGANIZATION_NOT_FOUND",
]


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    permission: Permission
    organization_id: uuid.UUID
    user_id: uuid.UUID
    membership_id: uuid.UUID | None
    role: OrganizationRole | None
    decision_code: DecisionCode
    safe_reason: str
