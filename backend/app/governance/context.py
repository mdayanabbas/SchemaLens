import uuid
from dataclasses import dataclass

from app.governance.permissions import Permission
from app.models.enums import OrganizationRole


@dataclass(frozen=True)
class AuthorizedOrganizationContext:
    user_id: uuid.UUID
    organization_id: uuid.UUID
    membership_id: uuid.UUID | None
    role: OrganizationRole | None
    is_platform_admin: bool
    permissions: frozenset[Permission]
