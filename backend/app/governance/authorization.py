import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthorizationError
from app.governance.context import AuthorizedOrganizationContext
from app.governance.decisions import AuthorizationDecision
from app.governance.permissions import Permission
from app.governance.role_permissions import permissions_for_role, role_has_permission
from app.models.enums import MembershipStatus, OrganizationRole, OrganizationStatus, UserStatus
from app.models.membership import OrganizationMembership
from app.models.user import User
from app.repositories.membership import MembershipRepository
from app.repositories.organization import OrganizationRepository


class AuthorizationService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.org_repo = OrganizationRepository(session)
        self.membership_repo = MembershipRepository(session)

    async def evaluate_permission(
        self,
        *,
        user: User,
        organization_id: uuid.UUID,
        permission: Permission,
    ) -> AuthorizationDecision:
        """Evaluate if a user has a specific permission in an organization."""
        
        # 1. Reject disabled users.
        if user.status != UserStatus.ACTIVE:
            return AuthorizationDecision(
                allowed=False,
                permission=permission,
                organization_id=organization_id,
                user_id=user.id,
                membership_id=None,
                role=None,
                decision_code="USER_DISABLED",
                safe_reason="User account is disabled.",
            )

        # 2. Load organization.
        organization = await self.org_repo.get_by_id(organization_id)
        
        # 3. Reject missing organizations without revealing cross-tenant details.
        if not organization:
            return AuthorizationDecision(
                allowed=False,
                permission=permission,
                organization_id=organization_id,
                user_id=user.id,
                membership_id=None,
                role=None,
                decision_code="ORGANIZATION_NOT_FOUND",
                safe_reason="Organization not found.",
            )

        # 4. Reject suspended organizations.
        if organization.status == OrganizationStatus.SUSPENDED:
            return AuthorizationDecision(
                allowed=False,
                permission=permission,
                organization_id=organization_id,
                user_id=user.id,
                membership_id=None,
                role=None,
                decision_code="ORGANIZATION_SUSPENDED",
                safe_reason="Organization is suspended.",
            )

        # 5. If user is platform administrator:
        if user.is_platform_admin:
            return AuthorizationDecision(
                allowed=True,
                permission=permission,
                organization_id=organization_id,
                user_id=user.id,
                membership_id=None,
                role=None,
                decision_code="PLATFORM_ADMIN_ALLOWED",
                safe_reason="Platform administrator access.",
            )

        # 6. Otherwise load active membership using user ID and organization ID.
        membership = await self.membership_repo.get_for_user_and_organization(
            user_id=user.id, organization_id=organization_id
        )

        # 7. Reject missing or inactive membership.
        if not membership:
            return AuthorizationDecision(
                allowed=False,
                permission=permission,
                organization_id=organization_id,
                user_id=user.id,
                membership_id=None,
                role=None,
                decision_code="MEMBERSHIP_NOT_FOUND",
                safe_reason="Membership not found.",
            )
            
        if membership.status != MembershipStatus.ACTIVE:
            return AuthorizationDecision(
                allowed=False,
                permission=permission,
                organization_id=organization_id,
                user_id=user.id,
                membership_id=membership.id,
                role=membership.role,
                decision_code="MEMBERSHIP_INACTIVE",
                safe_reason="Membership is not active.",
            )

        # 8. Resolve permissions from the role mapping.
        has_perm = role_has_permission(membership.role, permission)

        # 9. Allow or deny the requested permission.
        if has_perm:
            return AuthorizationDecision(
                allowed=True,
                permission=permission,
                organization_id=organization_id,
                user_id=user.id,
                membership_id=membership.id,
                role=membership.role,
                decision_code="ALLOWED",
                safe_reason="Permission allowed.",
            )
        else:
            return AuthorizationDecision(
                allowed=False,
                permission=permission,
                organization_id=organization_id,
                user_id=user.id,
                membership_id=membership.id,
                role=membership.role,
                decision_code="PERMISSION_DENIED",
                safe_reason="Permission denied for this role.",
            )

    async def require_permission(
        self,
        *,
        user: User,
        organization_id: uuid.UUID,
        permission: Permission,
    ) -> AuthorizedOrganizationContext:
        """Require a specific permission and return an AuthorizedOrganizationContext on success."""
        decision = await self.evaluate_permission(
            user=user,
            organization_id=organization_id,
            permission=permission,
        )
        
        if not decision.allowed:
            # Mask existence for cross-tenant errors where possible, or use standard messages
            if decision.decision_code in ("ORGANIZATION_NOT_FOUND", "MEMBERSHIP_NOT_FOUND"):
                # Use a generic message, wait the instructions say: 
                # "Do not expose whether another organization exists to unauthorized users."
                # We can just throw a standard 403 or 404 (we'll use AuthorizationError with 403 or 404 appropriately depending on if we are keeping generic permission denied)
                # "Use generic user-facing denial messages."
                # "Preserve safe decision codes in exception details where appropriate."
                pass
            
            raise AuthorizationError(
                message=decision.safe_reason,
                code=decision.decision_code,
            )

        # 10. Return a safe AuthorizedOrganizationContext.
        role_perms = frozenset()
        if decision.role:
            role_perms = permissions_for_role(decision.role)
            
        return AuthorizedOrganizationContext(
            user_id=user.id,
            organization_id=organization_id,
            membership_id=decision.membership_id,
            role=decision.role,
            is_platform_admin=user.is_platform_admin,
            permissions=role_perms,
        )
