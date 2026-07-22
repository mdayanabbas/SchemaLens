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

    def _is_audit_worthy(self, permission: Permission) -> bool:
        high_risk = {
            Permission.ORGANIZATION_MANAGE,
            Permission.MEMBERS_MANAGE,
            Permission.POLICIES_MANAGE,
            Permission.AUDIT_READ,
            Permission.CONNECTIONS_MANAGE,
            Permission.BUSINESS_METADATA_APPROVE,
            Permission.QUERIES_EXECUTE,
            # future query approval could be BUSINESS_METADATA_APPROVE or something else.
        }
        return permission in high_risk

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
        from app.audit.service import AuditService
        from app.audit.schemas import AuditEventCreate
        from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
        
        audit_service = AuditService(self.session)
        
        decision = await self.evaluate_permission(
            user=user,
            organization_id=organization_id,
            permission=permission,
        )
        
        # Prepare common audit metadata
        audit_metadata = {
            "requested_permission": permission.value,
            "authorization_decision_code": decision.decision_code,
            "selected_organization_id": str(organization_id),
        }
        if decision.membership_id:
            audit_metadata["membership_id"] = str(decision.membership_id)
            
        actor_type = AuditActorType.PLATFORM_ADMIN if user.is_platform_admin else AuditActorType.USER
        
        # We only record organization ID if it was successfully resolved (not 404),
        # but the decision provides it anyway.
        audit_org_id = organization_id if decision.decision_code != "ORGANIZATION_NOT_FOUND" else None

        if not decision.allowed:
            # Mask existence for cross-tenant errors where possible, or use standard messages
            await audit_service.record_denial(AuditEventCreate(
                organization_id=audit_org_id,
                actor_user_id=user.id,
                actor_type=actor_type,
                action=AuditAction.AUTHORIZATION_DENIED,
                outcome=AuditOutcome.DENIED,
                resource_type=AuditResourceType.AUTHORIZATION,
                metadata=audit_metadata,
            ))
            
            raise AuthorizationError(
                message=decision.safe_reason,
                code=decision.decision_code,
            )
        else:
            if decision.decision_code == "PLATFORM_ADMIN_ALLOWED":
                await audit_service.record_success(AuditEventCreate(
                    organization_id=audit_org_id,
                    actor_user_id=user.id,
                    actor_type=actor_type,
                    action=AuditAction.AUTHORIZATION_PLATFORM_ADMIN_BYPASS,
                    outcome=AuditOutcome.SUCCEEDED,
                    resource_type=AuditResourceType.AUTHORIZATION,
                    metadata=audit_metadata,
                ))
            elif self._is_audit_worthy(permission):
                await audit_service.record_success(AuditEventCreate(
                    organization_id=audit_org_id,
                    actor_user_id=user.id,
                    actor_type=actor_type,
                    action=AuditAction.AUTHORIZATION_ALLOWED,
                    outcome=AuditOutcome.SUCCEEDED,
                    resource_type=AuditResourceType.AUTHORIZATION,
                    metadata=audit_metadata,
                ))

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
