import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.db.transactions import transactional
from app.models.membership import OrganizationMembership
from app.repositories.membership import MembershipRepository
from app.repositories.organization import OrganizationRepository
from app.repositories.user import UserRepository
from app.schemas.membership import MembershipCreate, MembershipRead, MembershipUpdate


class MembershipService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.membership_repo = MembershipRepository(session)
        self.org_repo = OrganizationRepository(session)
        self.user_repo = UserRepository(session)
        from app.audit.service import AuditService
        self.audit_service = AuditService(session)

    async def add_member_authorized(
        self,
        context: "AuthorizedOrganizationContext",
        membership_in: MembershipCreate,
        exact_email: str | None = None,
    ) -> MembershipRead:
        """Create a new membership using an authorized context."""
        from app.audit.schemas import AuditEventCreate
        from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
        
        async with transactional(self.session):
            if exact_email:
                user = await self.user_repo.get_by_email(exact_email)
            else:
                user = await self.user_repo.get_by_id(membership_in.user_id)

            if not user:
                raise NotFoundError(
                    message="User not found.",
                    code="USER_NOT_FOUND",
                )

            exists = await self.membership_repo.membership_exists(
                user_id=user.id,
                organization_id=context.organization_id,
            )
            if exists:
                raise ConflictError(
                    message="User is already a member of this organization.",
                    code="MEMBERSHIP_ALREADY_EXISTS",
                )

            membership = OrganizationMembership(
                organization_id=context.organization_id,
                user_id=user.id,
                role=membership_in.role,
                status=membership_in.status,
            )
            self.membership_repo.add(membership)
            await self.membership_repo.flush()

            actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER
            await self.audit_service.record_success(AuditEventCreate(
                organization_id=context.organization_id,
                actor_user_id=context.user_id,
                actor_type=actor_type,
                action=AuditAction.MEMBERSHIP_CREATED,
                outcome=AuditOutcome.SUCCEEDED,
                resource_type=AuditResourceType.MEMBERSHIP,
                resource_id=membership.id,
                metadata={
                    "target_user_id": str(user.id),
                    "role": membership.role,
                    "status": membership.status,
                }
            ))
            await self.session.flush()

            return MembershipRead.model_validate(membership)

    async def get_membership(
        self, organization_id: uuid.UUID, membership_id: uuid.UUID
    ) -> MembershipRead:
        """Retrieve a specific membership for an organization."""
        membership = await self.membership_repo.get_by_id_for_organization(
            membership_id=membership_id, organization_id=organization_id
        )
        if not membership:
            raise NotFoundError(
                message="Membership not found.",
                code="MEMBERSHIP_NOT_FOUND",
            )
        return MembershipRead.model_validate(membership)

    async def update_member_authorized(
        self,
        context: "AuthorizedOrganizationContext",
        membership_id: uuid.UUID,
        update_in: MembershipUpdate,
    ) -> MembershipRead:
        """Update a membership's role or status safely."""
        from app.models.enums import MembershipStatus, OrganizationRole
        from app.core.exceptions import ValidationError
        from app.audit.schemas import AuditEventCreate
        from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
        
        async with transactional(self.session):
            membership = await self.membership_repo.get_by_id_for_organization(
                membership_id=membership_id, organization_id=context.organization_id
            )
            if not membership:
                raise NotFoundError(
                    message="Membership not found.",
                    code="MEMBERSHIP_NOT_FOUND",
                )

            is_admin = membership.role == OrganizationRole.ORGANIZATION_ADMIN
            is_active = membership.status == MembershipStatus.ACTIVE
            
            actor_type = AuditActorType.PLATFORM_ADMIN if context.is_platform_admin else AuditActorType.USER

            # Determine if this operation demotes or disables an active admin
            demoting_admin = update_in.role is not None and update_in.role != OrganizationRole.ORGANIZATION_ADMIN
            disabling_admin = update_in.status is not None and update_in.status != MembershipStatus.ACTIVE

            if is_admin and is_active and (demoting_admin or disabling_admin):
                # Lock row if necessary, count admins
                active_admins = await self.membership_repo.count_active_admins_for_organization(
                    context.organization_id
                )
                if active_admins <= 1:
                    # Last administrator protection
                    await self.audit_service.record_denial(AuditEventCreate(
                        organization_id=context.organization_id,
                        actor_user_id=context.user_id,
                        actor_type=actor_type,
                        action=AuditAction.MEMBERSHIP_UPDATED,
                        outcome=AuditOutcome.DENIED,
                        resource_type=AuditResourceType.MEMBERSHIP,
                        resource_id=membership.id,
                        metadata={
                            "reason": "LAST_ORGANIZATION_ADMIN_REQUIRED",
                            "target_user_id": str(membership.user_id),
                        }
                    ))
                    await self.session.flush()
                    raise ValidationError(
                        message="Cannot modify or disable the last active organization administrator.",
                        code="LAST_ORGANIZATION_ADMIN_REQUIRED",
                    )
            
            changed_fields = []
            previous_role = None
            new_role = None
            previous_status = None
            new_status = None
            
            if update_in.role is not None and membership.role != update_in.role:
                previous_role = membership.role
                membership.role = update_in.role
                new_role = update_in.role
                changed_fields.append("role")
            if update_in.status is not None and membership.status != update_in.status:
                previous_status = membership.status
                membership.status = update_in.status
                new_status = update_in.status
                changed_fields.append("status")
                
            await self.membership_repo.flush()
            
            if changed_fields:
                await self.audit_service.record_success(AuditEventCreate(
                    organization_id=context.organization_id,
                    actor_user_id=context.user_id,
                    actor_type=actor_type,
                    action=AuditAction.MEMBERSHIP_UPDATED,
                    outcome=AuditOutcome.SUCCEEDED,
                    resource_type=AuditResourceType.MEMBERSHIP,
                    resource_id=membership.id,
                    metadata={"changed_fields": changed_fields, "target_user_id": str(membership.user_id)}
                ))
                
                if new_role:
                    await self.audit_service.record_success(AuditEventCreate(
                        organization_id=context.organization_id,
                        actor_user_id=context.user_id,
                        actor_type=actor_type,
                        action=AuditAction.MEMBERSHIP_ROLE_CHANGED,
                        outcome=AuditOutcome.SUCCEEDED,
                        resource_type=AuditResourceType.MEMBERSHIP,
                        resource_id=membership.id,
                        metadata={"target_user_id": str(membership.user_id), "previous_role": previous_role, "new_role": new_role}
                    ))
                    
                if new_status == MembershipStatus.DISABLED:
                    await self.audit_service.record_success(AuditEventCreate(
                        organization_id=context.organization_id,
                        actor_user_id=context.user_id,
                        actor_type=actor_type,
                        action=AuditAction.MEMBERSHIP_DISABLED,
                        outcome=AuditOutcome.SUCCEEDED,
                        resource_type=AuditResourceType.MEMBERSHIP,
                        resource_id=membership.id,
                        metadata={"target_user_id": str(membership.user_id), "previous_status": previous_status, "new_status": new_status}
                    ))
                await self.session.flush()

            return MembershipRead.model_validate(membership)

    async def disable_member_authorized(
        self,
        context: "AuthorizedOrganizationContext",
        membership_id: uuid.UUID,
    ) -> MembershipRead:
        """Disable a membership safely."""
        from app.models.enums import MembershipStatus
        update_in = MembershipUpdate(status=MembershipStatus.DISABLED)
        return await self.update_member_authorized(context, membership_id, update_in)

    async def list_organization_memberships_authorized(
        self,
        context: "AuthorizedOrganizationContext",
        limit: int = 100,
        offset: int = 0,
        role: "OrganizationRole | None" = None,
        status: "MembershipStatus | None" = None,
    ) -> list["MembershipDetailedRead"]:
        """List memberships for a specific organization using authorized context."""
        from app.schemas.membership import MembershipDetailedRead
        
        memberships = await self.membership_repo.list_for_organization(
            organization_id=context.organization_id, limit=limit, offset=offset, role=role, status=status
        )
        return [
            MembershipDetailedRead(
                id=m.id,
                organization_id=m.organization_id,
                user_id=m.user_id,
                user_email=m.user.email,
                user_display_name=m.user.display_name,
                role=m.role,
                status=m.status,
                created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in memberships
        ]

    async def list_user_memberships(
        self, user_id: uuid.UUID, limit: int = 100, offset: int = 0
    ) -> list[MembershipRead]:
        """List memberships for a specific user across all their organizations."""
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundError(
                message="User not found.",
                code="USER_NOT_FOUND",
            )

        memberships = await self.membership_repo.list_for_user(
            user_id=user_id, limit=limit, offset=offset
        )
        return [MembershipRead.model_validate(m) for m in memberships]
