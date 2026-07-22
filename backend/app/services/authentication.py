import uuid
from datetime import datetime, UTC

from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.enums import AuditAction, AuditActorType, AuditOutcome, AuditResourceType
from app.audit.schemas import AuditEventCreate
from app.audit.service import AuditService
from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.passwords import PasswordService
from app.core.tokens import TokenService
from app.db.transactions import transactional
from app.models.enums import RefreshTokenStatus
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.repositories.refresh_token import RefreshTokenRepository
from app.repositories.user import UserRepository
from app.schemas.auth import TokenResponse


class AuthenticationException(AppError):
    def __init__(self, message: str = "Invalid email or password.", code: str = "INVALID_CREDENTIALS"):
        super().__init__(message=message, code=code, status_code=401)


class AuthenticationService:
    def __init__(self, session: AsyncSession, settings: Settings):
        self.session = session
        self.settings = settings
        self.user_repo = UserRepository(session)
        self.refresh_repo = RefreshTokenRepository(session)
        self.password_service = PasswordService()
        self.token_service = TokenService(settings)
        self.audit_service = AuditService(session)

    async def _record_event(
        self,
        action: AuditAction,
        outcome: AuditOutcome,
        actor_user_id: uuid.UUID | None = None,
        email: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        metadata = metadata or {}
        if email:
            metadata["email_hash"] = self.token_service.hash_refresh_token(email.lower().strip())
            
        # Determine actor type safely
        actor_type = AuditActorType.USER if actor_user_id else AuditActorType.ANONYMOUS
            
        event_in = AuditEventCreate(
            actor_user_id=actor_user_id,
            actor_type=actor_type,
            action=action,
            outcome=outcome,
            resource_type=AuditResourceType.AUTHENTICATION,
            ip_hash=self.token_service.hash_optional_ip_address(ip_address),
            user_agent_hash=self.token_service.hash_optional_user_agent(user_agent),
            metadata=metadata,
        )
        await self.audit_service.record(event_in)

    async def set_initial_password(
        self, user_id: uuid.UUID, password: str, password_confirmation: str
    ) -> None:
        if password != password_confirmation:
            raise AuthenticationException("Passwords do not match.", code="PASSWORD_MISMATCH")

        async with transactional(self.session):
            user = await self.user_repo.get_by_id(user_id)
            if not user or user.status != "active":
                raise AuthenticationException("User is not active or does not exist.", code="USER_DISABLED")

            password_hash = self.password_service.hash_password(password)
            await self.user_repo.set_password_hash(user.id, password_hash)
            await self.refresh_repo.revoke_all_for_user(user.id)
            
            await self._record_event(
                action=AuditAction.AUTH_PASSWORD_CHANGED, 
                outcome=AuditOutcome.SUCCEEDED, 
                actor_user_id=user.id
            )
            await self.session.flush()

    async def login(
        self, email: str, password: str, ip_address: str | None = None, user_agent: str | None = None
    ) -> TokenResponse:
        async with transactional(self.session):
            user = await self.user_repo.get_active_by_email(email)
            
            # Dummy verification to resist timing attacks if user missing or password unset
            if not user or not user.password_hash:
                self.password_service.verify_password(password, self.password_service.hash_password("dummy"))
                await self._record_event(
                    action=AuditAction.AUTH_LOGIN_FAILED, 
                    outcome=AuditOutcome.FAILED,
                    email=email, 
                    ip_address=ip_address, 
                    user_agent=user_agent,
                    metadata={"reason": "invalid_credentials"}
                )
                await self.session.flush()
                raise AuthenticationException()

            if not self.password_service.verify_password(password, user.password_hash):
                await self._record_event(
                    action=AuditAction.AUTH_LOGIN_FAILED, 
                    outcome=AuditOutcome.FAILED,
                    actor_user_id=user.id, 
                    email=email, 
                    ip_address=ip_address, 
                    user_agent=user_agent,
                    metadata={"reason": "invalid_credentials"}
                )
                await self.session.flush()
                raise AuthenticationException()

            now = datetime.now(UTC)
            await self.user_repo.update_last_login(user.id, now)

            access_token, expires_at = self.token_service.create_access_token(user.id)
            raw_refresh = self.token_service.generate_opaque_refresh_token()
            refresh_hash = self.token_service.hash_refresh_token(raw_refresh)
            
            family_id = uuid.uuid4()
            refresh_record = RefreshToken(
                user_id=user.id,
                family_id=family_id,
                token_hash=refresh_hash,
                expires_at=self.token_service.calculate_refresh_token_expiry(),
                created_ip_hash=self.token_service.hash_optional_ip_address(ip_address),
                created_user_agent_hash=self.token_service.hash_optional_user_agent(user_agent),
            )
            self.refresh_repo.add(refresh_record)

            await self._record_event(
                action=AuditAction.AUTH_LOGIN_SUCCEEDED,
                outcome=AuditOutcome.SUCCEEDED, 
                actor_user_id=user.id, 
                ip_address=ip_address, 
                user_agent=user_agent,
                metadata={"session_family_id": str(family_id)}
            )
            await self.session.flush()

            return TokenResponse(
                access_token=access_token,
                refresh_token=raw_refresh,
                expires_in=self.settings.access_token_expire_minutes * 60,
                refresh_expires_in=self.settings.refresh_token_expire_days * 86400,
            )

    async def refresh(
        self, refresh_token: str, ip_address: str | None = None, user_agent: str | None = None
    ) -> TokenResponse:
        token_hash = self.token_service.hash_refresh_token(refresh_token)
        
        async with transactional(self.session):
            record = await self.refresh_repo.get_by_hash(token_hash, lock=True)
            if not record:
                raise AuthenticationException("Invalid session.", code="INVALID_REFRESH_TOKEN")
                
            now = datetime.now(UTC)
            if record.expires_at < now:
                raise AuthenticationException("Session expired.", code="SESSION_EXPIRED")

            if record.status == RefreshTokenStatus.ROTATED:
                # Reuse detected
                await self.refresh_repo.revoke_family(record.user_id, record.family_id)
                await self.refresh_repo.mark_compromised(record.id)
                await self._record_event(
                    action=AuditAction.AUTH_REFRESH_TOKEN_REUSE_DETECTED,
                    outcome=AuditOutcome.DENIED,
                    actor_user_id=record.user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    metadata={"session_family_id": str(record.family_id), "affected_token_record_id": str(record.id), "family_revoked": True}
                )
                await self.session.flush()
                raise AuthenticationException("Session compromised.", code="TOKEN_REUSE_DETECTED")

            if record.status != RefreshTokenStatus.ACTIVE:
                raise AuthenticationException("Session revoked.", code="SESSION_REVOKED")

            user = await self.user_repo.get_active_by_id(record.user_id)
            if not user:
                raise AuthenticationException("User is no longer active.", code="USER_DISABLED")

            new_raw_refresh = self.token_service.generate_opaque_refresh_token()
            new_refresh_hash = self.token_service.hash_refresh_token(new_raw_refresh)
            
            new_record = RefreshToken(
                user_id=user.id,
                family_id=record.family_id,
                token_hash=new_refresh_hash,
                expires_at=self.token_service.calculate_refresh_token_expiry(),
                created_ip_hash=self.token_service.hash_optional_ip_address(ip_address),
                created_user_agent_hash=self.token_service.hash_optional_user_agent(user_agent),
            )
            self.refresh_repo.add(new_record)
            await self.session.flush()

            await self.refresh_repo.mark_rotated(record.id, new_record.id)
            
            record.used_at = now
            record.last_used_ip_hash = self.token_service.hash_optional_ip_address(ip_address)
            record.last_used_user_agent_hash = self.token_service.hash_optional_user_agent(user_agent)
            
            access_token, _ = self.token_service.create_access_token(user.id)
            
            await self._record_event(
                action=AuditAction.AUTH_TOKEN_REFRESHED,
                outcome=AuditOutcome.SUCCEEDED,
                actor_user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata={
                    "session_family_id": str(record.family_id),
                    "old_token_record_id": str(record.id),
                    "new_token_record_id": str(new_record.id),
                }
            )
            await self.session.flush()
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=new_raw_refresh,
                expires_in=self.settings.access_token_expire_minutes * 60,
                refresh_expires_in=self.settings.refresh_token_expire_days * 86400,
            )

    async def logout(self, refresh_token: str, ip_address: str | None = None, user_agent: str | None = None) -> None:
        token_hash = self.token_service.hash_refresh_token(refresh_token)
        async with transactional(self.session):
            record = await self.refresh_repo.get_by_hash(token_hash, lock=True)
            if record:
                await self.refresh_repo.mark_revoked(record.id)
                await self._record_event(
                    action=AuditAction.AUTH_LOGOUT,
                    outcome=AuditOutcome.SUCCEEDED,
                    actor_user_id=record.user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    metadata={"token_record_id": str(record.id), "already_revoked": record.status != RefreshTokenStatus.ACTIVE}
                )
            await self.session.flush()

    async def revoke_all_user_sessions(self, user_id: uuid.UUID) -> None:
        async with transactional(self.session):
            revoked_count = await self.refresh_repo.revoke_all_for_user(user_id)
            
            await self._record_event(
                action=AuditAction.AUTH_SESSIONS_REVOKED,
                outcome=AuditOutcome.SUCCEEDED,
                actor_user_id=user_id,
                metadata={"revoked_count": revoked_count}
            )
            await self.session.flush()

    async def authenticate_access_token(self, token: str) -> User:
        payload = self.token_service.decode_and_validate_access_token(token)
        user_id = uuid.UUID(payload["sub"])
        
        user = await self.user_repo.get_active_by_id(user_id)
        if not user:
            raise AuthenticationException("User is not active or does not exist.", code="USER_DISABLED")
        return user
