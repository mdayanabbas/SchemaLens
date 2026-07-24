import uuid

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.stored_secret import StoredSecret
from app.secrets.enums import SecretStatus


class StoredSecretRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id_for_organization(
        self,
        *,
        secret_id: uuid.UUID,
        organization_id: uuid.UUID,
        for_update: bool = False,
    ) -> StoredSecret | None:
        """
        Get a secret by ID scoped to the organization.
        Optionally lock the row for rotation.
        """
        stmt = select(StoredSecret).where(
            and_(
                StoredSecret.id == secret_id,
                StoredSecret.organization_id == organization_id,
            )
        )
        if for_update:
            stmt = stmt.with_for_update()

        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_active_by_name_for_organization(
        self,
        *,
        name: str,
        organization_id: uuid.UUID,
    ) -> StoredSecret | None:
        """
        Get an active secret by name scoped to the organization.
        """
        stmt = select(StoredSecret).where(
            and_(
                StoredSecret.name == name,
                StoredSecret.organization_id == organization_id,
                StoredSecret.status == SecretStatus.ACTIVE,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def name_exists_for_organization(
        self,
        *,
        name: str,
        organization_id: uuid.UUID,
    ) -> bool:
        """
        Check if a name is taken within the organization.
        """
        stmt = select(1).where(
            and_(
                StoredSecret.name == name,
                StoredSecret.organization_id == organization_id,
            )
        )
        result = await self.session.execute(stmt)
        return result.first() is not None

    async def list_for_organization(
        self,
        *,
        organization_id: uuid.UUID,
        offset: int,
        limit: int,
        status: SecretStatus | None = None,
    ) -> list[StoredSecret]:
        """
        List secrets bounded and scoped to the organization.
        """
        stmt = select(StoredSecret).where(
            StoredSecret.organization_id == organization_id
        )

        if status:
            stmt = stmt.where(StoredSecret.status == status)

        # Deterministic ordering
        stmt = stmt.order_by(StoredSecret.created_at.desc(), StoredSecret.id.desc())
        stmt = stmt.offset(offset).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    def add(self, secret: StoredSecret) -> None:
        """Add a secret to the session."""
        self.session.add(secret)

    async def flush(self) -> None:
        """Flush changes."""
        await self.session.flush()
