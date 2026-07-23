import uuid

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.connection_policy import ConnectionPolicy


class ConnectionPolicyRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_for_connection_and_organization(
        self,
        connection_id: uuid.UUID,
        organization_id: uuid.UUID,
        *,
        for_update: bool = False,
    ) -> ConnectionPolicy | None:
        """
        Get the policy for a connection, ensuring it belongs to the given organization.
        Optionally lock the row for update.
        """
        stmt = select(ConnectionPolicy).where(
            and_(
                ConnectionPolicy.connection_id == connection_id,
                ConnectionPolicy.organization_id == organization_id,
            )
        )
        if for_update:
            stmt = stmt.with_for_update()
            
        result = await self.session.execute(stmt)
        return result.scalars().first()

    def add(self, policy: ConnectionPolicy) -> None:
        """Add a policy to the session without committing."""
        self.session.add(policy)

    async def flush(self) -> None:
        """Flush pending changes to the database."""
        await self.session.flush()
