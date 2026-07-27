import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema_scan_transition import SchemaScanTransition


class SchemaScanTransitionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_for_scan_and_organization(
        self,
        *,
        schema_scan_id: uuid.UUID,
        organization_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> list[SchemaScanTransition]:
        """List transitions ordered chronologically."""
        stmt = select(SchemaScanTransition).where(
            SchemaScanTransition.schema_scan_id == schema_scan_id,
            SchemaScanTransition.organization_id == organization_id
        ).order_by(SchemaScanTransition.occurred_at.asc())
        
        stmt = stmt.offset(offset).limit(limit)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    def add(self, transition: SchemaScanTransition) -> None:
        """Add a transition to the session."""
        self.session.add(transition)
