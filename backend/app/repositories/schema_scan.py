import uuid
from datetime import datetime

from sqlalchemy import func, select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema_scan import SchemaScan
from app.models.schema_scan_enums import SchemaScanStatus


class SchemaScanRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id_for_organization(
        self,
        *,
        scan_id: uuid.UUID,
        organization_id: uuid.UUID,
        for_update: bool = False,
    ) -> SchemaScan | None:
        stmt = select(SchemaScan).where(
            SchemaScan.id == scan_id,
            SchemaScan.organization_id == organization_id
        )
        if for_update:
            stmt = stmt.with_for_update()
            
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_for_connection(
        self,
        *,
        connection_id: uuid.UUID,
        organization_id: uuid.UUID,
        for_update: bool = False,
    ) -> SchemaScan | None:
        active_statuses = [
            SchemaScanStatus.QUEUED,
            SchemaScanStatus.RUNNING,
            SchemaScanStatus.CANCELLATION_REQUESTED,
        ]
        stmt = select(SchemaScan).where(
            SchemaScan.connection_id == connection_id,
            SchemaScan.organization_id == organization_id,
            SchemaScan.status.in_(active_statuses)
        )
        if for_update:
            stmt = stmt.with_for_update()
            
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_for_connection(
        self,
        *,
        connection_id: uuid.UUID,
        organization_id: uuid.UUID,
        offset: int,
        limit: int,
        status: SchemaScanStatus | None = None,
    ) -> list[SchemaScan]:
        stmt = select(SchemaScan).where(
            SchemaScan.connection_id == connection_id,
            SchemaScan.organization_id == organization_id
        )
        
        if status:
            stmt = stmt.where(SchemaScan.status == status)
            
        stmt = stmt.order_by(SchemaScan.created_at.desc(), SchemaScan.id.desc())
        stmt = stmt.offset(offset).limit(limit)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_for_connection(
        self,
        *,
        connection_id: uuid.UUID,
        organization_id: uuid.UUID,
        status: SchemaScanStatus | None = None,
    ) -> int:
        stmt = select(func.count(SchemaScan.id)).where(
            SchemaScan.connection_id == connection_id,
            SchemaScan.organization_id == organization_id
        )
        if status:
            stmt = stmt.where(SchemaScan.status == status)
            
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def list_stale_running(
        self,
        *,
        stale_before: datetime,
        limit: int,
    ) -> list[SchemaScan]:
        """
        INTERNAL SYSTEM QUERY. NOT FOR TENANT API USE.
        Finds scans that are running or cancellation_requested and have not heartbeated recently,
        or never heartbeated and started long ago.
        """
        active_states = [SchemaScanStatus.RUNNING, SchemaScanStatus.CANCELLATION_REQUESTED]
        
        stmt = select(SchemaScan).where(
            SchemaScan.status.in_(active_states),
            SchemaScan.completed_at.is_(None),
            or_(
                SchemaScan.heartbeat_at < stale_before,
                # if never heartbeated, check if it was started long ago
                # (or just created long ago if started_at is also null)
                SchemaScan.created_at < stale_before
            )
        )
        stmt = stmt.order_by(SchemaScan.created_at.asc()).limit(limit)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    def add(self, scan: SchemaScan) -> None:
        self.session.add(scan)

    async def flush(self) -> None:
        await self.session.flush()
