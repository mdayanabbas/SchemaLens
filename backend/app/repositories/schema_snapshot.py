import uuid
from typing import Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema_snapshot import SchemaSnapshot
from app.models.schema_snapshot_enums import SchemaSnapshotStatus
from app.repositories.base import BaseRepository


class SchemaSnapshotRepository(BaseRepository[SchemaSnapshot]):
    def __init__(self, session: AsyncSession):
        super().__init__(SchemaSnapshot, session)

    async def get_latest_for_connection(self, connection_id: uuid.UUID) -> SchemaSnapshot | None:
        stmt = (
            select(SchemaSnapshot)
            .where(SchemaSnapshot.connection_id == connection_id)
            .order_by(SchemaSnapshot.snapshot_version.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_ready_snapshot(self, snapshot_id: uuid.UUID) -> SchemaSnapshot | None:
        stmt = select(SchemaSnapshot).where(
            SchemaSnapshot.id == snapshot_id,
            SchemaSnapshot.status == SchemaSnapshotStatus.READY
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def mark_superseded(self, connection_id: uuid.UUID, exclude_snapshot_id: uuid.UUID) -> None:
        stmt = (
            update(SchemaSnapshot)
            .where(
                SchemaSnapshot.connection_id == connection_id,
                SchemaSnapshot.id != exclude_snapshot_id,
                SchemaSnapshot.status == SchemaSnapshotStatus.READY
            )
            .values(status=SchemaSnapshotStatus.SUPERSEDED)
        )
        await self.session.execute(stmt)

    async def get_by_scan_id(self, scan_id: uuid.UUID) -> SchemaSnapshot | None:
        stmt = select(SchemaSnapshot).where(SchemaSnapshot.schema_scan_id == scan_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
