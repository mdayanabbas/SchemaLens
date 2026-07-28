import uuid
from typing import Sequence, TypeVar, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

from app.db.base_class import Base

T = TypeVar("T", bound=Base)


class SchemaSnapshotMetadataRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def bulk_insert(self, model: type[T], mappings: Sequence[dict[str, Any]]) -> None:
        """
        Bulk insert records for a specific metadata model.
        """
        if not mappings:
            return
            
        # Using insert().values() for bulk insert
        stmt = insert(model).values(mappings)
        await self.session.execute(stmt)
