import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.connection_schema_state import ConnectionSchemaState
from app.repositories.base import BaseRepository


class ConnectionSchemaStateRepository(BaseRepository[ConnectionSchemaState]):
    def __init__(self, session: AsyncSession):
        super().__init__(ConnectionSchemaState, session)

    async def get_by_connection_id(self, connection_id: uuid.UUID) -> ConnectionSchemaState | None:
        stmt = select(ConnectionSchemaState).where(ConnectionSchemaState.connection_id == connection_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
