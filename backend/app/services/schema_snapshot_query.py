import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.connection_schema_state import ConnectionSchemaState
from app.models.schema_snapshot import SchemaSnapshot
from app.models.schema_namespace import SchemaNamespace
from app.models.schema_relation import SchemaRelation
from app.models.schema_column import SchemaColumn
from app.models.schema_constraint import SchemaConstraint
from app.models.schema_index import SchemaIndex
from app.models.schema_routine import SchemaRoutine


class SchemaSnapshotQueryService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_connection_state(self, connection_id: uuid.UUID, organization_id: uuid.UUID) -> ConnectionSchemaState | None:
        stmt = select(ConnectionSchemaState).where(
            ConnectionSchemaState.connection_id == connection_id,
            ConnectionSchemaState.organization_id == organization_id
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
        
    async def get_snapshot(self, snapshot_id: uuid.UUID, organization_id: uuid.UUID) -> SchemaSnapshot | None:
        stmt = select(SchemaSnapshot).where(
            SchemaSnapshot.id == snapshot_id,
            SchemaSnapshot.organization_id == organization_id
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_namespaces(self, snapshot_id: uuid.UUID) -> Sequence[SchemaNamespace]:
        stmt = select(SchemaNamespace).where(SchemaNamespace.snapshot_id == snapshot_id).order_by(SchemaNamespace.name)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_relations(self, snapshot_id: uuid.UUID, schema_name: str | None = None) -> Sequence[SchemaRelation]:
        stmt = select(SchemaRelation).where(SchemaRelation.snapshot_id == snapshot_id)
        if schema_name:
            stmt = stmt.where(SchemaRelation.schema_name == schema_name)
        stmt = stmt.order_by(SchemaRelation.schema_name, SchemaRelation.name)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_columns(self, relation_id: uuid.UUID) -> Sequence[SchemaColumn]:
        stmt = select(SchemaColumn).where(SchemaColumn.relation_id == relation_id).order_by(SchemaColumn.ordinal_position)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_constraints(self, relation_id: uuid.UUID) -> Sequence[SchemaConstraint]:
        stmt = select(SchemaConstraint).options(
            selectinload(SchemaConstraint.columns)
        ).where(SchemaConstraint.relation_id == relation_id).order_by(SchemaConstraint.name)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_indexes(self, relation_id: uuid.UUID) -> Sequence[SchemaIndex]:
        stmt = select(SchemaIndex).options(
            selectinload(SchemaIndex.columns)
        ).where(SchemaIndex.relation_id == relation_id).order_by(SchemaIndex.name)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_routines(self, snapshot_id: uuid.UUID, schema_name: str | None = None) -> Sequence[SchemaRoutine]:
        stmt = select(SchemaRoutine).where(SchemaRoutine.snapshot_id == snapshot_id)
        if schema_name:
            stmt = stmt.where(SchemaRoutine.schema_name == schema_name)
        stmt = stmt.order_by(SchemaRoutine.schema_name, SchemaRoutine.name)
        result = await self.session.execute(stmt)
        return result.scalars().all()
