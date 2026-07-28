import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema_constraint import SchemaConstraint
from app.models.schema_relation import SchemaRelation
from app.models.schema_snapshot_enums import SchemaConstraintKind


class SchemaSnapshotValidationService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def validate_snapshot(self, snapshot_id: uuid.UUID) -> tuple[bool, str | None]:
        """
        Validates internal consistency of the persisted snapshot.
        For example, ensuring all foreign keys point to valid relations within the snapshot.
        Returns (is_valid, error_message).
        """
        # Validate foreign keys
        stmt = (
            select(SchemaConstraint)
            .where(
                SchemaConstraint.snapshot_id == snapshot_id,
                SchemaConstraint.kind == SchemaConstraintKind.FOREIGN_KEY
            )
        )
        result = await self.session.execute(stmt)
        fks = result.scalars().all()
        
        # Load all relations in snapshot
        rel_stmt = select(SchemaRelation.schema_name, SchemaRelation.name).where(SchemaRelation.snapshot_id == snapshot_id)
        rel_result = await self.session.execute(rel_stmt)
        relations_set = {(r.schema_name, r.name) for r in rel_result.all()}
        
        for fk in fks:
            if not fk.referenced_schema_name or not fk.referenced_relation_name:
                continue
                
            ref_key = (fk.referenced_schema_name, fk.referenced_relation_name)
            if ref_key not in relations_set:
                return False, f"Foreign key '{fk.name}' references missing relation '{fk.referenced_schema_name}.{fk.referenced_relation_name}'"
                
        return True, None
