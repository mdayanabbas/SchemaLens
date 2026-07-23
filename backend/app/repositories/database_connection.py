import uuid

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.connection_enums import ConnectionEnvironment, ConnectionStatus, DatabaseDialect
from app.models.database_connection import DatabaseConnection


class DatabaseConnectionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id_for_organization(
        self, connection_id: uuid.UUID, organization_id: uuid.UUID
    ) -> DatabaseConnection | None:
        """Get a connection by ID, ensuring it belongs to the given organization."""
        stmt = select(DatabaseConnection).where(
            and_(
                DatabaseConnection.id == connection_id,
                DatabaseConnection.organization_id == organization_id,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_name_for_organization(
        self, name: str, organization_id: uuid.UUID
    ) -> DatabaseConnection | None:
        """Get a connection by exact name within an organization."""
        stmt = select(DatabaseConnection).where(
            and_(
                DatabaseConnection.name == name,
                DatabaseConnection.organization_id == organization_id,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def name_exists_for_organization(self, name: str, organization_id: uuid.UUID) -> bool:
        """Check if a connection name is already in use within the organization."""
        stmt = select(1).where(
            and_(
                DatabaseConnection.name == name,
                DatabaseConnection.organization_id == organization_id,
            )
        )
        result = await self.session.execute(stmt)
        return result.first() is not None

    async def list_for_organization(
        self,
        organization_id: uuid.UUID,
        *,
        offset: int,
        limit: int,
        environment: ConnectionEnvironment | None = None,
        status: ConnectionStatus | None = None,
        dialect: DatabaseDialect | None = None,
    ) -> list[DatabaseConnection]:
        """List connections for an organization with optional filtering."""
        stmt = select(DatabaseConnection).where(
            DatabaseConnection.organization_id == organization_id
        )
        
        if environment:
            stmt = stmt.where(DatabaseConnection.environment == environment)
        if status:
            stmt = stmt.where(DatabaseConnection.status == status)
        if dialect:
            stmt = stmt.where(DatabaseConnection.dialect == dialect)
            
        # Deterministic ordering: updated_at descending, then ID as tie-breaker
        stmt = stmt.order_by(DatabaseConnection.updated_at.desc(), DatabaseConnection.id.asc())
        stmt = stmt.offset(offset).limit(limit)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_for_organization(
        self,
        organization_id: uuid.UUID,
        *,
        environment: ConnectionEnvironment | None = None,
        status: ConnectionStatus | None = None,
        dialect: DatabaseDialect | None = None,
    ) -> int:
        """Count connections for an organization with optional filtering."""
        stmt = select(func.count()).select_from(DatabaseConnection).where(
            DatabaseConnection.organization_id == organization_id
        )
        
        if environment:
            stmt = stmt.where(DatabaseConnection.environment == environment)
        if status:
            stmt = stmt.where(DatabaseConnection.status == status)
        if dialect:
            stmt = stmt.where(DatabaseConnection.dialect == dialect)
            
        result = await self.session.execute(stmt)
        return result.scalar_one()

    def add(self, connection: DatabaseConnection) -> None:
        """Add a connection to the session without committing."""
        self.session.add(connection)

    async def flush(self) -> None:
        """Flush pending changes to the database."""
        await self.session.flush()
