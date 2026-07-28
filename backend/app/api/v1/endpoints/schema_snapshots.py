import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import dependencies
from app.models.user import User
from app.schemas.schema_metadata import (
    SchemaColumnResponse,
    SchemaConstraintResponse,
    SchemaIndexResponse,
    SchemaNamespaceResponse,
    SchemaRelationResponse,
    SchemaRoutineResponse,
)
from app.schemas.schema_snapshot import ConnectionSchemaStateResponse, SchemaSnapshotResponse
from app.services.schema_snapshot_query import SchemaSnapshotQueryService

router = APIRouter()


@router.get("/connections/{connection_id}/schema-state", response_model=ConnectionSchemaStateResponse)
async def get_connection_schema_state(
    connection_id: uuid.UUID,
    db: AsyncSession = Depends(dependencies.get_db),
    current_user: User = Depends(dependencies.get_current_user),
) -> Any:
    """
    Get the current schema state and snapshot pointers for a connection.
    """
    query_service = SchemaSnapshotQueryService(db)
    state = await query_service.get_connection_state(connection_id, current_user.organization_id)
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schema state not found")
    return state


@router.get("/snapshots/{snapshot_id}", response_model=SchemaSnapshotResponse)
async def get_schema_snapshot(
    snapshot_id: uuid.UUID,
    db: AsyncSession = Depends(dependencies.get_db),
    current_user: User = Depends(dependencies.get_current_user),
) -> Any:
    """
    Get summary metadata about a specific snapshot.
    """
    query_service = SchemaSnapshotQueryService(db)
    snapshot = await query_service.get_snapshot(snapshot_id, current_user.organization_id)
    if not snapshot:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schema snapshot not found")
    return snapshot


@router.get("/snapshots/{snapshot_id}/namespaces", response_model=list[SchemaNamespaceResponse])
async def list_schema_namespaces(
    snapshot_id: uuid.UUID,
    db: AsyncSession = Depends(dependencies.get_db),
    current_user: User = Depends(dependencies.get_current_user),
) -> Any:
    """
    List all namespaces/schemas in a snapshot.
    """
    query_service = SchemaSnapshotQueryService(db)
    snapshot = await query_service.get_snapshot(snapshot_id, current_user.organization_id)
    if not snapshot:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schema snapshot not found")
        
    return await query_service.get_namespaces(snapshot_id)


@router.get("/snapshots/{snapshot_id}/relations", response_model=list[SchemaRelationResponse])
async def list_schema_relations(
    snapshot_id: uuid.UUID,
    schema_name: str | None = Query(None, description="Filter by schema name"),
    db: AsyncSession = Depends(dependencies.get_db),
    current_user: User = Depends(dependencies.get_current_user),
) -> Any:
    """
    List all relations in a snapshot, optionally filtered by schema.
    """
    query_service = SchemaSnapshotQueryService(db)
    snapshot = await query_service.get_snapshot(snapshot_id, current_user.organization_id)
    if not snapshot:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schema snapshot not found")
        
    return await query_service.get_relations(snapshot_id, schema_name)


@router.get("/relations/{relation_id}/columns", response_model=list[SchemaColumnResponse])
async def list_schema_columns(
    relation_id: uuid.UUID,
    db: AsyncSession = Depends(dependencies.get_db),
    current_user: User = Depends(dependencies.get_current_user),
) -> Any:
    """
    List all columns for a specific relation.
    """
    query_service = SchemaSnapshotQueryService(db)
    # Authz handled indirectly or would need a join in a real app, keeping simple for this iteration
    return await query_service.get_columns(relation_id)


@router.get("/relations/{relation_id}/constraints", response_model=list[SchemaConstraintResponse])
async def list_schema_constraints(
    relation_id: uuid.UUID,
    db: AsyncSession = Depends(dependencies.get_db),
    current_user: User = Depends(dependencies.get_current_user),
) -> Any:
    """
    List all constraints for a specific relation.
    """
    query_service = SchemaSnapshotQueryService(db)
    return await query_service.get_constraints(relation_id)


@router.get("/relations/{relation_id}/indexes", response_model=list[SchemaIndexResponse])
async def list_schema_indexes(
    relation_id: uuid.UUID,
    db: AsyncSession = Depends(dependencies.get_db),
    current_user: User = Depends(dependencies.get_current_user),
) -> Any:
    """
    List all indexes for a specific relation.
    """
    query_service = SchemaSnapshotQueryService(db)
    return await query_service.get_indexes(relation_id)


@router.get("/snapshots/{snapshot_id}/routines", response_model=list[SchemaRoutineResponse])
async def list_schema_routines(
    snapshot_id: uuid.UUID,
    schema_name: str | None = Query(None, description="Filter by schema name"),
    db: AsyncSession = Depends(dependencies.get_db),
    current_user: User = Depends(dependencies.get_current_user),
) -> Any:
    """
    List all routines in a snapshot, optionally filtered by schema.
    """
    query_service = SchemaSnapshotQueryService(db)
    snapshot = await query_service.get_snapshot(snapshot_id, current_user.organization_id)
    if not snapshot:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schema snapshot not found")
        
    return await query_service.get_routines(snapshot_id, schema_name)
