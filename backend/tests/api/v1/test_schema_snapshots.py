import uuid
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.connection_schema_state import ConnectionSchemaState
from app.models.schema_snapshot import SchemaSnapshot
from app.models.schema_snapshot_enums import SchemaSnapshotStatus

pytestmark = pytest.mark.asyncio

async def test_get_connection_schema_state_not_found(
    client: AsyncClient,
    normal_user_headers: dict[str, str]
):
    conn_id = uuid.uuid4()
    response = await client.get(f"/api/v1/organizations/current/connections/{conn_id}/schema-state", headers=normal_user_headers)
    assert response.status_code == 404

async def test_get_connection_schema_state(
    client: AsyncClient,
    normal_user_headers: dict[str, str],
    normal_user_organization_id: uuid.UUID,
    db_session: AsyncSession
):
    conn_id = uuid.uuid4()
    state = ConnectionSchemaState(
        organization_id=normal_user_organization_id,
        connection_id=conn_id,
    )
    db_session.add(state)
    await db_session.commit()
    
    response = await client.get(f"/api/v1/organizations/current/connections/{conn_id}/schema-state", headers=normal_user_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["connection_id"] == str(conn_id)

async def test_get_schema_snapshot(
    client: AsyncClient,
    normal_user_headers: dict[str, str],
    normal_user_organization_id: uuid.UUID,
    db_session: AsyncSession
):
    conn_id = uuid.uuid4()
    scan_id = uuid.uuid4()
    snapshot = SchemaSnapshot(
        organization_id=normal_user_organization_id,
        connection_id=conn_id,
        schema_scan_id=scan_id,
        status=SchemaSnapshotStatus.READY,
        snapshot_version=1,
        server_version="15.0",
        database_name="test_db"
    )
    db_session.add(snapshot)
    await db_session.commit()
    
    response = await client.get(f"/api/v1/organizations/current/snapshots/{snapshot.id}", headers=normal_user_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(snapshot.id)
    assert data["database_name"] == "test_db"
