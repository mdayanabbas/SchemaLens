import uuid

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_connections_unauthorized(client: AsyncClient):
    response = await client.get("/api/v1/organizations/current/connections")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_connections_authorized(client: AsyncClient, token_headers_admin, org_id):
    response = await client.get(
        "/api/v1/organizations/current/connections",
        headers={**token_headers_admin, "X-Organization-ID": str(org_id)}
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_create_connection(client: AsyncClient, token_headers_admin, org_id):
    payload = {
        "name": "Test DB",
        "description": "A test database",
        "environment": "development",
        "dialect": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database_name": "testdb",
        "ssl_mode": "require",
        "secret_provider": "environment",
        "secret_reference": "TEST_DB_SECRET",
    }
    
    response = await client.post(
        "/api/v1/organizations/current/connections",
        headers={**token_headers_admin, "X-Organization-ID": str(org_id)},
        json=payload,
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test DB"
    assert "redacted_secret_reference" in data
    assert "TEST_DB_SECRET" not in data["redacted_secret_reference"]
    assert data["redacted_secret_reference"] != "TEST_DB_SECRET"
    
    conn_id = data["id"]
    
    # Verify policy was created
    policy_response = await client.get(
        f"/api/v1/organizations/current/connections/{conn_id}/policy",
        headers={**token_headers_admin, "X-Organization-ID": str(org_id)}
    )
    assert policy_response.status_code == 200
    policy_data = policy_response.json()
    assert policy_data["connection_id"] == conn_id
    assert policy_data["allow_query_execution"] is False


@pytest.mark.asyncio
async def test_create_production_connection_without_ssl(client: AsyncClient, token_headers_admin, org_id):
    payload = {
        "name": "Prod DB",
        "environment": "production",
        "dialect": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database_name": "proddb",
        "ssl_mode": "disable", # Invalid for prod
        "secret_provider": "environment",
        "secret_reference": "PROD_SECRET",
    }
    
    response = await client.post(
        "/api/v1/organizations/current/connections",
        headers={**token_headers_admin, "X-Organization-ID": str(org_id)},
        json=payload,
    )
    
    # Custom API validation throws 422 if configured with production_ssl check
    assert response.status_code == 422
