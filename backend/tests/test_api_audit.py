import uuid

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_audit_events_unauthorized(client: AsyncClient):
    # Missing org header
    response = await client.get("/api/v1/organizations/current/audit-events")
    assert response.status_code == 403

    # With org header but no token
    response = await client.get(
        "/api/v1/organizations/current/audit-events",
        headers={"X-Organization-ID": str(uuid.uuid4())}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_audit_events_forbidden(client: AsyncClient, token_headers, org_id):
    # Member without AUDIT_READ permission (assuming token is a basic member)
    response = await client.get(
        "/api/v1/organizations/current/audit-events",
        headers={**token_headers, "X-Organization-ID": str(org_id)}
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_audit_events_authorized(client: AsyncClient, token_headers_admin, org_id):
    # Need an admin token for AUDIT_READ
    response = await client.get(
        "/api/v1/organizations/current/audit-events",
        headers={**token_headers_admin, "X-Organization-ID": str(org_id)}
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "offset" in data
    assert "limit" in data
    
    # Verify that listing audit events generated an audit event for itself
    response2 = await client.get(
        "/api/v1/organizations/current/audit-events",
        headers={**token_headers_admin, "X-Organization-ID": str(org_id)}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    items = data2["items"]
    assert any(item["action"] == "audit.events_viewed" for item in items)
