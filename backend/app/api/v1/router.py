from fastapi import APIRouter

from app.api.v1.endpoints import auth
from app.api.v1.endpoints import health
from app.api.v1.endpoints import memberships
from app.api.v1.endpoints import organizations
from app.api.v1.endpoints import audit
from app.api.v1.endpoints import connections
from app.api.v1.endpoints import secrets
from app.api.v1.endpoints import schema_scans
from app.api.v1.endpoints import schema_snapshots

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(organizations.router, prefix="/organizations", tags=["Organizations"])
api_router.include_router(memberships.router, prefix="/organizations/current/members", tags=["Memberships"])
api_router.include_router(audit.router, prefix="/organizations/current/audit-events", tags=["Audit"])
api_router.include_router(connections.router, prefix="/organizations/current/connections", tags=["Connections"])
api_router.include_router(secrets.router, prefix="/organizations/current/secrets", tags=["Secrets"])
api_router.include_router(schema_scans.router, prefix="/organizations/current", tags=["Schema Scans"])
api_router.include_router(schema_snapshots.router, prefix="/organizations/current", tags=["Schema Snapshots"])
