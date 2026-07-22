from fastapi import APIRouter

from app.api.v1.endpoints import auth
from app.api.v1.endpoints import health
from app.api.v1.endpoints import memberships
from app.api.v1.endpoints import organizations

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(organizations.router, prefix="/organizations", tags=["Organizations"])
api_router.include_router(memberships.router, prefix="/organizations/current/members", tags=["Memberships"])
