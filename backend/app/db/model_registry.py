"""
Database model registry.
Import all SQLAlchemy declarative models here so Alembic can discover them.
"""

from app.db.base import Base

from app.models import (
    AuditEvent,
    AuthenticationEvent,
    Organization,
    OrganizationMembership,
    RefreshToken,
    User,
    DatabaseConnection,
    ConnectionPolicy,
    StoredSecret,
    SchemaScan,
    SchemaScanTransition,
    SchemaSnapshot,
    SchemaNamespace,
    SchemaRelation,
    SchemaColumn,
    SchemaConstraint,
    SchemaConstraintColumn,
    SchemaIndex,
    SchemaIndexColumn,
    SchemaRoutine,
    ConnectionSchemaState,
)

target_metadata = Base.metadata
