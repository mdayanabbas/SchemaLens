import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.schema_snapshot_enums import SchemaSnapshotStatus


class SchemaSnapshotResponse(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    connection_id: uuid.UUID
    schema_scan_id: uuid.UUID
    
    status: SchemaSnapshotStatus
    snapshot_version: int
    
    fingerprint: str | None
    server_version: str
    database_name: str
    
    namespace_count: int
    relation_count: int
    column_count: int
    constraint_count: int
    index_count: int
    routine_count: int
    warning_count: int
    metadata_size_bytes: int
    
    created_at: datetime
    finalized_at: datetime | None = None
    invalidated_at: datetime | None = None
    safe_invalid_reason_code: str | None = None

    model_config = ConfigDict(from_attributes=True)


class ConnectionSchemaStateResponse(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    connection_id: uuid.UUID
    
    current_snapshot_id: uuid.UUID | None
    previous_snapshot_id: uuid.UUID | None
    latest_scan_id: uuid.UUID | None
    
    current_fingerprint: str | None
    promoted_at: datetime | None
    
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
