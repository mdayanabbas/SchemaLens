import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.schema_snapshot_enums import (
    MatchType,
    NullsOrder,
    ReferentialAction,
    SchemaConstraintKind,
    SchemaRelationKind,
    SortDirection,
)


class SchemaNamespaceResponse(BaseModel):
    id: uuid.UUID
    snapshot_id: uuid.UUID
    name: str
    comment: str | None = None
    
    model_config = ConfigDict(from_attributes=True)


class SchemaRelationResponse(BaseModel):
    id: uuid.UUID
    snapshot_id: uuid.UUID
    namespace_id: uuid.UUID
    schema_name: str
    name: str
    qualified_name: str
    kind: SchemaRelationKind
    comment: str | None = None
    estimated_rows: int | None = None
    is_partition: bool
    parent_schema_name: str | None = None
    parent_relation_name: str | None = None
    
    model_config = ConfigDict(from_attributes=True)


class SchemaColumnResponse(BaseModel):
    id: uuid.UUID
    snapshot_id: uuid.UUID
    relation_id: uuid.UUID
    schema_name: str
    relation_name: str
    name: str
    ordinal_position: int
    formatted_data_type: str
    base_data_type: str
    character_maximum_length: int | None = None
    numeric_precision: int | None = None
    numeric_scale: int | None = None
    datetime_precision: int | None = None
    is_nullable: bool
    has_default: bool
    default_expression: str | None = None
    default_expression_truncated: bool
    is_identity: bool
    identity_generation: str | None = None
    is_generated: bool
    collation: str | None = None
    comment: str | None = None
    
    model_config = ConfigDict(from_attributes=True)


class SchemaConstraintColumnResponse(BaseModel):
    id: uuid.UUID
    constraint_id: uuid.UUID
    column_id: uuid.UUID
    ordinal_position: int
    referenced_column_id: uuid.UUID | None = None
    referenced_column_name: str | None = None
    
    model_config = ConfigDict(from_attributes=True)


class SchemaConstraintResponse(BaseModel):
    id: uuid.UUID
    snapshot_id: uuid.UUID
    relation_id: uuid.UUID
    name: str
    kind: SchemaConstraintKind
    is_deferrable: bool
    initially_deferred: bool
    is_validated: bool
    check_expression: str | None = None
    check_expression_truncated: bool
    referenced_schema_name: str | None = None
    referenced_relation_name: str | None = None
    referenced_relation_id: uuid.UUID | None = None
    update_action: ReferentialAction | None = None
    delete_action: ReferentialAction | None = None
    match_type: MatchType | None = None
    columns: list[SchemaConstraintColumnResponse] = []
    
    model_config = ConfigDict(from_attributes=True)


class SchemaIndexColumnResponse(BaseModel):
    id: uuid.UUID
    index_id: uuid.UUID
    column_id: uuid.UUID | None = None
    ordinal_position: int
    expression: str | None = None
    expression_truncated: bool
    included: bool
    sort_direction: SortDirection | None = None
    nulls_order: NullsOrder | None = None
    
    model_config = ConfigDict(from_attributes=True)


class SchemaIndexResponse(BaseModel):
    id: uuid.UUID
    snapshot_id: uuid.UUID
    relation_id: uuid.UUID
    name: str
    is_unique: bool
    is_primary: bool
    is_valid: bool
    is_ready: bool
    access_method: str
    predicate_present: bool
    predicate_expression: str | None = None
    predicate_expression_truncated: bool
    expression_index: bool
    estimated_size_bytes: int | None = None
    columns: list[SchemaIndexColumnResponse] = []
    
    model_config = ConfigDict(from_attributes=True)


class SchemaRoutineResponse(BaseModel):
    id: uuid.UUID
    snapshot_id: uuid.UUID
    namespace_id: uuid.UUID
    schema_name: str
    name: str
    identity_arguments: str
    result_type: str
    routine_kind: str
    volatility: str
    parallel_safety: str
    security_definer: bool
    language: str
    
    model_config = ConfigDict(from_attributes=True)
