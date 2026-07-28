from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from app.connectors.types import WarningSeverity
from app.models.connection_enums import DatabaseDialect
from app.models.schema_snapshot_enums import (
    MatchType,
    NullsOrder,
    ReferentialAction,
    SchemaConstraintKind,
    SchemaObjectType,
    SchemaRelationKind,
    SortDirection,
)


@dataclass(frozen=True)
class IntrospectedNamespace:
    name: str
    comment: str | None = None


@dataclass(frozen=True)
class IntrospectedRelation:
    schema_name: str
    name: str
    kind: SchemaRelationKind
    comment: str | None = None
    estimated_rows: int | None = None
    is_partition: bool = False
    parent_schema_name: str | None = None
    parent_relation_name: str | None = None


@dataclass(frozen=True)
class IntrospectedColumn:
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
    is_nullable: bool = True
    has_default: bool = False
    default_expression: str | None = None
    default_expression_hash: str | None = None
    default_expression_truncated: bool = False
    is_identity: bool = False
    identity_generation: str | None = None
    is_generated: bool = False
    generation_expression_present: bool = False
    collation: str | None = None
    comment: str | None = None


@dataclass(frozen=True)
class IntrospectedConstraint:
    schema_name: str
    relation_name: str
    name: str
    kind: SchemaConstraintKind
    is_deferrable: bool = False
    initially_deferred: bool = False
    is_validated: bool = True
    check_expression: str | None = None
    check_expression_hash: str | None = None
    check_expression_truncated: bool = False
    foreign_schema_name: str | None = None
    foreign_relation_name: str | None = None
    foreign_constraint_name: str | None = None
    update_action: ReferentialAction | None = None
    delete_action: ReferentialAction | None = None
    match_type: MatchType | None = None


@dataclass(frozen=True)
class IntrospectedConstraintColumn:
    constraint_name: str
    schema_name: str
    relation_name: str
    column_name: str
    ordinal_position: int
    referenced_column_name: str | None = None


@dataclass(frozen=True)
class IntrospectedIndex:
    schema_name: str
    relation_name: str
    name: str
    is_unique: bool
    is_primary: bool
    is_valid: bool
    is_ready: bool
    access_method: str
    predicate_present: bool
    predicate_expression: str | None = None
    predicate_expression_hash: str | None = None
    predicate_expression_truncated: bool = False
    expression_index: bool = False
    estimated_size_bytes: int | None = None


@dataclass(frozen=True)
class IntrospectedIndexColumn:
    index_name: str
    schema_name: str
    relation_name: str
    ordinal_position: int
    column_name: str | None = None
    expression: str | None = None
    expression_hash: str | None = None
    expression_truncated: bool = False
    included: bool = False
    sort_direction: SortDirection | None = None
    nulls_order: NullsOrder | None = None


@dataclass(frozen=True)
class IntrospectedRoutine:
    schema_name: str
    name: str
    identity_arguments: str
    result_type: str
    routine_kind: str
    volatility: str
    parallel_safety: str
    security_definer: bool
    language: str


@dataclass(frozen=True)
class SchemaIntrospectionWarning:
    code: str
    message: str
    severity: WarningSeverity
    object_type: SchemaObjectType | None = None
    object_identifier: str | None = None


@dataclass(frozen=True)
class SchemaIntrospectionResult:
    dialect: DatabaseDialect
    server_version: str
    database_name: str
    approved_schemas: Sequence[str]
    namespaces: Sequence[IntrospectedNamespace] = field(default_factory=list)
    relations: Sequence[IntrospectedRelation] = field(default_factory=list)
    columns: Sequence[IntrospectedColumn] = field(default_factory=list)
    constraints: Sequence[IntrospectedConstraint] = field(default_factory=list)
    constraint_columns: Sequence[IntrospectedConstraintColumn] = field(default_factory=list)
    indexes: Sequence[IntrospectedIndex] = field(default_factory=list)
    index_columns: Sequence[IntrospectedIndexColumn] = field(default_factory=list)
    routines: Sequence[IntrospectedRoutine] = field(default_factory=list)
    warnings: Sequence[SchemaIntrospectionWarning] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    fingerprint_input_version: int = 1
