import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.exceptions import ValidationError
from app.models.connection_enums import ApprovalMode
from app.services.connection_validation import validate_schema_lists


class ConnectionPolicyCreate(BaseModel):
    approved_schemas: list[str] = Field(default_factory=list)
    blocked_schemas: list[str] = Field(default_factory=lambda: ["pg_catalog", "information_schema"])
    allow_schema_scanning: bool = False
    allow_query_generation: bool = False
    allow_query_execution: bool = False
    approval_mode: ApprovalMode = ApprovalMode.ALWAYS
    max_statement_timeout_ms: int = Field(30000, gt=0)
    max_lock_timeout_ms: int = Field(5000, gt=0)
    max_rows: int = Field(1000, gt=0)
    max_response_bytes: int = Field(5242880, gt=0)
    max_estimated_rows: int = Field(100000, gt=0)
    max_estimated_cost: float = Field(10000.0, gt=0)
    max_joined_tables: int = Field(8, gt=0)
    max_subquery_depth: int = Field(5, gt=0)
    allow_system_catalogs: bool = False
    allow_cross_joins: bool = False
    require_fully_qualified_tables: bool = True

    @model_validator(mode="after")
    def validate_logic(self):
        self.approved_schemas, self.blocked_schemas = validate_schema_lists(
            self.approved_schemas, self.blocked_schemas
        )
        if self.allow_query_execution and not self.allow_query_generation:
            raise ValidationError("Cannot enable execution while generation is disabled.", code="INVALID_CONNECTION_POLICY")
        if self.allow_system_catalogs:
            raise ValidationError("System catalog access is not currently supported.", code="SYSTEM_CATALOG_ACCESS_NOT_SUPPORTED")
        return self


class ConnectionPolicyUpdate(BaseModel):
    approved_schemas: list[str] | None = None
    blocked_schemas: list[str] | None = None
    allow_schema_scanning: bool | None = None
    allow_query_generation: bool | None = None
    allow_query_execution: bool | None = None
    approval_mode: ApprovalMode | None = None
    max_statement_timeout_ms: int | None = Field(None, gt=0)
    max_lock_timeout_ms: int | None = Field(None, gt=0)
    max_rows: int | None = Field(None, gt=0)
    max_response_bytes: int | None = Field(None, gt=0)
    max_estimated_rows: int | None = Field(None, gt=0)
    max_estimated_cost: float | None = Field(None, gt=0)
    max_joined_tables: int | None = Field(None, gt=0)
    max_subquery_depth: int | None = Field(None, gt=0)
    allow_system_catalogs: bool | None = None
    allow_cross_joins: bool | None = None
    require_fully_qualified_tables: bool | None = None

    @model_validator(mode="after")
    def validate_logic(self):
        if self.approved_schemas is not None and self.blocked_schemas is not None:
            self.approved_schemas, self.blocked_schemas = validate_schema_lists(
                self.approved_schemas, self.blocked_schemas
            )
        # Note: cross-field validation for execution/generation/approval mode will need to be verified at the service layer 
        # since an update might only provide one of the fields while relying on the existing db state for the other.
        if self.allow_system_catalogs is True:
            raise ValidationError("System catalog access is not currently supported.", code="SYSTEM_CATALOG_ACCESS_NOT_SUPPORTED")
        return self


class ConnectionPolicyRead(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    connection_id: uuid.UUID
    approved_schemas_json: list[str]
    blocked_schemas_json: list[str]
    allow_schema_scanning: bool
    allow_query_generation: bool
    allow_query_execution: bool
    approval_mode: ApprovalMode
    max_statement_timeout_ms: int
    max_lock_timeout_ms: int
    max_rows: int
    max_response_bytes: int
    max_estimated_rows: int
    max_estimated_cost: float
    max_joined_tables: int
    max_subquery_depth: int
    allow_system_catalogs: bool
    allow_cross_joins: bool
    require_fully_qualified_tables: bool
    created_by_user_id: uuid.UUID
    updated_by_user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
