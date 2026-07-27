from typing import Mapping

from app.core.config import get_settings
from app.core.exceptions import ConflictError, PolicyViolationError, ValidationError
from app.models.connection_enums import ConnectionStatus, ConnectionTestStatus, DatabaseDialect
from app.models.connection_policy import ConnectionPolicy
from app.models.database_connection import DatabaseConnection
from app.models.schema_scan import SchemaScan


class SchemaScanValidator:
    def __init__(self):
        self.settings = get_settings()

    def validate_scan_eligibility(
        self,
        *,
        connection: DatabaseConnection,
        policy: ConnectionPolicy,
        active_scan: SchemaScan | None,
        requested_schemas: list[str] | None,
    ) -> list[str]:
        """
        Validates scan eligibility and returns the effective normalized list of schemas to scan.
        """
        if connection.status != ConnectionStatus.ACTIVE:
            raise ConflictError(
                message="Connection is disabled or in draft state.",
                code="CONNECTION_DISABLED",
            )

        if connection.last_test_status != ConnectionTestStatus.SUCCEEDED:
            raise ConflictError(
                message="Connection must be tested successfully before scanning.",
                code="CONNECTION_NOT_TESTED",
            )

        # Dialect verification
        if connection.dialect not in [DatabaseDialect.POSTGRESQL]:
            raise ValidationError(
                message=f"Dialect {connection.dialect} is not supported for scanning.",
                code="UNSUPPORTED_DIALECT"
            )

        if not policy.allow_schema_scanning:
            raise PolicyViolationError(
                message="Schema scanning is disabled by connection policy.",
                code="SCHEMA_SCANNING_DISABLED",
            )

        approved_schemas = policy.approved_schemas_json or []
        if not approved_schemas:
            raise PolicyViolationError(
                message="No schemas are approved for scanning in the connection policy.",
                code="NO_APPROVED_SCHEMAS",
            )

        blocked_schemas = set(policy.blocked_schemas_json or [])
        
        # System schemas should always be blocked. Let's add postgres defaults.
        if connection.dialect == DatabaseDialect.POSTGRESQL:
            blocked_schemas.update({"pg_catalog", "information_schema", "pg_toast"})

        # Resolve requested schemas
        if not requested_schemas:
            # Empty list means "all currently approved schemas"
            effective_schemas = [s for s in approved_schemas if s not in blocked_schemas]
        else:
            effective_schemas = []
            for schema in requested_schemas:
                if schema in blocked_schemas:
                    raise PolicyViolationError(
                        message=f"Requested schema '{schema}' is explicitly blocked.",
                        code="REQUESTED_SCHEMA_BLOCKED",
                    )
                if schema not in approved_schemas:
                    raise PolicyViolationError(
                        message=f"Requested schema '{schema}' is not in the approved list.",
                        code="REQUESTED_SCHEMA_NOT_APPROVED",
                    )
                effective_schemas.append(schema)

        if not effective_schemas:
            raise PolicyViolationError(
                message="The effective list of schemas to scan is empty after applying policy blocks.",
                code="NO_ELIGIBLE_SCHEMAS",
            )
            
        if len(effective_schemas) > self.settings.schema_scan_max_requested_schemas:
            raise ConflictError(
                message=f"Effective schema list exceeds max limit of {self.settings.schema_scan_max_requested_schemas}.",
                code="SCHEMA_SCAN_LIMIT_EXCEEDED"
            )

        if active_scan:
            raise ConflictError(
                message="An active schema scan already exists for this connection.",
                code="ACTIVE_SCHEMA_SCAN_EXISTS",
                details={"active_scan_id": str(active_scan.id)}
            )

        return effective_schemas
