import asyncio
import time
from typing import Awaitable, Callable, Sequence

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.engine import Row

from app.connectors.exceptions import ConnectorError
from app.connectors.introspection_types import (
    IntrospectedColumn,
    IntrospectedConstraint,
    IntrospectedConstraintColumn,
    IntrospectedIndex,
    IntrospectedIndexColumn,
    IntrospectedNamespace,
    IntrospectedRelation,
    IntrospectedRoutine,
    SchemaIntrospectionResult,
    SchemaIntrospectionWarning,
)
from app.connectors.postgres.constants import POSTGRESQL_SYSTEM_SCHEMAS, POSTGRESQL_SYSTEM_SCHEMA_PREFIXES
from app.connectors.postgres.expression_sanitizer import ExpressionSanitizer
from app.connectors.postgres.introspection_queries import (
    COLUMN_QUERY,
    CONSTRAINT_COLUMN_QUERY,
    CONSTRAINT_QUERY,
    INDEX_COLUMN_QUERY,
    INDEX_QUERY,
    NAMESPACE_QUERY,
    RELATION_QUERY,
    ROUTINE_QUERY,
)
from app.connectors.types import WarningSeverity
from app.core.config import Settings
from app.core.exceptions import AppError
from app.models.connection_enums import DatabaseDialect
from app.models.connection_policy import ConnectionPolicy
from app.models.schema_snapshot_enums import (
    MatchType,
    NullsOrder,
    ReferentialAction,
    SchemaConstraintKind,
    SchemaRelationKind,
    SortDirection,
)

logger = structlog.get_logger(__name__)


class PostgreSQLSchemaIntrospector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sanitizer = ExpressionSanitizer(
            max_length=settings.schema_introspection_max_default_expression_length
        )
        self.comment_sanitizer = ExpressionSanitizer(
            max_length=settings.schema_introspection_max_comment_length
        )

    async def introspect(
        self,
        *,
        engine: AsyncEngine,
        approved_schemas: Sequence[str],
        policy: ConnectionPolicy,
        cancellation_check: Callable[[], Awaitable[None]] | None,
        progress_callback: Callable[[str, int, int], Awaitable[None]] | None,
    ) -> SchemaIntrospectionResult:
        if not approved_schemas:
            raise AppError(
                code="NO_APPROVED_SCHEMAS",
                message="Cannot introspect without approved schemas.",
            )

        # 1. Validate schemas against limits
        effective_limit = min(
            self.settings.schema_introspection_max_schemas,
            policy.schema_scan_max_schemas or self.settings.schema_introspection_max_schemas,
        )
        
        filtered_schemas = []
        for s in approved_schemas:
            if s in POSTGRESQL_SYSTEM_SCHEMAS or any(s.startswith(p) for p in POSTGRESQL_SYSTEM_SCHEMA_PREFIXES):
                continue
            filtered_schemas.append(s)
            
        if len(filtered_schemas) > effective_limit:
            raise AppError(
                code="SCHEMA_INTROSPECTION_LIMIT_EXCEEDED",
                message=f"Requested {len(filtered_schemas)} schemas, exceeding the limit of {effective_limit}.",
            )

        result_builder = SchemaIntrospectionResult(
            dialect=DatabaseDialect.POSTGRESQL,
            server_version="",
            database_name="",
            approved_schemas=filtered_schemas,
            started_at=time.time(), # Will be converted to datetime upstream if needed, but dataclass expects datetime, let's fix that. Wait, time.time() returns float. Let's use datetime.
            # I will fix datetime usage below.
        )
        
        from datetime import datetime, timezone
        result_builder = SchemaIntrospectionResult(
            dialect=DatabaseDialect.POSTGRESQL,
            server_version="",
            database_name="",
            approved_schemas=filtered_schemas,
            started_at=datetime.now(timezone.utc),
        )

        namespaces = []
        relations = []
        columns = []
        constraints = []
        constraint_columns = []
        indexes = []
        index_columns = []
        routines = []
        warnings = []

        total_stages = 8
        current_stage = 0

        async def _check_cancel():
            if cancellation_check:
                await cancellation_check()

        async def _progress(phase: str):
            nonlocal current_stage
            current_stage += 1
            if progress_callback:
                await progress_callback(phase, current_stage, total_stages)

        await _check_cancel()
        await _progress("validating")

        try:
            async with engine.connect() as conn:
                async with conn.begin():
                    # Set safe config
                    await conn.execute(text(f"SET statement_timeout = {self.settings.schema_introspection_statement_timeout_ms}"))
                    await conn.execute(text(f"SET lock_timeout = {self.settings.schema_introspection_lock_timeout_ms}"))
                    await conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
                    
                    db_name_res = await conn.execute(text("SELECT current_database()"))
                    database_name = db_name_res.scalar()
                    
                    version_res = await conn.execute(text("SHOW server_version"))
                    server_version = version_res.scalar()

                    # Phase 1: Namespaces (Required)
                    await _check_cancel()
                    await _progress("namespaces")
                    res = await conn.execute(NAMESPACE_QUERY, {"schemas": filtered_schemas})
                    for row in res:
                        comment, _, _ = self.comment_sanitizer.sanitize(row.comment)
                        namespaces.append(IntrospectedNamespace(name=row.name, comment=comment))
                    
                    # Phase 2: Relations (Required)
                    await _check_cancel()
                    await _progress("relations")
                    res = await conn.execute(RELATION_QUERY, {"schemas": filtered_schemas})
                    for row in res:
                        if len(relations) >= self.settings.schema_introspection_max_relations:
                            raise AppError("SCHEMA_INTROSPECTION_LIMIT_EXCEEDED", "Relation limit exceeded.")
                        comment, _, _ = self.comment_sanitizer.sanitize(row.comment)
                        
                        kind_map = {
                            'r': SchemaRelationKind.TABLE,
                            'p': SchemaRelationKind.PARTITIONED_TABLE,
                            'v': SchemaRelationKind.VIEW,
                            'm': SchemaRelationKind.MATERIALIZED_VIEW,
                            'f': SchemaRelationKind.FOREIGN_TABLE,
                        }
                        
                        relations.append(IntrospectedRelation(
                            schema_name=row.schema_name,
                            name=row.name,
                            kind=kind_map.get(row.kind, SchemaRelationKind.TABLE),
                            comment=comment,
                            estimated_rows=row.estimated_rows,
                            is_partition=row.is_partition,
                            parent_schema_name=row.parent_schema_name,
                            parent_relation_name=row.parent_relation_name
                        ))

                    # Phase 3: Columns (Required)
                    await _check_cancel()
                    await _progress("columns")
                    res = await conn.execute(COLUMN_QUERY, {"schemas": filtered_schemas})
                    for row in res:
                        if len(columns) >= self.settings.schema_introspection_max_columns:
                            raise AppError("SCHEMA_INTROSPECTION_LIMIT_EXCEEDED", "Column limit exceeded.")
                        
                        def_expr, def_hash, def_trunc = self.sanitizer.sanitize(row.default_expression)
                        comment, _, _ = self.comment_sanitizer.sanitize(row.comment)
                        
                        columns.append(IntrospectedColumn(
                            schema_name=row.schema_name,
                            relation_name=row.relation_name,
                            name=row.name,
                            ordinal_position=row.ordinal_position,
                            formatted_data_type=row.formatted_data_type,
                            base_data_type=row.base_data_type,
                            character_maximum_length=row.character_maximum_length,
                            numeric_precision=row.numeric_precision,
                            numeric_scale=row.numeric_scale,
                            datetime_precision=row.datetime_precision,
                            is_nullable=row.is_nullable,
                            has_default=row.has_default,
                            default_expression=def_expr,
                            default_expression_hash=def_hash,
                            default_expression_truncated=def_trunc,
                            is_identity=row.is_identity,
                            identity_generation=row.identity_generation,
                            is_generated=row.is_generated,
                            generation_expression_present=row.generation_expression_present,
                            collation=row.collation,
                            comment=comment
                        ))

                    # Phase 4: Constraints (Required)
                    await _check_cancel()
                    await _progress("constraints")
                    res = await conn.execute(CONSTRAINT_QUERY, {"schemas": filtered_schemas})
                    for row in res:
                        if len(constraints) >= self.settings.schema_introspection_max_constraints:
                            raise AppError("SCHEMA_INTROSPECTION_LIMIT_EXCEEDED", "Constraint limit exceeded.")
                            
                        chk_expr, chk_hash, chk_trunc = self.sanitizer.sanitize(row.check_expression)
                        
                        kind_map = {
                            'p': SchemaConstraintKind.PRIMARY_KEY,
                            'u': SchemaConstraintKind.UNIQUE,
                            'f': SchemaConstraintKind.FOREIGN_KEY,
                            'c': SchemaConstraintKind.CHECK,
                            'x': SchemaConstraintKind.EXCLUSION,
                        }
                        
                        ref_action_map = {
                            'a': ReferentialAction.NO_ACTION,
                            'r': ReferentialAction.RESTRICT,
                            'c': ReferentialAction.CASCADE,
                            'n': ReferentialAction.SET_NULL,
                            'd': ReferentialAction.SET_DEFAULT,
                        }
                        
                        match_type_map = {
                            'f': MatchType.FULL,
                            'p': MatchType.PARTIAL,
                            's': MatchType.SIMPLE,
                        }
                        
                        constraints.append(IntrospectedConstraint(
                            schema_name=row.schema_name,
                            relation_name=row.relation_name,
                            name=row.name,
                            kind=kind_map.get(row.kind, SchemaConstraintKind.CHECK),
                            is_deferrable=row.is_deferrable,
                            initially_deferred=row.initially_deferred,
                            is_validated=row.is_validated,
                            check_expression=chk_expr,
                            check_expression_hash=chk_hash,
                            check_expression_truncated=chk_trunc,
                            foreign_schema_name=row.foreign_schema_name,
                            foreign_relation_name=row.foreign_relation_name,
                            foreign_constraint_name=row.foreign_constraint_name,
                            update_action=ref_action_map.get(row.update_action),
                            delete_action=ref_action_map.get(row.delete_action),
                            match_type=match_type_map.get(row.match_type),
                        ))

                    # Phase 5: Constraint Columns (Required)
                    res = await conn.execute(CONSTRAINT_COLUMN_QUERY, {"schemas": filtered_schemas})
                    for row in res:
                        constraint_columns.append(IntrospectedConstraintColumn(
                            constraint_name=row.constraint_name,
                            schema_name=row.schema_name,
                            relation_name=row.relation_name,
                            column_name=row.column_name,
                            ordinal_position=row.ordinal_position,
                            referenced_column_name=row.referenced_column_name
                        ))

                    # Phase 6: Indexes (Optional)
                    await _check_cancel()
                    await _progress("indexes")
                    try:
                        res = await conn.execute(INDEX_QUERY, {"schemas": filtered_schemas})
                        for row in res:
                            if len(indexes) >= self.settings.schema_introspection_max_indexes:
                                raise AppError("SCHEMA_INTROSPECTION_LIMIT_EXCEEDED", "Index limit exceeded.")
                            
                            pred_expr, pred_hash, pred_trunc = self.sanitizer.sanitize(row.predicate_expression)
                            indexes.append(IntrospectedIndex(
                                schema_name=row.schema_name,
                                relation_name=row.relation_name,
                                name=row.name,
                                is_unique=row.is_unique,
                                is_primary=row.is_primary,
                                is_valid=row.is_valid,
                                is_ready=row.is_ready,
                                access_method=row.access_method,
                                predicate_present=row.predicate_present,
                                predicate_expression=pred_expr,
                                predicate_expression_hash=pred_hash,
                                predicate_expression_truncated=pred_trunc,
                                expression_index=row.expression_index,
                                estimated_size_bytes=row.estimated_size_bytes
                            ))
                            
                        res = await conn.execute(INDEX_COLUMN_QUERY, {"schemas": filtered_schemas})
                        for row in res:
                            if len(index_columns) >= self.settings.schema_introspection_max_index_columns:
                                raise AppError("SCHEMA_INTROSPECTION_LIMIT_EXCEEDED", "Index column limit exceeded.")
                                
                            expr, expr_hash, expr_trunc = self.sanitizer.sanitize(row.expression)
                            
                            dir_map = {
                                'ascending': SortDirection.ASCENDING,
                                'descending': SortDirection.DESCENDING,
                            }
                            nulls_map = {
                                'first': NullsOrder.FIRST,
                                'last': NullsOrder.LAST,
                            }
                            
                            index_columns.append(IntrospectedIndexColumn(
                                index_name=row.index_name,
                                schema_name=row.schema_name,
                                relation_name=row.relation_name,
                                ordinal_position=row.ordinal_position,
                                column_name=row.column_name,
                                expression=expr,
                                expression_hash=expr_hash,
                                expression_truncated=expr_trunc,
                                included=row.included,
                                sort_direction=dir_map.get(row.sort_direction),
                                nulls_order=nulls_map.get(row.nulls_order)
                            ))
                    except Exception as e:
                        logger.warning("optional_stage_failed", stage="indexes", error=str(e))
                        warnings.append(SchemaIntrospectionWarning(
                            code="SCHEMA_OPTIONAL_STAGE_FAILED",
                            message="Failed to retrieve indexes.",
                            severity=WarningSeverity.WARNING
                        ))
                        # Clear partially fetched optional data
                        indexes = []
                        index_columns = []

                    # Phase 7: Routines (Optional)
                    await _check_cancel()
                    await _progress("routines")
                    try:
                        res = await conn.execute(ROUTINE_QUERY, {"schemas": filtered_schemas})
                        for row in res:
                            if len(routines) >= self.settings.schema_introspection_max_routines:
                                raise AppError("SCHEMA_INTROSPECTION_LIMIT_EXCEEDED", "Routine limit exceeded.")
                            
                            routines.append(IntrospectedRoutine(
                                schema_name=row.schema_name,
                                name=row.name,
                                identity_arguments=row.identity_arguments,
                                result_type=row.result_type,
                                routine_kind=row.routine_kind,
                                volatility=row.volatility,
                                parallel_safety=row.parallel_safety,
                                security_definer=row.security_definer,
                                language=row.language
                            ))
                    except Exception as e:
                        logger.warning("optional_stage_failed", stage="routines", error=str(e))
                        warnings.append(SchemaIntrospectionWarning(
                            code="SCHEMA_OPTIONAL_STAGE_FAILED",
                            message="Failed to retrieve routines.",
                            severity=WarningSeverity.WARNING
                        ))
                        routines = []

                    await _check_cancel()
                    await _progress("finalizing_metadata")

        except AppError:
            raise
        except Exception as e:
            if "cancel" in str(e).lower() or "timeout" in str(e).lower():
                raise AppError("SCHEMA_INTROSPECTION_TIMEOUT", "Introspection query timed out or was cancelled.")
            logger.error("schema_introspection_failed", error=str(e))
            raise AppError("SCHEMA_INTROSPECTION_FAILED", "Failed to retrieve schema metadata.")

        return SchemaIntrospectionResult(
            dialect=DatabaseDialect.POSTGRESQL,
            server_version=server_version,
            database_name=database_name,
            approved_schemas=filtered_schemas,
            namespaces=namespaces,
            relations=relations,
            columns=columns,
            constraints=constraints,
            constraint_columns=constraint_columns,
            indexes=indexes,
            index_columns=index_columns,
            routines=routines,
            warnings=warnings,
            started_at=result_builder.started_at,
            completed_at=datetime.now(timezone.utc),
            fingerprint_input_version=1
        )
