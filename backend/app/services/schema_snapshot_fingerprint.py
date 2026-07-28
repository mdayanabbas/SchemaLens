import hashlib
import json
from typing import Any

from app.connectors.introspection_types import SchemaIntrospectionResult

class SchemaSnapshotFingerprintService:
    @staticmethod
    def compute_fingerprint(result: SchemaIntrospectionResult) -> str:
        """
        Computes a deterministic SHA-256 fingerprint for the introspected schema state.
        This is used to detect if the schema has changed between scans.
        """
        # We need a stable representation of the schema.
        # We will build a nested dictionary and serialize it to JSON with sorted keys.
        
        state: dict[str, Any] = {
            "version": result.fingerprint_input_version,
            "namespaces": [],
            "relations": [],
            "columns": [],
            "constraints": [],
            "indexes": [],
            "routines": [],
        }

        # Namespaces
        for ns in sorted(result.namespaces, key=lambda x: x.name):
            state["namespaces"].append({
                "name": ns.name,
                "comment": ns.comment,
            })

        # Relations
        for rel in sorted(result.relations, key=lambda x: (x.schema_name, x.name)):
            state["relations"].append({
                "schema_name": rel.schema_name,
                "name": rel.name,
                "kind": rel.kind.value,
                "comment": rel.comment,
                # We do NOT include estimated_rows in the fingerprint, because it changes frequently 
                # without the actual schema changing.
                "is_partition": rel.is_partition,
                "parent_schema_name": rel.parent_schema_name,
                "parent_relation_name": rel.parent_relation_name,
            })

        # Columns
        for col in sorted(result.columns, key=lambda x: (x.schema_name, x.relation_name, x.ordinal_position)):
            state["columns"].append({
                "schema_name": col.schema_name,
                "relation_name": col.relation_name,
                "name": col.name,
                "ordinal_position": col.ordinal_position,
                "formatted_data_type": col.formatted_data_type,
                "is_nullable": col.is_nullable,
                "has_default": col.has_default,
                "default_expression_hash": col.default_expression_hash,
                "is_identity": col.is_identity,
                "is_generated": col.is_generated,
                "collation": col.collation,
                "comment": col.comment,
            })

        # Constraints
        # For constraints, we also need to include their columns. 
        # We'll group columns by constraint.
        constraint_cols = {}
        for cc in result.constraint_columns:
            key = (cc.schema_name, cc.relation_name, cc.constraint_name)
            if key not in constraint_cols:
                constraint_cols[key] = []
            constraint_cols[key].append(cc)
            
        for con in sorted(result.constraints, key=lambda x: (x.schema_name, x.relation_name, x.name)):
            cols = constraint_cols.get((con.schema_name, con.relation_name, con.name), [])
            cols_state = [
                {
                    "column_name": c.column_name,
                    "ordinal_position": c.ordinal_position,
                    "referenced_column_name": c.referenced_column_name,
                }
                for c in sorted(cols, key=lambda x: x.ordinal_position)
            ]
            
            state["constraints"].append({
                "schema_name": con.schema_name,
                "relation_name": con.relation_name,
                "name": con.name,
                "kind": con.kind.value,
                "is_deferrable": con.is_deferrable,
                "check_expression_hash": con.check_expression_hash,
                "foreign_schema_name": con.foreign_schema_name,
                "foreign_relation_name": con.foreign_relation_name,
                "update_action": con.update_action.value if con.update_action else None,
                "delete_action": con.delete_action.value if con.delete_action else None,
                "match_type": con.match_type.value if con.match_type else None,
                "columns": cols_state,
            })

        # Indexes
        index_cols = {}
        for ic in result.index_columns:
            key = (ic.schema_name, ic.relation_name, ic.index_name)
            if key not in index_cols:
                index_cols[key] = []
            index_cols[key].append(ic)
            
        for idx in sorted(result.indexes, key=lambda x: (x.schema_name, x.relation_name, x.name)):
            cols = index_cols.get((idx.schema_name, idx.relation_name, idx.name), [])
            cols_state = [
                {
                    "ordinal_position": c.ordinal_position,
                    "column_name": c.column_name,
                    "expression_hash": c.expression_hash,
                    "included": c.included,
                    "sort_direction": c.sort_direction.value if c.sort_direction else None,
                    "nulls_order": c.nulls_order.value if c.nulls_order else None,
                }
                for c in sorted(cols, key=lambda x: x.ordinal_position)
            ]
            
            state["indexes"].append({
                "schema_name": idx.schema_name,
                "relation_name": idx.relation_name,
                "name": idx.name,
                "is_unique": idx.is_unique,
                "is_primary": idx.is_primary,
                "access_method": idx.access_method,
                "predicate_expression_hash": idx.predicate_expression_hash,
                "columns": cols_state,
            })

        # Routines
        for rtn in sorted(result.routines, key=lambda x: (x.schema_name, x.name, x.identity_arguments)):
            state["routines"].append({
                "schema_name": rtn.schema_name,
                "name": rtn.name,
                "identity_arguments": rtn.identity_arguments,
                "result_type": rtn.result_type,
                "routine_kind": rtn.routine_kind,
                "volatility": rtn.volatility,
            })
            
        json_bytes = json.dumps(state, separators=(',', ':'), sort_keys=True).encode('utf-8')
        return hashlib.sha256(json_bytes).hexdigest()
