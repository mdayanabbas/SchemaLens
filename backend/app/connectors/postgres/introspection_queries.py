from sqlalchemy import text

# Query to get namespaces
NAMESPACE_QUERY = text("""
SELECT 
    n.nspname AS name,
    d.description AS comment
FROM pg_namespace n
LEFT JOIN pg_description d ON d.objoid = n.oid AND d.objsubid = 0
WHERE n.nspname = ANY(:schemas)
ORDER BY n.nspname;
""")

# Query to get relations
RELATION_QUERY = text("""
SELECT 
    n.nspname AS schema_name,
    c.relname AS name,
    c.relkind AS kind,
    d.description AS comment,
    c.reltuples::bigint AS estimated_rows,
    c.relispartition AS is_partition,
    pn.nspname AS parent_schema_name,
    pc.relname AS parent_relation_name
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_description d ON d.objoid = c.oid AND d.objsubid = 0
LEFT JOIN pg_inherits i ON i.inhrelid = c.oid
LEFT JOIN pg_class pc ON pc.oid = i.inhparent
LEFT JOIN pg_namespace pn ON pn.oid = pc.relnamespace
WHERE n.nspname = ANY(:schemas)
  AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
ORDER BY n.nspname, c.relname;
""")

# Query to get columns
COLUMN_QUERY = text("""
SELECT 
    n.nspname AS schema_name,
    c.relname AS relation_name,
    a.attname AS name,
    a.attnum AS ordinal_position,
    pg_catalog.format_type(a.atttypid, a.atttypmod) AS formatted_data_type,
    t.typname AS base_data_type,
    CASE WHEN t.typtype = 'b' THEN
        (SELECT character_maximum_length FROM information_schema.columns 
         WHERE table_schema = n.nspname AND table_name = c.relname AND column_name = a.attname)
    END AS character_maximum_length,
    CASE WHEN t.typtype = 'b' THEN
        (SELECT numeric_precision FROM information_schema.columns 
         WHERE table_schema = n.nspname AND table_name = c.relname AND column_name = a.attname)
    END AS numeric_precision,
    CASE WHEN t.typtype = 'b' THEN
        (SELECT numeric_scale FROM information_schema.columns 
         WHERE table_schema = n.nspname AND table_name = c.relname AND column_name = a.attname)
    END AS numeric_scale,
    CASE WHEN t.typtype = 'b' THEN
        (SELECT datetime_precision FROM information_schema.columns 
         WHERE table_schema = n.nspname AND table_name = c.relname AND column_name = a.attname)
    END AS datetime_precision,
    NOT a.attnotnull AS is_nullable,
    a.atthasdef AS has_default,
    pg_get_expr(ad.adbin, ad.adrelid) AS default_expression,
    a.attidentity != '' AS is_identity,
    a.attidentity::text AS identity_generation,
    a.attgenerated != '' AS is_generated,
    a.attgenerated != '' AS generation_expression_present,
    coll.collname AS collation,
    d.description AS comment
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_type t ON t.oid = a.atttypid
LEFT JOIN pg_attrdef ad ON ad.adrelid = a.attrelid AND ad.adnum = a.attnum
LEFT JOIN pg_collation coll ON coll.oid = a.attcollation AND coll.collname != 'default'
LEFT JOIN pg_description d ON d.objoid = c.oid AND d.objsubid = a.attnum
WHERE n.nspname = ANY(:schemas)
  AND a.attnum > 0
  AND NOT a.attisdropped
  AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
ORDER BY n.nspname, c.relname, a.attnum;
""")

# Query to get constraints
CONSTRAINT_QUERY = text("""
SELECT 
    n.nspname AS schema_name,
    c.relname AS relation_name,
    con.conname AS name,
    con.contype AS kind,
    con.condeferrable AS is_deferrable,
    con.condeferred AS initially_deferred,
    con.convalidated AS is_validated,
    pg_get_expr(con.conbin, con.conrelid) AS check_expression,
    fn.nspname AS foreign_schema_name,
    fc.relname AS foreign_relation_name,
    fcon.conname AS foreign_constraint_name,
    con.confupdtype AS update_action,
    con.confdeltype AS delete_action,
    con.confmatchtype AS match_type,
    con.conkey AS conkey,
    con.confkey AS confkey
FROM pg_constraint con
JOIN pg_class c ON c.oid = con.conrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_class fc ON fc.oid = con.confrelid
LEFT JOIN pg_namespace fn ON fn.oid = fc.relnamespace
LEFT JOIN pg_constraint fcon ON fcon.conrelid = fc.oid AND fcon.contype = 'p' AND con.confrelid != 0
WHERE n.nspname = ANY(:schemas)
ORDER BY n.nspname, c.relname, con.conname;
""")

# Query to get constraint columns mapping
CONSTRAINT_COLUMN_QUERY = text("""
SELECT 
    n.nspname AS schema_name,
    c.relname AS relation_name,
    con.conname AS constraint_name,
    a.attname AS column_name,
    a.attnum AS ordinal_position,
    fa.attname AS referenced_column_name,
    -- We use generate_subscripts to unnest arrays with ordinality to match conkey/confkey
    idx.idx AS pk_ordinal
FROM pg_constraint con
JOIN pg_class c ON c.oid = con.conrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
CROSS JOIN generate_subscripts(con.conkey, 1) AS idx(idx)
JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = con.conkey[idx.idx]
LEFT JOIN pg_attribute fa ON fa.attrelid = con.confrelid AND fa.attnum = con.confkey[idx.idx]
WHERE n.nspname = ANY(:schemas)
ORDER BY n.nspname, c.relname, con.conname, idx.idx;
""")

# Query to get indexes
INDEX_QUERY = text("""
SELECT 
    n.nspname AS schema_name,
    c.relname AS relation_name,
    i.relname AS name,
    idx.indisunique AS is_unique,
    idx.indisprimary AS is_primary,
    idx.indisvalid AS is_valid,
    idx.indisready AS is_ready,
    am.amname AS access_method,
    idx.indpred IS NOT NULL AS predicate_present,
    pg_get_expr(idx.indpred, idx.indrelid) AS predicate_expression,
    idx.indexprs IS NOT NULL AS expression_index,
    pg_relation_size(i.oid) AS estimated_size_bytes
FROM pg_index idx
JOIN pg_class i ON i.oid = idx.indexrelid
JOIN pg_class c ON c.oid = idx.indrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_am am ON am.oid = i.relam
WHERE n.nspname = ANY(:schemas)
ORDER BY n.nspname, c.relname, i.relname;
""")

# Query to get index columns
INDEX_COLUMN_QUERY = text("""
SELECT 
    n.nspname AS schema_name,
    c.relname AS relation_name,
    i.relname AS index_name,
    a.attname AS column_name,
    pos.pos AS ordinal_position,
    -- Get expressions if the key is 0
    CASE WHEN idx.indkey[pos.pos - 1] = 0 THEN
        pg_get_indexdef(idx.indexrelid, pos.pos, true)
    END AS expression,
    pos.pos > idx.indnkeyatts AS included,
    -- Sort directions and nulls
    CASE 
        WHEN idx.indoption[pos.pos - 1] & 1 = 1 THEN 'descending'
        ELSE 'ascending'
    END AS sort_direction,
    CASE 
        WHEN idx.indoption[pos.pos - 1] & 2 = 2 THEN 'first'
        WHEN idx.indoption[pos.pos - 1] & 1 = 1 THEN 'first' -- DESC implicitly defaults to NULLS FIRST in PG
        ELSE 'last' -- ASC implicitly defaults to NULLS LAST
    END AS nulls_order
FROM pg_index idx
JOIN pg_class i ON i.oid = idx.indexrelid
JOIN pg_class c ON c.oid = idx.indrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
CROSS JOIN generate_series(1, array_length(idx.indkey, 1)) AS pos(pos)
LEFT JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = idx.indkey[pos.pos - 1]
WHERE n.nspname = ANY(:schemas)
ORDER BY n.nspname, c.relname, i.relname, pos.pos;
""")

# Query to get routines
ROUTINE_QUERY = text("""
SELECT 
    n.nspname AS schema_name,
    p.proname AS name,
    pg_get_function_identity_arguments(p.oid) AS identity_arguments,
    pg_get_function_result(p.oid) AS result_type,
    CASE p.prokind
        WHEN 'f' THEN 'function'
        WHEN 'p' THEN 'procedure'
        WHEN 'a' THEN 'aggregate'
        WHEN 'w' THEN 'window'
        ELSE 'unknown'
    END AS routine_kind,
    CASE p.provolatile
        WHEN 'i' THEN 'immutable'
        WHEN 's' THEN 'stable'
        WHEN 'v' THEN 'volatile'
        ELSE 'unknown'
    END AS volatility,
    CASE p.proparallel
        WHEN 's' THEN 'safe'
        WHEN 'r' THEN 'restricted'
        WHEN 'u' THEN 'unsafe'
        ELSE 'unknown'
    END AS parallel_safety,
    p.prosecdef AS security_definer,
    l.lanname AS language
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
JOIN pg_language l ON l.oid = p.prolang
WHERE n.nspname = ANY(:schemas)
ORDER BY n.nspname, p.proname, identity_arguments;
""")
