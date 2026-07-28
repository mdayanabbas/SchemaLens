from enum import StrEnum

class SchemaSnapshotStatus(StrEnum):
    BUILDING = "building"
    READY = "ready"
    INVALID = "invalid"
    SUPERSEDED = "superseded"


class SchemaRelationKind(StrEnum):
    TABLE = "table"
    PARTITIONED_TABLE = "partitioned_table"
    VIEW = "view"
    MATERIALIZED_VIEW = "materialized_view"
    FOREIGN_TABLE = "foreign_table"


class SchemaConstraintKind(StrEnum):
    PRIMARY_KEY = "primary_key"
    UNIQUE = "unique"
    FOREIGN_KEY = "foreign_key"
    CHECK = "check"
    EXCLUSION = "exclusion"


class ReferentialAction(StrEnum):
    NO_ACTION = "no_action"
    RESTRICT = "restrict"
    CASCADE = "cascade"
    SET_NULL = "set_null"
    SET_DEFAULT = "set_default"


class MatchType(StrEnum):
    SIMPLE = "simple"
    FULL = "full"
    PARTIAL = "partial"


class SortDirection(StrEnum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


class NullsOrder(StrEnum):
    FIRST = "first"
    LAST = "last"
    DATABASE_DEFAULT = "database_default"


class SchemaObjectType(StrEnum):
    NAMESPACE = "namespace"
    RELATION = "relation"
    COLUMN = "column"
    CONSTRAINT = "constraint"
    INDEX = "index"
    ROUTINE = "routine"
