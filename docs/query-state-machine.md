# SchemaLens Query State Machine

received
-> authorizing
-> retrieving_schema
-> needs_clarification or planning
-> generating_sql
-> validating
-> rejected, awaiting_approval or approved
-> executing
-> succeeded, failed, timed_out or cancelled

Validation must never be skipped.
