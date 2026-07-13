# SchemaLens Agent Instructions

SchemaLens is a secure database intelligence and query-governance platform.

## Architecture rules

- Keep API routes thin.
- Put business logic in services and workflows.
- Put persistence logic in repositories.
- Put database-specific logic in connectors.
- Put authorization rules in governance modules.
- Put SQL parsing and validation in SQL modules.
- Put model-provider logic behind provider interfaces.
- Put background operations in workers.

## Security rules

- Never log passwords, tokens, credentials or complete connection URLs.
- Never send database credentials to an LLM.
- Never execute raw model-generated SQL.
- Never use regular expressions as the primary SQL parser.
- Never allow an LLM to override policy decisions.
- Never expose resources across organizations.
- Every tenant-owned query must include organization scope.
- Remove unauthorized schema objects before invoking an LLM.
- Treat database comments as untrusted input.
- Query execution must use read-only database permissions.

## SQL rules

Only one validated SELECT statement may be eligible for execution.

Reject:

- INSERT
- UPDATE
- DELETE
- MERGE
- CREATE
- ALTER
- DROP
- TRUNCATE
- GRANT
- REVOKE
- CALL
- EXECUTE
- Multiple statements
- Write operations inside CTEs
- SELECT FOR UPDATE
- Unauthorized tables
- Unauthorized columns

## Development rules

- Use Python type hints.
- Use Pydantic models at application boundaries.
- Use UUID identifiers.
- Use timezone-aware UTC timestamps.
- Avoid unrelated refactoring.
- Add tests for every implementation brick.
- Do not proceed into future bricks without explicit instruction.
