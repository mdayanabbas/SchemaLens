import re

from app.core.exceptions import ValidationError
from app.models.connection_enums import SSLMode

# Basic regex for host validation
# Allows valid hostname, IPv4, IPv6
# Rejects protocols, credentials, paths, control characters
HOST_REGEX = re.compile(
    r"^(?:[a-zA-Z0-9]"
    r"(?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2,6}\.?$|"
    r"^(?:\d{1,3}\.){3}\d{1,3}$|"
    r"^\[?[a-fA-F0-9:]+\]?$"
)

# Extremely simple check for local un-routed hostnames (like `localhost` or `db`)
# In a real environment, you'd want more robust validation.
SIMPLE_HOST_REGEX = re.compile(r"^[a-zA-Z0-9-]{1,63}$")

def validate_connection_name(name: str) -> str:
    """Normalize and validate connection name."""
    name = name.strip()
    if not name:
        raise ValidationError("Connection name cannot be empty.", code="INVALID_CONNECTION_CONFIGURATION")
    return name

def validate_host(host: str) -> str:
    """Validate host format."""
    host = host.strip()
    if not host:
        raise ValidationError("Host cannot be empty.", code="INVALID_CONNECTION_HOST")
        
    # Reject common URL components
    if "://" in host or "@" in host or "/" in host or "?" in host or "#" in host:
        raise ValidationError("Host must not contain protocols, credentials, paths, or query parameters.", code="INVALID_CONNECTION_HOST")
        
    # Check for control characters or whitespace
    if re.search(r"[\x00-\x1F\x7F\s]", host):
        raise ValidationError("Host contains invalid characters.", code="INVALID_CONNECTION_HOST")
        
    # We do a loose regex to make sure it's at least vaguely host-like
    if not (HOST_REGEX.match(host) or SIMPLE_HOST_REGEX.match(host)):
        raise ValidationError("Host format is invalid.", code="INVALID_CONNECTION_HOST")
        
    return host

def validate_database_name(db_name: str) -> str:
    """Validate database name."""
    db_name = db_name.strip()
    if not db_name:
        raise ValidationError("Database name cannot be empty.", code="INVALID_DATABASE_NAME")
        
    # Reject URL delimiters and credentials
    if any(c in db_name for c in ["/", "?", "#", "@", ":"]):
        raise ValidationError("Database name contains invalid URL delimiters or credentials.", code="INVALID_DATABASE_NAME")
        
    # Reject control chars
    if re.search(r"[\x00-\x1F\x7F]", db_name):
        raise ValidationError("Database name contains invalid characters.", code="INVALID_DATABASE_NAME")
        
    return db_name

def validate_secret_reference(secret_ref: str) -> str:
    """Validate secret reference."""
    secret_ref = secret_ref.strip()
    if not secret_ref:
        raise ValidationError("Secret reference cannot be empty.", code="INVALID_SECRET_REFERENCE")
        
    if len(secret_ref) > 500:
        raise ValidationError("Secret reference is too long.", code="INVALID_SECRET_REFERENCE")
        
    if "\x00" in secret_ref or "\n" in secret_ref or "\r" in secret_ref:
        raise ValidationError("Secret reference contains invalid characters.", code="INVALID_SECRET_REFERENCE")
        
    # Reject obvious connection URLs or raw credentials
    if "://" in secret_ref and "@" in secret_ref:
        raise ValidationError("Secret reference looks like a complete connection URL.", code="INVALID_SECRET_REFERENCE")
        
    return secret_ref

def normalize_and_deduplicate_schemas(schemas: list[str], max_schemas: int = 100) -> list[str]:
    """Normalize, validate, and deduplicate a list of schema names."""
    if len(schemas) > max_schemas:
        raise ValidationError(f"Maximum of {max_schemas} schemas exceeded.", code="INVALID_CONNECTION_POLICY")
        
    seen = set()
    result = []
    
    for schema in schemas:
        s = schema.strip()
        if not s:
            raise ValidationError("Schema name cannot be empty.", code="INVALID_CONNECTION_POLICY")
            
        if re.search(r"[\x00-\x1F\x7F]", s):
            raise ValidationError("Schema name contains invalid characters.", code="INVALID_CONNECTION_POLICY")
            
        if s not in seen:
            seen.add(s)
            result.append(s)
            
    return result

def validate_schema_lists(approved: list[str], blocked: list[str]) -> tuple[list[str], list[str]]:
    """Validate approved and blocked schemas, ensuring no overlap."""
    app_norm = normalize_and_deduplicate_schemas(approved)
    blk_norm = normalize_and_deduplicate_schemas(blocked)
    
    overlap = set(app_norm).intersection(set(blk_norm))
    if overlap:
        raise ValidationError("Approved and blocked schemas cannot overlap.", code="INVALID_CONNECTION_POLICY")
        
    return app_norm, blk_norm

def validate_production_ssl(environment: str, ssl_mode: str) -> None:
    """Ensure production environments do not disable SSL."""
    if environment == "production" and ssl_mode == SSLMode.DISABLE:
        raise ValidationError("Production environments require SSL.", code="PRODUCTION_SSL_REQUIRED")

def redact_secret_reference(provider: str, secret_ref: str) -> str:
    """Redact a secret reference for safe display in the API."""
    if not secret_ref:
        return ""
        
    # e.g. arn:aws:secretsmanager:region:account:secret:prod/orders-db-AbCd
    # e.g. SCHEMALENS_CUSTOMER_DB_SECRET
    # e.g. local://encrypted/0192c45e-...
    
    provider_prefix = provider.replace("_", "-")
    
    # We want to show only a small suffix or a generic prefix
    if len(secret_ref) <= 8:
        # Too short to safely expose parts
        return f"{provider_prefix}:[REDACTED]"
        
    # Expose the last 4 characters for identification
    suffix = secret_ref[-4:]
    
    if provider == "environment":
        # Environment variables often have a recognizable prefix, let's expose the first 12 chars + suffix
        prefix = secret_ref[:12] if len(secret_ref) > 16 else secret_ref[:4]
        return f"{provider_prefix}:{prefix}...{suffix}"
    
    return f"{provider_prefix}:...{suffix}"
