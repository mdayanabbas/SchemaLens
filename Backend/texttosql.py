import requests
import json
import hashlib
import shelve
import re
from decimal import Decimal
from datetime import date, datetime

from sqlalchemy import create_engine, inspect, text
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn


DB_URL       = "postgresql://postgres:abbas@localhost:5432/temp"
LLM_URL      = "http://localhost:1234/v1/chat/completions"
LLM_MODEL    = "qwen2.5-coder"
MAX_RETRIES  = 3
CACHE_FILE   = "sql_cache"

SYSTEM_PROMPT = """You are a senior data engineer. Generate ONLY raw SQL queries — no markdown, no backticks, no explanations.

Rules:
- Always use the provided relationships for JOINs
- Never assume columns that are not listed in the schema
- Use explicit JOIN conditions (never implicit comma joins)
- Prefer INNER JOIN unless the question implies optional matches
- Use aliases for readability on multi-table queries
- Always qualify column names with table aliases when joining
- Use ILIKE for case-insensitive text matching in PostgreSQL
- Output a single SQL statement ending with a semicolon"""


engine = create_engine(DB_URL)

_schema_cache: str | None = None

def get_schema(force_refresh: bool = False) -> str:
    global _schema_cache
    if _schema_cache and not force_refresh:
        return _schema_cache

    inspector = inspect(engine)
    tables    = inspector.get_table_names()

    schema_text   = ""
    relationships = []

    with engine.connect() as con:
        for table in tables:
            # Row count
            try:
                row_count = con.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
            except Exception:
                row_count = "?"

            schema_text += f"Table: {table}  ({row_count} rows)\n"

            # Columns
            columns = inspector.get_columns(table)
            pk_cols = inspector.get_pk_constraint(table).get("constrained_columns", [])

            for col in columns:
                col_name = col["name"]
                col_type = str(col["type"])
                pk_tag   = " [PK]" if col_name in pk_cols else ""

                # Sample values
                try:
                    rows = con.execute(
                        text(f'SELECT DISTINCT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL LIMIT 3')
                    ).fetchall()
                    samples = ", ".join(str(r[0]) for r in rows)
                    sample_hint = f"  -- e.g. {samples}" if samples else ""
                except Exception:
                    sample_hint = ""

                schema_text += f"  - {col_name} ({col_type}){pk_tag}{sample_hint}\n"

            # Foreign keys
            for fk in inspector.get_foreign_keys(table):
                local_cols  = fk["constrained_columns"]
                remote_table = fk["referred_table"]
                remote_cols = fk["referred_columns"]
                for lc, rc in zip(local_cols, remote_cols):
                    relationships.append(f"  {table}.{lc} → {remote_table}.{rc}")

            schema_text += "\n"

    if relationships:
        schema_text += "Relationships:\n"
        schema_text += "\n".join(relationships) + "\n"

    _schema_cache = schema_text
    return schema_text



FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "ALTER", "TRUNCATE",
    "INSERT", "CREATE", "REPLACE", "EXEC", "EXECUTE",
    "GRANT", "REVOKE", "COPY", "VACUUM",
]

def is_safe(query: str) -> None:
    """Raises ValueError if the query contains destructive keywords."""
    upper = query.upper()
    # Strip string literals before checking (prevents false negatives via quoted words)
    stripped = re.sub(r"'[^']*'", "", upper)
    for word in FORBIDDEN_KEYWORDS:
        # Use word-boundary style check: preceded/followed by non-alpha
        if re.search(rf"\b{word}\b", stripped):
            raise ValueError(f"Unsafe operation detected: '{word}' is not allowed.")



def call_llm(messages: list[dict]) -> str:
    """Send messages to the local LLM and return the text response."""
    response = requests.post(
        LLM_URL,
        json={
            "model":       LLM_MODEL,
            "temperature": 0,
            "top_p":       1,
            "messages":    messages,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def clean_sql(raw: str) -> str:
    """Strip markdown fences and whitespace from LLM output."""
    raw = re.sub(r"```sql", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```",    "", raw)
    return raw.strip()


def generate_sql(question: str, chat_history: list[dict] | None = None) -> str:
    """
    Generate SQL from a natural-language question.
    Maintains optional chat_history for follow-up questions.
    """
    schema = get_schema()

    user_msg = {
        "role": "user",
        "content": f"Schema:\n{schema}\n\nQuestion:\n{question}",
    }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages += chat_history
    messages.append(user_msg)

    raw_sql = call_llm(messages)
    sql     = clean_sql(raw_sql)

    if chat_history is not None:
        chat_history.append(user_msg)
        chat_history.append({"role": "assistant", "content": sql})

    return sql



def serialize_row(row_mapping: dict) -> dict:
    """Convert non-JSON-serialisable types to plain Python types."""
    out = {}
    for key, value in row_mapping.items():
        if isinstance(value, Decimal):
            value = float(value)
        elif isinstance(value, (date, datetime)):
            value = value.isoformat()
        out[key] = value
    return out


def execute_with_retry(
    sql: str,
    original_question: str,
    max_retries: int = MAX_RETRIES,
) -> tuple[list[dict], str]:
    """
    Execute SQL. If it fails, ask the LLM to fix it and retry.
    Returns (results, final_sql_used).
    """
    current_sql = sql
    last_error  = None

    for attempt in range(max_retries):
        try:
            is_safe(current_sql)

            with engine.connect() as con:
                result_proxy = con.execute(text(current_sql))
                rows = [serialize_row(dict(row._mapping)) for row in result_proxy]
                return rows, current_sql

        except ValueError as e:
            # Safety violation — never retry, just raise
            raise

        except Exception as e:
            last_error = str(e)
            print(f"\n[Attempt {attempt + 1}] SQL Error: {last_error}")

            if attempt < max_retries - 1:
                print("Asking LLM to self-correct…")
                schema = get_schema()
                print(schema)
                fix_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Schema:\n{schema}\n\n"
                            f"Original question:\n{original_question}\n\n"
                            f"This SQL query failed:\n{current_sql}\n\n"
                            f"Error message:\n{last_error}\n\n"
                            "Fix the SQL query. Return ONLY the corrected SQL."
                        ),
                    },
                ]
                current_sql = clean_sql(call_llm(fix_messages))
                print(f"Corrected SQL:\n{current_sql}\n")

    raise RuntimeError(
        f"Query failed after {max_retries} attempts. Last error: {last_error}\n"
        f"Last SQL tried:\n{current_sql}"
    )



def cached_generate_sql(question: str, chat_history: list[dict] | None = None) -> str:
    """Return cached SQL if available, otherwise generate and cache it."""
    # Don't cache when there's conversation context (results may differ)
    if chat_history:
        return generate_sql(question, chat_history)

    key = hashlib.md5(question.lower().strip().encode()).hexdigest()
    with shelve.open(CACHE_FILE) as cache:
        if key in cache:
            print("(cache hit — skipping LLM call)")
            return cache[key]
        sql = generate_sql(question)
        cache[key] = sql
        return sql


def explain_results(question: str, sql: str, results: list[dict]) -> str:
    """Ask the LLM to narrate the query results in plain English."""
    preview = results[:10] 
    messages = [
        {
            "role": "user",
            "content": (
                f"The user asked: \"{question}\"\n\n"
                f"This SQL was run:\n{sql}\n\n"
                f"Results (first {len(preview)} rows):\n{json.dumps(preview, indent=2)}\n\n"
                "Summarise the findings in 2–3 plain-English sentences. "
                "Be specific: mention numbers, names, or dates from the results."
            ),
        }
    ]
    return call_llm(messages)


app = FastAPI(title="Text-to-SQL API", version="2.0")


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Text-to-SQL API",
        "docs": "/docs"
    }
    
_sessions: dict[str, list[dict]] = {}


class QueryRequest(BaseModel):
    question:   str
    session_id: str  | None = None   
    explain:    bool         = False 
    use_cache:  bool         = True


class QueryResponse(BaseModel):
    sql:         str
    results:     list[dict]
    explanation: str | None = None
    session_id:  str | None = None


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    history = None
    if req.session_id:
        if req.session_id not in _sessions:
            _sessions[req.session_id] = []
        history = _sessions[req.session_id]

    try:
        # Generate SQL
        if req.use_cache and not history:
            sql = cached_generate_sql(req.question)
        else:
            sql = generate_sql(req.question, chat_history=history)

        # Execute (with auto-retry)
        results, final_sql = execute_with_retry(sql, req.question)

        # Optional explanation
        explanation = None
        if req.explain:
            explanation = explain_results(req.question, final_sql, results)

        return QueryResponse(
            sql=final_sql,
            results=results,
            explanation=explanation,
            session_id=req.session_id,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema")
async def schema_endpoint(refresh: bool = False):
    """Return the current database schema (useful for debugging)."""
    return {"schema": get_schema(force_refresh=refresh)}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Wipe conversation history for a session."""
    _sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.delete("/cache")
async def clear_cache():
    """Wipe the SQL query cache."""
    with shelve.open(CACHE_FILE) as cache:
        cache.clear()
    return {"cleared": True}


# ─────────────────────────────────────────────
#  INTERACTIVE CLI  (multi-turn conversation)
# ─────────────────────────────────────────────
def run_cli():
    print("=" * 60)
    print("  Text-to-SQL  (type 'exit' to quit, 'new' to reset chat)")
    print("=" * 60)

    chat_history: list[dict] = []

    while True:
        print()
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            break
        if question.lower() == "new":
            chat_history.clear()
            print("Conversation reset.")
            continue
        if question.lower() == "schema":
            print(get_schema())
            continue

        try:
            # Generate
            sql = cached_generate_sql(question, chat_history=chat_history if chat_history else None)
            print(f"\n[SQL]\n{sql}\n")

            # Execute with retry
            results, final_sql = execute_with_retry(sql, question)

            if final_sql != sql:
                print(f"[Corrected SQL]\n{final_sql}\n")

            print(f"[Results — {len(results)} row(s)]")
            print(json.dumps(results, indent=2))

            # Plain-English summary
            if results:
                summary = explain_results(question, final_sql, results)
                print(f"\n[Summary]\n{summary}")

        except ValueError as e:
            print(f"[BLOCKED] {e}")
        except Exception as e:
            print(f"[ERROR] {e}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--api" in sys.argv:
        # Start the FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    else:
        # Run the interactive CLI
        run_cli()