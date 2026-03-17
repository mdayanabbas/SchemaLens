from __future__ import annotations
import json
import urllib.parse
import requests
import subprocess
from typing import Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, MetaData, inspect

app = Flask(__name__)
CORS(app)

# ==========================================
# CONFIGURATION
# ==========================================
# Since 192.168.29.41 worked in your first script, we will use it as the source of truth.
LM_STUDIO_HOST = "192.168.29.41"
LM_STUDIO_PORT = "1234"
LM_STUDIO_BASE_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1"

# Specific Endpoints
CHAT_URL = f"{LM_STUDIO_BASE_URL}/chat/completions"
MODELS_URL = f"{LM_STUDIO_BASE_URL}/models" # Note: No trailing slash!

MODEL_NAME = "qwen2.5-coder-7b-instruct"

# ==========================================
# DATABASE INTROSPECTION
# ==========================================
def get_dialect_from_url(url: str) -> str:
    """Determine SQLAlchemy dialect from URL scheme"""
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme
    dialect = scheme.split('+')[0]
    dialect_map = {
        'postgres': 'postgresql',
        'postgresql': 'postgresql',
        'mysql': 'mysql',
        'mariadb': 'mysql',
        'sqlite': 'sqlite',
        'oracle': 'oracle',
        'mssql': 'mssql',
    }
    return dialect_map.get(dialect, dialect)


def introspect_database(db_url: str, table_name: Optional[str] = None) -> dict[str, Any]:
    """Introspect database and return structured data"""
    dialect = get_dialect_from_url(db_url)
    engine = create_engine(db_url)
    inspector = inspect(engine)
    metadata = MetaData()
    
    if table_name:
        metadata.reflect(bind=engine, only=[table_name])
        table_names = [table_name] if inspector.has_table(table_name) else []
    else:
        metadata.reflect(bind=engine)
        table_names = inspector.get_table_names()
    
    tables_data = []
    relationships = []
    
    for tbl_name in table_names:
        table_info = {
            "name": tbl_name,
            "columns": [],
            "primary_keys": [],
            "foreign_keys": [],
            "indexes": []
        }
        
        columns = inspector.get_columns(tbl_name)
        for col in columns:
            table_info["columns"].append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "default": str(col["default"]) if col.get("default") else None
            })
        
        pk = inspector.get_pk_constraint(tbl_name)
        table_info["primary_keys"] = pk.get("constrained_columns", [])
        
        fks = inspector.get_foreign_keys(tbl_name)
        for fk in fks:
            referred_table = fk.get("referred_table")
            constrained_cols = fk.get("constrained_columns", [])
            referred_cols = fk.get("referred_columns", [])
            
            for i, col in enumerate(constrained_cols):
                ref_col = referred_cols[i] if i < len(referred_cols) else "id"
                rel = {
                    "from_table": tbl_name,
                    "from_column": col,
                    "to_table": referred_table,
                    "to_column": ref_col,
                    "type": "foreign_key"
                }
                table_info["foreign_keys"].append(rel)
                relationships.append(rel)
        
        indexes = inspector.get_indexes(tbl_name)
        for idx in indexes:
            table_info["indexes"].append({
                "name": idx["name"],
                "columns": idx.get("column_names", []),
                "unique": idx.get("unique", False)
            })
        
        tables_data.append(table_info)
    
    return {
        "tables": tables_data,
        "relationships": relationships,
        "dialect": dialect,
    }

# ==========================================
# LLM ANALYSIS
# ==========================================
def analyze_with_local_llm(schema_info: dict[str, Any]) -> dict[str, Any]:
    """Use local Qwen 2.5 7B via LM Studio to infer relationships and classify entities"""
    
    tables_summary = []
    for table in schema_info["tables"]:
        cols = [f"{c['name']}({c['type'].split('(')[0]})" for c in table["columns"]]
        pks = table.get("primary_keys", [])
        fks = [f"{fk['from_column']}->{fk['to_table']}" for fk in table.get("foreign_keys", [])]
        
        tables_summary.append({
            "name": table["name"],
            "columns": cols[:15], 
            "pk": pks,
            "fk": fks
        })
    
    prompt = f"""You are a database architect. Analyze this database schema and:
1. Infer logical relationships not explicitly defined by foreign keys (based on naming conventions)
2. Classify each table as: master_data, transaction, lookup, or junction
3. Suggest a concise display color category for each table (blue, green, purple, orange, red, gray)

Schema:
{json.dumps(tables_summary, indent=2)}

Respond with STRICT JSON only:
{{
  "inferred_relationships": [
    {{
      "from": "table_name",
      "to": "other_table",
      "type": "one_to_many",
      "confidence": "high|medium|low",
      "reason": "column naming pattern explanation"
    }}
  ],
  "table_classifications": [
    {{
      "table": "table_name",
      "category": "master_data|transaction|lookup|junction",
      "color": "blue|green|purple|orange|red|gray",
      "reason": "brief explanation"
    }}
  ]
}}"""

    try:
        response = requests.post(
            CHAT_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a database schema analyzer. Output valid JSON only. Do not include markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1, # Lowered to 0.1 to match your working script
                "max_tokens": 2000,
                "stream": False
            },
            timeout=120 # Increased timeout in case the model takes a moment to generate
        )
        
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"]
        print("--- LLM RAW RESPONSE ---")
        print(content)
        print("------------------------")
        
        # ROBUST JSON PARSING: Find the first '{' and the last '}'
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = content[start_idx:end_idx]
            llm_result = json.loads(json_str)
        else:
            raise ValueError("No JSON object could be found in the response.")
            
        return {
            "inferred_relationships": llm_result.get("inferred_relationships", []),
            "table_classifications": llm_result.get("table_classifications", [])
        }
        
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse LLM JSON: {e}")
        return {"inferred_relationships": [], "table_classifications": []}
    except requests.exceptions.ConnectionError:
        print(f"⚠️  LM Studio not reachable at {CHAT_URL}. Skipping LLM analysis.")
        return {"inferred_relationships": [], "table_classifications": []}
    except Exception as e:
        print(f"⚠️  LLM analysis failed: {e}")
        return {"inferred_relationships": [], "table_classifications": []}


# ==========================================
# API ENDPOINTS
# ==========================================
@app.route('/api/schema', methods=['POST'])
def get_schema():
    """API endpoint to get database schema with optional LLM enhancement"""
    try:
        data = request.json
        db_url = data.get('url')
        use_llm = data.get('use_llm', True)
        
        if not db_url:
            return jsonify({"error": "No database URL provided"}), 400
        
        print(f"🔌 Connecting to: {db_url}")
        schema_info = introspect_database(db_url, data.get('table_name'))
        
        if use_llm:
            print(f"🧠 Analyzing with {MODEL_NAME} via LM Studio...")
            llm_result = analyze_with_local_llm(schema_info)
            schema_info["inferred_relationships"] = llm_result["inferred_relationships"]
            schema_info["table_classifications"] = llm_result["table_classifications"]
            print(f"✅ Found {len(llm_result['inferred_relationships'])} inferred relationships")
        
        return jsonify(schema_info)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    llm_status = "disconnected"
    try:
        r = requests.get(MODELS_URL, timeout=3)
        if r.status_code == 200:
            llm_status = "connected"
    except Exception as e:
        print(f"Connection failed to \"{MODELS_URL}\": {e}")
        llm_status = "disconnected"
    
    return jsonify({
        "status": "ok",
        "llm_status": llm_status,
        "lm_studio_host": LM_STUDIO_HOST,
        "model": MODEL_NAME if llm_status == "connected" else None
    })

# ==========================================
# MAIN RUNNER
# ==========================================
if __name__ == '__main__':
    print(f"🚀 Starting Schema Visualizer API")
    print(f"🤖 LM Studio Chat URL: {CHAT_URL}")
    print(f"🤖 LM Studio Models URL: {MODELS_URL}")
    print(f"📝 Target Model: {MODEL_NAME}")
    print(f"📡 Flask Port: 5000")
    
    app.run(port=5000, host='0.0.0.0')