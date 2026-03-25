import requests
# from fastapi import FastAPI
from sqlalchemy import create_engine, inspect, text
import json
from decimal import Decimal

# app = FastAPI()

URL = "postgresql://postgres:abbas@localhost:5432/temp"
engine = create_engine(URL)
def get_schema():
    inspector = inspect(engine)

    tables = inspector.get_table_names()

    schema_text = ""
    relationships = []

    for table in tables:
        schema_text += f"Table: {table}\n"

        # Columns
        columns = inspector.get_columns(table)
        pk_cols = inspector.get_pk_constraint(table).get("constrained_columns", [])

        for col in columns:
            col_name = col["name"]
            col_type = str(col["type"])

            if col_name in pk_cols:
                schema_text += f"- {col_name} ({col_type}) [PK]\n"
            else:
                schema_text += f"- {col_name} ({col_type})\n"

        # Foreign Keys
        fks = inspector.get_foreign_keys(table)
        for fk in fks:
            local_cols = fk["constrained_columns"]
            remote_table = fk["referred_table"]
            remote_cols = fk["referred_columns"]

            for lc, rc in zip(local_cols, remote_cols):
                relationships.append(f"{table}.{lc} = {remote_table}.{rc}")

        schema_text += "\n"

    # Add relationships at end
    if relationships:
        schema_text += "Relationships:\n"
        for rel in relationships:
            schema_text += f"- {rel}\n"

    return schema_text


def is_safe(query):
    forbidden = ["DROP", "DELETE", "UPDATE", "ALTER"]
    return not any(word in query.upper() for word in forbidden)

schema_text = get_schema()
# print(schema_text)



def prompt_build(question):
    return f"""
Schema:
{schema_text}

Question:
{question}
"""
question = input("Enter your question: ")

res = requests.post(
    "http://localhost:1234/v1/chat/completions",
    json={
        "model": "qwen2.5-coder",
        "temperature": 0,
        "top_p": 1,
        "messages": [
            {
                "role": "system",
                "content": """You are a senior data engineer. Generate only SQL queries. No explanations.
                Rules:
                - Always use provided relationships for joins
                - Never assume columns that are not listed
                - Use explicit JOIN conditions
                - Prefer INNER JOIN unless specified otherwise"""
            },
            {
                "role": "user",
                "content": prompt_build(question)
            }
        ]
    }
)

result = res.json()
# print(result)
# print(result['choices'][0]['message']['content'])
query = result['choices'][0]['message']['content']
clean_sql = query.replace("```sql", "").replace("```", "").strip()
clean_sql = text(clean_sql)
print(clean_sql)

with engine.connect() as con:
    ans = con.execute(clean_sql)

    result = []
    for row in ans:
        row_dict = {}

        for key, value in row._mapping.items():
            if isinstance(value, Decimal):
                value = float(value)

            row_dict[key] = value

        result.append(row_dict)

    print(json.dumps(result, indent=2))