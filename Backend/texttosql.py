import requests
from fastapi import FastAPI

app = FastAPI()

schema_text = """
Table: users
- id
- name
- email

Table: orders
- id
- user_id
- amount
"""

def prompt_build(question):
    return f"""You are a Senior level Data Engineer who is great at SQL Querying. This is the schema of
            the database {schema_text} and this is the question {question}. 
            Write a SQL Query to answer the exact question given based on the provided schema, be mindful of the schema
            and how it is organized to extract the correct informartion. Only write SQL Query and nothing else."""


res = requests.post(
    "http://localhost:1234/v1/chat/completions",
    json={
        "model": "qwen2.5-coder",
        "messages": [{"role": "system", "content": "You are a helpful assistant for writing SQL Queries based on the provided database schema."},
                     {"role": "user", "content": "prompt_build('What is the total amount of orders for each user?')"}]
    }
)

result = res.json()
print(result['choices'][0]['message']['content'])