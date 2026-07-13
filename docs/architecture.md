# SchemaLens Architecture

Main application layers:

Frontend
-> FastAPI API
-> Services and workflows
-> Repositories and infrastructure
-> SchemaLens application database

External database access is isolated behind connector adapters and workers.
