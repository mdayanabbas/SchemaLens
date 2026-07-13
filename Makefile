backend-install:
	cd backend && python -m pip install -e ".[dev]"

backend-dev:
	cd backend && uvicorn app.main:app --reload

backend-test:
	cd backend && pytest -q

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-test:
	cd frontend && npm test

db-up:
	docker compose up -d postgres redis

db-down:
	docker compose down

db-reset:
	docker compose down -v
	docker compose up -d postgres redis

migrate:
	cd backend && alembic upgrade head

worker:
	cd backend && celery -A app.workers.celery_app worker --loglevel=info
