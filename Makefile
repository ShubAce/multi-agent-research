.PHONY: dev stop build lint test eval clean help

# ── Dev ─────────────────────────────────────────────────────────────────────
dev:
	@echo "🚀  Starting full stack..."
	docker compose up --build

dev-infra:
	@echo "🛠   Starting Redis + ChromaDB only..."
	docker compose up redis chromadb

dev-backend:
	@echo "🐍  Starting backend (hot-reload)..."
	cd backend && poetry run uvicorn app.main:app --reload --port 8000

dev-local:
	@echo "🚀  Starting backend + Celery worker concurrently..."
	python run_dev.py

dev-worker:
	@echo "⚙️   Starting Celery worker..."
	cd backend && poetry run celery -A app.workers.celery_app worker --loglevel=info

dev-frontend:
	@echo "⚛️   Starting Next.js..."
	cd frontend && pnpm dev

stop:
	docker compose down

# ── Build ────────────────────────────────────────────────────────────────────
build:
	docker compose build

# ── Lint & Type-check ────────────────────────────────────────────────────────
lint:
	@echo "🔍  Linting backend..."
	cd backend && poetry run ruff check . && poetry run ruff format --check .
	@echo "🔍  Linting frontend..."
	cd frontend && pnpm lint

typecheck:
	cd backend && poetry run mypy app/

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	@echo "🧪  Running backend tests..."
	cd backend && poetry run pytest tests/ -v --cov=app --cov-report=term-missing

test-e2e:
	@echo "🎭  Running Playwright E2E tests..."
	cd frontend && pnpm exec playwright test

# ── Evaluation ───────────────────────────────────────────────────────────────
eval:
	@echo "📊  Running RAGAS evaluation..."
	cd backend && poetry run python eval/ragas_eval.py

# ── Ingestion ────────────────────────────────────────────────────────────────
ingest:
	@echo "📚  Ingesting ArXiv corpus..."
	cd backend && poetry run python -m app.ingestion.pipeline

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	docker compose down -v
	find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find backend -name "*.pyc" -delete 2>/dev/null || true

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  make dev           → full stack (Docker)"
	@echo "  make dev-infra     → Redis + ChromaDB only"
	@echo "  make dev-backend   → FastAPI hot-reload"
	@echo "  make dev-local     → FastAPI + Celery worker concurrently"
	@echo "  make dev-worker    → Celery worker"
	@echo "  make dev-frontend  → Next.js dev server"
	@echo "  make lint          → ruff + eslint"
	@echo "  make test          → pytest"
	@echo "  make eval          → RAGAS evaluation"
	@echo "  make ingest        → ingest ArXiv corpus"
	@echo "  make clean         → remove containers + volumes"
	@echo ""
