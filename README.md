# Research Assistant

A **multi-agent RAG system** that takes a research question, decomposes it into sub-tasks, runs specialised agents (web search + ArXiv retrieval), and streams back a grounded answer with citations.

**Stack:** LangGraph · LlamaIndex · Groq LLaMA 3 · FastAPI · Redis · ChromaDB · Next.js · Docker

---

## Architecture

```
User Query
    ↓
[Planner Agent]     ← LLaMA 3 8B via Groq
    ↓ routes to
┌────────────────────────────────┐
│  [Web Search]   Tavily API     │
│  [RAG Agent]    ChromaDB       │  ← dense retrieval + cross-encoder reranking
└────────────────────────────────┘
    ↓
[Synthesiser]       ← LLaMA 3 70B via Groq
    ↓
Final answer + citations   →   FastAPI SSE   →   Next.js UI
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/research-assistant
cd research-assistant
cp .env.example .env
# Fill in GROQ_API_KEY and TAVILY_API_KEY in .env
```

### 2. Start everything

```bash
make dev
```

This starts: Redis · ChromaDB · FastAPI backend · Celery worker · Next.js frontend

### 3. Seed the corpus

```bash
make ingest          # indexes default ML Research papers from ArXiv
# or custom:
cd backend && poetry run python -m app.ingestion.pipeline --query "flash attention" --max-papers 20
```

### 4. Open the UI

```
http://localhost:3000
```

---

## Development

```bash
# Individual services (after `make dev-infra` for Redis + ChromaDB)
make dev-backend     # FastAPI hot-reload on :8000
make dev-worker      # Celery worker
make dev-frontend    # Next.js on :3000

# Code quality
make lint            # ruff + eslint
make test            # pytest with coverage

# Evaluation
make eval            # RAGAS scores against eval/data/qa_pairs.json
```

---

## Project structure

```
research-assistant/
├── backend/
│   ├── app/
│   │   ├── agents/          ← LangGraph nodes (planner, web_search, rag, synthesiser)
│   │   ├── api/v1/          ← FastAPI endpoints
│   │   ├── core/            ← config, logging, security, metrics
│   │   ├── ingestion/       ← ArXiv loader + chunkers + pipeline
│   │   ├── memory/          ← Redis (short-term) + ChromaDB (long-term)
│   │   ├── models/          ← Pydantic schemas
│   │   └── workers/         ← Celery tasks
│   ├── eval/                ← RAGAS evaluation + QA pairs
│   └── tests/               ← pytest unit + integration tests
└── frontend/
    ├── app/                 ← Next.js App Router pages
    ├── components/          ← ChatInput, MessageBubble, AgentStatusPanel, SourceCitations
    ├── hooks/               ← useResearch (SSE)
    ├── lib/                 ← API client, TypeScript types
    └── store/               ← Zustand chat state
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/research` | Submit a query → returns `job_id` |
| GET | `/api/v1/stream/{job_id}` | SSE stream of agent events |
| POST | `/api/v1/ingest` | Ingest ArXiv papers into ChromaDB |
| GET | `/api/v1/health` | Liveness probe |
| GET | `/api/v1/ready` | Readiness probe (checks Redis + ChromaDB) |
| GET | `/api/v1/metrics` | Prometheus metrics |

All write endpoints require `X-Api-Key` header.

Interactive docs: `http://localhost:8000/docs`

---

## Evaluation (RAGAS)

RAGAS scores are enforced as a CI gate — a PR that drops faithfulness below 0.70 cannot be merged.

```bash
make eval
```

| Metric | Threshold | What it measures |
|--------|-----------|-----------------|
| Faithfulness | ≥ 0.70 | Are claims grounded in retrieved context? |
| Answer Relevancy | ≥ 0.75 | Does the answer address the question? |
| Context Recall | ≥ 0.65 | Did retrieval find the right chunks? |
| Context Precision | ≥ 0.65 | Are retrieved chunks all relevant? |

Reports saved to `backend/eval/reports/`.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key (free tier available) |
| `TAVILY_API_KEY` | Yes | Tavily search API key |
| `REDIS_URL` | Yes | Redis connection string |
| `CHROMA_HOST` / `CHROMA_PORT` | Yes | ChromaDB connection |
| `API_KEY` | Yes | Secret for `X-Api-Key` header |
| `LANGCHAIN_API_KEY` | No | LangSmith tracing (optional) |

---

## Deployment

The project is Dockerised and ready to deploy to Railway / Fly.io / Render.

```bash
# Build all images
make build

# Deploy to Railway (set RAILWAY_TOKEN in GitHub secrets)
# CI/CD handles this automatically on push to main
```

---

## Get free API keys

- **Groq:** https://console.groq.com — free tier, very fast
- **Tavily:** https://app.tavily.com — 1,000 free searches/month
- **LangSmith:** https://smith.langchain.com — free tier for tracing
