# Research Assistant

A **multi-agent RAG system**: a planner splits your research question into sub-tasks, paper and web agents research them in parallel, a synthesiser writes a cited answer, a fact-checker removes unsupported claims, and a critic scores the result (researching again if it is weak). Progress streams live to a Next.js UI.

**Stack:** LangGraph · LlamaIndex · Groq (gpt-oss) · FastAPI · Celery · Redis · ChromaDB · Next.js · Docker

---

## Architecture

```
User question ──► FastAPI ──► Celery worker (LangGraph)
                                 │
   [Memory]      last turns from Redis, so follow-ups like "who proposed it?" resolve
      ↓
   [Planner]     gpt-oss-20b · 2–4 sub-tasks, each routed to papers or the web
      ↓
   [Research]    in parallel:
                   papers — query rewriting → dense + BM25 (RRF) → cross-encoder rerank
                   web    — Tavily search
                 (falls back to the web when the knowledge base has nothing relevant)
      ↓
   [Synthesiser] gpt-oss-120b · Markdown answer citing numbered sources [n]
                 → fact-check: every sentence / table row checked against the sources;
                   unsupported ones are removed
      ↓
   [Critic]      score 0–10 · if < 6, rewrite queries and research once more
      ↓
   done ──► Redis Stream (replayable event log) ──► SSE ──► Next.js UI
```

---

## Features

**Research**
- Answers cite numbered sources in reading order; hover a citation to preview the source, click it to jump to the source card
- Quality signals on every answer: grounding %, critic score, fact-check report (what was removed or inferred), and whether it was refined or fell back to the web
- Suggested follow-up questions; conversation memory for follow-ups
- Web search toggle (papers only vs papers + web)
- Stop a running job; re-ask any question
- Copy or download any answer — or a whole conversation — as Markdown with references

**Interface**
- Research history in the sidebar (stored in the browser), searchable, grouped by date
- **Run inspector:** a per-answer timeline of each agent stage — the plan, papers and web results found, grounding, critic feedback, timings
- **Library:** see what's in the knowledge base, add papers by arXiv topic, summarise and add a single paper by link, remove papers
- Live service status (API, LLM, worker, Redis, ChromaDB), with hints when something is down
- Light / dark / system theme; responsive down to phone width
- Reloading mid-run reconnects to the job and replays its progress

---

## Quickstart

### 1. Configure

```bash
cp .env.example backend/.env      # set GROQ_API_KEY (and TAVILY_API_KEY for web search)
cp .env.example frontend/.env     # only the NEXT_PUBLIC_* lines are used
```

Free keys: [Groq](https://console.groq.com) · [Tavily](https://app.tavily.com) (1,000 searches/month)

### 2. Run (local dev)

```bash
docker compose up redis chromadb       # infrastructure
python run_dev.py                      # FastAPI (:8000) + Celery worker — run with the backend venv's Python
cd frontend && pnpm install && pnpm dev   # UI on http://localhost:3000
```

Or everything in Docker: `make dev`.

### 3. Add papers

From the UI: **Library → Add papers by topic**. Or from the CLI:

```bash
make ingest                                                        # default ML corpus
cd backend && poetry run python -m app.ingestion.pipeline --query "flash attention" --max-papers 20
```

Papers already in the knowledge base are skipped, so re-running is safe.

---

## Development

```bash
make dev-infra       # Redis + ChromaDB only
make dev-backend     # FastAPI hot-reload on :8000
make dev-worker      # Celery worker
make dev-frontend    # Next.js on :3000

make lint            # ruff + eslint
make test            # pytest
make eval            # RAGAS scores for the agent pipeline (papers only)
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/research` | Submit a question → `job_id` (body: `query`, `session_id`, `use_web_search`) |
| GET | `/api/v1/stream/{job_id}` | SSE stream: `queued` · `job_started` · `agent_done` · `done` · `failed` · `cancelled`. Replayable for 1 hour; resumes via `Last-Event-ID` |
| POST | `/api/v1/research/{job_id}/cancel` | Stop a job at the next agent boundary |
| DELETE | `/api/v1/sessions/{session_id}` | Forget a conversation's server-side memory |
| GET | `/api/v1/knowledge/stats` | Paper and passage counts |
| GET | `/api/v1/knowledge/papers` | Papers in the knowledge base (`?q=` filter) |
| DELETE | `/api/v1/knowledge/papers/{arxiv_id}` | Remove a paper and its passages |
| POST | `/api/v1/ingest` | Fetch arXiv papers for a topic into the knowledge base |
| POST | `/api/v1/papers/summarise` | Structured summary of one arXiv paper (also ingests it) |
| GET | `/api/v1/health` | Liveness probe |
| GET | `/api/v1/ready` | Readiness: LLM models, Redis, ChromaDB, worker; configured features |
| GET | `/api/v1/metrics` | Prometheus metrics |

Write endpoints require an `X-Api-Key` header (not enforced when `ENVIRONMENT=development`). Interactive docs: `http://localhost:8000/docs`

---

## Evaluation (RAGAS)

`make eval` runs the questions in `backend/eval/data/qa_pairs.json` through the full agent graph with web search off and scores the results. Reports are saved to `backend/eval/reports/`.

| Metric | Threshold | What it measures |
|--------|-----------|-----------------|
| Faithfulness | ≥ 0.70 | Are claims grounded in retrieved context? |
| Answer Relevancy | ≥ 0.75 | Does the answer address the question? |
| Context Recall | ≥ 0.65 | Did retrieval find the right chunks? |
| Context Precision | ≥ 0.65 | Are retrieved chunks all relevant? |

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key |
| `GROQ_MODEL` / `GROQ_MODEL_FAST` | No | Defaults `openai/gpt-oss-120b` / `openai/gpt-oss-20b` |
| `TAVILY_API_KEY` | No | Enables web search |
| `REDIS_URL` | Yes | Redis connection string (queue, memory, job event log) |
| `CHROMA_HOST` / `CHROMA_PORT` | Yes | ChromaDB connection |
| `API_KEY` | Yes | Secret for the `X-Api-Key` header |
| `CORS_ORIGINS` | No | JSON list of allowed browser origins |
| `RERANK_MODEL` / `RERANK_MIN_SCORE` | No | Cross-encoder and relevance cutoff for paper passages |
| `LANGCHAIN_API_KEY` | No | LangSmith tracing |

---

## Troubleshooting

- **Every answer fails, or `/ready` shows "model not available":** the configured Groq model was retired. Set `GROQ_MODEL` / `GROQ_MODEL_FAST` to a model listed at `https://api.groq.com/openai/v1/models`.
- **Stuck on "Waiting for a worker":** the Celery worker isn't running — start `python run_dev.py` (or `make dev-worker`).
- **`EPERM: symlink` during a frontend build on Windows:** standalone output needs symlinks, so it is only enabled inside Docker (`BUILD_STANDALONE=1`); a plain `pnpm build` doesn't use it.
- **No papers cited:** the knowledge base may not cover the topic — add papers from the Library, or leave web search on.
