# Multi-Agent Research Assistant — Project Roadmap

### Portfolio Project | IIT Kharagpur | CDC & AI/ML Internship Prep

> **Stack:** LangGraph · LlamaIndex · FastAPI · Redis · ChromaDB · Next.js · Docker

---

## Table of Contents

1. [Project Overview](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#project-overview)
2. [Repo Structure](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#repo-structure)
3. [Phase 0 — Project Setup](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#phase-0)
4. [Phase 1 — RAG Pipeline](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#phase-1)
5. [Phase 2 — Multi-Agent System with LangGraph](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#phase-2)
6. [Phase 3 — Memory &amp; Persistence](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#phase-3)
7. [Phase 4 — FastAPI Backend](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#phase-4)
8. [Phase 5 — Next.js Frontend](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#phase-5)
9. [Phase 6 — Evaluation &amp; Observability](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#phase-6)
10. [Phase 7 — Docker &amp; Deployment](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#phase-7)
11. [GitHub Workflow](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#github-workflow)
12. [CV Bullet &amp; Interview Prep](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#interview-prep)
13. [Future Extensions](https://claude.ai/chat/d0e60a00-b9a8-4266-9c80-91278e6f07a5?onboarding=1#future-extensions)

---

## Project Overview `<a name="project-overview"></a>`

### What you are building

A **multi-agent AI research assistant** that takes a natural-language query, breaks it into sub-tasks, runs specialised agents (web search, RAG retrieval, code execution), combines the results, and streams back a grounded answer with citations — served through a FastAPI backend and a Next.js chat interface.

### Why this stands out

| What you build                    | Why recruiters care                                     |
| --------------------------------- | ------------------------------------------------------- |
| Multi-agent system with LangGraph | Shows system-design thinking beyond basic LLM calls     |
| RAG with RAGAS evaluation         | Proves you measure quality, not just vibe-check outputs |
| Streaming FastAPI + Next.js       | Real production pattern, not a Jupyter notebook         |
| Docker + CI/CD                    | Shows you can ship, not just prototype                  |

### Pick your domain (commit to one)

| Domain                             | Corpus       | Best for                         |
| ---------------------------------- | ------------ | -------------------------------- |
| **ML Research**(recommended) | ArXiv papers | AI/ML internships, research labs |
|                                    |              |                                  |

## Repo Structure `<a name="repo-structure"></a>`

The entire project lives in one monorepo with two top-level folders:

```
research-assistant/
├── backend/          ← Python: FastAPI + LangGraph + LlamaIndex
├── frontend/         ← Next.js: Chat UI
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── .env.example
├── Makefile
└── README.md
```

### `backend/` layout

```
backend/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── api/
│   │   └── v1/
│   │       ├── research.py      # POST /research, GET /stream/{id}
│   │       └── sessions.py      # GET /sessions
│   ├── agents/
│   │   ├── state.py             # Shared TypedDict state schema
│   │   ├── graph.py             # LangGraph workflow definition
│   │   ├── planner.py           # Planner agent node
│   │   ├── web_search.py        # Tavily web search node
│   │   ├── rag_agent.py         # LlamaIndex retrieval node
│   │   └── synthesiser.py       # Final answer node
│   ├── ingestion/
│   │   ├── pipeline.py          # Document ingestion (PDFs, ArXiv)
│   │   └── chunkers.py          # Chunking strategies
│   ├── memory/
│   │   ├── short_term.py        # Redis conversation buffer
│   │   └── long_term.py         # ChromaDB semantic memory
│   ├── core/
│   │   ├── config.py            # Settings via pydantic-settings
│   │   ├── security.py          # API key auth + input sanitisation
│   │   └── logging.py           # structlog setup
│   └── workers/
│       └── tasks.py             # Celery background tasks
├── tests/
│   ├── unit/
│   └── integration/
├── eval/
│   ├── ragas_eval.py            # RAGAS evaluation script
│   └── data/
│       └── qa_pairs.json        # 50–100 hand-crafted eval pairs
├── Dockerfile
└── pyproject.toml               # poetry dependencies
```

### `frontend/` layout

```
frontend/
├── app/
│   ├── page.tsx                 # Main chat page
│   ├── layout.tsx
│   └── api/
│       └── chat/
│           └── route.ts         # Next.js proxy → FastAPI
├── components/
│   ├── ChatWindow.tsx           # Message list + input box
│   ├── MessageBubble.tsx        # User / assistant message
│   ├── AgentStatusPanel.tsx     # Live agent progress
│   └── SourceCitations.tsx      # Collapsible citation list
├── hooks/
│   └── useResearch.ts           # SSE subscription hook
├── store/
│   └── chat.ts                  # Zustand state
├── Dockerfile
└── package.json                 # pnpm dependencies
```

---

## Phase 0 — Project Setup `<a name="phase-0"></a>`

**Duration:** 2 days

**Goal:** Get a clean, reproducible dev environment running before touching any AI code.

### Tasks

**Backend**

```bash
cd backend
poetry init
poetry add fastapi uvicorn langchain langgraph llama-index \
           chromadb redis celery openai groq tavily-python \
           pydantic-settings structlog python-dotenv
poetry add --group dev pytest pytest-asyncio httpx ruff mypy
```

**Frontend**

```bash
cd frontend
pnpm create next-app . --typescript --tailwind --app
pnpm add zustand @tanstack/react-query lucide-react
pnpm add -D playwright @playwright/test
```

**Environment Variables (`.env.example`)**

```
# LLM
OPENAI_API_KEY=
GROQ_API_KEY=

# Tools
TAVILY_API_KEY=

# Services
REDIS_URL=redis://localhost:6379
CHROMA_HOST=localhost
CHROMA_PORT=8001

# App
ENVIRONMENT=development
LOG_LEVEL=INFO
API_KEY=your-local-dev-key
```

**Local services via Docker Compose**

```yaml
# docker-compose.yml (dev only)
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  chromadb:
    image: chromadb/chroma:latest
    ports: ["8001:8001"]
    volumes: ["chroma_data:/chroma/chroma"]

volumes:
  chroma_data:
```

**Makefile shortcuts**

```makefile
dev:         # docker compose up + backend + frontend
lint:        # ruff check backend/ && pnpm lint
test:        # pytest + playwright
```

**Pre-commit hooks (`.pre-commit-config.yaml`)**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks: [{ id: ruff }, { id: ruff-format }]
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks: [{ id: mypy, args: [--strict] }]
  - repo: https://github.com/Yelp/detect-secrets
    hooks: [{ id: detect-secrets }]
```

### Done when

* `make dev` starts all services in one command
* `make lint` and `make test` pass on an empty codebase
* README has a working "Get started in 5 minutes" section

---

## Phase 1 — RAG Pipeline `<a name="phase-1"></a>`

**Duration:** 5–6 days

**Goal:** Build a document ingestion + retrieval pipeline that can answer domain-specific questions with source attribution, and measure its quality with RAGAS.

### Ingestion Pipeline

```python
# backend/app/ingestion/pipeline.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

class IngestionPipeline:
    def __init__(self, collection_name: str = "research_docs"):
        self.client = chromadb.HttpClient(host="localhost", port=8001)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

    def ingest_arxiv(self, query: str, max_results: int = 30) -> int:
        """Pull papers from ArXiv API and index them."""
        # Use arxiv Python library to fetch papers
        # Parse PDFs → chunk → embed → store in ChromaDB
        ...

    def ingest_local_pdfs(self, directory: str) -> int:
        """Ingest a folder of PDFs."""
        docs = SimpleDirectoryReader(directory).load_data()
        nodes = self.parser.get_nodes_from_documents(docs)
        index = VectorStoreIndex(
            nodes,
            vector_store=ChromaVectorStore(chroma_collection=self.collection),
        )
        return len(nodes)
```

### Chunking — try two strategies, pick the better one

| Strategy        | Setting                | When to use                    |
| --------------- | ---------------------- | ------------------------------ |
| Fixed-size      | 512 tokens, 50 overlap | Fast baseline                  |
| Sentence-window | 3-sentence window      | Better for dense academic text |

Run RAGAS on both and document the winner. This is the kind of empirical decision-making interviewers love asking about.

### Retrieval with Reranking

```python
# backend/app/agents/rag_agent.py
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

# Reranker narrows 10 candidates down to the 3 best — big quality boost
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    top_n=3,
)

query_engine = index.as_query_engine(
    node_postprocessors=[reranker],
    response_mode="compact",
)
```

Including a **reranker** is one of the highest-leverage RAG improvements and very few portfolio projects have it — mention it explicitly on your CV.

### RAGAS Baseline Evaluation

Create `eval/data/qa_pairs.json` with 50–80 hand-crafted question/answer pairs from your corpus. Run this before building agents so you have a baseline to compare against.

```python
# backend/eval/ragas_eval.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

dataset = load_eval_dataset("eval/data/qa_pairs.json")
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])

print(results)
# Save to eval/reports/baseline.csv
results.to_pandas().to_csv("eval/reports/baseline.csv")
```

Target scores: faithfulness ≥ 0.75, answer relevancy ≥ 0.80.

### Done when

* You can ask a question and get a grounded answer with source citations
* RAGAS baseline is measured and saved in `eval/reports/`
* A simple `GET /api/v1/query?q=...` endpoint works (no agents yet)

---

## Phase 2 — Multi-Agent System with LangGraph `<a name="phase-2"></a>`

**Duration:** 7–8 days

**Goal:** Replace the single Q&A chain with a LangGraph state machine where a Planner breaks the query into sub-tasks and dispatches specialised agents.

### Agent Flow

```
User Query
    ↓
[Planner Agent]  →  decides which agents to call and in what order
    ↓
┌─────────────────────────────────────┐
│  [Web Search Agent]  (Tavily API)   │  ← for current / external info
│  [RAG Agent]         (ChromaDB)     │  ← for domain knowledge
│  [Code Agent]        (Python REPL)  │  ← for calculations / data
└─────────────────────────────────────┘
    ↓
[Synthesiser Agent]  →  merges all results into one grounded answer
    ↓
Final Response + Citations
```

### State Schema

```python
# backend/app/agents/state.py
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages:      Annotated[Sequence[BaseMessage], operator.add]
    query:         str
    sub_tasks:     list[str]
    web_results:   list[dict]
    rag_results:   list[dict]
    code_output:   str | None
    final_answer:  str | None
    citations:     list[dict]
    error:         str | None
```

### Planner Node

```python
# backend/app/agents/planner.py
PLANNER_PROMPT = """
You are a research planner. Given a complex query, break it into 2–4 sub-tasks.
For each sub-task decide which agent should handle it:
  - web_search : for recent/external facts
  - rag        : for domain-specific documents in our knowledge base
  - code       : for numerical computation or data analysis

Return JSON: { "sub_tasks": [...], "routing": { "task": "agent_type" } }
"""

def planner_node(state: AgentState) -> AgentState:
    response = llm.invoke(PLANNER_PROMPT + "\n\nQuery: " + state["query"])
    plan = parse_json_safely(response.content)
    return {**state, "sub_tasks": plan["sub_tasks"], "_routing": plan["routing"]}
```

### Building the Graph

```python
# backend/app/agents/graph.py
from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents import planner, web_search, rag_agent, code_agent, synthesiser

workflow = StateGraph(AgentState)

workflow.add_node("planner",     planner.planner_node)
workflow.add_node("web_search",  web_search.web_search_node)
workflow.add_node("rag_agent",   rag_agent.rag_agent_node)
workflow.add_node("code_agent",  code_agent.code_agent_node)
workflow.add_node("synthesiser", synthesiser.synthesiser_node)

workflow.set_entry_point("planner")

def route_after_planner(state: AgentState) -> str:
    agents_needed = set(state["_routing"].values())
    if "code" in agents_needed:
        return "code_agent"
    if "web_search" in agents_needed:
        return "web_search"
    return "rag_agent"

workflow.add_conditional_edges("planner", route_after_planner, {
    "web_search": "web_search",
    "rag_agent":  "rag_agent",
    "code_agent": "code_agent",
})

workflow.add_edge("web_search", "rag_agent")
workflow.add_edge("rag_agent",  "synthesiser")
workflow.add_edge("code_agent", "synthesiser")
workflow.add_edge("synthesiser", END)

agent_graph = workflow.compile()
```

### Synthesiser

The Synthesiser receives results from all agents and produces a single answer with:

* Inline citations linked to source documents
* A confidence indicator based on how much of the answer is grounded vs. inferred
* A brief "reasoning trace" (which agents contributed what)

### Testing agents

* Test each node individually by passing a hand-crafted `AgentState` dict
* Test the full graph on 5–10 known queries and assert the correct agents were called
* Test graceful fallback: if web search returns nothing, the graph should still return an answer from RAG

---

## Phase 3 — Memory & Persistence `<a name="phase-3"></a>`

**Duration:** 3–4 days

**Goal:** Add two types of memory so the assistant remembers context within a session and learns from past research across sessions.

### Short-Term Memory: Redis (per session)

```python
# backend/app/memory/short_term.py
import redis.asyncio as aioredis, json, time

class ConversationMemory:
    def __init__(self, redis_url: str, ttl_seconds: int = 3600):
        self.r = aioredis.from_url(redis_url)
        self.ttl = ttl_seconds

    async def add_turn(self, session_id: str, role: str, content: str):
        key = f"session:{session_id}:history"
        turn = json.dumps({"role": role, "content": content, "ts": time.time()})
        await self.r.rpush(key, turn)
        await self.r.expire(key, self.ttl)

    async def get_history(self, session_id: str, last_n: int = 8) -> list[dict]:
        key = f"session:{session_id}:history"
        raw = await self.r.lrange(key, -last_n, -1)
        return [json.loads(r) for r in raw]
```

### Long-Term Memory: ChromaDB (cross-session facts)

After each research session, extract key facts and embed them. Future queries retrieve relevant memories and inject them into the system prompt.

```python
# backend/app/memory/long_term.py
class LongTermMemory:
    async def memorise(self, session_id: str, facts: list[str]):
        """Store extracted facts as embeddings after each session."""
        self.collection.add(
            documents=facts,
            ids=[f"{session_id}_{i}" for i in range(len(facts))],
            metadatas=[{"session_id": session_id} for _ in facts],
        )

    async def recall(self, query: str, n: int = 4) -> list[str]:
        """Retrieve semantically similar past facts."""
        results = self.collection.query(query_texts=[query], n_results=n)
        return results["documents"][0]
```

### Injecting memory into agents

```python
async def build_system_context(query: str, session_id: str) -> str:
    history  = await short_term.get_history(session_id)
    memories = await long_term.recall(query)
    return f"""
Conversation so far:
{format_history(history)}

Relevant facts from past research:
{chr(10).join(f"- {m}" for m in memories)}
"""
```

### LangGraph Checkpointing (bonus — easy to add)

```python
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

checkpointer = AsyncRedisSaver.from_conn_string(settings.REDIS_URL)
agent_graph = workflow.compile(checkpointer=checkpointer)
```

This lets you resume an interrupted agent run — a genuinely advanced feature worth mentioning in interviews.

---

## Phase 4 — FastAPI Backend `<a name="phase-4"></a>`

**Duration:** 4–5 days

**Goal:** Expose the agent system through a clean, async REST API with streaming support, auth, and proper error handling.

### Endpoints

```
POST   /api/v1/research          Submit a query → returns job_id
GET    /api/v1/stream/{job_id}   SSE stream of live agent events
GET    /api/v1/sessions          List past sessions
GET    /api/v1/health            Health check
```

### App Factory

```python
# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import research, sessions, health

def create_app() -> FastAPI:
    app = FastAPI(title="Research Assistant API", version="1.0.0")

    app.add_middleware(CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(research.router, prefix="/api/v1")
    app.include_router(sessions.router, prefix="/api/v1")
    app.include_router(health.router,   prefix="/api/v1")

    return app

app = create_app()
```

### Research Endpoint

```python
# backend/app/api/v1/research.py
from fastapi import APIRouter, Depends, BackgroundTasks
from app.models.request import ResearchRequest
from app.models.response import ResearchJobResponse
from app.workers.tasks import run_agent_pipeline

router = APIRouter()

@router.post("/research", response_model=ResearchJobResponse, status_code=202)
async def submit_research(
    req: ResearchRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),   # simple API key auth
):
    job_id = str(uuid4())
    # Kick off Celery task so the HTTP response is instant
    run_agent_pipeline.delay(
        job_id=job_id,
        query=req.query,
        session_id=req.session_id or str(uuid4()),
    )
    return ResearchJobResponse(job_id=job_id, status="queued")
```

### SSE Streaming Endpoint

```python
# backend/app/api/v1/research.py
from sse_starlette.sse import EventSourceResponse

@router.get("/stream/{job_id}")
async def stream_events(job_id: str):
    async def generator():
        async with redis.pubsub() as ps:
            await ps.subscribe(f"job:{job_id}:events")
            async for msg in ps.listen():
                if msg["type"] == "message":
                    data = json.loads(msg["data"])
                    yield {"event": data["type"], "data": json.dumps(data)}
                    if data["type"] == "done":
                        break

    return EventSourceResponse(generator())
```

### Celery Worker

```python
# backend/app/workers/tasks.py
from app.workers import celery_app
import redis, json

r = redis.Redis.from_url(settings.REDIS_URL)

@celery_app.task(bind=True, max_retries=3)
def run_agent_pipeline(self, job_id: str, query: str, session_id: str):
    try:
        for event in agent_graph.stream({"query": query, "session_id": session_id}):
            r.publish(f"job:{job_id}:events", json.dumps(event))
        r.publish(f"job:{job_id}:events", json.dumps({"type": "done"}))
    except Exception as exc:
        r.publish(f"job:{job_id}:events", json.dumps({"type": "error", "message": str(exc)}))
        raise self.retry(exc=exc)
```

### Request/Response Models

```python
# backend/app/models/request.py
from pydantic import BaseModel, Field

class ResearchRequest(BaseModel):
    query:      str  = Field(..., min_length=5, max_length=1000)
    session_id: str | None = None
    domain:     str = "arxiv"

# backend/app/models/response.py
class ResearchJobResponse(BaseModel):
    job_id:  str
    status:  str          # "queued" | "running" | "done" | "error"
```

### Auth (simple API key)

```python
# backend/app/core/security.py
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
```

### Done when

* `POST /api/v1/research` returns a `job_id` instantly
* Connecting to `/api/v1/stream/{job_id}` streams agent events live
* Tests cover: auth failure, rate limit, invalid request, SSE event order

---

## Phase 5 — Next.js Frontend `<a name="phase-5"></a>`

**Duration:** 4–5 days

**Goal:** A clean, responsive chat interface that streams agent updates in real time and displays source citations alongside the final answer.

### Pages

```
/          → Main chat interface
/sessions  → Past research sessions
```

### Core Components

**ChatWindow** — the main layout: sidebar (sessions) + chat area + agent panel

**MessageBubble** — renders user or assistant messages; assistant messages include a "Sources" toggle that expands the citation list

**AgentStatusPanel** — shows which agent is currently running with a simple animated indicator

```tsx
// frontend/components/AgentStatusPanel.tsx
const AGENT_ICONS: Record<string, string> = {
  planner:     "🧠",
  web_search:  "🌐",
  rag_agent:   "📚",
  code_agent:  "💻",
  synthesiser: "✨",
};

type AgentEvent = { agent: string; status: string; duration_ms?: number };

export function AgentStatusPanel({ events }: { events: AgentEvent[] }) {
  return (
    <div className="rounded-lg bg-slate-50 p-4 space-y-2">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
        Agent Activity
      </p>
      {events.map((e, i) => (
        <div key={i} className="flex items-center gap-2 text-sm">
          <span>{AGENT_ICONS[e.agent] ?? "🤖"}</span>
          <span className="font-medium text-slate-700">{e.agent}</span>
          <span className="text-slate-400">{e.status}</span>
          {e.duration_ms && (
            <span className="ml-auto text-xs text-slate-300">{e.duration_ms}ms</span>
          )}
        </div>
      ))}
    </div>
  );
}
```

### SSE Hook

```typescript
// frontend/hooks/useResearch.ts
import { useState } from "react";

export function useResearch() {
  const [agentEvents, setAgentEvents] = useState<AgentEvent[]>([]);
  const [answer, setAnswer]           = useState<string>("");
  const [loading, setLoading]         = useState(false);

  async function submit(query: string, sessionId: string) {
    setLoading(true);
    setAgentEvents([]);
    setAnswer("");

    // 1. Post query, get job_id
    const res  = await fetch("/api/v1/research", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Api-Key": process.env.NEXT_PUBLIC_API_KEY! },
      body: JSON.stringify({ query, session_id: sessionId }),
    });
    const { job_id } = await res.json();

    // 2. Open SSE stream
    const es = new EventSource(`/api/v1/stream/${job_id}`);

    es.addEventListener("agent_start", (e) => {
      setAgentEvents((prev) => [...prev, JSON.parse(e.data)]);
    });

    es.addEventListener("done", (e) => {
      const data = JSON.parse(e.data);
      setAnswer(data.final_answer);
      setLoading(false);
      es.close();
    });

    es.addEventListener("error", () => {
      setLoading(false);
      es.close();
    });
  }

  return { submit, agentEvents, answer, loading };
}
```

### Next.js API Route (proxy to FastAPI)

```typescript
// frontend/app/api/chat/route.ts
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const res = await fetch(`${process.env.BACKEND_URL}/api/v1/research`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Api-Key": process.env.API_KEY!,
    },
    body: JSON.stringify(body),
  });
  return NextResponse.json(await res.json());
}
```

### Zustand Store

```typescript
// frontend/store/chat.ts
import { create } from "zustand";

interface ChatStore {
  sessionId:   string;
  messages:    Message[];
  addMessage:  (msg: Message) => void;
  clearChat:   () => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  sessionId:  crypto.randomUUID(),
  messages:   [],
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  clearChat:  () => set({ messages: [], sessionId: crypto.randomUUID() }),
}));
```

### Done when

* Typing a query streams agent events live in the sidebar
* Final answer renders with a collapsible "Sources" section
* Playwright E2E test: submit query → assert agent panel populates → assert answer renders

---

## Phase 6 — Evaluation & Observability `<a name="phase-6"></a>`

**Duration:** 3 days

**Goal:** Make the system measurable and debuggable — the two things that separate an engineering project from a demo.

### RAGAS in CI

```python
# backend/eval/ragas_eval.py
import pytest
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

THRESHOLDS = {
    "faithfulness":     0.75,
    "answer_relevancy": 0.80,
    "context_recall":   0.70,
}

def test_ragas_scores():
    dataset = load_eval_dataset("eval/data/qa_pairs.json")
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
    for metric, threshold in THRESHOLDS.items():
        assert results[metric] >= threshold, \
            f"{metric} dropped to {results[metric]:.2f}, threshold is {threshold}"
```

Add this to GitHub Actions so a PR that degrades RAG quality cannot be merged. This is the most impressive single thing you can add — treating model quality as a first-class CI concern.

### LangSmith Tracing

Sign up for a free LangSmith account and add two env vars:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-key
```

Every LLM call, agent transition, and tool invocation is now recorded with latency, token counts, and full input/output. Invaluable for debugging agent loops — and a great thing to screenshot for your README.

### Structured Logging

```python
# backend/app/core/logging.py
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

# Usage — every agent event logs structured JSON:
log.info("agent_completed",
    agent="rag_agent",
    job_id=job_id,
    duration_ms=847,
    sources_retrieved=3,
)
```

### Basic Metrics Endpoint

```python
# backend/app/api/v1/health.py
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

requests_total = Counter("research_requests_total", "Total research requests", ["status"])

@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

---

## Phase 7 — Docker & Deployment `<a name="phase-7"></a>`

**Duration:** 3 days

**Goal:** Package everything into Docker containers and deploy to a live URL.

### Backend Dockerfile (multi-stage)

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
RUN pip install poetry==1.8.0
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true && \
    poetry install --only main --no-root

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv .venv
COPY app/ app/
ENV PATH="/app/.venv/bin:$PATH"

RUN adduser --disabled-password appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile

```dockerfile
# frontend/Dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile
COPY . .
RUN pnpm build

FROM node:20-alpine AS runtime
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
```

### Full `docker-compose.yml`

```yaml
services:
  backend:
    build: ./backend
    env_file: .env
    ports: ["8000:8000"]
    depends_on: [redis, chromadb]

  worker:
    build: ./backend
    command: celery -A app.workers.celery_app worker --loglevel=info
    env_file: .env
    depends_on: [redis]

  frontend:
    build: ./frontend
    env_file: .env
    ports: ["3000:3000"]
    depends_on: [backend]

  redis:
    image: redis:7-alpine
    volumes: ["redis_data:/data"]

  chromadb:
    image: chromadb/chroma:latest
    volumes: ["chroma_data:/chroma/chroma"]

volumes:
  redis_data:
  chroma_data:
```

### GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: cd backend && pip install poetry && poetry install
      - run: cd backend && poetry run pytest tests/ -v
      - run: cd backend && poetry run ruff check .

  eval:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - run: cd backend && poetry run python eval/ragas_eval.py

  deploy:
    runs-on: ubuntu-latest
    needs: [test, eval]
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to Railway
        run: railway up
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
```

### Deploy target

**Railway** is the simplest option — connect your GitHub repo, set env vars, and it deploys automatically on every push to `main`. Total cost is around $5–10/month during active development.

---

## GitHub Workflow `<a name="github-workflow"></a>`

### Branch naming

```
feature/rag-pipeline
feature/langgraph-agents
fix/rag-chunking-overlap
chore/docker-setup
```

### Commit style (Conventional Commits)

```
feat(agents): add web search agent with Tavily
fix(rag): correct sentence-window chunk overlap
perf(retrieval): add cross-encoder reranker, +12% context recall
test(eval): add RAGAS CI gate for faithfulness
docs: add architecture diagram to README
```

### Release milestones

Tag these in GitHub so you can demo the project's evolution:

| Tag        | What works                    |
| ---------- | ----------------------------- |
| `v0.1.0` | RAG pipeline + RAGAS baseline |
| `v0.2.0` | Multi-agent LangGraph system  |
| `v0.3.0` | FastAPI + streaming           |
| `v1.0.0` | Full stack deployed with CI   |

---

## CV Bullet & Interview Prep `<a name="interview-prep"></a>`

### CV Bullet (copy this)

> Engineered a **multi-agent RAG research assistant** using **LangGraph & LlamaIndex** with autonomous task decomposition, hybrid retrieval with cross-encoder reranking, and a two-tier memory system (Redis + ChromaDB); served via **FastAPI + Celery** with SSE streaming and a **Next.js** frontend — achieved faithfulness ≥ 0.75 and answer relevancy ≥ 0.80 on domain-specific QA benchmarks (RAGAS), enforced as a CI gate in GitHub Actions.

### Questions you will be asked

**"Walk me through the architecture."**
Planner breaks the query into sub-tasks → Worker agents (web search, RAG, code) run in parallel → Synthesiser merges results with citations → FastAPI streams events to the Next.js frontend via SSE. State is managed through a typed LangGraph `AgentState` dict.

**"Why LangGraph over LangChain agents or CrewAI?"**
LangGraph gives you explicit, debuggable state transitions. LangChain agents and CrewAI abstract away the control flow — they are fine for simple demos but hard to debug in production. LangGraph also supports checkpointing and human-in-the-loop interrupts natively.

**"How do you know the system is working?"**
RAGAS metrics — faithfulness, answer relevancy, context recall — run as a CI gate. A PR that drops faithfulness below 0.75 cannot be merged. LangSmith traces every LLM call so I can see exactly what the agents did.

**"What was the hardest part?"**
Getting the Synthesiser to produce well-cited answers even when one agent fails. The conditional routing logic in LangGraph and the typed `AgentState` schema were the key design decisions — they made partial failures explicit and easy to handle.

**"How would you scale this?"**
API layer is stateless — scale horizontally behind a load balancer. Celery workers scale independently. ChromaDB swaps to Qdrant Cloud for production scale. Redis cluster for session memory.

---

## Future Extensions `<a name="future-extensions"></a>`

Once the core system is working, these extensions can meaningfully raise the project's ceiling:

**Fine-tuned reranker** — collect thumbs-up/down feedback from the UI and fine-tune the cross-encoder on your domain. Can improve context recall by 15–20% with ~200 labelled examples.

**Cost-aware LLM routing** — route simple queries to `gpt-4o-mini` and complex ones to `gpt-4o`. A lightweight query complexity classifier makes the decision. Target: 50–60% cost reduction with minimal quality loss.

**Agentic literature review** — chain the system to auto-generate a structured literature review: fetch papers → cluster by theme → extract contributions → write a report. Good for a demo to research-oriented companies.

**Voice interface** — add Whisper for speech input and a TTS output layer. Useful for mobile demos and accessibility.

**Multi-tenancy** — per-user ChromaDB namespaces, separate session stores per user, simple account system. Turns the project into a deployable product.

---

*Keep this file updated as you build — a roadmap that matches the actual code is itself a signal of engineering discipline.*

**Total timeline:** 4–5 weeks at 3–4 hrs/day

**Estimated API cost:** $30–60 total during development

**Goal:** Have a live demo URL ready before every interview.
