"""
app/core/metrics.py
Prometheus counters / histograms — imported by agent nodes and API handlers.
"""

from prometheus_client import Counter, Histogram, Gauge

# ── API ───────────────────────────────────────────────────────────────────────
research_requests_total = Counter(
    "research_requests_total",
    "Total research job submissions",
    ["status"],          # queued | error
)

# ── Agents ────────────────────────────────────────────────────────────────────
agent_runs_total = Counter(
    "agent_runs_total",
    "Total individual agent node executions",
    ["agent_name", "status"],   # status: success | error
)

agent_duration_seconds = Histogram(
    "agent_duration_seconds",
    "Time spent inside each agent node",
    ["agent_name"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

# ── LLM ──────────────────────────────────────────────────────────────────────
llm_tokens_total = Counter(
    "llm_tokens_total",
    "LLM tokens consumed",
    ["model", "token_type"],    # token_type: prompt | completion
)

# ── RAG ──────────────────────────────────────────────────────────────────────
vector_db_query_seconds = Histogram(
    "vector_db_query_seconds",
    "ChromaDB query latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2],
)

documents_retrieved_total = Counter(
    "documents_retrieved_total",
    "Total source nodes returned by the RAG agent",
)

# ── Sessions ─────────────────────────────────────────────────────────────────
active_sessions = Gauge(
    "active_sessions",
    "Currently active research sessions (approximate)",
)
