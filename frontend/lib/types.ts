// frontend/lib/types.ts

export interface ResearchJobResponse {
  job_id: string;
  session_id: string;
  status: "queued" | "running" | "done" | "error";
  message: string;
}

/** One numbered source — `id` is the n in an answer's [n] citations. */
export interface Citation {
  id: number;
  type: "paper" | "web";
  title: string;
  authors: string[];
  url: string;
  arxiv_id: string;
  published: string;
  relevance_score: number;
  excerpt: string;
}

export interface FactCheck {
  checked: number;
  supported: number;
  inferred: number;
  removed: number;
  unverified?: boolean; // no claim could be matched to a source; answer left unchanged
  inferred_sentences: string[];
  removed_sentences: string[];
}

export type AgentName = "memory_injection" | "planner" | "parallel_agents" | "synthesiser" | "critic";

/** Payloads of the SSE events streamed from /api/v1/stream/{job_id}. */
export interface AgentDoneEvent {
  type: "agent_done";
  agent: AgentName;
  duration_ms: number;
  next_agent: AgentName | null;
  detail: Record<string, unknown>;
}

export interface DoneEvent {
  type: "done";
  final_answer: string;
  citations: Citation[];
  agents_used: string[];
  confidence_score: number | null;
  critic_score: number | null;
  critic_feedback: string | null;
  fact_check: FactCheck | null;
  follow_ups: string[];
  sub_tasks: string[];
  retries: number;
  web_fallback: boolean;
  duration_ms: number;
}

export interface PipelineStep {
  agent: AgentName;
  durationMs: number;
  detail: Record<string, unknown>;
}

export type RunStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Run {
  jobId?: string;
  status: RunStatus;
  steps: PipelineStep[];
  nextAgent: AgentName | null;
  queuedAt: number;
  startedAt?: number;
  finishedAt?: number;
  error?: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  // ── assistant only ──
  query?: string;
  run?: Run;
  citations?: Citation[];
  confidenceScore?: number | null;
  criticScore?: number | null;
  criticFeedback?: string | null;
  factCheck?: FactCheck | null;
  followUps?: string[];
  retries?: number;
  webFallback?: boolean;
  durationMs?: number;
}

export interface Conversation {
  id: string; // doubles as the backend session_id (short-term memory key)
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
}

// ── System status ────────────────────────────────────────────────────────────

export interface ReadyResponse {
  status: "ok" | "degraded";
  services: Record<string, string>;
  version: string;
  features: { llm?: boolean; web_search?: boolean };
  knowledge_chunks: number | null;
}

// ── Knowledge base ───────────────────────────────────────────────────────────

export interface KnowledgePaper {
  arxiv_id: string;
  title: string;
  authors: string[];
  published: string;
  url: string;
  categories: string;
  chunks: number;
}

export interface KnowledgeStats {
  collection: string;
  chunks: number;
  papers: number;
}

export interface IngestResult {
  status: string;
  papers_indexed: number;
  papers_found: number;
  papers_skipped: number;
  chunks_indexed: number;
  total_chunks: number;
  collection: string;
}

export interface PaperSummary {
  arxiv_id: string;
  title: string;
  authors: string[];
  published: string;
  url: string;
  one_liner: string;
  problem: string;
  method: string;
  key_results: string;
  limitations: string;
  contributions: string[];
  related_work: string[];
  nodes_ingested: number;
  already_indexed: boolean;
  message: string;
}
