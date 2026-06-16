// frontend/lib/types.ts

export type JobStatus = "queued" | "running" | "done" | "error";

export interface ResearchJobResponse {
  job_id: string;
  session_id: string;
  status: JobStatus;
  message: string;
}

export interface Citation {
  title: string;
  authors: string[];
  url: string;
  arxiv_id: string;
  published: string;
  relevance_score: number;
  excerpt: string;
}

export interface AgentEvent {
  type: "job_started" | "agent_done" | "done" | "error";
  timestamp: number;
  agent?: string;
  status?: string;
  final_answer?: string;
  citations?: Citation[];
  agents_used?: string[];
  confidence_score?: number;
  critic_score?: number;        // 0–10 from Critic node
  message?: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  agentsUsed?: string[];
  confidenceScore?: number;
  criticScore?: number;         // shown as quality badge
  timestamp: number;
}

// ── Paper Summariser ──────────────────────────────────────────────────────────

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
  message: string;
}

export interface Session {
  session_id: string;
  query: string;
  created_at: string;
}
