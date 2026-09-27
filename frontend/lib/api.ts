// frontend/lib/api.ts

import type {
  IngestResult,
  KnowledgePaper,
  KnowledgeStats,
  PaperSummary,
  ReadyResponse,
  ResearchJobResponse,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "";

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
  }
}

/** FastAPI errors are `{detail: string}` or `{detail: [{msg}]}` for validation errors. */
function describeError(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown })?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail[0]?.msg) {
    return String(detail[0].msg).replace(/^Value error,\s*/, "");
  }
  return fallback;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/v1${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", "X-Api-Key": API_KEY, ...init.headers },
    });
  } catch {
    throw new ApiError(`Can't reach the API at ${API_URL}. Is the backend running?`);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(describeError(body, `${res.status} ${res.statusText}`), res.status);
  }
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

// ── Research ──────────────────────────────────────────────────────────────────

export const submitResearch = (query: string, sessionId: string, useWebSearch: boolean) =>
  post<ResearchJobResponse>("/research", {
    query,
    session_id: sessionId,
    use_web_search: useWebSearch,
  });

export const cancelResearch = (jobId: string) => post<{ status: string }>(`/research/${jobId}/cancel`);

export const streamUrl = (jobId: string) => `${API_URL}/api/v1/stream/${jobId}`;

export const forgetSession = (sessionId: string) =>
  request<{ deleted: boolean }>(`/sessions/${sessionId}`, { method: "DELETE" });

// ── System ────────────────────────────────────────────────────────────────────

export const getReady = () => request<ReadyResponse>("/ready");

// ── Knowledge base ────────────────────────────────────────────────────────────

export const getKnowledgeStats = () => request<KnowledgeStats>("/knowledge/stats");

export const listPapers = (q = "", limit = 200) =>
  request<{ total: number; papers: KnowledgePaper[] }>(
    `/knowledge/papers?limit=${limit}&q=${encodeURIComponent(q)}`,
  );

export const deletePaper = (arxivId: string) =>
  request<{ chunks_removed: number }>(`/knowledge/papers/${encodeURIComponent(arxivId)}`, {
    method: "DELETE",
  });

export const ingestArxiv = (query: string, maxPapers: number) =>
  post<IngestResult>("/ingest", { arxiv_query: query, max_papers: maxPapers });

export const summarisePaper = (arxivInput: string) =>
  post<PaperSummary>("/papers/summarise", { arxiv_input: arxivInput });
