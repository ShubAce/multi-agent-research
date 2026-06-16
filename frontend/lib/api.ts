// frontend/lib/api.ts

import type { ResearchJobResponse, PaperSummary } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "";

function headers(): HeadersInit {
  return {
    "Content-Type": "application/json",
    "X-Api-Key": API_KEY,
  };
}

export async function submitResearch(
  query: string,
  sessionId?: string,
): Promise<ResearchJobResponse> {
  const res = await fetch(`${API_URL}/api/v1/research`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ query, session_id: sessionId, domain: "arxiv" }),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail ?? "Failed to submit research query");
  }
  return res.json();
}

export async function summarisePaper(arxivInput: string): Promise<PaperSummary> {
  const res = await fetch(`${API_URL}/api/v1/papers/summarise`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ arxiv_input: arxivInput }),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail ?? "Failed to summarise paper");
  }
  return res.json();
}

export async function ingestArxiv(query: string, maxPapers = 20) {
  const res = await fetch(`${API_URL}/api/v1/ingest`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ arxiv_query: query, max_papers: maxPapers }),
  });
  if (!res.ok) throw new Error("Ingestion failed");
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${API_URL}/api/v1/health`);
  return res.json();
}
