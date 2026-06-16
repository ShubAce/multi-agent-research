// frontend/components/PaperSummariser.tsx
"use client";

import { useState } from "react";
import { BookOpen, ExternalLink, Loader2, ChevronDown, ChevronUp } from "lucide-react";
import { summarisePaper } from "@/lib/api";
import type { PaperSummary } from "@/lib/types";

export function PaperSummariser() {
  const [input, setInput]     = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [summary, setSummary] = useState<PaperSummary | null>(null);
  const [expanded, setExpanded] = useState(false);

  const handleSubmit = async () => {
    if (!input.trim() || loading) return;
    setLoading(true);
    setError(null);
    setSummary(null);

    try {
      const result = await summarisePaper(input.trim());
      setSummary(result);
      setExpanded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to summarise paper");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <BookOpen size={14} className="text-slate-500" />
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Paper Summariser
        </p>
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          placeholder="arxiv.org/abs/2205.14135"
          disabled={loading}
          className="flex-1 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700 placeholder-slate-400 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200 disabled:opacity-50"
        />
        <button
          onClick={handleSubmit}
          disabled={loading || !input.trim()}
          className="flex-shrink-0 rounded-lg bg-blue-600 px-3 py-2 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? <Loader2 size={12} className="animate-spin" /> : "Summarise"}
        </button>
      </div>

      {error && (
        <p className="text-xs text-red-500 bg-red-50 rounded-lg px-3 py-2">{error}</p>
      )}

      {/* Summary card */}
      {summary && (
        <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
          {/* Header */}
          <div className="px-4 py-3 bg-slate-50 border-b border-slate-100">
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-slate-800 leading-snug line-clamp-2">
                  {summary.title}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  {summary.authors.slice(0, 2).join(", ")}
                  {summary.authors.length > 2 ? " et al." : ""}
                  {" · "}{summary.published}
                </p>
              </div>
              <a
                href={summary.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-shrink-0 text-blue-500 hover:text-blue-700"
              >
                <ExternalLink size={14} />
              </a>
            </div>

            {/* One-liner */}
            <p className="mt-2 text-xs text-blue-700 bg-blue-50 rounded-lg px-3 py-2 leading-relaxed">
              {summary.one_liner}
            </p>

            {/* Ingestion badge */}
            {summary.nodes_ingested > 0 && (
              <p className="mt-2 text-xs text-green-600">
                ✓ Added to knowledge base ({summary.nodes_ingested} chunks)
              </p>
            )}
          </div>

          {/* Toggle full summary */}
          <button
            onClick={() => setExpanded((v) => !v)}
            className="w-full flex items-center justify-between px-4 py-2 text-xs text-slate-500 hover:bg-slate-50 transition-colors"
          >
            <span>{expanded ? "Hide" : "Show"} full summary</span>
            {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>

          {expanded && (
            <div className="px-4 pb-4 space-y-3 text-xs text-slate-700">
              <Section label="Problem" content={summary.problem} />
              <Section label="Method" content={summary.method} />
              <Section label="Key Results" content={summary.key_results} />
              <Section label="Limitations" content={summary.limitations} />

              {summary.contributions.length > 0 && (
                <div>
                  <p className="font-semibold text-slate-600 mb-1">Contributions</p>
                  <ul className="space-y-1">
                    {summary.contributions.map((c, i) => (
                      <li key={i} className="flex gap-1.5">
                        <span className="text-blue-400 flex-shrink-0">•</span>
                        <span>{c}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {summary.related_work.length > 0 && (
                <div>
                  <p className="font-semibold text-slate-600 mb-1">Builds On</p>
                  <div className="flex flex-wrap gap-1">
                    {summary.related_work.map((r, i) => (
                      <span
                        key={i}
                        className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full text-xs"
                      >
                        {r}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Section({ label, content }: { label: string; content: string }) {
  if (!content) return null;
  return (
    <div>
      <p className="font-semibold text-slate-600 mb-0.5">{label}</p>
      <p className="text-slate-600 leading-relaxed">{content}</p>
    </div>
  );
}
