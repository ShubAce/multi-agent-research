// frontend/components/SourceCitations.tsx
"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, ExternalLink } from "lucide-react";
import type { Citation } from "@/lib/types";

interface Props {
  citations: Citation[];
}

export function SourceCitations({ citations }: Props) {
  const [open, setOpen] = useState(false);

  if (citations.length === 0) return null;

  return (
    <div className="mt-3 rounded-lg border border-slate-200 overflow-hidden text-sm">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-2 bg-slate-50 hover:bg-slate-100 transition-colors text-slate-600"
      >
        <span className="font-medium">
          {citations.length} Source{citations.length !== 1 ? "s" : ""}
        </span>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {open && (
        <ul className="divide-y divide-slate-100">
          {citations.map((c, i) => (
            <li key={i} className="px-4 py-3 hover:bg-slate-50 transition-colors">
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-slate-800 truncate">{c.title}</p>
                  {c.authors.length > 0 && (
                    <p className="text-xs text-slate-500 mt-0.5">
                      {c.authors.slice(0, 3).join(", ")}
                      {c.authors.length > 3 ? " et al." : ""}
                      {c.published ? ` · ${c.published}` : ""}
                    </p>
                  )}
                  {c.excerpt && (
                    <p className="text-xs text-slate-500 mt-1 line-clamp-2">
                      {c.excerpt}
                    </p>
                  )}
                </div>

                <div className="flex-shrink-0 flex flex-col items-end gap-1">
                  <span className="text-xs text-slate-400 tabular-nums">
                    {Math.round(c.relevance_score * 100)}%
                  </span>
                  {c.url && (
                    <a
                      href={c.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-500 hover:text-blue-700"
                    >
                      <ExternalLink size={14} />
                    </a>
                  )}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
