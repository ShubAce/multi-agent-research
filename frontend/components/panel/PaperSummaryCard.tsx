// frontend/components/panel/PaperSummaryCard.tsx
"use client";

import { useState } from "react";
import { ChevronDown, ExternalLink, X } from "lucide-react";
import { Badge, IconButton } from "@/components/ui";
import type { PaperSummary } from "@/lib/types";
import { cn, shortAuthors } from "@/lib/utils";

function Field({ label, text }: { label: string; text: string }) {
  if (!text) return null;
  return (
    <div>
      <p className="mb-0.5 text-[11px] font-semibold uppercase tracking-wider text-fg-subtle">{label}</p>
      <p className="text-xs leading-relaxed text-fg-muted">{text}</p>
    </div>
  );
}

export function PaperSummaryCard({
  summary,
  onClose,
  onAsk,
}: {
  summary: PaperSummary;
  onClose: () => void;
  onAsk: (q: string) => void;
}) {
  const [open, setOpen] = useState(true);

  return (
    <div className="animate-fade-in overflow-hidden rounded-xl border border-border bg-surface shadow-card">
      <div className="p-3.5">
        <div className="flex items-start gap-2">
          <a
            href={summary.url}
            target="_blank"
            rel="noopener noreferrer"
            className="group min-w-0 flex-1 text-[13px] font-semibold leading-snug text-fg hover:text-accent-text"
          >
            {summary.title}
            <ExternalLink size={11} className="ml-1 inline align-baseline text-fg-subtle group-hover:text-accent-text" />
          </a>
          <IconButton label="Dismiss" className="-mr-1 -mt-1 h-7 w-7" onClick={onClose}>
            <X size={14} />
          </IconButton>
        </div>
        <p className="mt-1 text-xs text-fg-subtle">
          {shortAuthors(summary.authors, 3)} · {summary.published}
        </p>
        <p className="mt-2.5 rounded-lg bg-accent-soft px-3 py-2 text-xs leading-relaxed text-accent-text">
          {summary.one_liner}
        </p>
        <div className="mt-2.5 flex items-center gap-2">
          <Badge tone={summary.nodes_ingested > 0 || summary.already_indexed ? "success" : "warning"}>
            {summary.message}
          </Badge>
        </div>
      </div>

      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between border-t border-border px-3.5 py-2 text-xs font-medium text-fg-muted hover:bg-surface-2/60"
      >
        Structured summary
        <ChevronDown size={14} className={cn("transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="space-y-3 border-t border-border px-3.5 py-3">
          <Field label="Problem" text={summary.problem} />
          <Field label="Method" text={summary.method} />
          <Field label="Key results" text={summary.key_results} />
          <Field label="Limitations" text={summary.limitations} />
          {summary.contributions.length > 0 && (
            <div>
              <p className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-fg-subtle">Contributions</p>
              <ul className="list-disc space-y-1 pl-4 text-xs leading-relaxed text-fg-muted">
                {summary.contributions.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </div>
          )}
          {summary.related_work.length > 0 && (
            <div>
              <p className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-fg-subtle">Builds on</p>
              <div className="flex flex-wrap gap-1">
                {summary.related_work.map((r, i) => (
                  <Badge key={i}>{r}</Badge>
                ))}
              </div>
            </div>
          )}
          <button
            onClick={() => onAsk(`What are the key ideas and results of the paper "${summary.title}"?`)}
            className="text-xs font-medium text-accent-text hover:underline"
          >
            Ask a question about this paper →
          </button>
        </div>
      )}
    </div>
  );
}
