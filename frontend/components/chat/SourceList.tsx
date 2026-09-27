// frontend/components/chat/SourceList.tsx
"use client";

import { ChevronDown, FileText, Globe } from "lucide-react";
import type { Citation } from "@/lib/types";
import { cn, hostname, shortAuthors } from "@/lib/utils";

export const COLLAPSED_SOURCES = 4;

export const sourceElementId = (messageId: string, n: number) => `src-${messageId}-${n}`;

function SourceCard({ c, messageId }: { c: Citation; messageId: string }) {
  const Icon = c.type === "web" ? Globe : FileText;
  const meta =
    c.type === "web"
      ? hostname(c.url)
      : [shortAuthors(c.authors), c.published?.slice(0, 4)].filter(Boolean).join(" · ") || "arXiv";

  return (
    <a
      id={sourceElementId(messageId, c.id)}
      href={c.url || undefined}
      target="_blank"
      rel="noopener noreferrer"
      title={c.excerpt || c.title}
      className="group flex scroll-mt-24 gap-3 rounded-lg border border-border bg-surface p-3 transition-colors hover:border-fg-subtle/40 hover:bg-surface-2/60"
    >
      <span className="flex h-5 min-w-5 shrink-0 items-center justify-center rounded-md bg-surface-2 px-1 text-[11px] font-semibold text-fg-muted ring-1 ring-inset ring-border">
        {c.id}
      </span>
      <span className="min-w-0 flex-1">
        <span className="line-clamp-2 text-[13px] font-medium leading-snug text-fg group-hover:text-accent-text">
          {c.title}
        </span>
        <span className="mt-1 flex items-center gap-1.5 text-xs text-fg-subtle">
          <Icon size={12} className="shrink-0" />
          <span className="truncate">{meta}</span>
        </span>
      </span>
    </a>
  );
}

export function SourceList({
  citations,
  messageId,
  expanded,
  onToggle,
}: {
  citations: Citation[];
  messageId: string;
  expanded: boolean;
  onToggle: () => void;
}) {
  if (!citations.length) return null;
  const shown = expanded ? citations : citations.slice(0, COLLAPSED_SOURCES);
  const hidden = citations.length - shown.length;
  const papers = citations.filter((c) => c.type === "paper").length;
  const web = citations.length - papers;

  return (
    <section>
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs font-medium text-fg-muted">
          Sources
          <span className="ml-1.5 font-normal text-fg-subtle">
            {[papers && `${papers} paper${papers > 1 ? "s" : ""}`, web && `${web} web`].filter(Boolean).join(" · ")}
          </span>
        </p>
        {citations.length > COLLAPSED_SOURCES && (
          <button
            onClick={onToggle}
            className="flex items-center gap-1 text-xs font-medium text-fg-muted hover:text-fg"
          >
            {expanded ? "Show fewer" : `Show all ${citations.length}`}
            <ChevronDown size={14} className={cn("transition-transform", expanded && "rotate-180")} />
          </button>
        )}
      </div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {shown.map((c) => (
          <SourceCard key={c.id} c={c} messageId={messageId} />
        ))}
      </div>
      {hidden > 0 && !expanded && (
        <button onClick={onToggle} className="mt-2 text-xs text-fg-subtle hover:text-fg-muted">
          +{hidden} more
        </button>
      )}
    </section>
  );
}
