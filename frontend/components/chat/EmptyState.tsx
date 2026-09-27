// frontend/components/chat/EmptyState.tsx
"use client";

import { ArrowRight, BookOpen, Globe, Layers, ShieldCheck } from "lucide-react";
import type { ReadyResponse } from "@/lib/types";

const EXAMPLES = [
  { q: "How does FlashAttention reduce the memory cost of attention?", tag: "Mechanism" },
  { q: "Compare RLHF and DPO for aligning language models", tag: "Comparison" },
  { q: "What are the main failure modes of retrieval-augmented generation?", tag: "Limitations" },
  { q: "What are the latest advances in diffusion models for image generation?", tag: "Recent work" },
];

const FEATURES = [
  { icon: Layers, text: "Planner splits your question; paper and web agents research in parallel" },
  { icon: BookOpen, text: "Every claim cites a numbered source you can open" },
  { icon: ShieldCheck, text: "Unsupported sentences are fact-checked out; a critic scores the result" },
];

export function EmptyState({
  onAsk,
  status,
}: {
  onAsk: (q: string) => void;
  status: ReadyResponse | null;
}) {
  const chunks = status?.knowledge_chunks;
  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col px-4 pb-10 pt-[12vh] sm:px-6">
      <h1 className="text-2xl font-semibold tracking-tight text-fg sm:text-[28px]">
        What would you like to research?
      </h1>
      <p className="mt-2 max-w-xl text-[15px] leading-relaxed text-fg-muted">
        Ask about machine-learning research. Answers are grounded in ArXiv papers from your knowledge base
        and, when enabled, the web — follow-up questions keep the context.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {EXAMPLES.map(({ q, tag }) => (
          <button
            key={q}
            onClick={() => onAsk(q)}
            className="group flex flex-col items-start gap-2 rounded-xl border border-border bg-surface p-4 text-left shadow-card transition-colors hover:border-accent/40 hover:bg-surface-2/50"
          >
            <span className="text-[11px] font-semibold uppercase tracking-wider text-fg-subtle">{tag}</span>
            <span className="text-sm leading-snug text-fg">{q}</span>
            <ArrowRight
              size={14}
              className="mt-auto text-fg-subtle transition-transform group-hover:translate-x-0.5 group-hover:text-accent-text"
            />
          </button>
        ))}
      </div>

      <ul className="mt-8 space-y-2.5">
        {FEATURES.map(({ icon: Icon, text }) => (
          <li key={text} className="flex items-start gap-2.5 text-sm text-fg-muted">
            <Icon size={16} className="mt-0.5 shrink-0 text-fg-subtle" />
            {text}
          </li>
        ))}
        {chunks != null && (
          <li className="flex items-start gap-2.5 text-sm text-fg-muted">
            <Globe size={16} className="mt-0.5 shrink-0 text-fg-subtle" />
            {chunks > 0
              ? `Knowledge base: ${chunks.toLocaleString()} indexed passages.`
              : "Your knowledge base is empty — add ArXiv papers from the Library panel, or rely on web search."}
          </li>
        )}
      </ul>
    </div>
  );
}
