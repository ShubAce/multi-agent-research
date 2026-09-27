// frontend/components/Markdown.tsx — answer renderer with inline citation chips
"use client";

import { createContext, memo, useContext, useMemo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText, Globe } from "lucide-react";
import type { Citation } from "@/lib/types";
import { hostname } from "@/lib/utils";

interface CiteContextValue {
  citations: Citation[];
  onCite?: (n: number) => void;
}

const CiteContext = createContext<CiteContextValue>({ citations: [] });

function expand(inner: string): number[] {
  const nums: number[] = [];
  for (const part of inner.split(",")) {
    const [a, b] = part.split(/[–-]/).map((x) => parseInt(x.trim(), 10));
    if (Number.isNaN(a)) continue;
    if (b && b > a && b - a <= 10) for (let n = a; n <= b; n++) nums.push(n);
    else nums.push(a);
  }
  return nums;
}

/** No maths renderer is bundled, so show stray LaTeX (\( \), \[ \]) as code instead of raw backslashes. */
function plainMath(text: string): string {
  return text
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, m: string) => `\n\`\`\`\n${m.trim()}\n\`\`\`\n`)
    .replace(/\\\((.+?)\\\)/g, (_, m: string) => `\`${m.trim()}\``);
}

/** "[1]", "[1, 3]" and "[2–4]" become links to #cite-n (rendered as chips). */
export function linkCitations(text: string, max: number): string {
  return text.replace(/\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()/g, (match, inner: string) => {
    const nums = expand(inner).filter((n) => n >= 1 && n <= max);
    return nums.length ? nums.map((n) => `[${n}](#cite-${n})`).join("") : match;
  });
}

function CitationChip({ n }: { n: number }) {
  const { citations, onCite } = useContext(CiteContext);
  const c = citations[n - 1];
  const Icon = c?.type === "web" ? Globe : FileText;
  return (
    <span className="group/cite relative mx-px inline-block align-[1px]">
      <button
        type="button"
        onClick={() => onCite?.(n)}
        className="cite inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-[5px] bg-surface-2 px-1 text-[11px] font-semibold leading-none text-fg-muted ring-1 ring-inset ring-border transition-colors hover:bg-accent-soft hover:text-accent-text hover:ring-accent/30"
        aria-label={c ? `Source ${n}: ${c.title}` : `Source ${n}`}
      >
        {n}
      </button>
      {c && (
        <span
          role="tooltip"
          // display:none (not visibility:hidden) so hidden cards don't widen the scroll area
          className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 hidden w-72 -translate-x-1/2 animate-fade-in rounded-lg border border-border bg-surface p-3 text-left shadow-pop group-hover/cite:block"
        >
          <span className="flex items-center gap-1.5 text-[11px] font-medium text-fg-subtle">
            <Icon size={12} />
            {c.type === "web" ? hostname(c.url) : `arXiv${c.published ? ` · ${c.published.slice(0, 4)}` : ""}`}
          </span>
          <span className="mt-1 line-clamp-2 block text-[13px] font-medium leading-snug text-fg">{c.title}</span>
          {c.excerpt && (
            <span className="mt-1 line-clamp-3 block text-xs leading-relaxed text-fg-muted">{c.excerpt}</span>
          )}
        </span>
      )}
    </span>
  );
}

const components: Components = {
  a({ href, children }) {
    const cite = href?.match(/^#cite-(\d+)$/);
    if (cite) return <CitationChip n={Number(cite[1])} />;
    return (
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    );
  },
  table({ children }) {
    return (
      <div className="overflow-x-auto rounded-lg border border-border">
        <table>{children}</table>
      </div>
    );
  },
};

export const Markdown = memo(function Markdown({
  content,
  citations = [],
  onCite,
}: {
  content: string;
  citations?: Citation[];
  onCite?: (n: number) => void;
}) {
  const text = useMemo(
    () => linkCitations(plainMath(content), citations.length),
    [content, citations.length],
  );
  const ctx = useMemo(() => ({ citations, onCite }), [citations, onCite]);
  return (
    <CiteContext.Provider value={ctx}>
      <div className="answer">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {text}
        </ReactMarkdown>
      </div>
    </CiteContext.Provider>
  );
});
