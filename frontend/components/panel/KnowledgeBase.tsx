// frontend/components/panel/KnowledgeBase.tsx
"use client";

import { useCallback, useEffect, useState, type FormEvent } from "react";
import { CircleAlert, CircleCheck, ExternalLink, Link2, Plus, RefreshCw, Search, Trash2 } from "lucide-react";
import { PaperSummaryCard } from "@/components/panel/PaperSummaryCard";
import { Button, IconButton, SectionLabel, Spinner } from "@/components/ui";
import { useSystemStatus } from "@/hooks/useSystemStatus";
import { deletePaper, getKnowledgeStats, ingestArxiv, listPapers, summarisePaper } from "@/lib/api";
import type { KnowledgePaper, KnowledgeStats, PaperSummary } from "@/lib/types";
import { cn, shortAuthors } from "@/lib/utils";

type Notice = { tone: "success" | "error"; text: string } | null;

function NoticeLine({ notice }: { notice: Notice }) {
  if (!notice) return null;
  const Icon = notice.tone === "success" ? CircleCheck : CircleAlert;
  return (
    <p
      className={cn(
        "flex gap-1.5 rounded-lg px-2.5 py-2 text-xs leading-relaxed",
        notice.tone === "success" ? "bg-success-soft text-success" : "bg-danger-soft text-danger",
      )}
    >
      <Icon size={13} className="mt-px shrink-0" />
      <span className="break-words">{notice.text}</span>
    </p>
  );
}

const inputClass =
  "h-9 w-full rounded-lg border border-border bg-surface px-3 text-[13px] text-fg outline-none transition-colors placeholder:text-fg-subtle focus:border-accent/50";

function AddByTopic({ onDone }: { onDone: () => void }) {
  const [topic, setTopic] = useState("");
  const [count, setCount] = useState(10);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState<Notice>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (topic.trim().length < 3 || loading) return;
    setLoading(true);
    setNotice(null);
    try {
      const r = await ingestArxiv(topic.trim(), count);
      const skipped = r.papers_skipped ? ` · ${r.papers_skipped} already indexed` : "";
      setNotice({
        tone: "success",
        text: r.papers_found
          ? `Added ${r.papers_indexed} paper${r.papers_indexed === 1 ? "" : "s"} (${r.chunks_indexed} passages)${skipped}.`
          : "ArXiv returned no papers for that topic.",
      });
      setTopic("");
      onDone();
    } catch (err) {
      setNotice({ tone: "error", text: err instanceof Error ? err.message : "Ingestion failed." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-2">
      <SectionLabel>Add papers by topic</SectionLabel>
      <input
        value={topic}
        onChange={(e) => setTopic(e.target.value)}
        placeholder="e.g. mixture of experts routing"
        className={inputClass}
        disabled={loading}
      />
      <div className="flex gap-2">
        <select
          value={count}
          onChange={(e) => setCount(Number(e.target.value))}
          disabled={loading}
          aria-label="Number of papers"
          className={cn(inputClass, "w-auto pr-8")}
        >
          {[5, 10, 20, 30].map((n) => (
            <option key={n} value={n}>
              {n} papers
            </option>
          ))}
        </select>
        <Button type="submit" variant="primary" className="flex-1" loading={loading} disabled={topic.trim().length < 3}>
          {!loading && <Plus size={14} />}
          {loading ? "Fetching from arXiv…" : "Add to library"}
        </Button>
      </div>
      <NoticeLine notice={notice} />
    </form>
  );
}

function AddByLink({ onDone, onAsk }: { onDone: () => void; onAsk: (q: string) => void }) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState<Notice>(null);
  const [summary, setSummary] = useState<PaperSummary | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (input.trim().length < 5 || loading) return;
    setLoading(true);
    setNotice(null);
    try {
      setSummary(await summarisePaper(input.trim()));
      setInput("");
      onDone();
    } catch (err) {
      setNotice({ tone: "error", text: err instanceof Error ? err.message : "Could not summarise that paper." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-2">
      <form onSubmit={submit} className="space-y-2">
        <SectionLabel>Summarise &amp; add one paper</SectionLabel>
        <div className="flex gap-2">
          <label className="relative flex-1">
            <Link2 size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-fg-subtle" />
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="arxiv.org/abs/2205.14135"
              className={cn(inputClass, "pl-8")}
              disabled={loading}
            />
          </label>
          <Button type="submit" loading={loading} disabled={input.trim().length < 5}>
            {loading ? "Reading…" : "Summarise"}
          </Button>
        </div>
      </form>
      <NoticeLine notice={notice} />
      {summary && <PaperSummaryCard summary={summary} onClose={() => setSummary(null)} onAsk={onAsk} />}
    </div>
  );
}

function PaperRow({ paper, onRemoved }: { paper: KnowledgePaper; onRemoved: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);

  const remove = async () => {
    setBusy(true);
    try {
      await deletePaper(paper.arxiv_id);
      onRemoved();
    } catch {
      setBusy(false);
      setConfirming(false);
    }
  };

  return (
    <li className="group flex items-start gap-2 px-1 py-2.5">
      <div className="min-w-0 flex-1">
        <a
          href={paper.url}
          target="_blank"
          rel="noopener noreferrer"
          className="line-clamp-2 text-[13px] font-medium leading-snug text-fg hover:text-accent-text"
        >
          {paper.title}
        </a>
        <p className="mt-0.5 truncate text-[11px] text-fg-subtle">
          {[shortAuthors(paper.authors), paper.published?.slice(0, 4), `${paper.chunks} passages`]
            .filter(Boolean)
            .join(" · ")}
        </p>
      </div>
      {confirming ? (
        <div className="flex shrink-0 items-center gap-1">
          <Button size="sm" variant="danger" onClick={remove} loading={busy}>
            Remove
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setConfirming(false)} disabled={busy}>
            Cancel
          </Button>
        </div>
      ) : (
        <div className="flex shrink-0 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
          <IconButton label="Open on arXiv" className="h-7 w-7" onClick={() => window.open(paper.url, "_blank", "noopener")}>
            <ExternalLink size={13} />
          </IconButton>
          <IconButton label="Remove from library" className="h-7 w-7 hover:text-danger" onClick={() => setConfirming(true)}>
            <Trash2 size={13} />
          </IconButton>
        </div>
      )}
    </li>
  );
}

export function KnowledgeBase({ onAsk }: { onAsk: (q: string) => void }) {
  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [papers, setPapers] = useState<KnowledgePaper[]>([]);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const refreshStatus = useSystemStatus((s) => s.refresh);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [s, list] = await Promise.all([getKnowledgeStats(), listPapers()]);
      setStats(s);
      setPapers(list.papers);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the knowledge base.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const changed = useCallback(() => {
    load();
    refreshStatus();
  }, [load, refreshStatus]);

  const needle = filter.trim().toLowerCase();
  const visible = needle
    ? papers.filter((p) => p.title.toLowerCase().includes(needle) || p.authors.some((a) => a.toLowerCase().includes(needle)))
    : papers;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-sm font-medium text-fg">Knowledge base</p>
          <p className="mt-0.5 text-xs text-fg-muted">
            {stats
              ? `${stats.papers.toLocaleString()} papers · ${stats.chunks.toLocaleString()} passages indexed`
              : loading
                ? "Loading…"
                : "Unavailable"}
          </p>
        </div>
        <IconButton label="Refresh" onClick={changed} disabled={loading}>
          <RefreshCw size={14} className={cn(loading && "animate-spin")} />
        </IconButton>
      </div>

      <AddByTopic onDone={changed} />
      <AddByLink onDone={changed} onAsk={onAsk} />

      <div className="space-y-2">
        <SectionLabel>Papers</SectionLabel>
        {papers.length > 6 && (
          <label className="relative block">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-fg-subtle" />
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter by title or author"
              className={cn(inputClass, "pl-8")}
            />
          </label>
        )}
        {error ? (
          <NoticeLine notice={{ tone: "error", text: error }} />
        ) : loading && !papers.length ? (
          <div className="flex justify-center py-6">
            <Spinner />
          </div>
        ) : !papers.length ? (
          <p className="rounded-lg border border-dashed border-border px-3 py-6 text-center text-xs leading-relaxed text-fg-subtle">
            No papers yet. Add a topic above to fetch papers from arXiv.
          </p>
        ) : (
          <ul className="divide-y divide-border">
            {visible.map((p) => (
              <PaperRow key={p.arxiv_id} paper={p} onRemoved={changed} />
            ))}
            {!visible.length && <li className="py-4 text-center text-xs text-fg-subtle">No matches</li>}
          </ul>
        )}
      </div>
    </div>
  );
}
