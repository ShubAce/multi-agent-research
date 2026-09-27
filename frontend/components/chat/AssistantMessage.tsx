// frontend/components/chat/AssistantMessage.tsx
"use client";

import { useCallback, useState } from "react";
import {
  ArrowRight,
  Check,
  ChevronDown,
  CircleAlert,
  CircleCheck,
  Copy,
  Download,
  Gauge,
  Globe,
  ListTree,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Square,
  TriangleAlert,
} from "lucide-react";
import { Markdown } from "@/components/Markdown";
import { Badge, Button, IconButton, Spinner } from "@/components/ui";
import { COLLAPSED_SOURCES, SourceList, sourceElementId } from "@/components/chat/SourceList";
import { useNow } from "@/hooks/useNow";
import { answerToMarkdown, downloadText, slug } from "@/lib/export";
import { STAGES, currentActivity, summariseStep } from "@/lib/pipeline";
import type { FactCheck, Message, Run } from "@/lib/types";
import { cn, formatDuration } from "@/lib/utils";
import { useChatStore } from "@/store/chat";

const WORKER_HINT_AFTER_MS = 15_000;

// ── Live progress while the agents work ──────────────────────────────────────

function LiveProgress({ run, onStop }: { run: Run; onStop: () => void }) {
  const now = useNow(true);
  const elapsed = now - (run.startedAt ?? run.queuedAt);
  const stuckInQueue = run.status === "queued" && now - run.queuedAt > WORKER_HINT_AFTER_MS;

  return (
    <div className="rounded-xl border border-border bg-surface p-4 shadow-card">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <Spinner size={16} className="text-accent-text" />
          <span className="truncate text-sm font-medium text-fg">{currentActivity(run)}…</span>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span className="text-xs tabular-nums text-fg-subtle">{formatDuration(elapsed)}</span>
          <Button size="sm" variant="ghost" onClick={onStop}>
            <Square size={12} className="fill-current" />
            Stop
          </Button>
        </div>
      </div>

      {run.steps.length > 0 && (
        <ol className="mt-3 space-y-1.5 border-t border-border pt-3">
          {run.steps.map((step, i) => {
            const meta = STAGES[step.agent];
            const Icon = meta?.icon ?? CircleCheck;
            return (
              <li key={i} className="flex animate-fade-in items-center gap-2.5 text-[13px]">
                <Icon size={14} className="shrink-0 text-fg-subtle" />
                <span className="font-medium text-fg-muted">{meta?.label ?? step.agent}</span>
                <span className="truncate text-fg-subtle">{summariseStep(step)}</span>
                <span className="ml-auto shrink-0 text-xs tabular-nums text-fg-subtle">
                  {formatDuration(step.durationMs)}
                </span>
              </li>
            );
          })}
        </ol>
      )}

      {stuckInQueue && (
        <p className="mt-3 flex gap-2 rounded-lg bg-warning-soft px-3 py-2 text-xs leading-relaxed text-warning">
          <TriangleAlert size={14} className="mt-px shrink-0" />
          No worker has picked this up yet. Make sure the Celery worker is running (
          <code className="font-mono">python run_dev.py</code>).
        </p>
      )}

      <div className="mt-4 space-y-2" aria-hidden>
        <div className="skeleton h-3 w-11/12" />
        <div className="skeleton h-3 w-full" />
        <div className="skeleton h-3 w-4/6" />
      </div>
    </div>
  );
}

// ── Quality signals ──────────────────────────────────────────────────────────

function QualityBar({ message }: { message: Message }) {
  const { confidenceScore: conf, criticScore: score } = message;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {conf != null && (
        <Badge
          tone={conf >= 0.8 ? "success" : conf >= 0.5 ? "warning" : "danger"}
          title="Share of checked claims that the fact-checker found supported by the sources"
        >
          <ShieldCheck size={12} />
          {Math.round(conf * 100)}% grounded
        </Badge>
      )}
      {score != null && (
        <Badge
          tone={score >= 8 ? "success" : score >= 6 ? "neutral" : "warning"}
          title={message.criticFeedback ? `Critic: ${message.criticFeedback}` : "Critic score"}
        >
          <Gauge size={12} />
          Quality {score}/10
        </Badge>
      )}
      {!!message.retries && (
        <Badge tone="accent" title="The critic judged the first draft weak, so the agents researched again">
          <RotateCcw size={12} />
          Refined
        </Badge>
      )}
      {message.webFallback && (
        <Badge tone="warning" title="Nothing relevant in the knowledge base, so this was answered from the web">
          <Globe size={12} />
          Web fallback
        </Badge>
      )}
      {message.durationMs != null && (
        <span className="ml-1 text-xs tabular-nums text-fg-subtle">{formatDuration(message.durationMs)}</span>
      )}
    </div>
  );
}

function FactCheckReport({ report }: { report: FactCheck }) {
  const [open, setOpen] = useState(false);
  const issues = report.unverified ? 0 : report.inferred + report.removed;
  return (
    <div className="rounded-lg border border-border">
      <button
        onClick={() => setOpen((v) => !v)}
        disabled={!issues}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-fg-muted enabled:hover:bg-surface-2/60"
      >
        {issues || report.unverified ? (
          <CircleAlert size={14} className="shrink-0 text-warning" />
        ) : (
          <CircleCheck size={14} className="shrink-0 text-success" />
        )}
        <span className="shrink-0 whitespace-nowrap font-medium text-fg">Fact-check</span>
        <span>
          {report.unverified
            ? `None of the ${report.checked} checked claims could be matched to a source`
            : issues
            ? [
                `${report.supported} supported`,
                report.inferred && `${report.inferred} inferred`,
                report.removed && `${report.removed} removed`,
              ]
                .filter(Boolean)
                .join(" · ")
            : `All ${report.checked} checked claims are supported by the sources`}
        </span>
        {!!issues && <ChevronDown size={14} className={cn("ml-auto transition-transform", open && "rotate-180")} />}
      </button>
      {open && (
        <div className="space-y-3 border-t border-border px-3 py-3 text-xs leading-relaxed">
          {report.removed_sentences.length > 0 && (
            <div>
              <p className="mb-1 font-medium text-danger">Removed — not supported by any source</p>
              <ul className="space-y-1 text-fg-muted">
                {report.removed_sentences.map((s, i) => (
                  <li key={i} className="line-through decoration-danger/40">{s}</li>
                ))}
              </ul>
            </div>
          )}
          {report.inferred_sentences.length > 0 && (
            <div>
              <p className="mb-1 font-medium text-warning">Inferred — goes beyond what the sources state</p>
              <ul className="list-disc space-y-1 pl-4 text-fg-muted">
                {report.inferred_sentences.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function FollowUps({ items, onAsk }: { items: string[]; onAsk: (q: string) => void }) {
  return (
    <section>
      <p className="mb-1.5 text-xs font-medium text-fg-muted">Follow up</p>
      <ul className="divide-y divide-border rounded-lg border border-border">
        {items.map((q) => (
          <li key={q}>
            <button
              onClick={() => onAsk(q)}
              className="group flex w-full items-center gap-3 px-3 py-2.5 text-left text-sm text-fg-muted transition-colors hover:bg-surface-2/60 hover:text-fg"
            >
              <span className="flex-1">{q}</span>
              <ArrowRight size={14} className="shrink-0 text-fg-subtle transition-transform group-hover:translate-x-0.5" />
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

// ── Message ──────────────────────────────────────────────────────────────────

interface Props {
  message: Message;
  isLatest: boolean;
  busy: boolean;
  onAsk: (q: string) => void;
  onStop: () => void;
}

export function AssistantMessage({ message, isLatest, busy, onAsk, onStop }: Props) {
  const inspectMessage = useChatStore((s) => s.inspectMessage);
  const setPanelTab = useChatStore((s) => s.setPanelTab);
  const inspected = useChatStore((s) => s.inspectedMessageId === message.id);
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const run = message.run;
  const status = run?.status ?? "done";
  const citations = message.citations ?? [];
  const question = message.query ?? "";

  const onCite = useCallback(
    (n: number) => {
      if (n > COLLAPSED_SOURCES) setSourcesExpanded(true);
      setTimeout(() => {
        const el = document.getElementById(sourceElementId(message.id, n));
        if (!el) return;
        el.scrollIntoView({ behavior: "smooth", block: "nearest" });
        el.classList.remove("animate-flash");
        void el.offsetWidth; // restart the animation
        el.classList.add("animate-flash");
      }, 30);
    },
    [message.id],
  );

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(answerToMarkdown(question, message));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (non-secure context) */
    }
  };

  const inspect = () => {
    inspectMessage(message.id);
    setPanelTab("run");
  };

  return (
    <div className="flex animate-fade-in gap-3">
      <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-accent-soft text-accent-text">
        <Sparkles size={15} />
      </div>

      <div className="min-w-0 flex-1 space-y-4">
        {(status === "queued" || status === "running") && run && <LiveProgress run={run} onStop={onStop} />}

        {status === "failed" && (
          <div className="flex items-start gap-3 rounded-xl border border-danger/30 bg-danger-soft px-4 py-3">
            <CircleAlert size={16} className="mt-0.5 shrink-0 text-danger" />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-fg">Research failed</p>
              <p className="mt-0.5 break-words text-sm text-fg-muted">{run?.error}</p>
            </div>
            {question && (
              <Button size="sm" onClick={() => onAsk(question)} disabled={busy}>
                <RotateCcw size={12} /> Retry
              </Button>
            )}
          </div>
        )}

        {status === "cancelled" && (
          <div className="flex items-center justify-between gap-3 rounded-xl border border-border px-4 py-3 text-sm text-fg-muted">
            Research stopped.
            {question && (
              <Button size="sm" variant="ghost" onClick={() => onAsk(question)} disabled={busy}>
                <RotateCcw size={12} /> Run again
              </Button>
            )}
          </div>
        )}

        {status === "done" && (
          <>
            {message.factCheck?.unverified && (
              <p className="flex gap-2 rounded-lg bg-warning-soft px-3 py-2 text-xs leading-relaxed text-warning">
                <TriangleAlert size={14} className="mt-px shrink-0" />
                Unverified answer: the fact-checker could not match its claims to the retrieved sources.
                Treat it as a starting point and check the sources below.
              </p>
            )}
            <Markdown content={message.content} citations={citations} onCite={onCite} />

            <div className="flex flex-wrap items-center justify-between gap-2">
              <QualityBar message={message} />
              <div className="flex items-center gap-0.5">
                <IconButton label={copied ? "Copied" : "Copy answer"} onClick={copy}>
                  {copied ? <Check size={15} className="text-success" /> : <Copy size={15} />}
                </IconButton>
                <IconButton
                  label="Download as Markdown"
                  onClick={() => downloadText(`${slug(question)}.md`, answerToMarkdown(question, message))}
                >
                  <Download size={15} />
                </IconButton>
                <IconButton label="Inspect run" active={inspected} onClick={inspect}>
                  <ListTree size={15} />
                </IconButton>
                {question && (
                  <IconButton label="Ask again" onClick={() => onAsk(question)} disabled={busy}>
                    <RotateCcw size={15} />
                  </IconButton>
                )}
              </div>
            </div>

            <SourceList
              citations={citations}
              messageId={message.id}
              expanded={sourcesExpanded}
              onToggle={() => setSourcesExpanded((v) => !v)}
            />

            {message.factCheck && <FactCheckReport report={message.factCheck} />}

            {isLatest && !busy && !!message.followUps?.length && (
              <FollowUps items={message.followUps} onAsk={onAsk} />
            )}
          </>
        )}
      </div>
    </div>
  );
}
