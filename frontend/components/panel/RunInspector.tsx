// frontend/components/panel/RunInspector.tsx
"use client";

import { BookOpen, CircleAlert, Globe, RotateCcw } from "lucide-react";
import { Badge, SectionLabel, Spinner } from "@/components/ui";
import { useNow } from "@/hooks/useNow";
import { STAGES, STAGE_ORDER, currentActivity, summariseStep } from "@/lib/pipeline";
import type { FactCheck, Message, PipelineStep } from "@/lib/types";
import { cn, formatDuration } from "@/lib/utils";
import { useActiveConversation, useChatStore } from "@/store/chat";

const STAGE_HELP: Record<string, string> = {
  memory_injection: "Loads recent turns so follow-ups like “who proposed it?” resolve correctly.",
  planner: "Splits the question into focused sub-tasks and routes each to papers or the web.",
  parallel_agents: "Hybrid search (dense + BM25, query rewriting, reranking) over papers, and web search, in parallel.",
  synthesiser: "Writes a cited answer, then fact-checks every sentence against the sources.",
  critic: "Scores the answer; if it is weak, the agents research again with better queries.",
};

function StepDetail({ step }: { step: PipelineStep }) {
  const d = step.detail as Record<string, unknown>;
  const error = typeof d.error === "string" ? d.error : null;

  return (
    <div className="mt-2 space-y-2 text-xs leading-relaxed text-fg-muted">
      {step.agent === "planner" && Array.isArray(d.plan) && (
        <ul className="space-y-1.5">
          {(d.plan as { task: string; agent: string }[]).map((p, i) => (
            <li key={i} className="flex items-start gap-2">
              {p.agent === "web_search" ? (
                <Globe size={12} className="mt-0.5 shrink-0 text-fg-subtle" />
              ) : (
                <BookOpen size={12} className="mt-0.5 shrink-0 text-fg-subtle" />
              )}
              <span>{p.task}</span>
            </li>
          ))}
        </ul>
      )}

      {step.agent === "parallel_agents" && (
        <div className="flex flex-wrap gap-1.5">
          <Badge>{Number(d.papers) || 0} papers</Badge>
          <Badge>{Number(d.passages) || 0} passages</Badge>
          <Badge>{Number(d.web_results) || 0} web results</Badge>
          {!!d.web_fallback && <Badge tone="warning">Web fallback</Badge>}
          {!!d.retry && <Badge tone="accent">Retry round</Badge>}
        </div>
      )}

      {step.agent === "synthesiser" && !!d.fact_check && (
        <p>
          {(d.fact_check as FactCheck).unverified
            ? `Fact-check: none of ${(d.fact_check as FactCheck).checked} claims matched a source (unverified)`
            : `Fact-check: ${(d.fact_check as FactCheck).supported} supported · ${(d.fact_check as FactCheck).inferred} inferred · ${(d.fact_check as FactCheck).removed} removed`}
        </p>
      )}

      {step.agent === "critic" && (
        <>
          {typeof d.feedback === "string" && d.feedback && (
            <p className="border-l-2 border-border pl-2.5 italic">{d.feedback}</p>
          )}
          {Array.isArray(d.improved_queries) && d.improved_queries.length > 0 && (
            <div>
              <p className="mb-1 flex items-center gap-1 font-medium text-accent-text">
                <RotateCcw size={12} /> Researching again with:
              </p>
              <ul className="list-disc space-y-0.5 pl-4">
                {(d.improved_queries as string[]).map((q) => (
                  <li key={q}>{q}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}

      {error && (
        <p className="flex gap-1.5 rounded-md bg-danger-soft px-2 py-1.5 text-danger">
          <CircleAlert size={12} className="mt-0.5 shrink-0" />
          <span className="break-words">{error}</span>
        </p>
      )}
    </div>
  );
}

function Timeline({ message }: { message: Message }) {
  const run = message.run!;
  const running = run.status === "queued" || run.status === "running";
  const now = useNow(running);

  return (
    <ol className="relative">
      {run.steps.map((step, i) => {
        const meta = STAGES[step.agent];
        const Icon = meta.icon;
        return (
          <li key={i} className="relative pb-5 pl-9">
            <span className="absolute bottom-0 left-[13px] top-7 w-px bg-border" />
            <span className="absolute left-0 top-0 flex h-7 w-7 items-center justify-center rounded-full border border-border bg-surface text-fg-muted">
              <Icon size={13} />
            </span>
            <div className="flex items-baseline justify-between gap-2 pt-1">
              <p className="text-[13px] font-medium text-fg">{meta.label}</p>
              <span className="text-[11px] tabular-nums text-fg-subtle">{formatDuration(step.durationMs)}</span>
            </div>
            <p className="text-xs text-fg-subtle">{summariseStep(step)}</p>
            <StepDetail step={step} />
          </li>
        );
      })}

      {running && (
        <li className="relative pl-9">
          <span className="absolute left-0 top-0 flex h-7 w-7 items-center justify-center rounded-full border border-accent/40 bg-accent-soft">
            <Spinner size={13} className="text-accent-text" />
          </span>
          <p className="pt-1 text-[13px] font-medium text-fg">{currentActivity(run)}…</p>
          <p className="text-xs tabular-nums text-fg-subtle">
            {formatDuration(now - (run.startedAt ?? run.queuedAt))} elapsed
          </p>
        </li>
      )}

      {run.status === "failed" && (
        <li className="pl-9 text-xs text-danger">{run.error}</li>
      )}
      {run.status === "cancelled" && <li className="pl-9 text-xs text-fg-subtle">Stopped by you.</li>}
    </ol>
  );
}

function HowItWorks() {
  return (
    <div className="space-y-4">
      <div>
        <p className="text-sm font-medium text-fg">How a question is answered</p>
        <p className="mt-1 text-xs leading-relaxed text-fg-muted">
          Each answer is produced by a pipeline of agents. Ask something to watch it run here.
        </p>
      </div>
      <ol className="space-y-3">
        {STAGE_ORDER.map((agent) => {
          const { icon: Icon, label } = STAGES[agent];
          return (
            <li key={agent} className="flex gap-3">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border bg-surface text-fg-muted">
                <Icon size={13} />
              </span>
              <div>
                <p className="text-[13px] font-medium text-fg">{label}</p>
                <p className="text-xs leading-relaxed text-fg-muted">{STAGE_HELP[agent]}</p>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export function RunInspector() {
  const conversation = useActiveConversation();
  const inspectedId = useChatStore((s) => s.inspectedMessageId);
  const inspectMessage = useChatStore((s) => s.inspectMessage);

  const answers = (conversation?.messages ?? []).filter((m) => m.role === "assistant" && m.run);
  const message = answers.find((m) => m.id === inspectedId) ?? answers[answers.length - 1];

  if (!message?.run) return <HowItWorks />;

  const status = message.run.status;
  const index = answers.indexOf(message);

  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center justify-between gap-2">
          <SectionLabel>
            Answer {index + 1} of {answers.length}
          </SectionLabel>
          <div className="flex items-center gap-1">
            {answers.length > 1 && (
              <>
                <button
                  className="rounded px-1.5 text-xs text-fg-muted hover:bg-surface-2 disabled:opacity-40"
                  disabled={index === 0}
                  onClick={() => inspectMessage(answers[index - 1].id)}
                >
                  ‹ Prev
                </button>
                <button
                  className="rounded px-1.5 text-xs text-fg-muted hover:bg-surface-2 disabled:opacity-40"
                  disabled={index === answers.length - 1}
                  onClick={() => inspectMessage(answers[index + 1].id)}
                >
                  Next ›
                </button>
              </>
            )}
          </div>
        </div>
        <p className="mt-1.5 line-clamp-3 text-sm font-medium leading-snug text-fg">{message.query}</p>
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <Badge
            tone={
              status === "done" ? "success" : status === "failed" ? "danger" : status === "cancelled" ? "neutral" : "accent"
            }
          >
            {status === "done" ? "Completed" : status === "failed" ? "Failed" : status === "cancelled" ? "Stopped" : "Running"}
          </Badge>
          {message.durationMs != null && <Badge>{formatDuration(message.durationMs)} total</Badge>}
          {message.run.jobId && (
            <span className="truncate font-mono text-[10px] text-fg-subtle" title="Job id">
              {message.run.jobId.slice(0, 8)}
            </span>
          )}
        </div>
      </div>

      <div className={cn("border-t border-border pt-4")}>
        {message.run.steps.length === 0 && status !== "queued" && status !== "running" ? (
          <p className="text-xs text-fg-subtle">No pipeline steps were recorded for this answer.</p>
        ) : (
          <Timeline message={message} />
        )}
      </div>
    </div>
  );
}
