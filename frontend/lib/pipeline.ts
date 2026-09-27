// frontend/lib/pipeline.ts — how each agent stage is labelled and summarised
import { Brain, History, Layers, PenLine, ShieldCheck, type LucideIcon } from "lucide-react";
import type { AgentName, PipelineStep, Run } from "./types";

export interface StageMeta {
  label: string;
  running: string; // shown while the stage is in progress
  icon: LucideIcon;
}

export const STAGES: Record<AgentName, StageMeta> = {
  memory_injection: { label: "Context", running: "Loading conversation context", icon: History },
  planner: { label: "Plan", running: "Breaking the question into sub-tasks", icon: Brain },
  parallel_agents: { label: "Research", running: "Searching papers and the web", icon: Layers },
  synthesiser: { label: "Synthesis", running: "Writing a cited answer and fact-checking it", icon: PenLine },
  critic: { label: "Review", running: "Scoring answer quality", icon: ShieldCheck },
};

export const STAGE_ORDER: AgentName[] = [
  "memory_injection",
  "planner",
  "parallel_agents",
  "synthesiser",
  "critic",
];

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** One-line outcome of a finished stage. */
export function summariseStep(step: PipelineStep): string {
  const d = step.detail as Record<string, unknown>;
  switch (step.agent) {
    case "memory_injection":
      return d.has_history ? `Using ${plural(Number(d.turns) || 1, "earlier turn")}` : "New conversation";
    case "planner": {
      const plan = (d.plan as { agent: string }[]) ?? [];
      const web = plan.filter((p) => p.agent === "web_search").length;
      return `${plural(plan.length, "sub-task")}${web ? ` · ${web} on the web` : ""}`;
    }
    case "parallel_agents": {
      const parts = [];
      if (Number(d.papers)) parts.push(plural(Number(d.papers), "paper"));
      if (Number(d.web_results)) parts.push(plural(Number(d.web_results), "web result"));
      const base = parts.join(" · ") || "No sources found";
      return d.retry ? `Retry: ${base}` : base;
    }
    case "synthesiser": {
      const conf = d.confidence as number | null;
      return `${plural(Number(d.sources) || 0, "source")}${conf != null ? ` · ${Math.round(conf * 100)}% grounded` : ""}`;
    }
    case "critic":
      return d.score != null ? `Scored ${d.score}/10` : "Review unavailable";
  }
}

/** Label for whatever is running right now. */
export function currentActivity(run: Run): string {
  if (run.status === "queued") return "Waiting for a worker";
  if (!run.nextAgent) return "Finishing up";
  const isRetry = run.nextAgent === "parallel_agents" && run.steps.some((s) => s.agent === "critic");
  return isRetry ? "Answer was weak — researching again" : STAGES[run.nextAgent].running;
}
