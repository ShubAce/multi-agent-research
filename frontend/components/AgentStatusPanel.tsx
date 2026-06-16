// frontend/components/AgentStatusPanel.tsx
"use client";

import type { AgentEvent } from "@/lib/types";
import { useChatStore } from "@/store/chat";

const AGENT_META: Record<string, { icon: string; label: string }> = {
  planner:          { icon: "🧠", label: "Planner" },
  web_search:       { icon: "🌐", label: "Web Search" },
  rag_agent:        { icon: "📚", label: "RAG Retrieval" },
  parallel_agents:  { icon: "⚡", label: "Parallel Agents" },
  synthesiser:      { icon: "✨", label: "Synthesiser" },
  job_started:      { icon: "🚀", label: "Started" },
};

interface Props {
  events: AgentEvent[];
  isLoading: boolean;
}

export function AgentStatusPanel({ events, isLoading }: Props) {
  const { lastLatencyMs } = useChatStore();

  if (events.length === 0 && !isLoading) return null;

  const agentDoneEvents = events.filter((e) => e.type === "agent_done");

  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Agent Activity
        </p>
        {isLoading && (
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-blue-500" />
        )}
        {!isLoading && lastLatencyMs !== null && (
          <span className="text-xs text-slate-400 tabular-nums">
            {(lastLatencyMs / 1000).toFixed(1)}s
          </span>
        )}
      </div>

      <div className="space-y-2">
        {agentDoneEvents.map((event, i) => {
          const meta = AGENT_META[event.agent ?? ""] ?? {
            icon: "🤖",
            label: event.agent ?? "Agent",
          };
          return (
            <div key={i} className="flex items-center gap-2 text-sm">
              <span className="text-base">{meta.icon}</span>
              <span className="font-medium text-slate-700">{meta.label}</span>
              <span className="text-xs text-slate-400 ml-auto">✓</span>
            </div>
          );
        })}

        {isLoading && (
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <span className="animate-spin inline-block">⚙️</span>
            <span>Running agents…</span>
          </div>
        )}
      </div>
    </div>
  );
}
