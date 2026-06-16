// frontend/app/page.tsx
"use client";

import { useEffect, useRef } from "react";
import { RotateCcw } from "lucide-react";

import { useChatStore } from "@/store/chat";
import { useResearch } from "@/hooks/useResearch";
import { MessageBubble } from "@/components/MessageBubble";
import { AgentStatusPanel } from "@/components/AgentStatusPanel";
import { ChatInput } from "@/components/ChatInput";
import { PaperSummariser } from "@/components/PaperSummariser";

export default function HomePage() {
  const { messages, isLoading, agentEvents, startNewSession } = useChatStore();
  const { submit } = useResearch();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, agentEvents]);

  return (
    <div className="flex h-screen bg-slate-50 font-sans">

      {/* ── Main chat area ────────────────────────────────────────────────── */}
      <main className="flex flex-1 flex-col overflow-hidden">

        {/* Header */}
        <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-4 shadow-sm">
          <div>
            <h1 className="text-lg font-semibold text-slate-900">Research Assistant</h1>
            <p className="text-xs text-slate-500">
              Multi-agent RAG · ArXiv corpus · Session memory · Self-reflection
            </p>
          </div>
          <button
            onClick={startNewSession}
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-50 transition-colors"
          >
            <RotateCcw size={12} />
            New session
          </button>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center text-slate-400 gap-3">
              <div className="text-5xl">🔬</div>
              <p className="text-lg font-medium text-slate-600">What do you want to research?</p>
              <p className="text-sm max-w-md">
                Ask anything about ML research. The system remembers context across turns —
                try asking a follow-up question after the first answer.
              </p>
              <div className="grid grid-cols-1 gap-2 mt-4 w-full max-w-md">
                {[
                  "What is Flash Attention and why does it matter?",
                  "How does retrieval-augmented generation work?",
                  "Explain RLHF in large language models",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => submit(q)}
                    className="text-left rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 hover:border-blue-300 hover:text-blue-600 transition-colors shadow-sm"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="border-t border-slate-200 bg-white px-6 py-4">
          <ChatInput onSubmit={submit} disabled={isLoading} />
          <p className="text-center text-xs text-slate-400 mt-2">
            Remembers context within a session · Shift+Enter for new line
          </p>
        </div>
      </main>

      {/* ── Right sidebar ─────────────────────────────────────────────────── */}
      <aside className="hidden lg:flex w-80 flex-col gap-5 border-l border-slate-200 bg-white p-5 overflow-y-auto">

        {/* Agent activity */}
        <AgentStatusPanel events={agentEvents} isLoading={isLoading} />

        {!isLoading && agentEvents.length === 0 && (
          <div className="rounded-xl border border-slate-100 bg-slate-50 p-4 text-xs text-slate-500 space-y-1.5">
            <p className="font-semibold text-slate-600 mb-2">Pipeline</p>
            {[
              { icon: "🧠", label: "Memory + Planner", desc: "Reads history, decomposes query" },
              { icon: "⚡", label: "Parallel Agents",  desc: "Web + RAG run concurrently" },
              { icon: "✨", label: "Synthesiser",       desc: "Merges results with citations" },
              { icon: "🎯", label: "Critic",            desc: "Scores answer, retries if weak" },
            ].map((s, i) => (
              <div key={i} className="flex items-start gap-2">
                <span>{s.icon}</span>
                <div>
                  <span className="font-medium text-slate-700">{s.label}</span>
                  <span className="text-slate-400"> — {s.desc}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Divider */}
        <div className="border-t border-slate-100" />

        {/* Paper Summariser */}
        <PaperSummariser />
      </aside>
    </div>
  );
}
