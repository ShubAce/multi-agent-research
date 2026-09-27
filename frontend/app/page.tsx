// frontend/app/page.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { PlugZap } from "lucide-react";
import { AssistantMessage } from "@/components/chat/AssistantMessage";
import { Composer } from "@/components/chat/Composer";
import { EmptyState } from "@/components/chat/EmptyState";
import { UserMessage } from "@/components/chat/UserMessage";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopBar } from "@/components/layout/TopBar";
import { RightPanel } from "@/components/panel/RightPanel";
import { Button } from "@/components/ui";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { useResearch, useResumeRuns } from "@/hooks/useResearch";
import { useSystemStatus, useSystemStatusPoller } from "@/hooks/useSystemStatus";
import { useApplyTheme } from "@/hooks/useTheme";
import { API_URL } from "@/lib/api";
import { useActiveConversation, useChatStore } from "@/store/chat";

function OfflineBanner() {
  const refresh = useSystemStatus((s) => s.refresh);
  return (
    <div className="mx-auto mb-3 flex w-full max-w-3xl items-center gap-3 px-4 sm:px-6">
      <div className="flex flex-1 items-center gap-3 rounded-xl border border-danger/30 bg-danger-soft px-4 py-2.5 text-sm">
        <PlugZap size={16} className="shrink-0 text-danger" />
        <span className="flex-1 text-fg-muted">
          Can&apos;t reach the backend at <span className="font-mono text-xs">{API_URL}</span>. Start it with{" "}
          <code className="font-mono text-xs">python run_dev.py</code>.
        </span>
        <Button size="sm" onClick={refresh}>
          Retry
        </Button>
      </div>
    </div>
  );
}

function Workspace() {
  const conversation = useActiveConversation();
  const activeId = useChatStore((s) => s.activeId);
  const panelOpen = useChatStore((s) => s.panelOpen);
  const setPanelOpen = useChatStore((s) => s.setPanelOpen);
  const { status, apiReachable } = useSystemStatus();
  const { ask, cancel } = useResearch();

  const wide = useMediaQuery("(min-width: 1280px)");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const panelVisible = wide ? panelOpen : drawerOpen;

  const messages = conversation?.messages ?? [];
  const running = messages.find((m) => m.run?.status === "queued" || m.run?.status === "running");
  const busy = Boolean(running);
  const stop = () => {
    if (conversation && running) cancel(conversation.id, running.id);
  };

  // Keep the newest exchange in view when a question is asked or a conversation opens
  const scrollRef = useRef<HTMLDivElement>(null);
  const lastActive = useRef<string | null>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const switched = lastActive.current !== activeId;
    lastActive.current = activeId;
    el.scrollTo({ top: el.scrollHeight, behavior: switched ? "auto" : "smooth" });
  }, [activeId, messages.length]);

  const askFromPanel = (q: string) => {
    ask(q);
    if (!wide) setDrawerOpen(false);
  };

  return (
    <div className="flex h-dvh overflow-hidden bg-bg">
      <Sidebar />

      <main className="flex min-w-0 flex-1 flex-col bg-surface">
        <TopBar
          panelVisible={panelVisible}
          onTogglePanel={() => (wide ? setPanelOpen(!panelOpen) : setDrawerOpen((v) => !v))}
        />

        <div ref={scrollRef} className="flex-1 overflow-y-auto overflow-x-hidden">
          {messages.length === 0 ? (
            <EmptyState onAsk={ask} status={status} />
          ) : (
            <div className="mx-auto w-full max-w-3xl space-y-8 px-4 py-8 sm:px-6">
              {messages.map((m, i) =>
                m.role === "user" ? (
                  <UserMessage key={m.id} message={m} />
                ) : (
                  <AssistantMessage
                    key={m.id}
                    message={m}
                    isLatest={i === messages.length - 1}
                    busy={busy}
                    onAsk={ask}
                    onStop={stop}
                  />
                ),
              )}
            </div>
          )}
        </div>

        {apiReachable === false && <OfflineBanner />}
        <Composer
          onSubmit={ask}
          onStop={stop}
          busy={busy}
          webAvailable={status?.features?.web_search}
          autoFocus
        />
      </main>

      {panelVisible && (
        <RightPanel
          overlay={!wide}
          onClose={() => (wide ? setPanelOpen(false) : setDrawerOpen(false))}
          onAsk={askFromPanel}
        />
      )}
    </div>
  );
}

export default function HomePage() {
  // Conversations live in localStorage, so render only after mount to avoid
  // a server/client hydration mismatch.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  useApplyTheme();
  useSystemStatusPoller();
  useResumeRuns();

  if (!mounted) return <div className="h-dvh bg-bg" />;
  return <Workspace />;
}
