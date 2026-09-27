// frontend/components/layout/Sidebar.tsx
"use client";

import { useMemo, useState } from "react";
import { ChevronDown, Microscope, Monitor, Moon, Plus, Search, Sun, Trash2, X } from "lucide-react";
import { IconButton, StatusDot } from "@/components/ui";
import { useSystemStatus } from "@/hooks/useSystemStatus";
import { forgetSession } from "@/lib/api";
import { cn, dateGroup } from "@/lib/utils";
import { useChatStore, type Theme } from "@/store/chat";

const GROUP_ORDER = ["Today", "Yesterday", "Previous 7 days", "Older"];

function ConversationList({ query }: { query: string }) {
  const conversations = useChatStore((s) => s.conversations);
  const order = useChatStore((s) => s.order);
  const activeId = useChatStore((s) => s.activeId);
  const select = useChatStore((s) => s.selectConversation);
  const remove = useChatStore((s) => s.deleteConversation);

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const byGroup = new Map<string, { id: string; title: string; running: boolean }[]>();
    for (const id of order) {
      const c = conversations[id];
      if (!c) continue;
      if (needle && !c.title.toLowerCase().includes(needle)) continue;
      const group = dateGroup(c.updatedAt);
      const running = c.messages.some((m) => m.run?.status === "running" || m.run?.status === "queued");
      byGroup.set(group, [...(byGroup.get(group) ?? []), { id, title: c.title, running }]);
    }
    return GROUP_ORDER.filter((g) => byGroup.has(g)).map((g) => [g, byGroup.get(g)!] as const);
  }, [conversations, order, query]);

  if (!groups.length) {
    return (
      <p className="px-3 py-6 text-center text-xs text-fg-subtle">
        {query ? "No matching conversations" : "Your research history will appear here"}
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {groups.map(([group, items]) => (
        <div key={group}>
          <p className="px-3 pb-1 text-[11px] font-medium text-fg-subtle">{group}</p>
          <ul className="space-y-px">
            {items.map((item) => (
              <li key={item.id} className="group relative">
                <button
                  onClick={() => select(item.id)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-lg px-3 py-2 pr-9 text-left text-[13px] transition-colors",
                    item.id === activeId
                      ? "bg-surface-2 font-medium text-fg"
                      : "text-fg-muted hover:bg-surface-2/60 hover:text-fg",
                  )}
                >
                  {item.running && <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-accent" />}
                  <span className="truncate">{item.title}</span>
                </button>
                <IconButton
                  label="Delete conversation"
                  className="absolute right-1 top-1/2 h-7 w-7 -translate-y-1/2 opacity-0 hover:text-danger focus:opacity-100 group-hover:opacity-100"
                  onClick={() => {
                    remove(item.id);
                    forgetSession(item.id).catch(() => undefined); // also drop server-side memory
                  }}
                >
                  <Trash2 size={14} />
                </IconButton>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

const SERVICE_LABELS: Record<string, string> = {
  llm: "Language model",
  worker: "Agent worker",
  redis: "Redis",
  chromadb: "ChromaDB",
};

function SystemStatus() {
  const { status, apiReachable } = useSystemStatus();
  const [open, setOpen] = useState<boolean | null>(null); // null = expand automatically on problems

  const rows: { name: string; ok: boolean; detail: string }[] = [
    {
      name: "API",
      ok: apiReachable === true,
      detail: apiReachable === false ? "Unreachable — start the backend" : "Connected",
    },
    ...Object.entries(status?.services ?? {}).map(([key, value]) => ({
      name: SERVICE_LABELS[key] ?? key,
      ok: value === "ok",
      detail: value === "ok" ? "Operational" : value.replace(/^error:\s*/, ""),
    })),
  ];
  const problems = apiReachable === null ? 0 : rows.filter((r) => !r.ok).length;
  const state = apiReachable === null ? "unknown" : problems === 0 ? "ok" : apiReachable ? "warn" : "error";
  const expanded = open ?? problems > 0;

  return (
    <div className="rounded-lg border border-border bg-surface">
      <button
        onClick={() => setOpen(!expanded)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs"
        aria-expanded={expanded}
      >
        <StatusDot state={state} />
        <span className="flex-1 font-medium text-fg-muted">
          {apiReachable === null
            ? "Checking services…"
            : problems === 0
              ? "All systems operational"
              : `${problems} service${problems > 1 ? "s" : ""} need attention`}
        </span>
        <ChevronDown size={14} className={cn("text-fg-subtle transition-transform", expanded && "rotate-180")} />
      </button>
      {expanded && (
        <ul className="space-y-1.5 border-t border-border px-3 py-2.5">
          {rows.map((r) => (
            <li key={r.name} className="flex items-start gap-2 text-xs" title={r.detail}>
              <span className="mt-1">
                <StatusDot state={r.ok ? "ok" : "error"} />
              </span>
              <span className="w-24 shrink-0 text-fg-muted">{r.name}</span>
              <span className={cn("min-w-0 flex-1 truncate", r.ok ? "text-fg-subtle" : "text-danger")}>
                {r.detail}
              </span>
            </li>
          ))}
          {status?.features && !status.features.web_search && (
            <li className="pt-1 text-[11px] leading-relaxed text-fg-subtle">
              Web search disabled: TAVILY_API_KEY is not set.
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

const THEMES: { value: Theme; icon: typeof Sun; label: string }[] = [
  { value: "light", icon: Sun, label: "Light" },
  { value: "dark", icon: Moon, label: "Dark" },
  { value: "system", icon: Monitor, label: "System" },
];

function ThemeSwitch() {
  const theme = useChatStore((s) => s.theme);
  const setTheme = useChatStore((s) => s.setTheme);
  return (
    <div className="flex rounded-lg bg-surface-2 p-0.5" role="radiogroup" aria-label="Theme">
      {THEMES.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          role="radio"
          aria-checked={theme === value}
          title={label}
          onClick={() => setTheme(value)}
          className={cn(
            "flex h-7 flex-1 items-center justify-center rounded-md text-fg-subtle transition-colors hover:text-fg",
            theme === value && "bg-surface text-fg shadow-card",
          )}
        >
          <Icon size={14} />
        </button>
      ))}
    </div>
  );
}

export function Sidebar() {
  const [query, setQuery] = useState("");
  const select = useChatStore((s) => s.selectConversation);
  const sidebarOpen = useChatStore((s) => s.sidebarOpen);
  const setSidebarOpen = useChatStore((s) => s.setSidebarOpen);

  return (
    <>
      {sidebarOpen && (
        <div className="fixed inset-0 z-30 bg-black/30 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-[272px] flex-col border-r border-border bg-bg transition-transform lg:static lg:translate-x-0",
          sidebarOpen ? "translate-x-0 shadow-pop" : "-translate-x-full",
        )}
      >
        <div className="flex h-14 items-center gap-2.5 px-4">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent text-accent-fg">
            <Microscope size={15} />
          </div>
          <div className="min-w-0 flex-1 leading-tight">
            <p className="text-sm font-semibold text-fg">Research Assistant</p>
            <p className="text-[11px] text-fg-subtle">Multi-agent RAG</p>
          </div>
          <IconButton label="Close sidebar" className="lg:hidden" onClick={() => setSidebarOpen(false)}>
            <X size={16} />
          </IconButton>
        </div>

        <div className="space-y-2 px-3">
          <button
            onClick={() => select(null)}
            className="flex h-9 w-full items-center gap-2 rounded-lg border border-border bg-surface px-3 text-sm font-medium text-fg shadow-card transition-colors hover:bg-surface-2"
          >
            <Plus size={15} />
            New research
          </button>
          <label className="flex h-9 items-center gap-2 rounded-lg px-3 text-fg-subtle ring-1 ring-inset ring-transparent focus-within:bg-surface focus-within:ring-border">
            <Search size={14} />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search history"
              className="w-full bg-transparent text-[13px] text-fg outline-none placeholder:text-fg-subtle"
            />
          </label>
        </div>

        <nav className="mt-3 flex-1 overflow-y-auto px-2 pb-4">
          <ConversationList query={query} />
        </nav>

        <div className="space-y-2 border-t border-border p-3">
          <SystemStatus />
          <ThemeSwitch />
        </div>
      </aside>
    </>
  );
}
