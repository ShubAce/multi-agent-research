// frontend/components/panel/RightPanel.tsx
"use client";

import { Library, ListTree, X } from "lucide-react";
import { KnowledgeBase } from "@/components/panel/KnowledgeBase";
import { RunInspector } from "@/components/panel/RunInspector";
import { IconButton } from "@/components/ui";
import { cn } from "@/lib/utils";
import { useChatStore, type PanelTab } from "@/store/chat";

const TABS: { id: PanelTab; label: string; icon: typeof ListTree }[] = [
  { id: "run", label: "Run inspector", icon: ListTree },
  { id: "library", label: "Library", icon: Library },
];

export function RightPanel({
  overlay,
  onClose,
  onAsk,
}: {
  overlay: boolean; // drawer on narrow screens, docked column on wide ones
  onClose: () => void;
  onAsk: (q: string) => void;
}) {
  const tab = useChatStore((s) => s.panelTab);
  const setTab = useChatStore((s) => s.setPanelTab);

  return (
    <>
      {overlay && <div className="fixed inset-0 z-30 bg-black/30" onClick={onClose} />}
      <aside
        className={cn(
          "flex w-[380px] max-w-[92vw] flex-col border-l border-border bg-bg",
          overlay ? "fixed inset-y-0 right-0 z-40 animate-fade-in shadow-pop" : "shrink-0",
        )}
      >
        <div className="flex h-14 shrink-0 items-center gap-1 border-b border-border px-3">
          <div className="flex flex-1 gap-1" role="tablist">
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                role="tab"
                aria-selected={tab === id}
                onClick={() => setTab(id)}
                className={cn(
                  "flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[13px] font-medium transition-colors",
                  tab === id ? "bg-surface-2 text-fg" : "text-fg-muted hover:text-fg",
                )}
              >
                <Icon size={14} />
                {label}
              </button>
            ))}
          </div>
          <IconButton label="Close panel" onClick={onClose}>
            <X size={16} />
          </IconButton>
        </div>
        <div className="flex-1 overflow-y-auto px-4 py-5">
          {tab === "run" ? <RunInspector /> : <KnowledgeBase onAsk={onAsk} />}
        </div>
      </aside>
    </>
  );
}
