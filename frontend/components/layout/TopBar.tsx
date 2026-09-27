// frontend/components/layout/TopBar.tsx
"use client";

import { Download, Menu, PanelRight, PanelRightClose, Trash2 } from "lucide-react";
import { IconButton } from "@/components/ui";
import { conversationToMarkdown, downloadText, slug } from "@/lib/export";
import { forgetSession } from "@/lib/api";
import { useActiveConversation, useChatStore } from "@/store/chat";

export function TopBar({ panelVisible, onTogglePanel }: { panelVisible: boolean; onTogglePanel: () => void }) {
  const conversation = useActiveConversation();
  const setSidebarOpen = useChatStore((s) => s.setSidebarOpen);
  const deleteConversation = useChatStore((s) => s.deleteConversation);
  const hasAnswers = conversation?.messages.some((m) => m.run?.status === "done");

  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b border-border px-3 sm:px-4">
      <IconButton label="Open history" className="lg:hidden" onClick={() => setSidebarOpen(true)}>
        <Menu size={18} />
      </IconButton>

      <h1 className="min-w-0 flex-1 truncate text-sm font-medium text-fg">
        {conversation?.title ?? "New research"}
      </h1>

      {conversation && (
        <>
          <IconButton
            label="Export conversation as Markdown"
            disabled={!hasAnswers}
            onClick={() => downloadText(`${slug(conversation.title)}.md`, conversationToMarkdown(conversation))}
          >
            <Download size={16} />
          </IconButton>
          <IconButton
            label="Delete conversation"
            className="hover:text-danger"
            onClick={() => {
              deleteConversation(conversation.id);
              forgetSession(conversation.id).catch(() => undefined);
            }}
          >
            <Trash2 size={16} />
          </IconButton>
          <span className="mx-1 h-5 w-px bg-border" />
        </>
      )}

      <IconButton label={panelVisible ? "Hide panel" : "Show run inspector & library"} onClick={onTogglePanel}>
        {panelVisible ? <PanelRightClose size={17} /> : <PanelRight size={17} />}
      </IconButton>
    </header>
  );
}
