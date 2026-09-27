// frontend/store/chat.ts
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { Conversation, Message } from "@/lib/types";
import { uid } from "@/lib/utils";

export type Theme = "light" | "dark" | "system";
export type PanelTab = "run" | "library";

const MAX_CONVERSATIONS = 50;

interface ChatStore {
  // ── Conversations (persisted) ────────────────────────────────────────────
  conversations: Record<string, Conversation>;
  order: string[]; // most recently updated first
  activeId: string | null;

  newConversation: (title?: string) => string;
  selectConversation: (id: string | null) => void;
  deleteConversation: (id: string) => void;
  clearConversations: () => void;
  addMessage: (conversationId: string, message: Message) => void;
  patchMessage: (
    conversationId: string,
    messageId: string,
    patch: Partial<Message> | ((m: Message) => Partial<Message>),
  ) => void;

  // ── Preferences (persisted) ──────────────────────────────────────────────
  useWebSearch: boolean;
  setUseWebSearch: (v: boolean) => void;
  theme: Theme;
  setTheme: (t: Theme) => void;
  panelOpen: boolean;
  setPanelOpen: (v: boolean) => void;
  panelTab: PanelTab;
  setPanelTab: (t: PanelTab) => void;

  // ── Transient UI state ───────────────────────────────────────────────────
  sidebarOpen: boolean; // mobile drawer
  setSidebarOpen: (v: boolean) => void;
  inspectedMessageId: string | null; // answer shown in the run inspector
  inspectMessage: (id: string | null) => void;
}

function touch(state: ChatStore, id: string, conv: Conversation) {
  return {
    conversations: { ...state.conversations, [id]: { ...conv, updatedAt: Date.now() } },
    order: [id, ...state.order.filter((x) => x !== id)],
  };
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      conversations: {},
      order: [],
      activeId: null,

      newConversation: (title = "New research") => {
        const id = uid();
        const now = Date.now();
        set((s) => {
          const order = [id, ...s.order].slice(0, MAX_CONVERSATIONS);
          const conversations: Record<string, Conversation> = {};
          for (const cid of order) conversations[cid] = s.conversations[cid];
          conversations[id] = { id, title, createdAt: now, updatedAt: now, messages: [] };
          return { conversations, order, activeId: id, inspectedMessageId: null };
        });
        return id;
      },

      selectConversation: (id) => set({ activeId: id, inspectedMessageId: null, sidebarOpen: false }),

      deleteConversation: (id) =>
        set((s) => {
          const { [id]: _removed, ...conversations } = s.conversations;
          return {
            conversations,
            order: s.order.filter((x) => x !== id),
            activeId: s.activeId === id ? null : s.activeId,
            inspectedMessageId: s.activeId === id ? null : s.inspectedMessageId,
          };
        }),

      clearConversations: () =>
        set({ conversations: {}, order: [], activeId: null, inspectedMessageId: null }),

      addMessage: (conversationId, message) =>
        set((s) => {
          const conv = s.conversations[conversationId];
          if (!conv) return {};
          return touch(s, conversationId, { ...conv, messages: [...conv.messages, message] });
        }),

      patchMessage: (conversationId, messageId, patch) =>
        set((s) => {
          const conv = s.conversations[conversationId];
          if (!conv) return {};
          const messages = conv.messages.map((m) =>
            m.id === messageId ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m,
          );
          return {
            conversations: { ...s.conversations, [conversationId]: { ...conv, messages } },
          };
        }),

      useWebSearch: true,
      setUseWebSearch: (v) => set({ useWebSearch: v }),
      theme: "system",
      setTheme: (t) => set({ theme: t }),
      panelOpen: true,
      setPanelOpen: (v) => set({ panelOpen: v }),
      panelTab: "run",
      setPanelTab: (t) => set({ panelTab: t, panelOpen: true }),

      sidebarOpen: false,
      setSidebarOpen: (v) => set({ sidebarOpen: v }),
      inspectedMessageId: null,
      inspectMessage: (id) => set({ inspectedMessageId: id }),
    }),
    {
      name: "research-assistant", // also read by the theme script in app/layout.tsx
      version: 1,
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({
        conversations: s.conversations,
        order: s.order,
        activeId: s.activeId,
        useWebSearch: s.useWebSearch,
        theme: s.theme,
        panelOpen: s.panelOpen,
        panelTab: s.panelTab,
      }),
    },
  ),
);

/** The conversation currently on screen (undefined on the start page). */
export function useActiveConversation(): Conversation | undefined {
  return useChatStore((s) => (s.activeId ? s.conversations[s.activeId] : undefined));
}
