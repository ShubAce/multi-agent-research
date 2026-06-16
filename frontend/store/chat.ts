// frontend/store/chat.ts
import { create } from "zustand";
import type { Message, AgentEvent, Citation } from "@/lib/types";

interface ChatStore {
  // ── Session ────────────────────────────────────────────────────────────────
  sessionId: string;

  // ── Messages ───────────────────────────────────────────────────────────────
  messages: Message[];
  addMessage: (msg: Message) => void;
  updateLastAssistantMessage: (patch: Partial<Message>) => void;

  // ── Streaming state ────────────────────────────────────────────────────────
  isLoading: boolean;
  setLoading: (v: boolean) => void;

  agentEvents: AgentEvent[];
  addAgentEvent: (e: AgentEvent) => void;
  clearAgentEvents: () => void;

  // ── Latency tracking ───────────────────────────────────────────────────────
  lastLatencyMs: number | null;
  setLastLatencyMs: (ms: number) => void;

  // ── Actions ────────────────────────────────────────────────────────────────
  startNewSession: () => void;
  clearAll: () => void;
}

function newSessionId(): string {
  return crypto.randomUUID();
}

export const useChatStore = create<ChatStore>((set) => ({
  sessionId: newSessionId(),
  messages: [],
  isLoading: false,
  agentEvents: [],
  lastLatencyMs: null,

  addMessage: (msg) =>
    set((s) => ({ messages: [...s.messages, msg] })),

  updateLastAssistantMessage: (patch) =>
    set((s) => {
      const msgs = [...s.messages];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") {
          msgs[i] = { ...msgs[i], ...patch };
          break;
        }
      }
      return { messages: msgs };
    }),

  setLoading: (v) => set({ isLoading: v }),
  setLastLatencyMs: (ms) => set({ lastLatencyMs: ms }),

  addAgentEvent: (e) =>
    set((s) => ({ agentEvents: [...s.agentEvents, e] })),

  clearAgentEvents: () => set({ agentEvents: [] }),

  startNewSession: () =>
    set({
      sessionId: newSessionId(),
      messages: [],
      agentEvents: [],
      isLoading: false,
      lastLatencyMs: null,
    }),

  clearAll: () =>
    set({
      sessionId: newSessionId(),
      messages: [],
      agentEvents: [],
      isLoading: false,
      lastLatencyMs: null,
    }),
}));
