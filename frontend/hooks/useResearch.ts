// frontend/hooks/useResearch.ts
"use client";

import { useCallback } from "react";
import { submitResearch } from "@/lib/api";
import { useChatStore } from "@/store/chat";
import type { AgentEvent, Message } from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function useResearch() {
  const {
    sessionId,
    addMessage,
    updateLastAssistantMessage,
    setLoading,
    addAgentEvent,
    clearAgentEvents,
    setLastLatencyMs,
  } = useChatStore();

  const submit = useCallback(
    async (query: string) => {
      setLoading(true);
      clearAgentEvents();
      const startTime = Date.now();

      // Add user message immediately
      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: query,
        timestamp: Date.now(),
      };
      addMessage(userMsg);

      // Add placeholder assistant message
      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "",
        citations: [],
        agentsUsed: [],
        timestamp: Date.now(),
      };
      addMessage(assistantMsg);

      try {
        const job = await submitResearch(query, sessionId);

        const es = new EventSource(`${API_URL}/api/v1/stream/${job.job_id}`);

        es.addEventListener("job_started", (e) => {
          addAgentEvent(JSON.parse(e.data) as AgentEvent);
        });

        es.addEventListener("agent_done", (e) => {
          addAgentEvent(JSON.parse(e.data) as AgentEvent);
        });

        es.addEventListener("done", (e) => {
          const data = JSON.parse(e.data) as AgentEvent;
          addAgentEvent(data);
          setLastLatencyMs(Date.now() - startTime);

          updateLastAssistantMessage({
            content:        data.final_answer ?? "No answer generated.",
            citations:      data.citations ?? [],
            agentsUsed:     data.agents_used ?? [],
            confidenceScore: data.confidence_score,
            criticScore:    data.critic_score,
          });

          setLoading(false);
          es.close();
        });

        es.addEventListener("error", (e) => {
          try {
            const data = JSON.parse((e as MessageEvent).data) as AgentEvent;
            addAgentEvent(data);
            updateLastAssistantMessage({
              content: `Error: ${data.message ?? "Something went wrong."}`,
            });
          } catch {
            updateLastAssistantMessage({
              content: "An unexpected error occurred. Please try again.",
            });
          }
          setLoading(false);
          es.close();
        });

        es.onerror = () => {
          updateLastAssistantMessage({
            content: "Connection lost. Please check your network and try again.",
          });
          setLoading(false);
          es.close();
        };

      } catch (err) {
        const message = err instanceof Error ? err.message : "Submission failed";
        updateLastAssistantMessage({ content: `Error: ${message}` });
        setLoading(false);
      }
    },
    [sessionId, addMessage, updateLastAssistantMessage, setLoading,
     addAgentEvent, clearAgentEvents, setLastLatencyMs]
  );

  return { submit };
}
