// frontend/hooks/useResearch.ts
"use client";

import { useCallback, useEffect } from "react";
import { cancelResearch, streamUrl, submitResearch } from "@/lib/api";
import type { AgentDoneEvent, DoneEvent, Message, Run } from "@/lib/types";
import { titleFromQuery, uid } from "@/lib/utils";
import { useChatStore } from "@/store/chat";

// One EventSource per in-flight answer, keyed by assistant message id
const streams = new Map<string, EventSource>();
const MAX_CONNECTION_ERRORS = 5;

function patchRun(conversationId: string, messageId: string, fn: (run: Run) => Partial<Run>) {
  useChatStore.getState().patchMessage(conversationId, messageId, (m) =>
    m.run ? { run: { ...m.run, ...fn(m.run) } } : {},
  );
}

function finish(conversationId: string, messageId: string, patch: Partial<Run>) {
  patchRun(conversationId, messageId, (run) =>
    run.status === "done" || run.status === "cancelled" || run.status === "failed"
      ? {}
      : { ...patch, nextAgent: null, finishedAt: Date.now() },
  );
}

function attachStream(conversationId: string, messageId: string, jobId: string) {
  if (streams.has(messageId)) return;

  // The server replays the whole event log (and resumes via Last-Event-ID on
  // reconnect), so connecting late — or after a page reload — loses nothing.
  const es = new EventSource(streamUrl(jobId));
  streams.set(messageId, es);
  let connectionErrors = 0;

  const close = () => {
    es.close();
    streams.delete(messageId);
  };
  const on = <T,>(type: string, handler: (data: T) => void) =>
    es.addEventListener(type, (e) => {
      connectionErrors = 0;
      try {
        handler(JSON.parse((e as MessageEvent).data) as T);
      } catch {
        /* ignore malformed event */
      }
    });

  on("job_started", () =>
    patchRun(conversationId, messageId, (run) =>
      run.status === "queued"
        ? { status: "running", startedAt: Date.now(), nextAgent: "memory_injection" }
        : {},
    ),
  );

  on<AgentDoneEvent>("agent_done", (d) =>
    patchRun(conversationId, messageId, (run) => ({
      status: "running",
      steps: [...run.steps, { agent: d.agent, durationMs: d.duration_ms, detail: d.detail ?? {} }],
      nextAgent: d.next_agent,
    })),
  );

  on<DoneEvent>("done", (d) => {
    useChatStore.getState().patchMessage(conversationId, messageId, (m) => ({
      content: d.final_answer,
      citations: d.citations ?? [],
      confidenceScore: d.confidence_score,
      criticScore: d.critic_score,
      criticFeedback: d.critic_feedback,
      factCheck: d.fact_check,
      followUps: d.follow_ups ?? [],
      retries: d.retries,
      webFallback: d.web_fallback,
      durationMs: d.duration_ms,
      run: m.run && { ...m.run, status: "done", nextAgent: null, finishedAt: Date.now() },
    }));
    close();
  });

  on<{ message?: string }>("failed", (d) => {
    finish(conversationId, messageId, { status: "failed", error: d.message ?? "Research failed." });
    close();
  });

  on("cancelled", () => {
    finish(conversationId, messageId, { status: "cancelled" });
    close();
  });

  es.onerror = () => {
    // EventSource reconnects on its own; give up only if it can't
    connectionErrors += 1;
    if (es.readyState === EventSource.CLOSED || connectionErrors >= MAX_CONNECTION_ERRORS) {
      finish(conversationId, messageId, {
        status: "failed",
        error: "Lost connection to the research server.",
      });
      close();
    }
  };
}

function findMessage(conversationId: string, messageId: string): Message | undefined {
  return useChatStore.getState().conversations[conversationId]?.messages.find((m) => m.id === messageId);
}

export function useResearch() {
  const ask = useCallback(async (query: string) => {
    const store = useChatStore.getState();
    const conversationId =
      store.activeId && store.conversations[store.activeId]
        ? store.activeId
        : store.newConversation(titleFromQuery(query));

    const now = Date.now();
    const assistantId = uid();
    store.addMessage(conversationId, { id: uid(), role: "user", content: query, createdAt: now });
    store.addMessage(conversationId, {
      id: assistantId,
      role: "assistant",
      content: "",
      createdAt: now,
      query,
      run: { status: "queued", steps: [], nextAgent: null, queuedAt: now },
    });
    store.inspectMessage(assistantId);

    try {
      const job = await submitResearch(query, conversationId, store.useWebSearch);
      patchRun(conversationId, assistantId, () => ({ jobId: job.job_id }));
      attachStream(conversationId, assistantId, job.job_id);
    } catch (err) {
      finish(conversationId, assistantId, {
        status: "failed",
        error: err instanceof Error ? err.message : "Could not submit the question.",
      });
    }
  }, []);

  const cancel = useCallback(async (conversationId: string, messageId: string) => {
    const jobId = findMessage(conversationId, messageId)?.run?.jobId;
    if (jobId) {
      try {
        await cancelResearch(jobId); // the stream then delivers "cancelled"
      } catch {
        /* fall through to local cancel */
      }
    }
    // Don't leave the UI waiting if the server never confirms
    setTimeout(() => {
      finish(conversationId, messageId, { status: "cancelled" });
      streams.get(messageId)?.close();
      streams.delete(messageId);
    }, jobId ? 3000 : 0);
  }, []);

  return { ask, cancel };
}

let resumed = false;

/** Re-attach to answers that were still running when the page was reloaded. */
export function useResumeRuns() {
  useEffect(() => {
    if (resumed) return;
    resumed = true;
    const { conversations } = useChatStore.getState();
    for (const conv of Object.values(conversations)) {
      for (const m of conv.messages) {
        const status = m.run?.status;
        if (status !== "queued" && status !== "running") continue;
        if (m.run?.jobId) {
          patchRun(conv.id, m.id, () => ({ steps: [], nextAgent: null })); // replay rebuilds them
          attachStream(conv.id, m.id, m.run.jobId);
        } else {
          finish(conv.id, m.id, { status: "failed", error: "Interrupted before the job was submitted." });
        }
      }
    }
  }, []);
}
