// frontend/components/chat/Composer.tsx
"use client";

import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { ArrowUp, Globe, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { useChatStore } from "@/store/chat";

const MAX_LENGTH = 1000;

interface Props {
  onSubmit: (query: string) => void;
  onStop: () => void;
  busy: boolean;
  webAvailable: boolean | undefined; // undefined while status is loading
  autoFocus?: boolean;
}

export function Composer({ onSubmit, onStop, busy, webAvailable, autoFocus }: Props) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  const useWebSearch = useChatStore((s) => s.useWebSearch);
  const setUseWebSearch = useChatStore((s) => s.setUseWebSearch);

  const webOn = useWebSearch && webAvailable !== false;
  const trimmed = value.trim();
  const tooShort = trimmed.length > 0 && trimmed.length < 5;

  useEffect(() => {
    if (autoFocus) ref.current?.focus();
  }, [autoFocus]);

  const resize = () => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  const submit = () => {
    if (busy || trimmed.length < 5) return;
    onSubmit(trimmed);
    setValue("");
    requestAnimationFrame(resize);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-4 sm:px-6">
      <div className="rounded-2xl border border-border bg-surface shadow-card transition-shadow focus-within:border-accent/50 focus-within:shadow-[0_0_0_4px_rgb(var(--accent)/0.08)]">
        <textarea
          ref={ref}
          value={value}
          maxLength={MAX_LENGTH}
          onChange={(e) => {
            setValue(e.target.value);
            resize();
          }}
          onKeyDown={onKeyDown}
          rows={1}
          placeholder="Ask a research question…"
          aria-label="Research question"
          className="block max-h-[200px] min-h-[52px] w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-[15px] leading-relaxed text-fg outline-none placeholder:text-fg-subtle"
        />
        <div className="flex items-center justify-between gap-2 px-2.5 pb-2.5">
          <button
            type="button"
            onClick={() => setUseWebSearch(!useWebSearch)}
            disabled={webAvailable === false}
            aria-pressed={webOn}
            title={
              webAvailable === false
                ? "Web search is unavailable — set TAVILY_API_KEY on the backend"
                : webOn
                  ? "Agents may search the web (click to use papers only)"
                  : "Papers only (click to allow web search)"
            }
            className={cn(
              "inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
              webOn
                ? "bg-accent-soft text-accent-text hover:bg-accent-soft/70"
                : "text-fg-muted ring-1 ring-inset ring-border hover:bg-surface-2",
            )}
          >
            <Globe size={14} />
            Web search {webOn ? "on" : "off"}
          </button>

          <div className="flex items-center gap-2">
            {tooShort && <span className="text-xs text-fg-subtle">At least 5 characters</span>}
            {value.length > MAX_LENGTH * 0.8 && (
              <span className="text-xs tabular-nums text-fg-subtle">
                {value.length}/{MAX_LENGTH}
              </span>
            )}
            {busy ? (
              <button
                type="button"
                onClick={onStop}
                aria-label="Stop research"
                title="Stop research"
                className="flex h-8 w-8 items-center justify-center rounded-lg bg-fg text-bg transition-opacity hover:opacity-85"
              >
                <Square size={12} className="fill-current" />
              </button>
            ) : (
              <button
                type="button"
                onClick={submit}
                disabled={trimmed.length < 5}
                aria-label="Send"
                title="Send (Enter)"
                className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-accent-fg transition-colors hover:bg-accent/90 disabled:bg-surface-2 disabled:text-fg-subtle"
              >
                <ArrowUp size={16} />
              </button>
            )}
          </div>
        </div>
      </div>
      <p className="mt-2 text-center text-[11px] text-fg-subtle">
        Enter to send · Shift+Enter for a new line · Answers can still contain mistakes — check the sources.
      </p>
    </div>
  );
}
