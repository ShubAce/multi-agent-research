// frontend/components/MessageBubble.tsx
"use client";

import type { Message } from "@/lib/types";
import { SourceCitations } from "./SourceCitations";

interface Props {
  message: Message;
}

function CriticBadge({ score }: { score: number }) {
  const color =
    score >= 8 ? "bg-green-100 text-green-700" :
    score >= 6 ? "bg-yellow-100 text-yellow-700" :
                 "bg-red-100 text-red-700";
  const label =
    score >= 8 ? "High quality" :
    score >= 6 ? "Good" :
                 "Weak answer";
  return (
    <span
      className={`text-xs font-medium px-2 py-0.5 rounded-full ${color}`}
      title={`Critic score: ${score}/10`}
    >
      {label} {score}/10
    </span>
  );
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-blue-600 px-4 py-3 text-white text-sm leading-relaxed shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  const isEmpty = !message.content;

  return (
    <div className="flex justify-start">
      <div className="max-w-[90%] space-y-2">
        <div className="rounded-2xl rounded-tl-sm bg-white border border-slate-200 px-4 py-3 text-slate-800 text-sm leading-relaxed shadow-sm">
          {isEmpty ? (
            <span className="flex gap-1 items-center text-slate-400">
              <span className="animate-bounce" style={{ animationDelay: "0ms" }}>·</span>
              <span className="animate-bounce" style={{ animationDelay: "150ms" }}>·</span>
              <span className="animate-bounce" style={{ animationDelay: "300ms" }}>·</span>
            </span>
          ) : (
            <div className="whitespace-pre-wrap">{message.content}</div>
          )}
        </div>

        {/* Badges row */}
        {message.agentsUsed && message.agentsUsed.length > 0 && (
          <div className="flex flex-wrap items-center gap-1 px-1">
            {/* Agent tags */}
            {message.agentsUsed.map((agent) => (
              <span
                key={agent}
                className="text-xs bg-slate-100 text-slate-500 px-2 py-0.5 rounded-full"
              >
                {agent}
              </span>
            ))}

            {/* Faithfulness score */}
            {message.confidenceScore !== undefined && (
              <span
                className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                  message.confidenceScore >= 0.8 ? "bg-green-100 text-green-700" :
                  message.confidenceScore >= 0.5 ? "bg-yellow-100 text-yellow-700" :
                  "bg-red-100 text-red-700"
                }`}
                title="Fraction of answer grounded in retrieved sources"
              >
                {Math.round(message.confidenceScore * 100)}% grounded
              </span>
            )}

            {/* Critic score */}
            {message.criticScore !== undefined && (
              <CriticBadge score={message.criticScore} />
            )}
          </div>
        )}

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <SourceCitations citations={message.citations} />
        )}
      </div>
    </div>
  );
}
