// frontend/components/chat/UserMessage.tsx
import type { Message } from "@/lib/types";

export function UserMessage({ message }: { message: Message }) {
  return (
    <div className="flex animate-fade-in justify-end">
      <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-surface-2 px-4 py-2.5 text-[15px] leading-relaxed text-fg ring-1 ring-inset ring-border/60">
        {message.content}
      </div>
    </div>
  );
}
