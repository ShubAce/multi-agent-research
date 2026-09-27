// frontend/lib/export.ts — Markdown export of answers and whole conversations
import type { Citation, Conversation, Message } from "./types";
import { hostname, shortAuthors } from "./utils";

function reference(c: Citation): string {
  if (c.type === "web") return `[${c.id}] ${c.title} — ${hostname(c.url)}. ${c.url}`;
  const meta = [shortAuthors(c.authors, 3), c.published?.slice(0, 4)].filter(Boolean).join(", ");
  return `[${c.id}] ${c.title}${meta ? ` — ${meta}` : ""}. ${c.url}`;
}

export function answerToMarkdown(question: string, answer: Message): string {
  const parts = [`## ${question}`, "", answer.content.trim()];
  if (answer.citations?.length) {
    parts.push("", "### References", "", ...answer.citations.map((c) => `${reference(c)}  `));
  }
  const quality: string[] = [];
  if (answer.confidenceScore != null) quality.push(`grounding ${Math.round(answer.confidenceScore * 100)}%`);
  if (answer.criticScore != null) quality.push(`critic ${answer.criticScore}/10`);
  if (quality.length) parts.push("", `_Quality: ${quality.join(" · ")}_`);
  return parts.join("\n");
}

export function conversationToMarkdown(conv: Conversation): string {
  const blocks = [`# ${conv.title}`, "", `_Exported ${new Date().toLocaleString()}_`];
  let question = "";
  for (const m of conv.messages) {
    if (m.role === "user") question = m.content;
    else if (m.run?.status === "done") blocks.push("", "---", "", answerToMarkdown(question, m));
  }
  return blocks.join("\n");
}

export function downloadText(filename: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown;charset=utf-8" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  a.click();
  URL.revokeObjectURL(url);
}

export function slug(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "research";
}
