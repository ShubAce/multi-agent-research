// frontend/hooks/useTheme.ts
"use client";

import { useEffect } from "react";
import { useChatStore } from "@/store/chat";

/** Keep the `dark` class on <html> in sync with the saved preference and the OS. */
export function useApplyTheme() {
  const theme = useChatStore((s) => s.theme);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () =>
      document.documentElement.classList.toggle(
        "dark",
        theme === "dark" || (theme === "system" && media.matches),
      );
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, [theme]);
}
