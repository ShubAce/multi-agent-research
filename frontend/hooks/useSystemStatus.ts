// frontend/hooks/useSystemStatus.ts
"use client";

import { useEffect } from "react";
import { create } from "zustand";
import { getReady } from "@/lib/api";
import type { ReadyResponse } from "@/lib/types";

const POLL_MS = 20_000;

interface StatusStore {
  status: ReadyResponse | null;
  apiReachable: boolean | null; // null until the first check finishes
  refresh: () => Promise<void>;
}

export const useSystemStatus = create<StatusStore>((set) => ({
  status: null,
  apiReachable: null,
  refresh: async () => {
    try {
      set({ status: await getReady(), apiReachable: true });
    } catch {
      set({ apiReachable: false });
    }
  },
}));

/** Mount once (in the page) to keep system status fresh. */
export function useSystemStatusPoller() {
  useEffect(() => {
    const { refresh } = useSystemStatus.getState();
    refresh();
    const id = setInterval(refresh, POLL_MS);
    const onFocus = () => refresh();
    window.addEventListener("focus", onFocus);
    return () => {
      clearInterval(id);
      window.removeEventListener("focus", onFocus);
    };
  }, []);
}
