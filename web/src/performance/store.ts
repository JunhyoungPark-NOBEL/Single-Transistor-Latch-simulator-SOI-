import { create } from "zustand";
import { request, runJob, type HttpConfiguration } from "../api/client";
import { getSessionToken, useConnection } from "../api/connection";
import { getBackend } from "../state/runner";
import { useStore } from "../state/store";
import type { PerformanceData } from "./types";

interface PerformanceState {
  data: PerformanceData | null;
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
  revision: number;
  calibrating: boolean;
  calibrationProgress: number;
  calibrationMessage: string;
}
export const usePerformance = create<PerformanceState>(() => ({ data: null, status: "idle", error: null, revision: 0, calibrating: false, calibrationProgress: 0, calibrationMessage: "" }));
let generation = 0;
let inFlight: Promise<void> | null = null;
let fallback: Record<string, unknown> | null = null;

export function performanceConfig(): HttpConfiguration {
  return { endpoint: useConnection.getState().endpoint, token: getSessionToken() };
}
export async function refreshPerformance(force = false): Promise<void> {
  if (inFlight) return inFlight;
  if (!force && usePerformance.getState().status === "ready") return;
  const epoch = generation;
  const online = useStore.getState().backend === "online";
  const config = performanceConfig();
  usePerformance.setState({ status: "loading", error: null });
  const task = (async () => {
    try {
      let data: PerformanceData;
      if (online) data = await request<PerformanceData>("GET", "/api/performance", undefined, 15000, config);
      else {
        if (!fallback) {
          const res = await fetch(`${import.meta.env.BASE_URL}data/performance-catalog.json`);
          if (!res.ok) throw new Error("performance-catalog-unavailable");
          fallback = await res.json() as Record<string, unknown>;
        }
        data = { catalog: fallback! };
      }
      if (epoch === generation) usePerformance.setState((s) => ({ data, status: "ready", revision: s.revision + 1 }));
    } catch (e) {
      if (epoch === generation) usePerformance.setState({ status: "error", error: (e as Error).message });
    }
  })().finally(() => { if (inFlight === task) inFlight = null; });
  inFlight = task;
  return task;
}

export async function calibratePerformance(): Promise<void> {
  if (usePerformance.getState().calibrating || useStore.getState().backend !== "online") return;
  const epoch = generation;
  usePerformance.setState({ calibrating: true, calibrationProgress: 0, calibrationMessage: "", error: null });
  try {
    await runJob(getBackend(), "performance_calibrate", {}, {
      isAborted: () => epoch !== generation,
      onStatus: (s) => { if (epoch === generation) usePerformance.setState({ calibrationProgress: s.progress, calibrationMessage: s.message }); },
    });
    if (epoch === generation) await refreshPerformance(true);
  } catch (e) {
    if (epoch === generation) usePerformance.setState({ error: (e as Error).message });
  } finally {
    if (epoch === generation) usePerformance.setState({ calibrating: false, calibrationMessage: "" });
  }
}

/** Connection-local metadata, never browser stopwatch timings, feeds host estimates. */
export function startPerformanceTracking(): () => void {
  let refreshTimer: ReturnType<typeof setTimeout> | undefined;
  const reset = () => {
    generation++;
    inFlight = null;
    usePerformance.setState((s) => ({ data: null, status: "idle", error: null, revision: s.revision + 1, calibrating: false, calibrationProgress: 0, calibrationMessage: "" }));
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => void refreshPerformance(), 100);
  };
  const unsubConnection = useConnection.subscribe((s, prev) => {
    if (s.endpoint !== prev.endpoint || s.status !== prev.status) reset();
  });
  const unsubStore = useStore.subscribe((s, prev) => {
    if (s.backend !== prev.backend || s.health?.version !== prev.health?.version) { reset(); return; }
    if (s.backend === "online" && Object.entries(s.results).some(([key, e]) => e.status === "done" && !e.cached && !e.mock && prev.results[key]?.status !== "done")) {
      clearTimeout(refreshTimer);
      refreshTimer = setTimeout(() => void refreshPerformance(true), 200);
    }
  });
  // Other users and restarts can change the queue or worker readiness without a local edit.
  const poll = setInterval(() => { if (useStore.getState().backend === "online") void refreshPerformance(true); }, 20000);
  void refreshPerformance();
  return () => { unsubConnection(); unsubStore(); clearTimeout(refreshTimer); clearInterval(poll); generation++; inFlight = null; };
}
