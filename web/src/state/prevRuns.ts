// Previous completed result per slot (branches, sweep_mc) for the "▲ +80.0 mV · 이전 대비" delta chips and the
// grey "이전" ghost curve: when a slot finishes with data from a different payload (dataKey) than the one on
// screen, that one becomes `prev`. The baseline resets when the preset or the mode changes (a delta across
// presets or across modes would compare unlike runs).
import { create } from "zustand";
import { useStore, type State } from "./store";

export interface RunSnap {
  data: unknown;
  dataKey: string;
}

export interface PrevRunsState {
  /** Result currently on screen per slot (the latest completed one). */
  cur: Record<string, RunSnap | undefined>;
  /** The completed result before `cur` (different dataKey), or undefined. */
  prev: Record<string, RunSnap | undefined>;
}

/** Slots whose previous result is kept. */
export const PREV_SLOTS = ["branches", "sweep_mc"] as const;

export const usePrevRuns = create<PrevRunsState>(() => ({ cur: {}, prev: {} }));

/**
 * Pure transition (unit-tested): the new state after the app state changed from `before` to `next` (`before`
 * null: initialisation). A preset or mode change empties both maps; the result then on screen is not a
 * baseline, only the next completed run is.
 */
export function nextPrevRuns(st: PrevRunsState, next: Pick<State, "results" | "preset" | "mode"> & Partial<Pick<State, "params">>, before: Pick<State, "results" | "preset" | "mode"> & Partial<Pick<State, "params">> | null): PrevRunsState {
  const reset = !!before && (next.preset !== before.preset || next.mode !== before.mode || (next.params?.device.model ?? "detailed") !== (before.params?.device.model ?? "detailed"));
  let cur = reset ? {} : st.cur;
  let prev = reset ? {} : st.prev;
  for (const k of PREV_SLOTS) {
    const e = next.results[k];
    if (!e || e.status !== "done" || !e.dataKey || e.data === undefined) continue;
    if (before && before.results[k] === e) continue; // this entry did not change
    const c = cur[k];
    if (c && c.dataKey === e.dataKey) {
      if (c.data !== e.data) cur = { ...cur, [k]: { data: e.data, dataKey: e.dataKey } }; // same payload re-run
      continue;
    }
    cur = { ...cur, [k]: { data: e.data, dataKey: e.dataKey } };
    prev = { ...prev, [k]: c };
  }
  return cur === st.cur && prev === st.prev ? st : { cur, prev };
}

if (typeof window !== "undefined") {
  usePrevRuns.setState(nextPrevRuns(usePrevRuns.getState(), useStore.getState(), null));
  useStore.subscribe((s, p) => {
    if (s.results === p.results && s.preset === p.preset && s.mode === p.mode && s.params.device.model === p.params.device.model) return;
    const st = usePrevRuns.getState();
    const n = nextPrevRuns(st, s, p);
    if (n !== st) usePrevRuns.setState(n, true);
  });
}

/** The previous result of `slot` (undefined until a second, different run completed). */
export function usePrevRun<T>(slot: (typeof PREV_SLOTS)[number]): { data: T; dataKey: string } | undefined {
  return usePrevRuns((s) => s.prev[slot]) as { data: T; dataKey: string } | undefined;
}
