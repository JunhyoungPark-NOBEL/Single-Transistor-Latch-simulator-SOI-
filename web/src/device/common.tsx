// Shared hooks/helpers for result panels.
import { useMemo } from "react";
import type { Arr, DeviceBlock, Kind } from "../api/types";
import { palette } from "../plots/theme";
import { useStore, type ResultEntry } from "../state/store";
import { canonical } from "../utils/object";

export function useEntry<T>(key: string): { entry: ResultEntry | undefined; data: T | undefined } {
  const entry = useStore((s) => s.results[key]);
  return { entry, data: entry?.data as T | undefined };
}

/** Canonical key of the payload the panel *would* compute now (for the "parameters changed" badge). */
export function useCurrentKey(kind: Kind, payload: unknown): string {
  return useMemo(() => canonical({ kind, payload }), [kind, payload]);
}

/** True when `entry` holds data computed from a different payload than `currentKey` (and is not re-running). */
export function isStale(entry: ResultEntry | undefined, currentKey: string): boolean {
  if (!entry?.dataKey || entry.data === undefined) return false;
  if (entry.status === "running" || entry.status === "queued") return false;
  return entry.dataKey !== currentKey;
}

export function usePalette() {
  const theme = useStore((s) => s.theme);
  return palette(theme);
}

/** Nulls instead of non-positive values (log axes). */
export const pos = (a: Arr | undefined): (number | null)[] => (a ?? []).map((v) => (typeof v === "number" && v > 0 ? v : null));
export const abs = (a: Arr | undefined): (number | null)[] => (a ?? []).map((v) => (typeof v === "number" ? Math.abs(v) : null));
export const nums = (a: Arr | undefined): (number | null)[] => (a ?? []).map((v) => (typeof v === "number" && Number.isFinite(v) ? v : null));

export function isPaperReference(d: DeviceBlock): boolean {
  const iph = d.light.mode === "power" ? d.light.power_mW * d.light.responsivity_pA_per_mW : d.light.iph_pA;
  return Math.abs(d.vg + 2) < 1e-6 && iph === 0;
}

/** Subsample indices for direction arrows (always includes the big jumps). */
export function arrowIndices(x: Arr, y: Arr, every: number): number[] {
  const idx = new Set<number>();
  for (let i = every; i < x.length - 1; i += every) idx.add(i);
  for (let i = 1; i < y.length; i++) {
    const a = y[i - 1];
    const b = y[i];
    if (typeof a === "number" && typeof b === "number" && a > 0 && b > 0 && Math.abs(Math.log10(b / a)) > 1) {
      idx.add(i - 1);
      idx.add(i);
    }
  }
  return [...idx].sort((a, b) => a - b);
}

export function pick<T>(a: T[] | undefined, idx: number[]): T[] {
  return idx.map((i) => (a ?? [])[i]);
}

/**
 * Explicit log-axis range for currents: the engine returns exponentially small values (e.g. 1e-58 A at
 * V_D → 0) that would stretch an autoranged log axis over 50 decades. The data stay untouched (hover/CSV);
 * only the initial view is limited to [max(floor, hi·10^-maxDecades), hi].
 */
export function logRange(arrays: ((number | null)[] | undefined)[], floor = 1e-17, maxDecades = 14): [number, number] | undefined {
  let hi = 0;
  let lo = Infinity;
  for (const a of arrays)
    for (const v of a ?? [])
      if (typeof v === "number" && v > 0 && Number.isFinite(v)) {
        if (v > hi) hi = v;
        if (v < lo) lo = v;
      }
  if (!(hi > 0)) return undefined;
  const bottom = Math.max(lo, floor, hi * 10 ** -maxDecades);
  return [Math.log10(bottom) - 0.15, Math.log10(hi) + 0.25];
}

/** Linear y-range from the values whose x lies in [x0, x1] (padding 8 %). */
export function rangeWithin(x: (number | null)[], y: (number | null)[], x0: number, x1: number): [number, number] | undefined {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < x.length; i++) {
    const xv = x[i];
    const yv = y[i];
    if (xv == null || yv == null || xv < x0 || xv > x1) continue;
    lo = Math.min(lo, yv);
    hi = Math.max(hi, yv);
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return undefined;
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
  return [lo - pad, hi + pad];
}
