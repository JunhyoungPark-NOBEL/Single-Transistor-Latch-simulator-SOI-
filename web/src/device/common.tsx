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
