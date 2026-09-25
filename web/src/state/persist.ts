// Restoring persisted UI state (localStorage) defensively: the stored JSON may come from an older schema,
// another app version or be corrupted by hand. Every field is validated; anything unknown or ill-typed
// falls back to the default instead of crashing the app (pure functions — unit-tested).
import type { Mode, PresetId } from "../api/types";
import { BENCH_ORDER } from "../params/benches";
import type { ParamRoot, Tab } from "../params/schema";
import { getPath, mergeDefaults, setPath, type Path } from "../utils/object";
import type { VgRange } from "../utils/payload";

export type Lang = "ko" | "en";
export type Theme = "light" | "dark";

/** Schema version of the stored object. 2: rise_s/fall_s default to "auto" (server default). */
export const PERSIST_VERSION = 2;

export interface Persisted {
  v: number;
  mode: Mode;
  lang: Lang;
  theme: Theme;
  autoRun: boolean;
  /** The user has set the auto-run switch at least once (until then auto-run defaults to on). */
  autoRunChosen?: boolean;
  preset: PresetId;
  params: ParamRoot;
  vgRange: VgRange;
  vgsRange: VgRange;
  tab: Tab;
}

export const TABS: readonly Tab[] = ["device", "circuit", "validation", "physics"];
export const MODES: readonly Mode[] = ["deterministic", "stochastic"];
const LANGS: readonly Lang[] = ["ko", "en"];
const THEMES: readonly Theme[] = ["light", "dark"];
export const PRESETS: readonly PresetId[] = ["paper", "photo", "custom"];

const isPlainObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const oneOf = <T>(v: unknown, allowed: readonly T[]): v is T => allowed.includes(v as T);

/** Parse the stored JSON into a partial state with only valid fields (never throws). */
export function parsePersisted(raw: string | null | undefined): Partial<Persisted> {
  if (!raw) return {};
  let v: unknown;
  try {
    v = JSON.parse(raw);
  } catch {
    return {};
  }
  if (!isPlainObject(v)) return {};
  const out: Partial<Persisted> = { v: typeof v.v === "number" && Number.isFinite(v.v) ? v.v : 1 };
  if (oneOf(v.mode, MODES)) out.mode = v.mode;
  if (oneOf(v.lang, LANGS)) out.lang = v.lang;
  if (oneOf(v.theme, THEMES)) out.theme = v.theme;
  if (oneOf(v.tab, TABS)) out.tab = v.tab;
  if (oneOf(v.preset, PRESETS)) out.preset = v.preset;
  if (typeof v.autoRun === "boolean") out.autoRun = v.autoRun;
  if (typeof v.autoRunChosen === "boolean") out.autoRunChosen = v.autoRunChosen;
  if (isPlainObject(v.params)) out.params = v.params as unknown as ParamRoot;
  if (isPlainObject(v.vgRange)) out.vgRange = v.vgRange as unknown as VgRange;
  if (isPlainObject(v.vgsRange)) out.vgsRange = v.vgsRange as unknown as VgRange;
  return out;
}

const LOCAL_MODES = ["none", "frozen", "evolving"] as const;
const LOCAL_ACTIONS = ["gidl", "local_avalanche", "junction", "multiplication"] as const;

/** Enumerated fields: a stored value outside the allowed set is replaced by the default. */
const ENUMS: [Path, readonly unknown[]][] = [
  [["device", "preset"], PRESETS],
  [["device", "light", "mode"], ["iph", "power"]],
  [["device", "ext", "loc_carriers"], [0, 1, 2]],
  [["stochastic", "local_state", "mode"], LOCAL_MODES],
  [["stochastic", "local_state", "action"], LOCAL_ACTIONS],
  [["stochastic", "engine"], ["auto", "general", "calibrated_lookup"]],
  [["circuit", "bench"], BENCH_ORDER],
  [["circuit", "solver", "method"], ["BE", "TRAP"]],
  [["circuit", "stochastic", "local_state", "mode"], LOCAL_MODES],
  [["circuit", "stochastic", "local_state", "action"], LOCAL_ACTIONS],
  [["circuit", "bench_params", "coupled", "source"], ["ramp", "pulse"]],
];

/**
 * Merge stored parameters over the defaults (unknown keys dropped, types checked by mergeDefaults), then
 * enforce enumerations and numeric-list contents so every value the UI and payload builders read is valid.
 */
export function restoreParams(base: ParamRoot, stored: unknown, version = PERSIST_VERSION): ParamRoot {
  let p = mergeDefaults(base, stored);
  if (version < 2) {
    // v1 stored the old built-in 1 µs edges as plain values; they were never a user choice → back to auto
    for (const bench of ["pulse", "pbit", "coupled"] as const)
      for (const k of ["rise_s", "fall_s"])
        if (p.circuit.bench_params[bench]?.[k] === 1e-6) p = setPath(p, ["circuit", "bench_params", bench, k], getPath(base, ["circuit", "bench_params", bench, k]));
  }
  for (const [path, allowed] of ENUMS) {
    if (!allowed.includes(getPath(p, path))) p = setPath(p, path, getPath(base, path));
  }
  // numeric lists of the circuit benches (e.g. amplitudes_V): keep finite numbers only
  for (const bench of BENCH_ORDER) {
    const bp = p.circuit.bench_params[bench];
    for (const [k, v] of Object.entries(bp)) {
      if (Array.isArray(v) && v.some((x) => typeof x !== "number" || !Number.isFinite(x))) {
        p = setPath(p, ["circuit", "bench_params", bench, k], v.filter((x) => typeof x === "number" && Number.isFinite(x)));
      }
    }
  }
  return p;
}

/** V_G range for the V_G-curve panels: finite, ordered, 2 ≤ n ≤ maxN. */
export function restoreRange(def: VgRange, stored: unknown, maxN = 61): VgRange {
  const r = mergeDefaults(def, stored);
  if (!Number.isFinite(r.min) || !Number.isFinite(r.max) || r.min >= r.max) return { ...def, n: clampN(r.n, def.n, maxN) };
  return { min: r.min, max: r.max, n: clampN(r.n, def.n, maxN) };
}
const clampN = (n: number, d: number, maxN: number) => (Number.isFinite(n) ? Math.max(2, Math.min(maxN, Math.round(n))) : d);

/**
 * Auto-run at start-up: on by default (Device · deterministic re-runs on every change) until the user has set
 * the switch; then the stored choice. Older stored states (autoRun false = the old default, no choice
 * recorded) therefore start with auto-run on.
 */
export function initialAutoRun(p: Partial<Persisted>): boolean {
  return p.autoRunChosen ? (p.autoRun ?? true) : true;
}

/** Group open/closed map (sidebar cards). */
export function parseOpenState(raw: string | null | undefined): Record<string, boolean> {
  try {
    const v: unknown = raw ? JSON.parse(raw) : {};
    if (!isPlainObject(v)) return {};
    return Object.fromEntries(Object.entries(v).filter(([, x]) => typeof x === "boolean")) as Record<string, boolean>;
  } catch {
    return {};
  }
}
