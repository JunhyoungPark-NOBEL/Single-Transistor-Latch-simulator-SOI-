// Typed client for user-drawn circuits: kind "circuit", bench "custom" (docs/WEB_CONTRACT.md §6).
// Request/response types, a result guard, the job submission (reuses api/client.ts runJob: polling,
// cancel, HTTP 429 retry) and client-side helpers (interpolation at a time, signal lookup).
import type { Backend } from "./client";
import { runJob, type RunOptions } from "./client";
import type { Arr, CircuitResult, DeviceBlock, LocalStateBlock, Mode } from "./types";

// ---------------------------------------------------------------- request (§6)
export type Wave =
  | { kind: "dc"; value: number }
  | { kind: "pulse"; v1: number; v2: number; td: number; tr: number; tf: number; pw: number; per: number; ncycles: number }
  | { kind: "pwl"; t: number[]; v: number[] }
  | { kind: "sine"; vo: number; va: number; freq: number; td: number; theta: number };
export type WaveKind = Wave["kind"];

export type CustomElement =
  | { type: "R" | "C"; name: string; nodes: [string, string]; value: number }
  | { type: "V" | "I"; name: string; nodes: [string, string]; wave: Wave }
  | {
      type: "STL";
      name: string;
      nodes: { d: string; g: string; s: string };
      device: DeviceBlock;
      light_pA: Wave | null;
      /** Extension (not in §6): the library device's local-state settings for stochastic runs. */
      local_state?: LocalStateBlock;
    }
  | {
      /** Comparator (§6.3): output (behavioural voltage source to ground) = v_high when V(in) − V(inm) > v_ref. */
      type: "CMP";
      name: string;
      nodes: { in: string; out: string; inm?: string };
      v_ref: number;
      v_high?: number;
      v_low?: number;
      hysteresis?: number;
      width?: number;
    };

export interface CustomTran {
  t_stop_s: number;
  t_start_save_s: number;
  dt_max_s: number;
  dt_min_s: number;
  method: "BE" | "TRAP";
  reltol: number;
}
export interface CustomStochastic {
  seed: number;
  n_runs: number;
  carrier_noise: boolean;
  ld_carrier_noise: boolean;
  local_state: LocalStateBlock;
  /** §6.3: true → `local_state` applies to every STL; false → each STL's own `local_state` (library device). */
  local_state_override: boolean;
}
export interface CustomCircuitRequest {
  bench: "custom";
  mode: Mode;
  netlist: { elements: CustomElement[] };
  tran: CustomTran;
  stochastic?: CustomStochastic;
  detect: { i_threshold_A: number; hysteresis: number };
  probes: string[] | null;
}

// ---------------------------------------------------------------- response (§4 + §6.2)
export interface Envelope {
  key: string;
  t: Arr;
  mean: Arr;
  sd: Arr;
  p05: Arr;
  p95: Arr;
}
export interface CustomEvent {
  run: number;
  kind: string;
  t: number;
  cell?: string;
  v_d?: number;
  value?: number;
  v_src?: number;
}
export interface ResolvedElement {
  type?: string;
  kind?: string;
  name: string;
  nodes: string[] | Record<string, string>;
  value?: number | string;
  wave?: unknown;
  [k: string]: unknown;
}
/** Per-comparator firing statistics (§6.3): one window per period of the periodic pulse source. */
export interface ComparatorStats {
  name: string;
  nodes: { in: string; inm: string; out: string };
  v_ref: number;
  v_high: number;
  v_low: number;
  hysteresis: number;
  window_source: string | null;
  t_windows: number[];
  /** runs × windows: 1 = the output was high within that window, null = not reached (may be capped in windows). */
  bits: (number | null)[][];
  p_fire_window: (number | null)[];
  p_fire_window_err: (number | null)[];
  p_fire: number | null;
  lag1: number | null;
  n_bits: number;
  p_fire_run: (number | null)[];
}

export interface CustomCircuitResult extends Omit<CircuitResult, "events" | "schematic"> {
  events: CustomEvent[];
  comparators?: ComparatorStats[];
  nodes?: string[];
  elements?: ResolvedElement[];
  /** Operating point at t = 0. Shape not fixed by the contract: flat {"V(n)": v} or {nodes:{}, currents:{}}. */
  op?: Record<string, unknown> | null;
  envelopes?: Envelope[];
  schematic?: CircuitResult["schematic"];
}

// ---------------------------------------------------------------- guard
const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

/** Missing/ill-typed top-level keys of a custom-circuit result (empty = renderable). */
export function checkCustomResult(v: unknown): string[] {
  if (!isObj(v)) return ["<result is not an object>"];
  const miss: string[] = [];
  if (!Array.isArray(v.runs)) miss.push("runs");
  else if (!v.runs.length || !isObj(v.runs[0]) || !Array.isArray((v.runs[0] as Record<string, unknown>).t) || !Array.isArray((v.runs[0] as Record<string, unknown>).signals)) miss.push("runs[0].t/signals");
  if (!Array.isArray(v.summary)) miss.push("summary");
  if (v.events !== undefined && !Array.isArray(v.events)) miss.push("events");
  if (v.envelopes !== undefined && v.envelopes !== null && !Array.isArray(v.envelopes)) miss.push("envelopes");
  return miss;
}

/** True when a job error means the server does not know bench "custom" yet (→ demo fallback). */
export function isUnknownBenchError(msg: string | undefined | null): boolean {
  return !!msg && /unknown bench|bench must be|not supported.*custom|custom.*not (yet )?(supported|implemented)/i.test(msg);
}

// ---------------------------------------------------------------- submit
export function runCustomCircuit(backend: Backend, req: CustomCircuitRequest, opt: RunOptions = {}): Promise<CustomCircuitResult> {
  return runJob<CustomCircuitResult>(backend, "circuit", req, opt);
}

// ---------------------------------------------------------------- helpers
/** Index of the last sample with t[i] <= x (binary search on a non-decreasing time axis; nulls skipped). */
export function searchTime(t: Arr, x: number): number {
  let lo = 0;
  let hi = t.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    const v = t[mid];
    if (v != null && v <= x) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** Linear interpolation of `values` (aligned with `t`) at time x; null outside/at gaps. */
export function valueAt(t: Arr, values: Arr, x: number): number | null {
  if (!t.length || !values.length) return null;
  const t0 = t[0];
  const tn = t[t.length - 1];
  if (t0 == null || tn == null) return null;
  if (x <= t0) return values[0] ?? null;
  if (x >= tn) return values[values.length - 1] ?? null;
  const i = searchTime(t, x);
  const ta = t[i];
  const tb = t[i + 1];
  const a = values[i];
  const b = values[i + 1];
  if (ta == null || a == null) return null;
  if (tb == null || b == null || tb <= ta) return a;
  return a + ((b - a) * (x - ta)) / (tb - ta);
}

/** Canonical probe keys (§6.1). */
export const vKey = (node: string) => `V(${node})`;
export const iKey = (name: string, terminal?: "d" | "g" | "s") => (terminal ? `I(${name}.${terminal})` : `I(${name})`);

/** Parse "V(n)" / "I(R1)" / "I(X1.d)" / "X1.u" → description. */
export function parseProbe(key: string): { type: "V"; node: string } | { type: "I"; el: string; terminal?: string } | { type: "state"; el: string; q: string } | null {
  let m = /^V\((.+)\)$/.exec(key);
  if (m) return { type: "V", node: m[1] };
  m = /^I\(([^.()]+)(?:\.([dgs]))?\)$/.exec(key);
  if (m) return { type: "I", el: m[1], terminal: m[2] };
  m = /^([^.()]+)\.([a-z_]+)$/i.exec(key);
  if (m) return { type: "state", el: m[1], q: m[2] };
  return null;
}
