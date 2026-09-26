// Source waveforms (SPICE semantics): evaluation, corner points for previews, SPICE text, validation
// and a breakpoint count for the run-time estimate.
import type { Wave, WaveKind } from "../api/circuitCustom";
import { toSpice } from "./si";

export const defaultWave = (kind: WaveKind, level = 1): Wave => {
  switch (kind) {
    case "dc":
      return { kind: "dc", value: level };
    case "pulse":
      return { kind: "pulse", v1: 0, v2: level, td: 0, tr: 10e-6, tf: 10e-6, pw: 200e-6, per: 1e-3, ncycles: 0 };
    case "pwl":
      return { kind: "pwl", t: [0, 1e-3, 2e-3], v: [0, level, 0] };
    case "sine":
      return { kind: "sine", vo: 0, va: level, freq: 1e3, td: 0, theta: 0 };
  }
};

/** Convert a waveform to another kind keeping its levels where they map naturally. */
export function convertWave(w: Wave, kind: WaveKind): Wave {
  if (w.kind === kind) return w;
  const hi = w.kind === "dc" ? w.value : w.kind === "pulse" ? w.v2 : w.kind === "sine" ? w.vo + w.va : Math.max(...w.v, 0);
  const lo = w.kind === "pulse" ? w.v1 : w.kind === "sine" ? w.vo : 0;
  const out = defaultWave(kind, hi || 1);
  if (out.kind === "pulse") out.v1 = lo === hi ? 0 : lo;
  if (out.kind === "sine") {
    out.vo = lo;
    out.va = hi - lo || 1;
  }
  return out;
}

/** Value of the waveform at time t (s). */
export function waveAt(w: Wave, t: number): number {
  switch (w.kind) {
    case "dc":
      return w.value;
    case "pulse": {
      if (t < w.td) return w.v1;
      const per = w.per > 0 ? w.per : Infinity;
      const k = Math.floor((t - w.td) / per);
      if (w.ncycles > 0 && k >= w.ncycles) return w.v1;
      const tl = t - w.td - (Number.isFinite(per) ? k * per : 0);
      if (tl < w.tr) return w.tr > 0 ? w.v1 + ((w.v2 - w.v1) * tl) / w.tr : w.v2;
      if (tl < w.tr + w.pw) return w.v2;
      if (tl < w.tr + w.pw + w.tf) return w.tf > 0 ? w.v2 + ((w.v1 - w.v2) * (tl - w.tr - w.pw)) / w.tf : w.v1;
      return w.v1;
    }
    case "pwl": {
      const n = Math.min(w.t.length, w.v.length);
      if (!n) return 0;
      if (t <= w.t[0]) return w.v[0];
      for (let i = 1; i < n; i++) {
        if (t <= w.t[i]) {
          const dt = w.t[i] - w.t[i - 1];
          return dt > 0 ? w.v[i - 1] + ((w.v[i] - w.v[i - 1]) * (t - w.t[i - 1])) / dt : w.v[i];
        }
      }
      return w.v[n - 1];
    }
    case "sine": {
      if (t < w.td) return w.vo;
      const s = t - w.td;
      return w.vo + w.va * Math.exp(-s * w.theta) * Math.sin(2 * Math.PI * w.freq * s);
    }
  }
}

/**
 * Points for a preview / feasibility over [0, tStop]: exact corners for DC/PULSE/PWL (≤ maxPts, later
 * cycles are dropped with `truncated`), uniform sampling for SINE.
 */
export function wavePoints(w: Wave, tStop: number, maxPts = 600): { t: number[]; v: number[]; truncated: boolean } {
  const T = Math.max(tStop, 1e-15);
  const t: number[] = [];
  const v: number[] = [];
  const push = (a: number, b: number) => {
    if (a > T) return;
    t.push(a);
    v.push(b);
  };
  let truncated = false;
  switch (w.kind) {
    case "dc":
      push(0, w.value);
      push(T, w.value);
      break;
    case "pulse": {
      push(0, w.v1);
      const per = w.per > 0 ? w.per : Infinity;
      const nMax = w.ncycles > 0 ? w.ncycles : Number.isFinite(per) ? Math.ceil((T - w.td) / per) + 1 : 1;
      for (let k = 0; k < nMax; k++) {
        const t0 = w.td + (Number.isFinite(per) ? k * per : 0);
        if (t0 > T) break;
        if (t.length > maxPts) {
          truncated = true;
          break;
        }
        push(t0, w.v1);
        push(t0 + w.tr, w.v2);
        push(t0 + w.tr + w.pw, w.v2);
        push(t0 + w.tr + w.pw + w.tf, w.v1);
      }
      if (t[t.length - 1] < T) push(T, waveAt(w, T));
      break;
    }
    case "pwl": {
      const n = Math.min(w.t.length, w.v.length);
      if (!n) break;
      if (w.t[0] > 0) push(0, w.v[0]);
      for (let i = 0; i < n && i < maxPts; i++) push(w.t[i], w.v[i]);
      if (w.t[n - 1] < T) push(T, w.v[n - 1]);
      else if (w.t[n - 1] > T) push(T, waveAt(w, T));
      break;
    }
    case "sine": {
      const cycles = w.freq > 0 ? (T - w.td) * w.freq : 0;
      const n = Math.min(maxPts, Math.max(64, Math.ceil(cycles * 24)));
      truncated = cycles * 24 > maxPts;
      for (let i = 0; i <= n; i++) {
        const x = (T * i) / n;
        push(x, waveAt(w, x));
      }
      break;
    }
  }
  return { t, v, truncated };
}

/** Number of corners (breakpoints) inside [0, tStop] — each costs the adaptive stepper a few steps. */
export function breakpointCount(w: Wave | null | undefined, tStop: number): number {
  if (!w) return 0;
  switch (w.kind) {
    case "dc":
      return 0;
    case "pulse": {
      const per = w.per > 0 ? w.per : Infinity;
      const span = Math.max(0, tStop - w.td);
      let n = Number.isFinite(per) ? Math.ceil(span / per) : 1;
      if (w.ncycles > 0) n = Math.min(n, w.ncycles);
      return 4 * n;
    }
    case "pwl":
      return w.t.filter((x) => x <= tStop).length;
    case "sine":
      return Math.ceil(Math.max(0, tStop - w.td) * w.freq * 8);
  }
}

/** SPICE source text: "PULSE(0 3.8 0 10u 10u 200u 1m 10)". */
export function waveSpice(w: Wave): string {
  const n = (x: number) => toSpice(x, 5);
  switch (w.kind) {
    case "dc":
      return `DC ${n(w.value)}`;
    case "pulse":
      return `PULSE(${[w.v1, w.v2, w.td, w.tr, w.tf, w.pw, w.per].map(n).join(" ")}${w.ncycles > 0 ? ` ${w.ncycles}` : ""})`;
    case "pwl": {
      const pts = w.t.map((x, i) => `${n(x)} ${n(w.v[i] ?? 0)}`);
      return `PWL(${pts.length > 8 ? [...pts.slice(0, 7), "…", pts[pts.length - 1]].join(" ") : pts.join(" ")})`;
    }
    case "sine":
      return `SINE(${[w.vo, w.va, w.freq, w.td, w.theta].map(n).join(" ")})`;
  }
}

export type WaveIssue = { level: "error" | "warning"; key: string; vars?: Record<string, string | number> };

/** Structural problems of a waveform (keys are i18n suffixes under "schematic.wave.err."). */
export function waveIssues(w: Wave): WaveIssue[] {
  const out: WaveIssue[] = [];
  const fin = (...xs: number[]) => xs.every((x) => typeof x === "number" && Number.isFinite(x));
  switch (w.kind) {
    case "dc":
      if (!fin(w.value)) out.push({ level: "error", key: "nan" });
      break;
    case "pulse":
      if (!fin(w.v1, w.v2, w.td, w.tr, w.tf, w.pw, w.per, w.ncycles)) out.push({ level: "error", key: "nan" });
      else {
        if (w.td < 0 || w.tr < 0 || w.tf < 0 || w.pw < 0 || w.per < 0 || w.ncycles < 0) out.push({ level: "error", key: "negative" });
        if (w.per > 0 && w.tr + w.pw + w.tf > w.per * (1 + 1e-9)) out.push({ level: "error", key: "period" });
        if (w.tr === 0 || w.tf === 0) out.push({ level: "warning", key: "zeroEdge" });
      }
      break;
    case "pwl": {
      if (w.t.length !== w.v.length || w.t.length < 1) out.push({ level: "error", key: "pwlLength" });
      else if (!fin(...w.t, ...w.v)) out.push({ level: "error", key: "nan" });
      else if (w.t.some((x, i) => i > 0 && x < w.t[i - 1])) out.push({ level: "error", key: "pwlOrder" });
      else if (w.t[0] < 0) out.push({ level: "error", key: "negative" });
      if (w.t.length > 2000) out.push({ level: "error", key: "pwlMax" });
      break;
    }
    case "sine":
      if (!fin(w.vo, w.va, w.freq, w.td, w.theta)) out.push({ level: "error", key: "nan" });
      else if (w.freq < 0 || w.td < 0) out.push({ level: "error", key: "negative" });
      break;
  }
  return out;
}

/** Validate an arbitrary JSON value as a Wave (import). */
export function parseWave(v: unknown): Wave | null {
  if (!v || typeof v !== "object") return null;
  const o = v as Record<string, unknown>;
  const num = (k: string) => (typeof o[k] === "number" && Number.isFinite(o[k] as number) ? (o[k] as number) : null);
  switch (o.kind) {
    case "dc": {
      const value = num("value");
      return value == null ? null : { kind: "dc", value };
    }
    case "pulse": {
      const ks = ["v1", "v2", "td", "tr", "tf", "pw", "per", "ncycles"] as const;
      const vals = ks.map(num);
      if (vals.some((x) => x == null)) return null;
      const [v1, v2, td, tr, tf, pw, per, ncycles] = vals as number[];
      return { kind: "pulse", v1, v2, td, tr, tf, pw, per, ncycles: Math.max(0, Math.round(ncycles)) };
    }
    case "pwl": {
      const t = Array.isArray(o.t) ? o.t : null;
      const vv = Array.isArray(o.v) ? o.v : null;
      if (!t || !vv || t.length !== vv.length || !t.length || t.length > 2000) return null;
      if (![...t, ...vv].every((x) => typeof x === "number" && Number.isFinite(x))) return null;
      return { kind: "pwl", t: t as number[], v: vv as number[] };
    }
    case "sine": {
      const ks = ["vo", "va", "freq", "td", "theta"] as const;
      const vals = ks.map(num);
      if (vals.some((x) => x == null)) return null;
      const [vo, va, freq, td, theta] = vals as number[];
      return { kind: "sine", vo, va, freq, td, theta };
    }
  }
  return null;
}
