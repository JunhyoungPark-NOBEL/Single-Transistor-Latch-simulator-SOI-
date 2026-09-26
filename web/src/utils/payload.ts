// Payload builders (docs/WEB_CONTRACT.md §1 / §4). Pure functions of the parameter root so they can be
// unit-tested and used as cache/staleness keys.
import type { DeviceBlock, Meta, Mode, PresetDef } from "../api/types";
import type { CircuitParams, ParamRoot } from "../params/schema";
import { BENCHES, DEFAULT_CIRCUIT_STOCH, DEFAULT_SOLVER } from "../params/benches";
import { clone } from "./object";
import { withModelDefaults } from "../params/model";
import { resolveBackGate, resolveGeometry } from "../params/geometry";

export interface VgRange {
  min: number;
  max: number;
  n: number;
}

/** Photocurrent in pA from the light block (I_PH = R · P in power mode) — mirrors params.iph_A(). */
export function iphPA(device: Pick<DeviceBlock, "light">): number {
  const l = device.light;
  return l.mode === "power" ? l.responsivity_pA_per_mW * l.power_mW : l.iph_pA;
}

/** Optical power (mW) that corresponds to the current light block. */
export function powerMW(device: Pick<DeviceBlock, "light">): number {
  const l = device.light;
  return l.mode === "power" ? l.power_mW : l.responsivity_pA_per_mW > 0 ? l.iph_pA / l.responsivity_pA_per_mW : 0;
}

/** Switch the light input mode keeping the photocurrent unchanged. */
export function switchLightMode(device: DeviceBlock, mode: "iph" | "power"): DeviceBlock {
  if (device.light.mode === mode) return device;
  const d = clone(device);
  const R = d.light.responsivity_pA_per_mW;
  if (mode === "power") d.light.power_mW = R > 0 ? d.light.iph_pA / R : 0;
  else d.light.iph_pA = R * d.light.power_mW;
  d.light.mode = mode;
  return d;
}

const device = (p: ParamRoot): DeviceBlock => clone(p.device);

export const branchesPayload = (p: ParamRoot) => ({ device: device(p), sweep: clone(p.sweep) });
export const chargeBalancePayload = (p: ParamRoot, vd: number) => ({ device: device(p), vd: round(vd, 6) });
/**
 * V_G assumed in the device block of a V_G-curve request. The server sets `vg` at every grid point itself, so
 * the curve does not depend on the sidebar V_G: a fixed value keeps the cache/staleness key unchanged when only
 * V_G moves (no re-run, no "parameters changed" badge; only the "현재 V_G" marker moves). −2 V is the reference
 * preset's V_G, so its snapshot key stays the same.
 */
export const VG_CURVE_CANONICAL_VG = -2;

export const vgCurvePayload = (p: ParamRoot, r: VgRange) => {
  const d = device(p);
  d.vg = VG_CURVE_CANONICAL_VG;
  return { device: d, vg_min: r.min, vg_max: r.max, n: Math.round(r.n) };
};
export const hazardPayload = (p: ParamRoot) => ({ device: device(p), sweep: clone(p.sweep) });
export const sweepMcPayload = (p: ParamRoot) => ({ device: device(p), sweep: clone(p.sweep), stochastic: clone(p.stochastic) });
export const vgStochPayload = (p: ParamRoot, r: VgRange) => ({
  device: device(p), sweep: clone(p.sweep), stochastic: clone(p.stochastic), vg_min: r.min, vg_max: r.max, n: Math.round(r.n),
});

export function circuitPayload(p: ParamRoot, mode: Mode) {
  const c = p.circuit;
  const out: Record<string, unknown> = {
    bench: c.bench,
    mode,
    device: device(p),
    bench_params: benchParamsPayload(c.bench_params[c.bench]),
    solver: clone(c.solver),
    detect: clone(c.detect),
  };
  if (mode === "stochastic") out.stochastic = clone(c.stochastic);
  return out;
}

/**
 * Bench parameters for the request: `null` ("auto") keys are omitted so the server's BENCH_DEFAULTS apply
 * (None there = resolved from the device; a number = documented default, e.g. rise_s 10 µs). The server's
 * merge would otherwise copy an explicit null over its default.
 */
export function benchParamsPayload(bp: Record<string, unknown> | undefined): Record<string, unknown> {
  return Object.fromEntries(Object.entries(clone(bp ?? {})).filter(([, v]) => v !== null && v !== undefined));
}

export const validationPayload = (level: "fast" | "full") => ({ level });

/** sweep_mc payload for one measured photo-device condition (photo preset, given V_G and power). */
export function photoConditionPayload(photo: PresetDef, vg: number, power_mW: number) {
  const d = clone(photo.device);
  d.preset = "photo";
  d.vg = vg;
  d.light = { ...d.light, mode: "power", power_mW };
  return { device: d, sweep: clone(photo.sweep), stochastic: clone(photo.stochastic) };
}

export function defaultCircuit(): CircuitParams {
  return {
    bench: "load_line",
    bench_params: Object.fromEntries(Object.values(BENCHES).map((b) => [b.id, { ...b.defaults }])) as CircuitParams["bench_params"],
    solver: clone(DEFAULT_SOLVER),
    stochastic: clone(DEFAULT_CIRCUIT_STOCH),
    detect: { i_threshold_A: 1e-8 },
  };
}

/** Parameter root for a preset (device/sweep/stochastic from the preset, circuit defaults). */
export function presetRoot(meta: Meta, preset: keyof Meta["presets"]): ParamRoot {
  const pr = meta.presets[preset] ?? meta.presets.paper;
  return { device: { ...withModelDefaults(clone(pr.device)), geometry: resolveGeometry(pr.device.geometry), vbg: resolveBackGate(pr.device.vbg) }, sweep: clone(pr.sweep), stochastic: clone(pr.stochastic), circuit: defaultCircuit() };
}

/** Midpoint between the folds (default V_D for the charge-balance panel). */
export function midFold(vlu: number | null | undefined, vld: number | null | undefined, fallback = 3.2): number {
  if (typeof vlu === "number" && typeof vld === "number") return round((vlu + vld) / 2, 3);
  if (typeof vlu === "number") return round(vlu - 0.3, 3);
  return fallback;
}

export function round(v: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(v * f) / f;
}

/** Detect which channel-seed option the extension block corresponds to. */
export function channelSeedOf(ext: DeviceBlock["ext"], options: Meta["channel_seed_options"]): string {
  const near = (a: number, b: number) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));
  const neutral = { gamma: 0, seed_ip_pA: 0, seed_S: 1 };
  for (const [id, over] of Object.entries(options)) {
    const target = { ...neutral, ...over } as Record<string, number>;
    if (near(ext.gamma, target.gamma) && near(ext.seed_ip_pA, target.seed_ip_pA) && (target.seed_ip_pA === 0 || near(ext.seed_S, target.seed_S))) return id;
  }
  return "custom";
}

/** Apply a channel-seed option: neutral values for the seed levers, then the option's overrides. */
export function applyChannelSeed(ext: DeviceBlock["ext"], id: string, options: Meta["channel_seed_options"]): DeviceBlock["ext"] {
  return { ...ext, gamma: 0, seed_ip_pA: 0, seed_S: 1, ...(options[id] ?? {}) } as DeviceBlock["ext"];
}
