// Offline demo backend. Deterministic, plausible-looking fixtures that follow the result shapes of
// docs/WEB_CONTRACT.md §2/§4 (folds at 3.70 / 2.60 V for the paper device at V_G = −2 V). These are
// NOT model results — the UI shows a "demo data" banner whenever this backend is active.
import type { Backend } from "./client";
import type {
  Arr, BranchesResult, ChargeBalanceResult, CircuitResult, CircuitRun, Components, Curve, DeviceBlock, HazardResult,
  JobStatus, Kind, Signal, Stats, StochasticBlock, SweepBlock, SweepMCResult, ValidationResult, VgCurveResult,
  VgCurveStochasticResult, XY,
} from "./types";
import { BUILTIN_META, RESPONSIVITY_PA_PER_MW } from "../state/presets";

// ---------------------------------------------------------------- helpers
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rand: () => number) {
  const u = Math.max(rand(), 1e-12);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const linspace = (a: number, b: number, n: number) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / Math.max(1, n - 1));
const smooth = (s: number) => s * s * (3 - 2 * s);
const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));

export function statsOf(values: Arr): Stats {
  const x = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const cens = values.length - x.length;
  if (!x.length) return { n: 0, mean: null, sd: null, median: null, p05: null, p95: null, min: null, max: null, censored: cens, lag1: null };
  const n = x.length;
  const mean = x.reduce((a, b) => a + b, 0) / n;
  const sd = n > 1 ? Math.sqrt(x.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : null;
  const s = [...x].sort((a, b) => a - b);
  const q = (p: number) => {
    const h = (n - 1) * p;
    const lo = Math.floor(h);
    return s[lo] + (s[Math.min(n - 1, lo + 1)] - s[lo]) * (h - lo);
  };
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    den += (x[i] - mean) ** 2;
    if (i > 0) num += (x[i] - mean) * (x[i - 1] - mean);
  }
  return { n, mean, sd, median: q(0.5), p05: q(0.05), p95: q(0.95), min: s[0], max: s[n - 1], censored: cens, lag1: den > 0 ? num / den : null };
}

export function histOf(values: Arr, bins = 30, range?: [number, number]) {
  const x = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const lo = range ? range[0] : Math.min(...x);
  const hi = range ? range[1] : Math.max(...x);
  const w = (hi - lo || 1) / bins;
  const edges = Array.from({ length: bins + 1 }, (_, i) => lo + i * w);
  const counts = new Array(bins).fill(0);
  for (const v of x) counts[clamp(Math.floor((v - lo) / w), 0, bins - 1)]++;
  return { edges, counts };
}
export function cdfOf(values: Arr) {
  const s = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v)).sort((a, b) => a - b);
  return { v: s, p: s.map((_, i) => (i + 1) / s.length) };
}

// ---------------------------------------------------------------- device model (mock)
interface Dev { vg: number; iph: number; dG: number; dE: number; vdMax: number }
function devOf(device: Partial<DeviceBlock> | undefined, sweep?: Partial<SweepBlock>): Dev {
  const d = device ?? {};
  const light = d.light;
  const iph = !light ? 0 : light.mode === "power" ? (light.responsivity_pA_per_mW ?? RESPONSIVITY_PA_PER_MW) * (light.power_mW ?? 0) : light.iph_pA ?? 0;
  return {
    vg: d.vg ?? -2,
    iph,
    dG: d.state?.delta_phi_G0_V ?? 0,
    dE: d.state?.delta_phi_E0_V ?? 0,
    vdMax: sweep?.vd_max_V ?? 4,
  };
}
function mockFolds(dev: Dev) {
  const { vg, iph, dG, dE } = dev;
  const taper = Math.sqrt(clamp((vg + 3.9) / 0.5, 0, 1) * clamp((-0.815 - vg) / 0.35, 0, 1));
  const VLD = 2.5979 - 41 * dE - 0.0007 * iph + 0.02 * (vg + 2) ** 2;
  const W0 = 1.1058 + 0.8035 * (vg + 2) - 0.36 * Math.max(0, vg + 2) ** 2 - 0.215 * iph - 0.8 * dG;
  const W = W0 * taper;
  const latch = W > 0.02;
  return { latch, VLU: VLD + Math.max(W, 0), VLD };
}
const I_HRS = (vd: number, VLU: number, vg: number, iph: number) =>
  1e-13 * Math.exp(vd / 0.55) * (1 + 2 * (vd / VLU) ** 14) * Math.exp(-0.4 * (vg + 2)) + iph * 1e-12 * 3;
const I_LRS = (vd: number, VLD: number) => 1.1e-6 + Math.max(0, vd - VLD) * 1.25e-5 + 2e-7 * Math.max(0, vd - VLD) ** 2;

function comps(vd: number[], id: number[], u: number[], vg: number, iph: number): Components {
  const n = vd.length;
  const m = (f: (i: number) => number): Arr => Array.from({ length: n }, (_, i) => f(i));
  return {
    channel: m((i) => id[i] * 0.93),
    seed: m((i) => id[i] * 0.05),
    ii_total: m((i) => id[i] * 0.02 * Math.exp(Math.min(6, (vd[i] - 3) / 0.45))),
    btbt_junction: m((i) => 2e-16 * Math.exp(vd[i] / 0.52)),
    gidl: m((i) => 6e-17 * Math.exp(vd[i] / 0.47) * Math.exp(-1.2 * (vg + 2))),
    photo: m(() => iph * 1e-12),
    loss_bulk_srh: m((i) => 3e-16 * Math.exp(u[i] / 0.055)),
    loss_diffusion: m((i) => 8e-17 * Math.exp(u[i] / 0.04)),
    loss_junction_srh: m((i) => 1.5e-16 * Math.exp(u[i] / 0.07)),
    net_F: m(() => 0),
    hole_drop_V: m((i) => 0.02 * u[i]),
    injection: m((i) => 1e-4 * Math.exp(u[i] / 0.05)),
    r_access_ohm: m(() => 1),
  };
}
function curve(vd: number[], id: number[], u: number[], vg: number, iph: number): Curve {
  return { vd, id, u, r: vd.map((v, i) => v - u[i]), comp: comps(vd, id, u, vg, iph) };
}

export function mockBranches(payload: { device?: DeviceBlock; sweep?: SweepBlock }): BranchesResult {
  const dev = devOf(payload.device, payload.sweep);
  const { latch, VLU, VLD } = mockFolds(dev);
  const vdMax = dev.vdMax;
  const nH = 140;
  const vH = linspace(0, VLU, nH);
  const iH = vH.map((v) => I_HRS(v, VLU, dev.vg, dev.iph));
  const uH = vH.map((v) => 0.1 + 0.42 * (v / VLU) ** 2);
  const ILU = iH[nH - 1];
  const ILD = I_LRS(VLD, VLD);
  const s = linspace(0, 1, 80);
  const vU = s.map((x) => VLU - (VLU - VLD) * smooth(x));
  const iU = s.map((x) => Math.exp(Math.log(ILU) + x * (Math.log(ILD) - Math.log(ILU))));
  const uU = s.map((x) => 0.52 + 0.24 * x);
  const vL = linspace(VLD, vdMax + 1, 120);
  const iL = vL.map((v) => I_LRS(v, VLD));
  const uL = vL.map((v) => 0.76 + 0.03 * (v - VLD));
  const up: XY = { vd: [], id: [] };
  const down: XY = { vd: [], id: [] };
  for (const v of linspace(0, vdMax, 201)) {
    up.vd.push(v);
    up.id.push(latch && v >= VLU ? I_LRS(v, VLD) : I_HRS(Math.min(v, VLU), VLU, dev.vg, dev.iph));
  }
  for (const v of linspace(vdMax, 0, 201)) {
    down.vd.push(v);
    down.id.push(latch && v >= VLD && vdMax >= VLU ? I_LRS(v, VLD) : I_HRS(Math.min(v, VLU), VLU, dev.vg, dev.iph));
  }
  const HRS = curve(vH, iH, uH, dev.vg, dev.iph);
  const unstable = curve(vU, iU, uU, dev.vg, dev.iph);
  const LRS = curve(vL, iL, uL, dev.vg, dev.iph);
  const full = curve([...vH, ...vU, ...vL], [...iH, ...iU, ...iL], [...uH, ...uU, ...uL], dev.vg, dev.iph);
  return {
    latch,
    HRS, unstable, LRS, full,
    folds: latch
      ? { V_LU: VLU, V_LD: VLD, I_LU: ILU, I_LD: ILD, u_LU: 0.52, u_LD: 0.76, window_V: VLU - VLD }
      : { V_LU: null, V_LD: null, I_LU: null, I_LD: null, u_LU: null, u_LD: null, window_V: null },
    double_sweep: { up, down },
    iph_A: dev.iph * 1e-12,
    p: [],
    runtime_s: 0.21,
    warnings: latch ? [] : ["no latch (single-valued I–V) at this bias"],
  };
}

export function mockChargeBalance(payload: { device?: DeviceBlock; vd?: number }): ChargeBalanceResult {
  const dev = devOf(payload.device);
  const { VLU, VLD } = mockFolds(dev);
  const vd = payload.vd ?? (VLU + VLD) / 2;
  const fr = (vd - VLD) / Math.max(VLU - VLD, 1e-3);
  const f = clamp(fr, 0, 1);
  const off = fr > 1 ? (fr - 1) * 30 : fr < 0 ? fr * 30 : 0;
  const u1 = 0.4 + 0.08 * f;
  const u2 = 0.78 - 0.3 * f;
  const u3 = 0.8 + 0.02 * f;
  const u = linspace(0.05, 0.95, 241);
  const lnGL = u.map((x) => 6 * Math.tanh((-900 * (x - u1) * (x - u2) * (x - u3) + off) / 6));
  const L = u.map((x) => 1e-16 * Math.exp(x / 0.045));
  const G = L.map((l, i) => l * Math.exp(lnGL[i]));
  const du = u[1] - u[0];
  const pot: number[] = [];
  let acc = 0;
  for (let i = 0; i < u.length; i++) {
    acc -= lnGL[i] * du * 60;
    pot.push(acc);
  }
  const roots: ChargeBalanceResult["roots"] = [];
  for (let i = 1; i < u.length; i++) {
    if (Math.sign(lnGL[i]) !== Math.sign(lnGL[i - 1])) {
      const x = u[i - 1] + (du * lnGL[i - 1]) / (lnGL[i - 1] - lnGL[i]);
      roots.push({ u: x, Q_C: 2.1e-15 * x, kind: lnGL[i - 1] > 0 ? "stable" : "unstable", id: 1e-13 * Math.exp(x / 0.06) });
    }
  }
  const ref = roots.find((r) => r.kind === "stable");
  const i0 = ref ? u.findIndex((x) => x >= ref.u) : 0;
  const p0 = pot[Math.max(0, i0)];
  return {
    vd, u, r: u.map((x) => vd - x), id: u.map((x) => 1e-13 * Math.exp(x / 0.06)), Q_C: u.map((x) => 2.1e-15 * x),
    generation_A: G, loss_A: L, unit_A: G.map((g) => g * 0.4), ii_A: G.map((g) => g * 0.6), F_A: G.map((g, i) => g - L[i]),
    potential: pot.map((p) => p - p0), roots, runtime_s: 0.05, warnings: [],
  };
}

export function mockVgCurve(payload: { device?: DeviceBlock; vg_min?: number; vg_max?: number; n?: number }): VgCurveResult {
  const vgs = linspace(payload.vg_min ?? -4.5, payload.vg_max ?? -0.5, payload.n ?? 41);
  const base = devOf(payload.device);
  const rows = vgs.map((vg) => mockFolds({ ...base, vg }));
  const inWin = vgs.filter((_, i) => rows[i].latch);
  return {
    vg: vgs,
    V_LU: rows.map((r) => (r.latch ? r.VLU : null)),
    V_LD: rows.map((r) => (r.latch ? r.VLD : null)),
    I_LU: rows.map((r) => (r.latch ? I_HRS(r.VLU, r.VLU, -2, base.iph) : null)),
    latch: rows.map((r) => r.latch),
    window: { vg_low: inWin.length ? Math.min(...inWin) : null, vg_high: inWin.length ? Math.max(...inWin) : null },
    runtime_s: 0.9,
    warnings: [],
  };
}

export function mockHazard(payload: { device?: DeviceBlock; sweep?: SweepBlock }): HazardResult {
  const dev = devOf(payload.device, payload.sweep);
  const { VLU, VLD } = mockFolds(dev);
  const rate = payload.sweep?.rate_V_per_s ?? 0.4;
  const w = 0.008;
  const A = (rate / w) * Math.exp(0.06 / w) * 0.69;
  const voltage = linspace(VLU - 0.2, VLU, 121);
  const hazard = voltage.map((v) => A * Math.exp((v - VLU) / w));
  const survival = voltage.map((v) => Math.exp(-(A * w * Math.exp((v - VLU) / w)) / rate));
  const prob = linspace(0.01, 0.99, 99);
  const qv = prob.map((p) => VLU + w * Math.log((-Math.log(1 - p) * rate) / (A * w)));
  return {
    fold_V: VLU, VLD_fold_V: VLD, voltage, hazard, survival, quantiles: { prob, v: qv }, stats: statsOf(qv),
    rate_V_per_s: rate, runtime_s: 3.2, warnings: [],
  };
}

export function mockSweepMC(payload: { device?: DeviceBlock; sweep?: SweepBlock; stochastic?: StochasticBlock }): SweepMCResult {
  const dev = devOf(payload.device, payload.sweep);
  const st = payload.stochastic;
  const n = st?.n_cycles ?? 100;
  const rand = mulberry32(st?.seed ?? 1);
  const { VLU, VLD } = mockFolds(dev);
  const ls = st?.local_state;
  const sig = ls && ls.mode !== "none" ? (ls.action === "gidl" || ls.action === "junction" ? ls.sigma : ls.sigma * 0.15) : 0;
  const rho = ls?.mode === "evolving" ? 0.45 : 0.05;
  const state: number[] = [];
  let s0 = gauss(rand);
  for (let i = 0; i < n; i++) {
    s0 = rho * s0 + Math.sqrt(1 - rho * rho) * gauss(rand);
    state.push(sig * s0);
  }
  const noise = st?.carrier_noise === false ? 0 : 0.0078;
  const vlu: Arr = state.map((d) => {
    const v = VLU - 0.06 - 0.8 * d + noise * gauss(rand);
    return v > dev.vdMax ? null : v;
  });
  const vld: Arr = vlu.map((v) => (v == null ? null : VLD + 0.1 + 0.0195 * gauss(rand)));
  const range = (a: Arr): [number, number] => {
    const x = a.filter((v): v is number => v != null);
    return [Math.min(...x), Math.max(...x)];
  };
  const nTr = Math.min(n, st?.n_traces ?? 12);
  const traces = Array.from({ length: nTr }, (_, k) => {
    const lu = vlu[k];
    const ld = vld[k];
    const up: XY = { vd: [], id: [] };
    const down: XY = { vd: [], id: [] };
    for (const v of linspace(0, dev.vdMax, 161)) {
      up.vd.push(v);
      up.id.push((lu != null && v >= lu ? I_LRS(v, VLD) : I_HRS(Math.min(v, VLU), VLU, dev.vg, dev.iph)) * (1 + 0.08 * gauss(rand)));
    }
    for (const v of linspace(dev.vdMax, 0, 161)) {
      down.vd.push(v);
      down.id.push((lu != null && ld != null && v >= ld ? I_LRS(v, VLD) : I_HRS(Math.min(v, VLU), VLU, dev.vg, dev.iph)) * (1 + 0.08 * gauss(rand)));
    }
    return { cycle: k, V_LU: lu, V_LD: ld, up, down };
  });
  const br = mockBranches({ device: payload.device, sweep: payload.sweep });
  const isPaper = Math.abs(dev.vg + 2) < 1e-9 && dev.iph === 0;
  let measured: SweepMCResult["measured"] = null;
  if (isPaper || payload.device?.preset === "photo") {
    const r2 = mulberry32(7);
    const mLU = Array.from({ length: isPaper ? 100 : 400 }, () => (isPaper ? 3.63 : VLU - 0.05) + (isPaper ? 0.123 : 0.173) * gauss(r2));
    const mLD = isPaper ? Array.from({ length: 100 }, () => 2.7 + 0.0195 * gauss(r2)) : null;
    measured = {
      label: isPaper ? "Paper device (demo)" : "Photo device (demo)",
      V_LU: mLU,
      V_LD: mLD,
      stats: { LU: statsOf(mLU), LD: mLD ? statsOf(mLD) : null },
    };
  }
  const deltas = linspace(-3 * (sig || 0.05), 3 * (sig || 0.05), 25);
  return {
    engine: isPaper && ls?.action === "gidl" ? "calibrated_lookup" : "general",
    V_LU: vlu, V_LD: vld,
    stats: { LU: statsOf(vlu), LD: statsOf(vld) },
    hist: { LU: histOf(vlu, 30, range(vlu)), LD: histOf(vld, 30, range(vld)) },
    cdf: { LU: cdfOf(vlu), LD: cdfOf(vld) },
    traces,
    cycle_state: state,
    fold_table: { delta: deltas, V_LU: deltas.map((d) => VLU - 0.8 * d), V_LD: deltas.map(() => VLD) },
    centre: {
      V_LU: VLU, V_LD: VLD,
      HRS: { vd: br.HRS.vd, id: br.HRS.id },
      LRS: { vd: br.LRS.vd, id: br.LRS.id },
    },
    measured,
    runtime_s: 1.4,
    warnings: [],
  };
}

export function mockVgStoch(payload: { device?: DeviceBlock; stochastic?: StochasticBlock; vg_min?: number; vg_max?: number; n?: number }): VgCurveStochasticResult {
  const vgs = linspace(payload.vg_min ?? -2.6, payload.vg_max ?? -0.9, payload.n ?? 9);
  const base = devOf(payload.device);
  const sig = payload.stochastic?.local_state?.sigma ?? 0.1534;
  const rows = vgs.map((vg) => mockFolds({ ...base, vg }));
  const stateSd = vgs.map((vg) => 1000 * 0.8 * sig * (0.75 + 0.3 * Math.exp(-(((vg + 1.25) / 0.6) ** 2))));
  const noiseSd = vgs.map(() => 7.8);
  const measured =
    payload.device?.preset === "photo"
      ? [
          { vg: -1.8, power_mW: 0, mean_V: 3.806, sd_mV: 173.2 },
          { vg: -1.1, power_mW: 0, mean_V: 3.408, sd_mV: 53.9 },
        ]
      : [];
  return {
    vg: vgs,
    mean_VLU: rows.map((r) => (r.latch ? r.VLU - 0.06 : null)),
    sd_VLU_mV: stateSd.map((s, i) => Math.hypot(s, noiseSd[i])),
    state_sd_mV: stateSd,
    noise_sd_mV: noiseSd,
    fold_centre_V: rows.map((r) => (r.latch ? r.VLU : null)),
    VLD_fold_V: rows.map((r) => (r.latch ? r.VLD : null)),
    no_latch_weight: rows.map((r) => (r.latch ? 0 : 1)),
    measured,
    runtime_s: 24,
    warnings: [],
  };
}

export function mockValidation(payload: { level?: string }): ValidationResult {
  const full = payload.level === "full";
  const c = (id: string, ko: string, en: string, expected: string, computed: string, tolerance: string, pass: boolean | null, seconds: number, note?: string) => ({
    id, label: { ko, en }, expected, computed, tolerance, pass, seconds, note,
  });
  const checks = [
    c("folds_paper_m2", "논문 모델 V_G = −2 V 암조건 fold", "Paper model folds, V_G = −2 V dark", "3.7037 / 2.5979 V", "3.7037 / 2.5979 V", "±1 mV", true, 0.4),
    c("folds_paper_m18", "논문 모델 V_G = −1.8 V fold", "Paper model folds, V_G = −1.8 V", "3.8644 / 2.5979 V", "3.8644 / 2.5979 V", "±1 mV", true, 0.4),
    c("folds_photo", "광조사 모델 I_PH = 2.63 pA fold", "Photo model folds, I_PH = 2.63 pA", "3.2913 / 2.596 V", "3.2913 / 2.5959 V", "±2 mV", true, 0.5),
    c("ext_zero", "확장 항 = 0 → 논문 모델과 동일", "Extensions = 0 → identical to paper model", "Δ < 1e-12 V", "0", "1e-12 V", true, 0.3),
    c("fpt_node", "FPT 노드 (중심 상태, 0.4 V/s)", "FPT node (centre states, 0.4 V/s)", "3.644 V, SD ≈ 8 mV", full ? "3.6442 V, 6.8 mV" : "—", "±5 mV / ±3 mV", full ? true : null, full ? 4.2 : 0, full ? undefined : "full only"),
    c("mc_100", "동적 MC 100 스윕", "Dynamic MC, 100 sweeps", "3.63 V / 120 mV; 2.70 V / 20 mV", "3.629 V / 121 mV; 2.702 V / 19.6 mV", "±20 mV / ±25 %", true, 1.1),
    c("light_conv", "광 변환 I_PH = R·P", "Light conversion I_PH = R·P", "0.86 / 1.91 / 2.63 pA", "0.8625 / 1.9125 / 2.6325 pA", "±0.01 pA", true, 0.0),
  ];
  return { checks, runtime_s: checks.reduce((a, b) => a + b.seconds, 0), warnings: ["demo data — backend offline"] };
}

// ---------------------------------------------------------------- circuit (mock)
function sig(key: string, ko: string, en: string, unit: string, axis: Signal["axis"], values: Arr): Signal {
  return { key, label: { ko, en }, unit, axis, values };
}
export function mockCircuit(payload: {
  bench?: string; mode?: string; device?: DeviceBlock; bench_params?: Record<string, unknown>; stochastic?: { n_runs?: number; seed?: number };
}): CircuitResult {
  const bench = payload.bench ?? "load_line";
  const mode = payload.mode ?? "deterministic";
  const dev = devOf(payload.device);
  const { VLU, VLD } = mockFolds(dev);
  const rand = mulberry32(payload.stochastic?.seed ?? 11);
  const nRuns = mode === "stochastic" ? Math.min(8, payload.stochastic?.n_runs ?? 8) : 1;
  const bp = payload.bench_params ?? {};
  const runs: CircuitRun[] = [];
  const events: CircuitResult["events"] = [];
  const nT = 600;
  let traj: XY | undefined;
  const vluRuns: number[] = [];
  for (let k = 0; k < nRuns; k++) {
    const jitter = mode === "stochastic" ? 0.12 * gauss(rand) : 0;
    const vlu = VLU - (mode === "stochastic" ? 0.06 : 0) + jitter;
    const vld = VLD + (mode === "stochastic" ? 0.1 + 0.02 * gauss(rand) : 0);
    vluRuns.push(vlu);
    if (bench === "load_line" || bench === "coupled") {
      const vmax = Number(bp.v_max_V ?? 4.5);
      const rate = Number(bp.rate_V_per_s ?? 0.4);
      const T = (2 * vmax) / rate;
      const t = linspace(0, T, nT);
      const vsrc = t.map((x) => (x < T / 2 ? rate * x : vmax - rate * (x - T / 2)));
      let on = false;
      const vd: number[] = [];
      const id: number[] = [];
      const qb: number[] = [];
      const rs = Number(bp.R_s_ohm ?? bp.R_s1_ohm ?? 1e3);
      t.forEach((x, i) => {
        const vs = vsrc[i];
        if (!on && vs >= vlu && x < T / 2) {
          on = true;
          events.push({ run: k, kind: "latch_up", t: x, v_src: vs, v_d: vs });
        }
        if (on && x > T / 2 && vs <= vld + 0.05) {
          on = false;
          events.push({ run: k, kind: "latch_down", t: x, v_src: vs, v_d: vs });
        }
        const i_d = on ? I_LRS(vs, VLD) : I_HRS(Math.min(vs, VLU), VLU, dev.vg, dev.iph);
        id.push(i_d);
        vd.push(vs - rs * i_d);
        qb.push((on ? 3.2e-15 : 1.2e-15) + 1e-16 * vs);
      });
      const signals = [
        sig("v_src", "전원 전압", "Source voltage", "V", "voltage", vsrc),
        sig("v_d", "드레인 전압", "Drain voltage", "V", "voltage", vd),
        sig("i_d", "드레인 전류", "Drain current", "A", "current", id),
        sig("q_b", "Body 전하", "Body charge", "C", "charge", qb),
      ];
      if (bench === "coupled") {
        signals.push(sig("v_d2", "드레인 전압 (소자 2)", "Drain voltage (device 2)", "V", "voltage", vd.map((v, i) => v * (0.98 + 0.01 * Math.sin(i / 40)))));
        signals.push(sig("i_d2", "드레인 전류 (소자 2)", "Drain current (device 2)", "A", "current", id.map((v) => v * 0.9)));
      }
      runs.push({ run: k, t, signals });
      if (k === 0) traj = { vd, id };
    } else if (bench === "pulse") {
      const amp = Number(bp.v_amp_V ?? VLU + 0.1);
      const width = Number(bp.width_s ?? 200e-6);
      const period = Number(bp.period_s ?? 1e-3);
      const np = Math.min(20, Number(bp.n_pulses ?? 10));
      const T = period * np;
      const t = linspace(0, T, nT);
      let on = false;
      const vsrc = t.map((x) => ((x % period) < width ? amp : 0));
      const vd: number[] = [];
      const id: number[] = [];
      const qb: number[] = [];
      let q = 1e-15;
      t.forEach((x, i) => {
        const high = vsrc[i] > 0;
        q = high ? q + 3e-17 * (1 + 0.3 * gauss(rand)) : Math.max(1e-15, q * 0.985);
        if (!on && high && q > 2.2e-15) {
          on = true;
          events.push({ run: k, kind: "latch_up", t: x, v_src: vsrc[i] });
        }
        if (on && !high) {
          on = false;
          events.push({ run: k, kind: "latch_down", t: x, v_src: 0 });
        }
        const i_d = high ? (on ? I_LRS(amp, VLD) : I_HRS(Math.min(amp, VLU), VLU, dev.vg, dev.iph)) : 1e-14;
        id.push(i_d);
        vd.push(vsrc[i] - 1e4 * i_d);
        qb.push(q);
      });
      runs.push({
        run: k, t,
        signals: [
          sig("v_src", "펄스 전압", "Pulse voltage", "V", "voltage", vsrc),
          sig("v_d", "드레인 전압", "Drain voltage", "V", "voltage", vd),
          sig("i_d", "드레인 전류", "Drain current", "A", "current", id),
          sig("q_b", "Body 전하", "Body charge", "C", "charge", qb),
        ],
      });
      if (k === 0) traj = { vd, id };
    } else {
      // pbit
      const T = Number(bp.clock_period_s ?? 1e-3) * Number(bp.n_clocks ?? 50);
      const t = linspace(0, T, nT);
      let state = 0;
      const bits: number[] = [];
      const vd: number[] = [];
      const id: number[] = [];
      const qb: number[] = [];
      t.forEach((x, i) => {
        if (rand() < 0.03) {
          state = 1 - state;
          events.push({ run: k, kind: "bit", t: x, value: state });
        }
        bits.push(state);
        const i_d = state ? 8e-6 : 2e-11;
        id.push(i_d * (1 + 0.05 * gauss(rand)));
        vd.push(state ? 2.9 : 3.5);
        qb.push(state ? 3e-15 : 1.3e-15);
        void i;
      });
      runs.push({
        run: k, t,
        signals: [
          sig("v_d", "드레인 전압", "Drain voltage", "V", "voltage", vd),
          sig("i_d", "드레인 전류", "Drain current", "A", "current", id),
          sig("q_b", "Body 전하", "Body charge", "C", "charge", qb),
          sig("bit", "비교기 출력", "Comparator bit", "1", "logic", bits),
        ],
      });
      if (k === 0) traj = { vd, id };
    }
  }
  const cell = (k: number, d: string, g: string) => [
    { kind: "STL", name: `X${k}`, nodes: [d, g, "0"], value: "paper model" },
    { kind: "V", name: `VG${k}`, nodes: [g, "0"], value: `${dev.vg} V (DC)` },
  ];
  const schematic: CircuitResult["schematic"] =
    bench === "coupled"
      ? {
          nodes: ["src", "d1", "d2", "g1", "g2", "0"],
          elements: [
            { kind: "V", name: "Vsrc", nodes: ["src", "0"], value: "triangle 0→4.5 V, 0.4 V/s" },
            { kind: "R", name: "Rs1", nodes: ["src", "d1"], value: "100 kΩ" },
            { kind: "R", name: "Rs2", nodes: ["src", "d2"], value: "100 kΩ" },
            { kind: "R", name: "Rc", nodes: ["d1", "d2"], value: "1 MΩ" },
            { kind: "C", name: "Cd1", nodes: ["d1", "0"], value: "2 fF" },
            { kind: "C", name: "Cd2", nodes: ["d2", "0"], value: "2 fF" },
            ...cell(1, "d1", "g1"),
            ...cell(2, "d2", "g2"),
          ],
        }
      : bench === "pbit"
        ? {
            nodes: ["clk", "d", "g", "0"],
            elements: [
              { kind: "V", name: "Vclk", nodes: ["clk", "0"], value: "clock 0/3.68 V, 0.001 s" },
              { kind: "R", name: "RL", nodes: ["clk", "d"], value: "100 kΩ" },
              { kind: "C", name: "Cd", nodes: ["d", "0"], value: "2 fF" },
              ...cell(1, "d", "g"),
              { kind: "CMP", name: "CMP", nodes: ["d"], value: "bit = [v_D < 3.67 V]" },
            ],
          }
        : {
            nodes: ["src", "d", "g", "0"],
            elements: [
              { kind: "V", name: "Vsrc", nodes: ["src", "0"], value: bench === "pulse" ? "pulses 0→3.8 V" : "triangle 0→4.5 V, 0.4 V/s" },
              { kind: "R", name: "Rs", nodes: ["src", "d"], value: "1 kΩ" },
              { kind: "C", name: "Cd", nodes: ["d", "0"], value: "2 fF" },
              ...cell(1, "d", "g"),
            ],
          };
  const st = statsOf(vluRuns);
  const summary: CircuitResult["summary"] =
    bench === "pbit"
      ? [
          { key: "p_one", label: { ko: "P(1)", en: "P(1)" }, value: 0.48, unit: "1", spread: mode === "stochastic" ? 0.04 : null },
          { key: "flip_rate", label: { ko: "뒤집힘 빈도", en: "Flip rate" }, value: 92, unit: "1/s", spread: mode === "stochastic" ? 11 : null },
        ]
      : [
          { key: "V_LU", label: { ko: "래치업 전압", en: "Latch-up voltage" }, value: st.mean, unit: "V", spread: mode === "stochastic" ? st.sd : null },
          { key: "V_LD", label: { ko: "래치다운 전압", en: "Latch-down voltage" }, value: VLD + (mode === "stochastic" ? 0.1 : 0), unit: "V", spread: mode === "stochastic" ? 0.02 : null },
          { key: "I_on", label: { ko: "LRS 전류", en: "LRS current" }, value: I_LRS(4, VLD), unit: "A" },
        ];
  const res: CircuitResult = {
    bench, mode, runs, events, summary, trajectory: traj, schematic,
    solver_stats: { steps: 1843, rejected: 37, newton_iters: 5210, runtime_s: 0.8 },
    runtime_s: 0.9,
    warnings: ["demo data — backend offline"],
  };
  if (mode === "stochastic") {
    res.distributions = [{ key: "V_LU", label: { ko: "V_LU (run별)", en: "V_LU per run" }, unit: "V", values: vluRuns }];
    if (bench === "pulse") {
      const amps = linspace(3.4, 4.4, 11);
      res.sweeps = [{
        key: "p_sw", label: { ko: "스위칭 확률 vs 진폭", en: "Switching probability vs amplitude" },
        x: amps, x_label: "amplitude", x_unit: "V", y: amps.map((a) => 1 / (1 + Math.exp(-(a - VLU + 0.05) / 0.08))), y_label: "P_sw", y_unit: "1",
        y_err: amps.map(() => 0.05),
      }];
    }
  }
  return res;
}

// ---------------------------------------------------------------- measured + design map (mock, backend-shaped)
export function mockMeasuredRaw(): unknown {
  const r = mulberry32(3);
  const vd = linspace(0, 4, 201);
  const up = vd.map((v) => (v < 3.63 ? I_HRS(v, 3.7037, -2, 0) : I_LRS(v, 2.5979)));
  const down = vd.map((v) => (v > 2.7 ? I_LRS(v, 2.5979) : I_HRS(v, 3.7037, -2, 0)));
  const band = (a: number[], f: number) => a.map((x) => x * f);
  const stats = (mean: number, sd: number) => ({ n: 400, mean, sd, median: mean, p05: mean - 1.64 * sd, p95: mean + 1.64 * sd, min: mean - 3 * sd, max: mean + 3 * sd, censored: 0, lag1: 0.1 });
  const table: [number, number, number, number][] = [
    [-1.8, 0, 3.806, 0.1732], [-1.8, 1.15, 3.5004, 0.1774], [-1.8, 2.55, 3.3356, 0.1186], [-1.8, 3.51, 3.0725, 0.1341],
    [-1.1, 0, 3.408, 0.0539], [-1.1, 1.15, 3.24, 0.061], [-1.1, 2.55, 3.1, 0.072], [-1.1, 3.51, 2.98, 0.08],
  ];
  const powers = [0, 0.5, 1.15, 2.0, 2.55, 3.51];
  return {
    photo: {
      conditions: table.map(([vg, p, m, s], k) => ({ index: k, label: `${vg}V ${p.toFixed(2)}mW`, vg, power_mW: p, stats: stats(m, s) })),
      V_LU: table.map(([, , m, s]) => Array.from({ length: 400 }, () => m + s * gauss(r))),
    },
    light_iv: {
      vd: linspace(0, 5, 101),
      power_mW: powers,
      id: powers.map((p) => linspace(0, 5, 101).map((v) => (v < 3.8 - 0.2 * p ? I_HRS(v, 3.8, -1.8, 0.75 * p) : I_LRS(v, 2.6)))),
    },
    paper_idvd: {
      up: { vd, median: up, p10: band(up, 0.6), p90: band(up, 1.6) },
      down: { vd, median: down, p10: band(down, 0.7), p90: band(down, 1.4) },
      V_LU: Array.from({ length: 100 }, () => 3.63 + 0.123 * gauss(r)),
      V_LD: Array.from({ length: 100 }, () => 2.7 + 0.0195 * gauss(r)),
    },
  };
}

export function mockDesignMapRaw(): unknown {
  const L = Array.from({ length: 41 }, (_, i) => 3 * Math.pow(100 / 3, i / 40));
  const D = linspace(0, 0.85, 31);
  const sphi = D.map((d) => L.map((l) => 1000 * (0.3 + d) * Math.pow(l / 3, -0.9) + 5));
  return {
    arrays: {
      length_nm: L,
      depth_fraction: D,
      sigma_phi_mV: sphi,
      sigma_VLU_mV: sphi.map((row) => row.map((s) => Math.hypot(0.8 * s * 0.5, 10))),
      latched_fraction: sphi.map((row) => row.map((s) => clamp(1.02 - s / 5000, 0.8, 1))),
      sigma_VLU_sweep5p2V_mV: sphi.map((row) => row.map((s) => Math.hypot(0.8 * s * 0.6, 11))),
      expected_trap_count: D.map(() => L.map((l) => 1e12 * (l * 1e-7) ** 2)),
      line_Nt: [1e11, 5e11, 1e12, 5e12],
      line_L0_device: [47.7, 21.3, 15.1, 6.7],
      line_L0_50: [111, 49.7, 35.1, 15.7],
    },
    scalars: { Nt_cm2: 1e12, device_sigma_phi_mV: 153.39, phi_50mV: 65.8 },
  };
}

// ---------------------------------------------------------------- mock backend (in-memory jobs)
const RUNTIME_MS: Partial<Record<Kind, number>> = {
  branches: 450, charge_balance: 250, vg_curve: 900, hazard: 1300, sweep_mc: 1500, vg_curve_stochastic: 2600,
  circuit: 1200, validation: 1500,
};
const MESSAGES: Partial<Record<Kind, string>> = {
  branches: "classify: tracing branch", hazard: "FPT nodes", sweep_mc: "sweeping cycles", vg_curve: "V_G points",
  vg_curve_stochastic: "V_G nodes", circuit: "transient", validation: "checks", charge_balance: "lattice",
};

function compute(kind: Kind, payload: unknown): unknown {
  const p = payload as never;
  switch (kind) {
    case "branches": return mockBranches(p);
    case "charge_balance": return mockChargeBalance(p);
    case "vg_curve": return mockVgCurve(p);
    case "hazard": return mockHazard(p);
    case "sweep_mc": return mockSweepMC(p);
    case "vg_curve_stochastic": return mockVgStoch(p);
    case "circuit": return mockCircuit(p);
    case "validation": return mockValidation(p);
  }
  throw new Error(`unknown kind ${kind as string}`);
}

interface MockJob { id: string; kind: Kind; start: number; dur: number; payload: unknown; cancelled: boolean }

export function createMockBackend(speed = 1): Backend {
  const jobs = new Map<string, MockJob>();
  let seq = 0;
  const status = (j: MockJob): JobStatus => {
    const el = (performance.now() - j.start) / 1000;
    const frac = Math.min(1, (el * 1000) / j.dur);
    if (j.cancelled) return { job_id: j.id, kind: j.kind, status: "cancelled", progress: frac, message: "cancelled", cached: false, elapsed_s: el };
    if (frac >= 1) {
      try {
        return { job_id: j.id, kind: j.kind, status: "done", progress: 1, message: "done", result: compute(j.kind, j.payload), cached: false, elapsed_s: el };
      } catch (e) {
        return { job_id: j.id, kind: j.kind, status: "error", progress: frac, message: "error", error: String((e as Error).message), cached: false, elapsed_s: el };
      }
    }
    return { job_id: j.id, kind: j.kind, status: "running", progress: frac, message: `${MESSAGES[j.kind] ?? "running"} (demo)`, cached: false, elapsed_s: el };
  };
  return {
    isMock: true,
    health: async () => ({ ok: true, version: "mock", workers: 0 }),
    meta: async () => BUILTIN_META,
    submit: async (kind, payload) => {
      const j: MockJob = { id: `mock-${++seq}`, kind, start: performance.now(), dur: (RUNTIME_MS[kind] ?? 600) * speed, payload, cancelled: false };
      jobs.set(j.id, j);
      return status(j);
    },
    job: async (id) => {
      const j = jobs.get(id);
      if (!j) throw new Error("unknown job");
      return status(j);
    },
    cancel: async (id) => {
      const j = jobs.get(id);
      if (j) j.cancelled = true;
    },
    measured: async () => mockMeasuredRaw(),
    designMap: async () => mockDesignMapRaw(),
  };
}
