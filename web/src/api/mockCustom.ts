// Demo backend for bench "custom" (docs/WEB_CONTRACT.md §6): a small MNA transient (backward Euler +
// Newton, dense LU) with R, C, V, I and a hysteretic STL stand-in (HRS/LRS currents of the mock device,
// latch-up above V_LU(V_GS, I_PH), latch-down below V_LD, body state relaxing with τ ≈ µs; stochastic
// runs add a per-run fold jitter and an escape hazard near V_LU; comparators switch on the previous step's input).
// NOT the model — used only when the
// backend is offline or does not know bench "custom" yet; the UI marks such results as demo data.
import type { ComparatorStats, CustomCircuitRequest, CustomCircuitResult, CustomElement, Envelope, Wave } from "./circuitCustom";
import type { Arr, CircuitRun, DeviceBlock, Signal, SummaryItem } from "./types";
import { waveAt, wavePoints } from "../schematic/waves";

const GROUND = new Set(["0", "gnd", "GND"]);

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r: () => number) {
  const u = Math.max(r(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

/** Mock folds (same fit as api/mock.ts): V_LD ≈ 2.60 V, V_LU ≈ 3.70 V at V_GS = −2 V, dark. */
function folds(vgs: number, iph: number, dG: number, dE: number) {
  const taper = Math.sqrt(Math.min(1, Math.max(0, (vgs + 3.9) / 0.5)) * Math.min(1, Math.max(0, (-0.815 - vgs) / 0.35)));
  const VLD = 2.5979 - 41 * dE - 0.0007 * iph + 0.02 * (vgs + 2) ** 2;
  const W = (1.1058 + 0.8035 * (vgs + 2) - 0.36 * Math.max(0, vgs + 2) ** 2 - 0.215 * iph - 0.8 * dG) * taper;
  return { VLU: VLD + Math.max(W, 0.02), VLD };
}
const iHRS = (v: number, VLU: number, vgs: number, iph: number) =>
  v <= 0 ? 1e-12 * v : 1e-13 * Math.exp(Math.min(v, 8) / 0.55) * (1 + 2 * Math.min(40, (v / VLU) ** 14)) * Math.exp(-0.4 * (vgs + 2)) + iph * 3e-12 + 1e-13 * v;
const iLRS = (v: number, VLD: number) => (v <= 0 ? 1e-9 * v : 1.1e-6 * Math.min(1, v / VLD) + Math.max(0, v - VLD) * 1.25e-5 + 2e-7 * Math.max(0, v - VLD) ** 2);

function devIph(d: DeviceBlock | undefined): number {
  if (!d) return 0;
  return d.light.mode === "power" ? d.light.power_mW * d.light.responsivity_pA_per_mW : d.light.iph_pA;
}

interface Cell {
  el: Extract<CustomElement, { type: "STL" }>;
  d: number;
  g: number;
  s: number;
  latched: boolean;
  frac: number;
  vluJ: number;
  vldJ: number;
  nUp: number;
  nDown: number;
  firstUp: number | null;
  firstUpVd: number | null;
}

function solve(A: Float64Array, b: Float64Array, n: number): Float64Array {
  // Gaussian elimination with partial pivoting (in place)
  for (let k = 0; k < n; k++) {
    let p = k;
    let best = Math.abs(A[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(A[i * n + k]);
      if (v > best) {
        best = v;
        p = i;
      }
    }
    if (best < 1e-300) throw new Error("singular MNA matrix — floating node or voltage-source loop");
    if (p !== k) {
      for (let j = 0; j < n; j++) {
        const tmp = A[k * n + j];
        A[k * n + j] = A[p * n + j];
        A[p * n + j] = tmp;
      }
      const tb = b[k];
      b[k] = b[p];
      b[p] = tb;
    }
    const piv = A[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const f = A[i * n + k] / piv;
      if (f === 0) continue;
      for (let j = k; j < n; j++) A[i * n + j] -= f * A[k * n + j];
      b[i] -= f * b[k];
    }
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= A[i * n + j] * x[j];
    x[i] = s / A[i * n + i];
  }
  return x;
}

function timeGrid(req: CustomCircuitRequest, maxPts = 1800): number[] {
  const T = req.tran.t_stop_s;
  const hMax = Math.min(req.tran.dt_max_s > 0 ? req.tran.dt_max_s : T / 1000, T / 600);
  const base = Math.min(maxPts * 0.6, Math.ceil(T / hMax));
  const pts = new Set<number>();
  for (let i = 0; i <= base; i++) pts.add((T * i) / base);
  const waves: Wave[] = [];
  for (const e of req.netlist.elements) {
    if (e.type === "V" || e.type === "I") waves.push(e.wave);
    if (e.type === "STL" && e.light_pA) waves.push(e.light_pA);
  }
  for (const w of waves) {
    if (w.kind === "sine" || w.kind === "dc") continue;
    const c = wavePoints(w, T, 400).t;
    for (const x of c) {
      if (pts.size > maxPts) break;
      pts.add(x);
      // resolve the edge: a few points just after each corner
      for (const f of [1e-3, 1e-2, 5e-2]) if (x + f * hMax < T) pts.add(x + f * hMax);
    }
  }
  return [...pts].filter((x) => x >= 0 && x <= T).sort((a, b) => a - b);
}

type CmpEl = Extract<CustomElement, { type: "CMP" }>;
const nodesOf = (e: CustomElement): string[] => Object.values(e.nodes);

interface RunOut {
  t: number[];
  sig: Map<string, number[]>;
  cells: Cell[];
  /** per comparator: output high at each recorded time */
  cmpHigh: boolean[][];
  events: CustomCircuitResult["events"];
  steps: number;
  newton: number;
}

function simulate(req: CustomCircuitRequest, grid: number[], run: number, rand: () => number, stochastic: boolean): RunOut {
  const els = req.netlist.elements;
  const nodeIdx = new Map<string, number>();
  const idx = (n: string) => {
    if (GROUND.has(n)) return -1;
    let k = nodeIdx.get(n);
    if (k === undefined) {
      k = nodeIdx.size;
      nodeIdx.set(n, k);
    }
    return k;
  };
  const two = els.filter((e) => ["R", "C", "V", "I"].includes(e.type)) as Extract<CustomElement, { type: "R" | "C" | "V" | "I" }>[];
  const cmps = els.filter((e) => e.type === "CMP") as CmpEl[];
  const cmpIdx = cmps.map((c) => ({ i: idx(c.nodes.in), o: idx(c.nodes.out) }));
  const nodesOf = two.map((e) => [idx(e.nodes[0]), idx(e.nodes[1])]);
  const cells: Cell[] = (els.filter((e) => e.type === "STL") as Cell["el"][]).map((el) => {
    const ls = req.stochastic?.local_state_override ? req.stochastic.local_state : el.local_state ?? req.stochastic?.local_state;
    const sigState = stochastic && ls && ls.mode !== "none" ? (["gidl", "junction"].includes(ls.action) ? 0.8 * ls.sigma : 0.05 * ls.sigma) : 0;
    return {
      el, d: idx(el.nodes.d), g: idx(el.nodes.g), s: idx(el.nodes.s), latched: false, frac: 0,
      vluJ: stochastic ? 0.035 * gauss(rand) + sigState * gauss(rand) - 0.05 : 0,
      vldJ: stochastic ? 0.015 * gauss(rand) + 0.08 : 0,
      nUp: 0, nDown: 0, firstUp: null, firstUpVd: null,
    };
  });
  const vsrc = two.map((e, i) => (e.type === "V" ? i : -1)).filter((i) => i >= 0);
  const N = nodeIdx.size;
  const M = vsrc.length;
  const n = N + M + cmps.length;
  if (N === 0) throw new Error("no node other than ground");
  const vIndex = new Map(vsrc.map((i, k) => [i, N + k]));
  let x: Float64Array = new Float64Array(n);
  let xPrev: Float64Array = new Float64Array(n);
  const V = (arr: Float64Array, k: number) => (k < 0 ? 0 : arr[k]);
  const out: RunOut = { t: [], sig: new Map(), cells, cmpHigh: cmps.map(() => []), events: [], steps: 0, newton: 0 };
  const cmpVal = (j: number, xs: Float64Array) => {
    const c = cmps[j];
    const hi = V(xs, cmpIdx[j].i) > c.v_ref;
    return { v: hi ? c.v_high ?? 1 : c.v_low ?? 0, hi };
  };
  const rec = (k: string, v: number) => {
    let a = out.sig.get(k);
    if (!a) out.sig.set(k, (a = []));
    a.push(v);
  };
  const tSave = req.tran.t_start_save_s;
  const nodeNames = [...nodeIdx.keys()];

  let cellPar: { vgs: number; iph: number; VLU: number; VLD: number }[] = [];
  const idOf = (c: Cell, k: number, vds: number) => (1 - c.frac) * iHRS(vds, cellPar[k].VLU, cellPar[k].vgs, cellPar[k].iph) + c.frac * iLRS(vds, cellPar[k].VLD);
  for (let step = 0; step < grid.length; step++) {
    const t = grid[step];
    const h = step === 0 ? 0 : t - grid[step - 1];
    // cell fold voltages at this step (gate/source from the previous solution; light waveform).
    // The operating point (step 0) is solved twice so V_GS comes from the circuit, not from zero.
    for (let pass = 0; pass < (step === 0 ? 2 : 1); pass++) {
    if (pass === 1) xPrev = x;
    cellPar = cells.map((c) => {
      const vgs = V(xPrev, c.g) - V(xPrev, c.s);
      const iph = c.el.light_pA ? waveAt(c.el.light_pA, t) : devIph(c.el.device);
      const f = folds(vgs, Math.max(0, iph), c.el.device?.state?.delta_phi_G0_V ?? 0, c.el.device?.state?.delta_phi_E0_V ?? 0);
      return { vgs, iph, VLU: f.VLU + c.vluJ, VLD: f.VLD + c.vldJ };
    });
    // Newton
    let it = 0;
    for (; it < 40; it++) {
      const A = new Float64Array(n * n);
      const b = new Float64Array(n);
      const add = (i: number, j: number, v: number) => {
        if (i >= 0 && j >= 0) A[i * n + j] += v;
      };
      for (let k = 0; k < N; k++) add(k, k, 1e-12); // gmin
      two.forEach((e, i) => {
        const [a, c] = nodesOf[i];
        if (e.type === "R") {
          const g = 1 / e.value;
          add(a, a, g); add(c, c, g); add(a, c, -g); add(c, a, -g);
        } else if (e.type === "C") {
          if (step === 0) return; // DC operating point: capacitors open
          const g = e.value / h;
          add(a, a, g); add(c, c, g); add(a, c, -g); add(c, a, -g);
          const ieq = g * (V(xPrev, a) - V(xPrev, c));
          if (a >= 0) b[a] += ieq;
          if (c >= 0) b[c] -= ieq;
        } else if (e.type === "V") {
          const j = vIndex.get(i)!;
          add(a, j, 1); add(c, j, -1); add(j, a, 1); add(j, c, -1);
          b[j] = waveAt(e.wave, t);
        } else if (e.type === "I") {
          const iv = waveAt(e.wave, t);
          if (a >= 0) b[a] -= iv;
          if (c >= 0) b[c] += iv;
        }
      });
      // comparators: behavioural sources driven by the previous step's input (explicit in the demo)
      cmps.forEach((_, j) => {
        const r = N + M + j;
        add(cmpIdx[j].o, r, 1);
        add(r, cmpIdx[j].o, 1);
        b[r] = cmpVal(j, xPrev).v;
      });
      cells.forEach((c, k) => {
        const vds = V(x, c.d) - V(x, c.s);
        const i0 = idOf(c, k, vds);
        const dv = 1e-6;
        const g = Math.max(1e-15, (idOf(c, k, vds + dv) - i0) / dv);
        const ieq = i0 - g * vds;
        add(c.d, c.d, g); add(c.s, c.s, g); add(c.d, c.s, -g); add(c.s, c.d, -g);
        if (c.d >= 0) b[c.d] -= ieq;
        if (c.s >= 0) b[c.s] += ieq;
      });
      const xn = solve(A, b, n);
      let maxDv = 0;
      for (let k = 0; k < n; k++) {
        let d = xn[k] - x[k];
        if (k < N) {
          if (d > 0.5) d = 0.5;
          if (d < -0.5) d = -0.5;
          maxDv = Math.max(maxDv, Math.abs(d));
        }
        xn[k] = x[k] + d;
      }
      x = xn;
      if (maxDv < 1e-7) break;
    }
    out.newton += it + 1;
    }
    out.steps++;
    // cell states after the step: thresholds with hysteresis, escape hazard near the fold, relaxation
    cells.forEach((c, k) => {
      const p = cellPar[k];
      const vds = V(x, c.d) - V(x, c.s);
      if (!c.latched) {
        let up = vds >= p.VLU;
        if (!up && stochastic && h > 0 && vds > p.VLU - 0.15) {
          const lam = 2e3 * Math.exp((vds - p.VLU) / 0.025);
          up = rand() < 1 - Math.exp(-lam * h);
        }
        if (up) {
          c.latched = true;
          c.nUp++;
          if (c.firstUp === null) {
            c.firstUp = t;
            c.firstUpVd = vds;
          }
          out.events.push({ run, kind: "latch_up", t, cell: c.el.name, v_d: vds });
        }
      } else if (vds < p.VLD) {
        c.latched = false;
        c.nDown++;
        out.events.push({ run, kind: "latch_down", t, cell: c.el.name, v_d: vds });
      }
      const tau = c.latched ? 2e-6 : 5e-6;
      c.frac += ((c.latched ? 1 : 0) - c.frac) * (h > 0 ? 1 - Math.exp(-h / tau) : 0);
    });
    if (t + 1e-18 < tSave) {
      xPrev = x;
      continue;
    }
    out.t.push(t);
    nodeNames.forEach((nm, k) => rec(`V(${nm})`, x[k]));
    two.forEach((e, i) => {
      const [a, c] = nodesOf[i];
      let cur = 0;
      if (e.type === "R") cur = (V(x, a) - V(x, c)) / e.value;
      else if (e.type === "C") cur = step === 0 ? 0 : (e.value * (V(x, a) - V(x, c) - (V(xPrev, a) - V(xPrev, c)))) / h;
      else if (e.type === "V") cur = x[vIndex.get(i)!];
      else cur = waveAt((e as { wave: Wave }).wave, t);
      rec(`I(${e.name})`, cur);
    });
    cmps.forEach((c, j) => {
      const hi = cmpVal(j, xPrev).hi;
      out.cmpHigh[j].push(hi);
      rec(`I(${c.name})`, x[N + M + j]);
      rec(`${c.name}.bit`, hi === (c.v_high ?? 1) > (c.v_low ?? 0) ? 1 : 0);
    });
    cells.forEach((c, k) => {
      const vds = V(x, c.d) - V(x, c.s);
      const id = idOf(c, k, vds);
      rec(`I(${c.el.name}.d)`, id);
      rec(`I(${c.el.name}.g)`, 0);
      rec(`I(${c.el.name}.s)`, -id);
      const u = 0.32 + 0.33 * c.frac + 0.015 * Math.max(0, vds);
      rec(`${c.el.name}.u`, u);
      rec(`${c.el.name}.r`, vds - u);
      rec(`${c.el.name}.q_b`, 1.2e-15 + 2e-15 * c.frac + 1e-16 * Math.max(0, vds));
    });
    xPrev = x;
  }
  return out;
}

const L = (ko: string, en: string) => ({ ko, en });

function signalMeta(key: string): Pick<Signal, "label" | "unit" | "axis"> {
  let m = /^V\((.+)\)$/.exec(key);
  if (m) return { label: L(`V(${m[1]})`, `V(${m[1]})`), unit: "V", axis: "voltage" };
  m = /^I\((.+)\)$/.exec(key);
  if (m) return { label: L(`I(${m[1]})`, `I(${m[1]})`), unit: "A", axis: "current" };
  m = /^(.+)\.bit$/.exec(key);
  if (m) return { label: L(`비교기 ${m[1]} 출력`, `Comparator ${m[1]} output`), unit: "1", axis: "logic" };
  m = /^(.+)\.(u|r|q_b)$/.exec(key);
  if (m) {
    if (m[2] === "q_b") return { label: L(`${m[1]} body 전하 Q_B`, `${m[1]} body charge Q_B`), unit: "C", axis: "charge" };
    return { label: L(`${m[1]} 상태 ${m[2]}`, `${m[1]} state ${m[2]}`), unit: "V", axis: "state" };
  }
  return { label: L(key, key), unit: "", axis: "state" };
}

function stats(vals: number[]) {
  const x = vals.filter((v) => Number.isFinite(v));
  if (!x.length) return { mean: null, sd: null };
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  const sd = x.length > 1 ? Math.sqrt(x.reduce((a, b) => a + (b - mean) ** 2, 0) / (x.length - 1)) : null;
  return { mean, sd };
}
function quantile(sorted: number[], p: number) {
  if (!sorted.length) return null;
  const h = (sorted.length - 1) * p;
  const lo = Math.floor(h);
  return sorted[lo] + (sorted[Math.min(sorted.length - 1, lo + 1)] - sorted[lo]) * (h - lo);
}

export function mockCustomCircuit(req: CustomCircuitRequest): CustomCircuitResult {
  const t0 = performance.now();
  const els = req.netlist?.elements ?? [];
  if (!els.length) throw new Error("the netlist has no elements");
  if (els.some((e) => ["MOS", "D", "BJT"].includes(e.type))) throw new Error("MOSFET · diode · BJT simulation requires the live calculation server (실시간 계산 서버가 필요합니다).");
  const touchesGround = els.some((e) => nodesOf(e).some((nd) => GROUND.has(nd)));
  if (!touchesGround) throw new Error("no ground reference: connect at least one element to node 0");
  if (!(req.tran?.t_stop_s > 0)) throw new Error("tran.t_stop_s must be > 0");
  const stochastic = req.mode === "stochastic";
  const nRuns = stochastic ? Math.max(1, Math.min(24, Math.round(req.stochastic?.n_runs ?? 8))) : 1;
  const rand = mulberry32(req.stochastic?.seed ?? 1);
  const grid = timeGrid(req);
  const outs: RunOut[] = [];
  for (let k = 0; k < nRuns; k++) outs.push(simulate(req, grid, k, rand, stochastic));

  const probeSet = req.probes && req.probes.length ? new Set(req.probes) : null;
  const keep = (k: string) => !probeSet || probeSet.has(k) || /\.(u|r|q_b)$/.test(k);
  const runs: CircuitRun[] = outs.slice(0, 8).map((o, k) => ({
    run: k,
    t: o.t,
    signals: [...o.sig.entries()].filter(([key]) => keep(key)).map(([key, values]) => ({ key, ...signalMeta(key), values })),
  }));
  const events = outs.flatMap((o) => o.events);

  // envelopes on a common grid (≤ 1000 points)
  let envelopes: Envelope[] | undefined;
  if (stochastic) {
    const tt = outs[0].t;
    const stride = Math.max(1, Math.ceil(tt.length / 1000));
    const idx = tt.map((_, i) => i).filter((i) => i % stride === 0 || i === tt.length - 1);
    envelopes = [...outs[0].sig.keys()].filter(keep).map((key) => {
      const mean: Arr = [];
      const sd: Arr = [];
      const p05: Arr = [];
      const p95: Arr = [];
      for (const i of idx) {
        const vals = outs.map((o) => o.sig.get(key)?.[i] ?? NaN).filter(Number.isFinite);
        const s = stats(vals);
        const sorted = [...vals].sort((a, b) => a - b);
        mean.push(s.mean);
        sd.push(s.sd ?? 0);
        p05.push(quantile(sorted, 0.05));
        p95.push(quantile(sorted, 0.95));
      }
      return { key, t: idx.map((i) => tt[i]), mean, sd, p05, p95 };
    });
  }

  // summary + distributions per STL
  const summary: SummaryItem[] = [];
  const distributions: NonNullable<CustomCircuitResult["distributions"]> = [];
  const cellNames = outs[0].cells.map((c) => c.el.name);
  cellNames.forEach((nm, ci) => {
    const per = outs.map((o) => o.cells[ci]);
    const ups = stats(per.map((c) => c.nUp));
    const downs = stats(per.map((c) => c.nDown));
    const firstT = per.map((c) => c.firstUp);
    const firstV = per.map((c) => c.firstUpVd);
    const ft = stats(firstT.filter((v): v is number => v != null));
    const fv = stats(firstV.filter((v): v is number => v != null));
    summary.push({ key: `${nm}.n_latch_up`, label: L(`${nm} 래치업 횟수`, `${nm} latch-ups`), value: stochastic ? ups.mean : per[0].nUp, unit: "1", spread: stochastic ? ups.sd : null });
    summary.push({ key: `${nm}.n_latch_down`, label: L(`${nm} 래치다운 횟수`, `${nm} latch-downs`), value: stochastic ? downs.mean : per[0].nDown, unit: "1", spread: stochastic ? downs.sd : null });
    summary.push({ key: `${nm}.t_first_lu`, label: L(`${nm} 첫 래치업 시각`, `${nm} first latch-up time`), value: stochastic ? ft.mean : per[0].firstUp, unit: "s", spread: stochastic ? ft.sd : null });
    summary.push({ key: `${nm}.vd_first_lu`, label: L(`${nm} 래치업 시 V_D`, `${nm} V_D at latch-up`), value: stochastic ? fv.mean : per[0].firstUpVd, unit: "V", spread: stochastic ? fv.sd : null });
    if (stochastic) {
      summary.push({ key: `${nm}.p_latched_end`, label: L(`${nm} 종료 시 래치 확률`, `${nm} P(latched at end)`), value: per.filter((c) => c.latched).length / per.length, unit: "1" });
      summary.push({ key: `${nm}.p_any_lu`, label: L(`${nm} 래치업 ≥ 1회 확률`, `${nm} P(≥ 1 latch-up)`), value: per.filter((c) => c.nUp > 0).length / per.length, unit: "1" });
      distributions.push({ key: `${nm}.t_first_lu`, label: L(`${nm} 첫 래치업 시각`, `${nm} first latch-up time`), unit: "s", values: firstT });
      distributions.push({ key: `${nm}.vd_first_lu`, label: L(`${nm} 래치업 시 V_D`, `${nm} V_D at latch-up`), unit: "V", values: firstV });
    } else summary.push({ key: `${nm}.final_state`, label: L(`${nm} 최종 상태`, `${nm} final state`), value: per[0].latched ? "LRS" : "HRS" });
  });
  if (stochastic) {
    for (const key of [...outs[0].sig.keys()].filter((k) => keep(k) && (k.startsWith("V(") || /^I\(.+\.d\)$/.test(k)))) {
      const meta = signalMeta(key);
      distributions.push({ key: `end:${key}`, label: L(`t_stop에서 ${key}`, `${key} at t_stop`), unit: meta.unit, values: outs.map((o) => o.sig.get(key)?.at(-1) ?? null) });
    }
  }

  const nodes = [...new Set(els.flatMap((e) => nodesOf(e)))];
  const op: Record<string, number> = {};
  for (const [k, v] of outs[0].sig) op[k] = v[0];
  const elementsEcho = els.map((e) => ({ ...e, nodes: Array.isArray(e.nodes) ? [...e.nodes] : { ...e.nodes } })) as unknown as CustomCircuitResult["elements"];
  // comparator firing per period of the periodic pulse source with the most periods (as the server)
  const comparators: ComparatorStats[] = [];
  const pulses = els.flatMap((e) => (e.type === "V" || e.type === "I") && e.wave.kind === "pulse" && e.wave.per > 0 ? [{ name: e.name, w: e.wave }] : []);
  const T = req.tran.t_stop_s;
  let win: { src: string; starts: number[] } | null = null;
  for (const p of pulses) {
    if (p.w.kind !== "pulse") continue;
    let nw = Math.floor((T - p.w.td - p.w.tr - p.w.pw) / p.w.per) + 1;
    if (p.w.ncycles > 0) nw = Math.min(nw, p.w.ncycles);
    if (nw >= 2 && (!win || nw > win.starts.length)) win = { src: p.name, starts: Array.from({ length: nw }, (_, k) => p.w.td + k * (p.w as { per: number }).per) };
  }
  els.forEach((e) => {
    if (e.type !== "CMP") return;
    const j = comparators.length;
    const bits = outs.map((o) => (win ? win.starts.map((s0, k) => {
      const s1 = k + 1 < win!.starts.length ? win!.starts[k + 1] : T;
      return o.t.some((tt, i) => tt >= s0 && tt < s1 && o.cmpHigh[j][i]) ? 1 : 0;
    }) : [])) as number[][];
    const all = bits.flat();
    const pw = win ? win.starts.map((_, k) => bits.reduce((a, b) => a + b[k], 0) / bits.length) : [];
    const p = all.length ? all.reduce((a, b) => a + b, 0) / all.length : null;
    comparators.push({
      name: e.name, nodes: { in: e.nodes.in, inm: e.nodes.inm ?? "0", out: e.nodes.out }, v_ref: e.v_ref, v_high: e.v_high ?? 1, v_low: e.v_low ?? 0, hysteresis: e.hysteresis ?? 0,
      window_source: win?.src ?? null, t_windows: win?.starts ?? [], bits, p_fire_window: pw, p_fire_window_err: pw.map((q) => Math.sqrt((q * (1 - q)) / bits.length)),
      p_fire: p, lag1: null, n_bits: all.length, p_fire_run: bits.map((b) => (b.length ? b.reduce((a, c) => a + c, 0) / b.length : null)),
    });
    if (p != null) summary.push({ key: `${e.name}.p_fire`, label: L(`${e.name} 발화 확률`, `${e.name} firing probability`), value: p, unit: "1" });
  });
  // I–V trajectory of the first STL (run 0), as the server returns it
  let trajectory: CustomCircuitResult["trajectory"];
  const c0 = outs[0].cells[0];
  if (c0) {
    const vd = outs[0].sig.get(`V(${c0.el.nodes.d})`);
    const vs = outs[0].sig.get(`V(${c0.el.nodes.s})`);
    const id = outs[0].sig.get(`I(${c0.el.name}.d)`);
    if (id) trajectory = { vd: id.map((_, i) => (vd?.[i] ?? 0) - (vs?.[i] ?? 0)), id, cell: c0.el.name } as CustomCircuitResult["trajectory"];
  }
  const runtime = (performance.now() - t0) / 1000;
  return {
    bench: "custom",
    mode: req.mode,
    runs,
    events,
    summary,
    distributions,
    envelopes,
    trajectory,
    nodes,
    elements: elementsEcho,
    comparators,
    op,
    schematic: {
      nodes,
      elements: els.map((e) => ({ kind: e.type, name: e.name, nodes: e.type === "STL" ? [e.nodes.d, e.nodes.g, e.nodes.s] : nodesOf(e), value: e.type === "R" || e.type === "C" ? String(e.value) : undefined })),
    },
    solver_stats: { steps: outs.reduce((a, o) => a + o.steps, 0), rejected: 0, newton_iters: outs.reduce((a, o) => a + o.newton, 0), runtime_s: runtime },
    runtime_s: runtime,
    warnings: ["demo data — mock transient (not the STL model)", ...(stochastic && (req.stochastic?.n_runs ?? 0) > nRuns ? [`demo: n_runs capped at ${nRuns}`] : [])],
  };
}
