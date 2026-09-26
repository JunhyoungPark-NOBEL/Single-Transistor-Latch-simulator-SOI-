import { describe, expect, it } from "vitest";
import { buildTemplate } from "../schematic/templates";
import { buildRequest } from "../schematic/netlist";
import { extractNets } from "../schematic/nets";
import { BUILTIN_META } from "../state/presets";
import { clone } from "../utils/object";
import { checkCustomResult, parseProbe, valueAt } from "./circuitCustom";
import { mockCustomCircuit } from "./mockCustom";

const STL = { libId: "builtin:paper", name: "ref", device: clone(BUILTIN_META.presets.paper.device) };

describe("custom-circuit mock (§6 shape)", () => {
  it("deterministic pulse train: signals for every node/element, latch events per cell, summary", () => {
    const d = buildTemplate("pulse", STL, "p");
    const req = buildRequest(d, extractNets(d), "deterministic", null);
    const r = mockCustomCircuit(req);
    expect(checkCustomResult(r)).toEqual([]);
    const keys = r.runs[0].signals.map((s) => s.key);
    for (const k of ["V(src)", "V(d)", "V(g)", "I(Rs)", "I(Cd)", "I(Vsrc)", "I(VG1)", "I(X1.d)", "I(X1.s)", "I(X1.g)", "X1.u", "X1.r", "X1.q_b"]) expect(keys).toContain(k);
    expect(r.events.filter((e) => e.kind === "latch_up" && e.cell === "X1").length).toBe(10);
    const t = r.runs[0].t;
    const vsrc = r.runs[0].signals.find((s) => s.key === "V(src)")!.values;
    expect(valueAt(t, vsrc, 1e-4)!).toBeCloseTo(3.8, 6);
    // source delivering power: negative I(Vsrc) while the cell conducts
    const iv = r.runs[0].signals.find((s) => s.key === "I(Vsrc)")!.values;
    expect(valueAt(t, iv, 1.5e-4)!).toBeLessThan(0);
    const id = r.runs[0].signals.find((s) => s.key === "I(X1.d)")!.values;
    const is = r.runs[0].signals.find((s) => s.key === "I(X1.s)")!.values;
    expect(valueAt(t, id, 1.5e-4)! + valueAt(t, is, 1.5e-4)!).toBeCloseTo(0, 15);
    expect(r.summary.find((s) => s.key === "X1.n_latch_up")?.value).toBe(10);
  });
  it("stochastic: envelopes on a common grid and distributions", () => {
    const d = buildTemplate("pbit", STL, "p");
    d.tran.t_stop_s = 5e-3;
    d.stoch.n_runs = 6;
    const r = mockCustomCircuit(buildRequest(d, extractNets(d), "stochastic", null));
    expect(r.runs.length).toBe(6);
    expect(r.envelopes!.length).toBeGreaterThan(5);
    const env = r.envelopes!.find((e) => e.key === "V(d)")!;
    expect(env.t.length).toBe(env.mean.length);
    expect(env.t.length).toBeLessThanOrEqual(1001);
    expect(r.distributions!.some((x) => x.key === "X1.t_first_lu")).toBe(true);
  });
  it("current-driven oscillator template runs in mock mode", () => {
    const d = buildTemplate("oscillator", STL, "osc");
    const r = mockCustomCircuit(buildRequest(d, extractNets(d), "deterministic", null));
    expect(checkCustomResult(r)).toEqual([]);
    const keys = r.runs[0].signals.map((s) => s.key);
    for (const k of ["V(out)", "I(X1.d)", "I(Iin)", "I(Cpar)"]) expect(keys).toContain(k);
    const iin = r.runs[0].signals.find((s) => s.key === "I(Iin)")!.values;
    expect(valueAt(r.runs[0].t, iin, 1e-3)!).toBeCloseTo(1e-9, 15);
  });
  it("p-bit template runs in mock mode: comparator output, bit trace and per-pulse firing raster", () => {
    const d = buildTemplate("pbit", STL, "p");
    d.stoch.n_runs = 4;
    const r = mockCustomCircuit(buildRequest(d, extractNets(d), "stochastic", null));
    expect(checkCustomResult(r)).toEqual([]);
    const keys = r.runs[0].signals.map((s) => s.key);
    for (const k of ["V(q)", "V(s)", "I(CMP1)", "CMP1.bit"]) expect(keys).toContain(k);
    const cmp = r.comparators![0];
    expect(cmp.name).toBe("CMP1");
    expect(cmp.window_source).toBe("Vpulse");
    expect(cmp.t_windows.length).toBe(20);
    expect(cmp.bits.length).toBe(4);
    expect(cmp.bits[0].length).toBe(20);
    expect(cmp.p_fire_window.length).toBe(20);
  });
  it("probe keys", () => {
    expect(parseProbe("V(d)")).toEqual({ type: "V", node: "d" });
    expect(parseProbe("I(X1.d)")).toEqual({ type: "I", el: "X1", terminal: "d" });
    expect(parseProbe("I(R1)")).toEqual({ type: "I", el: "R1", terminal: undefined });
    expect(parseProbe("X1.q_b")).toEqual({ type: "state", el: "X1", q: "q_b" });
  });
});
