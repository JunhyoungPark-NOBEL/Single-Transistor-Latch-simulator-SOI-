import { describe, expect, it } from "vitest";
import { BUILTIN_META } from "../state/presets";
import { clone } from "../utils/object";
import { hasErrors, runErc } from "./erc";
import { estimate } from "./feasibility";
import { newId, pinPositions, rotatePt, type ElKind, type Rot, type SchematicDoc, type SElement, type StlRef, type Wire } from "./model";
import { extractNets } from "./nets";
import { buildRequest, effectiveTran, netlistText } from "./netlist";
import { defaultDoc, exportDocJson, parseDoc, parseStored } from "./persist";
import { fmtSI, parseSI, toSpice } from "./si";
import { buildTemplate, TEMPLATE_ORDER } from "./templates";
import { breakpointCount, waveAt, waveIssues, wavePoints, waveSpice } from "./waves";

const STL: StlRef = { libId: "builtin:paper", name: "FDSOI ref", device: clone(BUILTIN_META.presets.paper.device) };
const el = (kind: ElKind, name: string, x: number, y: number, extra: Partial<SElement> = {}, rot: Rot = 0): SElement => ({ id: newId(), kind, name, x, y, rot, ...extra });
const w = (x1: number, y1: number, x2: number, y2: number): Wire => ({ id: newId("w"), x1, y1, x2, y2 });

/** V1 (0..) → R1 → node d with C1 to ground; V1 − grounded. */
function rcDoc(): SchematicDoc {
  const d = defaultDoc("rc");
  d.elements = [
    el("V", "V1", 0, 0, { wave: { kind: "dc", value: 1 } }), // + (0,-40), − (0,40)
    el("GND", "", 0, 40),
    el("R", "R1", 80, -80, { value: 1e3 }, 3), // p1 (40,-80) p2 (120,-80)
    el("C", "C1", 160, 0, { value: 2e-15 }), // (160,-40) (160,40)
    el("GND", "", 160, 40),
  ];
  d.wires = [w(0, -40, 0, -80), w(0, -80, 40, -80), w(120, -80, 160, -80), w(160, -80, 160, -40)];
  return d;
}

describe("SPICE numbers", () => {
  it("parses SI suffixes with SPICE semantics", () => {
    expect(parseSI("1k")).toBe(1e3);
    expect(parseSI("2f")).toBeCloseTo(2e-15, 30);
    expect(parseSI("10u")).toBeCloseTo(1e-5, 20);
    expect(parseSI("10µ")).toBeCloseTo(1e-5, 20);
    expect(parseSI("1meg")).toBe(1e6);
    expect(parseSI("1MEG")).toBe(1e6);
    expect(parseSI("1M")).toBeCloseTo(1e-3, 15); // SPICE: M = milli
    expect(parseSI("4.7n")).toBeCloseTo(4.7e-9, 20);
    expect(parseSI("100p")).toBeCloseTo(1e-10, 20);
    expect(parseSI("1e-3")).toBe(1e-3);
    expect(parseSI("-2")).toBe(-2);
    expect(parseSI("−2.5")).toBe(-2.5);
    expect(parseSI(" 2.2kΩ ")).toBeCloseTo(2200, 9);
    expect(parseSI("10us")).toBeCloseTo(1e-5, 20);
    expect(parseSI("5V")).toBe(5);
    expect(parseSI("1F")).toBeCloseTo(1e-15, 30); // SPICE gotcha, shown in the UI
    expect(parseSI("3.3e3k")).toBe(3.3e6);
    expect(parseSI("1g")).toBe(1e9);
    expect(parseSI("abc")).toBeNull();
    expect(parseSI("")).toBeNull();
    expect(parseSI("1k2")).toBeNull();
    expect(parseSI("1/s")).toBeNull();
  });
  it("formats and round-trips", () => {
    expect(fmtSI(2e-15, "F")).toBe("2 fF");
    expect(fmtSI(1e6, "Ω")).toBe("1 MΩ");
    expect(fmtSI(1e-5, "s")).toBe("10 µs");
    expect(fmtSI(0, "V")).toBe("0 V");
    expect(toSpice(1e6)).toBe("1meg");
    expect(toSpice(2e-15)).toBe("2f");
    expect(toSpice(1e-5)).toBe("10u");
    expect(toSpice(-3.8)).toBe("-3.8");
    for (const v of [1e6, 2e-15, 1e-5, 4.7e-9, 0.02, 1234.5, 1e-18, 3e15]) expect(Math.abs(parseSI(toSpice(v))! - v) / v).toBeLessThan(1e-12);
  });
});

describe("waveforms", () => {
  it("PULSE follows SPICE semantics and yields corner points", () => {
    const p = { kind: "pulse" as const, v1: 0, v2: 3.8, td: 1e-4, tr: 1e-5, tf: 1e-5, pw: 2e-4, per: 1e-3, ncycles: 2 };
    expect(waveAt(p, 0)).toBe(0);
    expect(waveAt(p, 1e-4 + 5e-6)).toBeCloseTo(1.9, 9);
    expect(waveAt(p, 1e-4 + 1e-5 + 1e-4)).toBe(3.8);
    expect(waveAt(p, 1e-4 + 1e-5 + 2e-4 + 5e-6)).toBeCloseTo(1.9, 9);
    expect(waveAt(p, 1.2e-3)).toBe(3.8); // second cycle flat top
    expect(waveAt(p, 1.5e-3)).toBe(0);
    expect(waveAt(p, 2.5e-3)).toBe(0); // after ncycles
    const pts = wavePoints(p, 3e-3);
    expect(pts.t[0]).toBe(0);
    expect(pts.t).toContain(1e-4);
    expect(pts.t).toContain(1e-4 + 1e-5);
    expect(pts.t.filter((x) => x > 0 && x < 3e-3).length).toBe(8); // 2 cycles × 4 corners
    expect(pts.t[pts.t.length - 1]).toBe(3e-3);
    expect(breakpointCount(p, 3e-3)).toBe(8);
    expect(waveSpice(p)).toBe("PULSE(0 3.8 100u 10u 10u 200u 1m 2)");
  });
  it("PWL, SINE and validation", () => {
    const pwl = { kind: "pwl" as const, t: [0, 10, 20], v: [0, 4, 0] };
    expect(waveAt(pwl, 5)).toBe(2);
    expect(waveAt(pwl, 30)).toBe(0);
    const s = { kind: "sine" as const, vo: 1, va: 2, freq: 1e3, td: 0, theta: 0 };
    expect(waveAt(s, 0.25e-3)).toBeCloseTo(3, 9);
    expect(waveIssues({ kind: "pulse", v1: 0, v2: 1, td: 0, tr: 1e-3, tf: 1e-3, pw: 1e-3, per: 2e-3, ncycles: 0 })[0].key).toBe("period");
    expect(waveIssues({ kind: "pwl", t: [0, 2, 1], v: [0, 1, 2] })[0].key).toBe("pwlOrder");
    expect(waveIssues(pwl)).toEqual([]);
  });
});

describe("geometry", () => {
  it("rotates clockwise in screen coordinates", () => {
    expect(rotatePt(0, -40, 1)).toEqual({ x: 40, y: 0 });
    expect(rotatePt(0, -40, 3)).toEqual({ x: -40, y: 0 });
    expect(rotatePt(-40, 0, 0, true)).toEqual({ x: 40, y: 0 });
    const r = el("R", "R1", 100, 100, {}, 3);
    expect(pinPositions(r).map((p) => [p.x, p.y])).toEqual([[60, 100], [140, 100]]);
  });
});

describe("net extraction (union-find)", () => {
  it("joins wires, pins and T-junctions; names ground 0 and others N001…", () => {
    const d = rcDoc();
    const c = extractNets(d);
    const names = c.nets.filter((n) => n.pins.length).map((n) => n.name).sort();
    expect(names).toEqual(["0", "N001", "N002"]);
    const v1 = d.elements[0];
    expect(c.pinNet.get(`${v1.id}:n`)!.name).toBe("0");
    expect(c.pinNet.get(`${v1.id}:p`)!.name).toBe("N001");
    expect(c.unconnected.length).toBe(0);
  });
  it("net labels merge nets and name them; T-junction gives a junction dot", () => {
    const d = rcDoc();
    d.elements.push(el("LABEL", "", 140, -80, { label: "out" }));
    d.wires.push(w(140, -80, 140, 0)); // branch from the middle of a wire
    d.elements.push(el("R", "R2", 140, 40, { value: 1e3 })); // (140,0) – (140,80)
    d.elements.push(el("GND", "", 140, 80));
    const c = extractNets(d);
    const out = c.byName.get("out")!;
    expect(out).toBeDefined();
    expect(out.pins.map((p) => p.el.name).sort()).toEqual(["C1", "R1", "R2"]);
    expect(c.junctions).toContainEqual({ x: 140, y: -80 });
    // a distant label with the same name joins the same net
    d.elements.push(el("LABEL", "", 400, 400, { label: "out" }));
    d.elements.push(el("C", "C9", 400, 440, { value: 1e-15 })); // pin (400,400)
    const c2 = extractNets(d);
    expect(c2.byName.get("out")!.pins.map((p) => p.el.name)).toContain("C9");
  });
  it("labels named gnd/0 are ground", () => {
    const d = rcDoc();
    d.elements = d.elements.filter((e) => e.kind !== "GND");
    d.elements.push(el("LABEL", "", 0, 40, { label: "gnd" }), el("LABEL", "", 160, 40, { label: "0" }));
    const c = extractNets(d);
    expect(c.nets.some((n) => n.ground && n.pins.length === 2)).toBe(true);
  });
});

describe("ERC", () => {
  it("passes a valid RC circuit", () => {
    const d = rcDoc();
    expect(runErc(d, extractNets(d)).filter((i) => i.level === "error")).toEqual([]);
  });
  it("finds missing ground, floating nodes, unconnected pins, V loops and DC-path problems", () => {
    const noGnd = rcDoc();
    noGnd.elements = noGnd.elements.filter((e) => e.kind !== "GND");
    expect(runErc(noGnd, extractNets(noGnd)).map((i) => i.code)).toContain("noGround");

    const dangling = rcDoc();
    dangling.elements.push(el("R", "R9", 300, 0, { value: 1 }));
    const codes = runErc(dangling, extractNets(dangling)).map((i) => i.code);
    expect(codes.filter((c) => c === "unconnected").length).toBe(2);

    const floating = rcDoc();
    floating.elements.push(el("R", "R9", 300, 0, { value: 1 })); // (300,-40)–(300,40)
    floating.wires.push(w(300, -40, 300, -100));
    floating.elements.push(el("GND", "", 300, 40));
    expect(runErc(floating, extractNets(floating)).map((i) => i.code)).toContain("floating");

    const vloop = rcDoc();
    vloop.elements.push(el("V", "V2", -100, 0, { wave: { kind: "dc", value: 2 } }));
    vloop.wires.push(w(-100, -40, -100, -80), w(-100, -80, 0, -80), w(-100, 40, 0, 40));
    const vi = runErc(vloop, extractNets(vloop));
    expect(vi.map((i) => i.code)).toContain("vLoop");
    expect(hasErrors(vi)).toBe(true);

    // capacitor-only node: no DC path to ground
    const cap = rcDoc();
    cap.elements = cap.elements.map((e) => (e.name === "R1" ? { ...e, kind: "C" as const, value: 1e-15 } : e));
    const ci = runErc(cap, extractNets(cap));
    expect(ci.find((i) => i.code === "noDcPath")?.vars?.node).toBe("N002");

    // current source in series with nothing
    const isrc = rcDoc();
    isrc.elements = isrc.elements.map((e) => (e.name === "R1" ? { ...e, kind: "I" as const, wave: { kind: "dc" as const, value: 1e-9 } } : e));
    isrc.elements = isrc.elements.map((e) => (e.name === "C1" ? { ...e, kind: "I" as const, name: "I2", wave: { kind: "dc" as const, value: 1e-9 } } : e));
    expect(runErc(isrc, extractNets(isrc)).map((i) => i.code)).toContain("iSeries");

    // bad values and duplicate names
    const bad = rcDoc();
    bad.elements[2].value = 0;
    bad.elements[3].name = "r1";
    const bi = runErc(bad, extractNets(bad)).map((i) => i.code);
    expect(bi).toContain("badR");
    expect(bi).toContain("dupName");
  });
  it("templates pass ERC and gate-only nodes are flagged", () => {
    for (const id of TEMPLATE_ORDER) {
      const d = buildTemplate(id, STL, id);
      const items = runErc(d, extractNets(d));
      expect(items.filter((i) => i.level === "error"), id).toEqual([]);
    }
    const d = buildTemplate("load_line", STL, "x");
    d.elements = d.elements.filter((e) => e.name !== "VG1");
    d.elements.push(el("C", "Cg", 300, 280, { value: 1e-15 }));
    expect(runErc(d, extractNets(d)).map((i) => i.code)).toContain("noDcPath");
  });
});

describe("netlist serialisation", () => {
  it("builds the §6 request with node names, waves, STL terminals and tran", () => {
    const d = buildTemplate("pulse", STL, "pulse");
    const c = extractNets(d);
    const req = buildRequest(d, c, "deterministic", ["V(d)"]);
    expect(req.bench).toBe("custom");
    expect(req.probes).toBeNull(); // save_all
    const x1 = req.netlist.elements.find((e) => e.name === "X1")!;
    expect(x1.type).toBe("STL");
    if (x1.type === "STL") {
      expect(x1.nodes).toEqual({ d: "d", g: "g", s: "0" });
      expect(x1.light_pA).toBeNull();
    }
    const rs = req.netlist.elements.find((e) => e.name === "Rs")!;
    expect(rs).toMatchObject({ type: "R", nodes: ["src", "d"], value: 1000 });
    const v = req.netlist.elements.find((e) => e.name === "Vsrc")!;
    expect(v).toMatchObject({ type: "V", nodes: ["src", "0"], wave: { kind: "pulse", v2: 3.8, per: 1e-3, ncycles: 10 } });
    expect(req.tran).toEqual({ t_stop_s: 10e-3, t_start_save_s: 0, dt_max_s: 5e-6, dt_min_s: 1e-15, method: "BE", reltol: 1e-3 });
    expect(req.stochastic).toBeUndefined();
    const sto = buildRequest({ ...d, save_all: false }, c, "stochastic", ["V(d)", "I(X1.d)", "V(gone)", "I(R77)"]);
    expect(sto.probes).toEqual(["V(d)", "I(X1.d)"]); // unknown keys are dropped (the server rejects them)
    expect(sto.stochastic?.n_runs).toBe(20);
    expect(sto.stochastic?.local_state_override).toBe(false);
    expect(buildRequest({ ...d, save_all: false }, c, "deterministic", ["V(nope)"]).probes).toBeNull();
    const ov = buildRequest({ ...d, stoch: { ...d.stoch, local_source: "override" } }, c, "stochastic", null);
    expect(ov.stochastic?.local_state_override).toBe(true);
    const text = netlistText(d, c, "deterministic");
    expect(text).toContain("Vsrc src 0 PULSE(0 3.8 0 10u 10u 200u 1m 10)");
    expect(text).toContain("Rs src d 1k");
    expect(text).toContain("Cd d 0 2f");
    expect(text).toMatch(/X1 d g 0 STL/);
    expect(text).toContain(".tran 0 10m 0 5u");
  });
  it("current-driven oscillator template: I_in into out, C_par and the STL drain on out, gate at a DC source", () => {
    const d = buildTemplate("oscillator", STL, "osc");
    const c = extractNets(d);
    expect(runErc(d, c).filter((i) => i.level === "error")).toEqual([]);
    const req = buildRequest(d, c, "deterministic", null);
    const byName = Object.fromEntries(req.netlist.elements.map((e) => [e.name, e]));
    // the source is drawn from ground (rotated 180°): the current flows 0 → out through it, i.e. INTO out
    expect(byName.Iin).toMatchObject({ type: "I", nodes: ["0", "out"], wave: { kind: "dc", value: 1e-9 } });
    expect(byName.Cpar).toMatchObject({ type: "C", nodes: ["out", "0"], value: 1e-12 });
    const x1 = byName.X1;
    expect(x1.type).toBe("STL");
    if (x1.type === "STL") {
      expect(x1.nodes.d).toBe("out");
      expect(x1.nodes.s).toBe("0");
      expect(byName.VG1).toMatchObject({ type: "V", nodes: [x1.nodes.g, "0"], wave: { kind: "dc", value: -2 } });
    }
    expect(req.tran).toMatchObject({ t_stop_s: 15e-3, dt_max_s: 5e-6, method: "BE" });
    const text = netlistText(d, c, "deterministic");
    expect(text).toContain("Iin 0 out DC 1n");
    expect(text).toContain("Cpar out 0 1p");
    expect(text).toContain(".tran 0 15m 0 5u");
    // the only labelled net is "out": the default traces after a run are V(out) and I(X1.d)
    expect(d.elements.filter((e) => e.kind === "LABEL").map((e) => e.label)).toEqual(["out"]);
    expect(estimate(d, "deterministic").level).toBe("ok");
    expect(estimate(d, "stochastic").level).not.toBe("refuse");
  });
  it("p-bit template: drain pulses, source resistor, comparator on the source node (carrier noise only)", () => {
    const d = buildTemplate("pbit", STL, "pbit");
    const c = extractNets(d);
    expect(runErc(d, c).filter((i) => i.level === "error")).toEqual([]);
    const req = buildRequest(d, c, "stochastic", null);
    const byName = Object.fromEntries(req.netlist.elements.map((e) => [e.name, e]));
    expect(byName.Vpulse).toMatchObject({ type: "V", nodes: ["d", "0"], wave: { kind: "pulse", v2: 3.69, tr: 20e-6, pw: 200e-6, per: 1e-3, ncycles: 20 } });
    expect(byName.RS).toMatchObject({ type: "R", nodes: ["s", "0"], value: 100e3 });
    expect(byName.CMP1).toMatchObject({ type: "CMP", nodes: { in: "s", out: "q" }, v_ref: 0.1, v_high: 1, v_low: 0, hysteresis: 0 });
    const x1 = byName.X1;
    if (x1.type === "STL") expect([x1.nodes.d, x1.nodes.s]).toEqual(["d", "s"]);
    expect(req.stochastic?.local_state_override).toBe(true);
    expect(req.stochastic?.local_state.mode).toBe("none");
    expect(req.tran.t_stop_s).toBeCloseTo(20e-3, 12);
    const text = netlistText(d, c, "stochastic");
    expect(text).toContain("CMP1 s 0 q CMP vref=100m vhigh=1 vlow=0");
    expect(text).toContain("RS s 0 100k");
    expect(estimate(d, "stochastic").level).not.toBe("refuse");
    // the comparator persists (export → import)
    const back = parseDoc(JSON.parse(exportDocJson(d)))!;
    expect(back.elements.find((e) => e.kind === "CMP")?.cmp).toEqual({ v_ref: 0.1, v_high: 1, v_low: 0, hysteresis: 0 });
    // ERC: a comparator output must not be driven by another source (Vx on the output net q)
    const bad = clone(d);
    bad.elements.push(el("V", "Vx", 700, 300, { wave: { kind: "dc", value: 1 } }), el("GND", "", 700, 340));
    bad.wires.push(w(660, 260, 700, 260));
    const codes = runErc(bad, extractNets(bad)).map((i) => i.code);
    expect(codes).toContain("cmpDriven");
  });
  it("auto time steps", () => {
    expect(effectiveTran({ t_stop_s: 1, t_start_save_s: 0, dt_max_s: null, dt_min_s: null, method: "TRAP", reltol: 1e-4 })).toEqual({ t_stop_s: 1, t_start_save_s: 0, dt_max_s: 5e-4, dt_min_s: 1e-13, method: "TRAP", reltol: 1e-4 });
  });
});

describe("feasibility and persistence", () => {
  it("flags a slow stochastic ramp", () => {
    const d = buildTemplate("load_line", STL, "ll");
    expect(estimate(d, "deterministic").level).toBe("ok");
    expect(estimate(d, "stochastic").level).toBe("refuse");
    expect(estimate(buildTemplate("pulse", STL, "p"), "stochastic").level).not.toBe("refuse");
    expect(estimate(buildTemplate("coupled", STL, "cp"), "stochastic").level).not.toBe("refuse");
  });
  it("round-trips and sanitises documents", () => {
    const d = buildTemplate("coupled", STL, "cp");
    const back = parseDoc(JSON.parse(exportDocJson(d)));
    expect(back).toEqual(d);
    expect(parseDoc({ elements: [{ kind: "nope", x: 0, y: 0 }, { kind: "R", x: 3, y: 7 }], wires: [{ x1: 0 }] })!.elements).toMatchObject([{ kind: "R", x: 0, y: 10, value: 1000 }]);
    expect(parseDoc(null)).toBeNull();
    expect(parseStored("{bad").doc).toBeNull();
    expect(parseStored(JSON.stringify({ doc: d, saved: [{ name: "a", doc: d }, { name: 5 }] })).saved.length).toBe(1);
  });
});
