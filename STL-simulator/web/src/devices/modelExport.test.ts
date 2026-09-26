import { describe, expect, it } from "vitest";
import type { BranchesResult } from "../api/types";
import reference from "./fixtures/export-reference.json";
import { buildLtspiceExport, buildVerilogAExport, ModelExportError, type ExportSelection } from "./modelExport";
import { REFERENCE_GEOMETRY } from "../params/geometry";

const selection = reference.selection as ExportSelection;
const result = reference.result as unknown as BranchesResult;

/** Parse the generated netlist, rather than reusing the exporter's interpolation implementation. */
function parsedCurrent(netlist: string, branch: string, v: number): number {
  if (v <= 0) return 0;
  const body = netlist.split(`.func ${branch}(v)`)[1].split("\n+ ))) }")[0];
  const end = body.indexOf("\n+ )))}");
  const pairs = body.slice(0, end).split("\n").filter((line) => line.startsWith("+ "))
    .map((line) => line.slice(2).split(",").slice(0, 2).map(Number));
  expect(pairs.length).toBeGreaterThan(2);
  if (v <= pairs[0][0]) return Math.exp(pairs[0][1]);
  for (let i = 1; i < pairs.length; i++) {
    const [x, y] = pairs[i], [xp, yp] = pairs[i - 1];
    if (v <= x) return Math.exp(yp + ((v - xp) / (x - xp)) * (y - yp));
  }
  return Math.exp(pairs.at(-1)![1]);
}

describe("LTspice calibration export", () => {
  const exported = buildLtspiceExport(selection, result);

  it("preserves the live engine ID–VD double sweep through independently parsed tables", () => {
    const lu = result.folds.V_LU!, ld = result.folds.V_LD!;
    expect(lu).toBeCloseTo(3.7037, 3);
    expect(ld).toBeCloseTo(2.5979, 3);
    for (const [direction, xy] of Object.entries(result.double_sweep)) {
      for (let i = 0; i < xy.vd.length; i++) {
        const v = xy.vd[i]!, current = xy.id[i]!;
        // A vertical fold has two legal currents at one voltage: compare its two sides separately.
        if (Math.abs(v - lu) < 1e-9 || Math.abs(v - ld) < 1e-9) continue;
        const branch = v > (direction === "up" ? lu : ld) ? "Ilrs" : "Ihrs";
        const actual = parsedCurrent(exported.circuit, branch, v);
        if (current === 0) expect(actual).toBe(0);
        else expect(Math.abs(actual / current - 1)).toBeLessThan(1e-9);
      }
    }
  });

  it("uses native positive hysteresis with the same upper and lower thresholds", () => {
    const switchLine = exported.circuit.split("\n").find((line) => line.startsWith(".model latch_memory"))!;
    const vt = Number(switchLine.match(/Vt=([^ ]+)/)![1]);
    const vh = Number(switchLine.match(/Vh=([^ )]+)/)![1]);
    expect(vh).toBeGreaterThan(0);
    expect(vt + vh).toBeCloseTo(result.folds.V_LU!, 12);
    expect(vt - vh).toBeCloseTo(result.folds.V_LD!, 12);
    expect(exported.circuit).toContain("Smemory logic state D S latch_memory OFF");
    expect(exported.circuit).toContain("Xdevice d 0 STL_Reference");
    expect(exported.circuit).toContain(".subckt STL_Reference D S");
  });

  it("keeps every effective calibration value and explicit model scope", () => {
    const comments = exported.circuit.split("* Export metadata (submitted calibration and actual engine vector):\n")[1].split("\n.end")[0];
    const metadata = JSON.parse(comments.split("\n").map((line) => line.slice(2)).join("\n"));
    expect(metadata.effective_engine_p).toEqual(result.p);
    expect(metadata.submitted_device).toEqual(selection.device);
    expect(metadata.valid_VDS_V).toEqual([0, selection.sweep.vd_max_V]);
    expect(metadata.fixed_geometry).toEqual(REFERENCE_GEOMETRY);
    expect(metadata.fixed_VBG_V).toBe(0);
    expect(exported.subcircuit).toContain('"fixed_geometry"');
    expect(exported.subcircuit).toContain('"effective_engine_p"');
    expect(exported.readme).toContain("body-charge ODE");
    expect(exported.readme).toContain("바디 전하");
    expect(exported.circuit).toContain("No physical body-charge transient");
  });

  it("rejects out-of-range sweeps, uncomputed cells and a result from a different gate bias", () => {
    expect(() => buildLtspiceExport({ ...selection, sweep: { ...selection.sweep, vd_max_V: 2 } }, result)).toThrow(ModelExportError);
    const invalid = structuredClone(result);
    invalid.LRS.id[5] = null;
    expect(() => buildLtspiceExport(selection, invalid)).toThrow(ModelExportError);
    expect(() => buildLtspiceExport({ ...selection, device: { ...selection.device, vg: -1.8 } }, result)).toThrow(ModelExportError);
  });

  it("exports a complete non-latching branch without inventing a hysteresis switch", () => {
    const mono = structuredClone(result);
    mono.latch = false;
    mono.double_sweep.up = { vd: [0, 1, 4], id: [0, 1e-10, 1e-7] };
    const out = buildLtspiceExport(selection, mono);
    expect(out.circuit).not.toContain("Smemory");
    expect(out.circuit).toContain("Bdrain D S I={Ihrs(V(D,S))}");
    expect(parsedCurrent(out.circuit, "Ihrs", 2)).toBeCloseTo(1e-9, 20);
  });

  it("keeps names from adding executable netlist lines", () => {
    const out = buildLtspiceExport({ ...selection, name: "my device\n.end\nVbad d 0 999" }, result);
    expect(out.modelName).toMatch(/^STL_[A-Za-z0-9_]+$/);
    expect(out.circuit.split("\n").filter((line) => line === ".end")).toHaveLength(1);
    expect(out.circuit).not.toMatch(/^Vbad /m);
  });

  it("rejects baseline or mismatched geometry results for a resized device", () => {
    const geometry = { ...REFERENCE_GEOMETRY, Lg_nm: 300, Tsi_nm: 35, EOT_nm: 10 };
    const resized = { ...selection, device: { ...selection.device, geometry } };
    for (const build of [buildLtspiceExport, buildVerilogAExport]) {
      expect(() => build(resized, result)).toThrow(ModelExportError);
      expect(() => build(resized, { ...result, geometry: REFERENCE_GEOMETRY })).toThrow(ModelExportError);
    }
  });

  it("preserves six-field geometry provenance and compact extended vectors in both formats", () => {
    // Serialization contract only; numerical geometry behavior is tested with the live Python solver.
    const geometry = { ...REFERENCE_GEOMETRY, W_nm: 400 };
    const selected = { ...selection, device: { ...selection.device, geometry } };
    const computed: BranchesResult = { ...result, p: [...result.p, 500, 400, 50, 14.1, 140, REFERENCE_GEOMETRY.Nbody_cm3], geometry,
      geometry_model: { version: "fdsoi-scaling-v1", calibrated_geometry: REFERENCE_GEOMETRY, tbox_source: "nominal assumption; absent from calibration", validated: false } };
    const spice = buildLtspiceExport(selected, computed);
    const verilog = buildVerilogAExport(selected, computed);
    expect(spice.circuit).toContain("W 400 nm");
    expect(verilog.source).toContain("W 400 nm");
    expect(spice.circuit).toContain('"fixed_geometry"');
    expect(verilog.source).toContain('"geometry_model"');
    expect(spice.readme).toContain("no tunable geometry parameters");
    expect(verilog.readme).toContain("no tunable geometry parameters");
  });

  it("rejects stale back-gate results and preserves a matching 33-element VBG result", () => {
    const selected = { ...selection, device: { ...selection.device, vbg: -2 } };
    const withBackGate: BranchesResult = { ...result, p: [...result.p, ...Object.values(REFERENCE_GEOMETRY), -2], vbg: -2, geometry: { ...REFERENCE_GEOMETRY } };
    for (const build of [buildLtspiceExport, buildVerilogAExport]) {
      expect(() => build(selected, result)).toThrow(ModelExportError);
      expect(() => build(selected, { ...withBackGate, vbg: 0 })).toThrow(ModelExportError);
      expect(() => build(selected, { ...withBackGate, p: [...withBackGate.p.slice(0, 32), 0] })).toThrow(ModelExportError);
      expect(() => build(selected, { ...withBackGate, p: withBackGate.p.slice(0, 32) })).toThrow(ModelExportError);
      expect(() => build(selected, withBackGate)).not.toThrow();
    }
    const spice = buildLtspiceExport(selected, withBackGate);
    const verilog = buildVerilogAExport(selected, withBackGate);
    expect(spice.subcircuit).toContain('"fixed_VBG_V": -2');
    expect(verilog.source).toContain('"fixed_VBG_V": -2');
    expect(spice.readme).toContain("VBG=-2");
    expect(verilog.readme).toContain("VBG=-2");
  });
});

/** Interpret only the emitted analog function expressions, not the generator's helper. */
function parsedVerilogCurrent(source: string, branch: string, v: number): number {
  if (v <= 0) return 0;
  const body = source.split(`analog function real ${branch};`)[1].split("endfunction")[0];
  const first = body.match(/else if \(v <= ([^)]+)\) \w+ = exp\(([^)]+)\);/)!;
  if (v <= Number(first[1])) return Math.exp(Number(first[2]));
  const clauses = [...body.matchAll(/else if \(v <= ([^)]+)\)\s+\w+ = exp\(([^ ]+) \+ \(v - ([^)]+)\) \* ([^)]+)\);/g)];
  expect(clauses.length).toBeGreaterThan(0);
  for (const [, end, logCurrent, start, slope] of clauses) {
    if (v <= Number(end)) return Math.exp(Number(logCurrent) + (v - Number(start)) * Number(slope));
  }
  return Math.exp(Number(body.match(/else \w+ = exp\(([^)]+)\);/)![1]));
}

describe("Verilog-A calibration export", () => {
  const exported = buildVerilogAExport(selection, result);

  it("preserves the engine's full up/down sweep through independently parsed analog functions", () => {
    const lu = Number(exported.source.match(/localparam real V_LU = ([^;]+);/)![1]);
    const ld = Number(exported.source.match(/localparam real V_LD = ([^;]+);/)![1]);
    expect(lu).toBe(result.folds.V_LU);
    expect(ld).toBe(result.folds.V_LD);
    for (const [direction, xy] of Object.entries(result.double_sweep)) {
      for (let i = 0; i < xy.vd.length; i++) {
        const v = xy.vd[i]!, current = xy.id[i]!;
        if (Math.abs(v - lu) < 1e-9 || Math.abs(v - ld) < 1e-9) continue;
        const branch = v > (direction === "up" ? lu : ld) ? "i_lrs" : "i_hrs";
        const actual = parsedVerilogCurrent(exported.source, branch, v);
        if (current === 0) expect(actual).toBe(0);
        else expect(Math.abs(actual / current - 1)).toBeLessThan(1e-9);
      }
    }
    expect(exported.source).toContain("@(initial_step) lrs_state = (V(D,S) >= V_LU) ? 1 : 0;");
    expect(exported.source).toContain("@(cross(V(D,S) - V_LU, +1)) lrs_state = 1;");
    expect(exported.source).toContain("@(cross(V(D,S) - V_LD, -1)) lrs_state = 0;");
  });

  it("retains the actual calibration and clearly separates static behavior from CSVM and TCAD", () => {
    const metadataText = exported.source.split("// Export metadata (submitted calibration and actual engine vector):\n")[1];
    const metadata = JSON.parse(metadataText.trim().split("\n").map((line) => line.slice(3)).join("\n"));
    expect(metadata.effective_engine_p).toEqual(result.p);
    expect(metadata.submitted_device).toEqual(selection.device);
    expect(metadata.fixed_IPH_A).toBe(result.iph_A);
    expect(exported.readme).toContain("No body-charge ODE");
    expect(exported.readme).toContain("CSVM Vtop/Vbottom/frequency");
    expect(exported.readme).toContain("not a Sentaurus physical-device deck");
    expect(exported.readme).toContain("No commercial Verilog-A simulator was executed");
  });

  it("keeps two-terminal topology and prevents names from adding source code", () => {
    const named = buildVerilogAExport({ ...selection, name: 'user\nendmodule\nmodule bad();' }, result);
    expect(named.modelName).toMatch(/^STL_[A-Za-z0-9_]+$/);
    expect(named.source.match(/^module /gm)).toHaveLength(1);
    expect(named.source.match(/^endmodule$/gm)).toHaveLength(1);
    expect(named.source).toContain(`module ${named.modelName}(D, S);`);
    expect(named.source).toContain('`include "disciplines.vams"');
  });

  it("exports non-latching conditions as a single current law and refuses incomplete data", () => {
    const mono = structuredClone(result);
    mono.latch = false;
    mono.double_sweep.up = { vd: [0, 1, 4], id: [0, 1e-10, 1e-7] };
    const out = buildVerilogAExport(selection, mono);
    expect(out.source).not.toContain("lrs_state");
    expect(out.source).not.toContain("@(cross");
    expect(parsedVerilogCurrent(out.source, "i_hrs", 2)).toBeCloseTo(1e-9, 20);
    expect(parsedVerilogCurrent(out.source, "i_hrs", -1)).toBe(0);
    expect(parsedVerilogCurrent(out.source, "i_hrs", 5)).toBeCloseTo(1e-7, 18);
    mono.double_sweep.up.id[1] = null;
    expect(() => buildVerilogAExport(selection, mono)).toThrow(ModelExportError);
    expect(() => buildVerilogAExport({ ...selection, device: { ...selection.device, vg: 0 } }, result)).toThrow(ModelExportError);
  });
});
