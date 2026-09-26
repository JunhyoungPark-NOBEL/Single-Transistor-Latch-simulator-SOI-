import { describe, expect, it } from "vitest";
import { legacyGeometryPayload } from "../api/geometryPolicy";
import { csvmPayload, DEFAULT_CSVM } from "../device/forcing";
import { sanitizeDevice } from "../devices/library";
import { buildLtspiceExport, buildVerilogAExport } from "../devices/modelExport";
import type { BranchesResult } from "../api/types";
import { restoreParams } from "../state/persist";
import { BUILTIN_META } from "../state/presets";
import { clone } from "../utils/object";
import { branchesPayload, presetRoot } from "../utils/payload";
import { hasSimpleModel, modelOf, parseHrsPoints, SIMPLE_DEFAULTS, simpleOf } from "./model";

describe("model identity and separate calibration", () => {
  it("legacy device settings and imported device files remain Detailed", () => {
    const base = presetRoot(BUILTIN_META, "paper");
    const legacy = clone(base); delete legacy.device.model; delete legacy.device.simple;
    expect(restoreParams(base, legacy).device.model).toBe("detailed");
    expect(sanitizeDevice(base.device, legacy.device).simple).toEqual(SIMPLE_DEFAULTS);
    expect(modelOf(legacy.device)).toBe("detailed");
  });
  it("round-trips Simple calibration through persistence and current-forcing payloads", () => {
    const base = presetRoot(BUILTIN_META, "paper");
    const changed = clone(base);
    changed.device.model = "simple";
    changed.device.simple = { ...SIMPLE_DEFAULTS, beta_ref: 1.6, tau_body_s: 3e-7 };
    const restored = restoreParams(base, JSON.parse(JSON.stringify(changed)));
    expect(restored.device.calib).toEqual(base.device.calib);
    expect(restored.device.simple).toEqual(changed.device.simple);
    expect(sanitizeDevice(base.device, changed.device)).toEqual(restored.device);
    expect(branchesPayload(restored).device.model).toBe("simple");
    const cell = csvmPayload(restored, "deterministic", DEFAULT_CSVM).netlist.elements.find((e) => e.type === "STL");
    expect(cell?.type === "STL" && cell.device.simple).toEqual(changed.device.simple);
    expect(cell?.type === "STL" && cell.device.model).toBe("simple");
  });
  it("finds Simple cells even in heterogeneous circuits", () => {
    const simple = { ...BUILTIN_META.presets.paper.device, model: "simple" };
    expect(hasSimpleModel({netlist: {elements: [{ device: simple }]}})).toBe(true);
    expect(hasSimpleModel({netlist: {elements: [{device: BUILTIN_META.presets.paper.device}]}})).toBe(false);
    expect(hasSimpleModel({model: {polarity: "nmos"}})).toBe(false);
  });
  it("preserves other devices' model dictionaries during Detailed snapshot migration", () => {
    const original = { elements: [{type: "MOS", model: {Vth_V: .5}}, {type: "STL", device: BUILTIN_META.presets.paper.device}] };
    const migrated = legacyGeometryPayload(original) as typeof original;
    expect(migrated.elements[0]).toEqual(original.elements[0]);
    expect(migrated.elements[1].device).not.toHaveProperty("model");
    expect(migrated.elements[1].device).not.toHaveProperty("simple");
  });
  it("never re-labels Detailed export tables as Simple", () => {
    const base = presetRoot(BUILTIN_META, "paper");
    const selection = {device: {...base.device, model: "simple" as const}, sweep: base.sweep};
    expect(() => buildLtspiceExport(selection, {} as BranchesResult)).toThrow("simple");
    expect(() => buildVerilogAExport(selection, {} as BranchesResult)).toThrow("simple");
  });
  it("sanitizes unknown model identifiers and non-finite optional defaults", () => {
    expect(sanitizeDevice(BUILTIN_META.presets.paper.device, {model: "faster"}).model).toBe("detailed");
    expect(simpleOf({beta_ref: NaN, tau_body_s: 3e-7})).toEqual({...SIMPLE_DEFAULTS, tau_body_s: 3e-7});
  });
});

describe("HRS data input", () => {
  it("accepts scientific notation with optional headers and comma/tab delimiters", () => {
    expect(parseHrsPoints("VD_V, ID_A\n1.2, 2e-12\n1.4\t3e-12")).toEqual([{vd_V:1.2,id_A:2e-12},{vd_V:1.4,id_A:3e-12}]);
    expect(parseHrsPoints("  ")).toEqual([]);
  });
  it.each(["1,0", "-1,1e-12", "1,NaN", "1,2,3", "1,1nA", "1,1e-12\nbad,row"])("rejects malformed/non-positive rows: %s", (text) => {
    expect(() => parseHrsPoints(text)).toThrow();
  });
  it("rejects excess rows without silently discarding data", () => {
    expect(() => parseHrsPoints(Array(201).fill("1,1e-12").join("\n"))).toThrow("limit");
  });
});
