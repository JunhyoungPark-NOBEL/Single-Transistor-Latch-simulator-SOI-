import { describe, expect, it } from "vitest";
import { BUILTIN_META } from "../state/presets";
import { builtinDevices, calibPart, exportLibraryJson, geometryLine, parseLibraryJson, uniqueName, validateDevice } from "./library";
import { parseStoredDevices, validationBase } from "./store";

describe("device library", () => {
  const base = validationBase();
  it("derives read-only built-ins from the presets", () => {
    const b = builtinDevices(BUILTIN_META);
    expect(b.map((d) => d.id)).toEqual(["builtin:paper", "builtin:photo"]);
    expect(b.every((d) => d.builtin && d.technology === "FDSOI")).toBe(true);
    expect(b[1].device.vg).toBeCloseTo(-1.8, 9);
    expect(geometryLine(b[0].geometry)).toBe("L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm");
    expect(calibPart("FDSOI · L_g 500 nm — reference calibration (dark)")).toBe("Reference calibration (dark)");
    expect(calibPart("Custom")).toBe("Custom");
  });
  it("validates stored/imported devices field by field", () => {
    const good = { id: "d1", name: "  A  ", device: { ...base.device, vg: -1.9, light: { ...base.device.light, mode: "bogus" } }, stochastic: { local_state: { mode: "weird", sigma: 0.2 } }, geometry: { Lg_nm: -5 } };
    const d = validateDevice(good, base)!;
    expect(d.name).toBe("A");
    expect(d.device.vg).toBe(-1.9);
    expect(d.device.light.mode).toBe(base.device.light.mode); // enum restored
    expect(d.stochastic.local_state.mode).toBe(base.stochastic.local_state.mode);
    expect(d.stochastic.local_state.sigma).toBe(0.2);
    expect(d.geometry.Lg_nm).toBe(500);
    expect(validateDevice({ name: "x" }, base)).toBeNull();
    expect(validateDevice({ name: "", device: {} }, base)).toBeNull();
    expect(validateDevice({ id: "builtin:paper", name: "copy", device: {} }, base)!.id).not.toBe("builtin:paper");
  });
  it("imports library files, bare arrays and single devices; rejects junk", () => {
    const d = validateDevice({ id: "d1", name: "A", device: base.device }, base)!;
    const text = exportLibraryJson([d]);
    const r = parseLibraryJson(text, base);
    expect(r.error).toBeUndefined();
    expect(r.devices[0].name).toBe("A");
    expect(parseLibraryJson(JSON.stringify([d, { junk: 1 }]), base)).toMatchObject({ skipped: 1 });
    expect(parseLibraryJson(JSON.stringify(d), base).devices.length).toBe(1);
    expect(parseLibraryJson("{nope", base).error).toBe("json");
    expect(parseLibraryJson('{"a":1}', base).error).toBe("format");
    expect(parseStoredDevices(JSON.stringify({ v: 1, devices: [d, d] })).map((x) => x.id)[0]).toBe("d1");
    expect(new Set(parseStoredDevices(JSON.stringify({ v: 1, devices: [d, d] })).map((x) => x.id)).size).toBe(2);
    expect(parseStoredDevices("garbage")).toEqual([]);
  });
  it("makes names unique", () => {
    expect(uniqueName("A", ["a"])).toBe("A (2)");
    expect(uniqueName("A (2)", ["A", "A (2)"])).toBe("A (3)");
    expect(uniqueName("B", ["A"])).toBe("B");
  });
});
