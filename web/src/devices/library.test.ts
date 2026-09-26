import { afterEach, describe, expect, it } from "vitest";
import { BUILTIN_META } from "../state/presets";
import { builtinDevices, calibPart, exportLibraryJson, geometryLine, isSupportedTechnology, parseLibraryJson, uniqueName, validateDevice } from "./library";
import { parseStoredDevices, useDeviceLib, validationBase } from "./store";
import { REFERENCE_GEOMETRY } from "../params/geometry";
import { stlRefFor } from "../schematic/store";
import { defaultDoc, exportDocJson, parseDoc } from "../schematic/persist";
import { loadDeviceIntoParams } from "./DeviceCard";
import { useStore } from "../state/store";
import { buildRequest } from "../schematic/netlist";
import { extractNets } from "../schematic/nets";
import { runErc } from "../schematic/erc";

afterEach(() => {
  useDeviceLib.setState({ devices: [] });
  useStore.getState().loadPreset("paper");
});

describe("device library", () => {
  const base = validationBase();
  it("derives read-only built-ins from the presets", () => {
    const b = builtinDevices(BUILTIN_META);
    expect(b.map((d) => d.id)).toEqual(["builtin:paper"]);
    expect(b[0].name).toBe("Device 1");
    expect(b[0].label).toEqual({ ko: "Device 1", en: "Device 1" });
    expect(b.every((d) => d.builtin && d.technology === "FDSOI")).toBe(true);
    expect(b[0].device).toEqual(BUILTIN_META.presets.paper.device);
    expect(builtinDevices(BUILTIN_META, ["photo"])[0].device.vg).toBeCloseTo(-1.8, 9);
    expect(geometryLine(b[0].geometry)).toBe("L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm · T_BOX 140 nm · N_body 2.296e+17 cm⁻³");
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
    expect(validateDevice({ name: "Unknown technology", technology: "unknown", device: base.device }, base)).toBeNull();
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
  it.each(["PDSOI", "Bulk"])("preserves imported %s records without loading, placement or FDSOI conversion", (technology) => {
    const raw = { id: `unsupported-${technology}`, name: technology, technology, device: { ...base.device, vg: -1.23 } };
    const imported = parseLibraryJson(JSON.stringify(raw), base).devices[0];
    expect(imported.technology).toBe(technology);
    expect(isSupportedTechnology(imported.technology)).toBe(false);
    expect(parseStoredDevices(exportLibraryJson([imported]))[0].technology).toBe(technology);
    useDeviceLib.getState().importMany([imported]);
    const before = useStore.getState().params;
    expect(loadDeviceIntoParams(imported)).toBe(false);
    expect(useStore.getState().params).toBe(before);
    expect(() => stlRefFor(imported.id)).toThrow("Only FDSOI is supported");
    useDeviceLib.getState().update(imported.id, { technology: "FDSOI", device: base.device });
    expect(useDeviceLib.getState().devices[0].technology).toBe(technology);
    expect(useDeviceLib.getState().devices[0].device.vg).toBe(-1.23);
    useDeviceLib.getState().rename(imported.id, "Archived device");
    expect(useDeviceLib.getState().devices[0].name).toBe("Archived device");
  });

  it.each(["PDSOI", "Bulk"])("retains unsupported %s circuit snapshots but blocks compilation and running", (technology) => {
    const d = defaultDoc();
    d.elements = [{ id: "stl-unsupported", name: "X1", kind: "STL", x: 0, y: 0, rot: 0, stl: { libId: "external", name: "Imported device", technology, device: base.device } }];
    const restored = parseDoc(JSON.parse(exportDocJson(d)))!;
    expect(restored.elements[0].stl?.technology).toBe(technology);
    const conn = extractNets(restored);
    expect(runErc(restored, conn)).toContainEqual(expect.objectContaining({ code: "unsupportedTechnology", level: "error", elementIds: ["stl-unsupported"] }));
    expect(() => buildRequest(restored, conn, "deterministic", null)).toThrow("Only FDSOI is supported");
  });

  it("makes names unique", () => {
    expect(uniqueName("A", ["a"])).toBe("A (2)");
    expect(uniqueName("A (2)", ["A", "A (2)"])).toBe("A (3)");
    expect(uniqueName("B", ["A"])).toBe("B");
  });

  it("retains all six active dimensions through save, export/import, circuit placement and reload", () => {
    const geometry = { Lg_nm: 300, W_nm: 400, Tsi_nm: 35, EOT_nm: 10, Tbox_nm: 100, Nbody_cm3: 3e17 };
    const source = builtinDevices(BUILTIN_META)[0];
    const saved = useDeviceLib.getState().add({ ...source, id: "scaled", name: "Scaled", device: { ...source.device, geometry, vbg: -1.5 } });
    expect(saved.geometry).toEqual(geometry);
    const imported = parseLibraryJson(exportLibraryJson([saved]), base).devices[0];
    expect(imported.geometry).toEqual(geometry);
    expect(imported.device.geometry).toEqual(geometry);
    expect(imported.device.vbg).toBe(-1.5);
    loadDeviceIntoParams(imported);
    expect(useStore.getState().params.device.geometry).toEqual(geometry);
    expect(useStore.getState().params.device.vbg).toBe(-1.5);
    const placed = stlRefFor(saved.id);
    expect(placed.device.geometry).toEqual(geometry);
    expect(placed.device.vbg).toBe(-1.5);
    const doc = defaultDoc();
    doc.elements = [{ id: "stl-1", kind: "STL", name: "U1", x: 0, y: 0, rot: 0, stl: placed }];
    const roundTrip = parseDoc(JSON.parse(exportDocJson(doc)))!;
    expect(roundTrip.elements[0].stl!.device.geometry).toEqual(geometry);
    expect(roundTrip.elements[0].stl!.device.vbg).toBe(-1.5);
    // A placed device owns its snapshot; changing the library does not resize existing circuits.
    const nextGeometry = { ...geometry, Lg_nm: 250 };
    useDeviceLib.getState().update(saved.id, { device: { ...saved.device, geometry: nextGeometry } });
    expect(useDeviceLib.getState().devices[0].geometry).toEqual(nextGeometry);
    expect(placed.device.geometry).toEqual(geometry);
  });

  it("migrates legacy descriptive geometry to the actual reference behavior", () => {
    const legacyDevice = { ...base.device };
    delete legacyDevice.geometry;
    delete legacyDevice.vbg;
    const legacy = validateDevice({ name: "Legacy", device: legacyDevice, geometry: { Lg_nm: 120, W_nm: 1234, Tsi_nm: 12, EOT_nm: 4 } }, base)!;
    expect(legacy.device.geometry).toEqual(REFERENCE_GEOMETRY);
    expect(legacy.device.vbg).toBe(0);
    expect(legacy.geometry).toEqual(REFERENCE_GEOMETRY);
    const doc = defaultDoc();
    doc.elements = [{ id: "old", kind: "STL", name: "U1", x: 0, y: 0, rot: 0, stl: { libId: "old", name: "Legacy", device: legacyDevice } }];
    expect(parseDoc(doc)!.elements[0].stl!.device.geometry).toEqual(REFERENCE_GEOMETRY);
    expect(parseDoc(doc)!.elements[0].stl!.device.vbg).toBe(0);
  });

  it("restores invalid saved back-gate values to zero without changing valid bias", () => {
    for (const vbg of [Infinity, -Infinity, NaN, 11, -11, "2", null]) {
      expect(validateDevice({ name: "Invalid VBG", device: { ...base.device, vbg } }, base)!.device.vbg).toBe(0);
    }
    expect(validateDevice({ name: "Valid VBG", device: { ...base.device, vbg: 2 } }, base)!.device.vbg).toBe(2);
  });
});
