import { describe, expect, it } from "vitest";
import { csvmPayload, DEFAULT_CSVM } from "../device/forcing";
import { restoreParams } from "../state/persist";
import { BUILTIN_META } from "../state/presets";
import { canonical, clone } from "../utils/object";
import { branchesPayload, presetRoot, sweepMcPayload, vgCurvePayload } from "../utils/payload";
import { GEOMETRY_KEYS, GEOMETRY_LIMITS, isReferenceGeometry, REFERENCE_GEOMETRY, resolveBackGate, resolveGeometry, usesGeometryModel } from "./geometry";

describe("geometry payload and migration", () => {
  it("restores legacy settings to the exact calibrated geometry without rounding the doping", () => {
    const base = presetRoot(BUILTIN_META, "paper");
    const legacy = clone(base);
    delete legacy.device.geometry;
    delete legacy.device.vbg;
    expect(restoreParams(base, legacy, 2).device.geometry).toEqual(REFERENCE_GEOMETRY);
    expect(restoreParams(base, legacy, 2).device.vbg).toBe(0);
    expect(REFERENCE_GEOMETRY.Nbody_cm3).toBe(2.295773162796593e17);
    expect(isReferenceGeometry(undefined)).toBe(true);
  });

  it("nonzero back-gate bias uses the extension even at the reference geometry and reaches forcing requests", () => {
    const original = presetRoot(BUILTIN_META, "paper");
    const edited = clone(original);
    edited.device.vbg = 2;
    expect(isReferenceGeometry(edited.device.geometry)).toBe(true);
    expect(usesGeometryModel(edited.device)).toBe(true);
    expect(usesGeometryModel(original.device)).toBe(false);
    expect(branchesPayload(edited).device.vbg).toBe(2);
    expect(canonical(csvmPayload(edited, "deterministic", DEFAULT_CSVM))).not.toBe(canonical(csvmPayload(original, "deterministic", DEFAULT_CSVM)));
    for (const invalid of [undefined, NaN, Infinity, -11, 11, "2"]) expect(resolveBackGate(invalid)).toBe(0);
    expect(resolveBackGate(-2)).toBe(-2);
  });

  it("restores invalid dimensions independently and preserves valid values", () => {
    const base = presetRoot(BUILTIN_META, "paper");
    const stored = clone(base);
    stored.device.geometry = { ...REFERENCE_GEOMETRY, Lg_nm: 300, Tsi_nm: -1, Nbody_cm3: 1e30, W_nm: NaN };
    expect(restoreParams(base, stored).device.geometry).toEqual({ ...REFERENCE_GEOMETRY, Lg_nm: 300 });
    expect(resolveGeometry({ EOT_nm: 10 })).toEqual({ ...REFERENCE_GEOMETRY, EOT_nm: 10 });
  });

  it.each(GEOMETRY_KEYS)("%s reaches voltage, Monte Carlo and current-forcing requests and changes their cache keys", (key) => {
    const original = presetRoot(BUILTIN_META, "paper");
    const edited = clone(original);
    edited.device.geometry![key] += GEOMETRY_LIMITS[key].step;
    const requests = [
      branchesPayload,
      sweepMcPayload,
      (p: typeof original) => vgCurvePayload(p, { min: -4, max: -1, n: 9 }),
      (p: typeof original) => csvmPayload(p, "deterministic", DEFAULT_CSVM),
    ];
    for (const request of requests) expect(canonical(request(edited))).not.toBe(canonical(request(original)));
    expect(branchesPayload(edited).device.geometry?.[key]).toBe(edited.device.geometry![key]);
    expect(isReferenceGeometry(edited.device.geometry)).toBe(false);
    expect(original.device.geometry).toEqual(REFERENCE_GEOMETRY);
  });
});
