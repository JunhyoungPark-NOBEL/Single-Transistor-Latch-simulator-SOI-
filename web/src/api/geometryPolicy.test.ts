import { describe, expect, it } from "vitest";
import { REFERENCE_GEOMETRY } from "../params/geometry";
import { GEOMETRY_LIVE_REQUIRED, hasChangedGeometry, legacyGeometryPayload } from "./geometryPolicy";
import { createMockBackend } from "./mock";
import { Snapshot, createSnapshotBackend, snapshotKey } from "./snapshot";

describe("geometry provenance", () => {
  const legacy = { device: { vg: -2, calib: { beta: 7 } }, sweep: { vd_max_V: 4 } };
  const baseline = { ...legacy, device: { ...legacy.device, vbg: 0, geometry: { ...REFERENCE_GEOMETRY } } };
  const changed = { ...baseline, device: { ...baseline.device, geometry: { ...REFERENCE_GEOMETRY, Lg_nm: 300 } } };

  it("detects changed dimensions in individual devices and heterogeneous circuits", () => {
    expect(hasChangedGeometry(baseline)).toBe(false);
    expect(hasChangedGeometry({ netlist: { elements: [{ device: baseline.device }, { device: changed.device }] } })).toBe(true);
    expect(hasChangedGeometry({ device: { vg: -2, geometry: { Tsi_nm: -1 } } })).toBe(true);
    expect(hasChangedGeometry({ device: { vg: -2, vbg: 1, geometry: REFERENCE_GEOMETRY } })).toBe(true);
  });

  it("keeps exact request hashes and permits only a reference-equivalent legacy lookup", () => {
    expect(snapshotKey("branches", baseline)).not.toBe(snapshotKey("branches", legacy));
    expect(legacyGeometryPayload(baseline)).toEqual(legacy);
    expect(legacyGeometryPayload(changed)).toEqual({ ...changed, device: { vg: changed.device.vg, calib: changed.device.calib, geometry: changed.device.geometry } });
    expect(legacyGeometryPayload({ device: { ...baseline.device, vbg: 1 } })).toEqual({ device: { vg: baseline.device.vg, calib: baseline.device.calib, vbg: 1 } });
    expect(baseline.device.geometry).toEqual(REFERENCE_GEOMETRY);
  });

  it("reads authentic legacy baseline snapshots but never serves them for another geometry", async () => {
    const snap = new Snapshot({ format: 1, created: "2026-09-25", data: {}, compute: [{ kind: "branches", key: snapshotKey("branches", legacy), file: "baseline.json", bytes: 2, label: "baseline" }] }, "snap/", async () => new Response('{"source":"recorded-reference"}'));
    const backend = createSnapshotBackend(snap, { fallback: () => { throw new Error("must not use fallback"); } });
    expect((await backend.submit("branches", baseline)).result).toEqual({ source: "recorded-reference" });
    await expect(backend.submit("branches", changed)).rejects.toThrow(GEOMETRY_LIVE_REQUIRED);
  });

  it("refuses fabricated geometry results in demo mode", async () => {
    const mock = createMockBackend(0);
    const result = await mock.submit("branches", changed);
    expect(result.status).toBe("error");
    expect(result.error).toBe(GEOMETRY_LIVE_REQUIRED);
  });
});
