import { describe, expect, it } from "vitest";
import { REFERENCE_GEOMETRY } from "../params/geometry";
import { GEOMETRY_LIVE_REQUIRED, geometryError, hasChangedGeometry, legacyGeometryPayload } from "./geometryPolicy";
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

describe("server model-domain messages", () => {
  const vbg = "geometry-domain-unavailable: back-gate coupling beyond the linear (depleted back-interface) range; reduce |V_BG| or EOT/Tbox (EOT/(Tbox + Tsi/3) x |V_BG| = 34.3 V > 2 V)";
  const short = "geometry-domain-unavailable: L = 120 nm fully depletes the lateral neutral base assumed by this compact model at Nbody = 2.3e+17 cm^-3 (L must exceed 153.6 nm); increase L or Nbody";
  const field = "geometry-domain-unavailable: the avalanche-field table for this Nbody (2e+18 cm^-3) ends at a reverse bias of 1.24 V (< 3 V); lower Nbody";

  it("names the limit that was crossed, in Korean and English", () => {
    expect(geometryError(vbg, "ko")).toBe("백게이트 결합이 선형 범위를 넘습니다(EOT/(Tbox + Tsi/3) × |V_BG| ≤ 2 V). |V_BG|나 EOT를 줄이거나 Tbox를 늘려 주세요.");
    expect(geometryError(vbg, "en")).toContain("beyond its linear range");
    expect(geometryError(short, "ko")).toBe("이 Nbody에서는 L이 너무 짧아 중성 바디가 남지 않습니다. L을 153.6 nm보다 길게 하거나 Nbody를 높여 주세요.");
    expect(geometryError(short, "en")).toBe("At this Nbody, L is too short to leave a neutral body. Make L longer than 153.6 nm or raise Nbody.");
    expect(geometryError(field, "ko")).toContain("Nbody를 낮춰 주세요");
    expect(geometryError("geometry-domain-unavailable: the specified Nbody is outside the finite avalanche-field domain of this compact model", "en")).toContain("Lower Nbody");
    expect(geometryError("geometry-domain-unavailable: this L/Nbody combination fully depletes the lateral neutral base", "en")).toContain("Increase L or raise Nbody");
    expect(geometryError(GEOMETRY_LIVE_REQUIRED, "ko")).toBe("Geometry·V_BG를 바꾼 계산은 계산 서버에 연결해야 실행됩니다.");
  });

  it("keeps the circuit element named by the server", () => {
    expect(geometryError(`X7: ${short}`, "en")).toBe("X7: At this Nbody, L is too short to leave a neutral body. Make L longer than 153.6 nm or raise Nbody.");
    expect(geometryError(`X2: ${vbg}`, "ko").startsWith("X2: 백게이트 결합이")).toBe(true);
    expect(geometryError("geometry-stochastic-unavailable: Geometry scaling currently supports deterministic VSCM/CSVM; use the reference geometry (X3)", "en"))
      .toBe("X3: Use deterministic mode for changed geometry or V_BG. The stochastic model is calibrated at the reference dimensions and V_BG=0.");
    expect(geometryError("run 0: DC operating point at t = 0 did not converge", "ko")).toBe("run 0: DC operating point at t = 0 did not converge");
  });

  it("translates the step-budget refusal without solver advice", () => {
    const osc = "circuit-step-budget: estimated ~7.35e+06 time steps per run exceed 2 x solver.max_steps = 2000000 (~515 s per run; relaxation oscillation: ~1797 predicted latch-up/latch-down cycles in t_stop, each resolved with several hundred steps, more with event-level carrier noise near the fold). Shorten t_stop, increase the capacitance or reduce the drive current (longer period), or raise solver.max_steps.";
    expect(geometryError(osc, "ko")).toBe("이 조건에서는 래치업·래치다운이 약 1,797번 반복되어 계산 스텝 한도를 넘습니다. 시간을 줄이거나, 커패시턴스를 키우거나, 전류를 줄여 주세요.");
    expect(geometryError(osc, "en")).not.toContain("max_steps");
    const long = "circuit-step-budget: estimated ~3.1e+06 time steps per run exceed 2 x solver.max_steps = 2000000 (~9 s per run; the waveforms are long)";
    expect(geometryError(long, "ko")).toBe("계산 스텝 수가 한도를 넘어 실행하지 않았습니다 (예상 약 3.1e+6 스텝). 시뮬레이션 시간이나 반복 수를 줄여 주세요.");
    expect(geometryError("circuit-step-budget: estimated total work ~5e+07 time steps (~60 min) exceeds the per-request limit 4e+07; reduce n_runs or t_stop", "en"))
      .toBe("Not run: the number of time steps is beyond the limit (about 5.0e+7 steps). Shorten the simulated time or reduce the repetitions.");
  });
});
