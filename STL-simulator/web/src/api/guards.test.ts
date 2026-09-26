import { describe, expect, it } from "vitest";
import { BUILTIN_META } from "../state/presets";
import { presetRoot, sweepMcPayload } from "../utils/payload";
import { checkResult, finite } from "./guards";
import { normalizeDesignMap, normalizeMeasured } from "./measured";
import {
  mockBranches, mockChargeBalance, mockCircuit, mockDesignMapRaw, mockHazard, mockMeasuredRaw, mockSweepMC, mockValidation, mockVgCurve,
  mockVgStoch, statsOf,
} from "./mock";

const paper = presetRoot(BUILTIN_META, "paper");

describe("result-shape guards", () => {
  it("accept the mock fixtures (which follow the contract shapes)", () => {
    expect(checkResult("branches", mockBranches({ device: paper.device, sweep: paper.sweep }))).toEqual([]);
    expect(checkResult("charge_balance", mockChargeBalance({ device: paper.device, vd: 3.1 }))).toEqual([]);
    expect(checkResult("vg_curve", mockVgCurve({ device: paper.device }))).toEqual([]);
    expect(checkResult("hazard", mockHazard({ device: paper.device, sweep: paper.sweep }))).toEqual([]);
    expect(checkResult("sweep_mc", mockSweepMC(sweepMcPayload(paper)))).toEqual([]);
    expect(checkResult("vg_curve_stochastic", mockVgStoch({ device: paper.device }))).toEqual([]);
    expect(checkResult("validation", mockValidation({ level: "fast" }))).toEqual([]);
    for (const bench of ["load_line", "pulse", "pbit", "coupled"])
      for (const mode of ["deterministic", "stochastic"]) expect(checkResult("circuit", mockCircuit({ bench, mode, device: paper.device }))).toEqual([]);
  });
  it("report missing keys", () => {
    expect(checkResult("branches", { HRS: { vd: [], id: [] } })).toContain("folds");
    expect(checkResult("sweep_mc", null)).toEqual(["<result is not an object>"]);
    expect(checkResult("validation", { checks: "x" })).toEqual(["checks"]);
  });
  it("finite filters nulls/NaN", () => {
    expect(finite([1, null, NaN, 2, undefined])).toEqual([1, 2]);
  });
});

describe("mock fixtures", () => {
  it("paper device folds at 3.70 / 2.60 V", () => {
    const b = mockBranches({ device: paper.device, sweep: paper.sweep });
    expect(b.latch).toBe(true);
    expect(b.folds.V_LU).toBeCloseTo(3.7037, 3);
    expect(b.folds.V_LD).toBeCloseTo(2.5979, 3);
  });
  it("stats match a direct computation", () => {
    const s = statsOf([1, 2, 3, null, 4]);
    expect(s.n).toBe(4);
    expect(s.censored).toBe(1);
    expect(s.mean).toBe(2.5);
    expect(s.median).toBe(2.5);
    expect(s.sd).toBeCloseTo(1.2909944, 6);
  });
  it("sweep_mc is reproducible for a seed", () => {
    const a = mockSweepMC(sweepMcPayload(paper));
    const b = mockSweepMC(sweepMcPayload(paper));
    expect(a.V_LU).toEqual(b.V_LU);
  });
});

describe("data normalisers", () => {
  it("measured (backend shape → panels)", () => {
    const m = normalizeMeasured(mockMeasuredRaw());
    expect(m.photo).toHaveLength(8);
    expect(m.photo[0]).toMatchObject({ vg: -1.8, power_mW: 0 });
    expect(m.photo[0].sd_mV).toBeCloseTo(173.2, 1);
    expect(m.photo[0].raw?.length).toBe(400);
    expect(m.light_iv).toHaveLength(6);
    expect(m.paper_iv?.median_up.length).toBeGreaterThan(10);
    expect(normalizeMeasured(null)).toEqual({ photo: [], light_iv: [], paper_iv: null });
  });
  it("design map", () => {
    const d = normalizeDesignMap(mockDesignMapRaw());
    expect(d.length_nm.length).toBeGreaterThan(2);
    expect(d.fields.sigma_VLU_mV.length).toBe(d.depth_fraction.length);
    expect(d.fields.sigma_VLU_mV[0].length).toBe(d.length_nm.length);
    expect(d.lines?.Nt).toHaveLength(4);
    expect(d.scalars.device_sigma_phi_mV).toBeCloseTo(153.39, 2);
  });
});
