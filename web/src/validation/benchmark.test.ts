import { describe, expect, it } from "vitest";
import { curveError } from "./benchmark";

describe("measured ID–VD comparison", () => {
  it("matches the same curve for both sweep directions and interpolates at measurement voltages", () => {
    for (const model of [{ vd: [0, 2], id: [0, 4] }, { vd: [2, 0], id: [4, 0] }]) {
      expect(curveError(model, { vd: [0, 1, 2], id: [0, 2, 4] })).toEqual({ n: 3, logN: 2, logRmse: 0, normalizedRmsePct: 0 });
    }
  });
  it("reports a factor-of-ten error as one decade, with independent linear normalization", () => {
    const e = curveError({ vd: [1, 2], id: [10, 100] }, { vd: [1, 2], id: [1, 10] });
    expect(e.logRmse).toBe(1);
    expect(e.normalizedRmsePct).toBeCloseTo(100 * Math.sqrt((9 ** 2 + 90 ** 2) / 2) / 9);
  });
  it("does not extrapolate, manufacture data, or divide by a zero current span", () => {
    expect(curveError({ vd: [1, 2], id: [0, 0] }, { vd: [0, 1, 2, 3, null], id: [5, 0, 0, 7, 1] })).toEqual({ n: 2, logN: 0, logRmse: null, normalizedRmsePct: null });
    expect(curveError({ vd: [], id: [] }, { vd: [1], id: [1] }).n).toBe(0);
  });
  it("keeps signed currents for linear errors but excludes them from log errors", () => {
    expect(curveError({ vd: [0, 1], id: [-1, 1] }, { vd: [0, 1], id: [-1, 1] })).toEqual({ n: 2, logN: 1, logRmse: 0, normalizedRmsePct: 0 });
  });
});
