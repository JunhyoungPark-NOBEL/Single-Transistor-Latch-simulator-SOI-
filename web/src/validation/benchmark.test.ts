import { describe, expect, it } from "vitest";
import { currentFloor, curveError, switchVoltage } from "./benchmark";

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
  it("leaves out samples at or below the floor when asked", () => {
    const model = { vd: [0, 1, 2, 3], id: [1e-15, 1e-14, 1e-9, 1e-6] };
    const measured = { vd: [0, 1, 2, 3], id: [4e-13, 4e-13, 1e-9, 1e-6] };
    const all = curveError(model, measured);
    expect(all.logN).toBe(4);
    expect(all.logRmse!).toBeGreaterThan(1);
    expect(curveError(model, measured, 1.2e-12)).toMatchObject({ n: 2, logN: 2, logRmse: 0 });
  });
});

describe("floor and switching voltages", () => {
  it("takes the median of the lowest fifth of the positive currents as the floor", () => {
    const low = Array.from({ length: 20 }, (_, i) => (i < 4 ? [3e-13, 4e-13, 5e-13, 6e-13][i] : 1e-9 * (i + 1)));
    expect(currentFloor(low)).toBe(4e-13);
    expect(currentFloor([0, -1, null as unknown as number])).toBeNull();
  });
  it("finds the latch-up and latch-down jumps of a double sweep", () => {
    const up = { vd: [3.6, 3.65, 3.66, 3.7], id: [1e-12, 2e-12, 1e-6, 2e-6] };
    const down = { vd: [3.0, 2.7, 2.69, 2.5], id: [1e-6, 5e-7, 1e-12, 1e-13] };
    expect(switchVoltage(up, "up")).toBeCloseTo(3.655, 6);
    expect(switchVoltage(down, "down")).toBeCloseTo(2.695, 6);
    // a model sweep jumps at the fold itself
    expect(switchVoltage({ vd: [3.7, 3.7037, 3.7037, 3.8], id: [1e-12, 2e-12, 1e-6, 1e-6] }, "up")).toBeCloseTo(3.7037, 6);
    expect(switchVoltage({ vd: [0, 1, 2], id: [1e-12, 2e-12, 3e-12] }, "up")).toBeNull();
  });
});
