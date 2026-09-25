import { describe, expect, it } from "vitest";
import type { CustomCircuitResult, CustomEvent } from "../api/circuitCustom";
import { analyzeCsvm } from "./csvmMetrics";

function trace(t: (number | null)[], values: (number | null)[], events: CustomEvent[] = []): CustomCircuitResult {
  return {
    bench: "custom", mode: "deterministic", runtime_s: 0, warnings: [], summary: [],
    runs: [{ run: 0, t, signals: [{ key: "V(drain)", label: { ko: "V_D", en: "V_D" }, unit: "V", values }] }],
    events, solver_stats: { steps: t.length, rejected: 0, newton_iters: 0, runtime_s: 0 },
  };
}

/** Cold first peak is 7 V. Subsequent measured peaks/valleys are 4/2 V, period 2 s. */
function oscillator(stop = 12, withEvents = true) {
  const t = Array.from({ length: stop + 1 }, (_, i) => i);
  const voltage = t.map((time) => time === 0 ? 0 : time === 1 ? 7 : time % 2 === 1 ? 4 : 2);
  const events = withEvents ? t.slice(1).map((time): CustomEvent => ({
    run: 0, cell: "X1", t: time, kind: time % 2 === 1 ? "latch_up" : "latch_down",
    // Threshold crossing voltages deliberately differ from the sampled extrema.
    v_d: time % 2 === 1 ? 3.8 : 2.2,
  })) : [];
  return trace(t, voltage, events);
}

describe("CSVM transient measurements", () => {
  it("measures actual cycle extrema and periods, excluding cold startup", () => {
    expect(analyzeCsvm(oscillator())).toMatchObject({
      vTop_V: 4, vBottom_V: 2, frequency_Hz: 0.5, period_s: 2,
      observedCycles: 5, cycles: 4, status: "oscillating", source: "events",
      windowStart_s: 3, windowEnd_s: 11,
    });
  });

  it("excludes cycles starting before the settling window", () => {
    const measured = analyzeCsvm(oscillator(), { settleFraction: 0.5 });
    expect(measured.cycles).toBe(2);
    expect(measured.observedCycles).toBe(5);
    expect(measured.windowStart_s).toBe(7);
    expect(measured.frequency_Hz).toBe(0.5);
  });

  it("requires at least two retained full cycles before reporting a frequency", () => {
    expect(analyzeCsvm(oscillator(6))).toMatchObject({
      vTop_V: 4, vBottom_V: 2, frequency_Hz: null, period_s: null,
      cycles: 1, observedCycles: 2, status: "insufficient-cycles",
    });
    expect(analyzeCsvm(oscillator(6), { minCycles: 1 }).frequency_Hz).toBeNull();
  });

  it("does not present the excluded first peak as a steady cycle", () => {
    expect(analyzeCsvm(oscillator(4))).toMatchObject({
      vTop_V: null, vBottom_V: null, frequency_Hz: null, observedCycles: 1, cycles: 0,
      status: "insufficient-cycles",
    });
  });

  it("can measure prominent waveform cycles when no event records were supplied", () => {
    expect(analyzeCsvm(oscillator(12, false))).toMatchObject({
      source: "waveform", status: "oscillating", vTop_V: 4, vBottom_V: 2,
      frequency_Hz: 0.5, cycles: 4, observedCycles: 5,
    });
  });

  it("ignores an unfinished final cycle", () => {
    const measured = analyzeCsvm(oscillator(10));
    expect(measured.windowEnd_s).toBe(9);
    expect(measured.cycles).toBe(3);
    expect(measured.frequency_Hz).toBe(0.5);
  });

  it("does not count repeated up events without an intervening down event as oscillations", () => {
    const result = oscillator();
    result.events = result.events.filter((event) => event.kind === "latch_up");
    expect(analyzeCsvm(result)).toMatchObject({ status: "no-oscillation", observedCycles: 0, frequency_Hz: null });
  });

  it("reports a single latch as no observed oscillation, ignoring predicted summaries", () => {
    const result = trace([0, 1, 2, 3, 4], [0, 4, 2.7, 2.65, 2.65], [
      { run: 0, cell: "X1", t: 1, kind: "latch_up" },
    ]);
    result.summary = [{ key: "X1.f_osc", label: { ko: "예측", en: "Prediction" }, value: 123, unit: "Hz" }];
    expect(analyzeCsvm(result)).toMatchObject({ status: "no-oscillation", frequency_Hz: null });
  });

  it("does not confuse a constant voltage or numerical ripple with oscillation", () => {
    const t = Array.from({ length: 100 }, (_, i) => i * 1e-3);
    for (const values of [t.map(() => 3), t.map((_, i) => 3 + 1e-6 * Math.sin(i))]) {
      expect(analyzeCsvm(trace(t, values))).toMatchObject({ status: "no-oscillation", frequency_Hz: null });
    }
  });

  it("reports too few cycles for an incomplete charge-discharge transient", () => {
    expect(analyzeCsvm(oscillator(2))).toMatchObject({
      status: "insufficient-cycles", observedCycles: 0, frequency_Hz: null,
    });
  });

  it("does not bridge missing-data gaps when evaluating a complete cycle", () => {
    const result = oscillator();
    result.runs[0].signals[0].values[4] = null;
    const measured = analyzeCsvm(result);
    expect(measured.observedCycles).toBe(4);
    expect(measured.cycles).toBe(3);
    expect(measured.windowStart_s).toBe(5);
    expect(measured.frequency_Hz).toBe(0.5);
  });

  it("does not invent cycle extrema from sparse endpoints and dense latch events", () => {
    const events = oscillator().events;
    const endpoints = analyzeCsvm(trace([0, 12], [3, 3], events));
    expect(endpoints).toMatchObject({ vTop_V: null, vBottom_V: null, frequency_Hz: null });
    const sparse = analyzeCsvm(trace([0, 6, 12], [0, 4, 3], events));
    expect(sparse).toMatchObject({
      status: "insufficient-cycles", cycles: 0, vTop_V: null, vBottom_V: null, frequency_Hz: null,
      reason: { en: "Waveform resolution too low" },
    });
  });

  it("requires resolved voltage swing even when repeated latch events exist", () => {
    const result = oscillator();
    result.runs[0].signals[0].values = result.runs[0].t.map(() => 3);
    expect(analyzeCsvm(result)).toMatchObject({
      status: "insufficient-cycles", cycles: 0, vTop_V: null, vBottom_V: null, frequency_Hz: null,
    });
  });

  it("selects the requested trace and ignores other runs and other cells", () => {
    const result = oscillator();
    result.runs.push({ ...result.runs[0], run: 8, signals: [{ ...result.runs[0].signals[0], values: result.runs[0].t.map(() => 3) }] });
    result.events.push(...result.events.map((event) => ({ ...event, run: 8, cell: "X2" })));
    expect(analyzeCsvm(result).frequency_Hz).toBe(0.5);
    expect(analyzeCsvm(result, { run: 8 })).toMatchObject({ status: "no-oscillation", frequency_Hz: null });
    expect(analyzeCsvm(result, { run: 7 }).status).toBe("no-data");
  });

  it("supports alternate probe keys and cell names", () => {
    const result = oscillator();
    result.runs[0].signals[0].key = "V(out)";
    result.events = result.events.map((event) => ({ ...event, cell: "X3" }));
    expect(analyzeCsvm(result).status).toBe("no-data");
    expect(analyzeCsvm(result, { drainKey: "V(out)", cell: "X3" }).frequency_Hz).toBe(0.5);
  });

  it("returns no data for invalid/missing voltage series", () => {
    const result = trace([0, 1, 2], [null, NaN, Infinity]);
    expect(analyzeCsvm(result).status).toBe("no-data");
    result.runs = [];
    expect(analyzeCsvm(result).status).toBe("no-data");
    expect(analyzeCsvm(trace([0, 2, 1], [0, 3, 2])).status).toBe("no-data");
  });

  it("averages measured periods rather than assuming uniform sample spacing", () => {
    const result = oscillator();
    result.runs[0].t = result.runs[0].t.map((t) => t! * t!);
    result.events = result.events.map((event) => ({ ...event, t: event.t ** 2 }));
    const measured = analyzeCsvm(result, { settleFraction: 0 });
    // Retained [3²,5²], [5²,7²], [7²,9²], [9²,11²]: mean period = 28 s.
    expect(measured.cycles).toBe(4);
    expect(measured.period_s).toBe(28);
    expect(measured.frequency_Hz).toBeCloseTo(1 / 28, 12);
  });
});
