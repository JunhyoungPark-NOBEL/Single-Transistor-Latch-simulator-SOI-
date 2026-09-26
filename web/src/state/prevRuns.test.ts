import { describe, expect, it } from "vitest";
import { deltaText } from "../device/KpiStrip";
import { nextPrevRuns, type PrevRunsState } from "./prevRuns";
import type { ResultEntry, State } from "./store";

const done = (dataKey: string, data: unknown = { k: dataKey }): ResultEntry => ({ status: "done", kind: "branches", data, dataKey, progress: 1, message: "", token: 1 });
const running = (prev: ResultEntry): ResultEntry => ({ ...prev, status: "running" });
type S = Pick<State, "results" | "preset" | "mode">;
const st = (results: Record<string, ResultEntry>, preset: S["preset"] = "paper", mode: S["mode"] = "deterministic"): S => ({ results, preset, mode });
const empty: PrevRunsState = { cur: {}, prev: {} };

describe("previous-run baseline (delta chips, 이전 ghost)", () => {
  it("a completed run with another payload becomes current; the one before becomes prev", () => {
    const r1 = done("k1");
    let s = nextPrevRuns(empty, st({ branches: r1 }), st({}));
    expect(s.cur.branches?.dataKey).toBe("k1");
    expect(s.prev.branches).toBeUndefined();
    // re-running (running → done) with a new payload
    const r1b = running(r1);
    s = nextPrevRuns(s, st({ branches: r1b }), st({ branches: r1 }));
    expect(s.cur.branches?.dataKey).toBe("k1"); // a running entry changes nothing
    const r2 = done("k2");
    s = nextPrevRuns(s, st({ branches: r2 }), st({ branches: r1b }));
    expect(s.cur.branches?.dataKey).toBe("k2");
    expect(s.prev.branches?.dataKey).toBe("k1");
    // the same payload again keeps the baseline (no "변화 없음" after a plain re-run of the same point)
    const r2b = done("k2", { k: "k2", again: true });
    s = nextPrevRuns(s, st({ branches: r2b }), st({ branches: r2 }));
    expect(s.prev.branches?.dataKey).toBe("k1");
    expect(s.cur.branches?.data).toEqual({ k: "k2", again: true });
  });

  it("unrelated state changes return the same object; errors keep the baseline", () => {
    const r1 = done("k1");
    const s = nextPrevRuns(empty, st({ branches: r1 }), st({}));
    expect(nextPrevRuns(s, st({ branches: r1 }), st({ branches: r1 }))).toBe(s);
    const err: ResultEntry = { ...r1, status: "error", error: "x" };
    expect(nextPrevRuns(s, st({ branches: err }), st({ branches: r1 }))).toBe(s);
  });

  it("a preset or mode change resets the baseline; the result on screen does not become one", () => {
    const r1 = done("k1");
    const r2 = done("k2");
    let s = nextPrevRuns(empty, st({ branches: r1 }), st({}));
    s = nextPrevRuns(s, st({ branches: r2 }), st({ branches: r1 }));
    expect(s.prev.branches?.dataKey).toBe("k1");
    s = nextPrevRuns(s, st({ branches: r2 }, "photo"), st({ branches: r2 }));
    expect(s).toEqual({ cur: {}, prev: {} });
    const r3 = done("k3");
    s = nextPrevRuns(s, st({ branches: r3 }, "photo"), st({ branches: r2 }, "photo"));
    expect(s.cur.branches?.dataKey).toBe("k3");
    expect(s.prev.branches).toBeUndefined(); // no delta across presets
    s = nextPrevRuns(s, st({ branches: r3 }, "photo", "stochastic"), st({ branches: r3 }, "photo"));
    expect(s).toEqual({ cur: {}, prev: {} });
  });

  it("keeps branches and sweep_mc apart", () => {
    const b = done("b1");
    const m = { ...done("m1"), kind: "sweep_mc" };
    const s = nextPrevRuns(empty, st({ branches: b, sweep_mc: m }), null);
    expect(s.cur.branches?.dataKey).toBe("b1");
    expect(s.cur.sweep_mc?.dataKey).toBe("m1");
  });
});

describe("delta chip text", () => {
  it("▲/▼ with mV to one decimal, no change below 0.05 mV", () => {
    expect(deltaText(0.0801)).toEqual({ text: "▲ +80.1 mV", dir: "up" });
    expect(deltaText(-0.0012)).toEqual({ text: "▼ −1.2 mV", dir: "down" });
    expect(deltaText(0.00004)).toBeNull();
    expect(deltaText(1.2345)).toEqual({ text: "▲ +1.234 V", dir: "up" });
    expect(deltaText(NaN)).toBeNull();
  });
});
