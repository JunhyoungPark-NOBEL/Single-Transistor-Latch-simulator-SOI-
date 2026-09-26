import { describe, expect, it } from "vitest";
import type { SummaryItem } from "../api/types";
import { pickSummary, summaryColor } from "./SummaryStrip";
import { fmtCoefValue, splitSpread, splitUnit } from "./summary";

const L = { ko: "", en: "" };
const item = (key: string, value: number | string = 1): SummaryItem => ({ key, label: L, value });

describe("result summary: at most four cells by priority", () => {
  it("quick benches: V_LU, V_LD, window first; run metadata never takes a cell", () => {
    const items = ["runs", "n_latch_up", "window", "V_LD", "fold_V_LU", "V_LU", "steps_per_run", "lag_LU"].map((k) => item(k));
    const [shown, rest] = pickSummary(items, ["V_LU", "V_LD", "window", "p_any_lu", "P1", "vd_first_lu", "n_latch_up", "final_state"]);
    expect(shown.map((s) => s.key)).toEqual(["V_LU", "V_LD", "window", "n_latch_up"]);
    expect(rest.map((s) => s.key).sort()).toEqual(["fold_V_LU", "lag_LU", "runs", "steps_per_run"]);
  });

  it("fills up to four with the server order when few priority keys exist", () => {
    const items = ["P_sw", "delay", "amplitude", "fold_V_LU", "P_retained"].map((k) => item(k));
    const [shown, rest] = pickSummary(items, ["V_LU", "V_LD", "P_sw"]);
    expect(shown.map((s) => s.key)).toEqual(["P_sw", "delay", "amplitude", "fold_V_LU"]);
    expect(rest).toHaveLength(1);
  });

  it("schematic: element-scoped keys follow the first STL; *.p_fire matches any comparator", () => {
    const items = ["X2.n_latch_up", "X1.n_latch_down", "X1.n_latch_up", "X1.t_first_lu", "X1.vd_first_lu", "CMP1.p_fire", "X1.p_any_lu"].map((k) => item(k));
    const [shown] = pickSummary(items, ["n_latch_up", "*.p_fire", "f_osc", "vd_first_lu", "p_any_lu", "final_state"], 4, "X1");
    expect(shown.map((s) => s.key)).toEqual(["X1.n_latch_up", "CMP1.p_fire", "X1.vd_first_lu", "X1.p_any_lu"]);
  });

  it("p-bit: the latched fraction yields its cell when it equals P(1)", () => {
    const items = [item("P1", 0.487), item("P_latched", 0.4871), item("lag1", -0.0003), item("n_bits", 300), item("v_th", 0.1)];
    const [shown, rest] = pickSummary(items, ["V_LU", "V_LD", "window", "p_any_lu", "P1"]);
    expect(shown.map((s) => s.key)).toEqual(["P1", "lag1", "n_bits", "v_th"]);
    expect(rest.map((s) => s.key)).toEqual(["P_latched"]);
    const [shown2] = pickSummary([item("P1", 0.487), item("P_latched", 0.6)], ["P1"]);
    expect(shown2.map((s) => s.key)).toEqual(["P1", "P_latched"]);
  });

  it("formats coefficients and spreads without noise digits", () => {
    expect(fmtCoefValue(-0.000278)).toBe("≈ 0");
    expect(fmtCoefValue(-0.071)).toBe("−0.07");
    expect(splitSpread(0.0532, "1")).toEqual({ value: "0.053", unit: "" });
    expect(splitSpread(1.68e-8, "V")).toBeNull();
    expect(splitSpread(0.119, "V")).toEqual({ value: "119", unit: "mV" });
    expect(splitUnit(-0.5, "V")).toEqual({ value: "−0.500", unit: "V" });
  });

  it("colours latch-up voltages blue (--hrs) and latch-down voltages red (--lrs)", () => {
    expect(summaryColor("V_LU")).toBe("var(--hrs)");
    expect(summaryColor("X1.vd_first_lu")).toBe("var(--hrs)");
    expect(summaryColor("V_LD_src")).toBe("var(--lrs)");
    expect(summaryColor("X1.n_latch_up")).toBeUndefined();
  });
});
