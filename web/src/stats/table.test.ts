import { describe as suite, expect, it } from "vitest";
import { describe } from "./describe";
import { fmtCoef, fmtLevel, fmtP, fmtPct, fmtShare, fmtSpread, levelDecimals, planUnits } from "./format";
import { ALL_COLUMNS, cellValue, COLUMN_GROUP, COMPACT_COLUMNS, computeRow, computeRows, GROUP_COLUMNS, statsCsv, visibleColumns } from "./table";

suite("unit plans and cell formatting", () => {
  it("voltages: levels in V, spreads in mV, level decimals follow the spread (≥ mV resolution)", () => {
    const p = planUnits("V", 3.8, 0.174);
    expect([p.levelUnit, p.spreadUnit, p.levelFactor, p.spreadFactor]).toEqual(["V", "mV", 1, 1e-3]);
    expect(levelDecimals(0.174, p)).toBe(3); // 3.808
    expect(levelDecimals(0.0055, p)).toBe(4); // 3.8028
    expect(levelDecimals(0.0000123, p)).toBe(6);
    expect(levelDecimals(null, p)).toBeNull();
    expect(fmtLevel(3.80783861, p, 3)).toBe("3.808");
    expect(fmtLevel(-0.03, p, 3)).toBe("−0.030");
    expect(fmtLevel(3.80783861, p, null)).toBe("3.808"); // 4 significant digits
    expect(fmtSpread(0.17380106, p)).toBe("173.8");
    expect(fmtSpread(0.011192, p)).toBe("11.2");
    expect(fmtSpread(0.00548, p)).toBe("5.48");
    expect(fmtSpread(0.00178, p, true)).toBe("+1.78");
    expect(fmtSpread(-0.0111, p, true)).toBe("−11.1");
    expect(fmtSpread(0, p)).toBe("0");
    expect(fmtSpread(null, p)).toBe("—");
    // a spread of ≥ 1 V is printed in V; a level below 10 mV gets a prefix
    expect(planUnits("V", 3.8, 1.5).spreadUnit).toBe("V");
    expect(planUnits("V", 0.004, 0.0002).levelUnit).toBe("mV");
  });
  it("other SI units choose prefixes from the magnitudes; dimensionless units print plainly", () => {
    const s = planUnits("s", 1.2e-4, 3e-6);
    expect([s.levelUnit, s.spreadUnit]).toEqual(["µs", "µs"]);
    expect(fmtLevel(1.234e-4, s, levelDecimals(3e-6, s))).toBe("123.4");
    const a = planUnits("A", 2e-9, 5e-12);
    expect([a.levelUnit, a.spreadUnit]).toEqual(["nA", "pA"]);
    const one = planUnits("1", 0.5, 0.1);
    expect([one.levelUnit, one.spreadUnit, one.levelFactor]).toEqual(["", "", 1]);
    expect(planUnits("%", 50, 3).levelUnit).toBe("%");
  });
  it("coefficients, percentages and p-values", () => {
    expect(fmtPct(0.0456)).toBe("4.56");
    expect(fmtCoef(-0.0309)).toBe("−0.03");
    expect(fmtCoef(0.05, 3)).toBe("0.050");
    expect(fmtP(0.6994)).toBe("0.70");
    expect(fmtP(0.0366)).toBe("0.037");
    expect(fmtP(0.0004)).toBe("<0.001");
    expect(fmtP(null)).toBe("—");
    expect(fmtShare(0)).toBe("0 %");
    expect(fmtShare(0.01)).toBe("1.0 %");
    expect(fmtShare(0.91)).toBe("91 %");
    expect(fmtShare(0.0004)).toBe("0.04 %");
  });
});

suite("table model", () => {
  it("columns and groups", () => {
    expect(ALL_COLUMNS.length).toBe(Object.values(GROUP_COLUMNS).flat().length);
    expect(COLUMN_GROUP.ks_p).toBe("compare");
    expect(COLUMN_GROUP.lag1).toBe("sequence");
    // the comparison group follows centre and spread
    expect(ALL_COLUMNS.indexOf("dmean")).toBeLessThan(ALL_COLUMNS.indexOf("p05"));
  });
  it("computeRow: describe + measured comparison (Δmean, SD ratio, KS)", () => {
    const r = computeRow({ key: "V_LU", label: "V_LU", values: [3.6, 3.7, null, 3.8, 3.9], measured: [3.5, 3.6, 3.7, 3.8] });
    expect(r.unit).toBe("V");
    expect(r.d.n).toBe(4);
    expect(r.d.censored).toBe(1);
    expect(r.m?.n).toBe(4);
    expect(r.cmp?.dmean).toBeCloseTo(0.1, 12);
    expect(r.cmp?.sd_ratio).toBeCloseTo(1, 12);
    expect(r.cmp?.ks_d).toBeCloseTo(0.25, 12);
    expect(cellValue(r, "m_mean")).toBeCloseTo(3.65, 12);
    expect(cellValue(r, "ks_d", "measured")).toBeNull();
    expect(cellValue(r, "mean", "measured")).toBeCloseTo(3.65, 12);
    expect(cellValue(r, "ci95")).toBe(r.d.ci95_half);
    expect(cellValue(r, "censored")).toBe(1);
  });
  it("measured by key (contract §9 form), scale, and precomputed analytic rows", () => {
    const rows = computeRows(
      [
        { key: "a", label: "a", values: [1, 2, 3], scale: 1e3, unit: "mV" },
        { key: "h", label: "h", stats: { mean: 3.8, sd: 0.005, q1: 3.79, q3: 3.8 }, secondary: true },
      ],
      { a: [1, 2, 4] },
    );
    expect(rows[0].d.mean).toBeCloseTo(2000, 9);
    expect(rows[0].m?.mean).toBeCloseTo(2333.333333, 5);
    expect(rows[1].d.iqr).toBeCloseTo(0.01, 12); // derived from q1/q3
    expect(rows[1].d.cv).toBeCloseTo(0.005 / 3.8, 12); // derived from sd/mean
    expect(Number.isNaN(rows[1].d.n)).toBe(true); // unknown count → shown as "—"
    expect(cellValue(rows[1], "n")).toBeNull();
    expect(rows[1].cmp).toBeNull();
  });
  it("statsCsv: model + measured rows, full precision in the row unit", () => {
    const rows = computeRows([{ key: "V_LU", label: "V_LU", values: [3.6, 3.7, 3.8, 3.9], measured: [3.61, 3.72, 3.83] }]);
    const csv = statsCsv(rows, { model: "model", measured: "measured" }).trim().split("\n");
    expect(csv.length).toBe(3);
    const head = csv[0].split(",");
    const model = csv[1].split(",");
    const meas = csv[2].split(",");
    const col = (k: string) => head.indexOf(k);
    expect(model[col("quantity")]).toBe("V_LU");
    expect(model[col("series")]).toBe("model");
    expect(meas[col("series")]).toBe("measured");
    expect(Number(model[col("mean")])).toBeCloseTo(3.75, 12);
    expect(Number(model[col("sd")])).toBeCloseTo(describe([3.6, 3.7, 3.8, 3.9]).sd!, 14);
    expect(Number(model[col("ks_D")])).toBeGreaterThan(0);
    expect(meas[col("ks_D")]).toBe("");
    expect(model[col("unit")]).toBe("V");
  });
});

suite("compact columns", () => {
  const vals = [3.6, 3.7, 3.65, 3.62, 3.68];
  it("mean, SD, p5, p95, Δmeasured, KS p — the comparison only with measured data", () => {
    expect(COMPACT_COLUMNS).toEqual(["mean", "sd", "p05", "p95", "dmean", "ks_p"]);
    const withMeas = computeRows([{ key: "V_LU", label: "V_LU", values: vals, measured: [3.61, 3.66, 3.7] }]);
    const all = ALL_COLUMNS.filter((c) => c !== "m_mean" && c !== "m_sd");
    expect(all).toHaveLength(19);
    expect(visibleColumns(all, COMPACT_COLUMNS, withMeas)).toEqual(COMPACT_COLUMNS);
    const noMeasCols = ALL_COLUMNS.filter((c) => COLUMN_GROUP[c] !== "compare");
    const noMeas = computeRows([{ key: "V_LU", label: "V_LU", values: vals }]);
    expect(visibleColumns(noMeasCols, COMPACT_COLUMNS, noMeas)).toEqual(["mean", "sd", "p05", "p95"]);
    expect(visibleColumns(all, null, withMeas)).toBe(all);
  });
  it("adds the censored count when a row lost cycles, so it never goes unnoticed", () => {
    const cens = computeRows([{ key: "V_LU", label: "V_LU", values: [...vals, null, NaN] }]);
    expect(visibleColumns(ALL_COLUMNS.filter((c) => COLUMN_GROUP[c] !== "compare"), COMPACT_COLUMNS, cens)).toEqual(["mean", "sd", "p05", "p95", "censored"]);
  });
});
