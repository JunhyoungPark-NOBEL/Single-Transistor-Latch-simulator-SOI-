import { describe, expect, it } from "vitest";
import { BRAND_STRINGS } from "../i18n/strings.brand";
import { signed, subs, withUnit } from "./labels";

describe("plot label helpers", () => {
  it("turns server symbols into Plotly subscripts", () => {
    expect(subs("V_LU (drain node)")).toBe("V<sub>LU</sub> (drain node)");
    expect(subs("ΔQ_B")).toBe("ΔQ<sub>B</sub>");
    expect(subs("X1 V_D at latch-up")).toBe("X1 V<sub>D</sub> at latch-up");
    expect(subs("V(n001)")).toBe("V(n001)");
  });
  it("formats units and signs", () => {
    expect(withUnit("Delay", "s")).toBe("Delay (s)");
    expect(withUnit("P(1)", "1")).toBe("P(1)");
    expect(withUnit("Count", "")).toBe("Count");
    expect(signed(-1.8, 1)).toBe("−1.8");
  });
});

describe("axis.* dictionary", () => {
  const axis = Object.entries(BRAND_STRINGS).filter(([k]) => k.startsWith("axis."));
  it("has both languages, no raw underscores and the same placeholders in KO and EN", () => {
    expect(axis.length).toBeGreaterThan(50);
    for (const [k, v] of axis) {
      expect(v.ko, k).toBeTruthy();
      expect(v.en, k).toBeTruthy();
      // symbols use <sub> (Plotly HTML), never "V_D"
      expect(/[A-Za-zσφ]_[A-Za-z]/.test(v.ko + v.en), k).toBe(false);
      const ph = (s: string) => (s.match(/\{\w+\}/g) ?? []).sort().join();
      expect(ph(v.ko), k).toBe(ph(v.en));
    }
  });
  it("puts units last in parentheses: Name symbol (unit)", () => {
    for (const k of ["axis.vd", "axis.vg", "axis.idAbs", "axis.icomp", "axis.cb.u", "axis.vlu", "axis.power", "axis.dmap.L"] as const)
      for (const s of [BRAND_STRINGS[k].ko, BRAND_STRINGS[k].en]) expect(s, k).toMatch(/ \((V|A|fC|mW|nm)\)$/);
    expect(BRAND_STRINGS["axis.vd"].en).toBe("Drain voltage V<sub>D</sub> (V)");
    expect(BRAND_STRINGS["axis.count"]).toEqual({ ko: "빈도", en: "Count" });
  });
});
