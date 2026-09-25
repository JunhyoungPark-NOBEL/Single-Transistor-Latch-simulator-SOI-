// Summary value formatting shared by the quick benches and the schematic results.
import { fmtSig, isNum, siPrefix } from "../utils/format";

/** Typographic minus for display (−0.07, not -0.07). */
export const tminus = (s: string) => s.replace(/^-/, "−");

/** Correlation-like keys (lag-1, corr): 2 decimals, "≈ 0" below 0.005 (their standard error is ~1/√n ≫ 0.001). */
export const isCoefKey = (key: string) => /(^|\.)(lag1|lag_\w+|corr\w*)$/.test(key);
export function fmtCoefValue(v: number): string {
  return Math.abs(v) < 0.005 ? "≈ 0" : tminus(v.toFixed(2));
}

/** Summary value → display value + unit (V with 3 decimals, A/C/s with SI prefixes, dimensionless plain). */
export function splitUnit(v: number | string | null | undefined, unit?: string, force?: "mV"): { value: string; unit: string } {
  if (typeof v === "string") return { value: v, unit: unit && unit !== "1" ? unit : "" };
  if (!isNum(v)) return { value: "—", unit: "" };
  const out = splitUnitRaw(v, unit, force);
  return { ...out, value: tminus(out.value) };
}

/** A spread (± …) in the value's unit with at most 2 significant digits; null when it rounds to 0 there. */
export function splitSpread(sd: number | null | undefined, unit?: string): { value: string; unit: string } | null {
  if (!isNum(sd) || sd <= 0) return null;
  if (unit === "V") {
    const mv = sd * 1e3;
    return mv < 0.05 ? null : { value: fmtSig(mv, mv >= 10 ? 3 : 2), unit: "mV" };
  }
  if (unit === "A" || unit === "C" || unit === "s" || unit === "F" || unit === "Ω") {
    const [f, p] = siPrefix(sd);
    return { value: fmtSig(sd / f, 2), unit: `${p}${unit}` };
  }
  const s = fmtSig(sd, 2);
  return Number(s) === 0 ? null : { value: s, unit: unit && unit !== "1" ? unit : "" };
}

function splitUnitRaw(v: number, unit?: string, force?: "mV"): { value: string; unit: string } {
  if (force === "mV") return { value: (v * 1e3).toFixed(1), unit: "mV" };
  if (unit === "V") return Math.abs(v) < 0.1 && v !== 0 ? { value: fmtSig(v * 1e3, 3), unit: "mV" } : { value: v.toFixed(3), unit: "V" };
  if (unit === "A" || unit === "C" || unit === "s" || unit === "F" || unit === "Ω") {
    const [f, p] = siPrefix(v);
    return { value: (v / f).toPrecision(3), unit: `${p}${unit}` };
  }
  if (Number.isInteger(v)) return { value: v.toLocaleString("en-US"), unit: unit && unit !== "1" ? unit : "" };
  return { value: fmtSig(v, 3), unit: unit && unit !== "1" ? unit : "" };
}
