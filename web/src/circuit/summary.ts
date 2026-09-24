// Summary value formatting shared by the quick benches and the schematic results.
import { fmtSig, isNum, siPrefix } from "../utils/format";

/** Summary value → display value + unit (V with 3 decimals, A/C/s with SI prefixes, dimensionless plain). */
export function splitUnit(v: number | string | null | undefined, unit?: string, force?: "mV"): { value: string; unit: string } {
  if (typeof v === "string") return { value: v, unit: unit && unit !== "1" ? unit : "" };
  if (!isNum(v)) return { value: "—", unit: "" };
  if (force === "mV") return { value: (v * 1e3).toFixed(1), unit: "mV" };
  if (unit === "V") return Math.abs(v) < 0.1 && v !== 0 ? { value: fmtSig(v * 1e3, 3), unit: "mV" } : { value: v.toFixed(3), unit: "V" };
  if (unit === "A" || unit === "C" || unit === "s" || unit === "F" || unit === "Ω") {
    const [f, p] = siPrefix(v);
    return { value: (v / f).toPrecision(3), unit: `${p}${unit}` };
  }
  if (Number.isInteger(v)) return { value: v.toLocaleString("en-US"), unit: unit && unit !== "1" ? unit : "" };
  return { value: fmtSig(v, 3), unit: unit && unit !== "1" ? unit : "" };
}
