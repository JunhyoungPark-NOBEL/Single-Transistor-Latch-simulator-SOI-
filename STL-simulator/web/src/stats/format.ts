// Unit-aware formatting of statistics cells: levels (mean, median, percentiles, min/max) in the base unit
// (V for voltages), spreads (SD, IQR, CI, Δmean) with their own SI prefix (mV for voltages). Decimals of a
// level follow the spread of its row, so a 174 mV spread prints 3.808 V and an 8 mV spread 3.6442 V.
import { fmtSig, isNum, siPrefix } from "../utils/format";

export const DASH = "—";

/** Units that take SI prefixes. Everything else (dimensionless "1", "", "%", "dec", …) is printed plainly. */
const PREFIXABLE = new Set(["V", "A", "s", "C", "F", "Ω", "W", "Hz", "S", "J"]);

export interface UnitPlan {
  /** physical unit of the values ("V", "s", "1", …) */
  unit: string;
  levelFactor: number;
  levelUnit: string;
  spreadFactor: number;
  spreadUnit: string;
}

/**
 * Display units for a row (or a homogeneous table): `levelMag` = largest |level|, `spreadMag` = largest SD.
 * Voltages: levels in V (unless below 10 mV), spreads in mV (unless ≥ 1 V).
 */
export function planUnits(unit: string, levelMag: number | null, spreadMag: number | null): UnitPlan {
  const u = unit ?? "";
  if (!PREFIXABLE.has(u)) return { unit: u, levelFactor: 1, levelUnit: u === "1" ? "" : u, spreadFactor: 1, spreadUnit: u === "1" ? "" : u };
  let lf = 1;
  let lp = "";
  let sf = 1;
  let sp = "";
  const lm = isNum(levelMag) ? Math.abs(levelMag) : 0;
  const sm = isNum(spreadMag) ? Math.abs(spreadMag) : 0;
  if (u === "V") {
    if (lm > 0 && lm < 0.01) [lf, lp] = siPrefix(lm);
    if (sm < 1) [sf, sp] = [1e-3, "m"];
  } else {
    if (lm > 0) [lf, lp] = siPrefix(lm);
    if (sm > 0) [sf, sp] = siPrefix(sm);
    else [sf, sp] = [lf, lp];
  }
  return { unit: u, levelFactor: lf, levelUnit: `${lp}${u}`, spreadFactor: sf, spreadUnit: `${sp}${u}` };
}

/**
 * Decimals for levels given the row's SD: resolve about a tenth of the spread, and at least one spread unit
 * (levels in V with spreads in mV → ≥ 3 decimals, i.e. mV resolution). 0 … 6 decimals.
 */
export function levelDecimals(sd: number | null | undefined, plan: UnitPlan): number | null {
  if (!isNum(sd) || sd <= 0) return null;
  const s = sd / plan.levelFactor;
  const minDec = Math.max(0, Math.round(Math.log10(plan.levelFactor / plan.spreadFactor)));
  return Math.min(6, Math.max(minDec, Math.ceil(-Math.log10(s)) + 1));
}

const minus = (s: string) => (s.startsWith("-") ? `−${s.slice(1)}` : s);

export function fmtLevel(v: number | null | undefined, plan: UnitPlan, decimals: number | null): string {
  if (!isNum(v)) return DASH;
  const x = v / plan.levelFactor;
  return minus(decimals === null ? fmtSig(x, 4) : x.toFixed(decimals));
}

export function fmtSpread(v: number | null | undefined, plan: UnitPlan, signed = false): string {
  if (!isNum(v)) return DASH;
  const x = v / plan.spreadFactor;
  const s = x === 0 ? "0" : Math.abs(x) >= 100 ? x.toFixed(1) : fmtSig(x, 3);
  return minus(signed && x > 0 ? `+${s}` : s);
}

/** Coefficient of variation (fraction) in %. */
export function fmtPct(v: number | null | undefined): string {
  return isNum(v) ? fmtSig(v * 100, 3) : DASH;
}

/** Share (fraction) as a percentage string for prose: 0.0125 → "1.3 %", 0 → "0 %". */
export function fmtShare(v: number | null | undefined): string {
  if (!isNum(v)) return DASH;
  if (v === 0) return "0 %";
  const p = v * 100;
  // 99.74 % must not print as "100 %" (that would claim every cycle), nor 0.04 % as "0 %"
  if (p >= 99 && p < 100) return `${Math.min(p, 99.9).toFixed(1)} %`;
  return `${p >= 10 ? p.toFixed(0) : p >= 0.1 ? p.toFixed(1) : p.toPrecision(1)} %`;
}

export function fmtCoef(v: number | null | undefined, decimals = 2): string {
  return isNum(v) ? minus(v.toFixed(decimals)) : DASH;
}

/** p-values: 0.70, 0.037, <0.001 */
export function fmtP(p: number | null | undefined): string {
  if (!isNum(p)) return DASH;
  if (p >= 0.1) return p.toFixed(2);
  if (p >= 0.001) return p.toFixed(3);
  return "<0.001";
}
