// Unit-aware number formatting (SI prefixes, tabular display, input strings).

const PREFIXES: [number, string][] = [
  [1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""], [1e-3, "m"], [1e-6, "µ"], [1e-9, "n"],
  [1e-12, "p"], [1e-15, "f"], [1e-18, "a"],
];

export const DASH = "—";

export function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

/** Significant-digit formatting without exponent for moderate magnitudes; trailing zeros kept for alignment. */
export function fmtSig(v: number | null | undefined, sig = 3): string {
  if (!isNum(v)) return DASH;
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(Math.max(0, sig - 1)).replace("e+", "e");
  const decimals = Math.max(0, sig - 1 - Math.floor(Math.log10(a)));
  return v.toFixed(Math.min(decimals, 12));
}

/** Pick an SI prefix so that the mantissa lies in [1, 1000). */
export function siPrefix(v: number): [number, string] {
  const a = Math.abs(v);
  if (a === 0 || !Number.isFinite(a)) return [1, ""];
  for (const [f, p] of PREFIXES) if (a >= f * 0.9995) return [f, p];
  return PREFIXES[PREFIXES.length - 1];
}

/** 3.2e-12, "A" → "3.20 pA". Units that should not take prefixes (dimensionless "1", "%", "dec") are printed plainly. */
export function fmtSI(v: number | null | undefined, unit = "", sig = 3): string {
  if (!isNum(v)) return DASH;
  if (unit === "1" || unit === "" || unit === "%" || unit === "cm⁻³" || unit === "cm^-3") {
    return `${fmtSig(v, sig)}${unit && unit !== "1" ? " " + unit : ""}`;
  }
  const [f, p] = siPrefix(v);
  return `${fmtSig(v / f, sig)} ${p}${unit}`;
}

/** Voltage with fixed decimals (fold voltages): 3.70372 → "3.704 V". */
export function fmtV(v: number | null | undefined, decimals = 3): string {
  return isNum(v) ? `${v.toFixed(decimals)} V` : DASH;
}

/** Volts → millivolts string: 0.1203 → "120.3 mV". */
export function fmtmV(vVolt: number | null | undefined, decimals = 1): string {
  return isNum(vVolt) ? `${(vVolt * 1e3).toFixed(decimals)} mV` : DASH;
}

export function fmtDuration(s: number | null | undefined): string {
  if (!isNum(s)) return DASH;
  if (s < 1) return `${(s * 1e3).toFixed(0)} ms`;
  if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - 60 * m)} s`;
}

export function fmtInt(v: number | null | undefined): string {
  return isNum(v) ? Math.round(v).toLocaleString("en-US") : DASH;
}

/** Compact string for a numeric <input>: up to `sig` significant digits, no trailing zeros. */
export function toInputString(v: number | null | undefined, sig = 6): string {
  if (!isNum(v)) return "";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e7 || a < 1e-5) {
    const s = v.toExponential(sig - 1);
    const [m, e] = s.split("e");
    return `${m.includes(".") ? m.replace(/0+$/, "").replace(/\.$/, "") : m}e${e.replace("+", "")}`;
  }
  const s = Number(v.toPrecision(sig)).toString();
  return s;
}

/** Parse a user-typed number; accepts "1e-3", "1,5" (comma decimal), unicode minus, surrounding spaces. */
export function parseNumber(s: string): number | null {
  const t = s.trim().replace(/−/g, "-").replace(/,/g, ".").replace(/\s+/g, "");
  if (t === "" || t === "-" || t === ".") return null;
  if (!/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(t)) return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

/** Relative closeness used for "changed from default" markers. */
export function nearlyEqual(a: unknown, b: unknown, rel = 1e-9): boolean {
  if (typeof a === "number" && typeof b === "number") {
    if (a === b) return true;
    return Math.abs(a - b) <= rel * Math.max(Math.abs(a), Math.abs(b), 1e-300);
  }
  return a === b;
}

const SUP: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
/** Scientific notation for reading, not editing: 2.30×10¹⁷ (2 decimals by default; typographic minus). */
export function sciText(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return "—";
  const [m, e] = v.toExponential(digits).split("e");
  return `${m.replace("-", "−")}×10${String(Number(e)).split("").map((c) => SUP[c] ?? c).join("")}`;
}
