// SPICE-style numbers: "1k", "2f", "10u", "1meg", "4.7µ", "100n", "1e-3", "2.2kΩ", "10us".
// SPICE semantics: the scale factor is case-insensitive, so "m"/"M" = milli and "meg" = mega; letters
// after the scale factor (units such as V, A, s, F, Ω, ohm, Hz) are ignored. As in SPICE, a bare "F"
// is femto ("1F" = 1e-15) — the UI shows the interpreted value next to every input to make this visible.

const SCALE: [RegExp, number][] = [
  [/^meg/i, 1e6],
  [/^mil/i, 25.4e-6],
  [/^t/i, 1e12],
  [/^g/i, 1e9],
  [/^k/i, 1e3],
  [/^m/i, 1e-3],
  [/^[uµμ]/i, 1e-6],
  [/^n/i, 1e-9],
  [/^p/i, 1e-12],
  [/^f/i, 1e-15],
];

const NUM = /^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(.*)$/i;

/** Parse a SPICE number with an optional scale suffix and unit letters. Returns null when invalid. */
export function parseSI(input: string | number | null | undefined): number | null {
  if (typeof input === "number") return Number.isFinite(input) ? input : null;
  if (input == null) return null;
  const s = String(input).trim().replace(/−/g, "-").replace(/\s+/g, "");
  if (!s) return null;
  const m = NUM.exec(s);
  if (!m) return null;
  let v = Number(m[1]);
  const rest = m[2];
  if (rest) {
    // units may only contain letters and Ω/µ (and a trailing "/s" style is not supported)
    if (!/^[a-zA-ZµμΩ]+$/.test(rest)) return null;
    for (const [re, f] of SCALE) {
      if (re.test(rest)) {
        v *= f;
        break;
      }
    }
  }
  return Number.isFinite(v) ? v : null;
}

const PREFIX: [number, string, string][] = [
  // [factor, display prefix, SPICE suffix]
  [1e12, "T", "t"],
  [1e9, "G", "g"],
  [1e6, "M", "meg"],
  [1e3, "k", "k"],
  [1, "", ""],
  [1e-3, "m", "m"],
  [1e-6, "µ", "u"],
  [1e-9, "n", "n"],
  [1e-12, "p", "p"],
  [1e-15, "f", "f"],
];

function pick(v: number, display = false): [number, string, string] {
  const a = Math.abs(v);
  if (a === 0 || !Number.isFinite(a)) return PREFIX[4];
  for (const p of PREFIX) if (a >= p[0] * 0.99995) return p;
  // display only (SPICE has no atto suffix)
  if (display && a >= 1e-18 * 0.99995) return [1e-18, "a", ""];
  return PREFIX[PREFIX.length - 1];
}

function trimNum(x: number, sig: number): string {
  const s = Number(x.toPrecision(sig)).toString();
  return s.includes("e") ? x.toPrecision(sig) : s;
}

/** Display string with an SI prefix and unit: 2e-15, "F" → "2 fF"; 1e6, "Ω" → "1 MΩ". */
export function fmtSI(v: number | null | undefined, unit = "", sig = 4): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (v === 0) return `0${unit ? " " + unit : ""}`;
  const [f, p] = pick(v, true);
  const a = Math.abs(v);
  // below atto: plain exponent
  if (a < 1e-18 * 0.99995) return `${v.toExponential(Math.max(0, sig - 1))}${unit ? " " + unit : ""}`;
  return `${trimNum(v / f, sig)} ${p}${unit}`;
}

/** Compact SPICE token (round-trips through parseSI): 1e6 → "1meg", 2e-15 → "2f", 1e-5 → "10u". */
export function toSpice(v: number | null | undefined, sig = 6): string {
  if (v == null || !Number.isFinite(v)) return "";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a < 1e-15 * 0.99995 || a >= 1e15) return trimNum(v, sig).replace("e+", "e");
  const [f, , s] = pick(v);
  return `${trimNum(v / f, sig)}${s}`;
}
