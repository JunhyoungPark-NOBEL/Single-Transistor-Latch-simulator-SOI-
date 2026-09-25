// Parameter guide integrity: every UI parameter has a guide, three effect lines in both languages, and the
// arrows / magnitudes of the V_LU and V_LD lines agree with the sensitivity data (sensitivity.json).
import { describe, expect, it } from "vitest";
import { GROUPS } from "../../params/schema";
import { PARAM_GUIDE } from "./guide";
import sensitivity from "./sensitivity.json";

interface Stats { n?: number; sd_VLU_mV?: number; sd_VLD_mV?: number }
interface Side {
  base?: Stats | null;
  up?: { window_exists?: boolean } | null;
  dVLU_up_mV?: number | null;
  dVLD_up_mV?: number | null;
  dVLU_down_mV?: number | null;
  dVLD_down_mV?: number | null;
  slope?: { VLU_mV?: number; VLD_mV?: number } | null;
  nonmonotonic?: { VLU?: boolean; VLD?: boolean };
}
type Block = { ref?: Side; photo?: Side; photo_2p55mW?: Side } & Record<string, unknown>;

const S = sensitivity as unknown as Record<string, Block>;
const SENS_KEYS = Object.keys(S).filter((k) => !k.startsWith("_") && k !== "baseline");
/** Custom light block (sidebar/ParamGroup.tsx LIGHT_FIELDS) + the light group itself. */
const LIGHT_KEYS = ["light", "iph_pA", "power_mW", "resp"];
const SCHEMA_KEYS = GROUPS.flatMap((g) => g.fields.map((f) => f.key));

/** Where the numbers of a guide entry come from (reference calibration unless aliased). */
function sourceOf(key: string): Side | undefined {
  if (key === "iph_pA" || key === "power_mW") return S.light?.ref;
  if (key === "resp") return S.light?.photo_2p55mW; // I_PH ×2 at 2.55 mW ≙ R ×2
  return S[key]?.ref;
}

const ARROWS = /[↑↓→]/u;
type Which = "VLU" | "VLD";

/** Expected change for "increase": the up step; for on/off toggles (no up step) the reverse of "off";
 *  when the step lost the latch window, the local slope. Monte-Carlo means: |Δ| < 3 SE counts as "→". */
function expected(side: Side, which: Which): { delta: number; tol: number } | null {
  let delta = side[`d${which}_up_mV`];
  if (delta == null) {
    const down = side[`d${which}_down_mV`];
    if (down != null) delta = -down;
    else if (side.slope && typeof side.slope[`${which}_mV`] === "number") delta = side.slope[`${which}_mV`];
  }
  if (delta == null) return null;
  const sd = side.base?.[`sd_${which}_mV`];
  const n = side.base?.n;
  const tol = typeof sd === "number" && typeof n === "number" && n > 0 ? Math.max(1, (3 * sd) / Math.sqrt(n)) : 1;
  return { delta, tol };
}
const arrowFor = (d: number, tol: number) => (Math.abs(d) < tol ? "→" : d > 0 ? "↑" : "↓");

/** First "<number> mV|V" after the first arrow, in mV. */
function magnitudeAfterArrow(line: string): { mV: number; unit: string } | null {
  const i = line.search(ARROWS);
  const m = line.slice(i + 1).match(/(\d+(?:\.\d+)?)\s*(mV|V)(?![A-Za-z/])/);
  if (!m) return null;
  return { mV: m[2] === "V" ? Number(m[1]) * 1e3 : Number(m[1]), unit: m[2] };
}

describe("parameter guide", () => {
  it("covers every schema field, light-block field and sensitivity key", () => {
    const need = [...new Set([...SCHEMA_KEYS, ...LIGHT_KEYS, ...SENS_KEYS])];
    const missing = need.filter((k) => !PARAM_GUIDE[k]);
    expect(missing).toEqual([]);
  });

  it("has no stray keys", () => {
    const known = new Set([...SCHEMA_KEYS, ...LIGHT_KEYS, ...SENS_KEYS]);
    expect(Object.keys(PARAM_GUIDE).filter((k) => !known.has(k))).toEqual([]);
  });

  for (const [key, g] of Object.entries(PARAM_GUIDE)) {
    describe(key, () => {
      it("is well formed in both languages", () => {
        expect(g.key).toBe(key);
        expect(g.effect).toHaveLength(3);
        const texts = [g.intuitive, ...g.effect, ...(g.caveat ? [g.caveat] : [])];
        for (const t of texts) {
          expect(t.ko.trim().length).toBeGreaterThan(0);
          expect(t.en.trim().length).toBeGreaterThan(0);
        }
        // very short, no equations
        for (const s of [g.intuitive.ko, g.intuitive.en]) expect(s).not.toMatch(/=|∝|\^|exp\(/);
        expect(g.intuitive.ko.length).toBeLessThanOrEqual(130);
        expect(g.intuitive.en.length).toBeLessThanOrEqual(230);
        for (const t of g.effect) {
          expect(t.ko.length).toBeLessThanOrEqual(70);
          expect(t.en.length).toBeLessThanOrEqual(110);
        }
        // line 1 = V_LU, line 2 = V_LD, each with a direction arrow
        for (const lang of ["ko", "en"] as const) {
          expect(g.effect[0][lang]).toMatch(/^V_LU\b/);
          expect(g.effect[1][lang]).toMatch(/^V_LD\b/);
          expect(g.effect[0][lang]).toMatch(ARROWS);
          expect(g.effect[1][lang]).toMatch(ARROWS);
        }
        if (SENS_KEYS.includes(key)) expect(g.basis?.trim().length ?? 0).toBeGreaterThan(0);
      });

      const side = sourceOf(key);
      if (!side) return;

      it("agrees with sensitivity.json (arrow and magnitude)", () => {
        (["VLU", "VLD"] as const).forEach((which, i) => {
          const exp = expected(side, which);
          if (!exp) return;
          const want = arrowFor(exp.delta, exp.tol);
          for (const lang of ["ko", "en"] as const) {
            const line = g.effect[i][lang];
            const got = line.match(ARROWS)?.[0];
            expect(got, `${key} ${lang} line ${i + 1}: "${line}" (Δ = ${exp.delta} mV, tol ${exp.tol.toFixed(1)} mV)`).toBe(want);
            if (want === "→") continue;
            const mag = magnitudeAfterArrow(line);
            expect(mag, `${key} ${lang} line ${i + 1} has no magnitude`).not.toBeNull();
            const target = Math.abs(exp.delta);
            const slack = 0.06 * target + (mag!.unit === "V" ? 5 : 0.05);
            expect(Math.abs(mag!.mV - target), `${key} ${lang} line ${i + 1}: ${mag!.mV} mV vs ${target} mV`).toBeLessThanOrEqual(slack);
          }
        });
      });

      it("has a caveat where the step loses the window or the response is non-monotonic", () => {
        const lost = side.up?.window_exists === false;
        const nonmono = !!(side.nonmonotonic?.VLU || side.nonmonotonic?.VLD);
        if (lost || nonmono) expect(g.caveat).toBeDefined();
      });
    });
  }
});
