// Guide UI helpers: every main field renders two parsed arrow chips (V_LU, V_LD) in both languages, the
// first-sentence cut, the effect-line parser, group leads, and the bench_* fallback to the plain tooltip.
import { describe, expect, it } from "vitest";
import { GUIDE_LEGEND, PARAM_GUIDE } from "../content/params/guide";
import { BENCHES } from "./benches";
import { effectChips, firstSentence, groupLead, guideFor, guideVerb, isMain, isMainAnywhere, parseEffect, splitLead } from "./guideUi";
import { GROUPS, LIGHT_FIELDS, type Ctx, type FieldDef } from "./schema";
import { BUILTIN_META } from "../state/presets";
import { presetRoot } from "../utils/payload";

const ALL_FIELDS: FieldDef[] = [...GROUPS.flatMap((g) => g.fields), ...Object.values(LIGHT_FIELDS)];
const root = presetRoot(BUILTIN_META, "paper");
const ctx = (mode: Ctx["mode"], tab: Ctx["tab"] = "device"): Ctx => ({ root, mode, tab });

describe("main fields (inline guide)", () => {
  const main = ALL_FIELDS.filter(isMainAnywhere);

  it("are the §2.5 set", () => {
    expect(main.map((f) => f.key).sort()).toEqual(["c_runs", "iph_pA", "ls_mode", "ls_sigma", "n_cycles", "power_mW", "rate", "vd_max", "vg"].sort());
    const rate = ALL_FIELDS.find((f) => f.key === "rate")!;
    expect(isMain(rate, ctx("deterministic"))).toBe(false);
    expect(isMain(rate, ctx("stochastic"))).toBe(true);
    expect(isMain(LIGHT_FIELDS.resp, ctx("deterministic"))).toBe(false);
  });

  for (const f of main) {
    it(`${f.key}: has a guide whose first two lines parse to V_LU and V_LD arrows`, () => {
      const g = guideFor(f.key);
      expect(g).toBeDefined();
      for (const lang of ["ko", "en"] as const) {
        const [lu, ld] = effectChips(g, lang);
        expect(lu?.q, `${f.key} ${lang} line 1`).toBe("V_LU");
        expect(ld?.q, `${f.key} ${lang} line 2`).toBe("V_LD");
        expect(["up", "down", "flat"]).toContain(lu!.dir);
        expect(["up", "down", "flat"]).toContain(ld!.dir);
        // a moving value carries its magnitude; "→" never does (it renders as "그대로")
        for (const e of [lu!, ld!]) {
          if (e.dir === "flat") expect(e.mag).toBeUndefined();
        }
      }
    });
  }

  it("V_G reads “V_LU ↑ 80 mV · V_LD → 그대로”", () => {
    const [lu, ld] = effectChips(guideFor("vg"), "ko");
    expect(lu).toMatchObject({ q: "V_LU", dir: "up", mag: "80 mV", mean: false });
    expect(ld).toMatchObject({ q: "V_LD", dir: "flat", mean: false });
    const [luEn] = effectChips(guideFor("vg"), "en");
    expect(luEn).toMatchObject({ q: "V_LU", dir: "up", mag: "80 mV" });
  });

  it("light: V_LU ↓ 320 mV, V_LD ↓ 1.2 mV (the magnitude after the arrow, not the step in brackets)", () => {
    const [lu, ld] = effectChips(guideFor("iph_pA"), "ko");
    expect(lu).toMatchObject({ q: "V_LU", dir: "down", mag: "320 mV" });
    expect(ld).toMatchObject({ q: "V_LD", dir: "down", mag: "1.2 mV" });
  });

  it("ls_sigma: the mean stays, the spread grows (σ ↑)", () => {
    const [lu, ld] = effectChips(guideFor("ls_sigma"), "ko");
    expect(lu).toMatchObject({ q: "V_LU", dir: "flat", mean: true, sigma: "up" });
    expect(ld).toMatchObject({ q: "V_LD", dir: "flat", mean: true, sigma: "up" });
    const [luEn] = effectChips(guideFor("ls_sigma"), "en");
    expect(luEn).toMatchObject({ dir: "flat", mean: true, sigma: "up" });
  });
});

describe("every guide entry", () => {
  for (const [key, g] of Object.entries(PARAM_GUIDE)) {
    it(`${key}: V_LU / V_LD lines parse in both languages`, () => {
      for (const lang of ["ko", "en"] as const) {
        const [lu, ld] = effectChips(g, lang);
        expect(lu?.q).toBe("V_LU");
        expect(ld?.q).toBe("V_LD");
      }
    });
  }
});

describe("parseEffect", () => {
  it("reads direction, magnitude and qualifiers", () => {
    expect(parseEffect("V_LU ↑ 약 80 mV (+0.1 V)")).toEqual({ q: "V_LU", dir: "up", mag: "80 mV", mean: false });
    expect(parseEffect("V_LD ↓ ≈ 1.2 V")).toEqual({ q: "V_LD", dir: "down", mag: "1.2 V", mean: false });
    expect(parseEffect("V_LD → 거의 그대로 (<1 mV)")).toEqual({ q: "V_LD", dir: "flat", mean: false });
    expect(parseEffect("V_LU mean ↓ ≈ 59 mV (off → on)")).toEqual({ q: "V_LU", dir: "down", mag: "59 mV", mean: true });
    expect(parseEffect("V_LU ↓ ≈ 1.7 mV (+0.1 V/dec)")?.mag).toBe("1.7 mV");
    expect(parseEffect("V_LU 평균 ↓ (소자 탭 기준 보정: 약 59 mV)")?.mag).toBe("59 mV");
    expect(parseEffect("V_LU ↑ (창 사라짐)")).toEqual({ q: "V_LU", dir: "up", mean: false });
  });
  it("returns null without an arrow, and q = null for other quantities", () => {
    expect(parseEffect("no arrow here")).toBeNull();
    expect(parseEffect("")).toBeNull();
    expect(parseEffect(undefined)).toBeNull();
    expect(parseEffect("GIDL 정공 공급 ↓ → 래치업이 늦어짐")?.q).toBeNull();
  });
});

describe("firstSentence", () => {
  it("cuts at the first sentence end (Korean and English)", () => {
    expect(firstSentence(PARAM_GUIDE.vg.intuitive.ko)).toMatch(/손잡이입니다\.$/);
    expect(firstSentence(PARAM_GUIDE.vg.intuitive.en)).toMatch(/field\)\.$/);
    expect(firstSentence("하나입니다. 둘입니다.")).toBe("하나입니다.");
    expect(firstSentence("One. Two.")).toBe("One.");
  });
  it("keeps a single sentence, numbers and abbreviations whole", () => {
    expect(firstSentence("Only one sentence.")).toBe("Only one sentence.");
    expect(firstSentence("It is 0.75 pA per mW. Next.")).toBe("It is 0.75 pA per mW.");
    expect(firstSentence("Matters once on, e.g. at V_G = −1.1 V. Next.")).toBe("Matters once on, e.g. at V_G = −1.1 V.");
  });
  it("every guide entry has a non-empty first sentence no longer than the text", () => {
    for (const g of Object.values(PARAM_GUIDE)) {
      for (const lang of ["ko", "en"] as const) {
        const s = firstSentence(g.intuitive[lang]);
        expect(s.length).toBeGreaterThan(0);
        expect(g.intuitive[lang].startsWith(s)).toBe(true);
      }
    }
  });
});

describe("groupLead / fallbacks", () => {
  it("uses the group's own entry, else the bucket picture", () => {
    expect(groupLead("light")).toBe(PARAM_GUIDE.light.intuitive);
    expect(groupLead("bias")).toBe(GUIDE_LEGEND.picture);
    expect(groupLead(undefined)).toBe(GUIDE_LEGEND.picture);
  });
  it("bench_* fields have no guide (the popover falls back to the technical tooltip)", () => {
    const keys = Object.values(BENCHES).flatMap((b) => b.fields.map((f) => `bench_${f.key}`));
    expect(keys.length).toBeGreaterThan(5);
    for (const k of keys) expect(guideFor(k), k).toBeUndefined();
    expect(guideFor("toString")).toBeUndefined();
    expect(guideFor("")).toBeUndefined();
  });
  it("verb follows the field kind", () => {
    const byKey = (k: string) => ALL_FIELDS.find((f) => f.key === k)!;
    expect(guideVerb(byKey("vg"))).toBe("raise");
    expect(guideVerb(byKey("carrier_noise"))).toBe("on");
    expect(guideVerb(byKey("ls_mode"))).toBe("change");
    expect(guideVerb(byKey("engine"))).toBe("change");
  });
  it("splitLead separates the coloured token", () => {
    expect(splitLead("V_LU ↑ 80 mV")).toEqual({ q: "V_LU", rest: " ↑ 80 mV" });
    expect(splitLead("GIDL ↓")).toEqual({ q: null, rest: "GIDL ↓" });
  });
});
