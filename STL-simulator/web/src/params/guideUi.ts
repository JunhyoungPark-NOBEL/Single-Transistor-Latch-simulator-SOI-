// Read-side helpers for the parameter guide (content/params/guide.ts, owned by the guide author).
// The UI never edits the guide: it picks an entry, cuts the first sentence for the inline line, and parses
// the V_LU / V_LD effect lines into arrow chips ("V_LU ↑ 80 mV", "V_LD → 그대로").
import { GUIDE_LEGEND, PARAM_GUIDE, type ParamGuide } from "../content/params/guide";
import type { L10n } from "../content/physics/types";
import type { Ctx, FieldDef } from "./schema";

export type { ParamGuide };

/** Guide entry for a parameter key (FieldDef.key, the light-block keys and the "light" group). `bench_*`
 *  keys have none: the UI then falls back to the plain technical tooltip. */
export function guideFor(key: string | null | undefined): ParamGuide | undefined {
  if (!key) return undefined;
  return Object.prototype.hasOwnProperty.call(PARAM_GUIDE, key) ? PARAM_GUIDE[key] : undefined;
}

// abbreviations whose period does not end a sentence
const NO_BREAK = /(?:^|[\s(])(?:e\.g|i\.e|vs|cf|approx|etc|incl|excl|cal)$/i;

/** First sentence of a guide text: up to the first ". " (Korean "다. " included); the whole text if none. */
export function firstSentence(s: string): string {
  const text = s.trim();
  let from = 0;
  for (;;) {
    const i = text.indexOf(". ", from);
    if (i < 0) return text;
    if (!NO_BREAK.test(text.slice(0, i))) return text.slice(0, i + 1);
    from = i + 2;
  }
}

/** Guide text for the inline line beside a main field: the whole intuitive picture (both sentences, so the
 *  plain "what happens" half is never cut off) with parenthetical glosses collapsed to their first term:
 *  "(GIDL, 게이트 유도 드레인 누설: …)" → "(GIDL)". Short parentheticals without a gloss ("(유량)") stay.
 *  The popover and the Details window keep the full text. */
export function inlineLead(s: string): string {
  return s
    .trim()
    .replace(/\(([^()]{1,12}?)\s*[,:：][^()]*\)/g, "($1)")
    .replace(/\s{2,}/g, " ");
}

export type EffectDir = "up" | "down" | "flat";
export interface Effect {
  /** Quantity the line is about: a leading "V_LU" or "V_LD". */
  q: "V_LU" | "V_LD" | null;
  /** Direction from the first arrow: ↑ up, ↓ down, → flat ("그대로"). */
  dir: EffectDir;
  /** First "<number> mV|V" after the arrow (the same rule as guide.test.ts); omitted for "→". */
  mag?: string;
  /** The line is about a Monte-Carlo mean ("평균" / "mean" before the arrow). */
  mean: boolean;
  /** The spread σ moves in a later "· …" clause (e.g. "→ 거의 그대로 · σ 약 2배"). */
  sigma?: "up" | "down";
  /** How much the parameter was raised for this number: the bracket after the magnitude ("+0.1 V",
   *  "0 → 1.9 pA", "×10"; only its first comma-separated part). Omitted for "→" lines and for brackets
   *  that qualify the result instead ("<1 mV", "거의 그대로"). */
  step?: string;
}

const ARROW = /[↑↓→]/u;
const MAG = /(\d+(?:\.\d+)?)\s*(mV|V)(?![A-Za-z/])/;

/** Parse one effect line ("V_LU ↑ 약 80 mV (+0.1 V)"). Returns null when it has no direction arrow. */
export function parseEffect(line: string | null | undefined): Effect | null {
  if (!line) return null;
  const s = line.trim();
  const i = s.search(ARROW);
  if (i < 0) return null;
  const arrow = s[i];
  const dir: EffectDir = arrow === "↑" ? "up" : arrow === "↓" ? "down" : "flat";
  const q = /^V_LU\b/.test(s) ? "V_LU" : /^V_LD\b/.test(s) ? "V_LD" : null;
  const head = s.slice(0, i);
  const out: Effect = { q, dir, mean: /평균|\bmean\b/i.test(head) };
  if (dir !== "flat") {
    const m = s.slice(i + 1).match(MAG);
    if (m) out.mag = `${m[1]} ${m[2]}`;
    const b = s.slice(i + 1).match(/\(([^()]*)\)/);
    const step = b?.[1].split(/[,，]/)[0].trim();
    if (step && /^[+−-]?\s*[\d.]|[+×→]/.test(step) && !/[<≤]|그대로|unchanged|같음|same/i.test(step)) out.step = step;
  }
  // a later "· σ …" clause: the mean may stay put while the spread grows or shrinks
  const rest = s.slice(i + 1);
  const dot = rest.indexOf("·");
  if (dot >= 0) {
    const tail = rest.slice(dot + 1);
    if (/σ/.test(tail)) {
      if (/↑|2배|doubl/i.test(tail)) out.sigma = "up";
      else if (/↓|halv|절반/i.test(tail)) out.sigma = "down";
    }
  }
  return out;
}

/** The V_LU and V_LD chips of an entry (null where a line does not parse). */
export function effectChips(g: ParamGuide | undefined, lang: keyof L10n): [Effect | null, Effect | null] {
  if (!g) return [null, null];
  return [parseEffect(g.effect[0][lang]), parseEffect(g.effect[1][lang])];
}

/** Lead sentence of a group's guide block: its own entry when one exists (e.g. "light"), else the bucket picture. */
export function groupLead(groupId: string | null | undefined): L10n {
  const g = guideFor(groupId ?? "");
  return g ? g.intuitive : GUIDE_LEGEND.picture;
}

export { GUIDE_LEGEND };

/** How the guide's "if you increase it" reads for a field: numbers are raised, switches turned on, choices changed. */
export type GuideVerb = "raise" | "on" | "change";
export function guideVerb(f: Pick<FieldDef, "type"> | null | undefined): GuideVerb {
  if (!f) return "raise";
  if (f.type === "toggle") return "on";
  if (f.type === "select" || f.type === "segmented") return "change";
  return "raise";
}

/** Main (always visible, with an inline guide) in this context. */
export function isMain(f: Pick<FieldDef, "main">, c: Ctx): boolean {
  return typeof f.main === "function" ? f.main(c) : !!f.main;
}
/** Main in some context (the Physics-tab guide lists these first). */
export const isMainAnywhere = (f: Pick<FieldDef, "main">): boolean => !!f.main;

/** Split a guide line into [leading quantity token, rest] so the token can take its colour. */
export function splitLead(line: string): { q: "V_LU" | "V_LD" | null; rest: string } {
  const m = line.match(/^(V_LU|V_LD)\b/);
  return m ? { q: m[1] as "V_LU" | "V_LD", rest: line.slice(m[1].length) } : { q: null, rest: line };
}
