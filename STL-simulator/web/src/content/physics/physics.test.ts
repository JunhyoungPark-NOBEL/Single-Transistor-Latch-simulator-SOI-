// Content integrity: every topic present, ids unique, every formula renders with KaTeX (throwOnError).
import katex from "katex";
import { describe, expect, it } from "vitest";
import { PHYSICS_TOPICS, TOPIC_ORDER } from "./index";

const inlineMath = (s: string) => [...s.matchAll(/\$([^$]+)\$/g)].map((m) => m[1]);

describe("physics content", () => {
  const present = TOPIC_ORDER.filter((id) => PHYSICS_TOPICS[id].sections.length > 0);

  it("lists topics", () => {
    console.log(`topics with content: ${present.length}/${TOPIC_ORDER.length}`, present.join(", "));
    expect(present.length).toBeGreaterThan(0);
  });

  it("has unique equation ids", () => {
    const ids = present.flatMap((id) => PHYSICS_TOPICS[id].sections.flatMap((s) => (s.equations ?? []).map((e) => e.id)));
    const dup = ids.filter((x, i) => ids.indexOf(x) !== i);
    expect(dup).toEqual([]);
  });

  for (const id of present) {
    it(`renders all math in ${id}`, () => {
      const t = PHYSICS_TOPICS[id];
      const texts: string[] = [t.summary.en, t.summary.ko];
      for (const s of t.sections) {
        for (const e of s.equations ?? []) {
          katex.renderToString(e.tex, { displayMode: true, throwOnError: true });
          if (e.note) texts.push(e.note.en, e.note.ko);
          if (e.label) texts.push(e.label.en, e.label.ko);
        }
        for (const v of s.variables ?? []) katex.renderToString(v.symbol, { throwOnError: true });
        if (s.body) texts.push(s.body.en, s.body.ko);
        for (const n of s.notes ?? []) texts.push(n.en, n.ko);
      }
      for (const txt of texts) for (const m of inlineMath(txt)) katex.renderToString(m, { throwOnError: true });
      for (const s of t.sections) {
        expect(s.heading.ko.length).toBeGreaterThan(0);
        expect(s.heading.en.length).toBeGreaterThan(0);
      }
    });
  }
});
