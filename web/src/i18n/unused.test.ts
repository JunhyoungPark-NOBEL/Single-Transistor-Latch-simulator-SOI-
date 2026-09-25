// Every dictionary key is used somewhere in the UI code: a quoted literal ("kpi.vlu"), or a key built from a
// template prefix (`stats.c.${c}`, `schematic.erc.${code}`). Dead keys pile up when a view is removed; this test
// lists them so they are deleted together with the view.
import { describe, expect, it } from "vitest";

const sources = import.meta.glob<string>(["../**/*.ts", "../**/*.tsx"], { query: "?raw", import: "default", eager: true });
// glob keys are relative to this file: "./strings.ts" here, "../device/KpiStrip.tsx" elsewhere
const isDict = (f: string) => /^\.\/strings(\.[a-z]+)?\.ts$/.test(f);

describe("i18n dictionaries", () => {
  it("have no unused keys", () => {
    const keys: [string, string][] = [];
    for (const [f, src] of Object.entries(sources)) {
      if (!isDict(f)) continue;
      for (const m of src.matchAll(/^\s*"([a-zA-Z0-9_.-]+)":\s*\{\s*ko:/gm)) keys.push([f, m[1]]);
    }
    expect(keys.length).toBeGreaterThan(300);
    const code = Object.entries(sources)
      .filter(([f]) => !isDict(f) && !/\.test\.tsx?$/.test(f))
      .map(([, src]) => src)
      .join("\n");
    // key prefixes built at run time: `prefix.${…}`
    const prefixes = [...new Set([...code.matchAll(/`([a-zA-Z][a-zA-Z0-9_]*\.[a-zA-Z0-9_.]*)\$\{/g)].map((m) => m[1]))];
    const used = (k: string) => ['"', "'", "`"].some((q) => code.includes(`${q}${k}${q}`)) || prefixes.some((p) => k.startsWith(p));
    expect(keys.filter(([, k]) => !used(k)).map(([f, k]) => `${f}: ${k}`)).toEqual([]);
  });
});
