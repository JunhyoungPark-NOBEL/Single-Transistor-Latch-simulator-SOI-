import { describe, expect, it } from "vitest";
import { fitScale, MIN_FIT_SCALE } from "../components/Tex";
import { modeHint } from "../components/Header";
import { translate } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { presetRoot } from "../utils/payload";
import { parseOpenState, parsePersisted, PERSIST_VERSION, restoreParams, restoreRange } from "./persist";
import { BUILTIN_META } from "./presets";
import { runContext } from "./runner";

const base = () => presetRoot(BUILTIN_META, "paper");

describe("parsePersisted (localStorage from junk / older schemas)", () => {
  it("never throws and keeps only valid fields", () => {
    for (const raw of [null, "", "{not json", "null", "[1,2,3]", "42", '"str"']) expect(parsePersisted(raw)).toEqual({});
    const p = parsePersisted(JSON.stringify({ tab: "zzz", mode: "foo", lang: "fr", theme: 7, preset: "weird", autoRun: "yes", params: "x", vgRange: [1] }));
    expect(p).toEqual({ v: 1 });
    const ok = parsePersisted(JSON.stringify({ v: 2, tab: "circuit", mode: "stochastic", lang: "en", theme: "dark", preset: "photo", autoRun: true, params: {} }));
    expect(ok).toMatchObject({ v: 2, tab: "circuit", mode: "stochastic", lang: "en", theme: "dark", preset: "photo", autoRun: true, params: {} });
  });
});

describe("restoreParams", () => {
  it("falls back to defaults for ill-typed blocks and invalid enums", () => {
    const b = base();
    const p = restoreParams(b, {
      device: { vg: "abc", light: { mode: "laser", iph_pA: 3 }, ext: { loc_carriers: 7 } },
      stochastic: { local_state: { mode: "sometimes", action: "magic", sigma: 0.2 }, engine: "fast" },
      circuit: { bench: "nope", bench_params: 5, solver: { method: "RK4", reltol: 1e-4 }, stochastic: { local_state: 3 } },
    });
    expect(p.device.vg).toBe(-2);
    expect(p.device.light).toMatchObject({ mode: "iph", iph_pA: 3 });
    expect(p.device.ext.loc_carriers).toBe(0);
    expect(p.stochastic.local_state).toMatchObject({ mode: "evolving", action: "gidl", sigma: 0.2 });
    expect(p.stochastic.engine).toBe("auto");
    expect(p.circuit.bench).toBe("load_line");
    expect(p.circuit.bench_params).toEqual(b.circuit.bench_params);
    expect(p.circuit.solver).toMatchObject({ method: "BE", reltol: 1e-4 });
    expect(p.circuit.stochastic.local_state).toEqual(b.circuit.stochastic.local_state);
  });
  it("cleans numeric lists and migrates the old built-in 1 µs edges (v1) to auto", () => {
    const b = base();
    const stored = { circuit: { bench: "pulse", bench_params: { pulse: { amplitudes_V: [3.6, "x", null, 3.8], rise_s: 1e-6, fall_s: 1e-6 }, pbit: { rise_s: 2e-6 } } } };
    const v1 = restoreParams(b, stored, 1);
    expect(v1.circuit.bench).toBe("pulse");
    expect(v1.circuit.bench_params.pulse.amplitudes_V).toEqual([3.6, 3.8]);
    expect(v1.circuit.bench_params.pulse.rise_s).toBeNull();
    expect(v1.circuit.bench_params.pulse.fall_s).toBeNull();
    expect(v1.circuit.bench_params.pbit.rise_s).toBe(2e-6); // a real user choice is kept
    const v2 = restoreParams(b, stored, PERSIST_VERSION);
    expect(v2.circuit.bench_params.pulse.rise_s).toBe(1e-6); // explicit value saved by the current schema
  });
  it("restoreRange keeps ordered finite ranges and clamps n", () => {
    const d = { min: -4.5, max: -0.5, n: 41 };
    expect(restoreRange(d, { min: -3, max: -1, n: 500 })).toEqual({ min: -3, max: -1, n: 61 });
    expect(restoreRange(d, { min: 1, max: -1, n: 1 })).toEqual({ min: -4.5, max: -0.5, n: 2 });
    expect(restoreRange(d, "x")).toEqual(d);
  });
  it("parseOpenState", () => {
    expect(parseOpenState("[1,2")).toEqual({});
    expect(parseOpenState("null")).toEqual({});
    expect(parseOpenState('{"bias":false,"x":3}')).toEqual({ bias: false });
  });
});

describe("display-equation fit", () => {
  it("fits, scales down to 80 %, then scrolls", () => {
    expect(fitScale(400, 472)).toEqual({ scale: 1, scroll: false });
    const s = fitScale(489, 472);
    expect(s.scroll).toBe(false);
    expect(s.scale).toBeLessThan(1);
    expect(489 * s.scale).toBeLessThanOrEqual(472);
    expect(fitScale(1000, 472)).toEqual({ scale: MIN_FIT_SCALE, scroll: true });
    expect(fitScale(0, 472)).toEqual({ scale: 1, scroll: false });
  });
});

describe("mode banner text follows the tab", () => {
  const mk = (lang: "ko" | "en") =>
    Object.assign((k: StrKey, v?: Record<string, string | number>) => translate(lang, k, v), { lang, l: () => "" });
  it("circuit tab describes the transient integrator / event increments, not the device model", () => {
    for (const lang of ["ko", "en"] as const) {
      const t = mk(lang);
      const det = modeHint(t, "circuit", "deterministic", "TRAP");
      const sto = modeHint(t, "circuit", "stochastic", "BE");
      expect(det).toContain("TRAP");
      expect(det).toContain("MNA");
      expect(det).not.toContain("fold");
      expect(sto).toContain("Eq. 2");
      expect(sto).toContain("Q_B");
      expect(modeHint(t, "device", "deterministic", "BE")).toBe(translate(lang, "mode.deterministic.hint"));
      expect(modeHint(t, "validation", "stochastic", "BE")).toBe(translate(lang, "mode.validation.hint"));
    }
  });
  it("run-bar context", () => {
    expect(runContext("device", "stochastic")).toBe("stochastic");
    expect(runContext("circuit", "deterministic")).toBe("circuit deterministic");
    expect(runContext("physics", "deterministic")).toBeNull();
  });
});
