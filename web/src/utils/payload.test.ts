import { describe, expect, it } from "vitest";
import { BENCHES } from "../params/benches";
import { BUILTIN_META, PHOTO_GAMMA } from "../state/presets";
import {
  applyChannelSeed, branchesPayload, channelSeedOf, circuitPayload, iphPA, midFold, photoConditionPayload, powerMW, presetRoot,
  sweepMcPayload, switchLightMode, VG_CURVE_CANONICAL_VG, vgCurvePayload,
} from "./payload";
import { canonical, deepEqual, getPath, mergeDefaults, setPath } from "./object";

const paper = () => presetRoot(BUILTIN_META, "paper");
const photo = () => presetRoot(BUILTIN_META, "photo");

describe("unit conversions", () => {
  it("I_PH = R · P (power mode) and plain pA (iph mode)", () => {
    const d = photo().device;
    d.light.power_mW = 3.51;
    expect(iphPA(d)).toBeCloseTo(2.6325, 6);
    expect(powerMW(d)).toBe(3.51);
    const s = switchLightMode(d, "iph");
    expect(s.light.mode).toBe("iph");
    expect(s.light.iph_pA).toBeCloseTo(2.6325, 9);
    expect(iphPA(s)).toBeCloseTo(iphPA(d), 12);
    const back = switchLightMode(s, "power");
    expect(back.light.power_mW).toBeCloseTo(3.51, 9);
    expect(d.light.mode).toBe("power"); // original untouched
  });
  it("validation light conversion numbers (0.86 / 1.91 / 2.63 pA)", () => {
    const d = photo().device;
    const got = [1.15, 2.55, 3.51].map((p) => iphPA({ light: { ...d.light, power_mW: p } }));
    expect(got.map((x) => x.toFixed(2))).toEqual(["0.86", "1.91", "2.63"]);
  });
});

describe("payload builders", () => {
  it("branches / sweep_mc carry the full blocks (SI values as in params.py)", () => {
    const p = paper();
    const b = branchesPayload(p);
    expect(b.device.vg).toBe(-2);
    expect(b.device.calib.tau_bulk_s).toBeCloseTo(9.266283807625294e-7, 20);
    expect(b.sweep).toEqual({ vd_max_V: 4, rate_V_per_s: 0.4, dv_V: 0.002 });
    const m = sweepMcPayload(p);
    expect(m.stochastic.n_cycles).toBe(100);
    expect(m.stochastic.local_state.mode).toBe("evolving");
    expect(m.stochastic.ld_carrier_noise).toBe(true);
    // payloads are copies
    b.device.vg = 0;
    expect(p.device.vg).toBe(-2);
  });
  it("vg_curve rounds the point count", () => {
    expect(vgCurvePayload(paper(), { min: -4, max: -1, n: 12.6 }).n).toBe(13);
  });
  it("vg_curve uses a canonical V_G: a V_G change keeps the payload (and its cache key) unchanged", () => {
    const r = { min: -4.5, max: -0.5, n: 41 };
    const a = paper();
    const b = setPath(paper(), ["device", "vg"], -1.9);
    expect(vgCurvePayload(a, r).device.vg).toBe(VG_CURVE_CANONICAL_VG);
    expect(VG_CURVE_CANONICAL_VG).toBe(-2); // the reference preset's V_G: its snapshot key is unchanged
    expect(canonical(vgCurvePayload(b, r))).toBe(canonical(vgCurvePayload(a, r)));
    expect(b.device.vg).toBe(-1.9); // the parameter root is not modified
    // the photo preset (V_G −1.8 V) also sends −2; every other device field is passed through
    const ph = vgCurvePayload(photo(), r).device;
    expect(ph.vg).toBe(-2);
    expect(ph.light).toEqual(photo().device.light);
    // anything else still changes the key
    const c = setPath(paper(), ["device", "light", "iph_pA"], 1);
    expect(canonical(vgCurvePayload(c, r))).not.toBe(canonical(vgCurvePayload(a, r)));
  });
  it("circuit payload: stochastic block only in stochastic mode", () => {
    const p = paper();
    const det = circuitPayload(p, "deterministic");
    expect(det.stochastic).toBeUndefined();
    expect(det.bench).toBe("load_line");
    expect(det.bench_params).toEqual({ v_min_V: 0, n_cycles: 1, R_s_ohm: 1e3, C_d_F: 2e-15 }); // auto (null) keys omitted
    const sto = circuitPayload(p, "stochastic");
    expect((sto.stochastic as { n_runs: number }).n_runs).toBeGreaterThan(0);
    expect(sto.mode).toBe("stochastic");
  });
  it("photo condition payload", () => {
    const pl = photoConditionPayload(BUILTIN_META.presets.photo, -1.1, 2.55);
    expect(pl.device.vg).toBe(-1.1);
    expect(pl.device.light).toMatchObject({ mode: "power", power_mW: 2.55 });
    expect(pl.device.ext.gamma).toBeCloseTo(PHOTO_GAMMA, 12);
    expect(pl.sweep.rate_V_per_s).toBe(1200);
    expect(pl.stochastic.n_cycles).toBe(400);
  });
  it("midFold", () => {
    expect(midFold(3.7037, 2.5979)).toBe(3.151);
    expect(midFold(null, null, 3.2)).toBe(3.2);
  });
});

describe("channel-seed option", () => {
  const opts = BUILTIN_META.channel_seed_options;
  it("detects and applies", () => {
    expect(channelSeedOf(paper().device.ext, opts)).toBe("none");
    expect(channelSeedOf(photo().device.ext, opts)).toBe("body_coupling");
    const hv = applyChannelSeed(photo().device.ext, "high_vd_seed", opts);
    expect(hv.gamma).toBe(0);
    expect(hv.seed_ip_pA).toBe(1.33);
    expect(hv.seed_S).toBe(0.8);
    expect(channelSeedOf(hv, opts)).toBe("high_vd_seed");
    const none = applyChannelSeed(hv, "none", opts);
    expect(none).toMatchObject({ gamma: 0, seed_ip_pA: 0, seed_S: 1 });
    expect(channelSeedOf({ ...none, gamma: 0.1 }, opts)).toBe("custom");
  });
});

describe("object helpers", () => {
  it("setPath is immutable, getPath reads", () => {
    const p = paper();
    const q = setPath(p, ["device", "calib", "beta"], 2);
    expect(getPath(q, ["device", "calib", "beta"])).toBe(2);
    expect(p.device.calib.beta).not.toBe(2);
    expect(q.sweep).toBe(p.sweep); // untouched branches shared
  });
  it("deepEqual with tolerance; mergeDefaults drops unknown keys and wrong types", () => {
    expect(deepEqual({ a: 1, b: [1, 2] }, { a: 1 + 1e-13, b: [1, 2] })).toBe(true);
    expect(deepEqual({ a: 1 }, { a: 1, b: 2 })).toBe(false);
    const m = mergeDefaults({ a: 1, b: { c: "x", d: 2 } }, { a: 5, b: { c: 3, e: 1 }, z: 9 });
    expect(m).toEqual({ a: 5, b: { c: "x", d: 2 } });
  });
  it("canonical JSON sorts keys", () => {
    expect(canonical({ b: 1, a: [2, { d: 1, c: 2 }] })).toBe('{"a":[2,{"c":2,"d":1}],"b":1}');
  });
});

describe("circuit defaults mirror server/compute/circuit/benches.py", () => {
  it("auto (null) bench values are omitted so the server defaults apply; lists stay arrays", () => {
    const p = paper();
    const ll = circuitPayload(p, "deterministic");
    const llp = ll.bench_params as Record<string, unknown>;
    expect(llp).toMatchObject({ v_min_V: 0, n_cycles: 1, R_s_ohm: 1e3, C_d_F: 2e-15 });
    for (const k of ["v_max_V", "rate_V_per_s", "vg_V"]) expect(k in llp).toBe(false);
    expect((ll.solver as { dt_min_s: unknown }).dt_min_s).toBeNull(); // solver: server default is None too
    const pulse = circuitPayload({ ...p, circuit: { ...p.circuit, bench: "pulse" } }, "stochastic");
    const pp = pulse.bench_params as Record<string, unknown>;
    expect(pp).toMatchObject({ amplitudes_V: [], n_pulses: 10, width_s: 200e-6 });
    expect("v_amp_V" in pp).toBe(false);
    expect((pulse.stochastic as { seed: number }).seed).toBe(2026092920);
  });
  it("rise/fall edges default to auto (server: pulse/coupled 10 µs, p-bit 20 µs) and stay overridable", () => {
    const p = paper();
    for (const b of ["pulse", "pbit", "coupled"] as const) {
      expect(p.circuit.bench_params[b].rise_s).toBeNull();
      expect(p.circuit.bench_params[b].fall_s).toBeNull();
      const pl = circuitPayload({ ...p, circuit: { ...p.circuit, bench: b } }, "deterministic").bench_params as Record<string, unknown>;
      expect("rise_s" in pl || "fall_s" in pl).toBe(false);
    }
    const over = setPath(setPath(p, ["circuit", "bench"], "pbit"), ["circuit", "bench_params", "pbit", "rise_s"], 5e-6);
    expect((circuitPayload(over, "deterministic").bench_params as Record<string, unknown>).rise_s).toBe(5e-6);
  });
  it("unit conversions of bench fields: stored SI, displayed kΩ / fF / µs", () => {
    const f = (b: keyof typeof BENCHES, k: string) => BENCHES[b].fields.find((x) => x.key === k)!;
    expect(1e3 * f("load_line", "R_s_ohm").scale!).toBe(1); // 1 kΩ
    expect(2e-15 * f("load_line", "C_d_F").scale!).toBeCloseTo(2, 12); // 2 fF
    expect(200e-6 * f("pulse", "width_s").scale!).toBeCloseTo(200, 9); // 200 µs
    expect(100e3 * f("pbit", "R_S_ohm").scale!).toBe(100); // 100 kΩ
  });
  it("persisted auto fields survive mergeDefaults (null ↔ number, arrays)", () => {
    const base = paper().circuit.bench_params.pulse;
    const m = mergeDefaults(base, { v_amp_V: 3.9, amplitudes_V: [3.6, 3.8], R_s_ohm: "x", old_key: 1 });
    expect(m.v_amp_V).toBe(3.9);
    expect(m.amplitudes_V).toEqual([3.6, 3.8]);
    expect(m.R_s_ohm).toBe(1e3);
    expect("old_key" in m).toBe(false);
    expect(mergeDefaults(base, { v_amp_V: null }).v_amp_V).toBeNull();
  });
});
