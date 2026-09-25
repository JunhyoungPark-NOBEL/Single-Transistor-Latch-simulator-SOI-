import { describe, expect, it } from "vitest";
import accepted from "./fixtures/csvm-payload.json";
import { BUILTIN_META } from "../state/presets";
import { canonical } from "../utils/object";
import { presetRoot } from "../utils/payload";
import { csvmPayload, DEFAULT_CSVM, restoreForcing } from "./forcing";
import { legacyGeometryPayload } from "../api/geometryPolicy";
import { REFERENCE_GEOMETRY } from "../params/geometry";

describe("device forcing", () => {
  it("matches the recorded legacy request after removing explicit reference geometry and zero back-gate bias", () => {
    const payload = csvmPayload(presetRoot(BUILTIN_META, "paper"), "deterministic", DEFAULT_CSVM);
    expect(payload.netlist.elements[3]).toMatchObject({ device: { geometry: REFERENCE_GEOMETRY, vbg: 0 } });
    // The recording predates these explicit defaults. Its accepted request and numerical result stay unchanged;
    // only the known-equivalent baseline fields are omitted for this compatibility comparison.
    expect(legacyGeometryPayload(payload)).toEqual(accepted);
  });
  it("preserves valid mode/settings and rejects corrupt persisted settings", () => {
    expect(restoreForcing("not JSON")).toEqual({ forcing: "vscm", settings: DEFAULT_CSVM });
    expect(restoreForcing(JSON.stringify({ forcing: "csvm", settings: { current_A: 2e-9, capacitance_F: -1, duration_s: "bad" } }))).toEqual({
      forcing: "csvm", settings: { ...DEFAULT_CSVM, current_A: 2e-9 },
    });
    expect(restoreForcing(JSON.stringify({ forcing: "invalid", settings: { duration_s: 1e50 } })).forcing).toBe("vscm");
  });
  it("uses positive drain current, selected capacitance, gate bias, and the full calibrated/light device", () => {
    const p = presetRoot(BUILTIN_META, "paper");
    p.device.vg = -1.8;
    p.device.light.iph_pA = 2.6;
    const req = csvmPayload(p, "deterministic", { current_A: 2e-9, capacitance_F: 3e-12, duration_s: .02 });
    expect(req.netlist.elements[0]).toMatchObject({ type: "I", nodes: ["0", "drain"], wave: { value: 2e-9 } });
    expect(req.netlist.elements[1]).toMatchObject({ type: "C", value: 3e-12 });
    expect(req.netlist.elements[2]).toMatchObject({ type: "V", wave: { value: -1.8 } });
    expect(req.netlist.elements[3]).toMatchObject({ type: "STL", device: p.device, light_pA: null });
    expect(req.tran.t_stop_s).toBe(.02);
    expect(req.stochastic).toBeUndefined();
  });
  it("preserves stochastic noise, seed, local state and uses one time-domain trace", () => {
    const p = presetRoot(BUILTIN_META, "paper");
    p.stochastic.seed = 101;
    p.stochastic.carrier_noise = false;
    p.stochastic.local_state.mode = "frozen";
    const req = csvmPayload(p, "stochastic", DEFAULT_CSVM);
    expect(req.stochastic).toMatchObject({ seed: 101, carrier_noise: false, n_runs: 1, local_state_override: true, local_state: { mode: "frozen" } });
  });
  it("invalidates results for forcing, calibration or noise changes but not irrelevant voltage-sweep settings", () => {
    const p = presetRoot(BUILTIN_META, "paper");
    const key = () => canonical(csvmPayload(p, "stochastic", DEFAULT_CSVM));
    const base = key();
    p.sweep.vd_max_V += 1;
    p.stochastic.n_cycles += 20;
    expect(key()).toBe(base);
    p.device.vg += .1;
    expect(key()).not.toBe(base);
    expect(canonical(csvmPayload(p, "stochastic", { ...DEFAULT_CSVM, current_A: 2e-9 }))).not.toBe(key());
    p.stochastic.seed += 1;
    expect(key()).not.toBe(base);
  });
});
