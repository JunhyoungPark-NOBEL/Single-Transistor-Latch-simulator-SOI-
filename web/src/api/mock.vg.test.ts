import { describe, expect, it } from "vitest";
import { BUILTIN_META } from "../state/presets";
import { mockBranches, mockVgCurve } from "./mock";

// The demo backend must respond to V_G like the model does qualitatively (it is still labelled demo data):
// model at V_D = 3 V, V_G −2 → −1.5 V: GIDL 1.21e-13 → 6.75e-15 A, channel 9.9e-22 → 5.2e-17 A, V_LU 3.704 → 4.111 V.
describe("mock responds to V_G", () => {
  const pr = BUILTIN_META.presets.paper;
  const at = (vg: number) => {
    const r = mockBranches({ device: { ...pr.device, vg }, sweep: pr.sweep });
    const i = r.HRS.vd.findIndex((v) => (v as number) >= 3);
    return { r, gidl: r.HRS.comp.gidl[i] as number, channel: r.HRS.comp.channel[i] as number, id: r.HRS.id[i] as number };
  };

  it("GIDL falls ≈ ×18 and the channel current rises ≈ 4.7 decades per +0.5 V; V_LU rises ≈ 0.4 V", () => {
    const a = at(-2);
    const b = at(-1.5);
    expect(a.r.folds.V_LU).toBeCloseTo(3.704, 2);
    expect(a.gidl).toBeGreaterThan(8e-14);
    expect(a.gidl).toBeLessThan(2e-13);
    expect(a.gidl / b.gidl).toBeGreaterThan(10);
    expect(a.gidl / b.gidl).toBeLessThan(30);
    expect(Math.log10(b.channel / a.channel)).toBeGreaterThan(4);
    expect(Math.log10(b.channel / a.channel)).toBeLessThan(5.5);
    expect(b.r.folds.V_LU! - a.r.folds.V_LU!).toBeGreaterThan(0.3);
    expect(b.r.folds.V_LU! - a.r.folds.V_LU!).toBeLessThan(0.5);
    expect(b.id).toBeLessThan(a.id); // the HRS current is GIDL-dominated at V_D = 3 V
  });

  it("the V_G curve's fold current follows V_G", () => {
    const c = mockVgCurve({ device: pr.device, vg_min: -2.5, vg_max: -1.5, n: 3 });
    const ilu = c.I_LU.filter((x): x is number => typeof x === "number");
    expect(ilu).toHaveLength(3);
    expect(new Set(ilu).size).toBe(3);
  });
});
