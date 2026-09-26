import { describe, expect, it } from "vitest";
import { useStore, type State } from "../state/store";
import { branchesPayload, vgCurvePayload } from "../utils/payload";
import { canonical } from "../utils/object";
import { csvmPayload, DEFAULT_CSVM } from "../device/forcing";
import { currentRunJobs, type RunContext } from "./runGroup";
const context: RunContext = { layout: "simple", forcing: "vscm", csvm: DEFAULT_CSVM, circuitView: "schematic" };
const state = (): State => ({ ...useStore.getState(), tab: "device", mode: "deterministic", results: {}, cbVd: null });

describe("pre-run estimate matches the run action", () => {
  it("includes only a branch job in the compact voltage workspace", () => {
    const s = state();
    expect(currentRunJobs(s, context)).toEqual([{ key: "branches", kind: "branches", payload: branchesPayload(s.params) }]);
  });
  it("includes all expert jobs and branches-to-charge dependency", () => {
    const s = state();
    const jobs = currentRunJobs(s, { ...context, layout: "all" });
    expect(jobs.map(j => j.kind)).toEqual(["branches", "vg_curve", "charge_balance"]);
    expect(jobs[2].depends_on).toEqual(["branches"]);
    s.cbVd = 2.6;
    expect(currentRunJobs(s, { ...context, layout: "all" })[2].depends_on).toBeUndefined();
  });
  it("omits an exact reusable VG result and unsupported Simple charge balance", () => {
    const s = state();
    s.params = { ...s.params, device: { ...s.params.device, model: "simple" } };
    s.results.vg_curve = { status: "done", data: {}, kind: "vg_curve", progress: 1, message: "", token: 1, dataKey: canonical({ kind: "vg_curve", payload: vgCurvePayload(s.params, s.vgRange) }) };
    expect(currentRunJobs(s, { ...context, layout: "all" }).map(j => j.kind)).toEqual(["branches"]);
  });
  it("includes the entire stochastic group", () => {
    const s = { ...state(), mode: "stochastic" as const };
    expect(currentRunJobs(s, { ...context, layout: "all" }).map(j => j.kind)).toEqual(["sweep_mc", "branches", "hazard"]);
  });
  it("uses exact current CSVM and custom circuit requests", () => {
    const s = state();
    const settings = { ...DEFAULT_CSVM, current_A: 2e-9, duration_s: .003 };
    const req = csvmPayload(s.params, s.mode, settings);
    expect(currentRunJobs(s, { ...context, forcing: "csvm", csvm: settings })[0].payload).toEqual(req);
    s.tab = "circuit";
    expect(currentRunJobs(s, { ...context, customRequest: req })).toEqual([{ key: "schematic", kind: "circuit", payload: req }]);
    expect(currentRunJobs(s, context)).toEqual([]);
  });
  it("never estimates a hidden Run on performance or guide tabs", () => {
    expect(currentRunJobs({ ...state(), tab: "performance" }, context)).toEqual([]);
    expect(currentRunJobs({ ...state(), tab: "physics" }, context)).toEqual([]);
  });
});
