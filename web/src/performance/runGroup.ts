// The exact jobs started by the active Run action, including client-side reuse and dependencies.
import type { State } from "../state/store";
import type { LayoutMode } from "../state/layout";
import type { CustomCircuitRequest } from "../api/circuitCustom";
import type { BranchesResult } from "../api/types";
import { csvmPayload, type CsvmSettings, type ForcingMode } from "../device/forcing";
import { canonical } from "../utils/object";
import { branchesPayload, chargeBalancePayload, circuitPayload, hazardPayload, midFold, sweepMcPayload, vgCurvePayload } from "../utils/payload";
import type { EstimateJob } from "./types";

export interface RunContext { layout: LayoutMode; forcing: ForcingMode; csvm: CsvmSettings; circuitView: string; customRequest?: CustomCircuitRequest }
export function currentRunJobs(s: State, ctx: RunContext): EstimateJob[] {
  const p = s.params;
  const branches: EstimateJob = { key: "branches", kind: "branches", payload: branchesPayload(p) };
  if (s.tab === "circuit") return ctx.circuitView === "schematic"
    ? ctx.customRequest ? [{ key: "schematic", kind: "circuit", payload: ctx.customRequest }] : []
    : [{ key: "circuit", kind: "circuit", payload: circuitPayload(p, s.mode) }, { ...branches, key: "circuit_branches" }];
  if (s.tab !== "device") return [];
  if (ctx.forcing === "csvm") return [{ key: "device_csvm", kind: "circuit", payload: csvmPayload(p, s.mode, ctx.csvm) }];
  if (s.mode === "stochastic") return [
    { key: "sweep_mc", kind: "sweep_mc", payload: sweepMcPayload(p) }, branches,
    ...(ctx.layout === "all" ? [{ key: "hazard", kind: "hazard" as const, payload: hazardPayload(p) }] : []),
  ];
  if (ctx.layout === "simple") return [branches];
  const jobs = [branches];
  const vg = vgCurvePayload(p, s.vgRange);
  const oldVg = s.results.vg_curve;
  if (!(oldVg?.status === "done" && oldVg.data !== undefined && oldVg.dataKey === canonical({ kind: "vg_curve", payload: vg }))) {
    jobs.push({ key: "vg_curve", kind: "vg_curve", payload: vg });
  }
  if (p.device.model !== "simple") {
    const oldBranches = s.results.branches;
    const matching = oldBranches?.dataKey === canonical({ kind: branches.kind, payload: branches.payload });
    const folds = matching ? (oldBranches.data as BranchesResult | undefined)?.folds : undefined;
    // Automatic charge-balance voltage is resolved from the next branches result. This initial
    // mid-fold value affects its state search, not the requested number of evaluation points.
    const vd = s.cbVd ?? midFold(folds?.V_LU, folds?.V_LD, 0.8 * p.sweep.vd_max_V);
    jobs.push({ key: "charge_balance", kind: "charge_balance", payload: chargeBalancePayload(p, vd), ...(s.cbVd === null ? { depends_on: ["branches"] } : {}) });
  }
  return jobs;
}
