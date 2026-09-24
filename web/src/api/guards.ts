// Lightweight runtime shape guards for compute results. A panel only renders a result that passes its
// guard; otherwise it shows "unexpected result shape (missing: …)" instead of crashing.
import type { Kind } from "./types";

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const isArr = (v: unknown): v is unknown[] => Array.isArray(v);

type Check = [path: string, test: (v: unknown) => boolean];
const arr: Check[1] = isArr;
const obj: Check[1] = isObj;
const numOrNull: Check[1] = (v) => v === null || typeof v === "number";

const SPECS: Record<Kind, Check[]> = {
  branches: [
    ["HRS.vd", arr], ["HRS.id", arr], ["LRS.vd", arr], ["LRS.id", arr], ["unstable.vd", arr], ["folds", obj],
    ["folds.V_LU", numOrNull], ["folds.V_LD", numOrNull], ["double_sweep.up.vd", arr], ["double_sweep.down.vd", arr],
  ],
  charge_balance: [["u", arr], ["generation_A", arr], ["loss_A", arr], ["potential", arr], ["roots", arr]],
  vg_curve: [["vg", arr], ["V_LU", arr], ["V_LD", arr], ["window", obj]],
  hazard: [["voltage", arr], ["hazard", arr], ["survival", arr], ["stats", obj]],
  sweep_mc: [["V_LU", arr], ["V_LD", arr], ["stats.LU", obj], ["stats.LD", obj], ["hist.LU.edges", arr], ["cdf.LU.v", arr], ["traces", arr], ["centre", obj]],
  vg_curve_stochastic: [["vg", arr], ["mean_VLU", arr], ["sd_VLU_mV", arr]],
  circuit: [["runs", arr], ["summary", arr], ["schematic.nodes", arr], ["schematic.elements", arr]],
  validation: [["checks", arr]],
};

function at(v: unknown, path: string): unknown {
  let cur = v;
  for (const k of path.split(".")) {
    if (!isObj(cur)) return undefined;
    cur = cur[k];
  }
  return cur;
}

/** Returns the list of missing/ill-typed paths (empty when the result looks valid). */
export function checkResult(kind: Kind, v: unknown): string[] {
  if (!isObj(v)) return ["<result is not an object>"];
  return (SPECS[kind] ?? []).filter(([p, test]) => !test(at(v, p))).map(([p]) => p);
}

export function isResult(kind: Kind, v: unknown): boolean {
  return checkResult(kind, v).length === 0;
}

/** Filter an array of maybe-null numbers into finite numbers. */
export function finite(a: readonly (number | null | undefined)[] | null | undefined): number[] {
  return (a ?? []).filter((x): x is number => typeof x === "number" && Number.isFinite(x));
}
