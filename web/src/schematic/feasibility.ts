// Rough run-cost estimate for the live feasibility hint (heuristic — the server does the real check and
// returns its own `feasibility` block). Deterministic: steps ≈ t_stop/Δt_max + ~30 steps per source corner
// + ~300 per possible latch transition. Stochastic: the carrier noise is resolved near the folds with small
// steps, so the cost grows with the simulated time spent near a fold (taken as 30 % of t_stop at ~1 µs steps).
// Calibrated on the live backend (4 CPUs): ≈ 50 µs per step and STL deterministic, ≈ 70–90 µs stochastic
// (examples: pulse train 13 k steps 0.9 s; p-bit 37 k steps 2.0 s; 8 stochastic pulse runs 7.8 s).
import type { Mode } from "../api/types";
import type { SchematicDoc } from "./model";
import { effectiveTran } from "./netlist";
import { breakpointCount } from "./waves";

export interface Estimate {
  stepsPerRun: number;
  runs: number;
  totalSteps: number;
  seconds: number;
  level: "ok" | "slow" | "heavy" | "refuse";
}

export const STEP_CAP = 2_000_000;
const NOISE_H = 1e-6; // effective stochastic step near a fold (s)

export function estimate(doc: SchematicDoc, mode: Mode): Estimate {
  const tr = effectiveTran(doc.tran);
  const T = Math.max(tr.t_stop_s, 0);
  const els = doc.elements;
  const nStl = els.filter((e) => e.kind === "STL").length;
  let corners = 0;
  for (const e of els) corners += breakpointCount(e.wave, T) + breakpointCount(e.light ?? null, T);
  const base = tr.dt_max_s > 0 ? T / tr.dt_max_s : 1e9;
  // every source corner may trigger a latch transition in each cell
  const transitions = nStl ? Math.min(corners, 4000) * 0.5 * nStl : 0;
  const tolScale = Math.max(1, Math.log10(1e-3 / Math.max(tr.reltol, 1e-9)) + 1);
  let steps = (base + 30 * corners + 300 * transitions * tolScale) * (tr.method === "TRAP" ? 1.1 : 1);
  if (mode === "stochastic" && nStl) steps += (nStl * 0.3 * T) / NOISE_H;
  steps = Math.max(50, Math.round(steps));
  const runs = mode === "stochastic" ? Math.max(1, Math.round(doc.stoch.n_runs)) : 1;
  const perStep = (mode === "stochastic" ? 75e-6 : 50e-6) * Math.max(1, nStl) + 0.5e-6 * els.length; // per step
  const seconds = 0.2 + steps * runs * perStep;
  const level: Estimate["level"] = steps > STEP_CAP ? "refuse" : seconds > 60 || steps > 0.5 * STEP_CAP ? "heavy" : seconds > 10 ? "slow" : "ok";
  return { stepsPerRun: steps, runs, totalSteps: steps * runs, seconds, level };
}
