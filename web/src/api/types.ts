// TypeScript mirror of docs/WEB_CONTRACT.md §1 (payload blocks), §2 (result shapes), §3 (JobStatus)
// and §4 (circuit). Keep in sync with the contract; the backend returns NaN/inf as null.
import type { L10n } from "../content/physics/types";

export type Arr = (number | null)[];
export type PresetId = "paper" | "photo" | "custom";
export type Mode = "deterministic" | "stochastic";

// ---------------------------------------------------------------- payload blocks (§1)
export interface LightBlock {
  mode: "iph" | "power";
  iph_pA: number;
  power_mW: number;
  responsivity_pA_per_mW: number;
}
export interface CalibBlock {
  beta: number; tau_bulk_s: number; tau_junction_s: number; r_contact_ohm: number; l_gidl_nm: number;
  t_access_nm: number; na_access_cm3: number; l_access_nm: number; tau_ratio: number;
  phi_gidl0_V: number; phi_emitter0_V: number; channel_ii_scale: number;
}
export interface ExtBlock {
  dibl: number; gamma: number; kappa: number; seed_ip_pA: number; seed_S: number; dj: number; dm: number;
  aloc: number; isat_pA: number; dloc: number; loc_carriers: number; kappaF: number;
}
export interface DeviceBlock {
  preset: PresetId;
  vg: number;
  light: LightBlock;
  calib: CalibBlock;
  ext: ExtBlock;
  state: { delta_phi_G0_V: number; delta_phi_E0_V: number };
  numerics: { grid: number };
}
export interface SweepBlock { vd_max_V: number; rate_V_per_s: number; dv_V: number }
export type LocalStateMode = "none" | "frozen" | "evolving";
export type LocalStateAction = "gidl" | "local_avalanche" | "junction" | "multiplication";
export interface LocalStateBlock {
  mode: LocalStateMode;
  action: LocalStateAction;
  sigma: number;
  tau_s: number;
  sigma_E_V: number;
  tau_E_s: number;
  acquisition_trend: boolean;
}
export interface StochasticBlock {
  n_cycles: number;
  seed: number;
  carrier_noise: boolean;
  ld_carrier_noise: boolean;
  local_state: LocalStateBlock;
  engine: "auto" | "general" | "calibrated_lookup";
  n_traces: number;
  fold_nodes: number;
  hazard_nodes: number;
}
export interface PresetDef {
  label: L10n;
  device: DeviceBlock;
  sweep: SweepBlock;
  stochastic: StochasticBlock;
}
export interface Meta {
  presets: Record<PresetId, PresetDef>;
  constants?: Record<string, unknown>;
  channel_seed_options: Record<string, Partial<ExtBlock>>;
  measured_photo_conditions?: { vg: number; power_mW: number }[];
  kinds?: string[];
  caps?: Record<string, number>;
  [k: string]: unknown;
}
export interface Health { ok: boolean; version?: string; workers?: number }

// ---------------------------------------------------------------- jobs (§3)
export type JobState = "queued" | "running" | "done" | "error" | "cancelled";
export interface JobStatus<T = unknown> {
  job_id: string;
  kind: string;
  status: JobState;
  progress: number;
  message: string;
  result?: T;
  error?: string;
  cached: boolean;
  elapsed_s: number;
}
export type Kind =
  | "branches" | "charge_balance" | "vg_curve" | "hazard" | "sweep_mc" | "vg_curve_stochastic"
  | "circuit" | "validation";

// ---------------------------------------------------------------- results (§2)
export interface Stats {
  n: number; mean: number | null; sd: number | null; median: number | null;
  p05: number | null; p95: number | null; min: number | null; max: number | null;
  censored: number; lag1: number | null;
}
export interface Components {
  channel: Arr; seed: Arr; ii_total: Arr; btbt_junction: Arr; gidl: Arr; photo: Arr;
  loss_bulk_srh: Arr; loss_diffusion: Arr; loss_junction_srh: Arr; net_F: Arr;
  hole_drop_V: Arr; injection: Arr; r_access_ohm: Arr;
}
export interface Curve { vd: Arr; id: Arr; u: Arr; r: Arr; comp: Components }
export interface XY { vd: Arr; id: Arr }
export interface Folds {
  V_LU: number | null; V_LD: number | null; I_LU: number | null; I_LD: number | null;
  u_LU: number | null; u_LD: number | null; window_V: number | null;
}
export interface Common { runtime_s: number; warnings: string[] }
export interface BranchesResult extends Common {
  latch: boolean;
  HRS: Curve; unstable: Curve; LRS: Curve; full: Curve;
  folds: Folds;
  double_sweep: { up: XY; down: XY };
  iph_A: number; p: number[];
}
export interface ChargeBalanceResult extends Common {
  vd: number; u: Arr; r: Arr; id: Arr; Q_C: Arr;
  generation_A: Arr; loss_A: Arr; unit_A: Arr; ii_A: Arr; F_A: Arr; potential: Arr;
  roots: { u: number; Q_C: number; kind: "stable" | "unstable"; id: number }[];
}
export interface VgCurveResult extends Common {
  vg: Arr; V_LU: Arr; V_LD: Arr; I_LU: Arr; latch: boolean[];
  window: { vg_low: number | null; vg_high: number | null };
}
export interface HazardResult extends Common {
  fold_V: number | null; VLD_fold_V: number | null; voltage: Arr; hazard: Arr; survival: Arr;
  quantiles: { prob: Arr; v: Arr }; stats: Stats; rate_V_per_s: number;
}
export interface Hist { edges: number[]; counts: number[] }
export interface Cdf { v: number[]; p: number[] }
export interface SweepTrace { cycle: number; V_LU: number | null; V_LD: number | null; up: XY; down: XY }
export interface SweepMCResult extends Common {
  engine: "calibrated_lookup" | "general";
  V_LU: Arr; V_LD: Arr;
  stats: { LU: Stats; LD: Stats };
  hist: { LU: Hist; LD: Hist };
  cdf: { LU: Cdf; LD: Cdf };
  traces: SweepTrace[];
  cycle_state: Arr;
  fold_table: { delta: Arr; V_LU: Arr; V_LD: Arr } | null;
  centre: { V_LU: number | null; V_LD: number | null; HRS: XY; LRS: XY };
  measured: { label: string; V_LU: Arr; V_LD: Arr | null; stats: { LU: Stats; LD: Stats | null } } | null;
}
export interface VgCurveStochasticResult extends Common {
  vg: Arr; mean_VLU: Arr; sd_VLU_mV: Arr; state_sd_mV: Arr; noise_sd_mV: Arr;
  fold_centre_V: Arr; VLD_fold_V: Arr; no_latch_weight: Arr;
  measured: { vg: number; power_mW: number; mean_V: number; sd_mV: number }[];
}
export interface ValidationCheck {
  id: string; label: L10n; expected: string; computed: string; pass: boolean | null;
  tolerance: string; note?: string; seconds: number;
}
export interface ValidationResult extends Common { checks: ValidationCheck[] }

// ---------------------------------------------------------------- circuit (§4)
export type BenchId = "load_line" | "pulse" | "pbit" | "coupled";
export interface Signal {
  key: string; label: L10n; unit: string; values: Arr;
  axis?: "voltage" | "current" | "charge" | "state" | "logic";
}
export interface CircuitRun { run: number; t: Arr; signals: Signal[] }
export interface SummaryItem { key: string; label: L10n; value: number | string | null; unit?: string; spread?: number | null }
export interface CircuitEvent { run: number; kind: string; t: number; value?: number; v_src?: number; v_d?: number }
export interface SchematicElement { kind: "V" | "R" | "C" | "STL" | "I" | "CMP" | string; name: string; nodes: string[]; value?: string }
export interface CircuitResult extends Common {
  bench: string; mode: string;
  runs: CircuitRun[];
  events: CircuitEvent[];
  summary: SummaryItem[];
  distributions?: { key: string; label: L10n; unit: string; values: Arr }[];
  sweeps?: { key: string; label: L10n; x: Arr; x_label: string; x_unit: string; y: Arr; y_label: string; y_unit: string; y_err?: Arr }[];
  trajectory?: XY;
  schematic: { nodes: string[]; elements: SchematicElement[] };
  solver_stats: { steps: number; rejected: number; newton_iters: number; runtime_s: number };
}
export interface SolverBlock { method: "BE" | "TRAP"; dt_min_s: number; dt_max_s: number; reltol: number; max_steps: number }
export interface CircuitStochBlock { seed: number; n_runs: number; carrier_noise: boolean; local_state: LocalStateBlock }

// ---------------------------------------------------------------- data endpoints (normalised, see api/measured.ts)
export interface MeasuredPhotoCondition {
  vg: number; power_mW: number; label: string; n: number; mean_V: number; sd_mV: number; lag1: number | null;
  raw?: number[];
}
export interface MeasuredLightCurve { label: string; power_mW: number | null; vd: number[]; id: number[] }
export interface MeasuredPaperIV {
  vd_up: number[]; median_up: number[]; p10_up: number[]; p90_up: number[];
  vd_down: number[]; median_down: number[]; p10_down: number[]; p90_down: number[];
  V_LU: number[]; V_LD: number[];
}
export interface MeasuredData {
  photo: MeasuredPhotoCondition[];
  light_iv: MeasuredLightCurve[];
  paper_iv: MeasuredPaperIV | null;
}
export interface DesignMapData {
  length_nm: number[];
  depth_fraction: number[];
  fields: Record<string, (number | null)[][]>;
  lines: { Nt: number[]; L0_device: number[]; L0_50: number[] } | null;
  scalars: Record<string, number>;
}
