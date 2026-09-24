// Small helpers shared by the physics-content B topics (stochastic, circuit, design map, validation).
// Not a topic module: it has no default export, so the registry in ../index.ts ignores it.
import type { L10n } from "../types";

/** Bilingual string (ko first, en second). */
export const L = (ko: string, en: string): L10n => ({ ko, en });

/** Code paths used in several topics (relative to engine/ or server/). */
export const CODE = {
  compound: "model/janus_calibration_20260920/hypothesis_study_20260920/compound_fpt.py",
  conditional: "model/janus_calibration_20260920/hypothesis_study_20260920/conditional_table.py",
  avalanche: "model/janus_calibration_20260920/hypothesis_study_20260920/avalanche_clusters.py",
  luFpt: "model/janus_calibration_20260920/stl_mc_v3/lu_fpt.py",
  ldFpt: "model/janus_calibration_20260920/stl_mc_v3/ld_fpt.py",
  gateDyn: "model/janus_calibration_20260920/claude_crosscheck_20260920/joint_model/gate_dynamic_compare.py",
  checkEscape: "model/janus_calibration_20260920/claude_crosscheck_20260920/joint_model/check_escape.py",
  gateCal: "model/janus_calibration_20260920/claude_crosscheck_20260920/joint_model/gate_dynamic_calibration.json",
  gateLookup: "model/janus_calibration_20260920/claude_crosscheck_20260920/joint_model/gate_state_lookup.npz",
  gateFpt: "model/janus_calibration_20260920/reader_fig3_20260921/gate_model/gate_fpt.py",
  photoFpt: "photo_extension/photo_fpt.py",
  photoMean: "photo_extension/photo_mean.py",
  setupPhoto: "photo_extension/setup_photo.py",
  mcCycles: "photo_extension/mc_cycles.py",
  vgSweep: "photo_extension/vg_sweep.py",
  predict: "photo_extension/predict.py",
  hypotheses: "photo_extension/hypotheses.py",
  verifyHloc: "photo_extension/verify_hloc.py",
  calibrate: "photo_extension/calibrate.py",
  params: "server/params.py",
  stochastic: "server/compute/stochastic.py",
  stochCore: "server/compute/stoch_core.py",
  circuit: "server/compute/circuit/",
} as const;
