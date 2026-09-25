// Built-in copy of server/params.py PRESETS (GET /api/meta). Used before /api/meta answers and in
// mock/offline mode. The live values from the backend always replace these when available.
import type { CalibBlock, DeviceBlock, DeviceGeometry, ExtBlock, Meta, PresetDef, PresetId, StochasticBlock } from "../api/types";
import { clone } from "../utils/object";
import { REFERENCE_GEOMETRY, resolveGeometry } from "../params/geometry";

export const RESPONSIVITY_PA_PER_MW = 0.7500000000000002;
export const PHOTO_GAMMA = 0.2794239352207005;
export const SIGMA_PHI_G_V = 0.15339035678526572;
export const SIGMA_PHI_E_V = 0.00043696121025623943;
export const TAU_E_S = 1.6205564347468981;
export const TAU_G_UP_S = 5.0;
export const PHOTO_DELTA_PHI_G0_V = 0.07443208588005665;
export const PHOTO_SIGMA_PHI_V = 0.21536460239140515;

// Device technology + geometry (server/params.py TECHNOLOGY / GEOMETRY, also sent per preset in /api/meta).
// Geometry is editable relative to this reference calibration. PDSOI and bulk remain unavailable.
export type Technology = "FDSOI" | "PDSOI" | "Bulk";
export type { DeviceGeometry } from "../api/types";
export interface PresetTech { technology: Technology; geometry: DeviceGeometry }
export const TECHNOLOGIES: { id: Technology; available: boolean }[] = [
  { id: "FDSOI", available: true },
  { id: "PDSOI", available: false },
  { id: "Bulk", available: false },
];
export const FDSOI_GEOMETRY: DeviceGeometry = { ...REFERENCE_GEOMETRY };
const GEOMETRY_TEXT = "L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm";
const FDSOI: PresetTech = { technology: "FDSOI", geometry: FDSOI_GEOMETRY };

const CALIB: CalibBlock = {
  beta: 7.166501201841884,
  tau_bulk_s: 9.266283807625294e-7,
  tau_junction_s: 5.358373401760838e-9,
  r_contact_ohm: 1.0000000009063326,
  l_gidl_nm: 28.754392875051614,
  t_access_nm: 3.312683950426714,
  na_access_cm3: 1.0002399957771266e17,
  l_access_nm: 70.0,
  tau_ratio: 117.42471623552643,
  phi_gidl0_V: 0.0002721229842976664,
  phi_emitter0_V: 5.908461490149247e-5,
  channel_ii_scale: 1.0,
};

export const NEUTRAL_EXT: ExtBlock = {
  dibl: 0, gamma: 0, kappa: 0, seed_ip_pA: 0, seed_S: 1, dj: 0, dm: 0, aloc: 0, isat_pA: 20, dloc: 0,
  loc_carriers: 0, kappaF: 0,
};

const DEVICE_BASE: DeviceBlock = {
  preset: "paper",
  geometry: { ...REFERENCE_GEOMETRY },
  vg: -2.0,
  vbg: 0,
  light: { mode: "iph", iph_pA: 0, power_mW: 0, responsivity_pA_per_mW: RESPONSIVITY_PA_PER_MW },
  calib: CALIB,
  ext: NEUTRAL_EXT,
  state: { delta_phi_G0_V: 0, delta_phi_E0_V: 0 },
  numerics: { grid: 601 },
};

const STOCH_PAPER: StochasticBlock = {
  n_cycles: 100, seed: 2026092920, carrier_noise: true, ld_carrier_noise: true,
  local_state: { mode: "evolving", action: "gidl", sigma: SIGMA_PHI_G_V, tau_s: TAU_G_UP_S, sigma_E_V: SIGMA_PHI_E_V, tau_E_s: TAU_E_S, acquisition_trend: true },
  engine: "auto", n_traces: 12, fold_nodes: 25, hazard_nodes: 5,
};

const paper: PresetDef & PresetTech = {
  label: {
    ko: `FDSOI · ${GEOMETRY_TEXT} — 기준 보정 (암조건, V_G = −2 V, 0.4 V/s)`,
    en: `FDSOI · ${GEOMETRY_TEXT} — reference calibration (dark, V_G = −2 V, 0.4 V/s)`,
  },
  ...FDSOI,
  device: clone({ ...DEVICE_BASE, preset: "paper" }),
  sweep: { vd_max_V: 4.0, rate_V_per_s: 0.4, dv_V: 0.002 },
  stochastic: clone(STOCH_PAPER),
};

const photo: PresetDef & PresetTech = {
  label: {
    ko: `FDSOI · ${GEOMETRY_TEXT} — 광조사 보정 (V_G = −1.8 V, 1200 V/s)`,
    en: `FDSOI · ${GEOMETRY_TEXT} — illumination calibration (V_G = −1.8 V, 1200 V/s)`,
  },
  ...FDSOI,
  device: clone({
    ...DEVICE_BASE,
    preset: "photo",
    vg: -1.8,
    light: { mode: "power", iph_pA: 0, power_mW: 0, responsivity_pA_per_mW: RESPONSIVITY_PA_PER_MW },
    ext: { ...NEUTRAL_EXT, gamma: PHOTO_GAMMA },
    state: { delta_phi_G0_V: PHOTO_DELTA_PHI_G0_V, delta_phi_E0_V: 0 },
  } as DeviceBlock),
  sweep: { vd_max_V: 5.0, rate_V_per_s: 1200.0, dv_V: 0.002 },
  stochastic: {
    n_cycles: 400, seed: 20260922, carrier_noise: true, ld_carrier_noise: true,
    local_state: { mode: "frozen", action: "gidl", sigma: PHOTO_SIGMA_PHI_V, tau_s: TAU_G_UP_S, sigma_E_V: 0, tau_E_s: TAU_E_S, acquisition_trend: false },
    engine: "auto", n_traces: 12, fold_nodes: 25, hazard_nodes: 5,
  },
};

const custom: PresetDef & PresetTech = {
  ...clone(paper),
  label: {
    ko: `FDSOI · ${GEOMETRY_TEXT} — 사용자 정의 (기준 보정 값에서 시작)`,
    en: `FDSOI · ${GEOMETRY_TEXT} — custom (starts from the reference calibration)`,
  },
  device: clone({ ...paper.device, preset: "custom" as PresetId }),
};

export const BUILTIN_META: Meta = {
  presets: { paper, photo, custom },
  channel_seed_options: {
    none: {},
    body_coupling: { gamma: PHOTO_GAMMA },
    high_vd_seed: { seed_ip_pA: 1.33, seed_S: 0.8 },
  },
  measured_photo_conditions: [
    { vg: -1.8, power_mW: 0 }, { vg: -1.8, power_mW: 1.15 }, { vg: -1.8, power_mW: 2.55 }, { vg: -1.8, power_mW: 3.51 },
    { vg: -1.1, power_mW: 0 }, { vg: -1.1, power_mW: 1.15 }, { vg: -1.1, power_mW: 2.55 }, { vg: -1.1, power_mW: 3.51 },
  ],
  constants: { NA_cm3: REFERENCE_GEOMETRY.Nbody_cm3, geometry: { L_nm: 500, W_nm: 200, T_Si_nm: 50, EOT_nm: 14.1, Tbox_nm: 140 } },
  technology: "FDSOI",
  geometry: FDSOI_GEOMETRY,
  technologies: TECHNOLOGIES,
};

export const PRESET_IDS: PresetId[] = ["paper", "photo", "custom"];

/** Technology + geometry of a preset: from /api/meta when the server sends them, else the built-in copy. */
export function presetTech(meta: Meta, id: PresetId): PresetTech {
  const p = meta.presets[id] as (PresetDef & Partial<PresetTech>) | undefined;
  const fb = (BUILTIN_META.presets[id] as PresetDef & PresetTech | undefined) ?? paper;
  return { technology: p?.technology ?? fb.technology, geometry: resolveGeometry(p?.device.geometry ?? p?.geometry ?? fb.geometry) };
}
