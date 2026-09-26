// Device forcing is independent of the schematic editor and voltage-sweep configuration.
import { create } from "zustand";
import type { CustomCircuitRequest } from "../api/circuitCustom";
import type { Mode } from "../api/types";
import type { ParamRoot } from "../params/schema";

export type ForcingMode = "vscm" | "csvm";
export interface CsvmSettings { current_A: number; capacitance_F: number; duration_s: number }
export const DEFAULT_CSVM: CsvmSettings = { current_A: 1e-9, capacitance_F: 1e-12, duration_s: 15e-3 };
export const CSVM_LIMITS: Record<keyof CsvmSettings, readonly [number, number]> = {
  current_A: [1e-15, 1], capacitance_F: [1e-18, 1], duration_s: [1e-9, 10],
};
const STORAGE_KEY = "stl-device-forcing:v1";

export function restoreForcing(raw: string | null): { forcing: ForcingMode; settings: CsvmSettings } {
  let v: Record<string, unknown> = {};
  try { const parsed: unknown = raw ? JSON.parse(raw) : {}; if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) v = parsed as Record<string, unknown>; } catch { /* defaults */ }
  const settings = { ...DEFAULT_CSVM };
  const stored = v.settings && typeof v.settings === "object" ? v.settings as Record<string, unknown> : {};
  for (const key of Object.keys(settings) as (keyof CsvmSettings)[]) {
    const n = stored[key]; const [lo, hi] = CSVM_LIMITS[key];
    if (typeof n === "number" && Number.isFinite(n) && n >= lo && n <= hi) settings[key] = n;
  }
  return { forcing: v.forcing === "csvm" ? "csvm" : "vscm", settings };
}
function initial() { try { return restoreForcing(localStorage.getItem(STORAGE_KEY)); } catch { return restoreForcing(null); } }
export const useForcing = create<{
  forcing: ForcingMode; settings: CsvmSettings;
  setForcing: (forcing: ForcingMode) => void;
  setSetting: (key: keyof CsvmSettings, value: number) => void;
}>((set) => ({
  ...initial(),
  setForcing: (forcing) => set({ forcing }),
  setSetting: (key, value) => {
    const [lo, hi] = CSVM_LIMITS[key];
    if (Number.isFinite(value) && value >= lo && value <= hi) set((s) => ({ settings: { ...s.settings, [key]: value } }));
  },
}));
useForcing.subscribe((s) => { try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ forcing: s.forcing, settings: s.settings })); } catch { /* optional persistence */ } });

/** Positive Iin flows into drain; all other inputs are the current calibrated Device-tab state. */
export function csvmPayload(params: ParamRoot, mode: Mode, settings: CsvmSettings): CustomCircuitRequest {
  const { device, stochastic } = params;
  return {
    bench: "custom", mode,
    netlist: { elements: [
      { type: "I", name: "Iin", nodes: ["0", "drain"], wave: { kind: "dc", value: settings.current_A } },
      { type: "C", name: "Cdrain", nodes: ["drain", "0"], value: settings.capacitance_F },
      { type: "V", name: "VG", nodes: ["gate", "0"], wave: { kind: "dc", value: device.vg } },
      { type: "STL", name: "X1", nodes: { d: "drain", g: "gate", s: "0" }, device, light_pA: null, ...(mode === "stochastic" ? { local_state: stochastic.local_state } : {}) },
    ] },
    tran: {
      t_stop_s: settings.duration_s, t_start_save_s: 0,
      dt_max_s: settings.duration_s / 3000,
      dt_min_s: Math.max(1e-18, Math.min(1e-15, settings.duration_s * 1e-9)),
      method: "BE", reltol: 1e-3,
    },
    ...(mode === "stochastic" ? { stochastic: {
      seed: stochastic.seed, n_runs: 1, carrier_noise: stochastic.carrier_noise,
      ld_carrier_noise: stochastic.ld_carrier_noise, local_state: stochastic.local_state, local_state_override: true,
    } } : {}),
    detect: { i_threshold_A: 1e-8, hysteresis: 10 },
    probes: ["V(drain)", "I(X1.d)", "X1.vb"],
  };
}
