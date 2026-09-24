// Device library (docs/WEB_CONTRACT.md §7): built-in read-only FDSOI devices derived from the presets
// in /api/meta, user devices saved from the Device tab, JSON import/export with validation.
// Pure functions (unit-tested); persistence lives in ./store.ts.
import type { L10n } from "../content/physics/types";
import type { DeviceBlock, LocalStateBlock, Meta, PresetId, StochasticBlock } from "../api/types";
import { clone, getPath, mergeDefaults, setPath, type Path } from "../utils/object";

export type Technology = "FDSOI" | "PDSOI" | "Bulk";
export const TECHNOLOGIES: { id: Technology; active: boolean }[] = [
  { id: "FDSOI", active: true },
  { id: "PDSOI", active: false },
  { id: "Bulk", active: false },
];

export interface Geometry {
  Lg_nm: number;
  W_nm: number;
  Tsi_nm: number;
  EOT_nm: number;
}
export interface DeviceStochastic {
  local_state: LocalStateBlock;
  carrier_noise: boolean;
  ld_carrier_noise: boolean;
}
export interface LibDevice {
  id: string;
  name: string;
  technology: Technology;
  geometry: Geometry;
  calibration_label: L10n;
  device: DeviceBlock;
  /** Stochastic settings used when the device is simulated stochastically in a circuit. */
  stochastic: DeviceStochastic;
  created: string;
  notes: string;
  /** Built-in (read-only) entries carry a bilingual display name. */
  builtin?: boolean;
  label?: L10n;
}

export const LIBRARY_FORMAT = "stl-device-library";
export const LIBRARY_VERSION = 1;
export const FDSOI_GEOMETRY: Geometry = { Lg_nm: 500, W_nm: 200, Tsi_nm: 50, EOT_nm: 14.1 };
export const BUILTIN_IDS: PresetId[] = ["paper", "photo"];
export const builtinId = (p: PresetId) => `builtin:${p}`;

/** Geometry from /api/meta constants ({L_nm, W_nm, T_Si_nm, EOT_nm}); FDSOI defaults otherwise. */
export function geometryFromMeta(meta: Pick<Meta, "constants">): Geometry {
  const g = (meta.constants?.geometry ?? {}) as Record<string, unknown>;
  const n = (k: string, d: number) => (typeof g[k] === "number" && Number.isFinite(g[k] as number) ? (g[k] as number) : d);
  return { Lg_nm: n("L_nm", FDSOI_GEOMETRY.Lg_nm), W_nm: n("W_nm", FDSOI_GEOMETRY.W_nm), Tsi_nm: n("T_Si_nm", FDSOI_GEOMETRY.Tsi_nm), EOT_nm: n("EOT_nm", FDSOI_GEOMETRY.EOT_nm) };
}

const fmtNm = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1));
/** "L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm" (plain text, no TeX). */
export function geometryLine(g: Geometry): string {
  return `L_g ${fmtNm(g.Lg_nm)} nm · W ${fmtNm(g.W_nm)} nm · T_Si ${fmtNm(g.Tsi_nm)} nm · EOT ${fmtNm(g.EOT_nm)} nm`;
}

export function stochOf(s: Pick<StochasticBlock, "local_state" | "carrier_noise" | "ld_carrier_noise">): DeviceStochastic {
  return { local_state: clone(s.local_state), carrier_noise: s.carrier_noise, ld_carrier_noise: s.ld_carrier_noise };
}

/** "FDSOI · L_g … — reference calibration (dark, …)" → "reference calibration (dark, …)" (label unchanged otherwise). */
export function calibPart(label: string): string {
  const i = label.lastIndexOf(" — ");
  const s = i >= 0 ? label.slice(i + 3) : label;
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : label;
}

/** Built-in read-only devices, one per calibration preset (labels from the brand package via meta). */
export function builtinDevices(meta: Meta): LibDevice[] {
  const geometry = geometryFromMeta(meta);
  return BUILTIN_IDS.filter((p) => meta.presets[p]).map((p) => {
    const pr = meta.presets[p];
    return {
      id: builtinId(p),
      name: calibPart(pr.label?.en ?? p),
      label: pr.label ? { ko: calibPart(pr.label.ko), en: calibPart(pr.label.en) } : undefined,
      technology: "FDSOI",
      geometry,
      calibration_label: pr.label,
      device: clone({ ...pr.device, preset: p }),
      stochastic: stochOf(pr.stochastic),
      created: "",
      notes: "",
      builtin: true,
    };
  });
}

/** Display name (built-ins in the UI language). */
export const deviceName = (d: Pick<LibDevice, "name" | "label" | "builtin">, lang: "ko" | "en") => (d.builtin && d.label ? d.label[lang] || d.name : d.name);

// ---------------------------------------------------------------- validation
const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const LOCAL_MODES = ["none", "frozen", "evolving"];
const LOCAL_ACTIONS = ["gidl", "local_avalanche", "junction", "multiplication"];
const DEVICE_ENUMS: [Path, readonly unknown[]][] = [
  [["preset"], ["paper", "photo", "custom"]],
  [["light", "mode"], ["iph", "power"]],
  [["ext", "loc_carriers"], [0, 1, 2]],
];

export function sanitizeDevice(base: DeviceBlock, raw: unknown): DeviceBlock {
  let d = mergeDefaults(base, raw);
  for (const [path, allowed] of DEVICE_ENUMS) if (!allowed.includes(getPath(d, path))) d = setPath(d, path, getPath(base, path));
  return d;
}

export function sanitizeLocal(base: LocalStateBlock, raw: unknown): LocalStateBlock {
  const l = mergeDefaults(base, raw);
  if (!LOCAL_MODES.includes(l.mode)) l.mode = base.mode;
  if (!LOCAL_ACTIONS.includes(l.action)) l.action = base.action;
  return l;
}

const str = (v: unknown, max = 200) => (typeof v === "string" ? v.slice(0, max) : "");
const l10n = (v: unknown): L10n | null => (isObj(v) && typeof v.ko === "string" && typeof v.en === "string" ? { ko: v.ko, en: v.en } : typeof v === "string" && v ? { ko: v, en: v } : null);

/**
 * Validate one stored/imported device. Unknown keys are dropped, every device-block field is type-checked
 * against `base` (missing → base value), enumerations enforced. Returns null when it is not a device.
 */
export function validateDevice(raw: unknown, base: { device: DeviceBlock; stochastic: DeviceStochastic }, fallbackId?: string): LibDevice | null {
  if (!isObj(raw) || !isObj(raw.device)) return null;
  const name = str(raw.name, 80).trim();
  if (!name) return null;
  const technology = (["FDSOI", "PDSOI", "Bulk"] as const).find((x) => x === raw.technology) ?? "FDSOI";
  const g = isObj(raw.geometry) ? raw.geometry : {};
  const num = (v: unknown, d: number) => (typeof v === "number" && Number.isFinite(v) && v > 0 ? v : d);
  const geometry: Geometry = { Lg_nm: num(g.Lg_nm, FDSOI_GEOMETRY.Lg_nm), W_nm: num(g.W_nm, FDSOI_GEOMETRY.W_nm), Tsi_nm: num(g.Tsi_nm, FDSOI_GEOMETRY.Tsi_nm), EOT_nm: num(g.EOT_nm, FDSOI_GEOMETRY.EOT_nm) };
  const st = isObj(raw.stochastic) ? raw.stochastic : {};
  const id = str(raw.id, 80) || fallbackId || newDeviceId();
  return {
    id: id.startsWith("builtin:") ? newDeviceId() : id,
    name,
    technology,
    geometry,
    calibration_label: l10n(raw.calibration_label) ?? { ko: "사용자 정의", en: "Custom" },
    device: sanitizeDevice(base.device, raw.device),
    stochastic: {
      local_state: sanitizeLocal(base.stochastic.local_state, st.local_state),
      carrier_noise: typeof st.carrier_noise === "boolean" ? st.carrier_noise : base.stochastic.carrier_noise,
      ld_carrier_noise: typeof st.ld_carrier_noise === "boolean" ? st.ld_carrier_noise : base.stochastic.ld_carrier_noise,
    },
    created: str(raw.created, 40) || new Date().toISOString(),
    notes: str(raw.notes, 2000),
  };
}

export interface ImportResult {
  devices: LibDevice[];
  skipped: number;
  error?: "json" | "format" | "empty";
}

/** Parse a library export ({format, v, devices}), a bare array of devices or a single device. */
export function parseLibraryJson(text: string, base: { device: DeviceBlock; stochastic: DeviceStochastic }): ImportResult {
  let v: unknown;
  try {
    v = JSON.parse(text);
  } catch {
    return { devices: [], skipped: 0, error: "json" };
  }
  let list: unknown[];
  if (Array.isArray(v)) list = v;
  else if (isObj(v) && Array.isArray(v.devices)) list = v.devices;
  else if (isObj(v) && isObj(v.device)) list = [v];
  else return { devices: [], skipped: 0, error: "format" };
  const devices = list.map((x) => validateDevice(x, base)).filter((x): x is LibDevice => !!x);
  if (!devices.length) return { devices: [], skipped: list.length, error: list.length ? "format" : "empty" };
  return { devices, skipped: list.length - devices.length };
}

export function exportLibraryJson(devices: LibDevice[]): string {
  const clean = devices.map(({ builtin: _b, label: _l, ...d }) => d);
  return JSON.stringify({ format: LIBRARY_FORMAT, v: LIBRARY_VERSION, exported: new Date().toISOString(), devices: clean }, null, 2);
}

export function newDeviceId(): string {
  return `dev-${Date.now().toString(36)}-${Math.floor(Math.random() * 46656).toString(36)}`;
}

/** Unique name: "name", "name (2)", … */
export function uniqueName(name: string, taken: string[]): string {
  const set = new Set(taken.map((x) => x.toLowerCase()));
  if (!set.has(name.toLowerCase())) return name;
  const stem = name.replace(/\s*\(\d+\)$/, "");
  for (let k = 2; ; k++) if (!set.has(`${stem} (${k})`.toLowerCase())) return `${stem} (${k})`;
}
