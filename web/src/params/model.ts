import type { DeviceBlock, DeviceModel, HrsPoint, SimpleModelBlock } from "../api/types";

/** Mirror of the server's Simple defaults at the Device 1 reference geometry. */
export const SIMPLE_DEFAULTS: SimpleModelBlock = {
  beta_ref: 2.3, tau_body_s: 2e-7, cb_ref_F: 0.86e-15 * 200 / 650,
  r_lrs_ref_ohm: 44000 * 650 / 200, is_ref_A: 2e-16 * 200 / 650,
  vbr_ref_V: 2.35, avalanche_eta: 4, gamma_fg: 0.2, gamma_bg: 0.0525,
  vfb_V: -3.35, btbt_scale: 1, surface_fraction: 1, gidl_volume_scale: 100,
};
export const modelOf = (device: Pick<DeviceBlock, "model">): DeviceModel => device.model === "simple" ? "simple" : "detailed";
export const modelLabel = (device: Pick<DeviceBlock, "model">) => modelOf(device) === "simple" ? "Simple" : "Detailed";
export function simpleOf(value?: Partial<SimpleModelBlock>): SimpleModelBlock {
  const out = { ...SIMPLE_DEFAULTS };
  for (const k of Object.keys(out) as (keyof SimpleModelBlock)[]) {
    const v = value?.[k];
    if (typeof v === "number" && Number.isFinite(v)) out[k] = v;
  }
  return out;
}
/** Add optional schema defaults before mergeDefaults validates saved data. */
export const withModelDefaults = (device: DeviceBlock): DeviceBlock => ({ ...device, model: modelOf(device), simple: simpleOf(device.simple) });

export const SIMPLE_LIVE_REQUIRED = "simple-model-live-required";
export function hasSimpleModel(payload: unknown): boolean {
  if (Array.isArray(payload)) return payload.some(hasSimpleModel);
  if (!payload || typeof payload !== "object") return false;
  const obj = payload as Record<string, unknown>;
  if (obj.model === "simple" && ("vg" in obj || "calib" in obj || "simple" in obj)) return true;
  return Object.values(obj).some(hasSimpleModel);
}

/** Exactly two SI columns. Optional first-line labels; malformed rows never disappear silently. */
export function parseHrsPoints(text: string): HrsPoint[] {
  const lines = text.trim().split(/\r?\n/).filter((line) => line.trim());
  if (!text.trim()) return [];
  const points: HrsPoint[] = [];
  lines.forEach((line, row) => {
    const columns = line.trim().split(/[\s,;]+/);
    if (row === 0 && columns.length === 2 && /^v/i.test(columns[0]) && /^i/i.test(columns[1])) return;
    const [vd_V, id_A] = columns.map(Number);
    if (columns.length !== 2 || !Number.isFinite(vd_V) || !Number.isFinite(id_A) || vd_V <= 0 || id_A <= 0) throw new Error(`row:${row + 1}`);
    points.push({ vd_V, id_A });
  });
  if (points.length > 200) throw new Error("limit");
  return points;
}
