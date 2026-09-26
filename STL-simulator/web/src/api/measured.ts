// Normalisers for GET /api/data/measured and GET /api/data/design_map (server/compute/data.py).
// They accept the backend shape and return the compact types used by the panels; unknown or
// missing keys degrade to empty data instead of throwing.
import type { DesignMapData, MeasuredData, MeasuredLightCurve, MeasuredPhotoCondition } from "./types";

type Any = Record<string, unknown>;
const obj = (v: unknown): Any => (v && typeof v === "object" && !Array.isArray(v) ? (v as Any) : {});
const nums = (v: unknown): number[] => (Array.isArray(v) ? v.map((x) => (typeof x === "number" ? x : NaN)) : []);
const num = (v: unknown, d = NaN): number => (typeof v === "number" ? v : d);
const matrix = (v: unknown): (number | null)[][] =>
  Array.isArray(v) ? v.map((row) => (Array.isArray(row) ? row.map((x) => (typeof x === "number" ? x : null)) : [])) : [];

export function normalizeMeasured(raw: unknown): MeasuredData {
  const r = obj(raw);
  // photo device conditions
  const photo = obj(r.photo);
  const vluRows = matrix(photo.V_LU);
  const conds = Array.isArray(photo.conditions) ? photo.conditions : [];
  const photoOut: MeasuredPhotoCondition[] = conds.map((c, k) => {
    const cc = obj(c);
    const st = obj(cc.stats);
    const fs = obj(cc.file_stats);
    const mean = num(st.mean, num(fs.mean_V));
    const sd = typeof st.sd === "number" ? st.sd * 1e3 : num(fs.sd_mV);
    return {
      vg: num(cc.vg),
      power_mW: num(cc.power_mW, 0),
      label: String(cc.label ?? `#${k}`),
      n: num(st.n, num(fs.n, 0)),
      mean_V: mean,
      sd_mV: sd,
      lag1: typeof st.lag1 === "number" ? st.lag1 : typeof fs.lag1 === "number" ? fs.lag1 : null,
      raw: (vluRows[num(cc.index, k)] ?? []).filter((x): x is number => typeof x === "number"),
    };
  });

  // light I–V
  const li = obj(r.light_iv);
  const vd = nums(li.vd);
  const powers = nums(li.power_mW);
  const ids = matrix(li.id);
  const light: MeasuredLightCurve[] = ids.map((row, k) => ({
    label: Number.isFinite(powers[k]) ? `${powers[k].toFixed(2)} mW` : `#${k}`,
    power_mW: Number.isFinite(powers[k]) ? powers[k] : null,
    vd,
    id: row.map((x) => (x == null ? NaN : x)),
  }));

  // paper device double sweeps
  const pi = obj(r.paper_idvd);
  const up = obj(pi.up);
  const down = obj(pi.down);
  const paper =
    Array.isArray(up.vd) && Array.isArray(up.median)
      ? {
          vd_up: nums(up.vd), median_up: nums(up.median), p10_up: nums(up.p10), p90_up: nums(up.p90),
          vd_down: nums(down.vd), median_down: nums(down.median), p10_down: nums(down.p10), p90_down: nums(down.p90),
          V_LU: nums(pi.V_LU), V_LD: nums(pi.V_LD),
        }
      : null;
  return { photo: photoOut, light_iv: light, paper_iv: paper };
}

export function normalizeDesignMap(raw: unknown): DesignMapData {
  const r = obj(raw);
  const arrays = obj(r.arrays ?? r);
  const scalarsIn = obj(r.scalars);
  const fields: Record<string, (number | null)[][]> = {};
  for (const [k, v] of Object.entries(arrays)) {
    if (Array.isArray(v) && Array.isArray(v[0])) fields[k] = matrix(v);
  }
  const scalars: Record<string, number> = {};
  for (const [k, v] of Object.entries(scalarsIn)) if (typeof v === "number") scalars[k] = v;
  const hasLines = Array.isArray(arrays.line_Nt);
  return {
    length_nm: nums(arrays.length_nm),
    depth_fraction: nums(arrays.depth_fraction),
    fields,
    lines: hasLines ? { Nt: nums(arrays.line_Nt), L0_device: nums(arrays.line_L0_device), L0_50: nums(arrays.line_L0_50) } : null,
    scalars,
  };
}
