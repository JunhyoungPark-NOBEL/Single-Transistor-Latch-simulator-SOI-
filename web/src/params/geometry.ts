import type { DeviceBlock, DeviceGeometry } from "../api/types";

export type GeometryKey = keyof DeviceGeometry;

/** Reference-fit dimensions. Tbox is a nominal 140 nm assumption, not an extracted measurement. */
export const REFERENCE_GEOMETRY: Readonly<DeviceGeometry> = Object.freeze({
  Lg_nm: 500,
  W_nm: 200,
  Tsi_nm: 50,
  EOT_nm: 14.1,
  Tbox_nm: 140,
  Nbody_cm3: 2.295773162796593e17,
});

/** Numerical input domain (= server/params.py GEOMETRY_LIMITS), not a claim of calibration or FDSOI validity across
 *  this range. Nbody: below 3e16 cm⁻³ even the reference L leaves almost no neutral body; above ~1.14e18 cm⁻³ the
 *  avalanche-field model stops short of the latch-up voltage, so the server refuses those values. */
export const GEOMETRY_LIMITS: Readonly<Record<GeometryKey, { min: number; max: number; step: number }>> = {
  Lg_nm: { min: 100, max: 2000, step: 10 },
  W_nm: { min: 20, max: 10000, step: 10 },
  Tsi_nm: { min: 5, max: 200, step: 1 },
  EOT_nm: { min: 1, max: 100, step: 0.1 },
  Tbox_nm: { min: 10, max: 1000, step: 10 },
  Nbody_cm3: { min: 3e16, max: 1.1e18, step: 1e16 },
};

/** Shortest L (nm) the server accepts at this Nbody: L must exceed 2·w_d0(Nbody) + 1 nm, w_d0 being the zero-bias
 *  n⁺ source/drain depletion width (server/params.py min_length_nm; 153.6 nm at the reference Nbody). */
export function minLengthNm(nbody_cm3: number): number {
  const q = 1.602176634e-19, vt = 1.380649e-23 * 300 / q, eps = 11.7 * 8.8541878128e-12 / 100;
  const vbi = vt * Math.log(1e20 * nbody_cm3 / 1e20);
  return (2 * Math.sqrt(2 * eps * vbi / (q * nbody_cm3)) + 1e-7) * 1e7;
}

/** Back-gate coupling EOT/(Tbox + Tsi/3)·|V_BG| the server accepts (linear, depleted back interface). */
export const BACKGATE_SHIFT_MAX_V = 2;
export function backGateShiftV(geometry: DeviceGeometry, vbg: number): number {
  return Math.abs(geometry.EOT_nm / (geometry.Tbox_nm + geometry.Tsi_nm / 3) * vbg);
}
export const GEOMETRY_KEYS = Object.keys(REFERENCE_GEOMETRY) as GeometryKey[];

export function isGeometryValue(key: GeometryKey, value: unknown): value is number {
  const { min, max } = GEOMETRY_LIMITS[key];
  return typeof value === "number" && Number.isFinite(value) && value >= min && value <= max;
}

/** Legacy and corrupt storage migration. Invalid/missing fields restore the reference value. */
export function resolveGeometry(input?: Partial<DeviceGeometry> | null): DeviceGeometry {
  return Object.fromEntries(GEOMETRY_KEYS.map((key) => [key,
    isGeometryValue(key, input?.[key]) ? input![key] : REFERENCE_GEOMETRY[key],
  ])) as unknown as DeviceGeometry;
}

export function isReferenceGeometry(input?: Partial<DeviceGeometry> | null): boolean {
  const geometry = resolveGeometry(input);
  return GEOMETRY_KEYS.every((key) => geometry[key] === REFERENCE_GEOMETRY[key]);
}

export const BACK_GATE_LIMITS = { min: -10, max: 10 } as const;
export function resolveBackGate(input?: unknown): number {
  return typeof input === "number" && Number.isFinite(input) && input >= BACK_GATE_LIMITS.min && input <= BACK_GATE_LIMITS.max ? input : 0;
}

/** The reference stochastic kernel covers the original dimensions at V_BG = 0 only. */
export function usesGeometryModel(device: Pick<DeviceBlock, "geometry" | "vbg">): boolean {
  return !isReferenceGeometry(device.geometry) || resolveBackGate(device.vbg) !== 0;
}
