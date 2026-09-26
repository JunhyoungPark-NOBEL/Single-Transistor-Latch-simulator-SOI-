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

/** Numerical input domain, not a claim of calibration or FDSOI validity across this range. */
export const GEOMETRY_LIMITS: Readonly<Record<GeometryKey, { min: number; max: number; step: number }>> = {
  Lg_nm: { min: 100, max: 2000, step: 10 },
  W_nm: { min: 20, max: 10000, step: 10 },
  Tsi_nm: { min: 5, max: 200, step: 1 },
  EOT_nm: { min: 1, max: 100, step: 0.1 },
  Tbox_nm: { min: 10, max: 1000, step: 10 },
  Nbody_cm3: { min: 1e15, max: 1e19, step: 1e16 },
};
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
