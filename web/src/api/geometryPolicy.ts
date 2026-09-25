import { GEOMETRY_KEYS, REFERENCE_GEOMETRY } from "../params/geometry";

export const GEOMETRY_LIVE_REQUIRED = "geometry-live-required";
/** Shared device/circuit guidance for unsupported numerical-model requests. */
export function geometryError(message: string, lang: string): string {
  const ko = lang === "ko";
  if (message.includes(GEOMETRY_LIVE_REQUIRED)) return ko
    ? "Geometry·VBG를 변경한 계산은 실시간 계산 서버에 연결해 주세요."
    : "Connect to the live compute server to simulate changed geometry or VBG.";
  if (message.includes("geometry-stochastic-unavailable")) return ko
    ? "Geometry·VBG 변경은 결정론 모드로 계산해 주세요. 확률 모델은 기준 치수·VBG=0에서 보정돼 있습니다."
    : "Use deterministic mode for changed geometry or VBG. The stochastic model is calibrated at the reference dimensions and VBG=0.";
  if (message.includes("geometry-domain-unavailable")) return ko
    ? "이 L·Nbody 조합은 현재 수송·전계 모델의 범위를 벗어납니다. L 또는 Nbody를 조정해 주세요."
    : "This L/Nbody combination is outside the transport or field model's domain. Adjust L or Nbody.";
  return message;
}
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

/** Missing dimensions mean the historical reference device, never a different simulated shape. */
function changed(geometry: unknown): boolean {
  if (geometry === undefined) return false;
  if (!object(geometry)) return true;
  return GEOMETRY_KEYS.some((key) => geometry[key] !== undefined && geometry[key] !== REFERENCE_GEOMETRY[key]);
}

/** Geometry-model requests, including nonzero back gate and heterogeneous circuit snapshots. */
export function hasChangedGeometry(payload: unknown): boolean {
  if (Array.isArray(payload)) return payload.some(hasChangedGeometry);
  if (!object(payload)) return false;
  if (("vg" in payload || "calib" in payload) && "vbg" in payload && payload.vbg !== undefined && payload.vbg !== 0) return true;
  if ("geometry" in payload && ("vg" in payload || "calib" in payload) && changed(payload.geometry)) return true;
  return Object.values(payload).some(hasChangedGeometry);
}

/** Compatibility lookup only: snapshot hashes still represent the exact original request. */
export function legacyGeometryPayload(payload: unknown): unknown {
  if (Array.isArray(payload)) return payload.map(legacyGeometryPayload);
  if (!object(payload)) return payload;
  const omit = "geometry" in payload && ("vg" in payload || "calib" in payload) && !changed(payload.geometry);
  const omitBackGate = ("vg" in payload || "calib" in payload) && payload.vbg === 0;
  return Object.fromEntries(Object.entries(payload).filter(([key]) => !(omit && key === "geometry") && !(omitBackGate && key === "vbg"))
    .map(([key, value]) => [key, legacyGeometryPayload(value)]));
}
