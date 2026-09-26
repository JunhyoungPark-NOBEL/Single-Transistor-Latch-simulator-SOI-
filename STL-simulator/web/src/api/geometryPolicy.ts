import { GEOMETRY_KEYS, REFERENCE_GEOMETRY } from "../params/geometry";

export const GEOMETRY_LIVE_REQUIRED = "geometry-live-required";
/** Shared device/circuit guidance for unsupported numerical-model requests. */
export function geometryError(message: string, lang: string): string {
  const ko = lang === "ko";
  if (message.includes("simple-model-live-required")) return ko
    ? "Simple Model은 실시간 계산 서버에 연결해 주세요." : "Connect to the live compute server to use Simple Model.";
  if (message.includes("simple-model-method-unavailable")) return ko
    ? "Simple Model 회로는 적분법을 BE로 선택해 주세요." : "Choose BE integration for Simple Model circuits.";
  if (message.includes("simple-calibration-domain")) return ko
    ? "입력점이 안정한 HRS 범위를 벗어납니다. 래치 전 측정점과 현재 바이어스를 확인해 주세요." : "A point lies outside the stable HRS range. Check pre-latch data and the current bias.";
  if (message.includes("identifiability")) return ko
    ? "이 점들만으로 두 파라미터를 구분할 수 없습니다. 한 항목을 보정하거나 서로 다른 HRS 점을 추가해 주세요." : "These points cannot identify both parameters. Fit one parameter or add distinct HRS points.";
  if (message.includes("simple-model-stochastic-unavailable")) return ko
    ? "Simple Model은 현재 결정론적 해석을 지원합니다." : "Simple Model currently supports deterministic analysis.";
  if (message.includes("five-terminal-stochastic-unavailable")) return ko
    ? "BG·B 접점을 연결한 회로는 현재 결정론적 모드로 계산할 수 있습니다."
    : "Circuits with connected BG or B terminals currently require deterministic mode.";
  if (message.includes("five-terminal-method-unavailable")) return ko
    ? "BG·B 접점을 연결한 회로는 적분법을 BE로 선택해 주세요."
    : "Choose BE integration for circuits with connected BG or B terminals.";
  if (message.includes("backgate-domain-unavailable")) return ko
    ? "이 VBG·Geometry 조합은 현재 바디 바이어스 근사의 범위를 벗어납니다. VBG 크기를 줄여 주세요."
    : "This VBG/geometry combination is outside the body-bias approximation. Reduce the magnitude of VBG.";
  if (message.includes("local-avalanche-mode-unavailable")) return ko
    ? "선택한 국소 경로는 지원하지 않습니다. 고급 설정의 국소 경로 캐리어를 0 또는 1로 바꿔 주세요."
    : "This local path is unsupported. Choose carrier path 0 or 1 in advanced settings.";
  if (message.includes(GEOMETRY_LIVE_REQUIRED)) return ko
    ? "Geometry·VBG를 변경한 계산은 실시간 계산 서버에 연결해 주세요."
    : "Connect to the live compute server to simulate changed geometry or VBG.";
  if (message.includes("geometry-stochastic-unavailable")) return ko
    ? "Geometry·VBG 변경은 결정론적 모드로 계산해 주세요. 확률적 모델은 기준 치수·VBG=0에서 보정돼 있습니다."
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
  return Object.fromEntries(Object.entries(payload).filter(([key]) => !(omit && key === "geometry") && !(omitBackGate && key === "vbg") && !(("vg" in payload || "calib" in payload) && payload.model !== "simple" && (key === "model" || key === "simple")))
    .map(([key, value]) => [key, legacyGeometryPayload(value)]));
}
