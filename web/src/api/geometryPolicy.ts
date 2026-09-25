import { GEOMETRY_KEYS, REFERENCE_GEOMETRY } from "../params/geometry";

export const GEOMETRY_LIVE_REQUIRED = "geometry-live-required";
const STOCHASTIC = "geometry-stochastic-unavailable";
const DOMAIN = "geometry-domain-unavailable";
const STEP_BUDGET = "circuit-step-budget";

/** Element names a custom-circuit error names: "X1: <code>: …" (worker prefix) or "<code>: … (X1)" (suffix). */
function elementPrefix(message: string, at: number): string {
  const head = message.slice(0, at);
  if (/^(?:[A-Za-z][A-Za-z0-9_]*: )+$/.test(head)) return head;
  const tail = /\(([A-Za-z][A-Za-z0-9_]*)\)\s*$/.exec(message);
  return tail ? `${tail[1]}: ` : "";
}

function domainText(message: string, ko: boolean): string {
  if (message.includes("back-gate coupling")) return ko
    ? "백게이트 결합이 선형 범위를 넘습니다(EOT/(Tbox + Tsi/3) × |V_BG| ≤ 2 V). |V_BG|나 EOT를 줄이거나 Tbox를 늘려 주세요."
    : "The back-gate coupling is beyond its linear range (EOT/(Tbox + Tsi/3) × |V_BG| ≤ 2 V). Reduce |V_BG| or EOT, or increase Tbox.";
  if (message.includes("fully depletes")) {
    const min = /must exceed (\d+(?:\.\d+)?) nm/.exec(message)?.[1];
    return ko
      ? `이 Nbody에서는 L이 너무 짧아 중성 바디가 남지 않습니다. ${min ? `L을 ${min} nm보다 길게 하거나` : "L을 늘리거나"} Nbody를 높여 주세요.`
      : `At this Nbody, L is too short to leave a neutral body. ${min ? `Make L longer than ${min} nm` : "Increase L"} or raise Nbody.`;
  }
  if (message.includes("avalanche-field")) return ko
    ? "이 Nbody에서는 애벌랜치 전계 모델이 래치업 전압 범위까지 닿지 않습니다. Nbody를 낮춰 주세요."
    : "At this Nbody the avalanche-field model does not reach the latch-up voltage range. Lower Nbody.";
  return ko
    ? "이 L·Nbody 조합은 현재 수송·전계 모델의 범위를 벗어납니다. L 또는 Nbody를 조정해 주세요."
    : "This L/Nbody combination is outside the transport or field model's domain. Adjust L or Nbody.";
}

function stepBudgetText(message: string, ko: boolean): string {
  const cycles = /relaxation oscillation: ~(\d+) predicted/.exec(message)?.[1];
  if (cycles) return ko
    ? `이 조건에서는 래치업·래치다운이 약 ${Number(cycles).toLocaleString("ko-KR")}번 반복되어 계산 스텝 한도를 넘습니다. 시간을 줄이거나, 커패시턴스를 키우거나, 전류를 줄여 주세요.`
    : `This run would repeat latch-up and latch-down about ${Number(cycles).toLocaleString("en-US")} times, beyond the step limit. Shorten the time, increase the capacitance or lower the current.`;
  const steps = /estimated (?:total work )?~([\d.e+]+) time steps/.exec(message)?.[1];
  const n = steps ? Number(steps) : NaN;
  const est = Number.isFinite(n) ? (ko ? ` (예상 약 ${n.toExponential(1)} 스텝)` : ` (about ${n.toExponential(1)} steps)`) : "";
  return ko
    ? `계산 스텝 수가 한도를 넘어 실행하지 않았습니다${est}. 시뮬레이션 시간이나 반복 수를 줄여 주세요.`
    : `Not run: the number of time steps is beyond the limit${est}. Shorten the simulated time or reduce the repetitions.`;
}

/** Shared device/circuit guidance for unsupported numerical-model requests (server codes → KO/EN text). */
export function geometryError(message: string, lang: string): string {
  const ko = lang === "ko";
  if (message.includes(GEOMETRY_LIVE_REQUIRED)) return ko
    ? "Geometry·V_BG를 바꾼 계산은 계산 서버에 연결해야 실행됩니다."
    : "Connect to the live compute server to simulate changed geometry or V_BG.";
  const at = (code: string) => message.indexOf(code);
  if (at(STOCHASTIC) >= 0) return elementPrefix(message, at(STOCHASTIC)) + (ko
    ? "Geometry·V_BG 변경은 결정론 모드로 계산해 주세요. 확률 모델은 기준 치수·V_BG=0에서 보정돼 있습니다."
    : "Use deterministic mode for changed geometry or V_BG. The stochastic model is calibrated at the reference dimensions and V_BG=0.");
  if (at(DOMAIN) >= 0) return elementPrefix(message, at(DOMAIN)) + domainText(message, ko);
  if (at(STEP_BUDGET) >= 0) return stepBudgetText(message, ko);
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
