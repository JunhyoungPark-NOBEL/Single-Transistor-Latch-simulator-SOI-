// Parameter groups and fields for the sidebar (docs/WEB_CONTRACT.md §5). Every field addresses the
// parameter root by `path`; values are stored exactly as the payload schema defines them (SI, or the
// unit in the key name, e.g. iph_pA) and converted for display with `scale` (display = stored × scale).
import type { L10n, TopicId } from "../content/physics/types";
import type { BenchId, CircuitStochBlock, DeviceBlock, Mode, SolverBlock, StochasticBlock, SweepBlock } from "../api/types";
import type { StrKey } from "../i18n/strings";
import type { Path } from "../utils/object";
import type { BenchValue } from "./benches";

export type Tab = "device" | "circuit" | "validation" | "physics";

export interface CircuitParams {
  bench: BenchId;
  bench_params: Record<BenchId, Record<string, BenchValue>>;
  solver: SolverBlock;
  stochastic: CircuitStochBlock;
  detect: { i_threshold_A: number };
}
export interface ParamRoot {
  device: DeviceBlock;
  sweep: SweepBlock;
  stochastic: StochasticBlock;
  circuit: CircuitParams;
}
export interface Ctx {
  root: ParamRoot;
  mode: Mode;
  tab: Tab;
}

export interface Option {
  value: string | number | boolean;
  label: L10n | StrKey;
  experimental?: boolean;
  /** Shown but not selectable; `note` explains why (tooltip). */
  disabled?: boolean;
  note?: L10n;
}
export interface FieldDef {
  key: string;
  path: Path;
  sym?: string;
  label: L10n;
  help: L10n;
  code?: string;
  unit?: string | ((c: Ctx) => string);
  scale?: number | ((c: Ctx) => number);
  min?: number;
  max?: number;
  step?: number;
  slider?: boolean | "log";
  int?: boolean;
  /** Value may be null = "auto" (resolved server-side). */
  auto?: boolean;
  type?: "number" | "toggle" | "select" | "segmented" | "list";
  options?: Option[];
  show?: (c: Ctx) => boolean;
  experimental?: boolean;
  /** Main field (간단히 layout): always visible, with a slider and the inline guide. Other fields of a basic
   *  group fold behind "고급 항목 n개" unless their value differs from the default. */
  main?: boolean | ((c: Ctx) => boolean);
}
export type CustomBlock = "light" | "seed" | "local-warning" | "bench";
export interface GroupDef {
  id: string;
  title: StrKey;
  desc: StrKey;
  topic: TopicId;
  tabs: Tab[];
  fields: FieldDef[];
  /** Extra paths reset by the group's reset link (besides its fields). */
  extraPaths?: Path[];
  custom?: CustomBlock[];
  stochasticOnly?: boolean;
  collapsed?: boolean;
  /** Advanced group (간단히 layout): listed under the "고급 설정" disclosure, closed by default. */
  advanced?: boolean;
  show?: (c: Ctx) => boolean;
}

const L = (ko: string, en: string): L10n => ({ ko, en });
const isDevice = (c: Ctx) => c.tab === "device" || c.tab === "validation" || c.tab === "physics";
const isVolt = (c: Ctx) => ["gidl", "junction"].includes(c.root.stochastic.local_state.action);
const isVoltC = (c: Ctx) => ["gidl", "junction"].includes(c.root.circuit.stochastic.local_state.action);

export const LOCAL_MODE_OPTIONS: Option[] = [
  { value: "none", label: "local.mode.none" },
  { value: "frozen", label: "local.mode.frozen" },
  { value: "evolving", label: "local.mode.evolving" },
];
export const LOCAL_ACTION_OPTIONS: Option[] = [
  { value: "gidl", label: "local.action.gidl" },
  { value: "local_avalanche", label: "local.action.local_avalanche", experimental: true },
  { value: "junction", label: "local.action.junction", experimental: true },
  { value: "multiplication", label: "local.action.multiplication", experimental: true },
];

/** Fields of the illumination block (the "light" group renders them itself: I_PH or P, plus R in power mode).
 *  Exported for the guide index (Physics tab) — every key has a PARAM_GUIDE entry. */
export const LIGHT_FIELDS: Record<"iph" | "power" | "resp", FieldDef> = {
  iph: { key: "iph_pA", path: ["device", "light", "iph_pA"], sym: "I_{PH}", label: L("광전류", "Photocurrent"), help: L("body로 들어가는 균일한 광생성 정공 전류", "Uniform photogenerated hole current into the body"), code: "p[13]", unit: "pA", min: 0, max: 100, step: 0.01, slider: true, main: true },
  power: { key: "power_mW", path: ["device", "light", "power_mW"], sym: "P", label: L("광 파워", "Optical power"), help: L("입사 광 파워 (I_PH = R·P)", "Incident optical power (I_PH = R·P)"), code: "p[13] = R·P", unit: "mW", min: 0, max: 50, step: 0.01, slider: true, main: true },
  resp: { key: "resp", path: ["device", "light", "responsivity_pA_per_mW"], sym: "R", label: L("응답도", "Responsivity"), help: L("광 변환 계수 (이 소자 보정값 0.75 pA/mW)", "Light conversion factor (this device: 0.75 pA/mW)"), unit: "pA/mW", min: 0, max: 100, step: 0.01 },
};

export const GROUPS: GroupDef[] = [
  {
    id: "bias",
    title: "g.bias",
    desc: "g.bias.desc",
    topic: "charge-balance",
    tabs: ["device", "circuit"],
    fields: [
      { key: "vg", path: ["device", "vg"], sym: "V_G", label: L("게이트 전압", "Gate voltage"), help: L("게이트-소스 전압 — 채널 전류와 게이트 가장자리 GIDL 전계에 들어갑니다", "Gate–source voltage — enters the channel current and the gate-edge GIDL field"), code: "p[11]", unit: "V", min: -6, max: 1, step: 0.01, slider: true, main: true },
      { key: "vd_max", path: ["sweep", "vd_max_V"], sym: "V_{D,\\max}", label: L("스윕 최대 전압", "Sweep peak"), help: L("삼각 스윕 0 → V_D,max → 0의 최고점 (서버 상한 8 V)", "Peak drain voltage of the triangular sweep 0 → V_D,max → 0 (server limit 8 V)"), unit: "V", min: 0.5, max: 8, step: 0.05, slider: true, show: isDevice, main: true },
      { key: "rate", path: ["sweep", "rate_V_per_s"], sym: "\\dot V_D", label: L("램프 속도", "Ramp rate"), help: L("드레인 전압 스윕 속도 — hazard 적분과 MC 시간축을 정합니다", "Drain-voltage sweep rate — sets the hazard integral and the MC time axis"), unit: "V/s", min: 1e-3, max: 1e5, slider: "log", show: isDevice, main: (c) => c.mode === "stochastic" },
      { key: "dv", path: ["sweep", "dv_V"], sym: "\\Delta V", label: L("전압 스텝", "Voltage step"), help: L("스윕 전압 간격 (MC 시간 스텝 Δt = ΔV / 램프 속도)", "Sweep voltage step (MC time step Δt = ΔV / ramp rate)"), unit: "mV", scale: 1e3, min: 0.1, max: 50, step: 0.1, show: isDevice },
    ],
  },
  {
    id: "light",
    title: "g.light",
    desc: "g.light.desc",
    topic: "photo",
    tabs: ["device", "circuit"],
    custom: ["light"],
    extraPaths: [["device", "light"]],
    fields: [],
  },
  {
    id: "state",
    advanced: true,
    title: "g.state",
    desc: "g.state.desc",
    topic: "local-states",
    tabs: ["device", "circuit"],
    fields: [
      { key: "dphiG0", path: ["device", "state", "delta_phi_G0_V"], sym: "\\delta\\phi_{G0}", label: L("드레인 쪽 이동", "Drain shift"), help: L("드레인 가장자리 국소 상태의 평균 이동 — GIDL 오프셋 p[9]에 더해집니다 (dV_LU/dφ_G ≈ −0.80 V/V)", "Mean shift of the drain-edge local state, added to the GIDL offset p[9] (dV_LU/dφ_G ≈ −0.80 V/V)"), code: "p[9] +", unit: "mV", scale: 1e3, min: -500, max: 500, step: 0.1, slider: true },
      { key: "dphiE0", path: ["device", "state", "delta_phi_E0_V"], sym: "\\delta\\phi_{E0}", label: L("소스 쪽 이동", "Source shift"), help: L("소스 가장자리 국소 상태의 평균 이동 — 이미터 오프셋 p[10]에 더해집니다 (dV_LD/dφ_E ≈ −41 V/V)", "Mean shift of the source-edge local state, added to the emitter offset p[10] (dV_LD/dφ_E ≈ −41 V/V)"), code: "p[10] +", unit: "mV", scale: 1e3, min: -10, max: 10, step: 0.01, slider: true },
    ],
  },
  {
    id: "calib",
    advanced: true,
    title: "g.calib",
    desc: "g.calib.desc",
    topic: "parameters",
    tabs: ["device", "circuit"],
    collapsed: true,
    fields: [
      { key: "beta", path: ["device", "calib", "beta"], sym: "\\beta", label: L("주입 비율", "Injection ratio"), help: L("소스로 빠져나가는 정공 하나당 주입되는 전자 수 — 클수록 정공 손실(I_DIFF ∝ 1/β)이 작습니다", "Electrons injected per hole that leaves through the source — larger β means a smaller hole loss (I_DIFF ∝ 1/β)"), code: "p[0]", unit: "", min: 0.01, max: 100, slider: "log" },
      { key: "tau_bulk", path: ["device", "calib", "tau_bulk_s"], sym: "\\tau_{\\mathrm{bulk}}", label: L("벌크 수명", "Bulk lifetime"), help: L("바디 벌크의 SRH 재결합 수명", "SRH recombination lifetime in the body bulk"), code: "p[1]", unit: "µs", scale: 1e6, min: 1e-4, max: 1e4, slider: "log" },
      { key: "tau_junction", path: ["device", "calib", "tau_junction_s"], sym: "\\tau_{j}", label: L("접합 수명", "Junction lifetime"), help: L("공핍 영역의 SRH 수명", "SRH lifetime in the depletion region"), code: "p[2]", unit: "ns", scale: 1e9, min: 1e-3, max: 1e5, slider: "log" },
      { key: "r_contact", path: ["device", "calib", "r_contact_ohm"], sym: "R_c", label: L("접촉 저항", "Contact R"), help: L("직렬 접촉 저항", "Series contact resistance"), code: "p[3]", unit: "Ω", min: 0, max: 1e6 },
      { key: "l_gidl", path: ["device", "calib", "l_gidl_nm"], sym: "l_{\\mathrm{GIDL}}", label: L("GIDL 전계 길이", "GIDL length"), help: L("GIDL 전계를 정하는 유효 길이: E_G = max(u + r − V_G − 0.3 − 1.12 + φ_GIDL, 0) / l_GIDL (BTBT 영역의 크기가 아님)", "Effective length that sets the GIDL field: E_G = max(u + r − V_G − 0.3 − 1.12 + φ_GIDL, 0) / l_GIDL (not the size of the BTBT region)"), code: "p[4]", unit: "nm", min: 0.1, max: 500, slider: "log" },
      { key: "t_access", path: ["device", "calib", "t_access_nm"], sym: "t_{\\mathrm{acc}}", label: L("접근 영역 두께", "Access thickness"), help: L("소스·드레인 접근 영역의 두께", "Thickness of the source/drain access regions"), code: "p[5]", unit: "nm", min: 0.1, max: 100, slider: "log" },
      { key: "na_access", path: ["device", "calib", "na_access_cm3"], sym: "N_{A,\\mathrm{acc}}", label: L("접근 영역 도핑", "Access doping"), help: L("접근 영역의 억셉터 농도", "Acceptor concentration in the access regions"), code: "p[6]", unit: "10¹⁷ cm⁻³", scale: 1e-17, min: 1e-3, max: 1e4, slider: "log" },
      { key: "l_access", path: ["device", "calib", "l_access_nm"], sym: "L_{\\mathrm{acc}}", label: L("접근 영역 길이", "Access length"), help: L("접근 영역의 길이", "Length of the access regions"), code: "p[7]", unit: "nm", min: 1, max: 1000, slider: "log" },
      { key: "tau_ratio", path: ["device", "calib", "tau_ratio"], sym: "\\tau_p/\\tau_n", label: L("수명 비", "Lifetime ratio"), help: L("정공과 전자의 SRH 수명 비", "Hole-to-electron SRH lifetime ratio"), code: "p[8]", unit: "", min: 1e-3, max: 1e5, slider: "log" },
      { key: "phi_gidl0", path: ["device", "calib", "phi_gidl0_V"], sym: "\\phi_{G,0}", label: L("GIDL 전위 오프셋", "GIDL offset"), help: L("GIDL 전위 오프셋 — 드레인 가장자리 국소 상태의 평균 (보정값)", "GIDL potential offset — mean of the drain-edge local state (calibrated)"), code: "p[9]", unit: "mV", scale: 1e3, min: -500, max: 500, step: 0.001 },
      { key: "phi_emitter0", path: ["device", "calib", "phi_emitter0_V"], sym: "\\phi_{E,0}", label: L("이미터 오프셋", "Emitter offset"), help: L("이미터 전위 오프셋 — 소스 가장자리 국소 상태의 평균 (보정값)", "Emitter potential offset — mean of the source-edge local state (calibrated)"), code: "p[10]", unit: "mV", scale: 1e3, min: -50, max: 50, step: 0.001 },
      { key: "ch_ii", path: ["device", "calib", "channel_ii_scale"], sym: "s_{\\mathrm{II,ch}}", label: L("채널 II 배율", "Channel-II scale"), help: L("채널 전자에 의한 충돌 이온화의 배율", "Scale factor of channel-electron impact ionization"), code: "p[12]", unit: "", min: 0, max: 100, slider: "log" },
    ],
  },
  {
    id: "ext",
    advanced: true,
    title: "g.ext",
    desc: "g.ext.desc",
    topic: "open-problems",
    tabs: ["device", "circuit"],
    collapsed: true,
    custom: ["seed"],
    fields: [
      { key: "dibl", path: ["device", "ext", "dibl"], sym: "\\eta", label: L("DIBL", "DIBL"), help: L("드레인 유도 장벽 저하(DIBL) 계수", "Drain-induced barrier lowering coefficient"), code: "p[14]", unit: "V/V", min: 0, max: 1, step: 0.001 },
      { key: "gamma", path: ["device", "ext", "gamma"], sym: "\\gamma", label: L("바디 결합", "Body coupling"), help: L("바디 전위가 채널에 주는 결합 (−1.1 V 채널 시드 옵션)", "Body-to-channel coupling (−1.1 V channel-seed option)"), code: "p[15]", unit: "", min: 0, max: 2, step: 0.0001 },
      { key: "kappa", path: ["device", "ext", "kappa"], sym: "\\kappa", label: L("기울기 계수", "Slope factor"), help: L("채널 전류 기울기 보정", "Correction to the channel-current slope"), code: "p[16]", unit: "1/V", min: -10, max: 10, step: 0.01 },
      { key: "seed_ip", path: ["device", "ext", "seed_ip_pA"], sym: "I_p", label: L("채널 시드 전류", "Seed current"), help: L("높은 V_D 채널 시드의 V_G = −1.8 V 전류: I_p·10^((V_G+1.8)/S)", "High-V_D channel-seed current at V_G = −1.8 V: I_p·10^((V_G+1.8)/S)"), code: "p[17]", unit: "pA", min: 0, max: 1e4, step: 0.01 },
      { key: "seed_S", path: ["device", "ext", "seed_S"], sym: "S", label: L("시드 기울기", "Seed slope"), help: L("시드 전류의 V_G 기울기", "V_G slope of the seed current"), code: "p[18]", unit: "V/dec", min: 0.01, max: 10, step: 0.01 },
      { key: "dj", path: ["device", "ext", "dj"], sym: "\\Delta_j", label: L("접합 오프셋", "Jct. offset"), help: L("접합 전위 오프셋 — 가설 검증용 레버 (hypotheses.py)", "Junction potential offset — hypothesis lever (hypotheses.py)"), code: "p[19]", unit: "V", min: -1, max: 1, step: 0.001, experimental: true },
      { key: "dm", path: ["device", "ext", "dm"], sym: "\\Delta_M", label: L("(M−1) 배율", "M−1 scale"), help: L("증배 인자 (M−1)의 로그 이동 — 가설 검증용 레버", "Log shift of the multiplication factor (M−1) — hypothesis lever"), code: "p[20]", unit: "ln", min: -5, max: 5, step: 0.01, experimental: true },
      { key: "aloc", path: ["device", "ext", "aloc"], sym: "a_{\\mathrm{loc}}", label: L("애벌랜치", "Avalanche"), help: L("국소 애벌랜치 경로의 세기 (광조사에서 σ가 유지된다는 가설)", "Strength of the local avalanche path (hypothesis for the σ kept under illumination)"), code: "p[21]", unit: "", min: 0, max: 1e3, step: 0.01, experimental: true },
      { key: "isat", path: ["device", "ext", "isat_pA"], sym: "I_{\\mathrm{sat}}", label: L("경로 포화", "Saturation"), help: L("국소 애벌랜치 경로의 포화 전류", "Saturation current of the local avalanche path"), code: "p[22]", unit: "pA", min: 0, max: 1e6, step: 0.1, experimental: true },
      { key: "dloc", path: ["device", "ext", "dloc"], sym: "\\delta_{\\mathrm{loc}}", label: L("경로 요동", "Fluctuation"), help: L("국소 애벌랜치 경로의 로그 요동 (국소 상태 작용점)", "Log fluctuation of the local avalanche path (local-state action point)"), code: "p[23]", unit: "ln", min: -10, max: 10, step: 0.01, experimental: true },
      {
        key: "loc_carriers", path: ["device", "ext", "loc_carriers"], sym: "c_{\\mathrm{loc}}", label: L("국소 경로 캐리어", "Local-path carriers"), help: L("0: 가장자리(채널 포함), 1: 벌크. 2(가장자리, 채널 제외)는 현재 엔진에서 1과 같습니다 (photo_mean.py의 p[24] > 0.5 검사)", "0: edge incl. channel, 1: bulk. 2 (edge excl. channel) currently behaves like 1 in the engine (photo_mean.py checks p[24] > 0.5)"), code: "p[24]", type: "select", experimental: true,
        options: [
          { value: 0, label: L("0 · 가장자리 (채널 포함)", "0 · edge incl. channel") },
          { value: 1, label: L("1 · 벌크", "1 · bulk") },
          {
            value: 2, label: L("2 · 가장자리 (채널 제외)", "2 · edge excl. channel"), disabled: true,
            note: L("현재 엔진에서는 1과 같음 (photo_mean.py의 p[24] > 0.5 검사)", "Currently behaves like 1 in the engine (photo_mean.py checks p[24] > 0.5)"),
          },
        ],
      },
      { key: "kappaF", path: ["device", "ext", "kappaF"], sym: "\\kappa_F", label: L("경로 전계 기울기", "Path field slope"), help: L("국소 애벌랜치 경로의 전계 기울기", "Field slope of the local avalanche path"), code: "p[25]", unit: "1/V", min: -10, max: 10, step: 0.01, experimental: true },
    ],
  },
  {
    id: "stoch",
    title: "g.stoch",
    desc: "g.stoch.desc",
    topic: "stochastic-events",
    tabs: ["device"],
    stochasticOnly: true,
    fields: [
      { key: "n_cycles", path: ["stochastic", "n_cycles"], sym: "N_{\\mathrm{cyc}}", label: L("사이클 수", "Cycles"), help: L("MC 스윕 사이클 수 (서버 상한 2000)", "Number of MC sweep cycles (server limit 2000)"), unit: "", min: 1, max: 2000, step: 1, int: true, slider: "log", main: true },
      { key: "seed", path: ["stochastic", "seed"], sym: "\\mathrm{seed}", label: L("난수 시드", "Random seed"), help: L("같은 시드는 같은 난수열을 만듭니다", "The same seed gives the same random stream"), unit: "", min: 0, max: 2 ** 32 - 1, step: 1, int: true },
      { key: "carrier_noise", path: ["stochastic", "carrier_noise"], type: "toggle", sym: "\\text{Eq. 2}", label: L("캐리어 잡음 (Eq. 2)", "Carrier noise (Eq. 2)"), help: L("II 클러스터 + 단위 사건으로 이루어진 첫 통과 잡음. 끄면 fold에서 바로 탈출합니다", "Compound first-passage noise (II clusters + unit events). When off, the device escapes exactly at the fold") },
      { key: "ld_carrier_noise", path: ["stochastic", "ld_carrier_noise"], type: "toggle", label: L("latch-down 첫 통과도 계산", "Latch-down first passage"), help: L("하향 스윕의 첫 통과(FPT)도 계산합니다 (더 느림)", "Also compute the first passage of the down sweep (slower)") },
      {
        key: "engine", path: ["stochastic", "engine"], type: "select", label: L("엔진", "Engine"), help: L("자동: 기준 보정 소자에 GIDL 작용점이면 보정 lookup 표, 그 밖에는 일반 엔진", "Auto: calibrated lookup for the reference-calibration device with the GIDL action point, general engine otherwise"),
        options: [
          { value: "auto", label: "engine.auto" },
          { value: "general", label: "engine.general" },
          { value: "calibrated_lookup", label: "engine.calibrated_lookup" },
        ],
      },
      { key: "n_traces", path: ["stochastic", "n_traces"], sym: "N_{\\mathrm{tr}}", label: L("저장 궤적 수", "Stored traces"), help: L("I–V 패널에 그릴 사이클 궤적 수", "Number of cycle traces returned for the I–V panel"), unit: "", min: 0, max: 50, step: 1, int: true },
    ],
  },
  {
    id: "local",
    title: "g.local",
    desc: "g.local.desc",
    topic: "local-states",
    tabs: ["device"],
    stochasticOnly: true,
    custom: ["local-warning"],
    fields: [
      { key: "ls_mode", path: ["stochastic", "local_state", "mode"], type: "segmented", label: L("상태 모드", "State mode"), help: L("고정: 사이클마다 한 번 추출, 진화: 시간에 따른 OU 과정", "Frozen: one draw per cycle; evolving: OU process in time"), options: LOCAL_MODE_OPTIONS, main: true },
      { key: "ls_action", path: ["stochastic", "local_state", "action"], type: "select", label: L("작용점", "Action point"), help: L("국소 상태의 요동이 작용하는 모델 파라미터", "Model parameter the fluctuating local state acts on"), options: LOCAL_ACTION_OPTIONS, show: (c) => c.root.stochastic.local_state.mode !== "none" },
      {
        key: "ls_sigma", path: ["stochastic", "local_state", "sigma"], sym: "\\sigma_{\\phi}", label: L("상태 표준편차", "State SD"), help: L("작용점 상태의 표준편차 (GIDL·접합은 V, 그 밖에는 ln 단위)", "SD of the action-point state (V for GIDL/junction, ln units otherwise)"),
        unit: (c) => (isVolt(c) ? "mV" : "ln"), scale: (c) => (isVolt(c) ? 1e3 : 1), min: 0, max: 2000, step: 0.1, slider: true, show: (c) => c.root.stochastic.local_state.mode !== "none", main: true,
      },
      { key: "ls_tau", path: ["stochastic", "local_state", "tau_s"], sym: "\\tau_{\\phi}", label: L("상태 상관 시간", "Correlation time"), help: L("OU 상관 시간 (진화 모드)", "OU correlation time (evolving mode)"), unit: "s", min: 1e-3, max: 1e4, slider: "log", show: (c) => c.root.stochastic.local_state.mode === "evolving" },
      { key: "ls_sigmaE", path: ["stochastic", "local_state", "sigma_E_V"], sym: "\\sigma_{\\phi E}", label: L("이미터 상태 SD", "Emitter SD"), help: L("소스 가장자리 국소 상태의 표준편차 (p[10]); 0이면 끕니다", "SD of the source-edge local state (p[10]); 0 turns it off"), code: "p[10]", unit: "mV", scale: 1e3, min: 0, max: 20, step: 0.001, show: (c) => c.root.stochastic.local_state.mode !== "none" },
      { key: "ls_tauE", path: ["stochastic", "local_state", "tau_E_s"], sym: "\\tau_E", label: L("이미터 상관 시간", "Emitter corr. time"), help: L("이미터 상태의 OU 상관 시간", "OU correlation time of the emitter state"), unit: "s", min: 1e-3, max: 1e4, slider: "log", show: (c) => c.root.stochastic.local_state.mode === "evolving" },
      { key: "ls_trend", path: ["stochastic", "local_state", "acquisition_trend"], type: "toggle", label: L("획득 추세 (기준 측정 기록)", "Acquisition trend"), help: L("기준 측정 기록(암조건 100회 스윕)에서 보정한 상향 스윕 추세 — 그 기록에만 해당합니다", "Up-sweep trend calibrated on the reference record (dark, 100 sweeps) — applies to that record only"), show: (c) => c.root.stochastic.local_state.mode === "evolving" },
    ],
  },
  {
    id: "numerics",
    advanced: true,
    title: "g.numerics",
    desc: "g.numerics.desc",
    topic: "numerics",
    tabs: ["device"],
    collapsed: true,
    fields: [
      { key: "grid", path: ["device", "numerics", "grid"], sym: "N_{\\mathrm{grid}}", label: L("상태 격자 점", "Grid points"), help: L("classify()의 state_grid 점 수", "Number of state_grid points for classify()"), unit: "", min: 201, max: 2001, step: 1, int: true, slider: true },
      { key: "fold_nodes", path: ["stochastic", "fold_nodes"], sym: "N_{\\mathrm{fold}}", label: L("fold 노드", "Fold nodes"), help: L("상태별 fold 표의 노드 수 (≤ 61)", "Nodes of the fold-vs-state table (≤ 61)"), unit: "", min: 3, max: 61, step: 1, int: true, show: (c) => c.mode === "stochastic" },
      { key: "hazard_nodes", path: ["stochastic", "hazard_nodes"], sym: "N_{h}", label: L("hazard 노드", "Hazard nodes"), help: L("첫 통과 hazard의 노드 수 (≤ 9, 많을수록 느림)", "First-passage hazard nodes (≤ 9; more is slower)"), unit: "", min: 1, max: 9, step: 1, int: true, show: (c) => c.mode === "stochastic" },
    ],
  },
  // ------------------------------------------------------------------ circuit
  {
    id: "bench",
    title: "g.bench",
    desc: "g.bench.desc",
    topic: "circuit-element",
    tabs: ["circuit"],
    custom: ["bench"],
    fields: [],
  },
  {
    id: "solver",
    advanced: true,
    title: "g.solver",
    desc: "g.solver.desc",
    topic: "circuit-element",
    tabs: ["circuit"],
    fields: [
      { key: "method", path: ["circuit", "solver", "method"], type: "segmented", label: L("적분법", "Integrator"), help: L("후진 오일러(BE) 또는 사다리꼴(TRAP)", "Backward Euler (BE) or trapezoidal rule (TRAP)"), options: [{ value: "BE", label: L("BE", "BE") }, { value: "TRAP", label: L("TRAP", "TRAP") }] },
      { key: "dt_min", path: ["circuit", "solver", "dt_min_s"], sym: "\\Delta t_{\\min}", label: L("최소 Δt", "Minimum Δt"), help: L("적응 스텝의 하한 (자동: max(1e-15, 1e-13·t_end))", "Lower bound of the adaptive step (auto: max(1e-15, 1e-13·t_end))"), unit: "s", min: 1e-18, max: 1, auto: true },
      { key: "dt_max", path: ["circuit", "solver", "dt_max_s"], sym: "\\Delta t_{\\max}", label: L("최대 Δt", "Maximum Δt"), help: L("적응 스텝의 상한 (자동: t_end/2000)", "Upper bound of the adaptive step (auto: t_end/2000)"), unit: "s", min: 1e-15, max: 1e4, auto: true },
      { key: "reltol", path: ["circuit", "solver", "reltol"], sym: "\\epsilon_{\\mathrm{rel}}", label: L("상대 허용오차", "Rel. tolerance"), help: L("스텝당 변화 한계의 배율: reltol = 1e-3이면 |Δu| ≤ 10 mV, |Δln I| ≤ 0.2, |Δv| ≤ 20 mV", "Relative tolerance; scales the per-step limits: at reltol = 1e-3, |Δu| ≤ 10 mV, |Δln I| ≤ 0.2, |Δv| ≤ 20 mV"), unit: "", min: 1e-6, max: 0.5, slider: "log" },
      { key: "max_steps", path: ["circuit", "solver", "max_steps"], sym: "N_{\\max}", label: L("최대 스텝", "Max steps"), help: L("시간 스텝 수의 상한 (서버 상한 2·10⁶)", "Maximum number of time steps (server limit 2·10⁶)"), unit: "", min: 100, max: 2e6, step: 1, int: true, slider: "log" },
      // latch-event detection is used by both modes (deterministic and stochastic transients)
      { key: "c_ith", path: ["circuit", "detect", "i_threshold_A"], sym: "I_{\\mathrm{th}}", label: L("스위칭 판정 전류", "Detect threshold"), help: L("래치 사건 판정 문턱 — I_D가 이 값을 넘어 올라가면 latch-up, 내려가면 latch-down (두 모드 공통)", "Current threshold for latch events — I_D crossing it upward is a latch-up, downward a latch-down (both modes)"), unit: "A", min: 1e-15, max: 1e-3, slider: "log" },
      { key: "tau_frac", path: ["circuit", "solver", "tau_frac"], sym: "f_\\tau", label: L("확률 스텝 비율", "Step fraction"), help: L("확률 과도해석의 스텝 제한: h ≤ tau_frac · τ_rel", "Stochastic step limit: h ≤ tau_frac · τ_rel"), unit: "", min: 1e-4, max: 1, slider: "log", show: (c) => c.mode === "stochastic" },
      { key: "max_ev", path: ["circuit", "solver", "max_events_per_step"], sym: "N_{\\mathrm{ev}}", label: L("스텝당 최대 사건", "Max events/step"), help: L("스텝당 기대 사건 수의 상한", "Upper bound on the expected number of events per step"), unit: "", min: 1, max: 1e6, step: 1, int: true, show: (c) => c.mode === "stochastic" },
      { key: "noise_dt_min", path: ["circuit", "solver", "noise_dt_min_s"], sym: "\\Delta t_{\\mathrm{noise}}", label: L("잡음 최소 Δt", "Noise Δt floor"), help: L("tau_frac·τ_rel이 이 값 이상인 구간에서만 캐리어 잡음을 풀고, 나머지는 드리프트만 계산합니다", "Carrier noise is resolved only where tau_frac·τ_rel ≥ this value; elsewhere drift only"), unit: "s", min: 1e-15, max: 1, slider: "log", show: (c) => c.mode === "stochastic" },
      { key: "gauss_th", path: ["circuit", "solver", "gauss_threshold"], sym: "\\bar N_{G}", label: L("가우스 근사 문턱", "Gaussian limit"), help: L("평균 사건 수가 이 값보다 크면 Poisson 대신 가우스 근사를 씁니다", "Poisson counts with a mean above this value use the Gaussian limit"), unit: "", min: 1, max: 1e6, show: (c) => c.mode === "stochastic" },
    ],
  },
  {
    id: "cstoch",
    title: "g.cstoch",
    desc: "g.cstoch.desc",
    topic: "stochastic-events",
    tabs: ["circuit"],
    stochasticOnly: true,
    custom: ["local-warning"],
    fields: [
      { key: "c_runs", path: ["circuit", "stochastic", "n_runs"], sym: "N_{\\mathrm{run}}", label: L("실행 횟수", "Runs"), help: L("독립 과도해석 횟수 (≤ 200, 파형은 처음 8개만 반환)", "Number of independent transients (≤ 200; waveforms for the first 8)"), unit: "", min: 1, max: 200, step: 1, int: true, slider: "log", main: true },
      { key: "c_seed", path: ["circuit", "stochastic", "seed"], sym: "\\mathrm{seed}", label: L("난수 시드", "Random seed"), help: L("같은 시드는 같은 난수열을 만듭니다", "The same seed gives the same random stream"), unit: "", min: 0, max: 2 ** 32 - 1, step: 1, int: true },
      { key: "c_noise", path: ["circuit", "stochastic", "carrier_noise"], type: "toggle", label: L("캐리어 잡음 (Eq. 2)", "Carrier noise (Eq. 2)"), help: L("매 스텝 Q_B 증분: Poisson 단위 사건 + II 클러스터", "Q_B increments at every step: Poisson unit events + II clusters") },
      { key: "c_ls_mode", path: ["circuit", "stochastic", "local_state", "mode"], type: "segmented", label: L("국소 상태", "Local state"), help: L("고정: 실행마다 한 번 추출, 진화: OU 과정", "Frozen: one draw per run; evolving: OU process"), options: LOCAL_MODE_OPTIONS },
      { key: "c_ls_action", path: ["circuit", "stochastic", "local_state", "action"], type: "select", label: L("작용점", "Action point"), help: L("국소 상태의 요동이 작용하는 모델 파라미터", "Model parameter the fluctuating local state acts on"), options: LOCAL_ACTION_OPTIONS, show: (c) => c.root.circuit.stochastic.local_state.mode !== "none" },
      {
        key: "c_ls_sigma", path: ["circuit", "stochastic", "local_state", "sigma"], sym: "\\sigma_{\\phi}", label: L("상태 표준편차", "State SD"), help: L("작용점 상태의 표준편차", "SD of the action-point state"),
        unit: (c) => (isVoltC(c) ? "mV" : "ln"), scale: (c) => (isVoltC(c) ? 1e3 : 1), min: 0, max: 2000, step: 0.1, show: (c) => c.root.circuit.stochastic.local_state.mode !== "none",
      },
      { key: "c_ls_tau", path: ["circuit", "stochastic", "local_state", "tau_s"], sym: "\\tau_{\\phi}", label: L("상태 상관 시간", "Correlation time"), help: L("OU 상관 시간", "OU correlation time"), unit: "s", min: 1e-6, max: 1e4, slider: "log", show: (c) => c.root.circuit.stochastic.local_state.mode === "evolving" },
    ],
  },
];

export function groupVisible(g: GroupDef, c: Ctx): boolean {
  if (!g.tabs.includes(c.tab === "validation" || c.tab === "physics" ? "device" : c.tab)) return false;
  if (g.stochasticOnly && c.mode !== "stochastic") return false;
  return g.show ? g.show(c) : true;
}

export const unitOf = (f: FieldDef, c: Ctx) => (typeof f.unit === "function" ? f.unit(c) : f.unit ?? "");
export const scaleOf = (f: FieldDef, c: Ctx) => (typeof f.scale === "function" ? f.scale(c) : f.scale ?? 1);

/** Group paths to reset: every field path + extra paths. */
export function groupPaths(g: GroupDef): Path[] {
  return [...g.fields.map((f) => f.path), ...(g.extraPaths ?? [])];
}

export type { StochasticBlock, SweepBlock, DeviceBlock };
