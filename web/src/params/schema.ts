// Parameter groups and fields for the sidebar (docs/WEB_CONTRACT.md §5). Every field addresses the
// parameter root by `path`; values are stored exactly as the payload schema defines them (SI, or the
// unit in the key name, e.g. iph_pA) and converted for display with `scale` (display = stored × scale).
import type { L10n, TopicId } from "../content/physics/types";
import type { BenchId, CircuitStochBlock, DeviceBlock, Mode, SolverBlock, StochasticBlock, SweepBlock } from "../api/types";
import type { StrKey } from "../i18n/strings";
import type { Path } from "../utils/object";

export type Tab = "device" | "circuit" | "validation" | "physics";

export interface CircuitParams {
  bench: BenchId;
  bench_params: Record<BenchId, Record<string, number | string | boolean>>;
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
  type?: "number" | "toggle" | "select" | "segmented";
  options?: Option[];
  show?: (c: Ctx) => boolean;
  experimental?: boolean;
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

export const GROUPS: GroupDef[] = [
  {
    id: "bias",
    title: "g.bias",
    desc: "g.bias.desc",
    topic: "charge-balance",
    tabs: ["device", "circuit"],
    fields: [
      { key: "vg", path: ["device", "vg"], sym: "V_G", label: L("게이트 전압", "Gate voltage"), help: L("게이트-소스 전압 (채널 전류와 GIDL 전계에 작용)", "Gate–source voltage (enters the channel current and the gate-edge GIDL field)"), code: "p[11]", unit: "V", min: -6, max: 1, step: 0.01, slider: true },
      { key: "vd_max", path: ["sweep", "vd_max_V"], sym: "V_{D,\\max}", label: L("스윕 최대 V_D", "Sweep maximum V_D"), help: L("삼각 스윕 0 → V_D,max → 0의 꼭짓점 (서버 상한 8 V)", "Apex of the triangular sweep 0 → V_D,max → 0 (server cap 8 V)"), unit: "V", min: 0.5, max: 8, step: 0.05, slider: true, show: isDevice },
      { key: "rate", path: ["sweep", "rate_V_per_s"], sym: "\\dot V_D", label: L("램프 속도", "Ramp rate"), help: L("드레인 전압 스윕 속도 — hazard 적분과 MC 시간축", "Drain-voltage sweep rate — sets the hazard integral and the MC time axis"), unit: "V/s", min: 1e-3, max: 1e5, slider: "log", show: isDevice },
      { key: "dv", path: ["sweep", "dv_V"], sym: "\\Delta V", label: L("전압 스텝", "Voltage step"), help: L("스윕 전압 간격 (MC 시간 스텝 Δt = ΔV / 램프 속도)", "Sweep voltage step (MC time step Δt = ΔV / rate)"), unit: "mV", scale: 1e3, min: 0.1, max: 50, step: 0.1, show: isDevice },
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
    title: "g.state",
    desc: "g.state.desc",
    topic: "local-states",
    tabs: ["device", "circuit"],
    fields: [
      { key: "dphiG0", path: ["device", "state", "delta_phi_G0_V"], sym: "\\delta\\phi_{G0}", label: L("드레인 가장자리 상태 중심", "Drain-edge state centre"), help: L("GIDL 전계 오프셋 p[9]에 더해지는 평균 이동. dV_LU/dφ_G ≈ −0.80 V/V", "Mean shift added to the GIDL offset p[9]. dV_LU/dφ_G ≈ −0.80 V/V"), code: "p[9] +", unit: "mV", scale: 1e3, min: -500, max: 500, step: 0.1, slider: true },
      { key: "dphiE0", path: ["device", "state", "delta_phi_E0_V"], sym: "\\delta\\phi_{E0}", label: L("소스 가장자리 상태 중심", "Source-edge state centre"), help: L("에미터 오프셋 p[10]에 더해지는 평균 이동. dV_LD/dφ_E ≈ −41 V/V", "Mean shift added to the emitter offset p[10]. dV_LD/dφ_E ≈ −41 V/V"), code: "p[10] +", unit: "mV", scale: 1e3, min: -10, max: 10, step: 0.01, slider: true },
    ],
  },
  {
    id: "calib",
    title: "g.calib",
    desc: "g.calib.desc",
    topic: "parameters",
    tabs: ["device", "circuit"],
    collapsed: true,
    fields: [
      { key: "beta", path: ["device", "calib", "beta"], sym: "\\beta", label: L("확산 비율", "Diffusion ratio"), help: L("out-diffusion 손실 비율", "Out-diffusion loss ratio"), code: "p[0]", unit: "", min: 0.01, max: 100, slider: "log" },
      { key: "tau_bulk", path: ["device", "calib", "tau_bulk_s"], sym: "\\tau_{\\mathrm{bulk}}", label: L("벌크 수명", "Bulk lifetime"), help: L("body 벌크 SRH 재결합 수명", "Body bulk SRH recombination lifetime"), code: "p[1]", unit: "µs", scale: 1e6, min: 1e-4, max: 1e4, slider: "log" },
      { key: "tau_junction", path: ["device", "calib", "tau_junction_s"], sym: "\\tau_{j}", label: L("접합 수명", "Junction lifetime"), help: L("공핍 영역 SRH 수명", "Depletion-region SRH lifetime"), code: "p[2]", unit: "ns", scale: 1e9, min: 1e-3, max: 1e5, slider: "log" },
      { key: "r_contact", path: ["device", "calib", "r_contact_ohm"], sym: "R_c", label: L("접촉 저항", "Contact resistance"), help: L("직렬 접촉 저항", "Series contact resistance"), code: "p[3]", unit: "Ω", min: 0, max: 1e6 },
      { key: "l_gidl", path: ["device", "calib", "l_gidl_nm"], sym: "l_{\\mathrm{GIDL}}", label: L("GIDL 길이", "GIDL length"), help: L("게이트 가장자리 BTBT 영역 길이", "Gate-edge BTBT region length"), code: "p[4]", unit: "nm", min: 0.1, max: 500, slider: "log" },
      { key: "t_access", path: ["device", "calib", "t_access_nm"], sym: "t_{\\mathrm{acc}}", label: L("접근 영역 두께", "Access thickness"), help: L("소스/드레인 접근 영역 두께", "Source/drain access-region thickness"), code: "p[5]", unit: "nm", min: 0.1, max: 100, slider: "log" },
      { key: "na_access", path: ["device", "calib", "na_access_cm3"], sym: "N_{A,\\mathrm{acc}}", label: L("접근 영역 도핑", "Access doping"), help: L("접근 영역 억셉터 농도", "Access-region acceptor concentration"), code: "p[6]", unit: "10¹⁷ cm⁻³", scale: 1e-17, min: 1e-3, max: 1e4, slider: "log" },
      { key: "l_access", path: ["device", "calib", "l_access_nm"], sym: "L_{\\mathrm{acc}}", label: L("접근 영역 길이", "Access length"), help: L("접근 영역 길이", "Access-region length"), code: "p[7]", unit: "nm", min: 1, max: 1000, slider: "log" },
      { key: "tau_ratio", path: ["device", "calib", "tau_ratio"], sym: "\\tau_p/\\tau_n", label: L("수명 비", "Lifetime ratio"), help: L("정공/전자 SRH 수명 비", "Hole/electron SRH lifetime ratio"), code: "p[8]", unit: "", min: 1e-3, max: 1e5, slider: "log" },
      { key: "phi_gidl0", path: ["device", "calib", "phi_gidl0_V"], sym: "\\phi_{G,0}", label: L("GIDL 전위 오프셋", "GIDL potential offset"), help: L("드레인 가장자리 상태 평균 (보정값)", "Drain-edge state mean (calibrated)"), code: "p[9]", unit: "mV", scale: 1e3, min: -500, max: 500, step: 0.001 },
      { key: "phi_emitter0", path: ["device", "calib", "phi_emitter0_V"], sym: "\\phi_{E,0}", label: L("에미터 전위 오프셋", "Emitter potential offset"), help: L("소스 가장자리 상태 평균 (보정값)", "Source-edge state mean (calibrated)"), code: "p[10]", unit: "mV", scale: 1e3, min: -50, max: 50, step: 0.001 },
      { key: "ch_ii", path: ["device", "calib", "channel_ii_scale"], sym: "s_{\\mathrm{II,ch}}", label: L("채널 II 배율", "Channel-II scale"), help: L("채널 전자 충돌 이온화 배율", "Scale of channel-electron impact ionisation"), code: "p[12]", unit: "", min: 0, max: 100, slider: "log" },
    ],
  },
  {
    id: "ext",
    title: "g.ext",
    desc: "g.ext.desc",
    topic: "open-problems",
    tabs: ["device", "circuit"],
    collapsed: true,
    custom: ["seed"],
    fields: [
      { key: "dibl", path: ["device", "ext", "dibl"], sym: "\\eta", label: L("DIBL", "DIBL"), help: L("드레인 유도 장벽 저하 계수", "Drain-induced barrier lowering coefficient"), code: "p[14]", unit: "V/V", min: 0, max: 1, step: 0.001 },
      { key: "gamma", path: ["device", "ext", "gamma"], sym: "\\gamma", label: L("Body 결합", "Body coupling"), help: L("body 전위의 채널 결합 (−1.1 V 채널 seed 옵션)", "Body-to-channel coupling (−1.1 V channel-seed option)"), code: "p[15]", unit: "", min: 0, max: 2, step: 0.0001 },
      { key: "kappa", path: ["device", "ext", "kappa"], sym: "\\kappa", label: L("기울기 계수", "Slope factor"), help: L("채널 기울기 보정", "Channel slope correction"), code: "p[16]", unit: "1/V", min: -10, max: 10, step: 0.01 },
      { key: "seed_ip", path: ["device", "ext", "seed_ip_pA"], sym: "I_p", label: L("고 V_D 채널 seed", "High-V_D channel seed"), help: L("V_G = −1.8 V에서의 seed 전류, I_p·10^((V_G+1.8)/S)", "Seed current at V_G = −1.8 V, I_p·10^((V_G+1.8)/S)"), code: "p[17]", unit: "pA", min: 0, max: 1e4, step: 0.01 },
      { key: "seed_S", path: ["device", "ext", "seed_S"], sym: "S", label: L("seed 기울기", "Seed slope"), help: L("seed 전류의 V_G 기울기", "V_G slope of the seed current"), code: "p[18]", unit: "V/dec", min: 0.01, max: 10, step: 0.01 },
      { key: "dj", path: ["device", "ext", "dj"], sym: "\\Delta_j", label: L("접합 전위 오프셋", "Junction potential offset"), help: L("가설 레버 (hypotheses.py)", "Hypothesis lever (hypotheses.py)"), code: "p[19]", unit: "V", min: -1, max: 1, step: 0.001, experimental: true },
      { key: "dm", path: ["device", "ext", "dm"], sym: "\\Delta_M", label: L("(M−1) 로그 배율", "log scale of (M−1)"), help: L("증배 인자 로그 이동 (가설 레버)", "Log shift of the multiplication factor (hypothesis lever)"), code: "p[20]", unit: "ln", min: -5, max: 5, step: 0.01, experimental: true },
      { key: "aloc", path: ["device", "ext", "aloc"], sym: "a_{\\mathrm{loc}}", label: L("국소 avalanche 강도", "Local avalanche strength"), help: L("국소 경로 세기 (광조사 σ 유지 가설)", "Local path strength (photo σ hypothesis)"), code: "p[21]", unit: "", min: 0, max: 1e3, step: 0.01, experimental: true },
      { key: "isat", path: ["device", "ext", "isat_pA"], sym: "I_{\\mathrm{sat}}", label: L("국소 경로 포화", "Local path saturation"), help: L("국소 경로 포화 전류", "Local path saturation current"), code: "p[22]", unit: "pA", min: 0, max: 1e6, step: 0.1, experimental: true },
      { key: "dloc", path: ["device", "ext", "dloc"], sym: "\\delta_{\\mathrm{loc}}", label: L("국소 경로 로그 요동", "Local path log fluctuation"), help: L("국소 경로의 로그 요동 (local state 작용점)", "Log fluctuation of the local path (local-state action point)"), code: "p[23]", unit: "ln", min: -10, max: 10, step: 0.01, experimental: true },
      {
        key: "loc_carriers", path: ["device", "ext", "loc_carriers"], sym: "c_{\\mathrm{loc}}", label: L("국소 경로 carrier 정의", "Local path carriers"), help: L("0 가장자리(채널 포함), 1 벌크. 2(채널 제외)는 현재 엔진에서 1과 동일 (photo_mean.py p[24] > 0.5)", "0 edge incl. channel, 1 bulk. 2 (edge excl. channel) is currently identical to 1 in the engine (photo_mean.py p[24] > 0.5)"), code: "p[24]", type: "select", experimental: true,
        options: [
          { value: 0, label: L("0 · 가장자리 (채널 포함)", "0 · edge incl. channel") },
          { value: 1, label: L("1 · 벌크", "1 · bulk") },
          {
            value: 2, label: L("2 · 가장자리 (채널 제외)", "2 · edge excl. channel"), disabled: true,
            note: L("현재 엔진에서는 1과 동일 (photo_mean.py의 p[24] > 0.5 검사)", "currently identical to 1 in the engine (photo_mean.py p[24] check)"),
          },
        ],
      },
      { key: "kappaF", path: ["device", "ext", "kappaF"], sym: "\\kappa_F", label: L("κ_F", "κ_F"), help: L("국소 경로 전계 기울기", "Local path field slope"), code: "p[25]", unit: "1/V", min: -10, max: 10, step: 0.01, experimental: true },
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
      { key: "n_cycles", path: ["stochastic", "n_cycles"], sym: "N_{\\mathrm{cyc}}", label: L("사이클 수", "Cycles"), help: L("MC 스윕 사이클 수 (서버 상한 2000)", "Number of MC sweep cycles (server cap 2000)"), unit: "", min: 1, max: 2000, step: 1, int: true, slider: "log" },
      { key: "seed", path: ["stochastic", "seed"], sym: "\\mathrm{seed}", label: L("난수 시드", "Random seed"), help: L("재현 가능한 난수열", "Reproducible random stream"), unit: "", min: 0, max: 2 ** 32 - 1, step: 1, int: true },
      { key: "carrier_noise", path: ["stochastic", "carrier_noise"], type: "toggle", sym: "\\text{Eq. 2}", label: L("Carrier 잡음 (Eq. 2)", "Carrier noise (Eq. 2)"), help: L("II 클러스터 + unit 사건 첫 통과 잡음. 끄면 fold에서 탈출", "Compound first-passage noise (II clusters + unit events). Off → escape at the fold") },
      { key: "ld_carrier_noise", path: ["stochastic", "ld_carrier_noise"], type: "toggle", label: L("Latch-down FPT도 계산", "Latch-down FPT too"), help: L("하향 스윕 첫 통과도 계산 (느림)", "Also compute the latch-down first passage (slower)") },
      {
        key: "engine", path: ["stochastic", "engine"], type: "select", label: L("엔진", "Engine"), help: L("자동: 논문 기준 소자 + GIDL이면 보정 lookup", "auto: calibrated lookup for the paper reference device with GIDL"),
        options: [
          { value: "auto", label: "engine.auto" },
          { value: "general", label: "engine.general" },
          { value: "calibrated_lookup", label: "engine.calibrated_lookup" },
        ],
      },
      { key: "n_traces", path: ["stochastic", "n_traces"], sym: "N_{\\mathrm{tr}}", label: L("저장 궤적 수", "Stored traces"), help: L("I–V 패널에 그릴 사이클 궤적 수", "Cycle traces returned for the I–V panel"), unit: "", min: 0, max: 50, step: 1, int: true },
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
      { key: "ls_mode", path: ["stochastic", "local_state", "mode"], type: "segmented", label: L("상태 모드", "State mode"), help: L("frozen: 사이클당 1회 추출, evolving: 시간 OU 과정", "frozen: one draw per cycle; evolving: OU process in time"), options: LOCAL_MODE_OPTIONS },
      { key: "ls_action", path: ["stochastic", "local_state", "action"], type: "select", label: L("작용점", "Action point"), help: L("요동이 작용하는 모델 파라미터", "Model parameter the fluctuating state acts on"), options: LOCAL_ACTION_OPTIONS, show: (c) => c.root.stochastic.local_state.mode !== "none" },
      {
        key: "ls_sigma", path: ["stochastic", "local_state", "sigma"], sym: "\\sigma_{\\phi}", label: L("상태 표준편차", "State SD"), help: L("작용점 상태의 SD (GIDL/접합: V, 그 외: ln 단위)", "SD of the action-point state (V for GIDL/junction, ln-units otherwise)"),
        unit: (c) => (isVolt(c) ? "mV" : "ln"), scale: (c) => (isVolt(c) ? 1e3 : 1), min: 0, max: 2000, step: 0.1, slider: true, show: (c) => c.root.stochastic.local_state.mode !== "none",
      },
      { key: "ls_tau", path: ["stochastic", "local_state", "tau_s"], sym: "\\tau_{\\phi}", label: L("상태 상관 시간", "State correlation time"), help: L("OU 상관 시간 (evolving)", "OU correlation time (evolving)"), unit: "s", min: 1e-3, max: 1e4, slider: "log", show: (c) => c.root.stochastic.local_state.mode === "evolving" },
      { key: "ls_sigmaE", path: ["stochastic", "local_state", "sigma_E_V"], sym: "\\sigma_{\\phi E}", label: L("에미터 상태 SD", "Emitter state SD"), help: L("소스 가장자리 상태 SD (p[10]); 0이면 끔", "Source-edge state SD (p[10]); 0 disables"), code: "p[10]", unit: "mV", scale: 1e3, min: 0, max: 20, step: 0.001, show: (c) => c.root.stochastic.local_state.mode !== "none" },
      { key: "ls_tauE", path: ["stochastic", "local_state", "tau_E_s"], sym: "\\tau_E", label: L("에미터 상관 시간", "Emitter correlation time"), help: L("에미터 OU 상관 시간", "Emitter OU correlation time"), unit: "s", min: 1e-3, max: 1e4, slider: "log", show: (c) => c.root.stochastic.local_state.mode === "evolving" },
      { key: "ls_trend", path: ["stochastic", "local_state", "acquisition_trend"], type: "toggle", label: L("획득 추세 (논문 기록)", "Acquisition trend (paper record)"), help: L("보정된 상향 스윕 추세 — 논문 기록에만 해당", "Calibrated up-sweep trend — paper record only"), show: (c) => c.root.stochastic.local_state.mode === "evolving" },
    ],
  },
  {
    id: "numerics",
    title: "g.numerics",
    desc: "g.numerics.desc",
    topic: "numerics",
    tabs: ["device"],
    collapsed: true,
    fields: [
      { key: "grid", path: ["device", "numerics", "grid"], sym: "N_{\\mathrm{grid}}", label: L("상태 격자 점", "State-grid points"), help: L("classify()의 state_grid 점 수", "state_grid points for classify()"), unit: "", min: 201, max: 2001, step: 1, int: true, slider: true },
      { key: "fold_nodes", path: ["stochastic", "fold_nodes"], sym: "N_{\\mathrm{fold}}", label: L("fold 노드", "Fold nodes"), help: L("상태별 fold 표 노드 수 (≤ 61)", "Nodes of the fold-vs-state table (≤ 61)"), unit: "", min: 3, max: 61, step: 1, int: true, show: (c) => c.mode === "stochastic" },
      { key: "hazard_nodes", path: ["stochastic", "hazard_nodes"], sym: "N_{h}", label: L("hazard 노드", "Hazard nodes"), help: L("FPT hazard 노드 수 (≤ 9, 느림)", "FPT hazard nodes (≤ 9, slow)"), unit: "", min: 1, max: 9, step: 1, int: true, show: (c) => c.mode === "stochastic" },
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
    title: "g.solver",
    desc: "g.solver.desc",
    topic: "circuit-element",
    tabs: ["circuit"],
    fields: [
      { key: "method", path: ["circuit", "solver", "method"], type: "segmented", label: L("적분법", "Integrator"), help: L("후진 오일러(BE) 또는 사다리꼴(TRAP)", "Backward Euler (BE) or trapezoidal (TRAP)"), options: [{ value: "BE", label: L("BE", "BE") }, { value: "TRAP", label: L("TRAP", "TRAP") }] },
      { key: "dt_min", path: ["circuit", "solver", "dt_min_s"], sym: "\\Delta t_{\\min}", label: L("최소 Δt", "Minimum Δt"), help: L("적응 스텝 하한", "Adaptive step lower bound"), unit: "s", min: 1e-18, max: 1, slider: "log" },
      { key: "dt_max", path: ["circuit", "solver", "dt_max_s"], sym: "\\Delta t_{\\max}", label: L("최대 Δt", "Maximum Δt"), help: L("적응 스텝 상한 (스윕: ≤ 1 ms)", "Adaptive step upper bound (sweeps: ≤ 1 ms)"), unit: "s", min: 1e-15, max: 10, slider: "log" },
      { key: "reltol", path: ["circuit", "solver", "reltol"], sym: "\\epsilon_{\\mathrm{rel}}", label: L("상대 허용오차", "Relative tolerance"), help: L("Newton / 스텝 제어 허용오차", "Newton / step-control tolerance"), unit: "", min: 1e-10, max: 0.1, slider: "log" },
      { key: "max_steps", path: ["circuit", "solver", "max_steps"], sym: "N_{\\max}", label: L("최대 스텝", "Max steps"), help: L("시간 스텝 상한 (서버 상한 2·10⁶)", "Time-step cap (server cap 2·10⁶)"), unit: "", min: 100, max: 2e6, step: 1, int: true, slider: "log" },
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
      { key: "c_runs", path: ["circuit", "stochastic", "n_runs"], sym: "N_{\\mathrm{run}}", label: L("Run 수", "Runs"), help: L("독립 과도해석 수 (≤ 200, 파형은 처음 8개)", "Independent transients (≤ 200; waveforms for the first 8)"), unit: "", min: 1, max: 200, step: 1, int: true, slider: "log" },
      { key: "c_seed", path: ["circuit", "stochastic", "seed"], sym: "\\mathrm{seed}", label: L("난수 시드", "Random seed"), help: L("재현 가능한 난수열", "Reproducible random stream"), unit: "", min: 0, max: 2 ** 32 - 1, step: 1, int: true },
      { key: "c_noise", path: ["circuit", "stochastic", "carrier_noise"], type: "toggle", label: L("사건 증분 잡음 (Eq. 2)", "Event-increment noise (Eq. 2)"), help: L("Poisson unit 사건 + II 클러스터", "Poisson unit events + II clusters") },
      { key: "c_ls_mode", path: ["circuit", "stochastic", "local_state", "mode"], type: "segmented", label: L("Local state", "Local state"), help: L("frozen: run당 1회 추출, evolving: OU", "frozen: one draw per run; evolving: OU"), options: LOCAL_MODE_OPTIONS },
      { key: "c_ls_action", path: ["circuit", "stochastic", "local_state", "action"], type: "select", label: L("작용점", "Action point"), help: L("요동이 작용하는 모델 파라미터", "Model parameter the state acts on"), options: LOCAL_ACTION_OPTIONS, show: (c) => c.root.circuit.stochastic.local_state.mode !== "none" },
      {
        key: "c_ls_sigma", path: ["circuit", "stochastic", "local_state", "sigma"], sym: "\\sigma_{\\phi}", label: L("상태 표준편차", "State SD"), help: L("작용점 상태의 SD", "SD of the action-point state"),
        unit: (c) => (isVoltC(c) ? "mV" : "ln"), scale: (c) => (isVoltC(c) ? 1e3 : 1), min: 0, max: 2000, step: 0.1, show: (c) => c.root.circuit.stochastic.local_state.mode !== "none",
      },
      { key: "c_ls_tau", path: ["circuit", "stochastic", "local_state", "tau_s"], sym: "\\tau_{\\phi}", label: L("상태 상관 시간", "State correlation time"), help: L("OU 상관 시간", "OU correlation time"), unit: "s", min: 1e-6, max: 1e4, slider: "log", show: (c) => c.root.circuit.stochastic.local_state.mode === "evolving" },
      { key: "c_ith", path: ["circuit", "detect", "i_threshold_A"], sym: "I_{\\mathrm{th}}", label: L("스위칭 판정 전류", "Switch-detect current"), help: L("latch 사건 판정 문턱", "Threshold used to detect latch events"), unit: "A", min: 1e-15, max: 1e-2, slider: "log" },
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
