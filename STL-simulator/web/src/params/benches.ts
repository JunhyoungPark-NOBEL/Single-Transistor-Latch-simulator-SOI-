// Circuit test benches (docs/WEB_CONTRACT.md §4). Keys and defaults mirror
// server/compute/circuit/benches.py (BENCH_DEFAULTS, SOLVER_DEFAULTS, STOCHASTIC_DEFAULTS).
// `null` = "auto" (resolved server-side from the device, e.g. v_max from the preset sweep, v_amp from V_LU, or
// the server's documented default such as rise_s/fall_s). Null keys are omitted from the payload
// (utils/payload.ts circuitPayload) so server/compute/circuit/benches.py BENCH_DEFAULTS applies.
import type { L10n } from "../content/physics/types";
import type { BenchId, CircuitStochBlock, SolverBlock } from "../api/types";
import type { StrKey } from "../i18n/strings";

export type BenchValue = number | string | boolean | null | number[];

export interface BenchField {
  key: string;
  sym?: string;
  label: L10n;
  help: L10n;
  unit: string;
  scale?: number;
  min?: number;
  max?: number;
  step?: number;
  slider?: boolean | "log";
  int?: boolean;
  /** null allowed = automatic value chosen by the server. */
  auto?: boolean;
  type?: "number" | "list" | "select";
  options?: { value: string; label: L10n }[];
  /** Only shown when this predicate on the bench params holds. */
  when?: (bp: Record<string, BenchValue>) => boolean;
  /** List fields: an example in this field's units. */
  placeholder?: L10n;
  /** Auto value the client can resolve (shown as "자동 = …"): the device V_G (server: device.vg), or the
   *  preset's sweep peak / rate (server: the preset's sweep section, not the edited sidebar sweep). */
  autoFrom?: "vg" | "vd_max" | "rate" | "edge" | "vref";
}
export interface BenchDef {
  id: BenchId;
  title: StrKey;
  desc: StrKey;
  defaults: Record<string, BenchValue>;
  fields: BenchField[];
}

const L = (ko: string, en: string): L10n => ({ ko, en });

const f = {
  vMin: { key: "v_min_V", sym: "V_{\\min}", label: L("램프 시작/끝 전압", "Ramp start/end"), help: L("삼각파 최저 전압", "Triangle low level"), unit: "V", min: -8, max: 8, step: 0.05 },
  vMax: { key: "v_max_V", sym: "V_{\\max}", label: L("램프 최대 전압", "Ramp peak"), help: L("자동: 프리셋의 스윕 V_D,max (기준 보정 4 V, 광조사 보정 5 V)", "Auto: the preset's sweep V_D,max (reference 4 V, illumination 5 V)"), unit: "V", min: -8, max: 8, step: 0.05, auto: true, autoFrom: "vd_max" as const },
  rate: { key: "rate_V_per_s", sym: "\\dot V", label: L("램프 속도", "Ramp rate"), help: L("자동: 프리셋 스윕 속도 (확률 실행에서 너무 느리면 1200 V/s)", "auto: preset sweep rate (stochastic runs fall back to 1200 V/s when too slow)"), unit: "V/s", min: 1e-4, max: 1e8, slider: "log" as const, auto: true, autoFrom: "rate" as const },
  nCycles: { key: "n_cycles", sym: "N_{\\mathrm{cyc}}", label: L("삼각파 사이클", "Triangle cycles"), help: L("≤ 50", "≤ 50"), unit: "", min: 1, max: 50, step: 1, int: true },
  Rs: { key: "R_s_ohm", sym: "R_s", label: L("직렬 저항", "Series resistor"), help: L("전원과 드레인 사이 저항", "Resistor between source and drain"), unit: "kΩ", scale: 1e-3, min: 1e-6, max: 1e9, slider: "log" as const },
  Cd: { key: "C_d_F", sym: "C_d", label: L("드레인 노드 용량", "Drain-node capacitance"), help: L("드레인 노드의 접지 용량", "Drain-node capacitance to ground"), unit: "fF", scale: 1e15, min: 0, max: 1e9, slider: "log" as const },
  vg: { key: "vg_V", sym: "V_G", label: L("벤치 게이트 전압", "Bench gate voltage"), help: L("자동: 바이어스·스윕의 V_G", "auto: the V_G of Bias & sweep"), unit: "V", min: -6, max: 1, step: 0.01, auto: true, autoFrom: "vg" as const },
  vBase: { key: "v_base_V", sym: "V_{\\mathrm{base}}", label: L("기저 전압", "Base level"), help: L("펄스 사이 전압", "Voltage between pulses"), unit: "V", min: -8, max: 8, step: 0.05 },
  vAmp: { key: "v_amp_V", sym: "V_{\\mathrm{amp}}", label: L("펄스 진폭", "Pulse amplitude"), help: L("자동: 결정론 V_LU + 0.10 V", "auto: deterministic V_LU + 0.10 V"), unit: "V", min: -8, max: 8, step: 0.01, auto: true },
  width: { key: "width_s", sym: "t_w", label: L("펄스 폭 (평탄부)", "Pulse width (flat top)"), help: L("high 구간", "High-level duration"), unit: "µs", scale: 1e6, min: 1e-6, max: 1e12, slider: "log" as const },
  period: { key: "period_s", sym: "T", label: L("펄스 주기", "Pulse period"), help: L("펄스 반복 주기", "Repetition period"), unit: "µs", scale: 1e6, min: 1e-6, max: 1e12, slider: "log" as const },
  // rise/fall default to the server's values (pulse/coupled 10 µs, p-bit 20 µs): edges faster than ~1 µs kick
  // the floating body through the drain-depletion charge and trigger latch-up below the static fold.
  rise: {
    key: "rise_s", sym: "t_r", label: L("상승 시간", "Rise time"),
    help: L("자동: 서버 기본값 (펄스·결합 10 µs, p-bit 20 µs). ~1 µs보다 빠른 에지는 드레인 공핍 전하를 통해 떠 있는 바디를 차서 정적 fold 아래에서도 래치업을 일으킨다", "auto: server default (pulse/coupled 10 µs, p-bit 20 µs). Edges faster than ~1 µs kick the floating body through the drain-depletion charge and trigger latch-up below the static fold"),
    unit: "µs", scale: 1e6, min: 0, max: 1e9, slider: "log" as const, auto: true, autoFrom: "edge" as const,
  },
  fall: {
    key: "fall_s", sym: "t_f", label: L("하강 시간", "Fall time"),
    help: L("자동: 서버 기본값 (펄스·결합 10 µs, p-bit 20 µs). 빠른 에지는 드레인 공핍 결합으로 바디 전하를 흔든다", "auto: server default (pulse/coupled 10 µs, p-bit 20 µs). Fast edges disturb the body charge through the drain-depletion coupling"),
    unit: "µs", scale: 1e6, min: 0, max: 1e9, slider: "log" as const, auto: true, autoFrom: "edge" as const,
  },
  nPulses: { key: "n_pulses", sym: "N_p", label: L("펄스 수", "Pulses"), help: L("≤ 2000", "≤ 2000"), unit: "", min: 1, max: 2000, step: 1, int: true },
  delay: { key: "delay_s", sym: "t_0", label: L("첫 펄스 지연", "First-pulse delay"), help: L("첫 펄스 시작 시각", "Start of the first pulse"), unit: "µs", scale: 1e6, min: 0, max: 1e12 },
};

export const BENCHES: Record<BenchId, BenchDef> = {
  load_line: {
    id: "load_line",
    title: "c.bench.load_line",
    desc: "c.bench.load_line.desc",
    defaults: { v_min_V: 0, v_max_V: null, rate_V_per_s: null, n_cycles: 1, R_s_ohm: 1e3, C_d_F: 2e-15, vg_V: null },
    fields: [f.vMin, f.vMax, f.rate, f.nCycles, f.Rs, f.Cd, f.vg],
  },
  pulse: {
    id: "pulse",
    title: "c.bench.pulse",
    desc: "c.bench.pulse.desc",
    defaults: { v_base_V: 0, v_amp_V: null, width_s: 200e-6, period_s: 1e-3, rise_s: null, fall_s: null, n_pulses: 10, delay_s: 0, R_s_ohm: 1e3, C_d_F: 2e-15, vg_V: null, amplitudes_V: [] },
    fields: [
      f.vBase, f.vAmp, f.width, f.period, f.rise, f.fall, f.nPulses, f.delay, f.Rs, f.Cd, f.vg,
      { key: "amplitudes_V", sym: "\\{V_{\\mathrm{amp}}\\}", label: L("진폭 스윕 (선택)", "Amplitude sweep (optional)"), help: L("쉼표로 구분한 진폭 목록 → P_sw vs 진폭 (≤ 25개)", "Comma-separated amplitudes → P_sw vs amplitude (≤ 25)"), unit: "V", type: "list", min: -8, max: 8, placeholder: L("예: 3.6, 3.8, 4.0 (비우면 없음)", "e.g. 3.6, 3.8, 4.0 (leave empty for none)") },
    ],
  },
  pbit: {
    id: "pbit",
    title: "c.bench.pbit",
    desc: "c.bench.pbit.desc",
    defaults: { v_low_V: 0, v_high_V: null, clock_period_s: 1e-3, clock_width_s: 200e-6, rise_s: null, fall_s: null, n_clocks: 50, R_S_ohm: 100e3, v_ref_V: null, vg_V: null, vg_list_V: [], light_list_pA: [] },
    fields: [
      { key: "v_low_V", sym: "V_{\\mathrm{low}}", label: L("펄스 low", "Pulse low"), help: L("드레인 펄스의 low 전압", "Low level of the drain pulses"), unit: "V", min: -8, max: 8, step: 0.05 },
      { key: "v_high_V", sym: "V_{\\mathrm{high}}", label: L("펄스 high", "Pulse high"), help: L("자동: V_LU − 15 mV (확률 스위칭 영역, 기준 소자에서 P(1) ≈ 0.5; 결정론 모드에서는 0)", "auto: V_LU − 15 mV (stochastic switching regime: P(1) ≈ 0.5 for the reference device; 0 deterministically)"), unit: "V", min: -8, max: 8, step: 0.01, auto: true },
      { key: "clock_period_s", sym: "T_{\\mathrm{clk}}", label: L("펄스 주기", "Pulse period"), help: L("비트 한 개당 시간", "Time per bit"), unit: "µs", scale: 1e6, min: 1e-6, max: 1e12, slider: "log" },
      { key: "clock_width_s", sym: "t_{\\mathrm{clk}}", label: L("펄스 폭", "Pulse width"), help: L("high 평탄부", "Flat top of the pulse"), unit: "µs", scale: 1e6, min: 1e-6, max: 1e12, slider: "log" },
      f.rise, f.fall,
      { key: "n_clocks", sym: "N_{\\mathrm{clk}}", label: L("비트 수", "Pulses (bits)"), help: L("≤ 5000", "≤ 5000"), unit: "", min: 1, max: 5000, step: 1, int: true, slider: "log" },
      { key: "R_S_ohm", sym: "R_S", label: L("소스 저항", "Source resistor"), help: L("소스와 접지 사이. 래치되면 V(R_S) = R_S·I_D ≈ 0.4 V (100 kΩ)", "Between the source and ground. Latched: V(R_S) = R_S·I_D ≈ 0.4 V (100 kΩ)"), unit: "kΩ", scale: 1e-3, min: 1e-6, max: 1e6, slider: "log" },
      { key: "v_ref_V", sym: "V_{\\mathrm{ref}}", label: L("비교기 기준 전압", "Comparator V_ref"), help: L("자동: R_S × 1 µA. 비트 = [V(R_S) > V_ref], 펄스 평탄부 끝에서", "auto: R_S × 1 µA. bit = [V(R_S) > V_ref] at the end of the pulse flat top"), unit: "V", min: -8, max: 8, step: 0.01, auto: true, autoFrom: "vref" },
      f.vg,
      { key: "vg_list_V", sym: "\\{V_G\\}", label: L("V_G 스윕 (선택)", "V_G sweep (optional)"), help: L("쉼표로 구분 → P(1) vs V_G", "Comma-separated → P(1) vs V_G"), unit: "V", type: "list", min: -6, max: 1, placeholder: L("예: −2.2, −2.0, −1.8 (비우면 없음)", "e.g. −2.2, −2.0, −1.8 (leave empty for none)") },
      { key: "light_list_pA", sym: "\\{I_{PH}\\}", label: L("광전류 스윕 (선택)", "Light sweep (optional)"), help: L("V_G 목록이 비었을 때 사용 → P(1) vs I_PH", "Used when the V_G list is empty → P(1) vs I_PH"), unit: "pA", type: "list", min: 0, max: 1e6, placeholder: L("예: 0, 0.5, 1, 2 (비우면 없음)", "e.g. 0, 0.5, 1, 2 (leave empty for none)") },
    ],
  },
  coupled: {
    id: "coupled",
    title: "c.bench.coupled",
    desc: "c.bench.coupled.desc",
    defaults: {
      source: "ramp", v_min_V: 0, v_max_V: null, rate_V_per_s: null, n_cycles: 1, v_base_V: 0, v_amp_V: null, width_s: 200e-6, period_s: 1e-3, rise_s: null, fall_s: null,
      n_pulses: 10, delay_s: 0, R_s1_ohm: 100e3, R_s2_ohm: 100e3, R_c_ohm: 1e6, C_d_F: 2e-15, vg_V: null, vg2_V: null, iph2_pA: null,
    },
    fields: [
      { key: "source", label: L("공통 전원", "Common source"), help: L("삼각 램프 또는 펄스 열", "Triangular ramp or pulse train"), unit: "", type: "select", options: [{ value: "ramp", label: L("램프", "ramp") }, { value: "pulse", label: L("펄스", "pulse") }] },
      ...[f.vMin, f.vMax, f.rate, f.nCycles].map((x) => ({ ...x, when: (bp: Record<string, BenchValue>) => bp.source !== "pulse" })),
      ...[f.vBase, f.vAmp, f.width, f.period, f.rise, f.fall, f.nPulses, f.delay].map((x) => ({ ...x, when: (bp: Record<string, BenchValue>) => bp.source === "pulse" })),
      { ...f.Rs, key: "R_s1_ohm", sym: "R_{s1}", label: L("직렬 저항 (셀 1)", "Series resistor (cell 1)") },
      { ...f.Rs, key: "R_s2_ohm", sym: "R_{s2}", label: L("직렬 저항 (셀 2)", "Series resistor (cell 2)") },
      { key: "R_c_ohm", sym: "R_c", label: L("결합 저항", "Coupling resistor"), help: L("두 드레인 사이 저항", "Resistor between the drains"), unit: "kΩ", scale: 1e-3, min: 1e-6, max: 1e12, slider: "log" },
      f.Cd,
      { ...f.vg, key: "vg_V", sym: "V_{G,1}", label: L("셀 1 게이트", "Cell 1 gate") },
      { ...f.vg, key: "vg2_V", sym: "V_{G,2}", label: L("셀 2 게이트", "Cell 2 gate"), help: L("자동: 셀 1과 동일", "auto: same as cell 1") },
      { key: "iph2_pA", sym: "I_{PH,2}", label: L("셀 2 광전류", "Cell 2 photocurrent"), help: L("자동: 소자 광 설정", "auto: device light"), unit: "pA", min: 0, max: 1e6, step: 0.01, auto: true },
    ],
  },
};

export const BENCH_ORDER: BenchId[] = ["load_line", "pulse", "pbit", "coupled"];

export const DEFAULT_SOLVER: SolverBlock = {
  method: "BE",
  dt_min_s: null,
  dt_max_s: null,
  reltol: 1e-3,
  max_steps: 1_000_000,
  tau_frac: 0.05,
  max_events_per_step: 200,
  noise_dt_min_s: 2e-9,
  gauss_threshold: 100,
};

export const DEFAULT_CIRCUIT_STOCH: CircuitStochBlock = {
  seed: 2026092920,
  n_runs: 20,
  carrier_noise: true,
  local_state: { mode: "none", action: "gidl", sigma: 0.15339035678526572, tau_s: 5, sigma_E_V: 0.00043696121025623943, tau_E_s: 1.6205564347468981, acquisition_trend: false },
};
