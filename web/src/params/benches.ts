// Circuit test benches (docs/WEB_CONTRACT.md §4, engine/docs/CIRCUIT_ELEMENT_DESIGN.md "Test benches").
// Bench-parameter keys follow server/compute/circuit/benches.py; unknown keys are ignored server-side
// and missing ones take the server defaults, so the UI only needs a sensible subset.
import type { L10n } from "../content/physics/types";
import type { BenchId, CircuitStochBlock, SolverBlock } from "../api/types";
import type { StrKey } from "../i18n/strings";

export interface BenchField {
  key: string;
  sym: string;
  label: L10n;
  help: L10n;
  unit: string;
  scale?: number;
  min?: number;
  max?: number;
  step?: number;
  slider?: boolean | "log";
  int?: boolean;
  options?: { value: string; label: L10n }[];
}
export interface BenchDef {
  id: BenchId;
  title: StrKey;
  desc: StrKey;
  defaults: Record<string, number | string | boolean>;
  fields: BenchField[];
}

const L = (ko: string, en: string): L10n => ({ ko, en });

const R_SERIES: BenchField = {
  key: "r_series_ohm", sym: "R_S", label: L("직렬 저항", "Series resistor"), help: L("전원과 드레인 사이 부하 저항", "Load resistor between source and drain"),
  unit: "kΩ", scale: 1e-3, min: 0.001, max: 1e5, slider: "log",
};
const C_NODE: BenchField = {
  key: "c_node_F", sym: "C_D", label: L("드레인 노드 용량", "Drain-node capacitance"), help: L("드레인 노드의 기생/부하 용량", "Parasitic/load capacitance at the drain node"),
  unit: "fF", scale: 1e15, min: 0, max: 1e6, slider: "log",
};

export const BENCHES: Record<BenchId, BenchDef> = {
  load_line: {
    id: "load_line",
    title: "c.bench.load_line",
    desc: "c.bench.load_line.desc",
    defaults: { v_max_V: 4.5, rate_V_per_s: 0.4, r_series_ohm: 1e4, c_node_F: 1e-13 },
    fields: [
      { key: "v_max_V", sym: "V_{\\mathrm{src,max}}", label: L("전원 최대 전압", "Source peak voltage"), help: L("삼각 램프 0 → V_max → 0", "Triangular ramp 0 → V_max → 0"), unit: "V", min: 0.5, max: 8, step: 0.1, slider: true },
      { key: "rate_V_per_s", sym: "\\dot V", label: L("램프 속도", "Ramp rate"), help: L("전원 전압 변화율", "Source slew rate"), unit: "V/s", min: 1e-3, max: 1e6, slider: "log" },
      R_SERIES,
      C_NODE,
    ],
  },
  pulse: {
    id: "pulse",
    title: "c.bench.pulse",
    desc: "c.bench.pulse.desc",
    defaults: { amplitude_V: 4.2, base_V: 0, width_s: 1e-3, period_s: 5e-3, n_pulses: 5, rise_s: 1e-6, r_series_ohm: 1e4, c_node_F: 1e-13 },
    fields: [
      { key: "amplitude_V", sym: "V_{\\mathrm{pulse}}", label: L("펄스 진폭", "Pulse amplitude"), help: L("펄스 high 전압", "Pulse high level"), unit: "V", min: 0, max: 8, step: 0.05, slider: true },
      { key: "base_V", sym: "V_{\\mathrm{base}}", label: L("기저 전압", "Base level"), help: L("펄스 사이 전압", "Voltage between pulses"), unit: "V", min: 0, max: 8, step: 0.05 },
      { key: "width_s", sym: "t_w", label: L("펄스 폭", "Pulse width"), help: L("high 구간 길이", "High-level duration"), unit: "ms", scale: 1e3, min: 1e-6, max: 1e4, slider: "log" },
      { key: "period_s", sym: "T", label: L("펄스 주기", "Pulse period"), help: L("펄스 간격 (주기)", "Pulse repetition period"), unit: "ms", scale: 1e3, min: 1e-6, max: 1e5, slider: "log" },
      { key: "n_pulses", sym: "N_p", label: L("펄스 수", "Number of pulses"), help: L("펄스 열 길이", "Pulse train length"), unit: "", min: 1, max: 1000, step: 1, int: true },
      { key: "rise_s", sym: "t_r", label: L("상승 시간", "Rise time"), help: L("에지 상승/하강 시간", "Edge rise/fall time"), unit: "µs", scale: 1e6, min: 1e-6, max: 1e6, slider: "log" },
      R_SERIES,
      C_NODE,
    ],
  },
  pbit: {
    id: "pbit",
    title: "c.bench.pbit",
    desc: "c.bench.pbit.desc",
    defaults: { v_bias_V: 3.3, r_load_ohm: 1e5, v_threshold_V: 3.0, duration_s: 0.2, c_node_F: 1e-13 },
    fields: [
      { key: "v_bias_V", sym: "V_{\\mathrm{bias}}", label: L("바이어스 전압", "Bias voltage"), help: L("부하 저항 위 DC 전원 (래치 창 안)", "DC supply above the load resistor (inside the latch window)"), unit: "V", min: 0, max: 8, step: 0.01, slider: true },
      { key: "r_load_ohm", sym: "R_L", label: L("부하 저항", "Load resistor"), help: L("부하선 기울기", "Sets the load-line slope"), unit: "kΩ", scale: 1e-3, min: 0.001, max: 1e6, slider: "log" },
      { key: "v_threshold_V", sym: "V_{\\mathrm{th}}", label: L("비교기 문턱", "Comparator threshold"), help: L("드레인 전압 → 비트 판정 기준", "Drain-voltage threshold for the output bit"), unit: "V", min: 0, max: 8, step: 0.01 },
      { key: "duration_s", sym: "t_{\\mathrm{sim}}", label: L("시뮬레이션 시간", "Duration"), help: L("과도해석 총 시간", "Total transient time"), unit: "s", min: 1e-6, max: 1e3, slider: "log" },
      C_NODE,
    ],
  },
  coupled: {
    id: "coupled",
    title: "c.bench.coupled",
    desc: "c.bench.coupled.desc",
    defaults: { v_max_V: 4.5, rate_V_per_s: 0.4, r_series_ohm: 1e4, r_couple_ohm: 1e6, vg2_V: -2.0, c_node_F: 1e-13 },
    fields: [
      { key: "v_max_V", sym: "V_{\\mathrm{src,max}}", label: L("전원 최대 전압", "Source peak voltage"), help: L("공통 삼각 램프", "Shared triangular ramp"), unit: "V", min: 0.5, max: 8, step: 0.1, slider: true },
      { key: "rate_V_per_s", sym: "\\dot V", label: L("램프 속도", "Ramp rate"), help: L("전원 전압 변화율", "Source slew rate"), unit: "V/s", min: 1e-3, max: 1e6, slider: "log" },
      R_SERIES,
      { key: "r_couple_ohm", sym: "R_c", label: L("결합 저항", "Coupling resistor"), help: L("두 드레인 사이 저항", "Resistor between the two drains"), unit: "kΩ", scale: 1e-3, min: 0.001, max: 1e7, slider: "log" },
      { key: "vg2_V", sym: "V_{G,2}", label: L("소자 2 게이트", "Device 2 gate"), help: L("두 번째 STL의 V_G", "V_G of the second STL"), unit: "V", min: -4.5, max: 0, step: 0.01, slider: true },
      C_NODE,
    ],
  },
};

export const BENCH_ORDER: BenchId[] = ["load_line", "pulse", "pbit", "coupled"];

export const DEFAULT_SOLVER: SolverBlock = { method: "BE", dt_min_s: 1e-12, dt_max_s: 1e-3, reltol: 1e-4, max_steps: 200000 };

export const DEFAULT_CIRCUIT_STOCH: CircuitStochBlock = {
  seed: 1,
  n_runs: 20,
  carrier_noise: true,
  local_state: { mode: "frozen", action: "gidl", sigma: 0.15339035678526572, tau_s: 5, sigma_E_V: 0.00043696121025623943, tau_E_s: 1.6205564347468981, acquisition_trend: false },
};
