// Example circuits reproducing the four quick benches (defaults of server/compute/circuit/benches.py):
// load-line ramp, pulse train, p-bit (drain pulses, source resistor, comparator), resistively coupled pair.
// The fold-dependent "auto" levels of the benches are written out for the FDSOI reference device
// (V_LU ≈ 3.70 V): pulse amplitude V_LU + 0.10 V = 3.8 V, p-bit pulse high ≈ V_LU − 15 mV = 3.69 V.
// Plus the current-driven relaxation oscillator (integrate-and-fire neuron, docs/CIRCUIT_SIMULATOR.md §13):
// I_in = 1 nA into V(out), C_par = 1 pF, STL drain at V(out), source grounded, gate at −2 V DC.
import type { StrKey } from "../i18n/strings";
import { clone } from "../utils/object";
import { newId, type ElKind, type Rot, type SchematicDoc, type SElement, type StlRef, type Wave, type Wire } from "./model";
import { DEFAULT_LOCAL } from "./persist";
import { defaultDoc } from "./persist";

export type TemplateId = "load_line" | "pulse" | "pbit" | "coupled" | "oscillator";
export const TEMPLATE_ORDER: TemplateId[] = ["load_line", "pulse", "pbit", "coupled", "oscillator"];
export const TEMPLATE_TEXT: Record<TemplateId, { title: StrKey; desc: StrKey }> = {
  load_line: { title: "schematic.tpl.load_line", desc: "schematic.tpl.load_line.desc" },
  pulse: { title: "schematic.tpl.pulse", desc: "schematic.tpl.pulse.desc" },
  pbit: { title: "schematic.tpl.pbit", desc: "schematic.tpl.pbit.desc" },
  coupled: { title: "schematic.tpl.coupled", desc: "schematic.tpl.coupled.desc" },
  oscillator: { title: "schematic.tpl.oscillator", desc: "schematic.tpl.oscillator.desc" },
};

/** Current-driven oscillator defaults (measured with the circuit simulator, FDSOI reference device at V_G = −2 V):
 *  quasi-static period ≈ C_par (V_LU − V_LD) / I_in = 1.109 ms, simulated 1.163 ms (+4.8 % fold lags, 860 Hz); first
 *  latch-up at C_par V_LU / I_in ≈ 3.74 ms, ten latch-ups in 15 ms; ≈ 10 500 steps, ≈ 1 s deterministic; ≈ 11 000 steps
 *  and ≈ 1 s per stochastic run (ISI jitter ≈ 1–2 %). docs/CIRCUIT_SIMULATOR.md §13. */
export const OSC_DEFAULTS = { i_in_A: 1e-9, c_par_F: 1e-12, t_stop_s: 15e-3, dt_max_s: 5e-6 };

/** p-bit defaults (measured, reference device V_G = −2 V, carrier noise only): P(fire) = 0.075 / 0.34 / 0.50 / 0.75 at
 *  3.66 / 3.68 / 3.69 / 3.70 V (200 µs flat top, 20 µs edges, 1 ms period); deterministic: never below 3.70 V, every
 *  pulse from 3.72 V. Latched: V(s) = R_S I_D ≈ 0.44 V; unlatched: µV → V_ref = 0.1 V. ≈ 1.5 s per stochastic run. */
export const PBIT_DEFAULTS = { v_pulse_V: 3.69, width_s: 200e-6, period_s: 1e-3, edge_s: 20e-6, n_pulses: 20, R_S_ohm: 100e3, v_ref_V: 0.1 };

const el = (kind: ElKind, name: string, x: number, y: number, extra: Partial<SElement> = {}, rot: Rot = 0): SElement => ({ id: newId(), kind, name, x, y, rot, ...extra });
const gnd = (x: number, y: number) => el("GND", "", x, y);
const label = (text: string, x: number, y: number) => el("LABEL", "", x, y, { label: text });
/** Orthogonal polyline → wire segments. */
function path(...pts: [number, number][]): Wire[] {
  const out: Wire[] = [];
  for (let i = 1; i < pts.length; i++) out.push({ id: newId("w"), x1: pts[i - 1][0], y1: pts[i - 1][1], x2: pts[i][0], y2: pts[i][1] });
  return out;
}
const dc = (value: number): Wave => ({ kind: "dc", value });
const pulse = (v2: number, tr: number, pw: number, per: number, n: number): Wave => ({ kind: "pulse", v1: 0, v2, td: 0, tr, tf: tr, pw, per, ncycles: n });
const triangle = (vmax: number, rate: number): Wave => {
  const T = vmax / rate;
  return { kind: "pwl", t: [0, T, 2 * T], v: [0, vmax, 0] };
};

/** One STL cell with series resistor from `src`, drain capacitor and a gate source. */
function singleCell(src: Wave, rName: string, rValue: number, srcName: string, srcLabel: string, stl: StlRef, vg: number): Pick<SchematicDoc, "elements" | "wires"> {
  return {
    elements: [
      el("V", srcName, 100, 200, { wave: src }),
      gnd(100, 240),
      el("R", rName, 200, 120, { value: rValue }, 3),
      el("STL", "X1", 400, 200, { stl: clone(stl), light: null }),
      gnd(400, 240),
      el("C", "Cd", 640, 200, { value: 2e-15 }),
      gnd(640, 240),
      el("V", "VG1", 300, 280, { wave: dc(vg) }),
      gnd(300, 320),
      label(srcLabel, 100, 120),
      label("d", 520, 120),
      label("g", 330, 200),
    ],
    wires: [...path([100, 160], [100, 120], [160, 120]), ...path([240, 120], [640, 120], [640, 160]), ...path([400, 120], [400, 160]), ...path([360, 200], [300, 200], [300, 240])],
  };
}

export function buildTemplate(id: TemplateId, stl: StlRef, name: string): SchematicDoc {
  const d = defaultDoc(name);
  const vg = stl.device.vg ?? -2;
  switch (id) {
    case "load_line": {
      // triangle 0 → 4 V → 0 at 0.4 V/s through R_s = 1 kΩ, C_d = 2 fF (bench defaults for the reference device)
      Object.assign(d, singleCell(triangle(4, 0.4), "Rs", 1e3, "Vsrc", "src", stl, vg));
      d.tran = { ...d.tran, t_stop_s: 20, dt_max_s: null };
      break;
    }
    case "pulse": {
      // 10 pulses: V_LU + 0.1 V, 200 µs flat top, 1 ms period, 10 µs edges
      Object.assign(d, singleCell(pulse(3.8, 10e-6, 200e-6, 1e-3, 10), "Rs", 1e3, "Vsrc", "src", stl, vg));
      d.tran = { ...d.tran, t_stop_s: 10e-3, dt_max_s: null };
      break;
    }
    case "pbit": {
      // regular pulses on the drain just below the latch-up fold, source → R_S → ground, comparator on the source
      // node: a latched pulse drives V(s) = R_S·I_D ≈ 0.44 V above V_ref, an unlatched one leaves µV → random firing
      const P = PBIT_DEFAULTS;
      d.elements = [
        el("V", "Vpulse", 100, 200, { wave: pulse(P.v_pulse_V, P.edge_s, P.width_s, P.period_s, P.n_pulses) }),
        gnd(100, 240),
        el("STL", "X1", 400, 200, { stl: clone(stl), light: null }),
        el("V", "VG1", 300, 280, { wave: dc(vg) }),
        gnd(300, 320),
        el("R", "RS", 400, 320, { value: P.R_S_ohm }),
        gnd(400, 360),
        el("CMP", "CMP1", 560, 260, { cmp: { v_ref: P.v_ref_V, v_high: 1, v_low: 0, hysteresis: 0 } }),
        label("d", 250, 120),
        label("s", 460, 260),
        label("q", 640, 260),
      ];
      d.wires = [
        ...path([100, 160], [100, 120], [400, 120], [400, 160]),
        ...path([360, 200], [300, 200], [300, 240]),
        ...path([400, 240], [400, 280]),
        ...path([400, 260], [520, 260]),
        ...path([600, 260], [660, 260]),
      ];
      d.tran = { ...d.tran, t_stop_s: P.n_pulses * P.period_s, dt_max_s: null };
      d.stoch.local_state = { ...clone(DEFAULT_LOCAL), mode: "none" };
      break;
    }
    case "coupled": {
      d.elements = [
        el("V", "Vsrc", 80, 200, { wave: triangle(4, 0.4) }),
        gnd(80, 240),
        el("R", "Rs1", 280, 140, { value: 100e3 }),
        el("STL", "X1", 280, 280, { stl: clone(stl), light: null }),
        gnd(280, 320),
        el("C", "Cd1", 460, 260, { value: 2e-15 }),
        gnd(460, 300),
        el("R", "Rc", 540, 220, { value: 1e6 }, 3),
        el("R", "Rs2", 680, 140, { value: 100e3 }),
        el("STL", "X2", 680, 280, { stl: clone(stl), light: null }),
        gnd(680, 320),
        el("C", "Cd2", 860, 260, { value: 2e-15 }),
        gnd(860, 300),
        el("V", "VG1", 180, 360, { wave: dc(vg) }),
        gnd(180, 400),
        label("src", 80, 100),
        label("d1", 370, 220),
        label("d2", 770, 220),
        label("g", 210, 280),
        label("g", 610, 280),
      ];
      d.wires = [
        ...path([80, 160], [80, 100], [680, 100]),
        ...path([280, 180], [280, 240]),
        ...path([280, 220], [460, 220], [500, 220]),
        ...path([580, 220], [680, 220], [860, 220]),
        ...path([680, 180], [680, 240]),
        ...path([240, 280], [180, 280], [180, 320]),
        ...path([640, 280], [610, 280]),
      ];
      d.tran = { ...d.tran, t_stop_s: 20, dt_max_s: null };
      break;
    }
    case "oscillator": {
      // integrate-and-fire: I_in charges C_par until the STL latches at V_LU, the LRS current discharges C_par to
      // V_LD, the body unlatches, repeat (sawtooth V(out), current spikes I(X1.d)). The source is drawn from ground
      // (rot 180°: the arrow points up into the out rail); the out net is labelled so V(out) is a default trace.
      d.elements = [
        el("C", "Cpar", 240, 160, { value: OSC_DEFAULTS.c_par_F }),
        gnd(240, 200),
        el("STL", "X1", 400, 200, { stl: clone(stl), light: null }),
        gnd(400, 240),
        el("I", "Iin", 560, 160, { wave: dc(OSC_DEFAULTS.i_in_A) }, 2),
        gnd(560, 200),
        el("V", "VG1", 300, 280, { wave: dc(vg) }),
        gnd(300, 320),
        label("out", 460, 120),
      ];
      d.wires = [...path([240, 120], [560, 120]), ...path([400, 120], [400, 160]), ...path([360, 200], [300, 200], [300, 240])];
      d.tran = { ...d.tran, t_stop_s: OSC_DEFAULTS.t_stop_s, dt_max_s: OSC_DEFAULTS.dt_max_s };
      break;
    }
  }
  // the p-bit's randomness is the carrier noise: local states off (with the library's slowly evolving local states
  // the firing probability would differ from run to run); the other examples use each device's own setting
  d.stoch.local_source = id === "pbit" ? "override" : "device";
  return d;
}
