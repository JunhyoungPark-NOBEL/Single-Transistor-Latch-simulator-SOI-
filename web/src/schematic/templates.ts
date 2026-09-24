// Example circuits reproducing the four quick benches (defaults of server/compute/circuit/benches.py):
// load-line ramp, pulse train, p-bit with load resistor, resistively coupled pair. The fold-dependent
// "auto" levels of the benches are written out for the FDSOI reference device (V_LU ≈ 3.70 V):
// pulse amplitude V_LU + 0.10 V = 3.8 V, p-bit clock high V_LU − 0.02 V = 3.68 V.
import type { StrKey } from "../i18n/strings";
import { clone } from "../utils/object";
import { newId, type ElKind, type Rot, type SchematicDoc, type SElement, type StlRef, type Wave, type Wire } from "./model";
import { defaultDoc } from "./persist";

export type TemplateId = "load_line" | "pulse" | "pbit" | "coupled";
export const TEMPLATE_ORDER: TemplateId[] = ["load_line", "pulse", "pbit", "coupled"];
export const TEMPLATE_TEXT: Record<TemplateId, { title: StrKey; desc: StrKey }> = {
  load_line: { title: "schematic.tpl.load_line", desc: "schematic.tpl.load_line.desc" },
  pulse: { title: "schematic.tpl.pulse", desc: "schematic.tpl.pulse.desc" },
  pbit: { title: "schematic.tpl.pbit", desc: "schematic.tpl.pbit.desc" },
  coupled: { title: "schematic.tpl.coupled", desc: "schematic.tpl.coupled.desc" },
};

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
      // clocked supply V_LU − 0.02 V through R_L = 100 kΩ; bit = [v_D below threshold] at the end of each clock
      Object.assign(d, singleCell(pulse(3.68, 20e-6, 200e-6, 1e-3, 50), "RL", 100e3, "Vclk", "clk", stl, vg));
      d.tran = { ...d.tran, t_stop_s: 50e-3, dt_max_s: null };
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
  }
  d.stoch.local_source = "device";
  return d;
}
