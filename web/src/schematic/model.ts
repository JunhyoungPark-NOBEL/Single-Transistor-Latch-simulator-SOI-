// Schematic document model (editor state that is saved/exported) and element geometry: pin positions,
// rotation/mirroring and bounding boxes in world units (1 grid = GRID world px at zoom 1).
import type { DeviceBlock, LocalStateBlock } from "../api/types";
import type { Wave } from "../api/circuitCustom";

export type { Wave };
export const GRID = 10;
export const DOC_VERSION = 1;

export type Rot = 0 | 1 | 2 | 3; // multiples of 90° clockwise
export type ElKind = "R" | "C" | "V" | "I" | "STL" | "CMP" | "GND" | "LABEL";
/** Kinds that become netlist elements (GND and LABEL are connectivity symbols only). */
export const CIRCUIT_KINDS: ElKind[] = ["R", "C", "V", "I", "STL", "CMP"];

/** Comparator parameters (§6.3 CMP element): output = v_high when V(in) > v_ref (± hysteresis/2), else v_low. */
export interface CmpParams {
  v_ref: number;
  v_high: number;
  v_low: number;
  hysteresis: number;
}
export const DEFAULT_CMP: CmpParams = { v_ref: 0.1, v_high: 1, v_low: 0, hysteresis: 0 };

export interface StlRef {
  /** Library id the snapshot was taken from ("current" = the Device tab's unsaved device). */
  libId: string;
  /** Library name at the time of the snapshot (shown on the canvas). */
  name: string;
  /** Snapshot of the device block (§1) — later library edits never change it silently. */
  device: DeviceBlock;
  /** Local-state settings stored with the library device (used in stochastic runs). */
  local_state?: LocalStateBlock;
}

export interface SElement {
  id: string;
  kind: ElKind;
  name: string;
  x: number;
  y: number;
  rot: Rot;
  mirror?: boolean;
  /** R in Ω, C in F. */
  value?: number;
  /** V source in volts, I source in amperes. */
  wave?: Wave;
  stl?: StlRef;
  /** STL light waveform in pA; null = the device block's light setting. */
  light?: Wave | null;
  /** Net label text (LABEL). */
  label?: string;
  /** Comparator parameters (CMP). */
  cmp?: CmpParams;
}

export interface Wire {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface TranSettings {
  t_stop_s: number;
  t_start_save_s: number;
  /** null = automatic (t_stop / 2000). */
  dt_max_s: number | null;
  /** null = automatic (max(1e-15, 1e-13 · t_stop)). */
  dt_min_s: number | null;
  method: "BE" | "TRAP";
  reltol: number;
}

export interface StochSettings {
  n_runs: number;
  seed: number;
  carrier_noise: boolean;
  ld_carrier_noise: boolean;
  /** "device": each STL uses the local-state settings stored with its library device; "override": one setting for all. */
  local_source: "device" | "override";
  local_state: LocalStateBlock;
}

export interface SchematicDoc {
  v: number;
  name: string;
  elements: SElement[];
  wires: Wire[];
  tran: TranSettings;
  stoch: StochSettings;
  detect: { i_threshold_A: number; hysteresis: number };
  /** true: store every node voltage and element current (probes: null); false: only the chosen traces. */
  save_all: boolean;
}

export interface Pt {
  x: number;
  y: number;
}

export type PinName = "p" | "n" | "d" | "g" | "s" | "o" | "i" | "q";
export interface PinDef {
  name: PinName;
  x: number;
  y: number;
}

/** Local (unrotated) pin positions. Two-terminal parts are vertical: first node on top. */
export const PINS: Record<ElKind, PinDef[]> = {
  R: [{ name: "p", x: 0, y: -40 }, { name: "n", x: 0, y: 40 }],
  C: [{ name: "p", x: 0, y: -40 }, { name: "n", x: 0, y: 40 }],
  V: [{ name: "p", x: 0, y: -40 }, { name: "n", x: 0, y: 40 }],
  I: [{ name: "p", x: 0, y: -40 }, { name: "n", x: 0, y: 40 }],
  STL: [{ name: "d", x: 0, y: -40 }, { name: "g", x: -40, y: 0 }, { name: "s", x: 0, y: 40 }],
  // comparator: input (compared with V_ref, referenced to ground) on the left, output on the right
  CMP: [{ name: "i", x: -40, y: 0 }, { name: "q", x: 40, y: 0 }],
  GND: [{ name: "o", x: 0, y: 0 }],
  LABEL: [{ name: "o", x: 0, y: 0 }],
};

/** Local bounding boxes [x0, y0, x1, y1] (for hit tests / fit). */
const BOX: Record<ElKind, [number, number, number, number]> = {
  R: [-12, -40, 12, 40],
  C: [-16, -40, 16, 40],
  V: [-20, -40, 20, 40],
  I: [-20, -40, 20, 40],
  STL: [-40, -40, 26, 40],
  CMP: [-40, -26, 40, 26],
  GND: [-14, -2, 14, 22],
  LABEL: [-4, -12, 64, 12],
};

export function rotatePt(x: number, y: number, rot: Rot, mirror = false): Pt {
  let px = mirror ? -x : x;
  let py = y;
  for (let k = 0; k < rot; k++) {
    const nx = -py;
    const ny = px;
    px = nx;
    py = ny;
  }
  return { x: px + 0, y: py + 0 };
}

export interface PinPos {
  el: SElement;
  pin: PinName;
  x: number;
  y: number;
}

export function pinPositions(el: SElement): PinPos[] {
  return PINS[el.kind].map((p) => {
    const r = rotatePt(p.x, p.y, el.rot, el.mirror);
    return { el, pin: p.name, x: el.x + r.x, y: el.y + r.y };
  });
}

export function labelWidth(text: string): number {
  return Math.max(28, 12 + text.length * 7.2);
}

export function elementBox(el: SElement): [number, number, number, number] {
  let b = BOX[el.kind];
  if (el.kind === "LABEL") b = [-4, -12, labelWidth(el.label ?? "") + 10, 12];
  const pts = [rotatePt(b[0], b[1], el.rot, el.mirror), rotatePt(b[2], b[3], el.rot, el.mirror)];
  return [
    el.x + Math.min(pts[0].x, pts[1].x),
    el.y + Math.min(pts[0].y, pts[1].y),
    el.x + Math.max(pts[0].x, pts[1].x),
    el.y + Math.max(pts[0].y, pts[1].y),
  ];
}

export const snap = (v: number, g = GRID) => Math.round(v / g) * g;

export const ptKey = (x: number, y: number) => `${Math.round(x)},${Math.round(y)}`;

/** True when (x, y) lies on the axis-aligned wire strictly between its end points. */
export function onWireInterior(w: Wire, x: number, y: number): boolean {
  if (w.x1 === w.x2) return x === w.x1 && y > Math.min(w.y1, w.y2) && y < Math.max(w.y1, w.y2);
  if (w.y1 === w.y2) return y === w.y1 && x > Math.min(w.x1, w.x2) && x < Math.max(w.x1, w.x2);
  return false;
}

/** Distance from a point to a segment (hit testing). */
export function distToSeg(px: number, py: number, w: Wire): number {
  const dx = w.x2 - w.x1;
  const dy = w.y2 - w.y1;
  const L2 = dx * dx + dy * dy;
  const t = L2 ? Math.max(0, Math.min(1, ((px - w.x1) * dx + (py - w.y1) * dy) / L2)) : 0;
  return Math.hypot(px - (w.x1 + t * dx), py - (w.y1 + t * dy));
}

/** Default name prefix per kind (SPICE convention: X for subcircuit-like devices). */
export const NAME_PREFIX: Record<ElKind, string> = { R: "R", C: "C", V: "V", I: "I", STL: "X", CMP: "CMP", GND: "GND", LABEL: "L" };

export function nextName(els: SElement[], kind: ElKind): string {
  const pre = NAME_PREFIX[kind];
  const used = new Set(els.map((e) => e.name.toUpperCase()));
  for (let k = 1; ; k++) if (!used.has(`${pre}${k}`.toUpperCase())) return `${pre}${k}`;
}

let idSeq = 0;
export function newId(prefix = "e"): string {
  idSeq = (idSeq + 1) % 1e6;
  return `${prefix}${Date.now().toString(36)}${idSeq.toString(36)}${Math.floor(Math.random() * 1296).toString(36)}`;
}
