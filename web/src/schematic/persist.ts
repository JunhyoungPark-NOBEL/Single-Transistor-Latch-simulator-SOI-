// Validation of stored / imported schematic documents (never throws; junk falls back to defaults).
import type { LocalStateBlock } from "../api/types";
import { sanitizeDevice, sanitizeLocal } from "../devices/library";
import { useDeviceLib, validationBase } from "../devices/store";
import { DEFAULT_CIRCUIT_STOCH } from "../params/benches";
import { clone } from "../utils/object";
import { DEFAULT_CMP, DEFAULT_MOS, DEFAULT_DIODE, DEFAULT_BJT, DOC_VERSION, GRID, newId, type ElKind, type Rot, type SchematicDoc, type SElement, type StlRef, type Wire } from "./model";
import { parseWave } from "./waves";

export const SCHEMATIC_KEY = "stl-websim:schematic";
export const CIRCUIT_FORMAT = "stl-circuit";
const KINDS: ElKind[] = ["R", "C", "V", "I", "MOS", "D", "BJT", "STL", "CMP", "GND", "LABEL"];
const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export const DEFAULT_LOCAL: LocalStateBlock = clone({ ...DEFAULT_CIRCUIT_STOCH.local_state, mode: "frozen" });

export function defaultDoc(name = ""): SchematicDoc {
  return {
    v: DOC_VERSION,
    name,
    elements: [],
    wires: [],
    tran: { t_stop_s: 10e-3, t_start_save_s: 0, dt_max_s: null, dt_min_s: null, method: "BE", reltol: 1e-3 },
    stoch: { n_runs: 20, seed: DEFAULT_CIRCUIT_STOCH.seed, carrier_noise: true, ld_carrier_noise: false, local_source: "device", local_state: clone(DEFAULT_LOCAL) },
    detect: { i_threshold_A: 1e-8, hysteresis: 10 },
    save_all: true,
  };
}

function parseStl(v: unknown): StlRef | undefined {
  if (!isObj(v) || !isObj(v.device)) return undefined;
  const base = validationBase();
  return {
    libId: typeof v.libId === "string" ? v.libId.slice(0, 80) : "current",
    name: typeof v.name === "string" ? v.name.slice(0, 80) : "STL",
    technology: typeof v.technology === "string" ? v.technology.slice(0, 40) : useDeviceLib.getState().devices.find((d) => d.id === v.libId)?.technology,
    device: sanitizeDevice(base.device, v.device),
    local_state: isObj(v.local_state) ? sanitizeLocal(base.stochastic.local_state, v.local_state) : undefined,
  };
}

function parseElement(v: unknown, legacy = false): SElement | null {
  if (!isObj(v)) return null;
  const kind = KINDS.find((k) => k === v.kind);
  if (!kind || !fin(v.x) || !fin(v.y)) return null;
  const snapG = (x: number) => Math.round(x / GRID) * GRID;
  const el: SElement = {
    id: typeof v.id === "string" && v.id ? v.id.slice(0, 60) : newId(),
    kind,
    name: typeof v.name === "string" ? v.name.slice(0, 40) : "",
    x: snapG(v.x),
    y: snapG(v.y),
    rot: ([0, 1, 2, 3] as Rot[]).find((r) => r === v.rot) ?? 0,
  };
  if (v.mirror === true) el.mirror = true;
  if ((kind === "R" || kind === "C") && fin(v.value)) el.value = v.value;
  if (kind === "R" && el.value === undefined) el.value = 1e3;
  if (kind === "C" && el.value === undefined) el.value = 1e-15;
  if (kind === "V" || kind === "I") el.wave = parseWave(v.wave) ?? { kind: "dc", value: kind === "V" ? 1 : 1e-9 };
  if (kind === "STL") {
    el.stl = parseStl(v.stl);
    if (legacy || v.stlTerminalMode === "legacy3") el.stlTerminalMode = "legacy3";
    el.light = v.light == null ? null : parseWave(v.light);
  }
  const numericModel = <T extends object>(raw: unknown, defaults: T): T => {
    const obj = isObj(raw) ? raw : {};
    return Object.fromEntries(Object.entries(defaults).map(([key, value]) => [key, typeof value === "number" ? (fin(obj[key]) ? obj[key] : value) : (value === "nmos" && obj[key] === "pmos") || (value === "npn" && obj[key] === "pnp") ? obj[key] : value])) as T;
  };
  if (kind === "MOS") el.mos = numericModel(v.mos, DEFAULT_MOS);
  if (kind === "D") el.diode = numericModel(v.diode, DEFAULT_DIODE);
  if (kind === "BJT") el.bjt = numericModel(v.bjt, DEFAULT_BJT);
  if (kind === "LABEL") el.label = typeof v.label === "string" ? v.label.slice(0, 40) : "";
  if (kind === "CMP") {
    const c = isObj(v.cmp) ? v.cmp : {};
    const num = (x: unknown, d: number, lo: number, hi: number) => (fin(x) && x >= lo && x <= hi ? x : d);
    el.cmp = {
      v_ref: num(c.v_ref, DEFAULT_CMP.v_ref, -1000, 1000),
      v_high: num(c.v_high, DEFAULT_CMP.v_high, -1000, 1000),
      v_low: num(c.v_low, DEFAULT_CMP.v_low, -1000, 1000),
      hysteresis: num(c.hysteresis, DEFAULT_CMP.hysteresis, 0, 10),
    };
  }
  return el;
}

function parseWire(v: unknown): Wire | null {
  if (!isObj(v) || !fin(v.x1) || !fin(v.y1) || !fin(v.x2) || !fin(v.y2)) return null;
  return { id: typeof v.id === "string" && v.id ? v.id.slice(0, 60) : newId("w"), x1: v.x1, y1: v.y1, x2: v.x2, y2: v.y2 };
}

/** Validate a document (stored or imported). Returns null when it is not a schematic at all. */
export function parseDoc(v: unknown): SchematicDoc | null {
  if (!isObj(v)) return null;
  const src = isObj(v.doc) ? v.doc : v; // accept {format, doc} wrappers
  if (!Array.isArray(src.elements) || !Array.isArray(src.wires)) return null;
  const d = defaultDoc();
  d.name = typeof src.name === "string" ? src.name.slice(0, 80) : "";
  const ids = new Set<string>();
  for (const x of src.elements.slice(0, 200)) {
    const el = parseElement(x, !fin(src.v) || src.v < 2);
    if (!el) continue;
    if (ids.has(el.id)) el.id = newId();
    ids.add(el.id);
    d.elements.push(el);
  }
  for (const x of src.wires.slice(0, 1000)) {
    const w = parseWire(x);
    if (!w) continue;
    if (ids.has(w.id)) w.id = newId("w");
    ids.add(w.id);
    d.wires.push(w);
  }
  if (isObj(src.tran)) {
    const t = src.tran;
    if (fin(t.t_stop_s) && t.t_stop_s > 0) d.tran.t_stop_s = t.t_stop_s;
    if (fin(t.t_start_save_s) && t.t_start_save_s >= 0) d.tran.t_start_save_s = t.t_start_save_s;
    d.tran.dt_max_s = fin(t.dt_max_s) && t.dt_max_s > 0 ? t.dt_max_s : null;
    d.tran.dt_min_s = fin(t.dt_min_s) && t.dt_min_s > 0 ? t.dt_min_s : null;
    if (t.method === "BE" || t.method === "TRAP") d.tran.method = t.method;
    if (fin(t.reltol) && t.reltol > 0) d.tran.reltol = t.reltol;
  }
  if (isObj(src.stoch)) {
    const s = src.stoch;
    if (fin(s.n_runs)) d.stoch.n_runs = Math.max(1, Math.min(200, Math.round(s.n_runs)));
    if (fin(s.seed)) d.stoch.seed = Math.max(0, Math.round(s.seed));
    if (typeof s.carrier_noise === "boolean") d.stoch.carrier_noise = s.carrier_noise;
    if (typeof s.ld_carrier_noise === "boolean") d.stoch.ld_carrier_noise = s.ld_carrier_noise;
    if (s.local_source === "device" || s.local_source === "override") d.stoch.local_source = s.local_source;
    if (isObj(s.local_state)) d.stoch.local_state = sanitizeLocal(DEFAULT_LOCAL, s.local_state);
  }
  if (isObj(src.detect)) {
    if (fin(src.detect.i_threshold_A) && src.detect.i_threshold_A > 0) d.detect.i_threshold_A = Math.min(1e-3, src.detect.i_threshold_A);
    if (fin(src.detect.hysteresis) && src.detect.hysteresis >= 1) d.detect.hysteresis = src.detect.hysteresis;
  }
  if (typeof src.save_all === "boolean") d.save_all = src.save_all;
  return d;
}

export interface SavedCircuit {
  name: string;
  saved: string;
  doc: SchematicDoc;
}
export interface StoredSchematic {
  doc: SchematicDoc | null;
  saved: SavedCircuit[];
}

export function parseStored(raw: string | null | undefined): StoredSchematic {
  const out: StoredSchematic = { doc: null, saved: [] };
  if (!raw) return out;
  try {
    const v = JSON.parse(raw) as unknown;
    if (!isObj(v)) return out;
    out.doc = parseDoc(v.doc);
    if (Array.isArray(v.saved))
      for (const s of v.saved.slice(0, 100)) {
        if (!isObj(s) || typeof s.name !== "string") continue;
        const doc = parseDoc(s.doc);
        if (doc) out.saved.push({ name: s.name.slice(0, 80), saved: typeof s.saved === "string" ? s.saved : "", doc });
      }
  } catch {
    /* junk → defaults */
  }
  return out;
}

export function exportDocJson(doc: SchematicDoc): string {
  return JSON.stringify({ format: CIRCUIT_FORMAT, v: DOC_VERSION, exported: new Date().toISOString(), doc }, null, 1);
}
