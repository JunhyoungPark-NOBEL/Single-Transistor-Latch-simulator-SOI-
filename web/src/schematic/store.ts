// Schematic editor state (Zustand): the document with undo/redo, selection, active tool, viewport,
// waveform traces, time cursor and saved circuits. The document and saved circuits persist in
// localStorage["stl-websim:schematic"] (validated on load, ./persist.ts).
import { create } from "zustand";
import { builtinDevices, deviceName, stochOf, type LibDevice } from "../devices/library";
import { useDeviceLib } from "../devices/store";
import { useStore } from "../state/store";
import { clone } from "../utils/object";
import type { Estimate } from "./feasibility";
import { type ElKind, type Rot, type SchematicDoc, type StlRef } from "./model";
import { defaultDoc, parseStored, SCHEMATIC_KEY, type SavedCircuit } from "./persist";
import { buildTemplate, TEMPLATE_TEXT, type TemplateId } from "./templates";
import { translate } from "../i18n";
import type { StrKey } from "../i18n/strings";

export type Tool =
  | { kind: "select" }
  | { kind: "place"; el: ElKind; rot: Rot; mirror: boolean }
  | { kind: "wire" }
  | { kind: "probe" };

export interface View {
  x: number;
  y: number;
  k: number;
}

export interface SchState {
  doc: SchematicDoc;
  past: SchematicDoc[];
  future: SchematicDoc[];
  selection: string[];
  tool: Tool;
  /** Library id used for the next STL placement ("current" = the Device tab's settings). */
  stlChoice: string;
  view: View;
  fitNonce: number;
  traces: string[];
  cursorT: number | null;
  annotate: boolean;
  annotateSource: "run0" | "mean";
  logI: boolean;
  showRuns: boolean;
  showBand: boolean;
  highlight: { ids: string[]; nonce: number } | null;
  saved: SavedCircuit[];
  confirm: Estimate | null;
  toast: { msg: string; nonce: number; kind?: "ok" | "err" } | null;

  commit: (fn: (d: SchematicDoc) => SchematicDoc) => void;
  /** Change without an undo entry (e.g. typing in a field: the first keystroke is recorded by `commit`). */
  patchDoc: (fn: (d: SchematicDoc) => SchematicDoc) => void;
  replaceDoc: (d: SchematicDoc) => void;
  undo: () => void;
  redo: () => void;
  select: (ids: string[]) => void;
  setTool: (t: Tool) => void;
  setView: (v: View) => void;
  requestFit: () => void;
  setTraces: (t: string[]) => void;
  addTrace: (k: string) => void;
  removeTrace: (k: string) => void;
  setCursor: (t: number | null) => void;
  flash: (ids: string[]) => void;
  notify: (msg: string, kind?: "ok" | "err") => void;
  set: (p: Partial<SchState>) => void;
  saveCurrent: (name: string) => void;
  deleteSaved: (name: string) => void;
}

const HISTORY = 100;

/** Library entries available for placement: built-ins, the user's devices and the unsaved current device. */
export function libraryEntries(): LibDevice[] {
  const app = useStore.getState();
  const cur: LibDevice = {
    id: "current",
    name: translate(app.lang, "schematic.lib.current"),
    technology: "FDSOI",
    geometry: builtinDevices(app.meta)[0]?.geometry ?? { Lg_nm: 500, W_nm: 200, Tsi_nm: 50, EOT_nm: 14.1 },
    calibration_label: app.meta.presets[app.preset]?.label ?? { ko: "", en: "" },
    device: clone(app.params.device),
    stochastic: stochOf(app.params.stochastic),
    created: "",
    notes: "",
  };
  return [...builtinDevices(app.meta), ...useDeviceLib.getState().devices, cur];
}

export function stlRefFor(libId: string): StlRef {
  const lang = useStore.getState().lang;
  const all = libraryEntries();
  const d = all.find((x) => x.id === libId) ?? all[0];
  // canvas-friendly name: built-ins "FDSOI · Reference", user devices their own name
  const name = d.builtin ? `${d.technology} · ${translate(lang, `preset.${d.device.preset}` as StrKey)}` : d.id === "current" ? `${d.technology} · ${translate(lang, "schematic.lib.current")}` : deviceName(d, lang);
  // acquisition_trend belongs to the device-record lookup engine and is not used by the circuit simulator
  return { libId: d.id, name, device: clone(d.device), local_state: { ...clone(d.stochastic.local_state), acquisition_trend: false } };
}

export function templateDoc(id: TemplateId): SchematicDoc {
  const lang = useStore.getState().lang;
  return buildTemplate(id, stlRefFor("builtin:paper"), translate(lang, TEMPLATE_TEXT[id].title));
}

function initial(): { doc: SchematicDoc; saved: SavedCircuit[] } {
  let stored: ReturnType<typeof parseStored> = { doc: null, saved: [] };
  try {
    stored = parseStored(typeof localStorage !== "undefined" ? localStorage.getItem(SCHEMATIC_KEY) : null);
  } catch {
    /* ignore */
  }
  let doc = stored.doc;
  if (!doc) {
    try {
      doc = templateDoc("load_line");
    } catch {
      doc = defaultDoc();
    }
  }
  return { doc, saved: stored.saved };
}

const init = initial();

export const useSch = create<SchState>((set) => ({
  doc: init.doc,
  past: [],
  future: [],
  selection: [],
  tool: { kind: "select" },
  stlChoice: "builtin:paper",
  view: { x: 40, y: 40, k: 1 },
  fitNonce: 1,
  traces: [],
  cursorT: null,
  annotate: true,
  annotateSource: "run0",
  logI: true,
  showRuns: true,
  showBand: true,
  highlight: null,
  saved: init.saved,
  confirm: null,
  toast: null,

  commit: (fn) =>
    set((s) => {
      const next = fn(s.doc);
      if (next === s.doc) return {};
      return { doc: next, past: [...s.past.slice(-HISTORY + 1), s.doc], future: [] };
    }),
  patchDoc: (fn) => set((s) => ({ doc: fn(s.doc) })),
  replaceDoc: (d) => set((s) => ({ doc: d, past: [...s.past.slice(-HISTORY + 1), s.doc], future: [], selection: [], fitNonce: s.fitNonce + 1, cursorT: null })),
  undo: () =>
    set((s) => {
      const prev = s.past[s.past.length - 1];
      if (!prev) return {};
      return { doc: prev, past: s.past.slice(0, -1), future: [s.doc, ...s.future].slice(0, HISTORY), selection: s.selection.filter((id) => prev.elements.some((e) => e.id === id) || prev.wires.some((w) => w.id === id)) };
    }),
  redo: () =>
    set((s) => {
      const next = s.future[0];
      if (!next) return {};
      return { doc: next, past: [...s.past, s.doc].slice(-HISTORY), future: s.future.slice(1) };
    }),
  select: (ids) => set({ selection: ids }),
  setTool: (tool) => set({ tool }),
  setView: (view) => set({ view }),
  requestFit: () => set((s) => ({ fitNonce: s.fitNonce + 1 })),
  setTraces: (traces) => set({ traces }),
  addTrace: (k) => set((s) => (s.traces.includes(k) ? {} : { traces: [...s.traces, k] })),
  removeTrace: (k) => set((s) => ({ traces: s.traces.filter((x) => x !== k) })),
  setCursor: (cursorT) => set({ cursorT }),
  flash: (ids) => set((s) => ({ highlight: { ids, nonce: (s.highlight?.nonce ?? 0) + 1 }, selection: ids.length ? ids : s.selection })),
  notify: (msg, kind = "ok") => set((s) => ({ toast: { msg, kind, nonce: (s.toast?.nonce ?? 0) + 1 } })),
  set: (p) => set(p),
  saveCurrent: (name) =>
    set((s) => {
      const n = name.trim() || s.doc.name || "circuit";
      const doc = { ...clone(s.doc), name: n };
      const saved = [...s.saved.filter((x) => x.name !== n), { name: n, saved: new Date().toISOString(), doc }];
      return { saved, doc: { ...s.doc, name: n } };
    }),
  deleteSaved: (name) => set((s) => ({ saved: s.saved.filter((x) => x.name !== name) })),
}));

// persistence (debounced)
if (typeof window !== "undefined") {
  let timer: ReturnType<typeof setTimeout> | undefined;
  useSch.subscribe((s, prev) => {
    if (s.doc === prev.doc && s.saved === prev.saved) return;
    clearTimeout(timer);
    timer = setTimeout(() => {
      try {
        const st = useSch.getState();
        localStorage.setItem(SCHEMATIC_KEY, JSON.stringify({ v: 1, doc: st.doc, saved: st.saved }));
      } catch {
        /* storage full / unavailable */
      }
    }, 300);
  });
}
