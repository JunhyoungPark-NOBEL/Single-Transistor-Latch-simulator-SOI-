// Global UI + parameter + result state (Zustand). Job orchestration lives in ./runner.ts.
import { create } from "zustand";
import type { Health, Meta, Mode, PresetId } from "../api/types";
import type { TopicId } from "../content/physics/types";
import type { ParamRoot, Tab } from "../params/schema";
import { BUILTIN_META } from "./presets";
import { clone, getPath, mergeDefaults, setPath, type Path } from "../utils/object";
import { presetRoot, type VgRange } from "../utils/payload";
import { parsePersisted, PERSIST_VERSION, restoreParams, restoreRange, TABS, type Persisted } from "./persist";

export type { Lang, Theme } from "./persist";
import type { Lang, Theme } from "./persist";
export type BackendState = "checking" | "online" | "offline" | "mock";
export type EntryStatus = "idle" | "queued" | "running" | "done" | "error" | "cancelled";

export interface ResultEntry {
  status: EntryStatus;
  kind: string;
  data?: unknown;
  error?: string;
  progress: number;
  message: string;
  startedAt?: number;
  elapsed?: number;
  cached?: boolean;
  mock?: boolean;
  /** Canonical payload of the job that is running / last requested. */
  payloadKey?: string;
  /** Canonical payload that produced `data`. */
  dataKey?: string;
  token: number;
}

export interface PhysicsWin {
  open: boolean;
  topic: TopicId;
  history: TopicId[];
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface ActiveRun {
  keys: string[];
  label: string;
  startedAt: number;
  finishedAt?: number;
}

export interface DataSlot<T> {
  status: "idle" | "loading" | "done" | "error";
  data?: T;
  error?: string;
}

export interface State {
  tab: Tab;
  mode: Mode;
  lang: Lang;
  theme: Theme;
  sidebarOpen: boolean;
  autoRun: boolean;
  backend: BackendState;
  forcedMock: boolean;
  health: Health | null;
  meta: Meta;
  preset: PresetId;
  params: ParamRoot;
  cbVd: number | null;
  vgRange: VgRange;
  vgsRange: VgRange;
  results: Record<string, ResultEntry>;
  activeRun: ActiveRun | null;
  physics: PhysicsWin;
  physicsTarget: { topic: TopicId; nonce: number } | null;
  measured: DataSlot<import("../api/types").MeasuredData>;
  designMap: DataSlot<import("../api/types").DesignMapData>;

  setTab: (t: Tab) => void;
  setMode: (m: Mode) => void;
  setLang: (l: Lang) => void;
  setTheme: (t: Theme) => void;
  setSidebar: (open: boolean) => void;
  setAutoRun: (on: boolean) => void;
  setParam: (path: Path, value: unknown) => void;
  updateParams: (fn: (p: ParamRoot) => ParamRoot) => void;
  resetPaths: (paths: Path[]) => void;
  loadPreset: (id: PresetId) => void;
  setMeta: (m: Meta) => void;
  setCbVd: (v: number | null) => void;
  setVgRange: (r: Partial<VgRange>) => void;
  setVgsRange: (r: Partial<VgRange>) => void;
  patchResult: (key: string, patch: Partial<ResultEntry>) => void;
  openPhysics: (topic: TopicId, anchor?: DOMRect | null) => void;
  navigatePhysics: (topic: TopicId) => void;
  physicsBack: () => void;
  closePhysics: () => void;
  movePhysics: (pos: Partial<Pick<PhysicsWin, "x" | "y" | "w" | "h">>) => void;
  showInPhysicsTab: (topic: TopicId) => void;
}

// ---------------------------------------------------------------- persistence
// Same key since v1 (keeps users' settings); the stored object carries a schema version `v` and every
// field is validated on load (state/persist.ts) so junk or older data never crashes the app.
const STORAGE_KEY = "stl-websim:v1";
function loadPersisted(): Partial<Persisted> {
  try {
    return parsePersisted(typeof localStorage !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null);
  } catch {
    return {};
  }
}
export function savePersisted(s: State) {
  try {
    const p: Persisted = { v: PERSIST_VERSION, mode: s.mode, lang: s.lang, theme: s.theme, autoRun: s.autoRun, preset: s.preset, params: s.params, vgRange: s.vgRange, vgsRange: s.vgsRange, tab: s.tab };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
  } catch {
    /* storage unavailable (private mode, blocked) — ignore */
  }
}

function readHash(): { tab?: Tab; mode?: Mode } {
  try {
    const h = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    const tab = h.get("tab") as Tab | null;
    const mode = h.get("mode") as Mode | null;
    return {
      tab: tab && TABS.includes(tab) ? tab : undefined,
      mode: mode === "deterministic" || mode === "stochastic" ? mode : undefined,
    };
  } catch {
    return {};
  }
}

function systemTheme(): Theme {
  try {
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch {
    return "light";
  }
}

const P = loadPersisted();
const H = typeof window !== "undefined" ? readHash() : {};
const initialPreset: PresetId = P.preset ?? "paper";
const baseRoot = presetRoot(BUILTIN_META, initialPreset);
const initialParams = P.params ? restoreParams(baseRoot, P.params, P.v) : baseRoot;
const WIN_W = 560;
const WIN_H = 620;

export const DEFAULT_VG_RANGE: VgRange = { min: -4.5, max: -0.5, n: 41 };
export const DEFAULT_VGS_RANGE: VgRange = { min: -2.6, max: -0.9, n: 9 };

export const useStore = create<State>((set) => ({
  tab: H.tab ?? P.tab ?? "device",
  mode: H.mode ?? P.mode ?? "deterministic",
  lang: P.lang ?? "ko",
  theme: P.theme ?? systemTheme(),
  sidebarOpen: typeof window !== "undefined" ? !window.matchMedia?.("(max-width: 1100px)").matches : true, // same breakpoint as the CSS drawer
  autoRun: P.autoRun ?? false,
  backend: "checking",
  forcedMock: false,
  health: null,
  meta: BUILTIN_META,
  preset: initialPreset,
  params: initialParams,
  cbVd: null,
  vgRange: P.vgRange ? restoreRange(DEFAULT_VG_RANGE, P.vgRange) : DEFAULT_VG_RANGE,
  vgsRange: P.vgsRange ? restoreRange(DEFAULT_VGS_RANGE, P.vgsRange) : DEFAULT_VGS_RANGE,
  results: {},
  activeRun: null,
  physics: { open: false, topic: "overview", history: [], x: 80, y: 80, w: WIN_W, h: WIN_H },
  physicsTarget: null,
  measured: { status: "idle" },
  designMap: { status: "idle" },

  setTab: (tab) => set({ tab }),
  setMode: (mode) => set({ mode }),
  setLang: (lang) => set({ lang }),
  setTheme: (theme) => set({ theme }),
  setSidebar: (sidebarOpen) => set({ sidebarOpen }),
  setAutoRun: (autoRun) => set({ autoRun }),
  setParam: (path, value) => set((s) => ({ params: setPath(s.params, path, value) })),
  updateParams: (fn) => set((s) => ({ params: fn(s.params) })),
  resetPaths: (paths) =>
    set((s) => {
      const def = presetRoot(s.meta, s.preset);
      let params = s.params;
      for (const p of paths) params = setPath(params, p, clone(getPath(def, p)));
      return { params };
    }),
  loadPreset: (id) =>
    set((s) => {
      const def = presetRoot(s.meta, id);
      // keep circuit settings (bench, solver) — presets only define device/sweep/stochastic
      return { preset: id, params: { ...def, circuit: s.params.circuit }, cbVd: null };
    }),
  setMeta: (meta) =>
    set((s) => {
      // merge backend presets into the built-in ones (keeps the UI working if a key is missing)
      const merged: Meta = { ...BUILTIN_META, ...meta, presets: { ...BUILTIN_META.presets, ...(meta.presets ?? {}) } };
      const params = mergeDefaults(presetRoot(merged, s.preset), s.params);
      return { meta: merged, params };
    }),
  setCbVd: (cbVd) => set({ cbVd }),
  setVgRange: (r) => set((s) => ({ vgRange: { ...s.vgRange, ...r } })),
  setVgsRange: (r) => set((s) => ({ vgsRange: { ...s.vgsRange, ...r } })),
  patchResult: (key, patch) =>
    set((s) => {
      const prev = s.results[key] ?? { status: "idle", kind: patch.kind ?? key, progress: 0, message: "", token: 0 };
      return { results: { ...s.results, [key]: { ...prev, ...patch } } };
    }),
  openPhysics: (topic, anchor) =>
    set((s) => {
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const w = Math.min(s.physics.w || WIN_W, vw - 16);
      const h = Math.min(s.physics.h || WIN_H, Math.round(vh * 0.9));
      let x = s.physics.x;
      let y = s.physics.y;
      if (anchor) {
        x = anchor.right + 12;
        if (x + w > vw - 8) x = anchor.left - w - 12;
        if (x < 8) x = Math.max(8, Math.min(vw - w - 8, anchor.left));
        y = anchor.top - 12;
      }
      x = Math.max(8, Math.min(x, vw - w - 8));
      y = Math.max(8, Math.min(y, vh - h - 8));
      const history = s.physics.open && s.physics.topic !== topic ? [...s.physics.history, s.physics.topic] : s.physics.open ? s.physics.history : [];
      return { physics: { ...s.physics, open: true, topic, history, x, y, w, h } };
    }),
  navigatePhysics: (topic) =>
    set((s) => (s.physics.topic === topic ? {} : { physics: { ...s.physics, topic, history: [...s.physics.history, s.physics.topic] } })),
  physicsBack: () =>
    set((s) => {
      const h = [...s.physics.history];
      const prev = h.pop();
      return prev ? { physics: { ...s.physics, topic: prev, history: h } } : {};
    }),
  closePhysics: () => set((s) => ({ physics: { ...s.physics, open: false, history: [] } })),
  movePhysics: (pos) => set((s) => ({ physics: { ...s.physics, ...pos } })),
  showInPhysicsTab: (topic) => set((s) => ({ tab: "physics", physicsTarget: { topic, nonce: (s.physicsTarget?.nonce ?? 0) + 1 }, physics: { ...s.physics, open: false, history: [] } })),
}));

// persist + URL hash sync
if (typeof window !== "undefined") {
  let timer: ReturnType<typeof setTimeout> | undefined;
  useStore.subscribe((s, prev) => {
    if (s.params !== prev.params || s.mode !== prev.mode || s.lang !== prev.lang || s.theme !== prev.theme || s.autoRun !== prev.autoRun || s.preset !== prev.preset || s.tab !== prev.tab || s.vgRange !== prev.vgRange || s.vgsRange !== prev.vgsRange) {
      clearTimeout(timer);
      timer = setTimeout(() => savePersisted(useStore.getState()), 250);
    }
    if (s.tab !== prev.tab || s.mode !== prev.mode) {
      try {
        const hash = `#tab=${s.tab}&mode=${s.mode}`;
        if (window.location.hash !== hash) window.history.replaceState(null, "", hash);
      } catch {
        /* ignore */
      }
    }
  });
  window.addEventListener("hashchange", () => {
    const h = readHash();
    const s = useStore.getState();
    if (h.tab && h.tab !== s.tab) s.setTab(h.tab);
    if (h.mode && h.mode !== s.mode) s.setMode(h.mode);
  });
}

/** Preset defaults for the current preset (used by "changed" markers and reset links). */
let memo: { meta: Meta; preset: PresetId; root: ParamRoot } | null = null;
export function presetDefaults(s: Pick<State, "meta" | "preset">): ParamRoot {
  if (!memo || memo.meta !== s.meta || memo.preset !== s.preset) memo = { meta: s.meta, preset: s.preset, root: presetRoot(s.meta, s.preset) };
  return memo.root;
}
