// Job orchestration: backend selection (HTTP or mock), compute jobs per result key, run groups for the
// Run button, auto-run, and lazy loading of measured data / design map.
import { ApiError, httpBackend, JobAborted, runJob, type Backend } from "../api/client";
import { translate } from "../i18n";
import { checkResult } from "../api/guards";
import { normalizeDesignMap, normalizeMeasured } from "../api/measured";
import { createMockBackend } from "../api/mock";
import { createSnapshotBackend, isSnapshotFallback, loadSnapshot, type Snapshot } from "../api/snapshot";
import type { BranchesResult, Kind, Meta, PresetId } from "../api/types";
import { canonical } from "../utils/object";
import {
  branchesPayload, chargeBalancePayload, circuitPayload, hazardPayload, midFold, photoConditionPayload, sweepMcPayload,
  validationPayload, vgCurvePayload, vgStochPayload,
} from "../utils/payload";
import { useStore } from "./store";
import { useLayout } from "./layout";
import { csvmPayload, useForcing } from "../device/forcing";
import { GEOMETRY_LIVE_REQUIRED, geometryError, hasChangedGeometry } from "../api/geometryPolicy";

let backend: Backend = httpBackend;
let mockBackend: Backend | null = null;
/** Static snapshot (snapshot/index.json next to the page) — probed once, used instead of the demo backend. */
let snapshotProbe: Promise<Snapshot | null> | null = null;
let snapshotBackend: Backend | null = null;
let markReady: () => void = () => undefined;
/** Resolves once the first backend probe finished (HTTP or mock chosen). */
export const backendReady: Promise<void> = new Promise((r) => (markReady = r));
const tokens = new Map<string, number>();
let tokenSeq = 0;

export function getBackend(): Backend {
  return backend;
}

function useMock(forced: boolean) {
  mockBackend ??= createMockBackend();
  backend = mockBackend;
  useStore.setState({ backend: forced ? "mock" : "offline", forcedMock: forced, health: { ok: true, version: "mock", workers: 0 } });
}

/** Backend unreachable: use the static snapshot when the page ships one, else the demo backend. */
async function useSnapshotOrMock(): Promise<void> {
  snapshotProbe ??= loadSnapshot();
  const snap = await snapshotProbe;
  if (!snap) {
    useMock(false);
    return;
  }
  snapshotBackend ??= createSnapshotBackend(snap, { fallback: () => (mockBackend ??= createMockBackend()) });
  backend = snapshotBackend;
  useStore.setState({ backend: "snapshot", forcedMock: false, health: await snapshotBackend.health() });
  if (snap.index.meta?.presets) useStore.getState().setMeta(snap.index.meta);
}

function forcedMockFromUrl(): boolean {
  try {
    const q = new URLSearchParams(window.location.search);
    return q.get("mock") === "1" || q.get("mock") === "true";
  } catch {
    return false;
  }
}

/** Probe /api/health; switch to the HTTP backend when it answers, otherwise to the mock backend. */
export function initBackend(): Promise<void> {
  return probeBackend().finally(() => markReady());
}

async function probeBackend(): Promise<void> {
  if (forcedMockFromUrl()) {
    useMock(true);
    return;
  }
  useStore.setState({ backend: "checking" });
  try {
    const h = await httpBackend.health();
    if (!h || !h.ok) throw new Error("unhealthy");
    const wasMock = backend !== httpBackend; // mock or snapshot data → reload measured data from the server
    backend = httpBackend;
    useStore.setState({ backend: "online", health: h, ...(wasMock ? { measured: { status: "idle" as const }, designMap: { status: "idle" as const } } : {}) });
    try {
      const meta = (await httpBackend.meta()) as Meta;
      if (meta && meta.presets) useStore.getState().setMeta(meta);
    } catch {
      /* keep built-in presets */
    }
  } catch {
    await useSnapshotOrMock();
  }
}

/** Periodic health check (updates the status dot; reconnects when an offline backend comes back). */
export function startHealthPolling(intervalMs = 20000): () => void {
  const id = setInterval(async () => {
    const s = useStore.getState();
    if (s.forcedMock || s.backend === "snapshot") return; // snapshot: re-probe from the status dot instead
    try {
      const h = await httpBackend.health();
      if (!h || !h.ok) throw new Error("unhealthy");
      if (s.backend !== "online") {
        backend = httpBackend;
        useStore.setState({ backend: "online", health: h, measured: { status: "idle" }, designMap: { status: "idle" } });
        httpBackend.meta().then((m) => m?.presets && useStore.getState().setMeta(m)).catch(() => undefined);
      } else useStore.setState({ health: h });
    } catch {
      if (s.backend === "online") void useSnapshotOrMock();
    }
  }, intervalMs);
  return () => clearInterval(id);
}

// ---------------------------------------------------------------- single jobs
export interface RunResult<T> {
  ok: boolean;
  data?: T;
}

/** Run one compute job into result slot `key`. A newer run of the same key supersedes (and cancels) the older one. */
export async function runKey<T = unknown>(key: string, kind: Kind, payload: unknown): Promise<RunResult<T>> {
  await backendReady;
  const st = useStore.getState();
  logJob(key);
  const token = ++tokenSeq;
  tokens.set(key, token);
  const payloadKey = canonical({ kind, payload });
  const startedAt = performance.now();
  const isMock = backend.isMock;
  st.patchResult(key, { status: "queued", kind, progress: 0, message: "", error: undefined, startedAt, elapsed: 0, payloadKey, token });
  try {
    if (backend.isMock && hasChangedGeometry(payload)) throw new Error(GEOMETRY_LIVE_REQUIRED);
    const data = await runJob<T>(backend, kind, payload, {
      isAborted: () => tokens.get(key) !== token,
      // HTTP 429: the server queue is full — say so and retry once after Retry-After instead of failing
      onBusy: (sec) => {
        if (tokens.get(key) !== token) return;
        useStore.getState().patchResult(key, { status: "queued", progress: 0, message: translate(useStore.getState().lang, "busy.retry", { s: Math.ceil(sec) }) });
      },
      onStatus: (js) => {
        if (tokens.get(key) !== token) return;
        useStore.getState().patchResult(key, {
          status: js.status === "done" ? "running" : (js.status as "queued" | "running"),
          progress: js.progress ?? 0,
          message: js.message ?? "",
          cached: js.cached,
          elapsed: js.elapsed_s,
        });
      },
    });
    if (tokens.get(key) !== token) return { ok: false };
    const missing = checkResult(kind, data);
    if (missing.length) {
      useStore.getState().patchResult(key, { status: "error", error: `unexpected result shape (missing: ${missing.join(", ")})`, progress: 1 });
      return { ok: false };
    }
    useStore.getState().patchResult(key, {
      status: "done", data, dataKey: payloadKey, progress: 1, message: "", mock: isMock || isSnapshotFallback(data), elapsed: (performance.now() - startedAt) / 1000,
    });
    return { ok: true, data };
  } catch (e) {
    if (e instanceof JobAborted) {
      if (tokens.get(key) === token) useStore.getState().patchResult(key, { status: "cancelled", progress: 0, message: "" });
      return { ok: false };
    }
    if (tokens.get(key) !== token) return { ok: false };
    const busy = e instanceof ApiError && e.status === 429;
    const detail = (e as Error).message || String(e);
    const msg = busy ? translate(useStore.getState().lang, "busy.failed") : geometryError(detail, useStore.getState().lang);
    useStore.getState().patchResult(key, { status: "error", error: msg, progress: 0, message: "" });
    return { ok: false };
  }
}

/** Dev builds only: every submitted job key in `window.__stlJobs` (the e2e specs check what a change re-runs). */
function logJob(key: string) {
  try {
    if (import.meta.env?.DEV && typeof window !== "undefined") ((window as unknown as { __stlJobs?: string[] }).__stlJobs ??= []).push(key);
  } catch {
    /* ignore */
  }
}

/**
 * Like runKey, but keeps a completed result whose payload is the one requested (no job, no progress overlay).
 * Used for the V_G curve, whose payload does not change with V_G (utils/payload.ts, VG_CURVE_CANONICAL_VG).
 */
export async function runKeyIfChanged<T = unknown>(key: string, kind: Kind, payload: unknown): Promise<RunResult<T>> {
  const e = useStore.getState().results[key];
  if (e && e.status === "done" && e.data !== undefined && e.dataKey === canonical({ kind, payload })) return { ok: true, data: e.data as T };
  return runKey<T>(key, kind, payload);
}

export function cancelKey(key: string) {
  const cur = useStore.getState().results[key];
  tokens.set(key, ++tokenSeq); // supersede -> runJob cancels the server job on its next poll
  if (cur && (cur.status === "running" || cur.status === "queued")) useStore.getState().patchResult(key, { status: "cancelled", progress: 0, message: "" });
}

// ---------------------------------------------------------------- run groups (Run button)
/** Start a run group (Run bar progress). Returns its identity for endGroup. */
function beginGroup(keys: string[], label: string): number {
  const startedAt = performance.now();
  useStore.setState({ activeRun: { keys, label, startedAt } });
  return startedAt;
}
/** Finish a run group — only if it is still the active one (a newer run of the same keys may have replaced it). */
function endGroup(id: number) {
  const ar = useStore.getState().activeRun;
  if (ar && ar.startedAt === id && ar.finishedAt === undefined) useStore.setState({ activeRun: { ...ar, finishedAt: performance.now() } });
}

/** Run-group label for the Run bar of a tab/mode (the bar only reports runs of its own context). */
export function runContext(tab: string, mode: string): string | null {
  if (tab === "circuit") return `circuit ${mode}`;
  if (tab === "device") return useForcing.getState().forcing === "csvm" ? `csvm ${mode}` : mode;
  return null;
}

export function cancelActive() {
  const ar = useStore.getState().activeRun;
  if (!ar) return;
  for (const k of ar.keys) cancelKey(k);
  useStore.setState({ activeRun: { ...ar, finishedAt: performance.now() } });
}

/** Add a key to the running group (the Run bar then waits for it too). */
function extendGroup(id: number, key: string) {
  const ar = useStore.getState().activeRun;
  if (ar && ar.startedAt === id && ar.finishedAt === undefined && !ar.keys.includes(key)) useStore.setState({ activeRun: { ...ar, keys: [...ar.keys, key] } });
}

export async function runDeterministic() {
  if (useForcing.getState().forcing === "csvm") return runCsvm();
  const s = useStore.getState();
  const p = s.params;
  if (useLayout.getState().layout === "simple") {
    const gid = beginGroup(["branches"], "deterministic");
    try {
      const r = await runKey<BranchesResult>("branches", "branches", branchesPayload(p));
      // no latch at this V_G: the V_G curve tells the answer bar where the latch window is
      if (r.ok && r.data && !r.data.latch) {
        extendGroup(gid, "vg_curve");
        await runKeyIfChanged("vg_curve", "vg_curve", vgCurvePayload(p, useStore.getState().vgRange));
      }
    } finally { endGroup(gid); }
    return;
  }
  const keys = ["branches", "charge_balance", "vg_curve"];
  const gid = beginGroup(keys, "deterministic");
  const cbFixed = s.cbVd;
  const jobs: Promise<unknown>[] = [
    runKey<BranchesResult>("branches", "branches", branchesPayload(p)).then((r) => {
      if (cbFixed == null) {
        const f = r.data?.folds;
        const vd = midFold(f?.V_LU, f?.V_LD, 0.8 * p.sweep.vd_max_V);
        return runKey("charge_balance", "charge_balance", chargeBalancePayload(p, vd));
      }
      return undefined;
    }),
    // the V_G curve does not depend on the sidebar V_G: keep it when only V_G changed
    runKeyIfChanged("vg_curve", "vg_curve", vgCurvePayload(p, s.vgRange)),
  ];
  if (cbFixed != null) jobs.push(runKey("charge_balance", "charge_balance", chargeBalancePayload(p, cbFixed)));
  await Promise.all(jobs);
  endGroup(gid);
}

export async function runStochastic() {
  if (useForcing.getState().forcing === "csvm") return runCsvm();
  const s = useStore.getState();
  const p = s.params;
  if (useLayout.getState().layout === "simple") {
    const gid = beginGroup(["sweep_mc", "branches"], "stochastic");
    try { await Promise.all([runKey("sweep_mc", "sweep_mc", sweepMcPayload(p)), runKey("branches", "branches", branchesPayload(p))]); } finally { endGroup(gid); }
    return;
  }
  const keys = ["sweep_mc", "hazard", "branches"];
  const gid = beginGroup(keys, "stochastic");
  await Promise.all([
    runKey("sweep_mc", "sweep_mc", sweepMcPayload(p)),
    runKey("hazard", "hazard", hazardPayload(p)),
    runKey("branches", "branches", branchesPayload(p)),
  ]);
  endGroup(gid);
}

/** Result slot done for exactly this request. */
function freshFor(key: string, kind: Kind, payload: unknown): boolean {
  const e = useStore.getState().results[key];
  return !!e && e.status === "done" && e.data !== undefined && e.dataKey === canonical({ kind, payload });
}
const inFlight = (key: string) => {
  const st = useStore.getState().results[key]?.status;
  return st === "running" || st === "queued";
};

/**
 * "모두 보기" shows panels whose results the simple layout never computes (charge balance and V_G curve;
 * hazard in the stochastic mode). On switching to it, run the ones missing or stale for the current
 * parameters, but only once the main result (I–V branches / MC sweeps) exists for these parameters: opening
 * the layout never starts the main run by itself.
 */
export async function fillAllLayout() {
  const s = useStore.getState();
  if (s.tab !== "device" || useForcing.getState().forcing === "csvm" || useLayout.getState().layout !== "all") return;
  const p = s.params;
  const jobs: [string, Kind, unknown][] = [];
  if (s.mode === "deterministic") {
    if (!freshFor("branches", "branches", branchesPayload(p))) return;
    const f = (s.results.branches?.data as BranchesResult | undefined)?.folds;
    const vd = s.cbVd ?? midFold(f?.V_LU, f?.V_LD, 0.8 * p.sweep.vd_max_V);
    jobs.push(["charge_balance", "charge_balance", chargeBalancePayload(p, vd)], ["vg_curve", "vg_curve", vgCurvePayload(p, s.vgRange)]);
  } else {
    if (!freshFor("sweep_mc", "sweep_mc", sweepMcPayload(p))) return;
    jobs.push(["hazard", "hazard", hazardPayload(p)]);
  }
  const todo = jobs.filter(([key, kind, payload]) => !freshFor(key, kind, payload) && !inFlight(key));
  if (!todo.length) return;
  const gid = beginGroup(todo.map(([key]) => key), runContext("device", s.mode) ?? s.mode);
  try {
    await Promise.all(todo.map(([key, kind, payload]) => runKey(key, kind, payload)));
  } finally {
    endGroup(gid);
  }
}

/** Fill the "모두 보기" panels when the layout switches to it (App mounts this once). */
export function startAllLayoutFill(): () => void {
  return useLayout.subscribe((st, prev) => {
    if (st.layout === "all" && prev.layout !== "all") void fillAllLayout();
  });
}

/** Device CSVM uses the same live MNA/body-state solver as the free-form circuit editor. */
export async function runCsvm() {
  await backendReady;
  const s = useStore.getState();
  const payload = csvmPayload(s.params, s.mode, useForcing.getState().settings);
  const gid = beginGroup(["device_csvm"], `csvm ${s.mode}`);
  try {
    if (s.backend !== "online") {
      s.patchResult("device_csvm", {
        kind: "circuit", status: "error", progress: 0,
        error: s.lang === "ko" ? "CSVM은 과도 해석 서버 연결이 필요합니다." : "CSVM requires a live transient solver connection.",
      });
      return;
    }
    await runKey("device_csvm", "circuit", payload);
  } finally { endGroup(gid); }
}

export async function runChargeBalance(vd: number) {
  const p = useStore.getState().params;
  await runKey("charge_balance", "charge_balance", chargeBalancePayload(p, vd));
}

export async function runVgCurve() {
  const s = useStore.getState();
  await runKey("vg_curve", "vg_curve", vgCurvePayload(s.params, s.vgRange));
}

export async function runVgStochastic() {
  const s = useStore.getState();
  await runKey("vg_curve_stochastic", "vg_curve_stochastic", vgStochPayload(s.params, s.vgsRange));
}

export async function runCircuit() {
  const s = useStore.getState();
  const keys = ["circuit", "circuit_branches"];
  const gid = beginGroup(keys, runContext("circuit", s.mode) ?? "circuit");
  await Promise.all([
    runKey("circuit", "circuit", circuitPayload(s.params, s.mode)),
    runKey("circuit_branches", "branches", { device: s.params.device, sweep: s.params.sweep }),
  ]);
  endGroup(gid);
}

export async function runValidation(level: "fast" | "full") {
  const keys = ["validation"];
  const gid = beginGroup(keys, `validation ${level}`);
  await runKey("validation", "validation", validationPayload(level));
  endGroup(gid);
}

export async function runValidationIV() {
  const meta = useStore.getState().meta;
  const pr = meta.presets.paper;
  await runKey("val_branches_paper", "branches", { device: pr.device, sweep: pr.sweep });
}

/** Model V_LU statistics for the 8 measured photo-device conditions (sequential, cancellable). */
export async function runValidationPhoto() {
  const meta = useStore.getState().meta;
  const conds = meta.measured_photo_conditions ?? [];
  const keys = conds.map((_, k) => `val_photo_${k}`);
  const gid = beginGroup(keys, "photo conditions");
  for (let k = 0; k < conds.length; k++) {
    const c = conds[k];
    const r = await runKey(keys[k], "sweep_mc", photoConditionPayload(meta.presets.photo, c.vg, c.power_mW));
    if (!r.ok && useStore.getState().results[keys[k]]?.status === "cancelled") break;
  }
  endGroup(gid);
}

export async function runValidationVg() {
  const meta = useStore.getState().meta;
  const pr = meta.presets.paper;
  await runKey("val_vgs", "vg_curve_stochastic", { device: pr.device, sweep: pr.sweep, stochastic: pr.stochastic, vg_min: -3.6, vg_max: -0.9, n: 10 });
}

/** Circuit-tab Run override: the schematic editor registers one while its sub-view is active (circuit/view.ts). */
let circuitRunOverride: (() => Promise<unknown>) | null = null;
export function setCircuitRunOverride(fn: (() => Promise<unknown>) | null) {
  circuitRunOverride = fn;
}

/** Run button dispatch for the current tab and mode. */
export function runCurrent() {
  const s = useStore.getState();
  if (s.tab === "circuit") return circuitRunOverride ? circuitRunOverride() : runCircuit();
  if (s.tab === "validation") return runValidationIV();
  return s.mode === "stochastic" ? runStochastic() : runDeterministic();
}

// ---------------------------------------------------------------- data endpoints
export async function loadMeasured(force = false) {
  const s = useStore.getState();
  if (!force && (s.measured.status === "loading" || s.measured.status === "done")) return;
  useStore.setState({ measured: { status: "loading" } });
  await backendReady;
  try {
    const raw = await backend.measured();
    useStore.setState({ measured: { status: "done", data: normalizeMeasured(raw) } });
  } catch (e) {
    useStore.setState({ measured: { status: "error", error: (e as Error).message } });
  }
}

export async function loadDesignMap(force = false) {
  const s = useStore.getState();
  if (!force && (s.designMap.status === "loading" || s.designMap.status === "done")) return;
  useStore.setState({ designMap: { status: "loading" } });
  await backendReady;
  try {
    const raw = await backend.designMap();
    useStore.setState({ designMap: { status: "done", data: normalizeDesignMap(raw) } });
  } catch (e) {
    useStore.setState({ designMap: { status: "error", error: (e as Error).message } });
  }
}

// ---------------------------------------------------------------- auto-run (deterministic, device tab)
/** Device · deterministic with auto-run on and no I–V result yet (nor one on its way). */
function needsFirstRun(s: ReturnType<typeof useStore.getState>): boolean {
  if (!s.autoRun || s.tab !== "device" || s.mode !== "deterministic") return false;
  const b = s.results[useForcing.getState().forcing === "csvm" ? "device_csvm" : "branches"];
  return !b || (b.data === undefined && b.status !== "running" && b.status !== "queued");
}
let firstLoadRun = false;

/**
 * Auto-run (Device · deterministic; stochastic runs are never started automatically): re-runs 700 ms after a
 * device/sweep change, when the switch is turned on, and once when the page opens on Device · deterministic
 * (after the backend probe) or switches there without an I–V result.
 */
export function startAutoRun(debounceMs = 700): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const schedule = (delay = debounceMs) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      const now = useStore.getState();
      if (now.autoRun && now.mode === "deterministic" && now.tab === "device") void runDeterministic();
    }, delay);
  };
  const unsub = useStore.subscribe((s, prev) => {
    if (!s.autoRun || s.mode !== "deterministic" || s.tab !== "device") return;
    const changed = s.params.device !== prev.params.device || (useForcing.getState().forcing === "vscm" && s.params.sweep !== prev.params.sweep) || (s.autoRun && !prev.autoRun);
    const entered = (s.tab !== prev.tab || s.mode !== prev.mode) && firstLoadRun && needsFirstRun(s);
    if (!changed && !entered) return;
    schedule(entered && !changed ? 0 : debounceMs);
  });
  const unsubForcing = useForcing.subscribe((s, prev) => {
    const app = useStore.getState();
    if (!app.autoRun || app.tab !== "device" || app.mode !== "deterministic") return;
    if (s.forcing !== prev.forcing || (s.forcing === "csvm" && s.settings !== prev.settings)) schedule();
  });
  // first load: one run as soon as the backend is chosen (StrictMode mounts twice — run once)
  void backendReady.then(() => {
    if (firstLoadRun) return;
    firstLoadRun = true;
    if (needsFirstRun(useStore.getState())) void runDeterministic();
  });
  return () => {
    clearTimeout(timer);
    unsub();
    unsubForcing();
  };
}

export function presetLabel(meta: Meta, id: PresetId, lang: "ko" | "en"): string {
  return meta.presets[id]?.label?.[lang] ?? id;
}
