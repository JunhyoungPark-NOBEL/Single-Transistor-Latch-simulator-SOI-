import { hasSimpleModel, SIMPLE_LIVE_REQUIRED } from "../params/model";
// Job orchestration: backend selection (HTTP or mock), compute jobs per result key, run groups for the
// Run button, auto-run, and lazy loading of measured data / design map.
import { ApiError, authenticateServer, configureHttpBackend, httpBackend, JobAborted, readServerSession, retireHttpBackend, runJob, type Backend } from "../api/client";
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
import { connectionError, getSessionToken, selectEndpoint, setSessionToken, useConnection } from "../api/connection";
import { usePrevRuns } from "./prevRuns";

const offlineBackend: Backend = {
  isMock: false,
  health: async () => ({ ok: false }),
  meta: async () => { throw new Error("connection-required"); },
  submit: async () => { throw new Error("connection-required"); },
  job: async () => { throw new Error("connection-required"); },
  cancel: async () => undefined,
  measured: async () => { throw new Error("connection-required"); },
  designMap: async () => { throw new Error("connection-required"); },
};
let backend: Backend = offlineBackend;
let mockBackend: Backend | null = null;
let snapshotProbe: Promise<Snapshot | null> | null = null;
let markReady: () => void = () => undefined;
/** Resolves after the initial connection attempt, including the offline state. */
export const backendReady: Promise<void> = new Promise((r) => (markReady = r));
const tokens = new Map<string, number>();
let tokenSeq = 0;
let backendEpoch = 0;
let probeInFlight: Promise<void> | null = null;

export function getBackend(): Backend { return backend; }
export function getBackendEpoch(): number { return backendEpoch; }
export async function awaitBackendReady(): Promise<void> {
  await backendReady;
  while (probeInFlight) await probeInFlight;
}

/** Every active request keeps its old backend; late completions are discarded. */
function invalidateConnection(): number {
  backendEpoch++;
  tokens.clear();
  retireHttpBackend();
  useStore.setState({ results: {}, activeRun: null, measured: { status: "idle" }, designMap: { status: "idle" }, health: null });
  usePrevRuns.setState({ cur: {}, prev: {} });
  return backendEpoch;
}

/** A revoked session or lost transport cannot remain labelled as a live connection. */
export function reportConnectionFailure(error: unknown, epoch: number): void {
  if (epoch !== backendEpoch || !(error instanceof ApiError) || ![0, 401, 403].includes(error.status)) return;
  invalidateConnection();
  backend = offlineBackend;
  useStore.setState({ backend: "offline" });
  const auth = error.status === 401 || error.status === 403;
  if (auth) setSessionToken(null);
  useConnection.setState({ status: auth ? "auth-required" : "offline", error: error.message });
}

function useMock(): void {
  mockBackend ??= createMockBackend();
  backend = mockBackend;
  useStore.setState({ backend: "mock", forcedMock: true, health: { ok: true, version: "mock", workers: 0 } });
  useConnection.setState({ status: "demo", error: null });
}

/** Offline recordings are exact requests only. Missing results are never synthesized. */
async function useSnapshotOrOffline(epoch: number, error: string): Promise<void> {
  backend = offlineBackend;
  useStore.setState({ backend: "offline", forcedMock: false, health: null });
  useConnection.setState({ status: "offline", error });
  if (useConnection.getState().endpoint) return;
  snapshotProbe ??= loadSnapshot();
  const snap = await snapshotProbe;
  if (epoch !== backendEpoch || !snap) return;
  const recorded = createSnapshotBackend(snap, { fallback: () => offlineBackend, exactOnly: true });
  recorded.isCurrent = () => epoch === backendEpoch;
  backend = recorded;
  useStore.setState({ backend: "snapshot", forcedMock: false, health: await recorded.health() });
  useConnection.setState({ status: "snapshot", error: null });
  if (snap.index.meta?.presets) useStore.getState().setMeta(snap.index.meta);
}

function forcedMockFromUrl(): boolean {
  if (!import.meta.env.DEV) return false;
  try {
    const q = new URLSearchParams(window.location.search);
    return q.get("mock") === "1" || q.get("mock") === "true";
  } catch { return false; }
}

/** Explicit reconnect, or the initial probe. This also retires jobs from an earlier connection. */
export function initBackend(password?: string): Promise<void> {
  const epoch = invalidateConnection();
  const endpoint = useConnection.getState().endpoint;
  backend = configureHttpBackend({ endpoint, token: getSessionToken() });
  const task = probeBackend(epoch, password).finally(() => {
    markReady();
    if (probeInFlight === task) probeInFlight = null;
  });
  probeInFlight = task;
  return task;
}

export async function connectToServer(endpoint: string, password?: string): Promise<void> {
  selectEndpoint(endpoint);
  await initBackend(password);
  const current = useStore.getState();
  if (current.backend === "online" && current.autoRun && current.tab === "device" && current.mode === "deterministic") void runDeterministic();
}

async function probeBackend(epoch: number, password?: string): Promise<void> {
  if (forcedMockFromUrl()) { useMock(); return; }
  useStore.setState({ backend: "checking", forcedMock: false });
  useConnection.setState({ status: "checking", error: null });
  const target = httpBackend;
  const config = { endpoint: useConnection.getState().endpoint, token: getSessionToken() };
  try {
    const h = await target.health() as Awaited<ReturnType<Backend["health"]>> & { access_gate?: string; auth_required?: boolean };
    if (epoch !== backendEpoch) return;
    if (!h?.ok) throw new Error("unhealthy");
    if (h.access_gate === "misconfigured") throw new Error("server-misconfigured");
    if (h.auth_required || h.access_gate === "on") {
      const session = password ? await authenticateServer(config, password) : await readServerSession(config);
      if (epoch !== backendEpoch) return;
      if (!session.authenticated) throw new ApiError("authentication-required", 401);
      if (password) {
        setSessionToken(session.token ?? null);
        backend = configureHttpBackend({ endpoint: config.endpoint, token: getSessionToken() });
      }
    }
    const connected = httpBackend;
    try {
      const meta = await connected.meta();
      if (epoch !== backendEpoch) return;
      if (meta?.presets) useStore.getState().setMeta(meta);
    } catch (e) {
      if (e instanceof ApiError && (e.status === 401 || e.status === 403)) throw e;
      // Older compatible servers may not provide metadata; bundled parameter definitions remain available.
    }
    if (epoch !== backendEpoch) return;
    backend = connected;
    useStore.setState({ backend: "online", forcedMock: false, health: h });
    useConnection.setState({ status: "online", error: null });
  } catch (e) {
    if (epoch !== backendEpoch) return;
    retireHttpBackend();
    const message = (e as Error).message || "network error";
    if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
      setSessionToken(null);
      backend = offlineBackend;
      useStore.setState({ backend: "offline", health: null });
      useConnection.setState({ status: "auth-required", error: message });
      return;
    }
    await useSnapshotOrOffline(epoch, message);
  }
}

/** Health checks never replace real results with a demonstration curve. */
export function startHealthPolling(intervalMs = 20000): () => void {
  let pending = false;
  const id = setInterval(async () => {
    const s = useStore.getState();
    if (pending || probeInFlight || s.forcedMock || s.backend === "snapshot" || useConnection.getState().status === "auth-required") return;
    pending = true;
    const epoch = backendEpoch;
    try {
      if (s.backend !== "online") { await initBackend(); return; }
      const h = await httpBackend.health();
      if (epoch !== backendEpoch) return;
      if (!h?.ok) throw new Error("unhealthy");
      useStore.setState({ health: h });
    } catch (e) {
      if (epoch !== backendEpoch) return;
      const next = invalidateConnection();
      await useSnapshotOrOffline(next, (e as Error).message || "network error");
    } finally { pending = false; }
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
  await awaitBackendReady();
  const epoch = backendEpoch;
  const target = backend;
  const st = useStore.getState();
  logJob(key);
  const token = ++tokenSeq;
  tokens.set(key, token);
  const payloadKey = canonical({ kind, payload });
  const startedAt = performance.now();
  const isMock = target.isMock;
  st.patchResult(key, { status: "queued", kind, progress: 0, message: "", error: undefined, startedAt, elapsed: 0, payloadKey, token });
  try {
    if (target.isMock && (hasSimpleModel(payload) || kind === "simple_calibrate")) throw new Error(SIMPLE_LIVE_REQUIRED);
    if (target.isMock && hasChangedGeometry(payload)) throw new Error(GEOMETRY_LIVE_REQUIRED);
    const data = await runJob<T>(target, kind, payload, {
      isAborted: () => tokens.get(key) !== token || epoch !== backendEpoch,
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
    const msg = busy ? translate(useStore.getState().lang, "busy.failed") : connectionError(geometryError(detail, useStore.getState().lang), useStore.getState().lang);
    reportConnectionFailure(e, epoch);
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

export async function runDeterministic() {
  if (useForcing.getState().forcing === "csvm") return runCsvm();
  const s = useStore.getState();
  const p = s.params;
  if (useLayout.getState().layout === "simple") {
    const gid = beginGroup(["branches"], "deterministic");
    try { await runKey("branches", "branches", branchesPayload(p)); } finally { endGroup(gid); }
    return;
  }
  if (p.device.model === "simple") {
    const gid = beginGroup(["branches", "vg_curve"], "deterministic");
    try { await Promise.all([runKey("branches", "branches", branchesPayload(p)), runKeyIfChanged("vg_curve", "vg_curve", vgCurvePayload(p, s.vgRange))]); } finally { endGroup(gid); }
    return;
  }
  const keys = ["branches", "charge_balance", "vg_curve"];
  const gid = beginGroup(keys, "deterministic");
  const cbFixed = s.cbVd;
  const jobs: Promise<unknown>[] = [
    runKey<BranchesResult>("branches", "branches", branchesPayload(p)).then((r) => {
      if (r.ok && cbFixed == null) {
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

/** Device CSVM uses the same live MNA/body-state solver as the free-form circuit editor. */
export async function runCsvm() {
  await awaitBackendReady();
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
  if (s.tab !== "device") return;
  return s.mode === "stochastic" ? runStochastic() : runDeterministic();
}

// ---------------------------------------------------------------- data endpoints
export async function loadMeasured(force = false) {
  const s = useStore.getState();
  if (!force && (s.measured.status === "loading" || s.measured.status === "done")) return;
  useStore.setState({ measured: { status: "loading" } });
  await awaitBackendReady();
  const epoch = backendEpoch;
  const target = backend;
  try {
    const raw = await target.measured();
    if (epoch !== backendEpoch) return;
    useStore.setState({ measured: { status: "done", data: normalizeMeasured(raw) } });
  } catch (e) {
    if (epoch === backendEpoch) useStore.setState({ measured: { status: "error", error: connectionError((e as Error).message, useStore.getState().lang) } });
  }
}

export async function loadDesignMap(force = false) {
  const s = useStore.getState();
  if (!force && (s.designMap.status === "loading" || s.designMap.status === "done")) return;
  useStore.setState({ designMap: { status: "loading" } });
  await awaitBackendReady();
  const epoch = backendEpoch;
  const target = backend;
  try {
    const raw = await target.designMap();
    if (epoch !== backendEpoch) return;
    useStore.setState({ designMap: { status: "done", data: normalizeDesignMap(raw) } });
  } catch (e) {
    if (epoch === backendEpoch) useStore.setState({ designMap: { status: "error", error: connectionError((e as Error).message, useStore.getState().lang) } });
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
