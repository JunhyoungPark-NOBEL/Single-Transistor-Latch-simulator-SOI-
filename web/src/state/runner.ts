// Job orchestration: backend selection (HTTP or mock), compute jobs per result key, run groups for the
// Run button, auto-run, and lazy loading of measured data / design map.
import { httpBackend, JobAborted, runJob, type Backend } from "../api/client";
import { checkResult } from "../api/guards";
import { normalizeDesignMap, normalizeMeasured } from "../api/measured";
import { createMockBackend } from "../api/mock";
import type { BranchesResult, Kind, Meta, PresetId } from "../api/types";
import { canonical } from "../utils/object";
import {
  branchesPayload, chargeBalancePayload, circuitPayload, hazardPayload, midFold, photoConditionPayload, sweepMcPayload,
  validationPayload, vgCurvePayload, vgStochPayload,
} from "../utils/payload";
import { useStore } from "./store";

let backend: Backend = httpBackend;
let mockBackend: Backend | null = null;
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

function forcedMockFromUrl(): boolean {
  try {
    const q = new URLSearchParams(window.location.search);
    return q.get("mock") === "1" || q.get("mock") === "true";
  } catch {
    return false;
  }
}

/** Probe /api/health; switch to the HTTP backend when it answers, otherwise to the mock backend. */
export async function initBackend(): Promise<void> {
  if (forcedMockFromUrl()) {
    useMock(true);
    return;
  }
  useStore.setState({ backend: "checking" });
  try {
    const h = await httpBackend.health();
    if (!h || !h.ok) throw new Error("unhealthy");
    backend = httpBackend;
    useStore.setState({ backend: "online", health: h });
    try {
      const meta = (await httpBackend.meta()) as Meta;
      if (meta && meta.presets) useStore.getState().setMeta(meta);
    } catch {
      /* keep built-in presets */
    }
  } catch {
    useMock(false);
  }
}

/** Periodic health check (updates the status dot; reconnects when an offline backend comes back). */
export function startHealthPolling(intervalMs = 20000): () => void {
  const id = setInterval(async () => {
    const s = useStore.getState();
    if (s.forcedMock) return;
    try {
      const h = await httpBackend.health();
      if (s.backend !== "online") {
        backend = httpBackend;
        useStore.setState({ backend: "online", health: h });
        httpBackend.meta().then((m) => m?.presets && useStore.getState().setMeta(m)).catch(() => undefined);
      } else useStore.setState({ health: h });
    } catch {
      if (s.backend === "online") useMock(false);
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
  const st = useStore.getState();
  const token = ++tokenSeq;
  tokens.set(key, token);
  const payloadKey = canonical({ kind, payload });
  const startedAt = performance.now();
  const isMock = backend.isMock;
  st.patchResult(key, { status: "queued", kind, progress: 0, message: "", error: undefined, startedAt, elapsed: 0, payloadKey, token });
  try {
    const data = await runJob<T>(backend, kind, payload, {
      isAborted: () => tokens.get(key) !== token,
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
      status: "done", data, dataKey: payloadKey, progress: 1, message: "", mock: isMock, elapsed: (performance.now() - startedAt) / 1000,
    });
    return { ok: true, data };
  } catch (e) {
    if (e instanceof JobAborted) {
      if (tokens.get(key) === token) useStore.getState().patchResult(key, { status: "cancelled", progress: 0, message: "" });
      return { ok: false };
    }
    if (tokens.get(key) !== token) return { ok: false };
    useStore.getState().patchResult(key, { status: "error", error: (e as Error).message || String(e), progress: 0 });
    return { ok: false };
  }
}

export function cancelKey(key: string) {
  const cur = useStore.getState().results[key];
  tokens.set(key, ++tokenSeq); // supersede -> runJob cancels the server job on its next poll
  if (cur && (cur.status === "running" || cur.status === "queued")) useStore.getState().patchResult(key, { status: "cancelled", progress: 0, message: "" });
}

// ---------------------------------------------------------------- run groups (Run button)
function beginGroup(keys: string[], label: string) {
  useStore.setState({ activeRun: { keys, label, startedAt: performance.now() } });
}
function endGroup(keys: string[]) {
  const ar = useStore.getState().activeRun;
  if (ar && ar.keys.join() === keys.join()) useStore.setState({ activeRun: { ...ar, finishedAt: performance.now() } });
}

export function cancelActive() {
  const ar = useStore.getState().activeRun;
  if (!ar) return;
  for (const k of ar.keys) cancelKey(k);
  useStore.setState({ activeRun: { ...ar, finishedAt: performance.now() } });
}

export async function runDeterministic() {
  const s = useStore.getState();
  const p = s.params;
  const keys = ["branches", "charge_balance", "vg_curve"];
  beginGroup(keys, "deterministic");
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
    runKey("vg_curve", "vg_curve", vgCurvePayload(p, s.vgRange)),
  ];
  if (cbFixed != null) jobs.push(runKey("charge_balance", "charge_balance", chargeBalancePayload(p, cbFixed)));
  await Promise.all(jobs);
  endGroup(keys);
}

export async function runStochastic() {
  const s = useStore.getState();
  const p = s.params;
  const keys = ["sweep_mc", "hazard", "branches"];
  beginGroup(keys, "stochastic");
  await Promise.all([
    runKey("sweep_mc", "sweep_mc", sweepMcPayload(p)),
    runKey("hazard", "hazard", hazardPayload(p)),
    runKey("branches", "branches", branchesPayload(p)),
  ]);
  endGroup(keys);
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
  beginGroup(keys, "circuit");
  await Promise.all([
    runKey("circuit", "circuit", circuitPayload(s.params, s.mode)),
    runKey("circuit_branches", "branches", { device: s.params.device, sweep: s.params.sweep }),
  ]);
  endGroup(keys);
}

export async function runValidation(level: "fast" | "full") {
  const keys = ["validation"];
  beginGroup(keys, `validation ${level}`);
  await runKey("validation", "validation", validationPayload(level));
  endGroup(keys);
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
  beginGroup(keys, "photo conditions");
  const token = tokenSeq;
  for (let k = 0; k < conds.length; k++) {
    const c = conds[k];
    const r = await runKey(keys[k], "sweep_mc", photoConditionPayload(meta.presets.photo, c.vg, c.power_mW));
    if (!r.ok && useStore.getState().results[keys[k]]?.status === "cancelled") break;
    void token;
  }
  endGroup(keys);
}

export async function runValidationVg() {
  const meta = useStore.getState().meta;
  const pr = meta.presets.paper;
  await runKey("val_vgs", "vg_curve_stochastic", { device: pr.device, sweep: pr.sweep, stochastic: pr.stochastic, vg_min: -3.6, vg_max: -0.9, n: 10 });
}

/** Run button dispatch for the current tab and mode. */
export function runCurrent() {
  const s = useStore.getState();
  if (s.tab === "circuit") return runCircuit();
  if (s.tab === "validation") return runValidation("fast");
  return s.mode === "stochastic" ? runStochastic() : runDeterministic();
}

// ---------------------------------------------------------------- data endpoints
export async function loadMeasured(force = false) {
  const s = useStore.getState();
  if (!force && (s.measured.status === "loading" || s.measured.status === "done")) return;
  useStore.setState({ measured: { status: "loading" } });
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
  try {
    const raw = await backend.designMap();
    useStore.setState({ designMap: { status: "done", data: normalizeDesignMap(raw) } });
  } catch (e) {
    useStore.setState({ designMap: { status: "error", error: (e as Error).message } });
  }
}

// ---------------------------------------------------------------- auto-run (deterministic, device tab)
export function startAutoRun(debounceMs = 700): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const unsub = useStore.subscribe((s, prev) => {
    if (!s.autoRun || s.mode !== "deterministic" || s.tab !== "device") return;
    const changed = s.params.device !== prev.params.device || s.params.sweep !== prev.params.sweep || (s.autoRun && !prev.autoRun);
    if (!changed) return;
    clearTimeout(timer);
    timer = setTimeout(() => void runDeterministic(), debounceMs);
  });
  return () => {
    clearTimeout(timer);
    unsub();
  };
}

export function presetLabel(meta: Meta, id: PresetId, lang: "ko" | "en"): string {
  return meta.presets[id]?.label?.[lang] ?? id;
}
