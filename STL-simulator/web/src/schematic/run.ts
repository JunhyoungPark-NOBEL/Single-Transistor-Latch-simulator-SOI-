// Run orchestration for the schematic: ERC gate, feasibility confirmation, job submission through the
// shared job runner (polling, cancel, HTTP 429 retry), with no synthetic fallback results.
import { ApiError, JobAborted } from "../api/client";
import { checkCustomResult, runCustomCircuit, type CustomCircuitRequest, type CustomCircuitResult } from "../api/circuitCustom";
import { connectionError } from "../api/connection";
import { geometryError } from "../api/geometryPolicy";
import { isSnapshotFallback } from "../api/snapshot";
import { translate } from "../i18n";
import { awaitBackendReady, getBackend, getBackendEpoch, reportConnectionFailure } from "../state/runner";
import { useStore } from "../state/store";
import { canonical } from "../utils/object";
import { hasErrors, runErc } from "./erc";
import { estimate } from "./feasibility";
import { buildRequest } from "./netlist";
import { extractNets, optionalPinConnected } from "./nets";
import { useSch } from "./store";

export const RESULT_KEY = "schematic";
let token = 0;

export function currentRequest(): CustomCircuitRequest {
  const st = useSch.getState();
  const mode = useStore.getState().mode;
  return buildRequest(st.doc, extractNets(st.doc), mode, st.traces);
}

/** Canonical key of a request (staleness of the shown result). */
export const requestKey = (req: CustomCircuitRequest) => canonical({ kind: "circuit", payload: req });

export interface SchematicRunData {
  result: CustomCircuitResult;
  request: CustomCircuitRequest;
  demoFallback: boolean;
}

export async function runSchematic(opts: { confirmed?: boolean } = {}): Promise<void> {
  const st = useSch.getState();
  const app = useStore.getState();
  const lang = app.lang;
  const conn = extractNets(st.doc);
  const erc = runErc(st.doc, conn);
  if (hasErrors(erc)) {
    const errs = erc.filter((i) => i.level === "error");
    st.flash(errs.flatMap((i) => [...i.elementIds, ...(i.wireIds ?? [])]));
    st.notify(translate(lang, "schematic.run.ercBlocked", { n: errs.length }), "err");
    window.dispatchEvent(new CustomEvent("sch-open-erc"));
    return;
  }
  const externalStl = st.doc.elements.filter((e) => e.kind === "STL" && (optionalPinConnected(e.id, "bg", conn) || optionalPinConnected(e.id, "b", conn)));
  if (externalStl.length && (app.mode !== "deterministic" || st.doc.tran.method !== "BE")) {
    st.flash(externalStl.map((e) => e.id));
    st.notify(translate(lang, "schematic.stl.deterministicBE"), "err");
    return;
  }
  const est = estimate(st.doc, app.mode);
  if (!opts.confirmed && (est.level === "heavy" || est.level === "refuse")) {
    st.set({ confirm: est });
    return;
  }
  st.set({ confirm: null });
  const req = buildRequest(st.doc, conn, app.mode, st.traces);
  await awaitBackendReady();
  const my = ++token;
  const epoch = getBackendEpoch();
  const key = requestKey(req);
  const startedAt = performance.now();
  const label = `schematic ${app.mode}`;
  useStore.setState({ activeRun: { keys: [RESULT_KEY], label, startedAt } });
  const patch = useStore.getState().patchResult;
  patch(RESULT_KEY, { status: "queued", kind: "circuit", progress: 0, message: "", error: undefined, startedAt, elapsed: 0, payloadKey: key, token: my });
  // superseded by a newer run, or cancelled from the Run bar (cancelActive marks the slot "cancelled")
  const aborted = () => epoch !== getBackendEpoch() || token !== my || useStore.getState().results[RESULT_KEY]?.status === "cancelled";
  const finish = () => {
    const ar = useStore.getState().activeRun;
    if (ar && ar.startedAt === startedAt && ar.finishedAt === undefined) useStore.setState({ activeRun: { ...ar, finishedAt: performance.now() } });
  };
  const run = async (backend: ReturnType<typeof getBackend>) =>
    runCustomCircuit(backend, req, {
      isAborted: aborted,
      onBusy: (sec) => !aborted() && patch(RESULT_KEY, { status: "queued", progress: 0, message: translate(useStore.getState().lang, "busy.retry", { s: Math.ceil(sec) }) }),
      onStatus: (js) =>
        !aborted() &&
        patch(RESULT_KEY, { status: js.status === "done" ? "running" : (js.status as "queued" | "running"), progress: js.progress ?? 0, message: js.message ?? "", cached: js.cached, elapsed: js.elapsed_s }),
    });
  const backend = getBackend();
  const demoFallback = false;
  try {
    const result = await run(backend);
    if (aborted()) return;
    const missing = checkCustomResult(result);
    if (missing.length) {
      patch(RESULT_KEY, { status: "error", error: `unexpected result shape (missing: ${missing.join(", ")})`, progress: 1 });
      return;
    }
    const data: SchematicRunData = { result, request: req, demoFallback };
    patch(RESULT_KEY, { status: "done", data, dataKey: key, progress: 1, message: "", mock: backend.isMock || isSnapshotFallback(result), elapsed: (performance.now() - startedAt) / 1000 });
    afterRun(result);
  } catch (e) {
    if (e instanceof JobAborted) {
      if (!aborted()) patch(RESULT_KEY, { status: "cancelled", progress: 0, message: "" });
      return;
    }
    if (aborted()) return;
    const busy = e instanceof ApiError && e.status === 429;
    const msg = (e as Error).message || String(e);
    reportConnectionFailure(e, epoch);
    patch(RESULT_KEY, { status: "error", error: busy ? translate(useStore.getState().lang, "busy.failed") : connectionError(geometryError(msg, useStore.getState().lang), useStore.getState().lang), progress: 0, message: "" });
    // server-side ERC / validation errors name elements and nodes: highlight them on the schematic
    if (!busy) {
      const ids = idsFromMessage(msg);
      if (ids.length) useSch.getState().flash(ids);
    }
  } finally {
    finish();
  }
}

export function cancelSchematic() {
  token++;
  const cur = useStore.getState().results[RESULT_KEY];
  if (cur && (cur.status === "running" || cur.status === "queued")) useStore.getState().patchResult(RESULT_KEY, { status: "cancelled", progress: 0, message: "" });
}

/** Default traces after a run (kept when still present): labelled node voltages + STL drain currents. */
function afterRun(result: CustomCircuitResult) {
  const st = useSch.getState();
  const keys = new Set(result.runs[0]?.signals.map((s) => s.key) ?? []);
  let traces = st.traces.filter((k) => keys.has(k));
  if (!traces.length) {
    const labels = new Set(st.doc.elements.filter((e) => e.kind === "LABEL" && e.label).map((e) => `V(${e.label})`));
    const volts = [...keys].filter((k) => labels.has(k));
    const stl = [...keys].filter((k) => /^I\(.+\.[dc]\)$/.test(k));
    const body = [...keys].filter((k) => /\.vb$/.test(k));
    const bits = [...keys].filter((k) => /\.bit$/.test(k)); // comparator outputs as logic traces
    traces = [...(volts.length ? volts : [...keys].filter((k) => k.startsWith("V(")).slice(0, 3)), ...body.slice(0, 2), ...stl, ...bits].slice(0, 8);
  }
  const t = result.runs[0]?.t ?? [];
  const tEnd = t[t.length - 1];
  const cur = st.cursorT;
  const t0 = t[0];
  // keep a cursor that is still inside the new time span; otherwise just after the first latch-up (run 0),
  // or the middle of the span
  let cursorT: number | null = null;
  if (typeof t0 === "number" && typeof tEnd === "number") {
    // inside the latched interval: a quarter of the way to the next event of run 0 (or +2 % of the span)
    const ev0 = result.events.filter((e) => e.run === 0).sort((a, b) => a.t - b.t);
    const k = ev0.findIndex((e) => e.kind === "latch_up");
    const span = tEnd - t0;
    const next = k >= 0 ? ev0[k + 1] : undefined;
    const after = k >= 0 ? ev0[k].t + (next ? 0.25 * (next.t - ev0[k].t) : 0.02 * span) : null;
    cursorT = cur != null && cur >= t0 && cur <= tEnd ? cur : after != null ? Math.min(tEnd, after) : t0 + span / 2;
  }
  st.set({ traces, cursorT, tool: st.tool.kind === "select" ? { kind: "probe" } : st.tool });
}

const esc = (x: string) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Element / wire ids named in a server error ("node 'b' has no DC path … capacitors C1, C2"). */
export function idsFromMessage(msg: string): string[] {
  const doc = useSch.getState().doc;
  const conn = extractNets(doc);
  const ids = new Set<string>();
  for (const e of doc.elements) {
    if (!e.name || e.kind === "GND" || e.kind === "LABEL") continue;
    if (new RegExp(`(^|[^A-Za-z0-9_])${esc(e.name)}([^A-Za-z0-9_]|$)`).test(msg)) ids.add(e.id);
  }
  for (const n of conn.nets) {
    if (n.ground) continue;
    if (new RegExp(`node\\s+['"]?${esc(n.name)}['"]?([^A-Za-z0-9_]|$)`, "i").test(msg)) {
      for (const w of n.wires) ids.add(w.id);
      for (const p of n.pins) ids.add(p.el.id);
    }
  }
  return [...ids];
}
