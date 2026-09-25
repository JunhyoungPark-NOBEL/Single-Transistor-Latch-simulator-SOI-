// Typed HTTP client for the FastAPI service (docs/WEB_CONTRACT.md §3) and the job runner used by
// the store: POST /api/compute/{kind}?wait=…, then poll GET /api/jobs/{id}, cancel with DELETE.
import type { Health, JobStatus, Kind, Meta } from "./types";
import { createRecorder } from "./snapshot";

/** Dev-only snapshot recorder (localStorage["stl-websim:record"] = "1"; scripts/record-snapshot.mjs). Null in builds. */
const recorder = import.meta.env.DEV ? createRecorder() : null;
const rec = <T,>(name: "health" | "meta" | "measured" | "design_map", p: Promise<T>): Promise<T> =>
  recorder ? p.then((v) => (recorder.data(name, v), v)) : p;

export interface Backend {
  readonly isMock: boolean;
  health(): Promise<Health>;
  meta(): Promise<Meta>;
  submit(kind: Kind, payload: unknown, wait?: number): Promise<JobStatus>;
  job(id: string): Promise<JobStatus>;
  cancel(id: string): Promise<void>;
  measured(): Promise<unknown>;
  designMap(): Promise<unknown>;
}

export class ApiError extends Error {
  status: number;
  /** Seconds from a Retry-After header (HTTP 429 "queue full"), when given. */
  retryAfter?: number;
  constructor(message: string, status: number, retryAfter?: number) {
    super(message);
    this.status = status;
    this.retryAfter = retryAfter;
    this.name = "ApiError";
  }
}

/** Retry-After header → seconds (delta-seconds or an HTTP date); undefined when absent/invalid. */
export function parseRetryAfter(h: string | null | undefined, now = Date.now()): number | undefined {
  if (!h) return undefined;
  const t = h.trim();
  if (/^\d+(\.\d+)?$/.test(t)) return Number(t);
  const d = Date.parse(t);
  return Number.isFinite(d) ? Math.max(0, (d - now) / 1000) : undefined;
}

/** Extracts a human-readable message from a FastAPI error body ({detail: string | [{msg, loc}]}). */
export function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object") {
    const d = (body as { detail?: unknown; error?: unknown; message?: unknown }).detail ??
      (body as { error?: unknown }).error ?? (body as { message?: unknown }).message;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d
        .map((e) => {
          if (e && typeof e === "object") {
            const loc = Array.isArray((e as { loc?: unknown }).loc) ? ((e as { loc: unknown[] }).loc.join(".") + ": ") : "";
            return loc + String((e as { msg?: unknown }).msg ?? JSON.stringify(e));
          }
          return String(e);
        })
        .join("; ");
    }
    if (d && typeof d === "object") return JSON.stringify(d);
  }
  return fallback;
}

async function request<T>(method: string, url: string, body?: unknown, timeoutMs = 30000): Promise<T> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: ctl.signal,
    });
    const text = await res.text();
    let data: unknown = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = text;
    }
    if (!res.ok) throw new ApiError(errorMessage(data, `${res.status} ${res.statusText || "HTTP error"}`), res.status, parseRetryAfter(res.headers.get("Retry-After")));
    return data as T;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    if ((e as Error)?.name === "AbortError") throw new ApiError("request timed out", 0);
    throw new ApiError((e as Error)?.message || "network error", 0);
  } finally {
    clearTimeout(timer);
  }
}

export const httpBackend: Backend = {
  isMock: false,
  health: () => rec("health", request<Health>("GET", "/api/health", undefined, 4000)),
  meta: () => rec("meta", request<Meta>("GET", "/api/meta")),
  submit: (kind, payload, wait = 1.5) => request<JobStatus>("POST", `/api/compute/${kind}?wait=${wait}`, payload, 60000),
  job: (id) => request<JobStatus>("GET", `/api/jobs/${encodeURIComponent(id)}`),
  cancel: async (id) => {
    await request<unknown>("DELETE", `/api/jobs/${encodeURIComponent(id)}`);
  },
  measured: () => rec("measured", request<unknown>("GET", "/api/data/measured", undefined, 60000)),
  designMap: () => rec("design_map", request<unknown>("GET", "/api/data/design_map", undefined, 60000)),
};

export class JobAborted extends Error {
  constructor() {
    super("aborted");
    this.name = "JobAborted";
  }
}

export interface RunOptions {
  onStatus?: (s: JobStatus) => void;
  /** HTTP 429 (server queue full): called with the wait in seconds before the single automatic retry. */
  onBusy?: (seconds: number) => void;
  /** Resolves true when the caller wants the job abandoned (checked between polls). */
  isAborted?: () => boolean;
  pollMs?: number;
  wait?: number;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Default / bounds of the wait before retrying a submit rejected with HTTP 429. */
export const BUSY_RETRY = { defaultS: 5, minS: 0.2, maxS: 60 };

/** Submit; on HTTP 429 (queue full) wait Retry-After (bounded) and retry exactly once. */
async function submitWithRetry(backend: Backend, kind: Kind, payload: unknown, opt: RunOptions): Promise<JobStatus> {
  try {
    return await backend.submit(kind, payload, opt.wait ?? 1.5);
  } catch (e) {
    if (!(e instanceof ApiError) || e.status !== 429) throw e;
    const waitS = Math.min(BUSY_RETRY.maxS, Math.max(BUSY_RETRY.minS, e.retryAfter ?? BUSY_RETRY.defaultS));
    opt.onBusy?.(waitS);
    const until = Date.now() + waitS * 1000;
    while (Date.now() < until) {
      if (opt.isAborted?.()) throw new JobAborted();
      await sleep(Math.min(200, until - Date.now()));
    }
    if (opt.isAborted?.()) throw new JobAborted();
    return backend.submit(kind, payload, opt.wait ?? 1.5); // a second 429 propagates to the caller
  }
}

/** Submit a job and poll until it finishes. Resolves the result, rejects with ApiError / JobAborted. */
export async function runJob<T>(backend: Backend, kind: Kind, payload: unknown, opt: RunOptions = {}): Promise<T> {
  if (!recorder) return runJobCore<T>(backend, kind, payload, opt);
  const end = recorder.begin();
  try {
    const result = await runJobCore<T>(backend, kind, payload, opt);
    if (backend === httpBackend) recorder.compute(kind, payload, result); // live results only
    return result;
  } finally {
    end();
  }
}

async function runJobCore<T>(backend: Backend, kind: Kind, payload: unknown, opt: RunOptions): Promise<T> {
  const pollMs = opt.pollMs ?? 400;
  let st = await submitWithRetry(backend, kind, payload, opt);
  opt.onStatus?.(st);
  let failures = 0;
  while (st.status === "queued" || st.status === "running") {
    if (opt.isAborted?.()) {
      backend.cancel(st.job_id).catch(() => undefined);
      throw new JobAborted();
    }
    await sleep(pollMs);
    if (opt.isAborted?.()) {
      backend.cancel(st.job_id).catch(() => undefined);
      throw new JobAborted();
    }
    try {
      st = await backend.job(st.job_id);
      failures = 0;
    } catch (e) {
      if (++failures > 5) throw e;
      continue;
    }
    opt.onStatus?.(st);
  }
  if (st.status === "done") return st.result as T;
  if (st.status === "cancelled") throw new JobAborted();
  throw new ApiError(st.error || st.message || "job failed", 500);
}
