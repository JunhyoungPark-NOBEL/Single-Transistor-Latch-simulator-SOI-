import { describe, expect, it } from "vitest";
import { ApiError, BUSY_RETRY, errorMessage, JobAborted, parseRetryAfter, runJob, type Backend } from "./client";
import type { JobStatus } from "./types";

const done = (result: unknown): JobStatus => ({ job_id: "j1", kind: "branches", status: "done", progress: 1, message: "", result, cached: false, elapsed_s: 0 });

function fakeBackend(submits: (() => JobStatus)[]): Backend & { calls: number } {
  const b = {
    isMock: true,
    calls: 0,
    health: async () => ({ ok: true }),
    meta: async () => ({}) as never,
    submit: async () => {
      const f = submits[Math.min(b.calls, submits.length - 1)];
      b.calls++;
      return f();
    },
    job: async () => done(null),
    cancel: async () => undefined,
    measured: async () => ({}),
    designMap: async () => ({}),
  };
  return b;
}
const busy = (retryAfter?: number) => () => {
  throw new ApiError("job queue full", 429, retryAfter);
};

describe("HTTP client", () => {
  it("error bodies {detail: string} become the message", () => {
    expect(errorMessage({ detail: "vd_max_V must be <= 8" }, "fallback")).toBe("vd_max_V must be <= 8");
    expect(errorMessage("<html>", "500 Internal Server Error")).toBe("500 Internal Server Error");
  });
  it("parses Retry-After (seconds or HTTP date)", () => {
    expect(parseRetryAfter("10")).toBe(10);
    expect(parseRetryAfter(null)).toBeUndefined();
    const now = Date.parse("2026-09-24T12:00:00Z");
    expect(parseRetryAfter("Thu, 24 Sep 2026 12:00:05 GMT", now)).toBe(5);
    expect(parseRetryAfter("soon")).toBeUndefined();
  });
  it("429: waits Retry-After, reports it once and retries exactly once", async () => {
    const b = fakeBackend([busy(0.05), () => done({ ok: 1 })]);
    const waits: number[] = [];
    const r = await runJob<{ ok: number }>(b, "branches", {}, { onBusy: (s) => waits.push(s) });
    expect(r).toEqual({ ok: 1 });
    expect(b.calls).toBe(2);
    expect(waits).toEqual([Math.max(BUSY_RETRY.minS, 0.05)]);
  });
  it("429 twice: the second rejection propagates (no retry loop)", async () => {
    const b = fakeBackend([busy(0.01), busy(0.01), () => done(1)]);
    await expect(runJob(b, "branches", {}, {})).rejects.toMatchObject({ status: 429 });
    expect(b.calls).toBe(2);
  });
  it("cancelling while waiting for the retry aborts without resubmitting", async () => {
    const b = fakeBackend([busy(5), () => done(1)]);
    let aborted = false;
    const p = runJob(b, "branches", {}, { onBusy: () => setTimeout(() => (aborted = true), 20), isAborted: () => aborted });
    await expect(p).rejects.toBeInstanceOf(JobAborted);
    expect(b.calls).toBe(1);
  });
});
