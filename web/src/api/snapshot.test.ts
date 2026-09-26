import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { runJob, type Backend } from "./client";
import {
  canonicalJson, createSnapshotBackend, decodeBody, fileCandidates, isSnapshotFallback, loadSnapshot, nearDistance, nearInfo, sha256Hex,
  Snapshot, snapshotApprox, snapshotKey, type SnapshotIndex,
} from "./snapshot";
import type { JobStatus, Kind } from "./types";

// reference implementations: Web Crypto (secure-context API, present in Node) and CompressionStream
const refSha = async (s: string) =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s))), (x) => x.toString(16).padStart(2, "0")).join("");
const gzAsync = async (v: unknown) =>
  new Uint8Array(await new Response(new Blob([JSON.stringify(v)]).stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer());

describe("snapshot key", () => {
  it("canonical JSON sorts keys recursively and follows JSON semantics", () => {
    expect(canonicalJson({ b: 1, a: { d: [3, { z: 1, y: 2 }], c: "x" } })).toBe('{"a":{"c":"x","d":[3,{"y":2,"z":1}]},"b":1}');
    expect(canonicalJson({ a: undefined, b: NaN, c: [undefined, Infinity, -0], d: null })).toBe('{"b":null,"c":[null,null,0],"d":null}');
    expect(canonicalJson(undefined)).toBe("null");
    expect(canonicalJson("é")).toBe('"é"');
  });

  it("the key ignores key order and matches what JSON.stringify would send", async () => {
    const p1 = { device: { vg: -2, preset: "paper", light: { mode: "iph", iph_pA: 0 } }, sweep: { vd_max_V: 4.5, dv_V: 0.002 } };
    const p2 = { sweep: { dv_V: 0.002, vd_max_V: 4.5 }, device: { light: { iph_pA: 0, mode: "iph" }, preset: "paper", vg: -2, extra: undefined } };
    expect(snapshotKey("branches", p1)).toBe(snapshotKey("branches", p2));
    expect(snapshotKey("branches", p1)).toBe(snapshotKey("branches", JSON.parse(JSON.stringify(p1))));
    expect(snapshotKey("branches", p1)).not.toBe(snapshotKey("hazard", p1));
    expect(snapshotKey("branches", p1)).not.toBe(snapshotKey("branches", { ...p1, sweep: { ...p1.sweep, vd_max_V: 4.6 } }));
    expect(snapshotKey("branches", p1)).toBe(await refSha(canonicalJson({ kind: "branches", payload: p1 })));
    expect(snapshotKey("branches", p1)).toMatch(/^[0-9a-f]{64}$/);
  });

  it("plain-JS SHA-256 matches Web Crypto (block boundaries, unicode, long input)", async () => {
    const cases = ["", "abc", "a".repeat(55), "a".repeat(56), "a".repeat(63), "a".repeat(64), "a".repeat(65), "래치업 V_LU ±σ — µs", "x".repeat(100_003)];
    for (const c of cases) expect(sha256Hex(c)).toBe(await refSha(c));
    expect(sha256Hex("abc")).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  });
});

// ---------------------------------------------------------------- files
type Files = Record<string, Uint8Array | string>;
function fakeFetch(files: Files, log: string[] = []) {
  return async (url: string): Promise<Response> => {
    log.push(url);
    const body = files[url];
    if (body === undefined) return new Response("not found", { status: 404 });
    return new Response(typeof body === "string" ? body : new Blob([body as BlobPart]));
  };
}
const g = globalThis as { DecompressionStream?: unknown };
const realDS = g.DecompressionStream;
afterEach(() => {
  g.DecompressionStream = realDS;
});

describe("snapshot files (gzip / plain fallback)", () => {
  it("file candidates: gzip first when decodable, then the plain copy", () => {
    expect(fileCandidates({ file: "k.json.gz", plain: "k.json" }, true)).toEqual(["k.json.gz", "k.json"]);
    expect(fileCandidates({ file: "k.json.gz", plain: "k.json" }, false)).toEqual(["k.json"]);
    expect(fileCandidates({ file: "k.json.gz" }, false)).toEqual([]);
    expect(fileCandidates({ file: "k.json" }, false)).toEqual(["k.json"]);
  });

  it("decodes gzip bytes, and passes through bytes a host already inflated", async () => {
    expect(JSON.parse(await decodeBody(await gzAsync({ a: [1, 2] })))).toEqual({ a: [1, 2] });
    expect(JSON.parse(await decodeBody(new TextEncoder().encode('{"b":1}')))).toEqual({ b: 1 });
  });

  const index: SnapshotIndex = {
    format: 1,
    created: "2026-09-25T00:00:00Z",
    health: { ok: true, version: "abc" },
    data: { measured: { file: "measured.json.gz", plain: "measured.json", bytes: 1 } },
    compute: [{ kind: "branches", key: snapshotKey("branches", { x: 1 }), label: "t", file: "k1.json.gz", plain: "k1.json", bytes: 1 }],
  };
  const files: Files = {
    "s/index.json": JSON.stringify(index),
    "s/k1.json": JSON.stringify({ folds: { V_LU: 3.7 } }),
    "s/measured.json": JSON.stringify({ photo: 1 }),
  };
  beforeAll(async () => {
    files["s/k1.json.gz"] = await gzAsync({ folds: { V_LU: 3.7 } });
    files["s/measured.json.gz"] = await gzAsync({ photo: 1 });
  });

  it("uses the .json.gz file when DecompressionStream exists", async () => {
    const log: string[] = [];
    const snap = (await loadSnapshot("s/", fakeFetch(files, log)))!;
    expect(snap.size).toBe(1);
    expect(await snap.lookup("branches", { x: 1 })).toEqual({ folds: { V_LU: 3.7 } });
    expect(await snap.data("measured")).toEqual({ photo: 1 });
    expect(log).toContain("s/k1.json.gz");
    expect(log).not.toContain("s/k1.json");
    expect(await snap.lookup("branches", { x: 2 })).toBeUndefined();
    expect(await snap.lookup("hazard", { x: 1 })).toBeUndefined();
  });

  it("falls back to the plain .json copy without DecompressionStream", async () => {
    g.DecompressionStream = undefined;
    const log: string[] = [];
    const snap = (await loadSnapshot("s/", fakeFetch(files, log)))!;
    expect(await snap.lookup("branches", { x: 1 })).toEqual({ folds: { V_LU: 3.7 } });
    expect(log).toEqual(["s/index.json", "s/k1.json"]);
  });

  it("falls back to the plain copy when the gzip file is missing; a gzip-only entry is a miss without DecompressionStream", async () => {
    const { ["s/k1.json.gz"]: _drop, ...noGz } = files;
    void _drop;
    const snap = (await loadSnapshot("s/", fakeFetch(noGz)))!;
    expect(await snap.lookup("branches", { x: 1 })).toEqual({ folds: { V_LU: 3.7 } });
    g.DecompressionStream = undefined;
    const gzOnly = new Snapshot({ ...index, compute: [{ ...index.compute[0], plain: undefined }] }, "s/", fakeFetch(files));
    expect(await gzOnly.lookup("branches", { x: 1 })).toBeUndefined();
  });

  it("no snapshot: 404 or an HTML page (SPA fallback) → null", async () => {
    expect(await loadSnapshot("s/", fakeFetch({}))).toBeNull();
    expect(await loadSnapshot("s/", fakeFetch({ "s/index.json": "<!doctype html><html></html>" }))).toBeNull();
    expect(await loadSnapshot("s/", fakeFetch({ "s/index.json": '{"format":99,"compute":[],"data":{}}' }))).toBeNull();
  });

  it("backend: hits come from the snapshot, misses from the fallback and are marked", async () => {
    const snap = (await loadSnapshot("s/", fakeFetch(files)))!;
    let n = 0;
    const fallback: Backend = {
      isMock: true,
      health: async () => ({ ok: true }),
      meta: async () => ({ presets: {} }) as never,
      submit: async (kind: Kind): Promise<JobStatus> => ({ job_id: `m${++n}`, kind, status: "running", progress: 0.5, message: "", cached: false, elapsed_s: 0 }),
      job: async (id: string): Promise<JobStatus> => ({ job_id: id, kind: "branches", status: "done", progress: 1, message: "", result: { demo: true }, cached: false, elapsed_s: 0 }),
      cancel: async () => undefined,
      measured: async () => ({}),
      designMap: async () => ({}),
    };
    const be = createSnapshotBackend(snap, { fallback: () => fallback });
    expect(be.isMock).toBe(false);
    const hit = await runJob<{ folds: { V_LU: number } }>(be, "branches", { x: 1 }, { pollMs: 1 });
    expect(hit.folds.V_LU).toBe(3.7);
    expect(isSnapshotFallback(hit)).toBe(false);
    const miss = await runJob<{ demo: boolean }>(be, "branches", { x: 2 }, { pollMs: 1 });
    expect(miss).toEqual({ demo: true });
    expect(isSnapshotFallback(miss)).toBe(true);
    expect((await be.health()).ok).toBe(true);
    expect(await be.measured()).toEqual({ photo: 1 });
    await expect(be.designMap()).rejects.toThrow(/not in the static snapshot/);
  });
});

describe("nearest V_G / optical power", () => {
  const dev = (vg: number, extra: Record<string, unknown> = {}) => ({
    device: { preset: "paper", vg, light: { mode: "iph", iph_pA: 0, power_mW: 0, responsivity_pA_per_mW: 0.75 }, ...extra },
    sweep: { vd_max_V: 4 },
  });
  const photo = (vg: number, power_mW: number) => ({
    device: { preset: "photo", vg, light: { mode: "power", iph_pA: 0, power_mW, responsivity_pA_per_mW: 0.75 } },
    sweep: { vd_max_V: 5 },
  });

  it("the signature ignores only V_G (and the optical power in power mode)", () => {
    const a = nearInfo("branches", dev(-2))!;
    expect(a.vg).toBe(-2);
    expect(a.p).toBeNull();
    expect(nearInfo("branches", dev(-1.5))!.sig).toBe(a.sig);
    expect(nearInfo("branches", dev(-1.5, { preset: "custom" }))!.sig).not.toBe(a.sig);
    expect(nearInfo("hazard", dev(-2))).toBeNull(); // stochastic kinds never snap
    expect(nearInfo("charge_balance", { ...dev(-2), vd: 3.1 })!.sig).not.toBe(nearInfo("charge_balance", { ...dev(-2), vd: 3.2 })!.sig);
    const p = nearInfo("branches", photo(-1.8, 1.15))!;
    expect(p.p).toBe(1.15);
    expect(nearInfo("branches", photo(-1.1, 3.51))!.sig).toBe(p.sig);
    // iph mode: the power field is not a coordinate, a change is a different request
    const iph = { device: { ...dev(-2).device, light: { mode: "iph", iph_pA: 0, power_mW: 1, responsivity_pA_per_mW: 0.75 } }, sweep: { vd_max_V: 4 } };
    expect(nearInfo("branches", iph)!.sig).not.toBe(a.sig);
  });

  it("resolve: exact hit, nearest point (marked copy), out of reach → undefined; bundles by part", async () => {
    const entry = (vg: number, file: string, part?: string) => ({
      kind: "branches", key: snapshotKey("branches", dev(vg)), label: `vg ${vg}`, file, part, bytes: 1, near: nearInfo("branches", dev(vg))!,
    });
    const idx: SnapshotIndex = {
      format: 1, created: "x", data: {},
      compute: [entry(-2, "a.json"), entry(-1.9, "grid.json", "k19"), entry(-1.8, "grid.json", "k18")],
    };
    const log: string[] = [];
    const snap = new Snapshot(idx, "s/", fakeFetch({ "s/a.json": '{"v":-2}', "s/grid.json": '{"k19":{"v":-1.9},"k18":{"v":-1.8}}' }, log));
    const exact = await snap.resolve("branches", dev(-2));
    expect(exact).toEqual({ v: -2 });
    expect(snapshotApprox(exact)).toBeUndefined();
    const near = await snap.resolve("branches", dev(-1.93));
    expect(near).toEqual({ v: -1.9 });
    expect(snapshotApprox(near)).toEqual({ vg: -1.9, power_mW: null, requested: { vg: -1.93, power_mW: null } });
    const exact2 = await snap.resolve("branches", dev(-1.9));
    expect(snapshotApprox(exact2)).toBeUndefined(); // the recorded object itself stays unmarked
    expect(await snap.resolve("branches", dev(-1.8 + 1e-12))).toEqual({ v: -1.8 }); // float noise = same point
    expect(snapshotApprox(await snap.resolve("branches", dev(-1.8 + 1e-12)))).toBeUndefined();
    expect(await snap.resolve("branches", dev(-3))).toBeUndefined(); // beyond NEAR_MAX.vg
    expect(await snap.resolve("branches", dev(-1.9, { preset: "custom" }))).toBeUndefined();
    expect(log.filter((u) => u === "s/grid.json")).toHaveLength(1); // one fetch per bundle
  });

  it("nearDistance weighs V_G in 0.1 V and power in 1 mW steps", () => {
    expect(nearDistance({ vg: -2, p: null }, { vg: -1.9, p: null })).toBeCloseTo(1);
    expect(nearDistance({ vg: -1.8, p: 2 }, { vg: -1.8, p: 2.55 })).toBeCloseTo(0.55);
    expect(nearDistance({ vg: -1.8, p: 2 }, { vg: -1.8, p: null })).toBe(Infinity);
    expect(nearDistance({ vg: -1.8, p: 5 }, { vg: -1.8, p: 3.51 })).toBe(Infinity);
  });

  it("strict production replay never substitutes a nearby condition or a demo result", async () => {
    const payload = dev(-2);
    const snap = new Snapshot({ format: 1, created: "x", data: {}, compute: [
      { kind: "branches", key: snapshotKey("branches", payload), label: "reference", file: "a.json", bytes: 1, near: nearInfo("branches", payload)! },
    ] }, "s/", fakeFetch({ "s/a.json": '{"v":-2}' }));
    const backend = createSnapshotBackend(snap, { exactOnly: true, fallback: () => { throw new Error("demo must not be reached"); } });
    expect((await backend.submit("branches", payload)).result).toEqual({ v: -2 });
    await expect(backend.submit("branches", dev(-1.99))).rejects.toThrow("snapshot-missing");
  });
});
