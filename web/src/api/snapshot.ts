// Static snapshot backend: precomputed model results for the built-in presets and examples, served as static
// files next to the page (web/snapshot/, copied into the artifact build by scripts/build-artifact.mjs).
//
// When /api/health fails, the runner tries `snapshot/index.json` (relative to the page). If it exists the app
// runs in "snapshot" mode: every compute request is looked up by
//   key = SHA-256( canonicalJson({ kind, payload }) )        (sorted keys, JSON semantics, hex digest)
// and the recorded result is returned; requests that are not in the snapshot fall back to the demo (mock)
// backend and the result is marked (`isSnapshotFallback`) so the panel says it is example data. Files are
// gzip JSON (`<key>.json.gz`, decoded with DecompressionStream) with an optional plain `.json` copy that is
// used when DecompressionStream is unavailable. The recorder hooks at the end of this file are dev-only
// (localStorage["stl-websim:record"] = "1") and feed scripts/record-snapshot.mjs.
//
// Locked artifact build (`build:artifact -- --lock`): every snapshot file, the index included (`snapshot/index.wasm`),
// is encrypted ("STLENC1\0" · 12-byte IV · AES-256-GCM ciphertext+tag, AAD = the published path, see
// scripts/lock-crypto.mjs). The password gate (scripts/lock/lock.js) leaves the key in globalThis.__STL_LOCK__
// before it starts the app; `readSnapshotBytes` decrypts, gunzips and parses every file. Plain builds and dev mode
// never see encrypted bytes and behave as before.
import type { Backend } from "./client";
import type { Health, JobStatus, Kind, Meta } from "./types";
import { GEOMETRY_LIVE_REQUIRED, hasChangedGeometry, legacyGeometryPayload } from "./geometryPolicy";

export const SNAPSHOT_FORMAT = 1;
/** Folder of the snapshot relative to the page (document base URL). */
export const SNAPSHOT_BASE = "snapshot/";

// ---------------------------------------------------------------- key
/** Canonical JSON with JSON semantics (undefined dropped / null in arrays, NaN/±inf → null) and sorted keys. */
export function canonicalJson(v: unknown): string {
  if (v === undefined) return "null";
  return stable(JSON.parse(JSON.stringify(v)) as unknown);
}
function stable(v: unknown): string {
  if (v === null || typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(stable).join(",")}]`;
  const o = v as Record<string, unknown>;
  return `{${Object.keys(o)
    .sort()
    .map((k) => `${JSON.stringify(k)}:${stable(o[k])}`)
    .join(",")}}`;
}

// SHA-256 (FIPS 180-4) in plain JS: synchronous and independent of crypto.subtle (which needs a secure context).
const K256 = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
  0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
  0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);
const rotr = (x: number, n: number) => (x >>> n) | (x << (32 - n));

export function sha256(msg: Uint8Array): Uint8Array {
  const len = msg.length;
  const buf = new Uint8Array(((len + 9 + 63) >> 6) << 6);
  buf.set(msg);
  buf[len] = 0x80;
  const dv = new DataView(buf.buffer);
  dv.setUint32(buf.length - 8, Math.floor(len / 0x20000000)); // bit length, high word
  dv.setUint32(buf.length - 4, (len << 3) >>> 0);
  const H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const W = new Uint32Array(64);
  for (let off = 0; off < buf.length; off += 64) {
    for (let i = 0; i < 16; i++) W[i] = dv.getUint32(off + 4 * i);
    for (let i = 16; i < 64; i++) {
      const a = W[i - 15];
      const b = W[i - 2];
      W[i] = (W[i - 16] + (rotr(a, 7) ^ rotr(a, 18) ^ (a >>> 3)) + W[i - 7] + (rotr(b, 17) ^ rotr(b, 19) ^ (b >>> 10))) >>> 0;
    }
    let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (let i = 0; i < 64; i++) {
      const t1 = (h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + W[i]) >>> 0;
      const t2 = ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + t1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) >>> 0;
    }
    H[0] = (H[0] + a) >>> 0;
    H[1] = (H[1] + b) >>> 0;
    H[2] = (H[2] + c) >>> 0;
    H[3] = (H[3] + d) >>> 0;
    H[4] = (H[4] + e) >>> 0;
    H[5] = (H[5] + f) >>> 0;
    H[6] = (H[6] + g) >>> 0;
    H[7] = (H[7] + h) >>> 0;
  }
  const out = new Uint8Array(32);
  const ov = new DataView(out.buffer);
  for (let i = 0; i < 8; i++) ov.setUint32(4 * i, H[i]);
  return out;
}

export function sha256Hex(text: string): string {
  return Array.from(sha256(new TextEncoder().encode(text)), (x) => x.toString(16).padStart(2, "0")).join("");
}

/** Snapshot key of a compute request: SHA-256 of the canonical JSON of {kind, payload}. */
export function snapshotKey(kind: string, payload: unknown): string {
  return sha256Hex(canonicalJson({ kind, payload }));
}

// ---------------------------------------------------------------- nearest V_G / optical power
// Deterministic device results are also recorded on a V_G grid (and a small optical-power grid for the
// illumination preset). A request that differs from a recorded one ONLY in device.vg (and, in power mode, in
// device.light.power_mW) is answered with the nearest recorded point and marked (`snapshotApprox`).
export const NEAR_KINDS: readonly string[] = ["branches", "charge_balance", "vg_curve"];
/** Largest distance that is still answered with the nearest point (beyond: demo data). */
export const NEAR_MAX = { vg: 0.25, power_mW: 0.8 };
export interface NearInfo {
  /** SHA-256 of the canonical request with V_G (and the optical power in power mode) removed. */
  sig: string;
  vg: number;
  /** Optical power in mW when the light block is in power mode, else null. */
  p: number | null;
}

export function nearInfo(kind: string, payload: unknown): NearInfo | null {
  if (!NEAR_KINDS.includes(kind) || !payload || typeof payload !== "object") return null;
  const copy = JSON.parse(canonicalJson(payload)) as { device?: { vg?: unknown; light?: { mode?: unknown; power_mW?: unknown } } };
  const d = copy.device;
  if (!d || typeof d.vg !== "number") return null;
  const vg = d.vg;
  delete d.vg;
  let p: number | null = null;
  if (d.light && d.light.mode === "power" && typeof d.light.power_mW === "number") {
    p = d.light.power_mW;
    delete d.light.power_mW;
  }
  return { sig: sha256Hex(canonicalJson({ kind, payload: copy })), vg, p };
}

/** Distance in grid units (0.1 V for V_G, 1 mW for the power); Infinity when out of reach. */
export function nearDistance(want: Pick<NearInfo, "vg" | "p">, have: Pick<NearInfo, "vg" | "p">): number {
  const dvg = Math.abs(want.vg - have.vg);
  const dp = want.p === null || have.p === null ? (want.p === have.p ? 0 : Infinity) : Math.abs(want.p - have.p);
  if (!(dvg <= NEAR_MAX.vg + 1e-9) || !(dp <= NEAR_MAX.power_mW + 1e-9)) return Infinity;
  return dvg / 0.1 + dp / 1;
}

/** Marker of a result that answers a request with the nearest recorded V_G / optical power. */
export interface SnapshotApprox {
  vg: number;
  power_mW: number | null;
  requested: { vg: number; power_mW: number | null };
}

// ---------------------------------------------------------------- index + files
export interface SnapshotFileRef {
  /** Path relative to the snapshot folder (normally `<name>.json.gz`). */
  file: string;
  /** Optional plain JSON copy, used when DecompressionStream is unavailable. */
  plain?: string;
  /** Size of `file` in bytes. */
  bytes: number;
  /** Uncompressed JSON size in bytes. */
  json_bytes?: number;
}
export interface SnapshotComputeEntry extends SnapshotFileRef {
  kind: string;
  key: string;
  label: string;
  /** Key of this result inside `file` when the file bundles several results ({key: result}). */
  part?: string;
  /** For NEAR_KINDS: signature and coordinates for the nearest-V_G lookup. */
  near?: NearInfo;
}
export interface SnapshotIndex {
  format: number;
  created: string;
  source?: string;
  health?: Health & Record<string, unknown>;
  meta?: Meta;
  data: Partial<Record<"measured" | "design_map", SnapshotFileRef>>;
  compute: SnapshotComputeEntry[];
}

export function isSnapshotIndex(v: unknown): v is SnapshotIndex {
  if (!v || typeof v !== "object") return false;
  const o = v as Record<string, unknown>;
  return o.format === SNAPSHOT_FORMAT && Array.isArray(o.compute) && !!o.data && typeof o.data === "object";
}

const hasDecompressionStream = () => typeof (globalThis as { DecompressionStream?: unknown }).DecompressionStream === "function";

/** Files to try for one entry, in order: the gzip file when it can be decoded, then the plain copy. */
export function fileCandidates(ref: Pick<SnapshotFileRef, "file" | "plain">, canGunzip = hasDecompressionStream()): string[] {
  const out: string[] = [];
  const gz = /\.gz$/i.test(ref.file);
  if (!gz || canGunzip) out.push(ref.file);
  if (ref.plain && !out.includes(ref.plain)) out.push(ref.plain);
  return out;
}

async function gunzip(bytes: Uint8Array): Promise<string> {
  const DS = (globalThis as unknown as { DecompressionStream: new (f: string) => TransformStream<Uint8Array, Uint8Array> }).DecompressionStream;
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DS("gzip"));
  return new Response(stream).text();
}

/** Body bytes → JSON text: gzip (magic 1f 8b) is inflated, anything else is read as UTF-8 (a host may already
 *  have removed the gzip layer by serving the file with Content-Encoding: gzip). */
export async function decodeBody(bytes: Uint8Array): Promise<string> {
  if (bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b) {
    if (!hasDecompressionStream()) throw new Error("gzip snapshot file but DecompressionStream is unavailable");
    return gunzip(bytes);
  }
  return new TextDecoder().decode(bytes);
}

// ---------------------------------------------------------------- locked build (encrypted files)
/** "STLENC1\0": first bytes of an encrypted file of the locked build. */
export const LOCK_MAGIC: readonly number[] = [0x53, 0x54, 0x4c, 0x45, 0x4e, 0x43, 0x31, 0x00];
const LOCK_IV = 12;
const LOCK_TAG = 16;
/** Index file of a locked build (in place of index.json). */
export const LOCKED_INDEX = "index.wasm";

/** Key left by the password gate (scripts/lock/lock.js), or null (plain build / dev). */
export function lockKey(): CryptoKey | null {
  const k = (globalThis as { __STL_LOCK__?: { key?: unknown } }).__STL_LOCK__?.key;
  return k && typeof k === "object" ? (k as CryptoKey) : null;
}

export function isEncrypted(bytes: Uint8Array): boolean {
  return bytes.length >= LOCK_MAGIC.length + LOCK_IV + LOCK_TAG && LOCK_MAGIC.every((x, i) => bytes[i] === x);
}

/** AAD of an encrypted file = its published path relative to the page, without a leading "./" — the same string
 *  the build encrypted it with ("snapshot/index.wasm", "snapshot/<name>.wasm"). */
export function lockAad(url: string): string {
  return url.replace(/^(\.\/)+/, "");
}

/** Decrypt one file of the locked build; rejects on a wrong key, another path (AAD) or any modified byte. */
export async function decryptLocked(bytes: Uint8Array, url: string, key: CryptoKey): Promise<Uint8Array> {
  if (!isEncrypted(bytes)) throw new Error(`${url} is not an encrypted file`);
  const params: AesGcmParams = {
    name: "AES-GCM",
    iv: bytes.slice(LOCK_MAGIC.length, LOCK_MAGIC.length + LOCK_IV),
    additionalData: new TextEncoder().encode(lockAad(url)),
    tagLength: LOCK_TAG * 8,
  };
  return new Uint8Array(await globalThis.crypto.subtle.decrypt(params, key, bytes.slice(LOCK_MAGIC.length + LOCK_IV)));
}

/** The one reader of snapshot bytes (index and every result file): decrypt when the bytes are encrypted (locked
 *  build, key from the gate), gunzip when they are gzip, then JSON.parse. `url` is the path the file was fetched
 *  from, relative to the page (it is the AAD). */
export async function readSnapshotBytes(bytes: Uint8Array, url: string, key: CryptoKey | null = lockKey()): Promise<unknown> {
  let b = bytes;
  if (isEncrypted(b)) {
    if (!key) throw new Error(`${url} is encrypted and the page is not unlocked`);
    b = await decryptLocked(b, url, key);
  }
  return JSON.parse(await decodeBody(b)) as unknown;
}

type Fetch = (url: string, init?: RequestInit) => Promise<Response>;

export class Snapshot {
  readonly index: SnapshotIndex;
  private readonly base: string;
  private readonly fetchFn: Fetch;
  private readonly byKey = new Map<string, SnapshotComputeEntry>();
  private readonly bySig = new Map<string, SnapshotComputeEntry[]>();
  private readonly cache = new Map<string, Promise<unknown>>();

  constructor(index: SnapshotIndex, base = SNAPSHOT_BASE, fetchFn: Fetch = (u, i) => fetch(u, i)) {
    this.index = index;
    this.base = base;
    this.fetchFn = fetchFn;
    for (const e of index.compute) {
      if (!e || typeof e.key !== "string" || typeof e.file !== "string") continue;
      this.byKey.set(e.key, e);
      if (e.near && typeof e.near.sig === "string" && typeof e.near.vg === "number") {
        const list = this.bySig.get(e.near.sig) ?? [];
        list.push(e);
        this.bySig.set(e.near.sig, list);
      }
    }
  }

  get size(): number {
    return this.byKey.size;
  }

  has(kind: string, payload: unknown): boolean {
    return this.byKey.has(snapshotKey(kind, payload));
  }

  /** Read one file reference (gzip first when decodable, then the plain copy). Rejects when none works. */
  async read(ref: Pick<SnapshotFileRef, "file" | "plain">): Promise<unknown> {
    const tries = fileCandidates(ref);
    if (!tries.length) throw new Error("snapshot file is gzip and this browser cannot decode it");
    let last: unknown = null;
    for (const f of tries) {
      try {
        const url = this.base + f;
        const res = await this.fetchFn(url);
        if (!res.ok) throw new Error(`${res.status} ${f}`);
        return await readSnapshotBytes(new Uint8Array(await res.arrayBuffer()), url);
      } catch (e) {
        last = e;
      }
    }
    throw last instanceof Error ? last : new Error(String(last));
  }

  /** File contents, fetched once per file (bundles hold several results). */
  private cached(ref: Pick<SnapshotFileRef, "file" | "plain">): Promise<unknown> {
    let p = this.cache.get(ref.file);
    if (!p) {
      p = this.read(ref);
      p.catch(() => this.cache.delete(ref.file)); // a failed read may be retried
      this.cache.set(ref.file, p);
    }
    return p;
  }

  private async entryResult(e: SnapshotComputeEntry): Promise<unknown> {
    const v = await this.cached(e);
    if (e.part === undefined) return v;
    const r = v && typeof v === "object" ? (v as Record<string, unknown>)[e.part] : undefined;
    if (r === undefined) throw new Error(`${e.part} missing in ${e.file}`);
    return r;
  }

  /** Recorded result of exactly this request, or `undefined` when it is not in the snapshot / unreadable. */
  async lookup(kind: string, payload: unknown): Promise<unknown> {
    const e = this.byKey.get(snapshotKey(kind, payload));
    if (!e || e.kind !== kind) return undefined;
    try {
      return await this.entryResult(e);
    } catch {
      return undefined;
    }
  }

  /** Recorded entry nearest in V_G (and optical power) to a request that differs only in those. */
  nearest(kind: string, payload: unknown): { entry: SnapshotComputeEntry; want: NearInfo } | null {
    const want = nearInfo(kind, payload);
    const list = want && this.bySig.get(want.sig);
    if (!want || !list) return null;
    let best: SnapshotComputeEntry | null = null;
    let bestD = Infinity;
    for (const e of list) {
      const d = e.kind === kind && e.near ? nearDistance(want, e.near) : Infinity;
      if (d < bestD) {
        bestD = d;
        best = e;
      }
    }
    return best ? { entry: best, want } : null;
  }

  /** Exact result, else the nearest-V_G result (a marked shallow copy), else `undefined`. */
  async resolve(kind: string, payload: unknown): Promise<unknown> {
    const exact = await this.lookup(kind, payload);
    if (exact !== undefined) return exact;
    const n = this.nearest(kind, payload);
    if (!n?.entry.near) return undefined;
    let r: unknown;
    try {
      r = await this.entryResult(n.entry);
    } catch {
      return undefined;
    }
    if (!r || typeof r !== "object" || Array.isArray(r)) return undefined;
    const have = n.entry.near;
    const same = Math.abs(have.vg - n.want.vg) < 1e-6 && (have.p === null || n.want.p === null || Math.abs(have.p - n.want.p) < 1e-6);
    if (same) return r; // the same point up to float noise (e.g. a slider value)
    const copy = { ...(r as Record<string, unknown>) };
    approxResults.set(copy, { vg: have.vg, power_mW: have.p, requested: { vg: n.want.vg, power_mW: n.want.p } });
    return copy;
  }

  async data(name: "measured" | "design_map"): Promise<unknown> {
    const ref = this.index.data[name];
    if (!ref) throw new Error(`${name} is not in the static snapshot`);
    return this.cached(ref);
  }
}

/** Fetch and validate `snapshot/index.json` (`snapshot/index.wasm` on an unlocked locked build); null when absent
 *  or not a snapshot (e.g. an SPA HTML fallback) or when it cannot be decrypted. */
export async function loadSnapshot(base = SNAPSHOT_BASE, fetchFn?: Fetch): Promise<Snapshot | null> {
  const f: Fetch = fetchFn ?? ((u, i) => fetch(u, i));
  try {
    const url = `${base}${lockKey() ? LOCKED_INDEX : "index.json"}`;
    const res = await f(url, { cache: "no-cache" });
    if (!res.ok) return null;
    const idx = await readSnapshotBytes(new Uint8Array(await res.arrayBuffer()), url);
    return isSnapshotIndex(idx) ? new Snapshot(idx, base, f) : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- backend
const approxResults = new WeakMap<object, SnapshotApprox>();
/** Nearest-point marker of a result (see `Snapshot.resolve`), or undefined for exact / live / demo results. */
export function snapshotApprox(result: unknown): SnapshotApprox | undefined {
  return result && typeof result === "object" ? approxResults.get(result) : undefined;
}

const fallbackResults = new WeakSet<object>();
/** True for a result that came from the demo fallback because the request was not in the snapshot. */
export function isSnapshotFallback(result: unknown): boolean {
  return !!result && typeof result === "object" && fallbackResults.has(result);
}

const FB = "fb:";

/** Backend that answers from the snapshot and delegates everything else to `fallback` (the demo backend). */
export function createSnapshotBackend(snap: Snapshot, opt: { fallback: () => Backend }): Backend {
  let fb: Backend | null = null;
  const fallback = () => (fb ??= opt.fallback());
  let seq = 0;
  const wrap = (st: JobStatus): JobStatus => {
    if (st.status === "done" && st.result && typeof st.result === "object") fallbackResults.add(st.result);
    return { ...st, job_id: FB + st.job_id };
  };
  return {
    isMock: false,
    health: async () => ({
      ...(snap.index.health ?? {}),
      ok: true,
      workers: 0,
      version: String(snap.index.health?.version ?? "snapshot"),
      snapshot: { n: snap.size, created: snap.index.created },
    }),
    meta: async () => snap.index.meta ?? fallback().meta(),
    submit: async (kind: Kind, payload: unknown) => {
      let result = await snap.resolve(kind, payload);
      if (result === undefined && !hasChangedGeometry(payload)) {
        const legacy = legacyGeometryPayload(payload);
        if (canonicalJson(legacy) !== canonicalJson(payload)) result = await snap.resolve(kind, legacy);
      }
      if (result !== undefined) {
        return { job_id: `snap-${++seq}`, kind, status: "done", progress: 1, message: "snapshot", result, cached: true, elapsed_s: 0 };
      }
      if (hasChangedGeometry(payload)) throw new Error(GEOMETRY_LIVE_REQUIRED);
      return wrap(await fallback().submit(kind, payload));
    },
    job: async (id: string) => {
      if (id.startsWith(FB)) return wrap(await fallback().job(id.slice(FB.length)));
      throw new Error(`unknown job ${id}`);
    },
    cancel: async (id: string) => {
      if (id.startsWith(FB)) await fallback().cancel(id.slice(FB.length));
    },
    measured: () => snap.data("measured"),
    designMap: () => snap.data("design_map"),
  };
}

// ---------------------------------------------------------------- recorder (dev only)
// Enabled by localStorage["stl-websim:record"] = "1" in `vite dev` (client.ts only calls this under
// import.meta.env.DEV, so production builds drop it). scripts/record-snapshot.mjs drains window.__stlRecords.
export const RECORD_FLAG = "stl-websim:record";
export type SnapshotRecord =
  | { type: "compute"; kind: string; key: string; near: NearInfo | null; payload: unknown; result: unknown }
  | { type: "data"; name: "health" | "meta" | "measured" | "design_map"; value: unknown };
interface RecorderWindow {
  __stlRecords?: SnapshotRecord[];
  __stlInflight?: number;
}
export interface Recorder {
  begin(): () => void;
  compute(kind: string, payload: unknown, result: unknown): void;
  data(name: "health" | "meta" | "measured" | "design_map", value: unknown): void;
}

export function createRecorder(): Recorder | null {
  let on = false;
  try {
    on = typeof window !== "undefined" && window.localStorage?.getItem(RECORD_FLAG) === "1";
  } catch {
    on = false;
  }
  if (!on) return null;
  const w = window as unknown as RecorderWindow;
  w.__stlRecords ??= [];
  w.__stlInflight ??= 0;
  return {
    begin() {
      w.__stlInflight = (w.__stlInflight ?? 0) + 1;
      let done = false;
      return () => {
        if (!done) w.__stlInflight = (w.__stlInflight ?? 1) - 1;
        done = true;
      };
    },
    compute(kind, payload, result) {
      (w.__stlRecords ??= []).push({
        type: "compute", kind, key: snapshotKey(kind, payload), near: nearInfo(kind, payload), payload: JSON.parse(canonicalJson(payload)), result,
      });
    },
    data(name, value) {
      (w.__stlRecords ??= []).push({ type: "data", name, value });
    },
  };
}
