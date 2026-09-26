// Small helpers for path-addressed nested objects (parameter state) and structural comparison.
import { nearlyEqual } from "./format";

export type Path = readonly (string | number)[];

export function getPath(obj: unknown, path: Path): unknown {
  let cur: unknown = obj;
  for (const k of path) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string | number, unknown>)[k];
  }
  return cur;
}

/** Immutable set: returns a new object with `value` at `path` (copies every object on the path). */
export function setPath<T>(obj: T, path: Path, value: unknown): T {
  if (path.length === 0) return value as T;
  const [k, ...rest] = path;
  const src = (obj ?? {}) as Record<string | number, unknown>;
  const copy: Record<string | number, unknown> = Array.isArray(src) ? ([...src] as never) : { ...src };
  copy[k] = setPath(src[k], rest, value);
  return copy as T;
}

export function clone<T>(v: T): T {
  return v === undefined ? v : (JSON.parse(JSON.stringify(v)) as T);
}

/** Deep structural equality with relative tolerance on numbers. */
export function deepEqual(a: unknown, b: unknown): boolean {
  if (typeof a === "number" || typeof b === "number") return nearlyEqual(a, b);
  if (a === b) return true;
  if (a == null || b == null || typeof a !== "object" || typeof b !== "object") return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a as object);
  const kb = Object.keys(b as object);
  const keys = new Set([...ka, ...kb]);
  for (const k of keys) {
    if (!deepEqual((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k])) return false;
  }
  return true;
}

/** Deep merge used to fill a possibly partial/older persisted object from defaults (like params._merge). */
export function mergeDefaults<T>(base: T, over: unknown): T {
  if (over == null || typeof over !== "object" || Array.isArray(over)) return clone(base);
  const out = clone(base) as Record<string, unknown>;
  for (const [k, v] of Object.entries(over as Record<string, unknown>)) {
    if (!(k in out)) continue; // drop unknown keys from stale storage
    const bv = out[k];
    if (bv && typeof bv === "object" && !Array.isArray(bv)) out[k] = mergeDefaults(bv, v);
    else if (bv === null) out[k] = v === null || typeof v === "number" ? v : null; // "auto" fields: number | null
    else if (Array.isArray(bv)) out[k] = Array.isArray(v) ? v : bv;
    else if (v !== null && v !== undefined && typeof v === typeof bv) out[k] = v;
  }
  return out as T;
}

/** Canonical JSON (sorted keys) — used as a cache key for payloads. */
export function canonical(v: unknown): string {
  if (v === null || typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(canonical).join(",")}]`;
  const o = v as Record<string, unknown>;
  return `{${Object.keys(o)
    .sort()
    .filter((k) => o[k] !== undefined)
    .map((k) => `${JSON.stringify(k)}:${canonical(o[k])}`)
    .join(",")}}`;
}
