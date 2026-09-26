// Helpers shared by the plain and the locked artifact build (scripts/build-artifact.mjs, scripts/lock-build.mjs).
import fs from "node:fs";
import path from "node:path";

/** Publisher limits of the claude.ai artifact host. */
export const LIMITS = { files: 255, textBytes: 16 * 1024 * 1024, binaryBytes: 15 * 1024 * 1024, versionBytes: 64 * 1024 * 1024 };
export const TEXT_EXT = /\.(html|js|mjs|css|json|svg|txt|map)$/i;
export const fmtMB = (b) => `${(b / 1048576).toFixed(2)} MB`;

/** Files below `dir`, as sorted "/"-separated paths relative to `base`. */
export function walk(dir, base = dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p, base));
    else out.push(path.relative(base, p).split(path.sep).join("/"));
  }
  return out.sort();
}

/** Escape U+FFFD (the publisher rejects the raw character). JS: "�" is the same character in string, template
 *  and regex literals; CSS: hex escape. Returns [text, count]. */
export function escapeFffd(s, css) {
  const n = (s.match(/�/g) ?? []).length;
  return [n ? s.replace(/�/g, css ? "\\fffd " : "\\ufffd") : s, n];
}

/** Keep only the woff2 entries of every @font-face src list (all current browsers load woff2). Entries are
 *  `url(...)format("...")`; small fonts inlined as data: URIs contain ";" but no ")". */
export function woff2Only(css) {
  return css.replace(/src:((?:\s*url\([^)]*\)\s*format\([^)]*\)\s*,?)+)/g, (m, list) => {
    const parts = list.split(/,(?=\s*url\()/);
    const keep = parts.filter((x) => /format\(\s*["']?woff2/.test(x));
    return keep.length && keep.length < parts.length ? `src:${keep.map((x) => x.trim().replace(/,$/, "")).join(",")}` : m;
  });
}
