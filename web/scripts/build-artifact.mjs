#!/usr/bin/env node
// Builds the static artifact (web/dist-artifact/): the frontend with relative asset URLs plus the recorded
// snapshot (web/snapshot/, see scripts/record-snapshot.mjs), shaped for the claude.ai artifact host:
//   - `vite build --base ./` (no tsc here; run `npm run typecheck` separately)
//   - snapshot/ copied next to the page (the app fetches ./snapshot/index.json when /api is unreachable)
//   - U+FFFD escaped in .js/.css (the publisher rejects the raw character)
//   - KaTeX .woff/.ttf removed, @font-face lists reduced to woff2 (fewer files)
//   - stl-simulator.html = the entry WITHOUT doctype/html/head/body (the host wraps it): title, stylesheet and
//     modulepreload links, <div id="root">, the module script
//   - files.json = [{ path }] of every other published file (relative to dist-artifact)
// Usage: npm run build:artifact [-- --plain] [--no-snapshot] [--out dist-artifact]
//   --plain        publish the snapshot as plain .json instead of .json.gz (if the host refuses gzip files)
//   --no-snapshot  build without snapshot/ (the page then runs in demo mode)
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);
const flag = (n) => argv.includes(`--${n}`);
const opt = (n, d) => {
  const i = argv.indexOf(`--${n}`);
  return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[i + 1] : d;
};
const OUT = path.resolve(WEB, opt("out", "dist-artifact"));
const SNAP_SRC = path.resolve(WEB, opt("snapshot", "snapshot"));
const ENTRY = "stl-simulator.html";
const LIMITS = { files: 255, textBytes: 16 * 1024 * 1024, binaryBytes: 15 * 1024 * 1024, versionBytes: 64 * 1024 * 1024 };
const TEXT_EXT = /\.(html|js|mjs|css|json|svg|txt|map)$/i;
const fmtMB = (b) => `${(b / 1048576).toFixed(2)} MB`;
const log = (...m) => console.log("[artifact]", ...m);
const problems = [];

function walk(dir, base = dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p, base));
    else out.push(path.relative(base, p).split(path.sep).join("/"));
  }
  return out.sort();
}

// ---------------------------------------------------------------- 1. vite build
fs.rmSync(OUT, { recursive: true, force: true });
log(`vite build --base ./ → ${path.relative(WEB, OUT)}/`);
const vb = spawnSync(process.execPath, [path.join(WEB, "node_modules/vite/bin/vite.js"), "build", "--base", "./", "--outDir", OUT, "--emptyOutDir"], {
  cwd: WEB,
  stdio: "inherit",
});
if (vb.status !== 0) {
  console.error("[artifact] vite build failed");
  process.exit(1);
}

// ---------------------------------------------------------------- 2. snapshot
if (flag("no-snapshot")) log("--no-snapshot: the page will run in demo mode");
else if (!fs.existsSync(path.join(SNAP_SRC, "index.json"))) {
  console.error(`[artifact] ${path.relative(WEB, SNAP_SRC)}/index.json not found — run \`npm run snapshot:record\` first (or pass --no-snapshot)`);
  process.exit(1);
} else {
  const dst = path.join(OUT, "snapshot");
  fs.mkdirSync(dst, { recursive: true });
  const index = JSON.parse(fs.readFileSync(path.join(SNAP_SRC, "index.json"), "utf8"));
  const refs = [...Object.values(index.data ?? {}), ...(index.compute ?? [])];
  for (const ref of refs) {
    const src = path.join(SNAP_SRC, ref.file);
    if (!fs.existsSync(src)) {
      problems.push(`snapshot file missing: ${ref.file}`);
      continue;
    }
    if (flag("plain") && ref.file.endsWith(".gz")) {
      const plain = ref.file.replace(/\.gz$/, "");
      const buf = gunzipSync(fs.readFileSync(src));
      fs.writeFileSync(path.join(dst, plain), buf);
      ref.file = plain;
      ref.bytes = buf.length;
      delete ref.plain;
    } else {
      fs.copyFileSync(src, path.join(dst, ref.file));
      if (ref.plain && fs.existsSync(path.join(SNAP_SRC, ref.plain))) fs.copyFileSync(path.join(SNAP_SRC, ref.plain), path.join(dst, ref.plain));
      else delete ref.plain;
    }
  }
  fs.writeFileSync(path.join(dst, "index.json"), JSON.stringify(index, null, 1).replace(/�/g, "\\ufffd"));
  log(`snapshot: ${index.compute?.length ?? 0} results recorded ${index.created} from ${index.source ?? "?"}${flag("plain") ? " (plain .json)" : ""}`);
}

// ---------------------------------------------------------------- 3./4. escape U+FFFD, KaTeX fonts
let fffd = 0;
let dropped = 0;
for (const rel of walk(OUT)) {
  const p = path.join(OUT, rel);
  if (/KaTeX_.*\.(woff|ttf)$/i.test(rel)) {
    fs.rmSync(p);
    dropped++;
    continue;
  }
  if (!/\.(js|mjs|css)$/i.test(rel)) continue;
  let s = fs.readFileSync(p, "utf8");
  const n = (s.match(/�/g) ?? []).length;
  if (n) {
    // JS: � is the same character in string, template and regex literals; CSS: hex escape
    s = s.replace(/�/g, rel.endsWith(".css") ? "\\fffd " : "\\ufffd");
    fffd += n;
  }
  if (rel.endsWith(".css")) {
    // keep only the woff2 entries of every @font-face src list (all current browsers load woff2)
    // (entries are `url(...)format("...")`; small fonts are inlined as data: URIs, which contain ";" but no ")")
    s = s.replace(/src:((?:\s*url\([^)]*\)\s*format\([^)]*\)\s*,?)+)/g, (m, list) => {
      const parts = list.split(/,(?=\s*url\()/);
      const keep = parts.filter((x) => /format\(\s*["']?woff2/.test(x));
      return keep.length && keep.length < parts.length ? `src:${keep.map((x) => x.trim().replace(/,$/, "")).join(",")}` : m;
    });
  }
  fs.writeFileSync(p, s);
}
log(`escaped ${fffd} U+FFFD character(s); removed ${dropped} KaTeX .woff/.ttf file(s)`);

// ---------------------------------------------------------------- 5. entry
const html = fs.readFileSync(path.join(OUT, "index.html"), "utf8");
const tags = (re) => [...html.matchAll(re)].map((m) => m[0]);
const styles = tags(/<link\b[^>]*rel="stylesheet"[^>]*>/g);
const preloads = tags(/<link\b[^>]*rel="modulepreload"[^>]*>/g);
const scripts = tags(/<script\b[^>]*type="module"[^>]*><\/script>/g);
if (!scripts.length) {
  console.error("[artifact] no module script found in index.html");
  process.exit(1);
}
const entry = ["<title>STL Simulator</title>", ...styles, ...preloads, '<div id="root"></div>', ...scripts].join("\n") + "\n";
fs.writeFileSync(path.join(OUT, ENTRY), entry);
fs.rmSync(path.join(OUT, "index.html"));
for (const tag of ["<!doctype", "<html", "<head", "<body"]) if (entry.toLowerCase().includes(tag)) problems.push(`entry contains ${tag}`);

// ---------------------------------------------------------------- 6. checks + files.json
const all = walk(OUT).filter((f) => f !== "files.json");
const published = all.filter((f) => f !== ENTRY);
// assets the page never references (e.g. public/favicon.svg — the host supplies the tab icon)
const referenced = (f) => {
  if (f.startsWith("snapshot/")) return true;
  const name = path.basename(f);
  return all.some((g) => g !== f && TEXT_EXT.test(g) && !g.startsWith("snapshot/") && fs.readFileSync(path.join(OUT, g), "utf8").includes(name));
};
const unused = published.filter((f) => !f.includes("/") && !referenced(f)); // top-level copies of public/ only
for (const f of unused) fs.rmSync(path.join(OUT, f));
const files = published.filter((f) => !unused.includes(f));
fs.writeFileSync(
  path.join(OUT, "files.json"),
  JSON.stringify(files.map((f) => (f.endsWith(".gz") ? { path: f, contentType: "application/gzip" } : { path: f })), null, 1) + "\n",
);

let total = 0;
const sizes = [];
for (const f of [ENTRY, ...files]) {
  const p = path.join(OUT, f);
  const size = fs.statSync(p).size;
  total += size;
  sizes.push([f, size]);
  const text = TEXT_EXT.test(f);
  if (text && fs.readFileSync(p, "utf8").includes("�")) problems.push(`U+FFFD left in ${f}`);
  if (size > (text ? LIMITS.textBytes : LIMITS.binaryBytes)) problems.push(`${f} is ${fmtMB(size)} (limit ${fmtMB(text ? LIMITS.textBytes : LIMITS.binaryBytes)})`);
}
if (files.length + 1 > LIMITS.files) problems.push(`${files.length + 1} files > ${LIMITS.files}`);
if (total > LIMITS.versionBytes) problems.push(`total ${fmtMB(total)} > ${fmtMB(LIMITS.versionBytes)}`);

const snapFiles = files.filter((f) => f.startsWith("snapshot/"));
const snapBytes = sizes.filter(([f]) => f.startsWith("snapshot/")).reduce((s, [, b]) => s + b, 0);
log(`${path.relative(WEB, OUT)}/${ENTRY} + ${files.length} files (${snapFiles.length} snapshot, ${files.length - snapFiles.length} app) = ${files.length + 1} files, ${fmtMB(total)} (snapshot ${fmtMB(snapBytes)})`);
if (unused.length) log(`dropped unreferenced: ${unused.join(", ")}`);
log("largest files:");
for (const [f, b] of sizes.sort((a, b) => b[1] - a[1]).slice(0, 8)) log(`  ${fmtMB(b).padStart(9)}  ${f}`);
log(`publish: file_path ${path.relative(WEB, OUT)}/${ENTRY}, root ${path.relative(WEB, OUT)}, files from ${path.relative(WEB, OUT)}/files.json`);
if (problems.length) {
  for (const p of problems) console.error(`[artifact] PROBLEM: ${p}`);
  process.exit(1);
}
