// Locked artifact build: `STL_ARTIFACT_PASSWORD=... npm run build:artifact -- --lock` (called by build-artifact.mjs).
// Only the password gate is readable; the app, its stylesheet and every data file are encrypted (scripts/lock-crypto.mjs):
//   stl-simulator.html  entry fragment: title, lock.css link, <div id="root">, the password card (logo =
//                       public/favicon.svg; no names — the gate gives a password guesser no vocabulary), ./lock.js
//   lock.js, lock.css   the gate (scripts/lock/), identical in every build
//   lock.json           {v, kdf, cipher, salt, iterations, app, css, index, files} — plaintext parameters only
//   assets/app-<rand>.wasm   the whole app as ONE classic IIFE script (vite JS API, inlineDynamicImports, no source
//                       maps), gzip → AES-256-GCM
//   assets/style-<rand>.wasm the app stylesheet (KaTeX included), gzip → AES-256-GCM; applied by the gate after unlock
//   assets/KaTeX_*.woff2    plain, byte-identical to node_modules/katex/dist/fonts (checked)
//   snapshot/index.wasm, snapshot/<rand>.wasm     every snapshot file (index with presets/meta included), gzip →
//                       AES-256-GCM; published as application/wasm (binary; see ENC_EXT)
// Key = PBKDF2-SHA256(password, random 16-byte salt, ≥ 600 000 iterations); every file has its own random IV and
// its published path as AAD. The build writes to a temporary sibling folder, decrypts everything once (self-check),
// scans every plaintext file for model strings (leak check) and every output file for the password, and only when
// all of that passes replaces the output folder (a failed build leaves nothing publishable behind).
import { createHash, randomBytes } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";
import { LIMITS, TEXT_EXT, escapeFffd, fmtMB, walk, woff2Only } from "./artifact-common.mjs";
import { MIN_ITERATIONS, SALT_BYTES, decryptFile, deriveKey, encryptFile, isEncrypted, newSalt } from "./lock-crypto.mjs";

// Encrypted files are published as .wasm (application/wasm): a binary type the artifact host serves unchanged
// (it refuses application/octet-stream); the bytes are only fetched and decrypted, never compiled.
export const ENC_EXT = ".wasm";
export const LOCK_INDEX = `snapshot/index${ENC_EXT}`;
export const ENC_TYPE = "application/wasm";
/** Strings that must never appear in a plaintext file of the locked build. */
export const LEAK_PATTERNS = [
  [/gidl/i, "GIDL"],
  [/p\[1[0-9]\]|z\[1[0-9]\]/, "parameter vector index"],
  [/3\.7037|2\.598[0-9]|3\.808[0-9]/, "fold voltage"],
  [/0\.2794/, "illumination coefficient"],
  [/calib/i, "calibration"],
  [/impact/i, "impact ionisation"],
  [/avalanche/i, "avalanche"],
  [/V_L[UD]|V<sub>L|래치/, "latch voltages"],
  [/latch/i, "latch"],
  [/biristor|stochastic|breakdown|illuminat|photocurrent/i, "device / model words"],
  [/NOBEL|nobelab|KAIST|카이스트|최양규|박준형|Yang-?Kyu|Junhyoung/i, "lab / people names (the gate gives no password vocabulary)"],
  [/tau_(bulk|junction|ratio)|phi_(gidl|emitter)|responsivity|na_access|t_access|l_access/i, "model parameter names"],
  [/MODEL_SPEC|stl_eval|charge_balance|vg_curve|design_map/, "model/code identifiers"],
  [/FDSOI|T_Si|EOT/, "technology details"],
];

const sha8 = (buf) => createHash("sha256").update(buf).digest("hex").slice(0, 10);
const randName = () => randomBytes(8).toString("hex");

/** Fingerprints of the calibrated numbers (presets' `calib` blocks): the mantissa up to its 6th significant digit
 *  ("7.16650", "0.000272122"), for values with more digits than that and at least 3 distinct digits. */
export function numberFingerprints(index) {
  const out = new Set();
  const visit = (v) => {
    if (typeof v === "number" && Number.isFinite(v)) {
      // plain and exponent spelling (minifiers write 2.72e-4 as 272e-6 or .000272, so both mantissas count)
      for (const m of [String(Math.abs(v)).split("e")[0], Math.abs(v).toExponential().split("e")[0]]) {
        let sig = 0;
        for (let i = 0; i < m.length; i++) {
          if (/[1-9]/.test(m[i]) || (sig > 0 && m[i] === "0")) sig++;
          if (sig === 6) {
            const fp = m.slice(0, i + 1);
            if (m.length > i + 1 && new Set(fp.replace(/[^0-9]/g, "")).size >= 3) out.add(fp);
            break;
          }
        }
      }
    } else if (v && typeof v === "object") for (const x of Object.values(v)) visit(x);
  };
  for (const p of Object.values(index?.meta?.presets ?? {})) visit(p?.device?.calib);
  return [...out];
}

/** SHA-256 of every stock KaTeX woff2 font (node_modules/katex/dist/fonts): published fonts must be one of them. */
export function stockFonts(WEB) {
  const dir = path.join(WEB, "node_modules/katex/dist/fonts");
  const out = new Set();
  if (fs.existsSync(dir)) for (const f of fs.readdirSync(dir)) if (f.endsWith(".woff2")) out.add(createHash("sha256").update(fs.readFileSync(path.join(dir, f))).digest("hex"));
  return out;
}

/** Scan `files` (paths under dir) that are not encrypted. Returns ["file: label (match)"]. A woff2 font is accepted
 *  only when it is byte-identical to a stock KaTeX font (`stock`, see stockFonts) — a binary is not text-scanned
 *  (short patterns match random bytes) but it must not be anything else either. */
export function scanLeaks(dir, files, { fingerprints = [], stock = null } = {}) {
  const hits = [];
  for (const f of files) {
    const buf = fs.readFileSync(path.join(dir, f));
    if (isEncrypted(buf)) continue;
    if (/\.woff2$/i.test(f)) {
      if (stock?.has(createHash("sha256").update(buf).digest("hex"))) continue;
      if (stock) {
        hits.push(`${f}: not a stock KaTeX font`);
        continue;
      }
    }
    const s = buf.toString(TEXT_EXT.test(f) ? "utf8" : "latin1");
    for (const [re, label] of LEAK_PATTERNS) {
      const m = re.exec(s);
      if (m) hits.push(`${f}: ${label} ("${m[0]}")`);
    }
    for (const fp of fingerprints) if (s.includes(fp)) hits.push(`${f}: calibration number ${fp}…`);
  }
  return hits;
}

/** Files containing the password (raw UTF-8, NFC, or base64) — must be none. Never prints the password. */
export function scanPassword(dir, files, password) {
  const needles = [...new Set([password, password.normalize("NFC")])].flatMap((p) => [Buffer.from(p, "utf8"), Buffer.from(Buffer.from(p, "utf8").toString("base64"))]);
  return files.filter((f) => {
    const buf = fs.readFileSync(path.join(dir, f));
    return needles.some((n) => buf.includes(n));
  });
}

/** The logo of the card: public/favicon.svg without comments, ids prefixed, sized and hidden from AT. */
function logoSvg(WEB) {
  let s = fs.readFileSync(path.join(WEB, "public/favicon.svg"), "utf8");
  s = s.replace(/<!--[\s\S]*?-->/g, "").replace(/<\?xml[^>]*>/, "");
  const ids = [...s.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
  for (const id of ids) s = s.split(`id="${id}"`).join(`id="stl-lk-${id}"`).split(`url(#${id})`).join(`url(#stl-lk-${id})`);
  s = s.replace(/<svg\b/, '<svg class="stl-lock__logo" width="56" height="56" aria-hidden="true" focusable="false"');
  return s.replace(/\n\s*\n/g, "\n").trim();
}

/** Vite build of the whole app as one classic IIFE script (in memory; vite.config.ts is used as is). */
async function buildIife(WEB, log) {
  const { build } = await import("vite");
  const res = await build({
    root: WEB,
    configFile: path.join(WEB, "vite.config.ts"),
    base: "./",
    mode: "production",
    logLevel: "warn",
    build: {
      write: false,
      outDir: path.join(WEB, "dist-artifact"), // unused (write: false)
      emptyOutDir: false,
      copyPublicDir: false,
      sourcemap: false,
      cssCodeSplit: false,
      modulePreload: false,
      reportCompressedSize: false,
      chunkSizeWarningLimit: 100_000,
      rollupOptions: {
        input: path.join(WEB, "src/main.tsx"),
        // codeSplitting: false = Rolldown's inlineDynamicImports (one chunk); the preload helper's import.meta.url is
        // then unused (every dynamic import is inlined, no deps to preload)
        onwarn(w, next) {
          if (w.code !== "EMPTY_IMPORT_META") next(w);
        },
        output: { format: "iife", codeSplitting: false, entryFileNames: "app.js", assetFileNames: "assets/[name]-[hash][extname]" },
      },
    },
  });
  const items = (Array.isArray(res) ? res : [res]).flatMap((o) => o.output ?? []);
  const chunks = items.filter((i) => i.type === "chunk");
  const assets = items.filter((i) => i.type === "asset");
  log(`IIFE bundle: ${chunks.length} chunk, ${assets.length} assets`);
  return { chunks, assets };
}

export async function buildLocked({ WEB, OUT, SNAP_SRC, ENTRY, password, log, iterations = MIN_ITERATIONS, noSnapshot = false }) {
  if (!Number.isInteger(iterations) || iterations < MIN_ITERATIONS) {
    console.error(`[artifact] --iterations must be an integer ≥ ${MIN_ITERATIONS}`);
    return false;
  }
  if (password.length < 8) log("WARNING: the password is short; its strength is the only protection of the page");
  // everything is written to a sibling temp folder; OUT is replaced only when every check passed
  const TMP = `${OUT}.tmp-${randomBytes(4).toString("hex")}`;
  let ok = false;
  try {
    ok = await writeLocked({ WEB, OUT, DIR: TMP, SNAP_SRC, ENTRY, password, log, iterations, noSnapshot });
    if (ok) {
      fs.rmSync(OUT, { recursive: true, force: true });
      fs.renameSync(TMP, OUT);
      log(`LOCKED build OK → ${path.relative(WEB, OUT)}/`);
    } else console.error(`[artifact] LOCKED build FAILED — nothing written to ${path.relative(WEB, OUT)}/ (the previous build there, if any, is unchanged)`);
    return ok;
  } finally {
    if (!ok) fs.rmSync(TMP, { recursive: true, force: true });
  }
}

async function writeLocked({ WEB, OUT, DIR, SNAP_SRC, ENTRY, password, log, iterations, noSnapshot }) {
  const problems = [];
  const rel = path.relative(WEB, OUT);

  // ------------------------------------------------------------ 1. app bundle (one IIFE) + stylesheet + fonts
  const { chunks, assets } = await buildIife(WEB, log);
  if (chunks.length !== 1) {
    console.error(`[artifact] expected one IIFE chunk, got ${chunks.length}`);
    return false;
  }
  const chunk = chunks[0];
  // (dynamicImports lists the chunk itself once every dynamic import is inlined)
  if (chunk.imports?.length || (chunk.dynamicImports ?? []).some((f) => f !== chunk.fileName)) problems.push("the IIFE bundle still imports other chunks");
  if (/^\s*["']use strict["']/.test(chunk.code)) problems.push("the IIFE bundle starts with a directive (the loader prepends a statement)");
  const cssAssets = assets.filter((a) => a.fileName.endsWith(".css"));
  if (cssAssets.length !== 1) problems.push(`expected one stylesheet, got ${cssAssets.length}`);

  fs.rmSync(DIR, { recursive: true, force: true });
  fs.mkdirSync(path.join(DIR, "assets"), { recursive: true });

  const [cssText, fffd] = escapeFffd(String(cssAssets[0]?.source ?? ""), true);
  const css = woff2Only(cssText);
  if (/@import\b/i.test(css)) problems.push("the app stylesheet uses @import (a constructable stylesheet ignores it)");
  let fonts = 0;
  let dropped = 0;
  for (const a of assets) {
    if (a.fileName.endsWith(".css")) continue;
    const base = path.posix.basename(a.fileName);
    if (css.includes(base)) {
      fs.writeFileSync(path.join(DIR, "assets", base), a.source);
      fonts++;
    } else if (chunk.code.includes(base)) {
      problems.push(`the app bundle references ${base}; script-relative asset URLs do not work in the locked loader`);
    } else dropped++;
  }

  // ------------------------------------------------------------ 2. key
  const salt = newSalt();
  if (salt.length !== SALT_BYTES) throw new Error("salt");
  const t0 = Date.now();
  const key = deriveKey(password, salt, iterations);
  log(`key: PBKDF2-SHA256, ${iterations} iterations, 16-byte random salt (${Date.now() - t0} ms)`);
  const encrypted = new Map(); // published path → kind (for the self-check)

  // ------------------------------------------------------------ 3. app + stylesheet (gzip → encrypt)
  const appPath = `assets/app-${randName()}${ENC_EXT}`;
  const appGz = gzipSync(Buffer.from(chunk.code, "utf8"), { level: 9 });
  fs.writeFileSync(path.join(DIR, appPath), encryptFile(appGz, key, appPath));
  encrypted.set(appPath, "app");
  log(`app: ${fmtMB(chunk.code.length)} JS → ${fmtMB(appGz.length)} gzip → ${appPath}`);
  // the stylesheet sits in assets/ like before, so its url(./KaTeX_….woff2) references keep their meaning
  const cssPath = `assets/style-${randName()}${ENC_EXT}`;
  fs.writeFileSync(path.join(DIR, cssPath), encryptFile(gzipSync(Buffer.from(css, "utf8"), { level: 9 }), key, cssPath));
  encrypted.set(cssPath, "css");
  log(`stylesheet: ${fmtMB(css.length)} (${fffd} U+FFFD escaped) → ${cssPath}; ${fonts} KaTeX woff2 font(s) plain, ${dropped} unused font file(s) dropped`);

  // ------------------------------------------------------------ 4. snapshot (every file, index included)
  let fingerprints = [];
  if (fs.existsSync(path.join(SNAP_SRC, "index.json"))) fingerprints = numberFingerprints(JSON.parse(fs.readFileSync(path.join(SNAP_SRC, "index.json"), "utf8")));
  if (noSnapshot) log("--no-snapshot: the unlocked page will run in demo mode");
  else if (!fs.existsSync(path.join(SNAP_SRC, "index.json"))) {
    console.error(`[artifact] ${path.relative(WEB, SNAP_SRC)}/index.json not found — run \`npm run snapshot:record\` first (or pass --no-snapshot)`);
    return false;
  } else {
    fs.mkdirSync(path.join(DIR, "snapshot"), { recursive: true });
    const index = JSON.parse(fs.readFileSync(path.join(SNAP_SRC, "index.json"), "utf8"));
    const names = new Map(); // source file → { file, bytes }
    const refs = [...Object.values(index.data ?? {}), ...(index.compute ?? [])];
    for (const ref of refs) {
      let out = names.get(ref.file);
      if (!out) {
        const src = path.join(SNAP_SRC, ref.file);
        if (!fs.existsSync(src)) {
          problems.push(`snapshot file missing: ${ref.file}`);
          continue;
        }
        const raw = fs.readFileSync(src);
        const gz = raw[0] === 0x1f && raw[1] === 0x8b ? raw : gzipSync(raw, { level: 9 });
        const file = `${randName()}${ENC_EXT}`;
        const enc = encryptFile(gz, key, `snapshot/${file}`);
        fs.writeFileSync(path.join(DIR, "snapshot", file), enc);
        encrypted.set(`snapshot/${file}`, "json");
        out = { file, bytes: enc.length };
        names.set(ref.file, out);
      }
      ref.file = out.file;
      ref.bytes = out.bytes;
      delete ref.plain;
    }
    const idx = gzipSync(Buffer.from(JSON.stringify(index), "utf8"), { level: 9 });
    fs.writeFileSync(path.join(DIR, LOCK_INDEX), encryptFile(idx, key, LOCK_INDEX));
    encrypted.set(LOCK_INDEX, "index");
    log(`snapshot: ${index.compute?.length ?? 0} results in ${names.size} files + index, recorded ${index.created} — all encrypted`);
  }

  // ------------------------------------------------------------ 5. gate: lock.js, lock.css, lock.json, entry
  for (const f of ["lock.js", "lock.css"]) fs.copyFileSync(path.join(WEB, "scripts/lock", f), path.join(DIR, f));
  const lockJson = {
    v: 1,
    kdf: "PBKDF2-SHA256",
    cipher: "AES-256-GCM",
    format: "STLENC1\\0 | iv(12) | ciphertext | tag(16); AAD = published path",
    salt: Buffer.from(salt).toString("base64"),
    iterations,
    app: appPath,
    css: cssPath,
    index: noSnapshot ? null : LOCK_INDEX,
    files: [...encrypted.keys()].sort(),
  };
  fs.writeFileSync(path.join(DIR, "lock.json"), JSON.stringify(lockJson, null, 1) + "\n");
  const entry = fs.readFileSync(path.join(WEB, "scripts/lock/entry.html"), "utf8").replace("{{LOGO}}", logoSvg(WEB));
  fs.writeFileSync(path.join(DIR, ENTRY), entry);
  for (const tag of ["<!doctype", "<html", "<head", "<body", "{{"]) if (entry.toLowerCase().includes(tag)) problems.push(`entry contains ${tag}`);
  const links = [...entry.matchAll(/<(?:link|script)\b[^>]*\b(?:href|src)="([^"]+)"/g)].map((m) => m[1]);
  if (links.join(" ") !== "./lock.css ./lock.js") problems.push(`the entry loads more than the gate: ${links.join(", ")}`);

  // ------------------------------------------------------------ 6. files.json + limits
  const files = walk(DIR).filter((f) => f !== ENTRY && f !== "files.json");
  fs.writeFileSync(
    path.join(DIR, "files.json"),
    JSON.stringify(files.map((f) => (f.endsWith(ENC_EXT) ? { path: f, contentType: ENC_TYPE } : { path: f })), null, 1) + "\n",
  );
  let total = 0;
  const sizes = [];
  for (const f of [ENTRY, ...files]) {
    const p = path.join(DIR, f);
    const size = fs.statSync(p).size;
    total += size;
    sizes.push([f, size]);
    const text = TEXT_EXT.test(f);
    if (text && fs.readFileSync(p, "utf8").includes("�")) problems.push(`U+FFFD left in ${f}`);
    if (size > (text ? LIMITS.textBytes : LIMITS.binaryBytes)) problems.push(`${f} is ${fmtMB(size)} (limit ${fmtMB(text ? LIMITS.textBytes : LIMITS.binaryBytes)})`);
  }
  if (files.length + 1 > LIMITS.files) problems.push(`${files.length + 1} files > ${LIMITS.files}`);
  if (total > LIMITS.versionBytes) problems.push(`total ${fmtMB(total)} > ${fmtMB(LIMITS.versionBytes)}`);

  // ------------------------------------------------------------ 7. self-check: every encrypted file opens
  for (const [f, kind] of encrypted) {
    try {
      const buf = fs.readFileSync(path.join(DIR, f));
      if (!isEncrypted(buf)) throw new Error("no magic");
      const plain = gunzipSync(decryptFile(buf, key, f));
      if (kind === "app" && plain.toString("utf8") !== chunk.code) throw new Error("app bundle differs");
      if (kind === "css" && plain.toString("utf8") !== css) throw new Error("stylesheet differs");
      if (kind === "json" || kind === "index") JSON.parse(plain.toString("utf8"));
      if (kind === "index") {
        const index = JSON.parse(plain.toString("utf8"));
        for (const ref of [...Object.values(index.data ?? {}), ...(index.compute ?? [])])
          if (!encrypted.has(`snapshot/${ref.file}`)) throw new Error(`index points to a missing file ${ref.file}`);
      }
      let moved = false; // the AAD binds the file to its path
      try {
        decryptFile(buf, key, `${f}.moved`);
        moved = true;
      } catch {
        /* expected */
      }
      if (moved) throw new Error("decrypts under another path");
    } catch (e) {
      problems.push(`self-check ${f}: ${e.message}`);
    }
  }
  key.fill(0);

  // ------------------------------------------------------------ 8. leak check + password check
  const plainFiles = [ENTRY, ...files].filter((f) => !encrypted.has(f));
  const stock = stockFonts(WEB);
  if (!stock.size) problems.push("node_modules/katex/dist/fonts not found (needed to prove the fonts are stock)");
  const leaks = scanLeaks(DIR, plainFiles, { fingerprints, stock });
  for (const l of leaks) problems.push(`LEAK ${l}`);
  const withPassword = scanPassword(DIR, [ENTRY, "files.json", ...files], password);
  for (const f of withPassword) problems.push(`the password appears in ${f}`);
  log(`leak check: ${plainFiles.length} plaintext files × ${LEAK_PATTERNS.length} patterns + ${fingerprints.length} calibration numbers (fonts: stock KaTeX) → ${leaks.length} hit(s); password in ${withPassword.length} of ${files.length + 2} files`);

  const snapFiles = files.filter((f) => f.startsWith("snapshot/"));
  const snapBytes = sizes.filter(([f]) => f.startsWith("snapshot/")).reduce((s, [, b]) => s + b, 0);
  log(`${rel}/${ENTRY} + ${files.length} files (${snapFiles.length} snapshot, ${encrypted.size} encrypted) = ${files.length + 1} files, ${fmtMB(total)} (snapshot ${fmtMB(snapBytes)})`);
  log("plaintext files:", plainFiles.filter((f) => !/\.woff2$/.test(f)).join(", "), `+ ${plainFiles.filter((f) => /\.woff2$/.test(f)).length} woff2 fonts`);
  log(`publish: file_path ${rel}/${ENTRY}, root ${rel}, files from ${rel}/files.json (${ENC_EXT} as ${ENC_TYPE})`);
  if (problems.length) {
    for (const p of problems) console.error(`[artifact] PROBLEM: ${p}`);
    return false;
  }
  return true;
}
