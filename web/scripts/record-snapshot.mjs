#!/usr/bin/env node
// Records the static snapshot (web/snapshot/) from the live backend: starts a Vite dev server proxied to the
// FastAPI service, walks the default flows of the app in Chromium (both languages) with the dev-only recorder
// enabled (localStorage["stl-websim:record"] = "1", see src/api/snapshot.ts) and writes every compute
// request/result pair the app received plus the GET data endpoints:
//   snapshot/index.json            format, created, source, health, meta, data refs, compute[{kind,key,label,file,bytes}]
//   snapshot/<sha256>.json.gz      one result per distinct request   (key = sha256(canonical JSON {kind, payload}))
//   snapshot/grid-<kind>-<n>.json.gz  the V_G / optical-power grid (deterministic kinds), bundled {key: result}
//   snapshot/measured.json.gz, snapshot/design_map.json.gz
// Nothing is decimated: the files hold exactly what the app received.
//
// Flows (per language): device-det, device-sto, circuit-schematic, circuit-benches, validation; once (first
// language): device-grid = Device tab deterministic (branches, charge balance at the UI's default V_D, V_G curve)
// for V_G −3.8 … −0.9 V in 0.1 V steps for both presets (only V_G changed) and the illumination preset at the
// measured conditions (V_G × P = 0 / 1.15 / 2.55 / 3.51 mW). The page answers other V_G values with the nearest
// grid point (src/api/snapshot.ts, marked in the panel).
//
// Usage: npm run snapshot:record [-- --api http://127.0.0.1:8000] [--flows device-det,device-sto,circuit-schematic,
//        circuit-benches,validation,device-grid] [--langs ko,en] [--plain] [--with-val-vg] [--step-timeout 900] [--headed]
// Env:   STL_API (backend URL, default http://127.0.0.1:8000). Requires the backend to be running.
import { createHash } from "node:crypto";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";
import { chromium } from "@playwright/test";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = parseArgs(process.argv.slice(2));
const API = String(args.api ?? process.env.STL_API ?? "http://127.0.0.1:8000").replace(/\/+$/, "");
const OUT = path.resolve(WEB, String(args.out ?? "snapshot"));
const LANGS = String(args.langs ?? "ko,en").split(",").filter(Boolean);
const ALL_FLOWS = ["device-det", "device-sto", "circuit-schematic", "circuit-benches", "validation", "device-grid"];
const ONCE_FLOWS = new Set(["device-grid"]); // language-independent: first language only
const VG_GRID = Array.from({ length: 30 }, (_, i) => Number((-3.8 + 0.1 * i).toFixed(1))); // −3.8 … −0.9 V
const P_GRID = [0, 1.15, 2.55, 3.51]; // mW (the measured illumination conditions)
const BUNDLE_JSON_BYTES = 3 * 1024 * 1024; // grid bundles: ≈ 1.2 MB gzip each
const FLOWS = String(args.flows ?? ALL_FLOWS.join(",")).split(",").filter(Boolean);
const PLAIN = !!args.plain;
const WITH_VAL_VG = !!args["with-val-vg"];
const STEP_TIMEOUT_MS = Number(args["step-timeout"] ?? 900) * 1000;
const LIMITS = { bytes: 40 * 1024 * 1024, files: 150 };
const DATA_ENDPOINTS = { health: "/api/health", meta: "/api/meta", measured: "/api/data/measured", design_map: "/api/data/design_map" };

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const [k, v] = a.slice(2).split("=", 2);
    if (v !== undefined) out[k] = v;
    else if (argv[i + 1] && !argv[i + 1].startsWith("--")) out[k] = argv[++i];
    else out[k] = true;
  }
  return out;
}

const t0 = Date.now();
const log = (...m) => console.log(`[snapshot ${((Date.now() - t0) / 1000).toFixed(0).padStart(4)} s]`, ...m);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmtMB = (b) => `${(b / 1048576).toFixed(2)} MB`;

// ---------------------------------------------------------------- key (mirrors src/api/snapshot.ts)
function stable(v) {
  if (v === null || typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(stable).join(",")}]`;
  return `{${Object.keys(v).sort().map((k) => `${JSON.stringify(k)}:${stable(v[k])}`).join(",")}}`;
}
const canonicalJson = (v) => (v === undefined ? "null" : stable(JSON.parse(JSON.stringify(v))));
const snapshotKey = (kind, payload) => createHash("sha256").update(canonicalJson({ kind, payload }), "utf8").digest("hex");
/** JSON text with U+FFFD escaped (the artifact publisher rejects the raw character; only strings can hold it). */
const jsonText = (v) => JSON.stringify(v).replace(/�/g, "\\ufffd");

// ---------------------------------------------------------------- recording state
const compute = new Map(); // key → {kind, key, label, result}
const data = {}; // name → value
let dupes = 0;
let mismatches = 0;
const problems = [];

function ingest(records, label, grid = false) {
  let added = 0;
  let seen = 0;
  for (const r of records) {
    if (r.type === "data") {
      data[r.name] = r.value;
      continue;
    }
    if (r.type !== "compute") continue;
    seen++;
    const key = snapshotKey(r.kind, r.payload);
    if (key !== r.key) {
      mismatches++;
      problems.push(`key mismatch for ${r.kind} (${label}): browser ${r.key.slice(0, 12)} vs node ${key.slice(0, 12)}`);
    }
    if (compute.has(key)) {
      dupes++;
      continue;
    }
    compute.set(key, { kind: r.kind, key, label: `${label} · ${r.kind}`, near: r.near ?? null, grid, result: r.result });
    added++;
  }
  return { added, seen };
}

// ---------------------------------------------------------------- dev server
async function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.once("error", reject);
    s.listen(0, "127.0.0.1", () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });
}

/** Vite dev server (JS API) with HMR and file watching off: other edits to the source tree during a long recording
 *  must not hot-reload the page (a reload or remount aborts running jobs) — the code is served as it was at start. */
async function startVite() {
  const port = Number(args.port ?? (await freePort()));
  process.env.STL_API = API; // read by vite.config.ts (proxy /api → API)
  const { createServer } = await import("vite");
  const server = await createServer({
    root: WEB,
    configFile: path.join(WEB, "vite.config.ts"),
    logLevel: "warn",
    clearScreen: false,
    server: { port, strictPort: true, host: "127.0.0.1", hmr: false, watch: { ignored: ["**/*"] } },
  });
  await server.listen();
  return { vite: { kill: () => void server.close() }, base: `http://127.0.0.1:${port}` };
}

// ---------------------------------------------------------------- page helpers
async function inflight(page) {
  return page.evaluate(() => window.__stlInflight ?? 0);
}

/** Wait until no compute job has been in flight for `quietMs` (chained jobs start after a short gap). */
async function settle(page, quietMs = 1500) {
  const start = Date.now();
  let quietSince = null;
  while (Date.now() - start < STEP_TIMEOUT_MS) {
    const n = await inflight(page);
    if (n === 0) {
      quietSince ??= Date.now();
      if (Date.now() - quietSince >= quietMs) return;
    } else quietSince = null;
    await sleep(250);
  }
  throw new Error(`timeout (${STEP_TIMEOUT_MS / 1000} s) waiting for jobs to finish`);
}

async function drain(page, label, grid = false) {
  const recs = await page.evaluate(() => (window.__stlRecords ?? []).splice(0));
  return ingest(recs, label, grid);
}

async function step(page, label, action, { grid = false, retry = true } = {}) {
  const computeCount = () => page.evaluate(() => (window.__stlRecords ?? []).filter((r) => r.type === "compute").length);
  const before = await computeCount();
  const ts = Date.now();
  await action();
  const started = await page
    .waitForFunction((b) => (window.__stlInflight ?? 0) > 0 || (window.__stlRecords ?? []).filter((r) => r.type === "compute").length > b, before, { timeout: 10_000 })
    .then(() => true)
    .catch(() => false);
  await settle(page);
  const { added, seen } = await drain(page, label, grid);
  const errs = (await page.locator(".err-box").allTextContents()).map((s) => s.trim()).filter(Boolean);
  log(`  ${label}: +${added} new (${((Date.now() - ts) / 1000).toFixed(1)} s)${errs.length ? ` — ${errs.length} panel error(s)` : ""}${seen ? "" : " — nothing recorded"}`);
  if (!seen && !errs.length && retry) return step(page, `${label} (retry)`, action, { grid, retry: false });
  if (!started && !seen) problems.push(`${label}: no compute job started`);
  for (const e of errs) problems.push(`${label}: panel error "${e.slice(0, 160)}"`);
  return added;
}

async function openApp(browser, base, hash, lang, loadLabel = `${lang} · load`) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await ctx.addInitScript(() => {
    try {
      localStorage.setItem("stl-websim:record", "1");
    } catch {
      /* ignore */
    }
  });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  await page.goto(`${base}/${hash}`);
  await page.waitForFunction(() => document.querySelector("[data-testid=backend-status]")?.textContent?.includes("API"), null, { timeout: 60_000 });
  if (lang !== "ko") {
    await page.getByTestId("lang-toggle").click();
    await page.waitForFunction((l) => document.documentElement.lang === l, lang);
  }
  await settle(page);
  await drain(page, loadLabel);
  return { ctx, page };
}

const click = (page, id) => page.getByTestId(id).click();
const runButton = (page) => click(page, "run-button");

async function runSchematic(page) {
  await runButton(page);
  const confirm = page.getByTestId("sch-confirm-run");
  // the run chunk loads lazily and heavy circuits ask for confirmation first: wait for either
  for (let i = 0; i < 40; i++) {
    await sleep(150);
    if (await confirm.isVisible().catch(() => false)) {
      await confirm.click();
      return;
    }
    if ((await inflight(page)) > 0) return;
  }
}

/** Type a value into a sidebar field (opening its group if collapsed); the field commits while typing. */
async function setField(page, key, group, text) {
  const input = page.locator(`[data-testid=field-${key}] input.input`).first();
  if (!(await input.isVisible().catch(() => false))) await page.locator(`[data-testid=group-${group}] .group-toggle`).first().click();
  await input.fill(text);
  await input.press("Enter");
}

async function vgStochCompute(page) {
  const empty = page.getByTestId("vgs-compute");
  if (await empty.isVisible().catch(() => false)) return empty.click();
  return page.locator("[data-testid=panel-vg-sto] .panel-toolbar button.btn").last().click();
}

// ---------------------------------------------------------------- flows
const FLOW_IMPL = {
  // Device tab, deterministic: default state (reference preset as loaded), then each preset explicitly.
  async "device-det"(browser, base, lang) {
    const { ctx, page } = await openApp(browser, base, "#tab=device&mode=deterministic", lang);
    const L = `${lang} · device det`;
    await step(page, `${L} · default`, () => runButton(page));
    await step(page, `${L} · reference`, async () => (await click(page, "preset-paper"), runButton(page)));
    await step(page, `${L} · illumination`, async () => (await click(page, "preset-photo"), runButton(page)));
    await ctx.close();
  },
  // Device tab, stochastic: default state + stochastic V_G curve, then each preset explicitly.
  async "device-sto"(browser, base, lang) {
    const { ctx, page } = await openApp(browser, base, "#tab=device&mode=stochastic", lang);
    const L = `${lang} · device sto`;
    await step(page, `${L} · default`, () => runButton(page));
    await step(page, `${L} · default · V_G curve`, () => vgStochCompute(page));
    await step(page, `${L} · reference`, async () => (await click(page, "preset-paper"), runButton(page)));
    await step(page, `${L} · reference · V_G curve`, () => vgStochCompute(page));
    await step(page, `${L} · illumination`, async () => (await click(page, "preset-photo"), runButton(page)));
    await ctx.close();
  },
  // Circuit tab, schematic editor: the default circuit and every example template, deterministic and stochastic.
  async "circuit-schematic"(browser, base, lang) {
    for (const mode of ["deterministic", "stochastic"]) {
      const { ctx, page } = await openApp(browser, base, `#tab=circuit&mode=${mode}`, lang);
      const L = `${lang} · schematic ${mode === "deterministic" ? "det" : "sto"}`;
      await page.getByTestId("schematic-view").waitFor();
      await step(page, `${L} · default circuit`, () => runSchematic(page));
      await click(page, "menu-examples");
      const ids = await page.locator("[data-testid^=tpl-]").evaluateAll((els) => els.map((e) => e.getAttribute("data-testid").slice(4)));
      await page.keyboard.press("Escape");
      for (const id of ids) {
        await step(page, `${L} · example ${id}`, async () => {
          if (!(await page.getByTestId(`tpl-${id}`).isVisible().catch(() => false))) await click(page, "menu-examples");
          await click(page, `tpl-${id}`);
          await sleep(300);
          await runSchematic(page);
        });
      }
      await ctx.close();
    }
  },
  // Circuit tab, quick benches: every bench deterministic, the load line also stochastic.
  async "circuit-benches"(browser, base, lang) {
    const { ctx, page } = await openApp(browser, base, "#tab=circuit&mode=deterministic", lang);
    const L = `${lang} · bench`;
    await click(page, "circuit-view-benches");
    const benches = await page.locator("[data-testid=bench-picker] [role=radio]").evaluateAll((els) => els.map((e) => e.getAttribute("data-testid").slice(6)));
    for (const b of benches) await step(page, `${L} det · ${b}`, async () => (await click(page, `bench-${b}`), runButton(page)));
    await click(page, "mode-stochastic");
    await step(page, `${L} sto · load_line`, async () => (await click(page, "bench-load_line"), runButton(page)));
    await ctx.close();
  },
  // Device tab deterministic on the V_G grid (both presets) and the illumination preset on the V_G × P grid.
  // Resilient: a failed point reopens the page and is retried once.
  async "device-grid"(browser, base, lang) {
    let app = await openApp(browser, base, "#tab=device&mode=deterministic", lang, "grid · load");
    const tasks = [];
    for (const preset of ["paper", "photo"]) for (const vg of VG_GRID) tasks.push({ preset, vg, p: null });
    const vgs = [...new Set((data.meta?.measured_photo_conditions ?? []).map((c) => c.vg))];
    for (const vg of vgs.length ? vgs : [-1.8]) for (const p of P_GRID) tasks.push({ preset: "photo", vg, p });
    let preset = null;
    let failures = 0;
    for (const t of tasks) {
      const label = `grid · ${t.preset} · V_G ${t.vg.toFixed(1)}${t.p === null ? "" : ` · P ${t.p} mW`}`;
      for (let attempt = 0; ; attempt++) {
        try {
          if (!app) {
            app = await openApp(browser, base, "#tab=device&mode=deterministic", lang, "grid · reload");
            preset = null;
          }
          const { page } = app;
          if (preset !== t.preset) {
            await click(page, `preset-${t.preset}`);
            preset = t.preset;
          }
          await step(page, label, async () => {
            await setField(page, "vg", "bias", t.p === null ? t.vg.toFixed(1) : String(t.vg));
            if (t.p !== null) await setField(page, "power_mW", "light", String(t.p));
            await runButton(page);
          }, { grid: true });
          break;
        } catch (e) {
          log(`  ${label}: ${e.message.split("\n")[0]} — reopening the page`);
          await app?.ctx.close().catch(() => undefined);
          app = null;
          if (attempt >= 1 || ++failures > 6) throw e;
        }
      }
    }
    await app?.ctx.close();
  },
  // Validation tab: reference-record I–V (auto on open), fast checks, the 8 illumination conditions.
  async validation(browser, base, lang) {
    const L = `${lang} · validation`;
    const { ctx, page } = await openApp(browser, base, "#tab=validation", lang, `${L} · reference I–V`);
    await step(page, `${L} · fast checks`, () => click(page, "val-fast"));
    await step(page, `${L} · illumination conditions`, () => click(page, "val-photo-run"));
    if (WITH_VAL_VG) await step(page, `${L} · V_G figure`, () => page.locator("[data-testid=panel-val-vg] button.btn").first().click());
    await ctx.close();
  },
};

// ---------------------------------------------------------------- output
function writeSnapshot() {
  fs.mkdirSync(OUT, { recursive: true });
  for (const f of fs.readdirSync(OUT)) if (/\.json(\.gz)?$/.test(f)) fs.rmSync(path.join(OUT, f));
  const put = (name, value) => {
    const text = jsonText(value);
    const gz = gzipSync(Buffer.from(text, "utf8"), { level: 9 });
    fs.writeFileSync(path.join(OUT, `${name}.json.gz`), gz);
    const ref = { file: `${name}.json.gz`, bytes: gz.length, json_bytes: Buffer.byteLength(text) };
    if (PLAIN) {
      fs.writeFileSync(path.join(OUT, `${name}.json`), text);
      ref.plain = `${name}.json`;
    }
    return ref;
  };
  const { jobs: _jobs, ...health } = data.health ?? {};
  void _jobs;
  const index = {
    format: 1,
    created: new Date().toISOString(),
    source: API,
    health,
    meta: data.meta,
    data: {},
    compute: [],
  };
  for (const name of ["measured", "design_map"]) if (data[name] !== undefined) index.data[name] = put(name, data[name]);
  const order = [...compute.values()].sort((a, b) => a.label.localeCompare(b.label));
  const near = (c) => (c.near ? { near: c.near } : {});
  for (const c of order.filter((x) => !x.grid)) index.compute.push({ kind: c.kind, key: c.key, label: c.label, ...near(c), ...put(c.key, c.result) });
  // grid results: bundles of neighbouring points per kind ({key: result}), fetched once per bundle
  const grid = order.filter((x) => x.grid).sort((a, b) => a.kind.localeCompare(b.kind) || (a.near?.sig ?? "").localeCompare(b.near?.sig ?? "") || (a.near?.p ?? 0) - (b.near?.p ?? 0) || (a.near?.vg ?? 0) - (b.near?.vg ?? 0));
  const bundles = [];
  for (const c of grid) {
    const size = Buffer.byteLength(jsonText(c.result));
    const last = bundles[bundles.length - 1];
    if (!last || last.kind !== c.kind || last.size + size > BUNDLE_JSON_BYTES) bundles.push({ kind: c.kind, size, items: [c] });
    else {
      last.items.push(c);
      last.size += size;
    }
  }
  const nth = {};
  for (const b of bundles) {
    nth[b.kind] = (nth[b.kind] ?? 0) + 1;
    const ref = put(`grid-${b.kind}-${nth[b.kind]}`, Object.fromEntries(b.items.map((c) => [c.key, c.result])));
    for (const c of b.items) {
      index.compute.push({ kind: c.kind, key: c.key, label: c.label, ...near(c), part: c.key, ...ref, json_bytes: Buffer.byteLength(jsonText(c.result)) });
    }
  }
  const indexText = JSON.stringify(index, null, 1).replace(/�/g, "\\ufffd");
  fs.writeFileSync(path.join(OUT, "index.json"), indexText);
  const files = fs.readdirSync(OUT).filter((f) => /\.json(\.gz)?$/.test(f));
  const total = files.reduce((s, f) => s + fs.statSync(path.join(OUT, f)).size, 0);
  return { index, files, total };
}

// ---------------------------------------------------------------- main
async function main() {
  for (const f of FLOWS) if (!FLOW_IMPL[f]) throw new Error(`unknown flow "${f}" (known: ${ALL_FLOWS.join(", ")})`);
  let h;
  try {
    h = await (await fetch(`${API}/api/health`, { signal: AbortSignal.timeout(5000) })).json();
  } catch (e) {
    throw new Error(`backend not reachable at ${API} (${e.message}). Start it first: uvicorn server.main:app --port 8000`);
  }
  if (!h?.ok) throw new Error(`backend at ${API} is not healthy`);
  log(`backend ${API} · engine ${h.version} · ${h.workers} workers`);
  const { vite, base } = await startVite();
  log(`vite dev server ${base} (proxy /api → ${API})`);
  const browser = await chromium.launch({ headless: !args.headed });
  try {
    for (const [li, lang] of LANGS.entries()) {
      for (const flow of FLOWS) {
        if (ONCE_FLOWS.has(flow) && li > 0) continue;
        log(`flow ${flow} (${lang})`);
        try {
          await FLOW_IMPL[flow](browser, base, lang);
        } catch (e) {
          problems.push(`flow ${flow} (${lang}) failed: ${e.message.split("\n")[0]}`);
          log(`  FAILED: ${e.message.split("\n")[0]}`);
        }
      }
    }
  } finally {
    await browser.close();
    vite.kill();
  }
  // data endpoints the flows did not touch (e.g. the design map panel was never opened)
  for (const [name, url] of Object.entries(DATA_ENDPOINTS)) {
    if (data[name] !== undefined) continue;
    log(`fetching ${url} directly`);
    data[name] = await (await fetch(`${API}${url}`)).json();
  }
  const { index, files, total } = writeSnapshot();
  const byKind = {};
  for (const c of index.compute) byKind[c.kind] = (byKind[c.kind] ?? 0) + 1;
  const byFile = new Map();
  for (const c of index.compute) if (!byFile.has(c.file)) byFile.set(c.file, c);
  const largest = [...byFile.values()].sort((a, b) => b.bytes - a.bytes).slice(0, 8);
  log(`wrote ${path.relative(process.cwd(), OUT) || OUT}: ${index.compute.length} results (${Object.entries(byKind).map(([k, n]) => `${k} ${n}`).join(", ")}), ${files.length} files, ${fmtMB(total)}; ${dupes} duplicate requests skipped`);
  for (const c of largest) log(`  ${fmtMB(c.bytes).padStart(9)} gz  ${c.part ? c.file : `${c.label} (json ${fmtMB(c.json_bytes)})`}`);
  if (mismatches) problems.push(`${mismatches} browser/node key mismatches — src/api/snapshot.ts and this script disagree`);
  if (total > LIMITS.bytes) problems.push(`snapshot is ${fmtMB(total)} > ${fmtMB(LIMITS.bytes)} budget`);
  if (files.length > LIMITS.files) problems.push(`snapshot has ${files.length} files > ${LIMITS.files}`);
  if (problems.length) {
    log(`${problems.length} problem(s):`);
    for (const p of problems) log(`  - ${p}`);
  }
  const fatal = mismatches || total > LIMITS.bytes || files.length > LIMITS.files || index.compute.length === 0;
  process.exitCode = fatal ? 1 : 0;
}

main().catch((e) => {
  console.error(`[snapshot] ${e.stack || e.message}`);
  process.exit(1);
});
