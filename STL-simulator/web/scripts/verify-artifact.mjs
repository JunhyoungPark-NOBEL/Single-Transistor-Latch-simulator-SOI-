#!/usr/bin/env node
// Serves web/dist-artifact/ the way the artifact host does (the entry fragment wrapped in a bare HTML skeleton,
// static files only, /api → 404) and checks the snapshot page in Chromium:
//   snapshot banner · reference preset deterministic folds · illumination stochastic mean ± SD with the measured
//   overlay · a schematic example run with real waveforms · a changed V_G → "not in the snapshot" notice.
// A LOCKED build (lock.json present, see scripts/lock-build.mjs) is checked instead for: nothing readable without
// the password (static scan, no plaintext index, the app never runs on a wrong password), the right password boots
// the app with real (decrypted) snapshot results, the saved key (session / "remember" / damaged key discarded), and
// a CSP matrix — the page served with a Content-Security-Policy that allows only blob: scripts, only
// 'unsafe-inline' or only 'unsafe-eval' must still boot; one that allows none must show the bilingual error.
// Usage: node scripts/verify-artifact.mjs [--shots <dir>] [--port 5211] [--dist dist-artifact] [--serve]
//        locked build: STL_ARTIFACT_PASSWORD=... node scripts/verify-artifact.mjs [--full]   (the password is read
//        only from the variable — a command-line flag would show in `ps` and the shell history, so --password is
//        refused). --full also runs the plain app flow after unlocking.
//        --serve only serves (?csp=blob|inline|eval|nonce|strict|stylestrict|none adds that CSP, ?theme=dark|light
//        sets html[data-theme]).
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);
const opt = (n, d) => {
  const i = argv.indexOf(`--${n}`);
  return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[i + 1] : d;
};
const DIST = path.resolve(WEB, opt("dist", "dist-artifact"));
const SHOTS = path.resolve(opt("shots", path.join(WEB, "test-results/artifact")));
const PORT = Number(opt("port", 0));
const TYPES = { ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".gz": "application/gzip", ".woff2": "font/woff2", ".svg": "image/svg+xml", ".html": "text/html", ".bin": "application/octet-stream", ".wasm": "application/wasm" };
const LOCKED = fs.existsSync(path.join(DIST, "lock.json"));
if (argv.some((a) => a === "--password" || a.startsWith("--password="))) {
  console.error("[verify] the password is read only from STL_ARTIFACT_PASSWORD, not from the command line");
  process.exit(2);
}
const PASSWORD = process.env.STL_ARTIFACT_PASSWORD ?? "";
if (LOCKED && !PASSWORD && !argv.includes("--serve")) {
  console.error("[verify] locked build: pass the password in STL_ARTIFACT_PASSWORD");
  process.exit(2);
}

// CSP variants for the locked build (base = what the page must live with; each variant adds one script source)
const CSP_BASE = [
  "default-src 'self'",
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
  "font-src 'self' data: https://fonts.gstatic.com",
  "img-src 'self' data: blob:",
  "connect-src 'self'",
  "worker-src 'self' blob:",
];
const NONCE = "c3RsLXZlcmlmeS1ub25jZQ";
// stylestrict: blob: scripts, and style-src 'self' only (no inline styles) — the decrypted app stylesheet must still apply
const CSP_EXTRA = { blob: "blob:", inline: "'unsafe-inline'", eval: "'unsafe-eval'", nonce: `'nonce-${NONCE}'`, strict: `'nonce-${NONCE}' 'strict-dynamic'`, stylestrict: "blob:", none: "" };
const cspHeader = (v) =>
  [`script-src 'self' https://cdnjs.cloudflare.com${CSP_EXTRA[v] ? ` ${CSP_EXTRA[v]}` : ""}`, ...CSP_BASE]
    .map((d) => (v === "stylestrict" && d.startsWith("style-src") ? "style-src 'self'" : d))
    .join("; ");
/** nonce variants: the host would stamp its nonce on the page's own script tags */
const withNonce = (html, v) => (v === "nonce" || v === "strict" ? html.replace(/<script\b/g, `<script nonce="${NONCE}"`) : html);

// ---------------------------------------------------------------- static host
const entry = fs.readFileSync(path.join(DIST, "stl-simulator.html"), "utf8");
const page = (theme) =>
  `<!doctype html>\n<html lang="en"${theme ? ` data-theme="${theme}"` : ""}><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head>\n<body>\n${entry}</body></html>\n`;
const server = http.createServer((req, res) => {
  const url = decodeURIComponent((req.url ?? "/").split(/[?#]/)[0]);
  if (url === "/" || url === "/stl-simulator.html") {
    const q = new URL(req.url ?? "/", "http://x").searchParams;
    const headers = { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" };
    if (q.get("csp") in CSP_EXTRA) headers["Content-Security-Policy"] = cspHeader(q.get("csp"));
    res.writeHead(200, headers);
    return res.end(withNonce(page(/^(dark|light)$/.test(q.get("theme") ?? "") ? q.get("theme") : ""), q.get("csp")));
  }
  const file = path.join(DIST, path.normalize(url).replace(/^([/\\])+/, ""));
  if (!url.startsWith("/api/") && file.startsWith(DIST) && fs.existsSync(file) && fs.statSync(file).isFile()) {
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream" });
    return res.end(fs.readFileSync(file));
  }
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end('{"detail":"Not Found"}');
});
await new Promise((r) => server.listen(PORT, "127.0.0.1", r));
const BASE = `http://127.0.0.1:${server.address().port}`;
console.log(`[verify] serving ${path.relative(WEB, DIST)}${LOCKED ? " (LOCKED build)" : ""} at ${BASE}/ (no /api)`);
if (argv.includes("--serve")) await new Promise(() => undefined);

// ---------------------------------------------------------------- checks
fs.mkdirSync(SHOTS, { recursive: true });
const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  console.log(`[verify] ${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
};
const api = await fetch(`${BASE}/api/health`);
check("/api/health is 404 on the static host", api.status === 404, String(api.status));

const browser = await chromium.launch();
const errors = [];
const foreign = new Set();
/** The page the helpers below act on (the locked checks switch between several). */
let p = null;
const watch = (page) => {
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("request", (r) => {
    const u = new URL(r.url());
    if (u.origin !== BASE && !u.protocol.startsWith("data") && !u.protocol.startsWith("blob")) foreign.add(u.origin);
  });
  return page;
};
const shot = async (name) => {
  await p.waitForTimeout(700);
  await p.screenshot({ path: path.join(SHOTS, `${name}.png`) });
};
const text = async (id) => ((await p.getByTestId(id).first().textContent({ timeout: 5000 }).catch(() => "")) ?? "").replace(/\s+/g, " ").trim();
async function waitText(id, re, timeout = 20_000) {
  const end = Date.now() + timeout;
  let s = "";
  while (Date.now() < end) {
    s = await text(id);
    if (re.test(s)) return s;
    await p.waitForTimeout(200);
  }
  return s;
}
const visible = (sel) => p.locator(sel).first().isVisible().catch(() => false);

/** The plain snapshot page flow (also run after unlocking a locked build with --full). */
async function appFlow() {
  await p.getByTestId("snapshot-banner").waitFor({ timeout: 20_000 });
  check("snapshot banner shown", true, await text("snapshot-banner"));
  check("status dot says snapshot", /스냅샷|snapshot/i.test(await text("backend-status")), await text("backend-status"));
  await shot("01-snapshot-banner");

  // reference preset, deterministic
  await p.getByTestId("mode-deterministic").click();
  await p.getByTestId("preset-paper").click();
  await p.getByTestId("run-button").click();
  const vlu = await waitText("kpi-vlu-value", /3\.704/);
  const vld = await text("kpi-vld-value");
  check("reference deterministic folds 3.704 / 2.598 V", /3\.704/.test(vlu) && /2\.598/.test(vld), `V_LU ${vlu} · V_LD ${vld}`);
  await p.locator("[data-testid=panel-charge-balance] .js-plotly-plot").waitFor({ timeout: 20_000 }).catch(() => undefined);
  check("deterministic panels are snapshot hits", !(await visible("[data-testid=snapshot-miss]")));
  await shot("02-device-det-reference");

  // illumination preset, stochastic
  await p.getByTestId("mode-stochastic").click();
  await p.getByTestId("preset-photo").click();
  await p.getByTestId("run-button").click();
  const mean = await waitText("kpi-vlu-value", /3\.80\d.*±\s*17\d\.\d/);
  check("illumination stochastic mean ≈ 3.808 V ± 173.8 mV", /3\.808/.test(mean) && /173\.8/.test(mean), mean);
  await p.locator("[data-testid=panel-dist] .js-plotly-plot").waitFor({ timeout: 20_000 }).catch(() => undefined);
  const dist = ((await p.getByTestId("panel-dist").textContent()) ?? "").toLowerCase();
  check("measured overlay in the distribution panel", dist.includes("측정") || dist.includes("measured"));
  check("stochastic panels are snapshot hits", !(await visible("[data-testid=snapshot-miss]")));
  await shot("03-device-sto-illumination");

  // schematic example (circuit tab opens on the editor with the load-line example)
  await p.getByTestId("tab-circuit").click();
  await p.getByTestId("mode-deterministic").click();
  await p.getByTestId("schematic-view").waitFor({ timeout: 20_000 });
  await p.getByTestId("menu-examples").click();
  await p.locator("[data-testid^=tpl-]").first().click();
  await p.getByTestId("run-button").click();
  await p.waitForTimeout(400);
  if (await visible("[data-testid=sch-confirm-run]")) await p.getByTestId("sch-confirm-run").click();
  await p.locator("[data-testid=panel-sch-waves] .js-plotly-plot").waitFor({ timeout: 30_000 }).catch(() => undefined);
  const waves = await visible("[data-testid=panel-sch-waves] .js-plotly-plot");
  const miss = await visible("[data-testid=panel-sch-waves] [data-testid=snapshot-miss]");
  check("schematic example shows recorded waveforms", waves && !miss, `plot ${waves} · miss notice ${miss}`);
  await shot("04-circuit-example");

  // V_G between grid points → nearest precomputed V_G (real model result, marked)
  await p.getByTestId("tab-device").click();
  await p.getByTestId("mode-deterministic").click();
  await p.getByTestId("preset-paper").click();
  const vg = p.locator("[data-testid=field-vg] input.input").first();
  await vg.fill("-1.53");
  await vg.press("Enter");
  await p.getByTestId("run-button").click();
  await p.locator("[data-testid=panel-iv] [data-testid=snapshot-near]").waitFor({ timeout: 20_000 }).catch(() => undefined);
  const near = await text("snapshot-near");
  const vluNear = await waitText("kpi-vlu-value", /^(?!3\.704)/);
  check("V_G −1.53 V uses the nearest precomputed V_G (−1.50 V)", /−1\.50/.test(near) && !(await visible("[data-testid=snapshot-miss]")), `${near} · V_LU ${vluNear}`);
  check("banner mentions the nearest precomputed V_G", await visible("[data-testid=snapshot-banner-near]"));
  await shot("05-nearest-vg");

  // V_G far outside the grid → not in the snapshot → example data, clearly marked
  await vg.fill("-5.2");
  await vg.press("Enter");
  await p.getByTestId("run-button").click();
  await p.locator("[data-testid=panel-iv] [data-testid=snapshot-miss]").waitFor({ timeout: 20_000 }).catch(() => undefined);
  const note = await text("snapshot-miss");
  check("V_G outside the grid shows the not-in-snapshot notice", /스냅샷에 없어|not in the static snapshot/.test(note), note);
  check("banner flags example data on screen", await visible("[data-testid=snapshot-banner-some]"));
  await shot("06-changed-vg-miss");

  // English
  await p.getByTestId("lang-toggle").click();
  await p.waitForTimeout(300);
  check("English banner", /Static snapshot/.test(await text("snapshot-banner")), await text("snapshot-banner"));
  await shot("07-english-miss");
}

try {
  if (LOCKED) {
    const { lockedChecks } = await import("./verify-lock.mjs");
    await lockedChecks({
      WEB, browser, BASE, DIST, PASSWORD, check, watch, shot, text, waitText, visible, appFlow,
      csp: CSP_EXTRA, full: argv.includes("--full"), setPage: (page) => (p = page),
    });
  } else {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    p = watch(await ctx.newPage());
    await p.goto(`${BASE}/`);
    await appFlow();
  }
} catch (e) {
  check("script", false, e.message.split("\n")[0]);
  await shot("99-failure").catch(() => undefined);
} finally {
  check("no page errors", errors.length === 0, errors.slice(0, 3).join(" | "));
  check("no requests to other hosts", foreign.size === 0, [...foreign].join(", "));
  await browser.close();
  server.close();
}
const failed = results.filter((r) => !r.ok);
console.log(`[verify] ${results.length - failed.length}/${results.length} checks passed · screenshots in ${SHOTS}`);
process.exit(failed.length ? 1 : 0);
