// Locked-build checks of scripts/verify-artifact.mjs (called when the build has a lock.json). Never prints the
// password; screenshots show the password field masked.
import { createDecipheriv, hkdfSync, randomBytes } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { decryptFile, deriveKey, isEncrypted } from "./lock-crypto.mjs";
import { LEAK_PATTERNS, numberFingerprints, scanLeaks, scanPassword, stockFonts } from "./lock-build.mjs";

const ENTRY = "stl-simulator.html";
const WRONG = "wrong-password-000";
const PUBLIC_KEYS = ["v", "kdf", "cipher", "format", "salt", "iterations", "app", "css", "index", "files"];
const DAY = 86_400_000;

/** The gate's encrypted app settings (scripts/lock/lock.js sealState), opened in Node. */
function openState(sealed, raw, id) {
  const key = Buffer.from(hkdfSync("sha256", raw, Buffer.alloc(32), "stl-lock app state v1", 32));
  const b = Buffer.from(sealed, "base64");
  const d = createDecipheriv("aes-256-gcm", key, b.subarray(0, 12), { authTagLength: 16 });
  d.setAAD(Buffer.from(`state:${id}`, "utf8"));
  d.setAuthTag(b.subarray(b.length - 16));
  return JSON.parse(Buffer.concat([d.update(b.subarray(12, b.length - 16)), d.final()]).toString("utf8"));
}

export async function lockedChecks({ WEB, browser, BASE, DIST, PASSWORD, check, watch, full, setPage, appFlow, text, waitText, visible, shot, csp }) {
  const cfg = JSON.parse(fs.readFileSync(path.join(DIST, "lock.json"), "utf8"));
  const listed = JSON.parse(fs.readFileSync(path.join(DIST, "files.json"), "utf8"));
  const KEY = `stl-lock:${cfg.salt}`;
  const STATE = `stl-lock-state:${cfg.salt}`;
  // the calibrated numbers, from the (decrypted) index: nothing on disk or in browser storage may contain them
  const raw = deriveKey(PASSWORD, Buffer.from(cfg.salt, "base64"), cfg.iterations);
  const index = cfg.index ? JSON.parse(gunzipSync(decryptFile(fs.readFileSync(path.join(DIST, cfg.index)), raw, cfg.index)).toString("utf8")) : null;
  const fingerprints = index ? numberFingerprints(index) : [];
  /** Leak patterns + calibrated numbers over browser-storage values → ["key: label"]. */
  const scanValues = (o) => {
    const hits = [];
    for (const [k, v] of Object.entries(o)) {
      const s = `${k}=${v}`;
      for (const [re, label] of LEAK_PATTERNS) if (re.test(s)) hits.push(`${k}: ${label}`);
      for (const fp of fingerprints) if (s.includes(fp)) hits.push(`${k}: calibration number ${fp}`);
    }
    return hits;
  };

  // ------------------------------------------------------------ static
  check(
    "lock.json holds only public parameters",
    Object.keys(cfg).every((k) => PUBLIC_KEYS.includes(k)) && cfg.iterations >= 600_000 && Buffer.from(cfg.salt, "base64").length === 16,
    `${cfg.kdf} · ${cfg.iterations} iterations · ${cfg.cipher} · ${cfg.files.length} encrypted files`,
  );
  const notEnc = cfg.files.filter((f) => !isEncrypted(fs.readFileSync(path.join(DIST, f))));
  check("every encrypted file starts with the STLENC1 magic", notEnc.length === 0, notEnc.slice(0, 3).join(", "));
  const paths = listed.map((f) => f.path);
  const badType = listed.filter((f) => f.path.endsWith(".bin") !== (f.contentType === "application/octet-stream"));
  const missing = [...cfg.files, "lock.js", "lock.css", "lock.json"].filter((f) => !paths.includes(f));
  check("files.json lists the gate + every encrypted file (.bin as application/octet-stream)", !badType.length && !missing.length, [...badType.map((f) => f.path), ...missing].join(", "));
  const plain = [ENTRY, ...paths].filter((f) => !cfg.files.includes(f));
  const leaks = scanLeaks(DIST, plain, { fingerprints, stock: stockFonts(WEB) });
  check("leak scan of every plaintext file (fonts = stock KaTeX)", leaks.length === 0, leaks.slice(0, 3).join(" | ") || `${plain.length} plaintext files, ${fingerprints.length} calibration numbers`);
  const entryHtml = fs.readFileSync(path.join(DIST, ENTRY), "utf8");
  const loads = [...entryHtml.matchAll(/<(?:link|script)\b[^>]*\b(?:href|src)="([^"]+)"/g)].map((m) => m[1]);
  const plainCode = plain.filter((f) => /\.(css|js|mjs|html?)$/.test(f) && !["lock.css", "lock.js", ENTRY].includes(f));
  check(
    "the only plain code is the gate: the entry loads lock.css + lock.js, the app stylesheet is encrypted",
    loads.join(" ") === "./lock.css ./lock.js" && plainCode.length === 0 && cfg.files.includes(cfg.css),
    `entry loads ${loads.join(", ")}${plainCode.length ? ` · plain: ${plainCode.join(", ")}` : ""}`,
  );
  const withPw = scanPassword(DIST, [ENTRY, "files.json", ...paths], PASSWORD);
  check("the password is in no published file", withPw.length === 0, withPw.join(", "));
  const idx = await fetch(`${BASE}/snapshot/index.json`);
  check("no plaintext snapshot/index.json", idx.status === 404, String(idx.status));

  // ------------------------------------------------------------ helpers
  const newPage = async (ctx, url = `${BASE}/`) => {
    const page = watch(await ctx.newPage());
    page.reqs = [];
    page.on("request", (r) => page.reqs.push(new URL(r.url()).pathname));
    setPage(page);
    await page.goto(url);
    return page;
  };
  const booted = (page) => page.evaluate(() => globalThis.__STL_BOOTED__ === true).catch(() => false);
  const bootMethod = (page) => page.evaluate(() => globalThis.__STL_LOCK_BOOT__ ?? null).catch(() => null);
  const waitBoot = (page, timeout = 30_000) =>
    page.locator("#stl-lock").waitFor({ state: "detached", timeout }).then(
      () => booted(page),
      () => false,
    );
  const ready = (page) => page.locator("#stl-lock-open:not([disabled])").waitFor({ timeout: 15_000 });
  const submit = async (page, pw, remember = false) => {
    await ready(page);
    await page.fill("#stl-lock-pw", pw);
    if (remember) await page.check("#stl-lock-remember");
    await page.click("#stl-lock-open");
  };
  const storage = (page) =>
    page.evaluate(() => {
      const dump = (name) => {
        const o = {};
        try {
          const s = window[name];
          for (let i = 0; i < s.length; i++) o[s.key(i)] = s.getItem(s.key(i));
        } catch {
          /* blocked */
        }
        return o;
      };
      return { local: dump("localStorage"), session: dump("sessionStorage") };
    });
  const rootEmpty = async (page) => (await page.locator("#root").innerHTML()) === "";
  /** The app's persisted settings as the app sees them (through the gate's storage layer). */
  const appSetting = (page) =>
    page.evaluate(() => {
      try {
        return JSON.parse(localStorage.getItem("stl-websim:v1") ?? "null");
      } catch {
        return null;
      }
    });
  const cssState = (page) =>
    page.evaluate(async () => {
      const sheet = document.adoptedStyleSheets?.[0];
      let font = "";
      for (const r of sheet?.cssRules ?? []) {
        const m = r instanceof CSSFontFaceRule ? /url\("?([^")]+\.woff2)"?\)/.exec(r.style.getPropertyValue("src")) : null;
        if (m) {
          font = m[1];
          break;
        }
      }
      const fontStatus = font ? await fetch(font).then((r) => r.status, () => 0) : 0;
      const fontLoaded = (await document.fonts.load("16px KaTeX_Main").catch(() => [])).length > 0;
      return { method: globalThis.__STL_LOCK_CSS__ ?? null, rules: sheet?.cssRules.length ?? 0, font: font.replace(location.origin, ""), fontStatus, fontLoaded };
    });
  const errText = async (page) => {
    await page.locator("#stl-lock-err").waitFor({ state: "visible", timeout: 20_000 }).catch(() => undefined);
    return ((await page.locator("#stl-lock-err").textContent().catch(() => "")) ?? "").trim();
  };

  // ------------------------------------------------------------ wrong password
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  let page = await newPage(ctx);
  await ready(page);
  check("password card shown, app root empty, no app running", (await page.locator("#stl-lock-form").isVisible()) && (await rootEmpty(page)) && !(await booted(page)));
  check("the password field has the focus (autofocus)", (await page.evaluate(() => document.activeElement?.id)) === "stl-lock-pw");
  for (let t = Date.now(); !page.reqs.includes(`/${cfg.app}`) && Date.now() - t < 5000; ) await page.waitForTimeout(100);
  check("the encrypted app downloads before the password is entered", page.reqs.includes(`/${cfg.app}`), cfg.app);
  await page.evaluate(() => {
    const f = document.getElementById("stl-lock-form");
    new MutationObserver(() => {
      if (f.getAttribute("aria-busy") === "true") window.__busySeen = (document.getElementById("stl-lock-open-label")?.textContent ?? "") || "busy";
    }).observe(f, { attributes: true });
  });
  await submit(page, WRONG);
  const wrong = await errText(page);
  check("wrong password → bilingual error line", /비밀번호가 맞지 않습니다/.test(wrong) && /Wrong password/.test(wrong), wrong);
  check("busy state while deriving the key", /여는 중|Opening/.test(String(await page.evaluate(() => window.__busySeen ?? ""))), String(await page.evaluate(() => window.__busySeen ?? "")));
  const snapBefore = page.reqs.filter((r) => r.startsWith("/snapshot/"));
  check("wrong password → the app never ran and fetched no data", !(await booted(page)) && (await rootEmpty(page)) && snapBefore.length === 0, `${snapBefore.length} data requests`);
  check(
    "wrong password → field marked invalid, focus back in it",
    (await page.getAttribute("#stl-lock-pw", "aria-invalid")) === "true" && (await page.evaluate(() => document.activeElement?.id)) === "stl-lock-pw",
  );
  await shot("lock-01-wrong-password");

  // ------------------------------------------------------------ right password (not remembered)
  await submit(page, PASSWORD);
  const ok = await waitBoot(page);
  check("right password → card removed, app booted", ok, `via ${await bootMethod(page)}`);
  const vlu = ok ? await waitText("kpi-vlu-value", /3\.704/, 30_000) : "";
  const vld = ok ? await text("kpi-vld-value") : "";
  check("auto-run shows the recorded folds V_LU 3.704 V / V_LD 2.598 V", /3\.704/.test(vlu) && /2\.598/.test(vld), `V_LU ${vlu} · V_LD ${vld}`);
  const status = ok ? await text("backend-status") : "";
  check("status dot says snapshot", /스냅샷|snapshot/i.test(status), status);
  await page.waitForTimeout(1500);
  const snapReqs = page.reqs.filter((r) => r.startsWith("/snapshot/"));
  check(
    "snapshot data decrypted: index.bin + result files fetched, no misses on screen",
    snapReqs.includes("/snapshot/index.bin") && snapReqs.length >= 2 && !(await visible("[data-testid=snapshot-miss]")),
    `${snapReqs.length} snapshot files`,
  );
  const css = await cssState(page);
  check(
    "the decrypted app stylesheet is applied (constructable stylesheet) and its KaTeX font URLs load",
    css.method === "adopted" && css.rules > 50 && css.fontStatus === 200 && css.fontLoaded,
    JSON.stringify(css),
  );
  const s1 = await storage(page);
  check("not remembered: the key is in sessionStorage only, never the password", !!s1.session[KEY] && !s1.local[KEY] && !JSON.stringify(s1).includes(PASSWORD));
  const plainApp = Object.keys({ ...s1.local, ...s1.session }).filter((k) => k.startsWith("stl-websim:"));
  const hits1 = scanValues({ ...s1.local, ...s1.session });
  let st = {};
  try {
    st = s1.local[STATE] ? openState(s1.local[STATE], raw, cfg.salt) : {};
  } catch {
    st = { error: 1 };
  }
  const inside = fingerprints.filter((fp) => JSON.stringify(st).includes(fp)).length;
  check(
    "the app's saved settings are only in localStorage ENCRYPTED (no plaintext stl-websim:* entry, no model string / calibrated number in any storage value)",
    plainApp.length === 0 && hits1.length === 0 && typeof st["stl-websim:v1"] === "string",
    `${plainApp.length} plain app entries · ${hits1.slice(0, 3).join(", ") || "0 hits"} · encrypted state: ${Object.keys(st).join(", ")} (${inside} calibrated numbers inside)`,
  );
  await shot("lock-02-unlocked");
  if (full) await appFlow();

  // settings survive a reload (decrypted from the encrypted entry)
  const before = await appSetting(page);
  const toggled = await page.getByTestId("lang-toggle").first().click({ timeout: 5000 }).then(() => true, () => false);
  await page.waitForTimeout(900);
  const want = await appSetting(page);
  await page.reload();
  const again = await waitBoot(page);
  const after = await appSetting(page);
  check("reload in the same tab reopens with the session key", again);
  check(
    "the app's settings survive the reload (encrypted at rest, decrypted by the gate)",
    again && !!after && after.lang === want?.lang && (!toggled || want?.lang !== before?.lang),
    `lang ${before?.lang} → ${want?.lang} → after reload ${after?.lang}`,
  );

  // browser closed (sessionStorage gone): nothing readable is left in localStorage
  await page.evaluate(() => sessionStorage.clear());
  await page.reload();
  await ready(page);
  const s4 = await storage(page);
  const hits4 = scanValues({ ...s4.local, ...s4.session });
  check(
    "session ended: card back, and localStorage holds no plaintext settings (only the encrypted entry)",
    !(await booted(page)) && Object.keys(s4.local).every((k) => !k.startsWith("stl-websim:")) && hits4.length === 0 && !!s4.local[STATE],
    `localStorage keys: ${Object.keys(s4.local).join(", ")} · ${hits4.slice(0, 3).join(", ") || "0 hits"}`,
  );
  // plaintext settings of an earlier plain build of the page are removed before any unlock
  await page.evaluate(() => localStorage.setItem("stl-websim:v1", JSON.stringify({ v: 1, lang: "en", note: "calib 0.2794" })));
  await page.reload();
  await ready(page);
  const s5 = await storage(page);
  check("plaintext app settings left by an earlier build are wiped by the gate", Object.keys(s5.local).every((k) => !k.startsWith("stl-websim:")), Object.keys(s5.local).join(", "));
  page = await newPage(ctx);
  await page.waitForTimeout(1500);
  check("a new tab asks again (not remembered)", (await page.locator("#stl-lock-pw").isVisible()) && !(await booted(page)) && (await rootEmpty(page)));
  await ctx.close();

  // ------------------------------------------------------------ remember (keyboard only), damaged key
  const ctx2 = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  page = await newPage(ctx2);
  await ready(page);
  await page.keyboard.type(PASSWORD);
  await page.keyboard.press("Tab");
  const onRemember = (await page.evaluate(() => document.activeElement?.id)) === "stl-lock-remember";
  await page.keyboard.press("Space");
  await page.keyboard.press("Tab");
  const onButton = (await page.evaluate(() => document.activeElement?.id)) === "stl-lock-open";
  await page.keyboard.press("Enter");
  check("keyboard only: type → Tab → Space (remember) → Tab → Enter opens the app", onRemember && onButton && (await waitBoot(page)), `focus order ok: ${onRemember && onButton}`);
  const s2 = await storage(page);
  let saved = {};
  try {
    saved = JSON.parse(s2.local[KEY] ?? "{}");
  } catch {
    /* checked below */
  }
  const days = (saved.exp - Date.now()) / DAY;
  check(
    "remembered: a 32-byte raw key in localStorage with a 30-day expiry, never the password",
    Buffer.from(String(saved.k ?? ""), "base64").length === 32 && days > 29.9 && days <= 30 && !JSON.stringify(s2).includes(PASSWORD),
    `expires in ${days.toFixed(2)} days`,
  );
  await page.close();
  page = await newPage(ctx2);
  check("remember survives closing the tab: a new tab opens without the password", await waitBoot(page));
  check("the app can lock itself (__STL_LOCK__.lock)", await page.evaluate(() => typeof globalThis.__STL_LOCK__?.lock === "function"));
  // expired key → asked again, entry removed
  await page.evaluate(([k, v]) => localStorage.setItem(k, v), [KEY, JSON.stringify({ k: saved.k, exp: Date.now() - 1000 })]);
  await page.reload();
  await ready(page).catch(() => undefined);
  const s3a = await storage(page);
  check("an expired remembered key is removed and the form comes back", (await page.locator("#stl-lock-pw").isEnabled()) && !s3a.local[KEY] && !(await booted(page)));
  // "#lock" in the URL at load → saved key forgotten, card shown with a note
  await submit(page, PASSWORD, true);
  await waitBoot(page);
  page = await newPage(ctx2, `${BASE}/#lock`);
  await ready(page);
  const note = ((await page.locator("#stl-lock-status").textContent().catch(() => "")) ?? "").trim();
  const s3b = await storage(page);
  check(
    "opening the link with #lock forgets the saved key on this browser and shows the card",
    !(await booted(page)) && !Object.keys({ ...s3b.local, ...s3b.session }).some((k) => k.startsWith("stl-lock:")) && /지웠습니다/.test(note) && !(await page.evaluate(() => location.hash)),
    note,
  );
  // "#lock" set while the app runs → reload to the card
  await submit(page, PASSWORD, true);
  await waitBoot(page);
  await Promise.all([page.waitForEvent("load", { timeout: 15_000 }).catch(() => undefined), page.evaluate(() => (location.hash = "lock")).catch(() => undefined)]);
  await ready(page).catch(() => undefined);
  const s3c = await storage(page);
  check("#lock while the app runs → back to the card, saved key gone", !(await booted(page)) && !s3c.local[KEY] && !s3c.session[KEY]);
  // damaged key → discarded
  await page.evaluate(([k, v]) => localStorage.setItem(k, v), [KEY, JSON.stringify({ k: randomBytes(32).toString("base64"), exp: Date.now() + DAY })]);
  await page.reload();
  await ready(page).catch(() => undefined);
  const s3 = await storage(page);
  check("a saved key that fails to decrypt is discarded and the form comes back", (await page.locator("#stl-lock-pw").isEnabled()) && !s3.local[KEY] && !(await booted(page)));
  await ctx2.close();

  // ------------------------------------------------------------ CSP matrix
  for (const v of Object.keys(csp)) {
    const c = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    page = await newPage(c, `${BASE}/?csp=${v}`);
    await submit(page, PASSWORD);
    if (v === "none") {
      const t = await errText(page);
      check(
        "CSP with none of blob:/'unsafe-inline'/'unsafe-eval' → bilingual CSP error in the card (no blank page)",
        /보안 정책/.test(t) && /security policy/.test(t) && (await page.locator("#stl-lock-form").isVisible()) && !(await booted(page)),
        t,
      );
      await shot("lock-05-csp-none");
    } else {
      const booted_ = await waitBoot(page);
      const how = await bootMethod(page);
      const val = booted_ ? await waitText("kpi-vlu-value", /3\.704/, 30_000) : "";
      const want = { blob: ["blob"], stylestrict: ["blob"], inline: ["inline"], eval: ["eval"] }[v] ?? ["blob", "inline"]; // nonce: the created script carries it
      const css = booted_ ? await cssState(page) : {};
      const label = v === "stylestrict" ? "blob: and style-src 'self' (no inline styles)" : `${csp[v]} only`;
      check(
        `CSP script-src 'self' + ${label} → the app boots (via ${want.join("/")}) with its stylesheet`,
        booted_ && want.includes(how) && /3\.704/.test(val) && css.method === "adopted" && css.rules > 50,
        `method ${how} · css ${css.method} (${css.rules} rules) · V_LU ${val}`,
      );
    }
    await c.close();
  }
  // no constructable stylesheets (older browsers) → the <style> fallback
  {
    const c = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    await c.addInitScript(() => delete Document.prototype.adoptedStyleSheets);
    page = await newPage(c, `${BASE}/?csp=blob`);
    await submit(page, PASSWORD);
    const ok = await waitBoot(page);
    const how = await page.evaluate(() => globalThis.__STL_LOCK_CSS__ ?? null);
    const rules = await page.evaluate(() => Array.from(document.querySelectorAll("style")).reduce((n, s) => n + (s.sheet?.cssRules.length ?? 0), 0));
    check("without constructable stylesheets the app stylesheet falls back to a <style> element", ok && how === "style" && rules > 50, `css ${how} · ${rules} rules`);
    await c.close();
  }

  // ------------------------------------------------------------ 390 px, light / dark / host data-theme
  const themes = [
    ["light", "", "light"],
    ["dark", "", "dark"],
    ["light", "dark", "dark"],
    ["dark", "light", "light"],
  ];
  for (const [scheme, attr, want] of themes) {
    const c = await browser.newContext({ viewport: { width: 390, height: 844 }, colorScheme: scheme, deviceScaleFactor: 2 });
    page = await newPage(c, `${BASE}/${attr ? `?theme=${attr}` : ""}`);
    await ready(page);
    const m = await page.evaluate(() => {
      const card = document.querySelector(".stl-lock__card");
      const r = card.getBoundingClientRect();
      return { sw: document.documentElement.scrollWidth, left: Math.round(r.left), right: Math.round(r.right), bg: getComputedStyle(card).backgroundColor };
    });
    const bg = want === "dark" ? "rgb(22, 25, 32)" : "rgb(255, 255, 255)";
    const label = `${scheme}${attr ? `+data-theme=${attr}` : ""}`;
    check(`390 px (${label}): card fits with 16 px gutters, ${want} colours`, m.sw <= 390 && m.left >= 16 && m.right <= 374 && m.bg === bg, JSON.stringify(m));
    await shot(`lock-0${themes.findIndex((t) => t[0] === scheme && t[1] === attr) + 6}-390-${label}`);
    await c.close();
  }
}
