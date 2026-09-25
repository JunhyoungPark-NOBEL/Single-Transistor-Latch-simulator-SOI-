// STL Simulator: password gate of the locked artifact build (npm run build:artifact -- --lock).
// This file and lock.css are the only plain script/style of the page; the app bundle, the app stylesheet and every
// data file are encrypted (format: scripts/lock-crypto.mjs). Flow:
//   lock.json {salt, iterations, app, css} → the encrypted app + stylesheet start downloading at once (while the
//   user types) → a key saved on this browser (sessionStorage, or localStorage for 30 days when "remember" was
//   ticked; key "stl-lock:<salt>" = {k: raw AES key, exp}, never the password) or PBKDF2-SHA256(password) →
//   AES-256-GCM decrypt (AAD = published path) → gunzip → app stylesheet applied (constructable stylesheet →
//   <style> → blob: link) → the app's own localStorage entries ("stl-websim:*") are served from memory and saved
//   ENCRYPTED (one entry "stl-lock-state:<salt>", AES-GCM with an HKDF subkey of the page key), so the saved
//   settings — which hold the model parameters — are never on disk in plaintext → globalThis.__STL_LOCK__ =
//   { key, lock } → run the classic IIFE bundle with the first method the page's Content-Security-Policy allows:
//   <script src="blob:…"> → inline <script> → new Function / indirect eval (the created scripts carry this
//   script's nonce, if the host uses one).
//   Success = globalThis.__STL_BOOTED__ (set at the top of src/main.tsx); then the card is removed.
// "#lock" in the URL (at load, or set while the app runs) or __STL_LOCK__.lock() forgets the saved key on this
// browser and shows the card again.
// The pure helpers are exported for the unit tests (src/api/lock.test.ts); the gate runs only on a page with #stl-lock.

export const MAGIC = new Uint8Array([0x53, 0x54, 0x4c, 0x45, 0x4e, 0x43, 0x31, 0x00]); // "STLENC1\0"
const IV_BYTES = 12;
const TAG_BYTES = 16;
export const STORE_PREFIX = "stl-lock:";
export const STATE_PREFIX = "stl-lock-state:";
export const APP_PREFIX = "stl-websim:";
const DAY = 86_400_000;
export const REMEMBER_MS = 30 * DAY;
export const SESSION_MS = DAY;

export const lockAad = (p) => String(p).replace(/^(\.\/)+/, "");
export const isEncrypted = (b) => b.length >= MAGIC.length + IV_BYTES + TAG_BYTES && MAGIC.every((x, i) => b[i] === x);
export const isGzip = (b) => b.length >= 2 && b[0] === 0x1f && b[1] === 0x8b;

export function b64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
export function unb64(s) {
  return Uint8Array.from(atob(String(s)), (c) => c.charCodeAt(0));
}

/** PBKDF2-HMAC-SHA256(NFC(password), salt, iterations) → 32 raw key bytes (same as scripts/lock-crypto.mjs). */
export async function deriveKeyBytes(password, salt, iterations) {
  const subtle = globalThis.crypto.subtle;
  const base = await subtle.importKey("raw", new TextEncoder().encode(String(password).normalize("NFC")), "PBKDF2", false, ["deriveBits"]);
  return new Uint8Array(await subtle.deriveBits({ name: "PBKDF2", hash: "SHA-256", salt, iterations }, base, 256));
}

/** Non-extractable AES-GCM decrypt key from raw bytes. */
export const importAesKey = (raw) => globalThis.crypto.subtle.importKey("raw", raw, { name: "AES-GCM" }, false, ["decrypt"]);

/** Decrypt one file; rejects (OperationError) on a wrong key, another path or any modified byte. */
export async function decryptFile(bytes, key, aad) {
  if (!isEncrypted(bytes)) throw new Error("not an encrypted file");
  const iv = bytes.slice(MAGIC.length, MAGIC.length + IV_BYTES);
  const body = bytes.subarray(MAGIC.length + IV_BYTES);
  const params = { name: "AES-GCM", iv, additionalData: new TextEncoder().encode(lockAad(aad)), tagLength: TAG_BYTES * 8 };
  return new Uint8Array(await globalThis.crypto.subtle.decrypt(params, key, body));
}

export async function gunzip(bytes) {
  if (!isGzip(bytes)) return bytes;
  if (typeof globalThis.DecompressionStream !== "function") throw new LockError("old");
  const stream = new Blob([bytes]).stream().pipeThrough(new globalThis.DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// ---------------------------------------------------------------- messages (bilingual, one line each)
export const T = {
  open: "열기 / Open",
  busy: "여는 중… / Opening…",
  saved: "저장된 키로 여는 중… / Opening with the saved key…",
  locked: "이 브라우저에 저장된 키를 지웠습니다 / The key saved on this browser was removed",
  empty: "비밀번호를 입력하세요 / Enter the password",
  wrong: "비밀번호가 맞지 않습니다 / Wrong password",
  net: "파일을 받지 못했습니다. 연결을 확인하고 다시 시도하세요 / Could not download the page files; check the connection and try again",
  config: "잠금 설정(lock.json)을 읽지 못했습니다 / Could not read the lock settings (lock.json)",
  crypto: "이 브라우저에서는 열 수 없습니다 (Web Crypto 없음, HTTPS 필요) / This browser cannot open the page (no Web Crypto; HTTPS is required)",
  old: "브라우저가 너무 오래되었습니다 (DecompressionStream 없음) / This browser is too old (no DecompressionStream)",
  csp: "이 페이지의 보안 정책(CSP)이 앱 실행을 막았습니다. 다른 브라우저로 열어 보세요 / The page's security policy (CSP) blocked the app from starting; try another browser",
  crash: "앱을 시작하지 못했습니다. 새로고침해 보세요 / The app failed to start; try reloading",
};
export class LockError extends Error {
  constructor(code) {
    super(T[code] ?? String(code));
    this.code = code;
  }
}

// ---------------------------------------------------------------- browser storage (every access guarded)
function storage(name) {
  try {
    return globalThis[name] ?? null;
  } catch {
    return null; // storage blocked (private mode, sandbox)
  }
}
function stores() {
  const out = [];
  for (const name of ["sessionStorage", "localStorage"]) {
    const s = storage(name);
    if (s) out.push([name, s]);
  }
  return out;
}
/** Keys of `s` starting with `prefix` (snapshot: safe to remove while iterating). */
function keysOf(s, prefix) {
  const out = [];
  try {
    for (let i = 0; i < s.length; i++) {
      const k = s.key(i);
      if (k && k.startsWith(prefix)) out.push(k);
    }
  } catch {
    /* ignore */
  }
  return out;
}
function remove(s, k) {
  try {
    s.removeItem(k);
  } catch {
    /* ignore */
  }
}

// ---------------------------------------------------------------- saved key (per salt = per build)
/** The saved raw key for this build, or null. Expired, damaged or old-format entries are removed. */
export function readSavedKey(id, now = Date.now()) {
  for (const [, s] of stores()) {
    let v = null;
    try {
      v = s.getItem(STORE_PREFIX + id);
    } catch {
      continue;
    }
    if (!v) continue;
    try {
      const o = JSON.parse(v);
      const raw = unb64(o?.k);
      if (raw.length === 32 && Number.isFinite(o.exp) && o.exp > now) return raw;
    } catch {
      /* damaged: removed below */
    }
    remove(s, STORE_PREFIX + id);
  }
  return null;
}
export function forgetKey(id) {
  for (const [, s] of stores()) remove(s, STORE_PREFIX + id);
}
/** Forget every saved key (all builds) on this browser. */
export function forgetAll() {
  for (const [, s] of stores()) for (const k of keysOf(s, STORE_PREFIX)) remove(s, k);
}
export function saveKey(id, raw, remember, now = Date.now()) {
  forgetKey(id);
  for (const [name, s] of stores()) {
    // keys of earlier builds (other salts) can never open this page again
    for (const k of keysOf(s, STORE_PREFIX)) if (k !== STORE_PREFIX + id) remove(s, k);
    if (name !== (remember ? "localStorage" : "sessionStorage")) continue;
    try {
      s.setItem(STORE_PREFIX + id, JSON.stringify({ k: b64(raw), exp: now + (remember ? REMEMBER_MS : SESSION_MS) }));
    } catch {
      /* storage full or blocked: the page just asks again next time */
    }
  }
}

// ---------------------------------------------------------------- the app's saved settings, encrypted at rest
/** AES-GCM key for the app's saved settings: HKDF-SHA256 subkey of the page key (never the file key itself). */
export async function stateKey(raw) {
  const subtle = globalThis.crypto.subtle;
  const base = await subtle.importKey("raw", raw, "HKDF", false, ["deriveKey"]);
  const params = { name: "HKDF", hash: "SHA-256", salt: new Uint8Array(32), info: new TextEncoder().encode("stl-lock app state v1") };
  return subtle.deriveKey(params, base, { name: "AES-GCM", length: 256 }, false, ["encrypt", "decrypt"]);
}
/** {key: value} → base64(IV · ciphertext · tag), AAD "state:<id>". */
export async function sealState(obj, key, id) {
  const iv = globalThis.crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const params = { name: "AES-GCM", iv, additionalData: new TextEncoder().encode(`state:${id}`), tagLength: TAG_BYTES * 8 };
  const ct = new Uint8Array(await globalThis.crypto.subtle.encrypt(params, key, new TextEncoder().encode(JSON.stringify(obj))));
  const out = new Uint8Array(IV_BYTES + ct.length);
  out.set(iv);
  out.set(ct, IV_BYTES);
  return b64(out);
}
/** Inverse of sealState; rejects on another key / id or any modified byte. Keeps string values under APP_PREFIX. */
export async function openState(sealed, key, id) {
  const b = unb64(sealed);
  const params = { name: "AES-GCM", iv: b.slice(0, IV_BYTES), additionalData: new TextEncoder().encode(`state:${id}`), tagLength: TAG_BYTES * 8 };
  const o = JSON.parse(new TextDecoder().decode(await globalThis.crypto.subtle.decrypt(params, key, b.subarray(IV_BYTES))));
  return new Map(Object.entries(o && typeof o === "object" ? o : {}).filter(([k, v]) => k.startsWith(APP_PREFIX) && typeof v === "string"));
}
/** Plaintext app entries left on this browser (an earlier plain build of the page, or a gate before this one):
 *  removed from both storages at once; returns the localStorage ones so an unlock can keep them (encrypted). */
export function takePlainAppEntries() {
  const found = new Map();
  for (const [name, s] of stores()) {
    for (const k of keysOf(s, APP_PREFIX)) {
      try {
        const v = s.getItem(k);
        if (name === "localStorage" && typeof v === "string") found.set(k, v);
      } catch {
        /* ignore */
      }
      remove(s, k);
    }
  }
  return found;
}
/** Encrypted settings of other builds (other salts) can never be opened again. */
export function dropOtherStates(id) {
  const ls = storage("localStorage");
  if (ls) for (const k of keysOf(ls, STATE_PREFIX)) if (k !== STATE_PREFIX + id) remove(ls, k);
}
/** From now on localStorage.getItem/setItem/removeItem of "stl-websim:*" keys (the app's) use `state` (in memory)
 *  and call `persist()` on a change; every other key and sessionStorage behave as before. */
export function protectAppStorage(state, persist) {
  const ls = storage("localStorage");
  const S = globalThis.Storage;
  if (!ls || typeof S !== "function") return false;
  const P = S.prototype;
  const orig = { getItem: P.getItem, setItem: P.setItem, removeItem: P.removeItem };
  const ours = (self, k) => self === ls && String(k).startsWith(APP_PREFIX);
  P.getItem = function getItem(k) {
    if (!ours(this, k)) return orig.getItem.apply(this, arguments);
    const v = state.get(String(k));
    return v === undefined ? null : v;
  };
  P.setItem = function setItem(k, v) {
    if (!ours(this, k)) return orig.setItem.apply(this, arguments);
    state.set(String(k), String(v));
    persist();
  };
  P.removeItem = function removeItem(k) {
    if (!ours(this, k)) return orig.removeItem.apply(this, arguments);
    if (state.delete(String(k))) persist();
  };
  return true;
}
/** Serialised writer: every call (re)saves the whole `state` encrypted; a burst of calls saves once or twice. */
export function stateWriter(state, key, id, write) {
  let chain = Promise.resolve();
  let queued = false;
  return () => {
    if (queued) return chain;
    queued = true;
    chain = chain
      .then(async () => {
        queued = false;
        write(STATE_PREFIX + id, await sealState(Object.fromEntries(state), key, id));
      })
      .catch(() => undefined); // storage full / blocked: settings just are not kept
    return chain;
  };
}

// ---------------------------------------------------------------- the app stylesheet (decrypted)
/** Relative url(...) references → absolute (against `base`), so the stylesheet works from any container. */
export function absolutizeCss(css, base) {
  return css.replace(/url\(\s*(["']?)(?![a-z][a-z0-9+.-]*:|\/|#)([^"')]+)\1\s*\)/gi, (m, q, u) => {
    try {
      return `url(${q}${new URL(u, base).href}${q})`;
    } catch {
      return m;
    }
  });
}

// ---------------------------------------------------------------- running the decrypted bundle
const G = globalThis;
const ran = () => G.__STL_RAN__ === 1;
const booted = () => G.__STL_BOOTED__ === true;
const parent = () => document.head || document.documentElement;
/** Nonce of this script's own element, if the host uses a nonce-based CSP: scripts/styles created here carry it too. */
function ownNonce() {
  try {
    const el = Array.from(document.scripts).find((s) => /(^|\/)lock\.js([?#]|$)/.test(s.getAttribute("src") ?? ""));
    return (el && (el.nonce || el.getAttribute("nonce"))) || "";
  } catch {
    return "";
  }
}
function newEl(tag) {
  const s = document.createElement(tag);
  const nonce = ownNonce();
  if (nonce) s.nonce = nonce;
  return s;
}

/** Apply the stylesheet with the first method the page allows: a constructable stylesheet (CSSOM — not subject to
 *  style-src), a <style> element, a blob: <link>. Resolves the method, or null. */
export async function applyCss(css) {
  try {
    if (typeof CSSStyleSheet === "function" && "adoptedStyleSheets" in document) {
      const sheet = new CSSStyleSheet();
      sheet.replaceSync(css);
      document.adoptedStyleSheets = [...document.adoptedStyleSheets, sheet];
      return "adopted";
    }
  } catch {
    /* next method */
  }
  try {
    const s = newEl("style");
    s.textContent = css;
    parent().appendChild(s);
    if (s.sheet && s.sheet.cssRules.length > 0) return "style";
    s.remove(); // blocked by the CSP
  } catch {
    /* next method */
  }
  return new Promise((resolve) => {
    let url = "";
    const done = (how, link) => {
      clearTimeout(timer);
      if (!how && link) link.remove();
      try {
        URL.revokeObjectURL(url);
      } catch {
        /* ignore */
      }
      resolve(how);
    };
    const timer = setTimeout(() => done(null), 10000);
    try {
      url = URL.createObjectURL(new Blob([css], { type: "text/css" }));
      const l = newEl("link");
      l.rel = "stylesheet";
      l.onload = () => done("blob");
      l.onerror = () => done(null, l);
      l.href = url;
      parent().appendChild(l);
    } catch {
      done(null);
    }
  });
}

function viaBlob(src) {
  return new Promise((resolve) => {
    let url = "";
    let done = false;
    let timer = 0;
    const finish = (how) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      try {
        URL.revokeObjectURL(url);
      } catch {
        /* ignore */
      }
      resolve(how);
    };
    timer = setTimeout(() => finish(ran() ? "ran" : "blocked"), 20000);
    try {
      url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
      const s = newEl("script");
      s.onload = () => finish(ran() ? "ran" : "blocked");
      s.onerror = () => {
        s.remove();
        finish(ran() ? "ran" : "blocked"); // a CSP block fires "error" without running anything
      };
      s.src = url;
      parent().appendChild(s);
    } catch {
      finish("blocked"); // e.g. Trusted Types
    }
  });
}

function viaInline(src) {
  try {
    const s = newEl("script");
    s.textContent = src; // runs synchronously on insertion unless the CSP blocks it (then silently nothing)
    parent().appendChild(s);
    s.remove();
  } catch {
    /* Trusted Types: blocked */
  }
  return ran() ? "ran" : "blocked";
}

function viaEval(src) {
  let fn = null;
  try {
    fn = new Function(src);
  } catch {
    fn = null; // EvalError under a CSP without 'unsafe-eval'
  }
  try {
    if (fn) fn();
    else (0, eval)(src);
  } catch {
    /* blocked, or the app threw (reported through the window error event) */
  }
  return ran() ? "ran" : "blocked";
}

/** Run the bundle; resolves { method: "blob" | "inline" | "eval" | null, booted, error }. */
export async function runBundle(code) {
  const src = `globalThis.__STL_RAN__=1;\n${code}`;
  let error = "";
  const onError = (e) => {
    error ||= String(e?.error?.message ?? e?.message ?? "");
  };
  addEventListener("error", onError);
  try {
    for (const [method, run] of [["blob", viaBlob], ["inline", viaInline], ["eval", viaEval]]) {
      if ((await run(src)) === "ran") {
        G.__STL_LOCK_BOOT__ = method;
        return { method, booted: booted(), error };
      }
    }
    return { method: null, booted: false, error };
  } finally {
    removeEventListener("error", onError);
  }
}

// ---------------------------------------------------------------- the gate
const LOCK_HASH = /^#lock$/i;
/** Forget the saved key and show the card again (reload with #lock). */
function lockNow() {
  forgetAll();
  try {
    if (!LOCK_HASH.test(location.hash)) history.replaceState(history.state, "", `${location.pathname}${location.search}#lock`);
  } catch {
    /* ignore */
  }
  location.reload();
}

async function gate() {
  const $ = (id) => document.getElementById(id);
  const card = $("stl-lock");
  const form = $("stl-lock-form");
  const pw = $("stl-lock-pw");
  const remember = $("stl-lock-remember");
  const btn = $("stl-lock-open");
  const label = $("stl-lock-open-label");
  const err = $("stl-lock-err");
  const status = $("stl-lock-status");
  let busy = false;

  const say = (el, text) => {
    el.textContent = text || "";
    el.hidden = !text;
  };
  const setBusy = (on, text = "") => {
    busy = on;
    form.setAttribute("aria-busy", on ? "true" : "false");
    btn.disabled = on;
    pw.readOnly = on;
    remember.disabled = on;
    label.textContent = on ? T.busy : T.open;
    say(status, text);
  };
  const showError = (text, invalid = false) => {
    say(err, text);
    if (invalid) pw.setAttribute("aria-invalid", "true");
    else pw.removeAttribute("aria-invalid");
  };
  const fatal = (text, detail = "") => {
    setBusy(false);
    showError(detail ? `${text} (${detail})` : text);
    btn.disabled = true;
    pw.disabled = true;
    remember.disabled = true;
    card.dataset.state = "error";
  };

  // "#lock": forget the saved key(s) of this browser first
  let lockedNow = false;
  if (LOCK_HASH.test(location.hash)) {
    forgetAll();
    lockedNow = true;
    try {
      history.replaceState(history.state, "", `${location.pathname}${location.search}`);
    } catch {
      /* ignore */
    }
  }
  // plaintext app settings must not stay on this browser (they are kept encrypted after an unlock)
  const leftovers = takePlainAppEntries();

  if (!G.crypto?.subtle || typeof TextEncoder !== "function") return fatal(T.crypto);

  let cfg;
  let salt;
  try {
    const res = await fetch("lock.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(String(res.status));
    cfg = await res.json();
    salt = unb64(cfg.salt);
    if (cfg.v !== 1 || typeof cfg.app !== "string" || !(cfg.iterations >= 1) || salt.length < 16) throw new Error("lock.json");
  } catch {
    return fatal(T.config);
  }
  const id = cfg.salt;
  dropOtherStates(id);

  // the encrypted app (and stylesheet) download while the password is typed
  const get = (p) =>
    fetch(p).then(async (r) => {
      if (!r.ok) throw new Error(String(r.status));
      return new Uint8Array(await r.arrayBuffer());
    });
  let app;
  let css;
  const download = () => {
    app = get(cfg.app);
    css = typeof cfg.css === "string" ? get(cfg.css) : Promise.resolve(null);
    app.catch(() => undefined);
    css.catch(() => undefined);
  };
  download();

  /** Raw key → { raw, key, plain } when it opens the app, null for a wrong key; throws LockError("net"). */
  const tryKey = async (raw) => {
    let bytes;
    try {
      bytes = await app;
    } catch {
      download(); // retry on the next attempt
      throw new LockError("net");
    }
    const key = await importAesKey(raw);
    try {
      return { raw, key, plain: await decryptFile(bytes, key, cfg.app) };
    } catch {
      return null;
    }
  };

  /** The app's saved settings: decrypted into memory, re-encrypted on every change. */
  const openAppStorage = async (raw) => {
    const ls = storage("localStorage");
    if (!ls) return;
    try {
      const sk = await stateKey(raw);
      let state = new Map();
      let sealed = null;
      try {
        sealed = ls.getItem(STATE_PREFIX + id);
      } catch {
        /* ignore */
      }
      if (sealed) {
        try {
          state = await openState(sealed, sk, id);
        } catch {
          remove(ls, STATE_PREFIX + id); // damaged: start fresh
        }
      }
      for (const [k, v] of leftovers) if (!state.has(k)) state.set(k, v);
      const persist = stateWriter(state, sk, id, (k, v) => ls.setItem(k, v));
      protectAppStorage(state, persist);
      if (leftovers.size) void persist();
    } catch {
      /* no HKDF / storage: the app runs with its defaults */
    }
  };

  const start = async ({ raw, key, plain }) => {
    let sheet = null;
    try {
      sheet = await css;
    } catch {
      download();
      throw new LockError("net");
    }
    const code = new TextDecoder().decode(await gunzip(plain));
    if (sheet) {
      const text = new TextDecoder().decode(await gunzip(await decryptFile(sheet, key, cfg.css)));
      G.__STL_LOCK_CSS__ = await applyCss(absolutizeCss(text, new URL(cfg.css, document.baseURI).href));
    }
    await openAppStorage(raw);
    Object.defineProperty(G, "__STL_LOCK__", { value: Object.freeze({ key, lock: lockNow }), enumerable: false, configurable: true, writable: false });
    addEventListener("hashchange", () => {
      if (LOCK_HASH.test(location.hash)) lockNow();
    });
    const r = await runBundle(code);
    if (r.booted) {
      card.remove();
      return;
    }
    fatal(r.method ? T.crash : T.csp, r.error);
  };

  const saved = lockedNow ? null : readSavedKey(id);
  if (saved) {
    setBusy(true, T.saved);
    try {
      const ok = await tryKey(saved);
      if (ok) return await start(ok);
      forgetKey(id); // an old or damaged key
      setBusy(false);
    } catch (e) {
      setBusy(false);
      showError(e instanceof LockError ? e.message : T.crash);
    }
  }
  if (lockedNow) say(status, T.locked);

  btn.disabled = false;
  try {
    pw.focus({ preventScroll: true });
  } catch {
    /* ignore */
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (busy) return;
    const value = pw.value;
    if (!value) {
      showError(T.empty, true);
      pw.focus();
      return;
    }
    showError("");
    setBusy(true);
    try {
      let raw = await deriveKeyBytes(value, salt, cfg.iterations);
      let ok = await tryKey(raw);
      const trimmed = value.trim();
      if (!ok && trimmed && trimmed !== value) {
        raw = await deriveKeyBytes(trimmed, salt, cfg.iterations); // a pasted trailing space / newline
        ok = await tryKey(raw);
      }
      if (!ok) {
        setBusy(false);
        showError(T.wrong, true);
        pw.focus();
        pw.select();
        return;
      }
      saveKey(id, raw, remember.checked);
      pw.value = "";
      await start(ok);
    } catch (e) {
      if (card.dataset.state === "error") return;
      setBusy(false);
      showError(e instanceof LockError ? e.message : T.crash);
    }
  });
}

if (typeof document !== "undefined" && document.getElementById("stl-lock")) void gate();
