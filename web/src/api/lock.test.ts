// Locked artifact build: the Node encryption of the build (scripts/lock-crypto.mjs), the password gate's helpers
// (scripts/lock/lock.js) and the app's snapshot reader (readSnapshotBytes / loadSnapshot) must agree on one format.
// Dummy password only; a low iteration count keeps the tests fast (one test checks the real 600 000).
import { afterEach, describe, expect, it } from "vitest";
import { decryptFile as nodeDecrypt, deriveKey, encryptFile, MAGIC, MIN_ITERATIONS, newSalt } from "../../scripts/lock-crypto.mjs";
import * as gate from "../../scripts/lock/lock.js";
import {
  decodeBody, decryptLocked, isEncrypted, LOCK_MAGIC, LOCKED_INDEX, loadSnapshot, lockAad, readSnapshotBytes, snapshotKey, type SnapshotIndex,
} from "./snapshot";

const PASSWORD = "test-pass-123";
const ITER = 1000;
const enc = new TextEncoder();
const gz = async (v: unknown) =>
  new Uint8Array(await new Response(new Blob([JSON.stringify(v)]).stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer());
const aesKey = (raw: Uint8Array<ArrayBuffer>) => crypto.subtle.importKey("raw", raw, "AES-GCM", false, ["decrypt"]);

type G = { __STL_LOCK__?: { key: CryptoKey }; localStorage?: unknown; sessionStorage?: unknown; Storage?: unknown };
const g = globalThis as G;
afterEach(() => {
  delete g.__STL_LOCK__;
  delete g.localStorage;
  delete g.sessionStorage;
  delete g.Storage;
});

describe("locked format: build (Node) ↔ gate (Web Crypto) ↔ app reader", () => {
  it("magic, AAD normalisation and key derivation agree on both sides", async () => {
    expect(Array.from(MAGIC)).toEqual(LOCK_MAGIC);
    expect(Array.from(gate.MAGIC)).toEqual(LOCK_MAGIC);
    expect(new TextDecoder().decode(MAGIC)).toBe("STLENC1\0");
    expect(lockAad("./snapshot/index.bin")).toBe("snapshot/index.bin");
    expect(gate.lockAad("./assets/app-x.bin")).toBe("assets/app-x.bin");
    const salt = newSalt();
    expect(salt.length).toBe(16);
    const node = deriveKey(PASSWORD, salt, ITER);
    expect(node.length).toBe(32);
    expect(Array.from(await gate.deriveKeyBytes(PASSWORD, salt, ITER))).toEqual(Array.from(node));
    // NFC: a decomposed Korean password derives the same key as the composed one
    expect(Array.from(await gate.deriveKeyBytes("비밀".normalize("NFD"), salt, ITER))).toEqual(Array.from(deriveKey("비밀", salt, ITER)));
    expect(Array.from(deriveKey("other-pass", salt, ITER))).not.toEqual(Array.from(node));
  });

  it("the real iteration count (600 000) derives the same key in Node and Web Crypto", async () => {
    expect(MIN_ITERATIONS).toBeGreaterThanOrEqual(600_000);
    const salt = newSalt();
    expect(Array.from(await gate.deriveKeyBytes(PASSWORD, salt, MIN_ITERATIONS))).toEqual(Array.from(deriveKey(PASSWORD, salt, MIN_ITERATIONS)));
  }, 20_000);

  it("roundtrip with a dummy password: gzip JSON and plain JSON snapshot files, app bundle text", async () => {
    const salt = newSalt();
    const raw = deriveKey(PASSWORD, salt, ITER);
    const key = await aesKey(await gate.deriveKeyBytes(PASSWORD, salt, ITER));
    const f1 = encryptFile(await gz({ folds: { a: 3.7 } }), raw, "snapshot/abc.bin");
    expect(isEncrypted(f1)).toBe(true);
    expect(gate.isEncrypted(f1)).toBe(true);
    expect(f1.length).toBe(8 + 12 + (await gz({ folds: { a: 3.7 } })).length + 16);
    expect(await readSnapshotBytes(f1, "snapshot/abc.bin", key)).toEqual({ folds: { a: 3.7 } });
    expect(await readSnapshotBytes(f1, "./snapshot/abc.bin", key)).toEqual({ folds: { a: 3.7 } });
    const f2 = encryptFile(enc.encode('{"b":[1,2]}'), raw, "snapshot/p.bin");
    expect(await readSnapshotBytes(f2, "snapshot/p.bin", key)).toEqual({ b: [1, 2] });
    // two encryptions of the same bytes differ (random IV)
    expect(Array.from(encryptFile(enc.encode("x"), raw, "a")).join()).not.toBe(Array.from(encryptFile(enc.encode("x"), raw, "a")).join());
    // the gate: app bundle = gzip(JS) under its own path
    const js = "(function(){globalThis.__x=1})();";
    const app = encryptFile(new Uint8Array(await new Response(new Blob([js]).stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer()), raw, "assets/app-1.bin");
    const plain = await gate.decryptFile(app, key, "assets/app-1.bin");
    expect(gate.isGzip(plain)).toBe(true);
    expect(new TextDecoder().decode(await gate.gunzip(plain))).toBe(js);
    // and Node reads what it wrote
    expect(new TextDecoder().decode(nodeDecrypt(f2, raw, "snapshot/p.bin"))).toBe('{"b":[1,2]}');
  });

  it("a wrong key fails (no plaintext, no garbage)", async () => {
    const salt = newSalt();
    const file = encryptFile(await gz({ secret: 1 }), deriveKey(PASSWORD, salt, ITER), "snapshot/k.bin");
    const wrong = await aesKey(await gate.deriveKeyBytes("test-pass-124", salt, ITER));
    await expect(readSnapshotBytes(file, "snapshot/k.bin", wrong)).rejects.toThrow();
    await expect(gate.decryptFile(file, wrong, "snapshot/k.bin")).rejects.toThrow();
    const otherSalt = await aesKey(await gate.deriveKeyBytes(PASSWORD, newSalt(), ITER));
    await expect(decryptLocked(file, "snapshot/k.bin", otherSalt)).rejects.toThrow();
    expect(() => nodeDecrypt(file, deriveKey("test-pass-124", salt, ITER), "snapshot/k.bin")).toThrow();
  });

  it("the AAD binds a file to its published path; any modified byte fails", async () => {
    const salt = newSalt();
    const raw = deriveKey(PASSWORD, salt, ITER);
    const key = await aesKey(raw);
    const file = encryptFile(await gz({ v: 1 }), raw, "snapshot/a.bin");
    expect(await readSnapshotBytes(file, "snapshot/a.bin", key)).toEqual({ v: 1 });
    await expect(readSnapshotBytes(file, "snapshot/b.bin", key)).rejects.toThrow();
    await expect(readSnapshotBytes(file, "a.bin", key)).rejects.toThrow();
    await expect(gate.decryptFile(file, key, "assets/a.bin")).rejects.toThrow();
    for (const at of [8, 20, file.length - 1]) {
      const bad = file.slice();
      bad[at] ^= 1;
      await expect(readSnapshotBytes(bad, "snapshot/a.bin", key)).rejects.toThrow();
    }
  });

  it("plain passthrough: unencrypted JSON / gzip JSON are read as before, with or without a key", async () => {
    const key = await aesKey(deriveKey(PASSWORD, newSalt(), ITER));
    expect(await readSnapshotBytes(enc.encode('{"a":1}'), "snapshot/index.json", null)).toEqual({ a: 1 });
    expect(await readSnapshotBytes(await gz({ a: 2 }), "snapshot/x.json.gz", null)).toEqual({ a: 2 });
    expect(await readSnapshotBytes(enc.encode('{"a":3}'), "snapshot/index.json", key)).toEqual({ a: 3 });
    expect(JSON.parse(await decodeBody(enc.encode("[1]")))).toEqual([1]);
    expect(isEncrypted(enc.encode("STLENC1"))).toBe(false); // too short to be a file
    // encrypted bytes on a page that was never unlocked
    const file = encryptFile(enc.encode("{}"), deriveKey(PASSWORD, newSalt(), ITER), "s/x.bin");
    await expect(readSnapshotBytes(file, "s/x.bin", null)).rejects.toThrow(/not unlocked/);
  });
});

describe("locked snapshot layout (loadSnapshot with the gate's key)", () => {
  async function lockedFiles(password: string) {
    const salt = newSalt();
    const raw = deriveKey(password, salt, ITER);
    const index: SnapshotIndex = {
      format: 1,
      created: "2026-09-25T00:00:00Z",
      data: { measured: { file: "m1.bin", bytes: 1 } },
      compute: [{ kind: "branches", key: snapshotKey("branches", { x: 1 }), label: "t", file: "k1.bin", bytes: 1 }],
    };
    const files: Record<string, Uint8Array> = {
      [`s/${LOCKED_INDEX}`]: encryptFile(await gz(index), raw, `s/${LOCKED_INDEX}`),
      "s/k1.bin": encryptFile(await gz({ folds: { V: 3.7 } }), raw, "s/k1.bin"),
      "s/m1.bin": encryptFile(await gz({ photo: 1 }), raw, "s/m1.bin"),
    };
    return { salt, files };
  }
  const fakeFetch = (files: Record<string, Uint8Array>, log: string[]) => async (url: string) => {
    log.push(url);
    const b = files[url];
    return b ? new Response(new Blob([b as BlobPart])) : new Response("not found", { status: 404 });
  };

  it("with the key from the gate: index.bin and the result files are decrypted", async () => {
    const { salt, files } = await lockedFiles(PASSWORD);
    g.__STL_LOCK__ = { key: await gate.importAesKey(await gate.deriveKeyBytes(PASSWORD, salt, ITER)) };
    const log: string[] = [];
    const snap = (await loadSnapshot("s/", fakeFetch(files, log)))!;
    expect(snap).not.toBeNull();
    expect(snap.size).toBe(1);
    expect(await snap.lookup("branches", { x: 1 })).toEqual({ folds: { V: 3.7 } });
    expect(await snap.data("measured")).toEqual({ photo: 1 });
    expect(log).toEqual(["s/index.bin", "s/k1.bin", "s/m1.bin"]);
  });

  it("without a key (plain page) it asks for index.json; with a wrong key it finds no snapshot", async () => {
    const { salt, files } = await lockedFiles(PASSWORD);
    const log: string[] = [];
    expect(await loadSnapshot("s/", fakeFetch(files, log))).toBeNull();
    expect(log).toEqual(["s/index.json"]);
    g.__STL_LOCK__ = { key: await gate.importAesKey(await gate.deriveKeyBytes("wrong-pass-000", salt, ITER)) };
    expect(await loadSnapshot("s/", fakeFetch(files, []))).toBeNull();
  });
});

describe("password gate: saved key", () => {
  function fakeStorage(fail = false) {
    const m = new Map<string, string>();
    return {
      m,
      get length() {
        return m.size;
      },
      key: (i: number) => [...m.keys()][i] ?? null,
      getItem: (k: string) => {
        if (fail) throw new Error("blocked");
        return m.get(k) ?? null;
      },
      setItem: (k: string, v: string) => {
        if (fail) throw new Error("blocked");
        m.set(k, v);
      },
      removeItem: (k: string) => void m.delete(k),
    };
  }

  it("remember → localStorage, otherwise sessionStorage; the raw key is stored, never the password", async () => {
    const local = fakeStorage();
    const session = fakeStorage();
    g.localStorage = local;
    g.sessionStorage = session;
    local.m.set(`${gate.STORE_PREFIX}old-salt`, "AAAA"); // a key of an earlier build
    const raw = await gate.deriveKeyBytes(PASSWORD, newSalt(), ITER);
    const now = Date.now();
    gate.saveKey("salt1", raw, true, now);
    expect([...local.m.keys()]).toEqual([`${gate.STORE_PREFIX}salt1`]);
    expect(session.m.size).toBe(0);
    expect(JSON.parse(local.m.get(`${gate.STORE_PREFIX}salt1`)!)).toEqual({ k: gate.b64(raw), exp: now + gate.REMEMBER_MS });
    expect(gate.REMEMBER_MS).toBe(30 * 86_400_000);
    expect(local.m.get(`${gate.STORE_PREFIX}salt1`)).not.toContain(PASSWORD);
    expect(Array.from(gate.readSavedKey("salt1")!)).toEqual(Array.from(raw));
    gate.saveKey("salt1", raw, false);
    expect(local.m.size).toBe(0);
    expect(session.m.has(`${gate.STORE_PREFIX}salt1`)).toBe(true);
    expect(gate.readSavedKey("salt2")).toBeNull();
    gate.forgetKey("salt1");
    expect(gate.readSavedKey("salt1")).toBeNull();
    session.m.set(`${gate.STORE_PREFIX}salt1`, JSON.stringify({ k: "bm90IGEga2V5", exp: Date.now() + 1e6 })); // wrong length
    expect(gate.readSavedKey("salt1")).toBeNull();
    expect(session.m.size).toBe(0); // … and removed
    session.m.set(`${gate.STORE_PREFIX}salt1`, gate.b64(raw)); // old format (no expiry) → not accepted
    expect(gate.readSavedKey("salt1")).toBeNull();
    expect(session.m.size).toBe(0);
  });

  it("a remembered key expires after 30 days (session keys after a day) and is then removed", async () => {
    const local = fakeStorage();
    const session = fakeStorage();
    g.localStorage = local;
    g.sessionStorage = session;
    const raw = await gate.deriveKeyBytes(PASSWORD, newSalt(), ITER);
    const t0 = 1_700_000_000_000;
    gate.saveKey("s", raw, true, t0);
    expect(gate.readSavedKey("s", t0 + gate.REMEMBER_MS - 1)).not.toBeNull();
    expect(gate.readSavedKey("s", t0 + gate.REMEMBER_MS)).toBeNull();
    expect(local.m.size).toBe(0);
    gate.saveKey("s", raw, false, t0);
    expect(gate.readSavedKey("s", t0 + gate.SESSION_MS - 1)).not.toBeNull();
    expect(gate.readSavedKey("s", t0 + gate.SESSION_MS + 1)).toBeNull();
    expect(session.m.size).toBe(0);
  });

  it("forgetAll (#lock) removes every saved key of every build and nothing else", () => {
    const local = fakeStorage();
    const session = fakeStorage();
    g.localStorage = local;
    g.sessionStorage = session;
    local.m.set(`${gate.STORE_PREFIX}a`, "1");
    local.m.set(`${gate.STORE_PREFIX}b`, "2");
    local.m.set(`${gate.STATE_PREFIX}a`, "sealed");
    session.m.set(`${gate.STORE_PREFIX}a`, "3");
    gate.forgetAll();
    expect([...local.m.keys()]).toEqual([`${gate.STATE_PREFIX}a`]); // the encrypted settings stay (unreadable without the key)
    expect(session.m.size).toBe(0);
  });

  it("blocked or missing storage never throws", () => {
    g.localStorage = fakeStorage(true);
    g.sessionStorage = fakeStorage(true);
    expect(() => gate.saveKey("s", new Uint8Array(32), true)).not.toThrow();
    expect(gate.readSavedKey("s")).toBeNull();
    delete g.localStorage;
    delete g.sessionStorage;
    expect(() => gate.saveKey("s", new Uint8Array(32), false)).not.toThrow();
    expect(gate.readSavedKey("s")).toBeNull();
    expect(Array.from(gate.unb64(gate.b64(new Uint8Array([0, 255, 7]))))).toEqual([0, 255, 7]);
  });
});

describe("password gate: the app's saved settings are encrypted at rest", () => {
  class FakeStorage {
    m = new Map<string, string>();
    get length() {
      return this.m.size;
    }
    key(i: number) {
      return [...this.m.keys()][i] ?? null;
    }
    getItem(k: string) {
      return this.m.get(k) ?? null;
    }
    setItem(k: string, v: string) {
      this.m.set(k, String(v));
    }
    removeItem(k: string) {
      this.m.delete(k);
    }
  }
  const install = () => {
    g.Storage = FakeStorage;
    const local = new FakeStorage();
    const session = new FakeStorage();
    g.localStorage = local;
    g.sessionStorage = session;
    return { local, session, ls: local as unknown as Storage, ss: session as unknown as Storage };
  };

  it("sealState/openState: roundtrip; another build (id), another key or a modified byte fails", async () => {
    const raw = await gate.deriveKeyBytes(PASSWORD, newSalt(), ITER);
    const key = await gate.stateKey(raw);
    const calib = JSON.stringify({ params: { device: { calib: { beta: 7.166501201841884 } } } });
    const sealed = await gate.sealState({ "stl-websim:v1": calib, "other:x": "dropped" }, key, "salt1");
    expect(sealed).not.toContain("7.1665");
    expect(atob(sealed)).not.toContain("calib");
    expect(Object.fromEntries(await gate.openState(sealed, key, "salt1"))).toEqual({ "stl-websim:v1": calib });
    await expect(gate.openState(sealed, key, "salt2")).rejects.toThrow();
    await expect(gate.openState(sealed, await gate.stateKey(await gate.deriveKeyBytes("test-pass-124", newSalt(), ITER)), "salt1")).rejects.toThrow();
    const bad = gate.unb64(sealed);
    bad[20] ^= 1;
    await expect(gate.openState(gate.b64(bad), key, "salt1")).rejects.toThrow();
    // the state key is not the file key: a file key cannot open the settings
    const fileKey = await crypto.subtle.importKey("raw", raw, "AES-GCM", false, ["decrypt"]);
    await expect(gate.openState(sealed, fileKey, "salt1")).rejects.toThrow();
  });

  it("protectAppStorage: the app's localStorage keys live in memory, everything else passes through", () => {
    const { local, session, ls, ss } = install();
    const state = new Map<string, string>([["stl-websim:v1", "restored"]]);
    let writes = 0;
    expect(gate.protectAppStorage(state, () => void writes++)).toBe(true);
    expect(ls.getItem("stl-websim:v1")).toBe("restored");
    ls.setItem("stl-websim:schematic", "{calib}");
    expect(local.m.has("stl-websim:schematic")).toBe(false); // never on disk in plaintext
    expect(state.get("stl-websim:schematic")).toBe("{calib}");
    expect(ls.getItem("stl-websim:schematic")).toBe("{calib}");
    expect(writes).toBe(1);
    ls.removeItem("stl-websim:schematic");
    expect(ls.getItem("stl-websim:schematic")).toBeNull();
    expect(writes).toBe(2);
    ls.removeItem("stl-websim:missing");
    expect(writes).toBe(2);
    ls.setItem("stl-lock:x", "k"); // other keys and sessionStorage behave as before
    expect(local.m.get("stl-lock:x")).toBe("k");
    ss.setItem("stl-websim:t", "s");
    expect(session.m.get("stl-websim:t")).toBe("s");
    expect(ss.getItem("stl-websim:t")).toBe("s");
  });

  it("stateWriter: a burst of changes ends with the latest settings saved (encrypted)", async () => {
    const raw = await gate.deriveKeyBytes(PASSWORD, newSalt(), ITER);
    const key = await gate.stateKey(raw);
    const state = new Map<string, string>();
    const written: [string, string][] = [];
    const persist = gate.stateWriter(state, key, "id1", (k, v) => void written.push([k, v]));
    for (let i = 0; i < 5; i++) {
      state.set("stl-websim:v1", `v${i}`);
      void persist();
    }
    await persist();
    await new Promise((r) => setTimeout(r, 0));
    await persist();
    expect(written.length).toBeGreaterThanOrEqual(1);
    expect(written.length).toBeLessThanOrEqual(3);
    const [k, v] = written[written.length - 1];
    expect(k).toBe(`${gate.STATE_PREFIX}id1`);
    expect(Object.fromEntries(await gate.openState(v, key, "id1"))).toEqual({ "stl-websim:v1": "v4" });
  });

  it("plaintext app entries (earlier plain build) are taken out of both storages; other builds' settings dropped", () => {
    const { local, session } = install();
    local.m.set("stl-websim:v1", "plain-local");
    local.m.set("keep", "1");
    local.m.set(`${gate.STATE_PREFIX}old`, "x");
    local.m.set(`${gate.STATE_PREFIX}cur`, "y");
    session.m.set("stl-websim:v1", "plain-session");
    expect(Object.fromEntries(gate.takePlainAppEntries())).toEqual({ "stl-websim:v1": "plain-local" });
    gate.dropOtherStates("cur");
    expect([...local.m.keys()].sort()).toEqual(["keep", `${gate.STATE_PREFIX}cur`]);
    expect(session.m.size).toBe(0);
  });
});

describe("password gate: decrypted stylesheet", () => {
  it("absolutizeCss resolves relative url()s against the stylesheet's published path, leaves the rest", () => {
    const base = "https://h.test/p/assets/style-1.bin";
    const css = '@font-face{src:url(./KaTeX_A.woff2)format("woff2")}a{b:url("KaTeX_B.woff2")}c{d:url(data:font/woff2;base64,AAA=)}e{f:url(https://x.test/y.png)}g{h:url(#m)}';
    const out = gate.absolutizeCss(css, base);
    expect(out).toContain("url(https://h.test/p/assets/KaTeX_A.woff2)");
    expect(out).toContain('url("https://h.test/p/assets/KaTeX_B.woff2")');
    expect(out).toContain("url(data:font/woff2;base64,AAA=)");
    expect(out).toContain("url(https://x.test/y.png)");
    expect(out).toContain("url(#m)");
  });
});
