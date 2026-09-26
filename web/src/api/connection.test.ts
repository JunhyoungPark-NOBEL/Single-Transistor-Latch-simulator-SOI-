import { afterEach, describe, expect, it, vi } from "vitest";
import { CONNECTION_STORAGE_KEY, getSessionToken, normalizeEndpoint, readEndpoint, selectEndpoint, setSessionToken, useConnection } from "./connection";
import { authenticateServer, configureHttpBackend, createHttpBackend, JobAborted, retireHttpBackend, runJob } from "./client";

afterEach(() => {
  vi.unstubAllGlobals();
  setSessionToken(null);
  useConnection.setState({ endpoint: "", status: "offline" });
});

describe("connection configuration", () => {
  it("accepts HTTP(S) roots with subpaths and rejects credentials / ambiguous URL forms", () => {
    expect(normalizeEndpoint("  https://lab.example.edu/biristor/  ")).toBe("https://lab.example.edu/biristor");
    expect(normalizeEndpoint("http://127.0.0.1:8000")).toBe("http://127.0.0.1:8000");
    expect(normalizeEndpoint("")).toBe("");
    for (const invalid of ["lab.example.edu", "//lab.example.edu", "file:///tmp", "javascript:alert(1)", "https://user:secret@lab.edu", "https://lab.edu?q=secret", "https://lab.edu/#secret", "https://lab.edu/with space", "https://lab.edu\\bad"]) {
      expect(() => normalizeEndpoint(invalid)).toThrow("invalid-url");
    }
  });
  it("persists the normalized endpoint only; switching origins forgets the session token", () => {
    const saved = new Map<string, string>();
    vi.stubGlobal("localStorage", { getItem: (key: string) => saved.get(key), setItem: (key: string, value: string) => saved.set(key, value) });
    selectEndpoint("https://lab.edu/studio/");
    setSessionToken("ephemeral-test-token");
    expect(readEndpoint()).toBe("https://lab.edu/studio");
    expect(saved.get(CONNECTION_STORAGE_KEY)).toBe('{"version":1,"endpoint":"https://lab.edu/studio"}');
    expect([...saved.values()].join()).not.toContain("token");
    selectEndpoint("http://127.0.0.1:8000");
    expect(getSessionToken()).toBeNull();
  });
  it("invalid stored configuration returns to same-origin", () => {
    vi.stubGlobal("localStorage", { getItem: () => '{"version":1,"endpoint":"https://secret@lab.edu"}' });
    expect(readEndpoint()).toBe("");
  });
});

describe("HTTP transport isolation", () => {
  it("captures remote URL and token, omits cookies, and preserves nested server roots", async () => {
    const calls: [string, RequestInit][] = [];
    vi.stubGlobal("location", { origin: "http://localhost:5173" });
    vi.stubGlobal("fetch", async (url: string, options: RequestInit) => { calls.push([url, options]); return new Response('{"ok":true}'); });
    const backend = createHttpBackend({ endpoint: "https://lab.edu/biristor", token: "test-token" });
    selectEndpoint("https://different.edu");
    await backend.health();
    expect(calls[0][0]).toBe("https://lab.edu/biristor/api/health");
    expect(calls[0][1].credentials).toBe("omit");
    expect(calls[0][1].headers).toEqual({ Authorization: "Bearer test-token" });
  });
  it("uses same-origin cookies locally and posts passwords only in the session body", async () => {
    const calls: [string, RequestInit][] = [];
    vi.stubGlobal("fetch", async (url: string, options: RequestInit) => { calls.push([url, options]); return new Response('{"authenticated":true,"token":"test-token"}'); });
    await authenticateServer({ endpoint: "" }, "test-password");
    expect(calls[0][0]).toBe("/api/session");
    expect(calls[0][1].credentials).toBe("same-origin");
    expect(calls[0][1].body).toBe('{"password":"test-password"}');
    expect(calls[0][1].headers).toEqual({ "Content-Type": "application/json" });
  });
  it("rejects even an immediate done response when its server was retired during submission", async () => {
    let release!: (value: Response) => void;
    vi.stubGlobal("fetch", () => new Promise<Response>((resolve) => { release = resolve; }));
    const old = configureHttpBackend({ endpoint: "https://old.edu" });
    const job = runJob(old, "branches", {});
    retireHttpBackend();
    release(new Response(JSON.stringify({ job_id: "old-job", status: "done", result: { old: true } })));
    await expect(job).rejects.toBeInstanceOf(JobAborted);
  });
  it("cancels a late queued job on its original server after a switch", async () => {
    let release!: (value: Response) => void;
    const calls: [string, RequestInit][] = [];
    vi.stubGlobal("fetch", (url: string, options: RequestInit) => {
      calls.push([url, options]);
      if (options.method === "DELETE") return Promise.resolve(new Response("{}"));
      return new Promise<Response>((resolve) => { release = resolve; });
    });
    const old = configureHttpBackend({ endpoint: "https://old.edu", token: "old-session" });
    const job = runJob(old, "branches", {});
    configureHttpBackend({ endpoint: "https://new.edu", token: "new-session" });
    release(new Response(JSON.stringify({ job_id: "old-job", status: "queued" })));
    await expect(job).rejects.toBeInstanceOf(JobAborted);
    expect(calls.at(-1)?.[0]).toBe("https://old.edu/api/jobs/old-job");
    expect(calls.at(-1)?.[1].method).toBe("DELETE");
    expect(calls.at(-1)?.[1].headers).toEqual({ Authorization: "Bearer old-session" });
  });
});
