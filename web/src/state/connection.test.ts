import { afterEach, describe, expect, it, vi } from "vitest";
import { useConnection } from "../api/connection";
import { connectToServer, initBackend, runKey } from "./runner";
import { useStore } from "./store";
import { BUILTIN_META } from "./presets";

afterEach(() => {
  vi.unstubAllGlobals();
  useConnection.setState({ endpoint: "", status: "offline" });
  useStore.setState({ autoRun: false, results: {}, backend: "offline" });
});

describe("server selection and result provenance", () => {
  it("leaves an unreachable solver offline and refuses computation without producing curves", async () => {
    const urls: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => { urls.push(url); return new Response("not found", { status: 404 }); });
    useStore.setState({ autoRun: false });
    await initBackend();
    expect(useStore.getState().backend).toBe("offline");
    const run = await runKey("branches", "branches", {});
    expect(run.ok).toBe(false);
    expect(useStore.getState().results.branches.data).toBeUndefined();
    expect(useStore.getState().results.branches.error).toMatch(/연결|connect/i);
    expect(urls.some((url) => url.includes("/compute/"))).toBe(false);
  });

  it("clears all old results and ignores a delayed completion from the former server", async () => {
    let release!: (response: Response) => void;
    let submitted!: () => void;
    const didSubmit = new Promise<void>((resolve) => { submitted = resolve; });
    vi.stubGlobal("fetch", async (url: string) => {
      if (url.endsWith("/health")) return new Response('{"ok":true,"workers":1}');
      if (url.endsWith("/meta")) return new Response(JSON.stringify(BUILTIN_META));
      if (url.includes("/compute/")) { submitted(); return new Promise<Response>((resolve) => { release = resolve; }); }
      return new Response("{}", { status: 404 });
    });
    useStore.setState({ autoRun: false });
    await initBackend();
    const pending = runKey("branches", "branches", {});
    await didSubmit;
    await connectToServer("https://lab.example.edu/studio");
    expect(useStore.getState().results).toEqual({});
    release(new Response(JSON.stringify({ job_id: "former-server", status: "done", result: { stale: true } })));
    expect((await pending).ok).toBe(false);
    expect(useStore.getState().results).toEqual({});
    expect(useConnection.getState().endpoint).toBe("https://lab.example.edu/studio");
  });

  it("marks an expired server session as requiring a password, without a fallback result", async () => {
    vi.stubGlobal("fetch", async (url: string) => {
      if (url.endsWith("/health")) return new Response('{"ok":true}');
      if (url.endsWith("/meta")) return new Response(JSON.stringify(BUILTIN_META));
      return new Response('{"detail":"login required — reload the page to sign in"}', { status: 401 });
    });
    useStore.setState({ autoRun: false });
    await initBackend();
    expect((await runKey("branches", "branches", {})).ok).toBe(false);
    expect(useConnection.getState().status).toBe("auth-required");
    expect(useStore.getState().backend).toBe("offline");
    expect(useStore.getState().results.branches.data).toBeUndefined();
  });
});
