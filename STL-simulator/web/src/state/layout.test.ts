// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { initialLayout, LAYOUT_STORAGE_KEY } from "./layout";

/** In-memory storage; `broken` makes every call throw (blocked site data). */
function fakeStorage(init: Record<string, string> = {}, broken = false) {
  const m = new Map(Object.entries(init));
  return {
    m,
    getItem: (k: string) => {
      if (broken) throw new Error("blocked");
      return m.get(k) ?? null;
    },
    setItem: (k: string, v: string) => {
      if (broken) throw new Error("blocked");
      m.set(k, v);
    },
  };
}

describe("initialLayout (query > localStorage > simple)", () => {
  it("defaults to simple", () => {
    expect(initialLayout("", fakeStorage())).toBe("simple");
    expect(initialLayout("?mock=1", null)).toBe("simple");
  });
  it("uses the stored layout when there is no query", () => {
    expect(initialLayout("?mock=1", fakeStorage({ [LAYOUT_STORAGE_KEY]: "all" }))).toBe("all");
    expect(initialLayout("", fakeStorage({ [LAYOUT_STORAGE_KEY]: "junk" }))).toBe("simple");
  });
  it("lets ?view= win and saves it", () => {
    const st = fakeStorage({ [LAYOUT_STORAGE_KEY]: "all" });
    expect(initialLayout("?mock=1&view=simple", st)).toBe("simple");
    expect(st.m.get(LAYOUT_STORAGE_KEY)).toBe("simple");
    const st2 = fakeStorage();
    expect(initialLayout("?view=all", st2)).toBe("all");
    expect(st2.m.get(LAYOUT_STORAGE_KEY)).toBe("all");
  });
  it("ignores an invalid ?view= value", () => {
    expect(initialLayout("?view=everything", fakeStorage({ [LAYOUT_STORAGE_KEY]: "all" }))).toBe("all");
  });
  it("never throws when storage is blocked", () => {
    expect(initialLayout("", fakeStorage({}, true))).toBe("simple");
    expect(initialLayout("?view=all", fakeStorage({}, true))).toBe("all");
  });
});

describe("useLayout store (jsdom)", () => {
  beforeEach(() => {
    vi.resetModules();
    localStorage.clear();
  });
  afterEach(() => window.history.replaceState(null, "", "/"));

  it("starts from the query, persists it, and keeps the hash when toggling", async () => {
    window.history.replaceState(null, "", "/?mock=1&view=all#tab=device&mode=stochastic");
    const { useLayout } = await import("./layout");
    expect(useLayout.getState().layout).toBe("all");
    expect(localStorage.getItem(LAYOUT_STORAGE_KEY)).toBe("all");
    expect(document.documentElement.dataset.layout).toBe("all");
    useLayout.getState().setLayout("simple");
    expect(useLayout.getState().layout).toBe("simple");
    expect(localStorage.getItem(LAYOUT_STORAGE_KEY)).toBe("simple");
    expect(window.location.search).toBe("?mock=1&view=simple");
    expect(window.location.hash).toBe("#tab=device&mode=stochastic");
    expect(document.documentElement.dataset.layout).toBe("simple");
  });

  it("falls back to localStorage and leaves a query-less URL alone", async () => {
    window.history.replaceState(null, "", "/?mock=1#tab=circuit");
    localStorage.setItem(LAYOUT_STORAGE_KEY, "all");
    const { useLayout } = await import("./layout");
    expect(useLayout.getState().layout).toBe("all");
    useLayout.getState().setLayout("simple");
    expect(window.location.search).toBe("?mock=1");
    expect(window.location.hash).toBe("#tab=circuit");
    expect(localStorage.getItem(LAYOUT_STORAGE_KEY)).toBe("simple");
  });
});
