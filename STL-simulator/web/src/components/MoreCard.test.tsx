// @vitest-environment jsdom
import { act, useContext, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useLayout } from "../state/layout";
import type { ResultEntry } from "../state/store";
import { entryStatus, FocusLayout, mergeStatus, MoreCard, MoreCardContext, selectMoreTab, type MoreTab } from "./MoreCard";
import { Panel, type PanelMenuItem } from "./Panel";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function Probe({ id }: { id: string }) {
  const { embedded } = useContext(MoreCardContext);
  return <section data-testid={`panel-${id}`} data-embedded={String(embedded)} />;
}
const tabs = (extra: Partial<Record<string, Partial<MoreTab>>> = {}): MoreTab[] =>
  ["a", "b", "c"].map((id) => ({ id, label: id.toUpperCase(), panel: <Probe id={id} />, ...extra[id] }));

let host: HTMLDivElement;
let root: Root;
const q = (sel: string) => host.querySelector(sel);
const render = (el: ReactNode) => act(() => root.render(el));

beforeEach(() => {
  localStorage.clear();
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe("entryStatus / mergeStatus", () => {
  const e = (p: Partial<ResultEntry>): ResultEntry => ({ status: "done", kind: "x", progress: 1, message: "", token: 0, ...p });
  it("mirrors Panel: running > error > stale", () => {
    expect(entryStatus(undefined, "k")).toBeNull();
    expect(entryStatus(e({ status: "queued" }), "k")).toBe("running");
    expect(entryStatus(e({ status: "running", data: 1, dataKey: "old" }), "k")).toBe("running");
    expect(entryStatus(e({ status: "error", data: 1, dataKey: "old" }), "k")).toBe("error");
    expect(entryStatus(e({ data: 1, dataKey: "old" }), "k")).toBe("stale");
    expect(entryStatus(e({ data: 1, dataKey: "k" }), "k")).toBeNull();
    expect(entryStatus(e({ dataKey: "old" }), "k")).toBeNull(); // no data yet → nothing to be stale
    expect(entryStatus(e({ data: 1, dataKey: "old" }))).toBeNull(); // no current key → never stale
  });
  it("mergeStatus picks the most urgent", () => {
    expect(mergeStatus(null, "stale", undefined, "error")).toBe("error");
    expect(mergeStatus("stale", "running")).toBe("running");
    expect(mergeStatus()).toBeNull();
  });
});

describe("MoreCard", () => {
  it("renders only the active tab, embedded, and persists the choice per scope", () => {
    render(<MoreCard scope="t1" tabs={tabs({ c: { status: "error" } })} defaultTab="b" />);
    expect(q('[data-testid="more-t1"]')).not.toBeNull();
    expect(q('[data-testid="more-tab-b"]')?.getAttribute("aria-selected")).toBe("true");
    expect(q('[data-testid="panel-b"]')?.getAttribute("data-embedded")).toBe("true");
    expect(q('[data-testid="panel-a"]')).toBeNull();
    expect(q('[data-testid="more-tab-c"] .more-dot.error')).not.toBeNull();

    const onResize = vi.fn();
    window.addEventListener("resize", onResize);
    act(() => (q('[data-testid="more-tab-a"]') as HTMLButtonElement).click());
    expect(q('[data-testid="panel-a"]')).not.toBeNull();
    expect(q('[data-testid="panel-b"]')).toBeNull();
    expect(localStorage.getItem("stl-websim:more:t1")).toBe("a");
    return new Promise<void>((resolve) =>
      setTimeout(() => {
        expect(onResize).toHaveBeenCalled();
        window.removeEventListener("resize", onResize);
        resolve();
      }, 50),
    );
  });

  it("restores the stored tab, and falls back to the default when it is hidden", () => {
    localStorage.setItem("stl-websim:more:t2", "c");
    render(<MoreCard scope="t2" tabs={tabs()} defaultTab="a" />);
    expect(q('[data-testid="more-tab-c"]')?.getAttribute("aria-selected")).toBe("true");
    render(<MoreCard scope="t2" tabs={tabs({ c: { hidden: true } })} defaultTab="a" />);
    expect(q('[data-testid="more-tab-c"]')).toBeNull();
    expect(q('[data-testid="panel-a"]')).not.toBeNull();
    expect(localStorage.getItem("stl-websim:more:t2")).toBe("c"); // the choice survives for when c returns
  });

  it("switches from outside with selectMoreTab, and with arrow keys", () => {
    render(<MoreCard scope="t3" tabs={tabs()} defaultTab="a" />);
    act(() => selectMoreTab("t3", "c"));
    expect(q('[data-testid="panel-c"]')).not.toBeNull();
    act(() => q('[role="tablist"]')!.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true })));
    expect(q('[data-testid="panel-a"]')).not.toBeNull(); // wraps around
    expect(document.activeElement).toBe(q('[data-testid="more-tab-a"]'));
  });
});

describe("FocusLayout", () => {
  it("simple: hero + card; all: today's grid with every visible panel in allOrder", () => {
    act(() => useLayout.getState().setLayout("simple"));
    const el = <FocusLayout testId="panels-x" scope="t4" hero={<Probe id="hero" />} tabs={tabs({ b: { hidden: true } })} defaultTab="c" side allOrder={["hero", "c"]} />;
    render(el);
    const grid = q('[data-testid="panels-x"]')!;
    expect(grid.className).toBe("focus-grid side");
    expect(q('[data-testid="panel-hero"]')?.getAttribute("data-embedded")).toBe("false");
    expect(q('[data-testid="panel-c"]')?.getAttribute("data-embedded")).toBe("true");
    expect(q('[data-testid="panel-a"]')).toBeNull();

    act(() => useLayout.getState().setLayout("all"));
    const all = q('[data-testid="panels-x"]')!;
    expect(all.className).toBe("grid");
    expect([...all.children].map((c) => c.getAttribute("data-testid"))).toEqual(["panel-hero", "panel-c", "panel-a"]);
    expect(q('[data-testid="more-t4"]')).toBeNull();
    expect(q('[data-testid="panel-a"]')?.getAttribute("data-embedded")).toBe("false");
    act(() => useLayout.getState().setLayout("simple"));
  });
});

describe("Panel Step-0 props", () => {
  it("renders the footnote line and no menu trigger without items", () => {
    render(<Panel id="x" title="X" hasData foot={<span>fold I_LU 5.31 pA</span>} />);
    expect(q(".panel-footnote")?.textContent).toBe("fold I_LU 5.31 pA");
    expect(q('[data-testid="panel-menu-x"]')).toBeNull();
    expect(q(".panel")?.className).toBe("panel");
  });

  it("⋯ menu: 보기 + 내보내기 sections, check toggles, action closes, Esc closes", () => {
    const onChange = vi.fn();
    const onCsv = vi.fn();
    const items: PanelMenuItem[] = [
      { kind: "check", id: "meas", label: "측정", checked: true, onChange, testId: "m-meas" },
      { kind: "radio", id: "log", group: "y", label: "로그", checked: true, onSelect: () => {} },
      { kind: "action", id: "csv", label: "CSV", onSelect: onCsv, testId: "csv-x" },
    ];
    render(<Panel id="x" title="X" hasData menu={items} />);
    const trig = q('[data-testid="panel-menu-x"]') as HTMLButtonElement;
    act(() => trig.click());
    expect(q('[role="menu"]')).not.toBeNull();
    expect([...host.querySelectorAll(".pmenu-sec")].map((s) => s.getAttribute("aria-label"))).toEqual(["보기", "내보내기"]);
    act(() => (q('[data-testid="m-meas"]') as HTMLButtonElement).click());
    expect(onChange).toHaveBeenCalledWith(false);
    expect(q('[role="menu"]')).not.toBeNull(); // check items keep the menu open
    act(() => (q('[data-testid="csv-x"]') as HTMLButtonElement).click());
    expect(onCsv).toHaveBeenCalled();
    expect(q('[role="menu"]')).toBeNull();
    act(() => trig.click());
    act(() => q('[role="menu"]')!.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })));
    expect(q('[role="menu"]')).toBeNull();
    expect(document.activeElement).toBe(trig);
  });

  it("embedded in a MoreCard: borderless class and a visually hidden h3", () => {
    render(
      <MoreCardContext.Provider value={{ embedded: true }}>
        <Panel id="x" title="X" hasData primary />
      </MoreCardContext.Provider>,
    );
    expect(q(".panel")?.className).toBe("panel primary embedded");
    expect(q("h3")?.className).toContain("sr-only");
    expect(q('[data-testid="panel-x"]')?.getAttribute("aria-labelledby")).toBe("panel-x-title");
  });
});
