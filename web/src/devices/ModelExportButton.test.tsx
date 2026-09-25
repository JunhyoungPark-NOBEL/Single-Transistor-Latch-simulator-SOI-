// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { runJob } from "../api/client";
import { useStore } from "../state/store";
import { downloadText } from "../utils/csv";
import { ModelExportButton } from "./ModelExportButton";
import type { ExportSelection } from "./modelExport";
import reference from "./fixtures/export-reference.json";

vi.mock("../api/client", () => ({ runJob: vi.fn(), JobAborted: class extends Error {} }));
vi.mock("../state/runner", () => ({ getBackend: () => ({ isMock: false }) }));
vi.mock("../utils/csv", () => ({ downloadText: vi.fn() }));
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;
const selection = reference.selection as ExportSelection;
const find = (id: string) => document.querySelector(`[data-testid="${id}"]`) as HTMLButtonElement;
const click = async (id: string) => { await act(async () => { find(id).click(); }); };

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(runJob).mockResolvedValue(reference.result);
  useStore.setState({ lang: "ko", backend: "online" });
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root.render(<ModelExportButton {...selection} />));
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

describe("commercial simulator export selection", () => {
  it("offers three formats and blocks a misleading executable Sentaurus download", async () => {
    expect(find("model-export-open").textContent).toBe("상용 시뮬레이터로 내보내기");
    await click("model-export-open");
    expect(document.querySelectorAll('[name="model-export-format"]')).toHaveLength(3);
    expect(find("model-export-download").disabled).toBe(false);
    await click("model-export-sentaurus");
    expect(find("model-export-sentaurus-info").textContent).toContain("격자·도핑·접촉");
    expect(find("model-export-download").disabled).toBe(true);
    await click("model-export-download");
    expect(runJob).not.toHaveBeenCalled();
    expect(downloadText).not.toHaveBeenCalled();
  });

  it("downloads the selected live model and clears previous-format results", async () => {
    await click("model-export-open");
    await click("model-export-verilog-a");
    await click("model-export-download");
    expect(runJob).toHaveBeenCalledWith(expect.anything(), "branches", { device: selection.device, sweep: selection.sweep }, expect.anything());
    expect(downloadText).toHaveBeenLastCalledWith("STL_Reference.va", expect.stringContaining("module STL_Reference(D, S);"), expect.any(String));
    expect(document.querySelector(".model-export-result")).not.toBeNull();
    await click("model-export-ltspice");
    expect(document.querySelector(".model-export-result")).toBeNull();
    await click("model-export-download");
    expect(downloadText).toHaveBeenLastCalledWith("STL_Reference.cir", expect.stringContaining(".subckt STL_Reference D S"), expect.any(String));
  });

  it("does not export demo data while disconnected", async () => {
    act(() => useStore.setState({ backend: "mock" }));
    await click("model-export-open");
    expect(find("model-export-download").disabled).toBe(true);
    await click("model-export-verilog-a");
    expect(find("model-export-download").disabled).toBe(true);
    expect(document.querySelector(".model-export-note")?.textContent).toContain("계산 서버");
    expect(runJob).not.toHaveBeenCalled();
  });

  it("discards a model that finishes after the export dialog is closed", async () => {
    let complete!: (value: unknown) => void;
    vi.mocked(runJob).mockImplementationOnce(() => new Promise(resolve => { complete = resolve; }));
    await click("model-export-open");
    await click("model-export-download");
    expect(find("model-export-download").disabled).toBe(true);
    await click("modal-close");
    await act(async () => { complete(reference.result); });
    expect(downloadText).not.toHaveBeenCalled();
    expect(find("model-export-dialog")).toBeNull();
  });
});
