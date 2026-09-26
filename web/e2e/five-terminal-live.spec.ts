import { expect, test, type Page } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const live = process.env.STUDIO_LIVE === "1";
const review = "review/body-terminals";
const fixture = JSON.parse(readFileSync("e2e/fixtures/body-terminals/circuit.json", "utf8"));

async function worldClick(page: Page, x: number, y: number) {
  const pt = await page.getByTestId("sch-svg").evaluate((svg, point) => {
    const group = svg.querySelector(":scope > g") as SVGGraphicsElement;
    const screen = new DOMPoint(point.x, point.y).matrixTransform(group.getScreenCTM()!);
    return { x: screen.x, y: screen.y };
  }, { x, y });
  await page.mouse.click(pt.x, pt.y);
}

async function screenshot(page: Page, name: string) {
  await page.evaluate(() => { (document.activeElement as HTMLElement | null)?.blur(); window.scrollTo(0, 0); });
  await page.mouse.move(1650, 20);
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: `${review}/${name}.png`, fullPage: true });
}

test.describe("five terminal STL real circuit", () => {
  test.skip(!live, "STUDIO_LIVE=1 requires production frontend and real FastAPI server.");
  test.setTimeout(180_000);
  test.use({ viewport: { width: 1680, height: 1100 } });

  test("wire body C, edit R and run independent G/BG modulation with body probes", async ({ page, request }) => {
    mkdirSync(review, { recursive: true });
    expect((await request.get("/api/health")).ok()).toBe(true);
    const sent: any[] = [], finished: any[] = [];
    page.on("request", req => { if (req.method() === "POST" && /\/api\/compute\/circuit/.test(req.url())) sent.push(req.postDataJSON()); });
    page.on("response", async response => {
      if (/\/api\/(compute|jobs)\//.test(response.url()) && response.ok()) {
        try { const body = await response.json(); if (body.status === "done" && body.result?.runs) finished.push(body); } catch { /* navigation can abort requests */ }
      }
    });
    await page.addInitScript(() => {
      localStorage.clear();
      localStorage.setItem("stl-websim:v1", JSON.stringify({ v: 2, lang: "ko", theme: "light", mode: "deterministic", autoRun: false, autoRunChosen: true }));
    });
    await page.goto("/#tab=circuit&mode=deterministic");
    await expect(page.getByTestId("sch-svg")).toBeVisible();
    const imported = structuredClone(fixture);
    imported.doc.wires = imported.doc.wires.filter((wire: any) => wire.id !== "body-cap-wire");
    await page.getByTestId("file-import-input").setInputFiles({ name: "body-terminals.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(imported)) });
    await expect(page.locator('[data-el="X1"] .sch-pin-letter')).toHaveText(["D", "G", "S", "BG", "B"]);
    // Finish the external body-capacitor wire using the normal canvas interaction.
    await page.getByTestId("sch-canvas").focus();
    await page.keyboard.press("w");
    await worldClick(page, 700, 230);
    await worldClick(page, 700, 260);
    await page.getByTestId("tool-select").click();
    await expect(page.getByTestId("erc-pill")).toHaveClass(/ok/);
    await worldClick(page, 540, 300);
    await expect(page.getByTestId("insp-value")).toBeVisible();
    await page.getByTestId("insp-value").fill("20G");
    await page.getByTestId("insp-value").press("Enter");
    await worldClick(page, 360, 220);
    await expect(page.getByTestId("insp-terminal-defaults")).toContainText("BG 회로 연결");
    await expect(page.getByTestId("insp-terminal-defaults")).toContainText("B 회로 연결");
    await expect(page.getByTestId("insp-expand-terminals")).toHaveCount(0);
    await page.getByTestId("menu-edit").click();
    await page.getByTestId("tool-zoom-in").click();
    await page.getByTestId("menu-edit").click();
    await expect.poll(() => page.evaluate(() => {
      const saved = JSON.parse(localStorage.getItem("stl-websim:schematic") ?? "{}");
      return saved.doc?.wires.some((wire: any) => wire.x1 === 700 && wire.x2 === 700 && Math.min(wire.y1, wire.y2) === 230 && Math.max(wire.y1, wire.y2) === 260);
    })).toBe(true);
    await expect(page.getByTestId("sch-toast")).toHaveCount(0, { timeout: 5000 });
    await screenshot(page, "circuit-five-terminals");
    await page.getByTestId("run-button").click();
    await expect.poll(() => finished.length, { timeout: 150_000 }).toBeGreaterThan(0);
    const payload = sent.find(p => p.bench === "custom");
    expect(payload).toBeTruthy();
    expect(payload.netlist.elements.find((e: any) => e.type === "STL").nodes).toEqual({ d: "drain", g: "gate", s: "0", bg: "backgate", b: "body" });
    expect(payload.netlist.elements.find((e: any) => e.name === "VG").wave.kind).toBe("pulse");
    expect(payload.netlist.elements.find((e: any) => e.name === "VBG").wave.kind).toBe("sine");
    expect(payload.netlist.elements.find((e: any) => e.name === "Rb")).toMatchObject({ nodes: ["body", "0"], value: 2e10 });
    expect(payload.netlist.elements.find((e: any) => e.name === "Cb")).toMatchObject({ nodes: ["body", "0"], value: 1e-14 });
    const result = finished.at(-1).result;
    expect(result.elements.find((e: any) => e.type === "STL").device.geometry_model.validated).toBe(false);
    await expect(page.getByTestId("insp-stl-echo")).toHaveCount(0);
    const signals = new Map<string, number[]>(result.runs[0].signals.map((s: any) => [s.key, s.values]));
    const range = (key: string) => Math.max(...signals.get(key)!) - Math.min(...signals.get(key)!);
    for (const key of ["V(gate)", "V(backgate)", "V(body)", "X1.vb", "X1.vbody", "I(X1.b)"]) {
      expect(signals.get(key)!.length).toBeGreaterThan(50);
      expect(signals.get(key)!.every(Number.isFinite)).toBe(true);
    }
    expect(range("V(gate)")).toBeCloseTo(.05, 6);
    expect(range("V(backgate)")).toBeCloseTo(.2, 5);
    expect(range("X1.vb")).toBeGreaterThan(.001);
    expect(range("X1.vbody")).toBeGreaterThan(.0001);
    for (let i = 0; i < signals.get("X1.vbody")!.length; i++) expect(Math.abs(signals.get("X1.vbody")![i] - signals.get("V(body)")![i])).toBeLessThan(1e-7);
    const graph = page.getByTestId("panel-sch-waves").locator(".js-plotly-plot");
    await expect(graph).toBeVisible();
    await expect(page.getByTestId("trace-X1.vb")).toBeVisible();
    await page.getByTestId("trace-add").selectOption("X1.vbody");
    await expect(page.getByTestId("trace-X1.vbody")).toBeVisible();
    for (const key of ["V(drain)", "V(gate)", "V(backgate)", "V(body)", "I(X1.d)"]) {
      const chip = page.getByTestId(`trace-${key}`);
      if (await chip.count()) await chip.getByRole("button").click();
    }
    await expect.poll(() => graph.evaluate(g => (g as any).data.filter((d: any) => (d.y?.length ?? 0) > 50).length)).toBeGreaterThanOrEqual(2);
    await screenshot(page, "circuit-body-response");
    await graph.screenshot({ path: `${review}/circuit-body-waveforms.png` });
    writeFileSync(`${review}/live-request.json`, JSON.stringify(payload, null, 2));
    writeFileSync(`${review}/live-response.json`, JSON.stringify(finished.at(-1), null, 2));
    const finalDoc = await page.evaluate(() => JSON.parse(localStorage.getItem("stl-websim:schematic")!).doc);
    expect(finalDoc.elements.find((e: any) => e.name === "Rb").value).toBe(2e10);
    mkdirSync("../examples", { recursive: true });
    writeFileSync("../examples/body-rc.stl-circuit.json", JSON.stringify({ format: "stl-circuit", v: 2, doc: finalDoc }, null, 2));
  });
});
