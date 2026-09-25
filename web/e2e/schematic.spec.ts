// Schematic editor + device library (mock mode, no backend needed): draw a circuit (V + R + ground +
// wire), run it, probe a node, time-cursor annotations, the load-line, oscillator and p-bit examples, keyboard shortcuts, ERC,
// stochastic statistics and the device library (save → reload → load, place in a schematic).
// Runs in the 모두 보기 layout (?view=all: every panel, the full toolbar and the open netlist, as before);
// the 간단히 default is covered by e2e/ux-shell.spec.ts.
import { expect, test, type Page } from "@playwright/test";

const SHOTS = "e2e/screenshots/all";

async function fresh(page: Page, hash = "#tab=circuit&mode=deterministic") {
  await page.addInitScript(() => {
    try {
      if (!sessionStorage.getItem("e2e-init")) {
        localStorage.clear();
        sessionStorage.setItem("e2e-init", "1");
      }
    } catch {
      /* ignore */
    }
  });
  await page.goto(`/?mock=1&view=all${hash}`);
  await expect(page.getByTestId("mode-toggle")).toBeVisible();
}

/** Screen position of a world point (the view is centred on (0,0) at zoom 1 after "New circuit"). */
async function world(page: Page) {
  // Playwright's scroll-into-view for toolbar/menu clicks may leave the page scrolled: measure from the top
  await page.evaluate(() => window.scrollTo(0, 0));
  const box = (await page.getByTestId("sch-svg").boundingBox())!;
  const k = Number(await page.getByTestId("sch-svg").getAttribute("data-k"));
  return (x: number, y: number) => ({ x: box.x + box.width / 2 + x * k, y: box.y + box.height / 2 + y * k });
}

async function newCircuit(page: Page) {
  await page.getByTestId("menu-file").click();
  await page.getByTestId("file-new").click();
  await expect(page.locator("[data-testid=sch-svg] [data-el]")).toHaveCount(0);
}

async function place(page: Page, tool: string, x: number, y: number) {
  await page.getByTestId(`tool-${tool}`).click();
  const at = await world(page);
  const p = at(x, y);
  await page.mouse.click(p.x, p.y);
}

test.describe("schematic editor (mock mode)", () => {
  test("draw V + R + ground, wire, run, probe a node, time cursor annotations", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("schematic-view")).toBeVisible();
    await newCircuit(page);
    await place(page, "V", 0, 0); // pins (0,-40) (0,40)
    await place(page, "GND", 0, 40);
    await place(page, "R", 120, 0); // pins (120,-40) (120,40)
    await place(page, "GND", 120, 40);
    await page.getByTestId("tool-wire").click();
    let at = await world(page);
    await page.mouse.click(at(0, -40).x, at(0, -40).y);
    await page.mouse.click(at(120, -40).x, at(120, -40).y); // ends on the resistor pin
    await expect(page.locator("[data-testid=sch-svg] [data-wire]")).toHaveCount(1);
    await expect(page.getByTestId("erc-pill")).toHaveClass(/ok/);
    // edit the source: select it and type a SPICE value
    await page.getByTestId("tool-select").click();
    at = await world(page);
    await page.mouse.click(at(0, 0).x, at(0, 0).y);
    await expect(page.getByTestId("inspector")).toContainText("V1");
    const val = page.getByTestId("insp-wave-value");
    await val.fill("2");
    await val.press("Enter");
    await expect(page.getByTestId("netlist-text")).toContainText("V1 N001 0 DC 2");
    await expect(page.getByTestId("netlist-text")).toContainText("R1 N001 0 1k");
    await page.screenshot({ path: `${SHOTS}/schematic-drawn.png` });

    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("panel-sch-waves").locator(".js-plotly-plot")).toBeVisible({ timeout: 20_000 });
    at = await world(page);
    // probe tool is active after a run: click the wire → V(N001)
    await expect(page.getByTestId("tool-probe")).toHaveAttribute("aria-pressed", "true");
    await page.mouse.click(at(60, -40).x, at(60, -40).y);
    await expect(page.getByTestId("trace-V(N001)")).toBeVisible();
    // click the resistor body → I(R1)
    await page.mouse.click(at(120, 0).x, at(120, 0).y);
    await expect(page.getByTestId("trace-I(R1)")).toBeVisible();
    await expect(page.getByTestId("readout")).toContainText("V(N001)");
    // the time cursor annotates the schematic: node voltage and element currents with direction arrows
    await expect(page.locator("[data-ann-node=N001]")).toContainText("2.000 V");
    await expect(page.locator("[data-ann-el=R1]")).toContainText("2 mA");
    await expect(page.locator("[data-ann-el=R1] path")).toHaveCount(1);
    await expect(page.locator("[data-ann-el=V1]")).toContainText("2 mA");
    const before = await page.getByTestId("cursor-time").textContent();
    await page.getByTestId("cursor-slider").focus();
    await page.keyboard.press("End");
    await expect(page.getByTestId("cursor-time")).not.toHaveText(before ?? "");
    // remove a trace with its chip
    await page.getByTestId("trace-I(R1)").getByRole("button").click();
    await expect(page.getByTestId("trace-I(R1)")).toHaveCount(0);
    await page.screenshot({ path: `${SHOTS}/schematic-run.png`, fullPage: true });
  });

  test("load-line example: netlist, run, summary and events", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("menu-examples").click();
    await page.getByTestId("tpl-load_line").click();
    await expect(page.locator("[data-testid=sch-svg] [data-kind=STL]")).toHaveCount(1);
    await expect(page.locator("[data-testid=sch-svg] [data-kind=LABEL]")).toHaveCount(3);
    await expect(page.getByTestId("erc-pill")).toHaveClass(/ok/);
    const nl = page.getByTestId("netlist-text");
    await expect(nl).toContainText("Vsrc src 0 PWL(0 0 10 4 20 0)");
    await expect(nl).toContainText("Rs src d 1k");
    await expect(nl).toContainText("Cd d 0 2f");
    await expect(nl).toContainText("VG1 g 0 DC -2");
    await expect(nl).toContainText(/X1 d g 0 STL/);
    await expect(nl).toContainText(".tran 0 20 0 10m");
    await page.getByTestId("netlist-view").getByRole("radio", { name: /JSON/ }).click();
    await expect(nl).toContainText('"bench": "custom"');
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("sch-summary")).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId("kpi-sch-X1.n_latch_up-value")).toHaveText("1");
    await expect(page.getByTestId("sch-events-table")).toContainText("X1");
    await expect(page.locator("[data-ann-el=X1]")).toBeVisible();
    await page.screenshot({ path: `${SHOTS}/schematic-loadline.png`, fullPage: true });
  });

  test("current-driven oscillator example: I_in into out, repeated latch-ups", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("menu-examples").click();
    await page.getByTestId("tpl-oscillator").click();
    await expect(page.locator("[data-testid=sch-svg] [data-kind=STL]")).toHaveCount(1);
    await expect(page.locator("[data-testid=sch-svg] [data-kind=I]")).toHaveCount(1);
    await expect(page.locator("[data-testid=sch-svg] [data-kind=LABEL]")).toHaveCount(1);
    await expect(page.getByTestId("erc-pill")).toHaveClass(/ok/);
    const nl = page.getByTestId("netlist-text");
    await expect(nl).toContainText("Iin 0 out DC 1n");
    await expect(nl).toContainText("Cpar out 0 1p");
    await expect(nl).toContainText(/X1 out \S+ 0 STL/);
    await expect(nl).toContainText(".tran 0 15m 0 5u");
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("sch-summary")).toBeVisible({ timeout: 20_000 });
    const n = Number(await page.getByTestId("kpi-sch-X1.n_latch_up-value").textContent());
    expect(n).toBeGreaterThan(1);
    await expect(page.getByTestId("sch-events-table")).toContainText("X1");
    await page.screenshot({ path: `${SHOTS}/schematic-oscillator.png`, fullPage: true });
  });

  test("p-bit example (stochastic): comparator on the source resistor, firing raster and P(fire)", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=stochastic");
    await page.getByTestId("menu-examples").click();
    await page.getByTestId("tpl-pbit").click();
    await expect(page.locator("[data-testid=sch-svg] [data-kind=CMP]")).toHaveCount(1);
    await expect(page.getByTestId("erc-pill")).toHaveClass(/ok/);
    const nl = page.getByTestId("netlist-text");
    await expect(nl).toContainText("RS s 0 100k");
    await expect(nl).toContainText("CMP1 s 0 q CMP vref=100m");
    await page.getByTestId("sto-runs").fill("4");
    await page.getByTestId("sto-runs").press("Enter");
    await page.getByTestId("run-button").click();
    const confirm = page.getByTestId("sch-confirm-run");
    if (await confirm.isVisible({ timeout: 1500 }).catch(() => false)) await confirm.click();
    await expect(page.getByTestId("sch-cmp-stats-CMP1")).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId("sch-cmp-stats-CMP1")).toContainText("P(");
    await expect(page.getByTestId("panel-sch-cmp-CMP1").locator(".js-plotly-plot")).toBeVisible();
    await page.screenshot({ path: `${SHOTS}/schematic-pbit.png`, fullPage: true });
  });

  test("keyboard shortcuts: place, rotate, undo/redo, delete, Esc", async ({ page }) => {
    await fresh(page);
    await newCircuit(page);
    const at = await world(page);
    await page.getByTestId("sch-canvas").focus();
    await page.keyboard.press("r");
    await expect(page.getByTestId("tool-R")).toHaveAttribute("aria-pressed", "true");
    await page.keyboard.press("Control+r"); // rotate the part being placed
    await page.mouse.click(at(0, 0).x, at(0, 0).y);
    await expect(page.locator("[data-testid=sch-svg] [data-kind=R]")).toHaveCount(1);
    await expect(page.getByTestId("netlist-text")).toContainText("R1");
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("tool-select")).toHaveAttribute("aria-pressed", "true");
    await page.keyboard.press("Control+z");
    await expect(page.locator("[data-testid=sch-svg] [data-kind=R]")).toHaveCount(0);
    await page.keyboard.press("Control+y");
    await expect(page.locator("[data-testid=sch-svg] [data-kind=R]")).toHaveCount(1);
    await page.mouse.click(at(0, 0).x, at(0, 0).y);
    await page.keyboard.press("Control+d");
    await expect(page.locator("[data-testid=sch-svg] [data-kind=R]")).toHaveCount(2);
    await page.keyboard.press("Delete");
    await expect(page.locator("[data-testid=sch-svg] [data-kind=R]")).toHaveCount(1);
    // K places a comparator
    await page.keyboard.press("Escape");
    await page.keyboard.press("k");
    await expect(page.getByTestId("tool-CMP")).toHaveAttribute("aria-pressed", "true");
    await page.mouse.click(at(200, 0).x, at(200, 0).y);
    await expect(page.locator("[data-testid=sch-svg] [data-kind=CMP]")).toHaveCount(1);
    await expect(page.getByTestId("netlist-text")).toContainText("CMP1");
  });

  test("ERC blocks the run and highlights the problem", async ({ page }) => {
    await fresh(page);
    await newCircuit(page);
    await place(page, "R", 0, 0);
    await expect(page.getByTestId("erc-pill")).toHaveClass(/err/);
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("sch-toast")).toBeVisible();
    await expect(page.getByTestId("erc-list")).toBeVisible();
    await expect(page.getByTestId("erc-item-noGround")).toBeVisible();
    await expect(page.getByTestId("erc-item-unconnected").first()).toBeVisible();
    await page.getByTestId("erc-item-unconnected").first().click();
    await expect(page.locator("[data-testid=sch-svg] .sch-el.flag")).toHaveCount(1);
  });

  test("stochastic run: mean ± SD band, event statistics and distributions", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=stochastic");
    await page.getByTestId("menu-examples").click();
    await page.getByTestId("tpl-pulse").click();
    await expect(page.getByTestId("sch-sto")).toBeVisible();
    await page.getByTestId("sto-runs").fill("8");
    await page.getByTestId("sto-runs").press("Enter");
    await expect(page.getByTestId("feasibility")).toBeVisible();
    await page.getByTestId("run-button").click();
    const confirm = page.getByTestId("sch-confirm-run");
    if (await confirm.isVisible({ timeout: 1500 }).catch(() => false)) await confirm.click();
    await expect(page.getByTestId("sch-stats-table")).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId("sch-events-table")).toContainText("%");
    await expect(page.getByTestId("panel-sch-waves").locator(".js-plotly-plot")).toBeVisible();
    await page.screenshot({ path: `${SHOTS}/schematic-stochastic.png`, fullPage: true });
  });

  test("device library: save as device, persists, loads, places in a schematic", async ({ page }) => {
    await fresh(page, "#tab=device&mode=deterministic");
    await expect(page.getByTestId("geometry-Lg_nm")).toHaveValue("500");
    await expect(page.getByTestId("preset-card")).toContainText("Device 1");
    const vg = page.getByTestId("field-vg").locator("input.input");
    await vg.fill("-1.9");
    await vg.press("Enter");
    await page.getByTestId("dev-save-open").click();
    await page.getByTestId("dev-save-name").fill("Test device A");
    await page.getByTestId("dev-save-submit").click();
    await expect(page.getByTestId("dev-toast")).toContainText("Test device A");
    // back to the reference preset, reload: the saved device is still there and loads V_G = −1.9 V
    await page.getByTestId("preset-paper").click();
    await expect(vg).toHaveValue("-2");
    await page.reload();
    await expect(page.getByTestId("device-slot-1")).toContainText("Test device A");
    await page.getByTestId("device-slot-1").getByRole("button", { name: "불러오기", exact: true }).click();
    await expect(page.getByTestId("field-vg").locator("input.input")).toHaveValue("-1.9");
    // manager: duplicate, then place the device in the schematic
    await page.getByTestId("dev-manage").click();
    await expect(page.getByTestId("device-manager")).toBeVisible();
    await page.getByTestId("dm-row-Test device A").getByTestId("dm-duplicate").click();
    await expect(page.getByTestId("dm-user-list").locator("li")).toHaveCount(2);
    await page.screenshot({ path: "review/device-library.png" });
    await page.getByTestId("dm-row-Test device A").getByTestId("dm-place").click();
    await expect(page.getByTestId("schematic-view")).toBeVisible();
    await expect(page.getByTestId("tool-STL")).toHaveAttribute("aria-pressed", "true");
    await newCircuit(page);
    await page.getByTestId("tool-STL").click();
    const at = await world(page);
    await page.mouse.click(at(0, 0).x, at(0, 0).y);
    await expect(page.locator("[data-testid=sch-svg] [data-kind=STL]")).toHaveCount(1);
    await page.keyboard.press("Escape");
    await page.getByTestId("tool-select").click();
    const at2 = await world(page);
    await page.mouse.click(at2(0, 0).x, at2(0, 0).y);
    await expect(page.getByTestId("insp-stl-device").locator("option:checked")).toHaveText("Test device A");
    await expect(page.getByTestId("inspector")).toContainText("-1.90");
    // the sidebar library lists it too
    await expect(page.getByTestId("lib-Test device A")).toBeVisible();
  });

  test("free-form editor is the circuit view; examples are a secondary menu", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("circuit-view-benches")).toHaveCount(0);
    await expect(page.getByTestId("bench-picker")).toHaveCount(0);
    await expect(page.getByTestId("sch-canvas")).toBeVisible();
    await expect(page.getByTestId("menu-examples")).toBeVisible();
    await expect(page.getByTestId("sch-sim")).toBeVisible();
  });
});

test.describe("schematic editor (live backend on :8000)", () => {
  test.beforeEach(async ({ request }) => {
    let ok = false;
    try {
      const r = await request.get(`${process.env.STL_API ?? "http://127.0.0.1:8000"}/api/health`, { timeout: 3000 });
      ok = r.ok() && (await r.json()).ok === true;
    } catch {
      ok = false;
    }
    test.skip(!ok, "backend not running on :8000");
  });

  test("live: load-line example reproduces the latch-up fold; stochastic pulse train gives statistics", async ({ page }) => {
    test.setTimeout(240_000);
    await page.addInitScript(() => {
      try {
        if (!sessionStorage.getItem("e2e-live")) {
          localStorage.clear();
          sessionStorage.setItem("e2e-live", "1");
        }
      } catch {
        /* ignore */
      }
    });
    await page.goto("/?view=all#tab=circuit&mode=deterministic");
    await expect(page.getByTestId("backend-status")).toContainText("API", { timeout: 15_000 });
    await page.getByTestId("menu-examples").click();
    await page.getByTestId("tpl-load_line").click();
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("sch-summary")).toBeVisible({ timeout: 120_000 });
    await expect(page.getByTestId("sch-demo-fallback")).toHaveCount(0);
    await expect(page.getByTestId("panel-sch-waves").locator(".badge.demo")).toHaveCount(0);
    await expect(page.getByTestId("kpi-sch-X1.n_latch_up-value")).toHaveText("1");
    await expect(page.getByTestId("kpi-sch-X1.vd_first_lu-value")).toContainText("3.70");
    await expect(page.locator("[data-ann-el=X1]")).toBeVisible();
    await page.screenshot({ path: `${SHOTS}/schematic-live-loadline.png`, fullPage: true });

    await page.getByTestId("mode-stochastic").click();
    await page.getByTestId("menu-examples").click();
    await page.getByTestId("tpl-pulse").click();
    await page.getByTestId("sto-runs").fill("4");
    await page.getByTestId("sto-runs").press("Enter");
    await page.getByTestId("run-button").click();
    const confirm = page.getByTestId("sch-confirm-run");
    if (await confirm.isVisible({ timeout: 1500 }).catch(() => false)) await confirm.click();
    await expect(page.getByTestId("sch-stats-table")).toBeVisible({ timeout: 180_000 });
    await expect(page.getByTestId("kpi-sch-X1.p_any_lu-value")).toBeVisible();
    await expect(page.getByTestId("sch-events-table")).toContainText("%");
    await page.screenshot({ path: `${SHOTS}/schematic-live-stochastic.png`, fullPage: true });
  });
});
