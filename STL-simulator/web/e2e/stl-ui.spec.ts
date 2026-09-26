// UI refresh gates. Run against the production frontend served by the real Python API.
// These checks do not mock numerical results or submit a stochastic simulation.
import { expect, test, type Locator, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";

const review = "review/stl-ui";
const live = process.env.STUDIO_LIVE === "1";

async function fresh(page: Page) {
  await page.addInitScript(() => {
    if (!sessionStorage.getItem("stl-ui-qa")) {
      localStorage.clear();
      localStorage.setItem("stl-websim:v1", JSON.stringify({ v: 2, lang: "ko", theme: "light", mode: "deterministic", autoRun: false, autoRunChosen: true }));
      sessionStorage.setItem("stl-ui-qa", "1");
    }
  });
  await page.goto("/#tab=device&mode=deterministic");
  await expect(page.getByTestId("mode-deterministic")).toBeVisible();
}

async function noOverflow(page: Page) {
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
}

async function shot(page: Page, name: string) {
  mkdirSync(review, { recursive: true });
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => { (document.activeElement as HTMLElement | null)?.blur(); window.scrollTo(0, 0); });
  await expect.poll(() => page.evaluate(() => scrollY)).toBe(0);
  for (const close of await page.locator(".plotly-notifier .notifier-close").all()) await close.click();
  await noOverflow(page);
  await page.screenshot({ path: `${review}/${name}.png`, fullPage: true });
}

async function plotted(graph: Locator) {
  await expect(graph).toBeVisible({ timeout: 180_000 });
  await expect.poll(() => graph.evaluate(el => {
    const d = (el as any).data as { x?: number[]; y?: number[] }[] | undefined;
    return (d ?? []).some(tr => (tr.x?.length ?? 0) > 10 && tr.y?.some(Number.isFinite));
  }), { timeout: 180_000 }).toBe(true);
  await expect(graph.locator(".scatterlayer .trace path").first()).toHaveAttribute("d", /.+/);
}

async function mathStyle(label: Locator) {
  await expect(label.locator(".math-var").first()).toHaveCSS("font-style", "italic");
  for (const sub of await label.locator("sub").all()) await expect(sub).toHaveCSS("font-style", "normal");
}

function recordJobs(page: Page) {
  const jobs: unknown[] = [];
  page.on("response", async response => {
    if (/\/api\/(?:compute|jobs)\//.test(response.url()) && response.ok()) {
      try { const body = await response.json(); if (body.status === "done") jobs.push({ url: response.url(), body }); } catch { /* Navigation may abort a response read. */ }
    }
  });
  return jobs;
}

test.describe("STL simulator production UI refresh", () => {
  test.skip(!live, "Set STUDIO_LIVE=1 and serve the production frontend with the actual FastAPI backend.");
  test.setTimeout(240_000);
  test.beforeEach(async ({ request }) => {
    const health = await request.get("/api/health");
    expect(health.ok()).toBe(true);
    expect((await health.json()).ok).toBe(true);
  });

  test("FDSOI schematic selects matching geometry fields without changing values; modes and symbols are explicit", async ({ page }) => {
    await fresh(page);
    await expect(page).toHaveTitle(/STL simulator/);
    await expect(page.locator(".brand-title")).toHaveText("STL simulator");
    await expect(page.getByTestId("mode-deterministic")).toHaveText("결정론적");
    await expect(page.getByTestId("mode-stochastic")).toHaveText("확률적");
    await expect(page.getByTestId("mode-deterministic")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("technology-FDSOI")).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("technology-PDSOI")).toBeDisabled();
    await expect(page.getByTestId("technology-Bulk")).toBeDisabled();
    const keys = ["Lg_nm", "W_nm", "Tsi_nm", "EOT_nm", "Tbox_nm", "Nbody_cm3"];
    const initial = await Promise.all(keys.map(key => page.getByTestId(`geometry-${key}`).inputValue()));
    for (const key of keys) {
      const symbol = page.getByTestId(`schematic-${key}`);
      const input = page.getByTestId(`geometry-${key}`);
      await symbol.click();
      await expect(input).toBeFocused();
      await expect(symbol).toHaveAttribute("aria-pressed", "true");
      await input.blur();
      await symbol.focus();
      await page.keyboard.press("Enter");
      await expect(input).toBeFocused();
    }
    expect(await Promise.all(keys.map(key => page.getByTestId(`geometry-${key}`).inputValue()))).toEqual(initial);
    const geometry = page.getByTestId("geometry-controls");
    for (const label of await geometry.locator("label").all()) await mathStyle(label);
    expect(await page.getByTestId("geometry-Lg_nm").evaluate(el => parseFloat(getComputedStyle(el).fontSize))).toBeGreaterThanOrEqual(14);
    expect(await page.getByTestId("tab-device").evaluate(el => parseFloat(getComputedStyle(el).fontSize))).toBeGreaterThanOrEqual(14);
    expect(await page.getByTestId("mode-deterministic").evaluate(el => parseFloat(getComputedStyle(el).fontSize))).toBeGreaterThanOrEqual(13);
    await page.getByTestId("mode-stochastic").click();
    await expect(page.getByTestId("mode-stochastic")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("panels-stochastic")).toBeVisible();
    await page.getByTestId("mode-stochastic").press("ArrowLeft");
    await expect(page.getByTestId("mode-deterministic")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("mode-deterministic")).toBeFocused();
    await page.getByTestId("geometry-Lg_nm").fill("400");
    await page.getByTestId("geometry-Lg_nm").press("Enter");
    // The existing settings store debounces writes by 250 ms. Wait for its persisted state.
    await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem("stl-websim:v1")!).params.device.geometry.Lg_nm)).toBe(400);
    await page.reload();
    await expect(page.getByTestId("geometry-Lg_nm")).toHaveValue("400");
    await expect(page.getByTestId("technology-FDSOI")).toHaveAttribute("aria-pressed", "true");
    await noOverflow(page);
  });

  test("actual VSCM IDVD is legible in both themes and narrow layouts; geometry drawer remains usable", async ({ page }) => {
    const jobs = recordJobs(page);
    await fresh(page);
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704", { timeout: 180_000 });
    await expect(page.getByTestId("kpi-vld-value")).toContainText("2.598");
    const graph = page.getByTestId("panel-iv").locator(".js-plotly-plot");
    await plotted(graph);
    const titles = await graph.evaluate(el => ({ x: (el as any)._fullLayout.xaxis.title.text, y: (el as any)._fullLayout.yaxis.title.text }));
    expect(titles.x).toContain("<i>V</i><sub>D</sub>");
    expect(titles.y).toContain("<i>I</i><sub>D</sub>");
    expect(await graph.evaluate(el => (el as any)._fullLayout.xaxis.tickfont.size)).toBeGreaterThanOrEqual(12);
    expect(await graph.evaluate(el => (el as any)._fullLayout.yaxis.tickfont.size)).toBeGreaterThanOrEqual(12);
    const kpi = page.getByTestId("kpi-vlu").locator(".ans-sym");
    await expect(kpi).toHaveCSS("font-style", "italic");
    await expect(kpi.locator("sub")).toHaveCSS("font-style", "normal");
    await shot(page, "device-light");
    await page.getByTestId("geometry-controls").screenshot({ path: `${review}/geometry-detail.png` });
    const lightPlot = await graph.evaluate(el => (el as any)._fullLayout.font.color);
    await page.getByTestId("theme-toggle").click();
    await expect.poll(() => graph.evaluate(el => (el as any)._fullLayout.font.color)).not.toBe(lightPlot);
    await shot(page, "device-dark");
    await page.getByTestId("theme-toggle").click();
    await page.setViewportSize({ width: 390, height: 844 });
    const brand = (await page.locator(".brand").boundingBox())!;
    const menu = (await page.locator(".sidebar-btn").boundingBox())!;
    expect(brand.x).toBeGreaterThanOrEqual(menu.x + menu.width - 1);
    await expect(page.getByTestId("mode-deterministic")).toBeInViewport();
    await expect(page.getByTestId("mode-stochastic")).toBeInViewport();
    await shot(page, "device-mobile");
    await page.locator(".sidebar-btn").click();
    await expect(page.getByTestId("sidebar")).toHaveClass(/ open/);
    await expect(page.getByTestId("geometry-schematic")).toBeInViewport();
    expect(await page.getByTestId("geometry-Lg_nm").evaluate(el => parseFloat(getComputedStyle(el).fontSize))).toBeGreaterThanOrEqual(14);
    await page.getByTestId("schematic-Tsi_nm").click();
    await expect(page.getByTestId("geometry-Tsi_nm")).toBeFocused();
    await shot(page, "geometry-mobile");
    await page.getByTestId("sidebar").getByRole("button", { name: "닫기", exact: true }).click();
    await page.setViewportSize({ width: 320, height: 780 });
    await expect(page.getByTestId("mode-deterministic")).toBeInViewport();
    await expect(page.getByTestId("mode-stochastic")).toBeInViewport();
    await noOverflow(page);
    await expect.poll(() => jobs.length).toBeGreaterThan(0);
    writeFileSync(`${review}/live-vscm.json`, JSON.stringify(jobs, null, 2));
  });

  test("actual CSVM shows italic main variables and upright extrema subscripts with a computed waveform", async ({ page }) => {
    const jobs = recordJobs(page);
    await fresh(page);
    await page.getByTestId("device-forcing").getByRole("radio").nth(1).click();
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-frequency-value")).toContainText("860", { timeout: 180_000 });
    const graph = page.getByTestId("panel-csvm").locator(".js-plotly-plot");
    await plotted(graph);
    for (const key of ["vtop", "vbottom"]) {
      const label = page.getByTestId(`kpi-${key}`).locator(".ans-sym");
      await expect(label).toHaveCSS("font-style", "italic");
      await expect(label.locator("sub")).toHaveCSS("font-style", "normal");
    }
    await expect.poll(() => graph.evaluate(el => (el as any)._fullLayout.yaxis.title.text)).toContain("<i>V</i><sub>D</sub>");
    await shot(page, "csvm-light");
    await expect.poll(() => jobs.length).toBeGreaterThan(0);
    writeFileSync(`${review}/live-csvm.json`, JSON.stringify(jobs, null, 2));
    // Layout review only: the existing optional example is shown without a new circuit solve.
    await page.getByTestId("tab-circuit").click();
    await page.getByTestId("menu-examples").click();
    await page.getByTestId("tpl-load_line").click();
    await expect(page.getByTestId("sch-svg")).toBeVisible();
    await expect(page.getByTestId("sch-toast")).toHaveCount(0, { timeout: 5000 });
    await shot(page, "circuit-layout");
  });
});
