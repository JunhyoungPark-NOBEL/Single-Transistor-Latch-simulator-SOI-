// User workflows for the compact voltage-sweep workspace. Advanced analyses remain available through
// the full layout; the default run must not launch computations for panels the user did not request.
import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

async function fresh(page: Page, mode = "deterministic", all = false, lang = "ko") {
  await page.addInitScript((language) => {
    if (!sessionStorage.getItem("e2e-init")) {
      localStorage.clear();
      localStorage.setItem("stl-websim:v1", JSON.stringify({ v: 2, lang: language }));
      sessionStorage.setItem("e2e-init", "1");
    }
  }, lang);
  await page.goto(`/?mock=1${all ? "&view=all" : ""}#tab=device&mode=${mode}`);
  await expect(page.getByTestId("tab-device")).toBeVisible();
}
const jobs = (page: Page) => page.evaluate(() => (window as unknown as { __stlJobs?: string[] }).__stlJobs ?? []);
async function ready(page: Page) { await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704", { timeout: 15_000 }); }
async function setVg(page: Page, value: string) {
  const input = page.getByTestId("field-vg").locator("input.input");
  await input.fill(value);
  await input.press("Enter");
}
async function plotted(page: Page, id: string) { await expect(page.getByTestId(`panel-${id}`).locator(".js-plotly-plot")).toBeVisible({ timeout: 20_000 }); }
async function exportMenu(page: Page, id: string) {
  await plotted(page, id);
  await page.getByTestId(`panel-menu-${id}`).click();
  await expect(page.getByTestId(`csv-${id}`)).toBeVisible();
  await expect(page.getByTestId(`png-${id}`)).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId(`csv-${id}`)).toHaveCount(0);
}

test.describe("Device workspace", () => {
  test("first load shows three results and one voltage-sweep chart without hidden analysis jobs", async ({ page }) => {
    await fresh(page);
    await ready(page);
    await expect(page.getByTestId("layout-simple")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("kpi-vld-value")).toContainText("2.598");
    await expect(page.getByTestId("kpi-window-value")).toContainText("1.106");
    await expect(page.getByTestId("kpis").locator("[data-testid$='-value']")).toHaveCount(3);
    await plotted(page, "iv");
    await expect(page.locator(".js-plotly-plot")).toHaveCount(1);
    await expect(page.getByTestId("getting-started")).toHaveCount(0);
    await expect(page.getByTestId("more-device-det")).toHaveCount(0);
    await expect(page.getByTestId("panel-components")).toHaveCount(0);
    await expect(page.getByTestId("stats-table")).toHaveCount(0);
    await expect(page.getByTestId("panel-iv").locator(".legend .traces")).toHaveCount(2);
    expect(await jobs(page)).toEqual(["branches"]);
    const bounds = await page.getByTestId("panel-iv").locator(".js-plotly-plot").boundingBox();
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(900);
  });

  test("V_G changes update deltas; previous sweeps are optional and excluded from CSV", async ({ page }) => {
    await fresh(page);
    await ready(page);
    await expect(page.getByTestId("kpi-vlu-delta")).toHaveCount(0);
    await setVg(page, "-1.9");
    await expect(page.getByTestId("kpi-vlu-delta")).toContainText(/▲ \+8\d\.\d mV/, { timeout: 15_000 });
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.78");
    await expect(page.getByTestId("kpis")).not.toHaveClass(/stale/);
    await expect(page.getByTestId("panel-iv").locator(".legend")).not.toContainText("이전");
    await page.getByTestId("panel-menu-iv").click();
    await page.getByTestId("iv-prev").click();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("panel-iv").locator(".legend")).toContainText("이전");
    expect((await jobs(page)).every(k => k === "branches")).toBe(true);
    expect((await jobs(page)).length).toBeGreaterThanOrEqual(2);
    await page.getByTestId("panel-menu-iv").click();
    const [dl] = await Promise.all([page.waitForEvent("download"), page.getByTestId("csv-iv").click()]);
    const csv = readFileSync((await dl.path())!, "utf8");
    expect(csv.split("\n")[0]).toContain("상향 스윕");
    expect(csv).not.toContain("이전");
    expect(csv).not.toMatch(/unstable|불안정/);
    await expect(page.getByRole("menu")).toHaveCount(0);
    await expect(page.getByTestId("panel-menu-iv")).toBeFocused();
  });

  test("manual edits mark retained results stale until the next run", async ({ page }) => {
    await fresh(page);
    await ready(page);
    await page.getByTestId("autorun").click();
    await expect(page.getByTestId("autorun")).toHaveAttribute("aria-checked", "false");
    await setVg(page, "-1.9");
    await expect(page.getByTestId("kpis")).toHaveClass(/stale/);
    await expect(page.getByTestId("panel-iv").locator(".badge.stale")).toBeVisible();
    expect(await jobs(page)).toEqual(["branches"]);
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.78", { timeout: 15_000 });
    await expect(page.getByTestId("kpis")).not.toHaveClass(/stale/);
  });

  test("full-view link keeps detailed panels, unchanged V_G analysis, and accessible range controls", async ({ page }) => {
    await fresh(page, "deterministic", true);
    await ready(page);
    for (const id of ["iv", "components", "charge-balance", "vg"]) await plotted(page, id);
    await expect(page.getByTestId("panels-deterministic")).toHaveClass(/\bgrid\b/);
    expect(await jobs(page)).toEqual(expect.arrayContaining(["branches", "charge_balance", "vg_curve"]));
    const before = (await jobs(page)).filter(k => k === "vg_curve").length;
    await setVg(page, "-1.9");
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.78", { timeout: 15_000 });
    expect((await jobs(page)).filter(k => k === "vg_curve").length).toBe(before);
    await expect(page.getByTestId("panel-components").locator(".legend .traces")).toHaveCount(8);
    await page.getByTestId("vg-range").click();
    await expect(page.getByTestId("vg-range-min")).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("vg-range-min")).toHaveCount(0);
    await expect(page.getByTestId("vg-range")).toBeFocused();
  });

  test("CSV and PNG exports stay reachable for advanced deterministic and stochastic analyses", async ({ page }) => {
    await fresh(page, "deterministic", true);
    await ready(page);
    for (const id of ["iv", "vg", "components", "charge-balance"]) await exportMenu(page, id);
    await page.getByTestId("mode-stochastic").click();
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 20_000 });
    for (const id of ["mc-iv", "dist", "hazard", "cycles", "design-map"]) await exportMenu(page, id);
    await page.getByTestId("vgs-compute").click();
    await exportMenu(page, "vg-sto");
  });

  test("stochastic default has one sweep; full view retains statistical comparisons and censoring", async ({ page }) => {
    await fresh(page, "stochastic");
    await expect(page.getByTestId("empty-run-mc-iv")).toBeVisible();
    await expect(page.getByTestId("panel-stats")).toHaveCount(0);
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 20_000 });
    await plotted(page, "mc-iv");
    await expect(page.locator(".js-plotly-plot")).toHaveCount(1);
    expect((await jobs(page)).every(k => ["sweep_mc", "branches"].includes(k))).toBe(true);
    await page.getByTestId("layout-all").click();
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("stats-table")).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId("stats-meta")).toContainText("시드");
    await expect(page.getByTestId("stats-meta")).toContainText("보정 조회표");
    await expect(page.getByTestId("stats-censored")).toContainText("중도절단");
    await expect(page.getByTestId("stats-table").locator('th[data-col="lag1"]')).toBeVisible();
    await expect(page.getByTestId("stats-table").locator('tr[data-row="hazard"]')).toBeVisible({ timeout: 20_000 });
    await page.getByTestId("stats-table-all-cols").click();
    await expect(page.getByTestId("stats-table").locator("thead th[data-col]")).toHaveCount(6);
    await expect(page.getByTestId("stats-table").locator('tr[data-row="hazard"]')).toHaveCount(0);
    await page.reload();
    await expect(page.getByTestId("layout-all")).toHaveAttribute("aria-checked", "true");
    await page.getByTestId("layout-simple").click();
    await expect(page.getByTestId("panel-stats")).toHaveCount(0);
    await expect(page.getByTestId("panel-dist")).toHaveCount(0);
    await expect(page.getByTestId("panel-mc-iv")).toBeVisible();
  });

  test("dark mode and English preserve a compact readable stochastic chart", async ({ page }) => {
    await fresh(page, "stochastic", false, "en");
    await page.getByTestId("theme-toggle").click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 20_000 });
    await expect(page.getByTestId("kpi-vlu")).toContainText("Latch-up");
    await plotted(page, "mc-iv");
    await expect(page.getByTestId("panel-mc-iv")).not.toContainText("calibrated_lookup");
    await expect(page.locator(".js-plotly-plot")).toHaveCount(1);
    await expect(page.getByTestId("panel-mc-iv").locator(".legend")).toContainText("Up sweep");
  });

  for (const viewport of [{ width: 1024, height: 768 }, { width: 390, height: 844 }]) {
    test(`responsive ${viewport.width}px: results and plot fit without horizontal page scrolling`, async ({ page }) => {
      await page.setViewportSize(viewport);
      await fresh(page);
      await ready(page);
      await plotted(page, "iv");
      await expect(page.locator(".js-plotly-plot")).toHaveCount(1);
      const tops = await Promise.all(["kpi-vlu", "kpi-vld", "kpi-window"].map(async id => (await page.getByTestId(id).boundingBox())!.y));
      expect(Math.max(...tops) - Math.min(...tops)).toBeLessThanOrEqual(1);
      expect((await page.getByTestId("panel-iv").locator(".js-plotly-plot").boundingBox())!.height).toBeGreaterThanOrEqual(280);
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
    });
  }

  test("phone full-view statistics retain a compact table without page overflow", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 860 });
    await fresh(page, "stochastic", true);
    await page.getByTestId("empty-run-mc-iv").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 20_000 });
    await page.getByTestId("stats-table-all-cols").click();
    await expect(page.getByTestId("stats-table").locator("thead th[data-col]")).toHaveCount(6);
    const m = await page.evaluate(() => {
      const sc = document.querySelector(".stats-scroll") as HTMLElement;
      return { inner: sc.scrollWidth, outer: sc.clientWidth, page: document.documentElement.scrollWidth };
    });
    expect(m.inner).toBeLessThanOrEqual(m.outer + 1);
    expect(m.page).toBeLessThanOrEqual(390);
    await expect(page.getByTestId("stats-table").locator('td[data-col="dmean"]').first()).toBeHidden();
  });
});
