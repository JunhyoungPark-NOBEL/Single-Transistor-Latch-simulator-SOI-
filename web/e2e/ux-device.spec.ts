// Device tab in the default 간단히 layout (mock mode): three answer numbers, one hero plot, one tabbed analysis
// card; the first load runs by itself; a V_G change shows the delta and the "이전" ghost without re-running
// the V_G curve; stochastic statistics as one line with the table on demand. Writes the canonical
// screenshots (e2e/screenshots/*.png); the legacy specs write theirs to screenshots/all/ with ?view=all.
import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

const SHOTS = "e2e/screenshots";

async function fresh(page: Page, hash = "#tab=device&mode=deterministic", init: Record<string, string> = {}) {
  await page.addInitScript((extra) => {
    try {
      if (!sessionStorage.getItem("e2e-init")) {
        localStorage.clear();
        for (const [k, v] of Object.entries(extra)) localStorage.setItem(k, v);
        sessionStorage.setItem("e2e-init", "1");
      }
    } catch {
      /* ignore */
    }
  }, init);
  await page.goto(`/?mock=1${hash}`);
  await expect(page.getByTestId("tab-device")).toBeVisible();
}

/** Plotly plots that intersect the viewport, and how many of them lie fully inside it. */
async function plotsInView(page: Page) {
  return page.evaluate(() => {
    const vh = window.innerHeight;
    const vw = window.innerWidth;
    const rs = [...document.querySelectorAll(".js-plotly-plot")].map((e) => e.getBoundingClientRect()).filter((r) => r.width > 0 && r.height > 0);
    const inView = rs.filter((r) => r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw);
    return { inView: inView.length, fully: inView.filter((r) => r.top >= 0 && r.bottom <= vh + 1).length };
  });
}

/** Top of the hero plot, measured from the top of the main column (84 px of chrome above it at 1440 × 900). */
async function heroOffset(page: Page) {
  return page.evaluate(() => {
    const hero = document.querySelector(".panel.primary .js-plotly-plot")!.getBoundingClientRect();
    const main = document.querySelector("main")!.getBoundingClientRect();
    return { hero: Math.round(hero.top), main: Math.round(main.top) };
  });
}

const jobs = (page: Page) => page.evaluate(() => (window as unknown as { __stlJobs?: string[] }).__stlJobs ?? []);

async function setVg(page: Page, v: string) {
  const input = page.getByTestId("field-vg").locator("input.input");
  await input.fill(v);
  await input.press("Enter");
}

test.describe("Device · 간단히 layout (mock mode)", () => {
  test("first load runs by itself: 3 answers in V, one hero and one analysis card above the fold, no table", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("layout-simple")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("getting-started")).toBeVisible();
    // no click: auto-run is on by default and the first load computes Device · deterministic
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704", { timeout: 15_000 });
    await expect(page.getByTestId("kpi-vld-value")).toContainText("2.598");
    await expect(page.getByTestId("kpi-window-value")).toContainText("1.106");
    await expect(page.getByTestId("kpis").locator("[data-testid$='-value']")).toHaveCount(3);
    await expect(page.getByTestId("panel-iv").locator(".js-plotly-plot")).toBeVisible();
    await expect(page.getByTestId("panel-vg").locator(".js-plotly-plot")).toBeVisible({ timeout: 15_000 });
    await page.waitForTimeout(500);
    // exactly two plots, both fully above the fold; the hero plot starts ≤ 330 px (84 px chrome + 246 px)
    expect(await plotsInView(page)).toEqual({ inView: 2, fully: 2 });
    const off = await heroOffset(page);
    expect(off.hero - off.main).toBeLessThanOrEqual(246);
    // the other views are one click away, as tabs of the analysis card
    for (const id of ["vg", "components", "charge-balance"]) await expect(page.getByTestId(`more-tab-${id}`)).toBeVisible();
    await expect(page.getByTestId("more-tab-vg")).toHaveAttribute("aria-selected", "true");
    await expect(page.getByTestId("panel-components")).toHaveCount(0);
    await expect(page.getByTestId("stats-table")).toHaveCount(0);
    // I–V legend: at most 4 entries (HRS, LRS, 측정, 이전); no orange sweep overlay by default
    const legend = await page.getByTestId("panel-iv").locator(".legend .traces").count();
    expect(legend).toBeGreaterThan(0);
    expect(legend).toBeLessThanOrEqual(4);
    // each visible panel shows at most one control besides 📖 and ⋯
    for (const bar of await page.locator(".panel-tools, .panel.embedded > .panel-toolbar").all()) {
      if (await bar.isVisible()) expect(await bar.locator(":scope > *").count()).toBeLessThanOrEqual(1);
    }
    // the hint goes away for good
    await page.getByTestId("getting-started-ok").click();
    await expect(page.getByTestId("getting-started")).toHaveCount(0);
    await page.reload();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704", { timeout: 15_000 });
    await expect(page.getByTestId("getting-started")).toHaveCount(0);
  });

  test("V_G −2 → −1.9 with auto-run: ▲ +80 mV delta and the 이전 ghost; the V_G curve is neither stale nor re-run", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704", { timeout: 15_000 });
    await expect(page.getByTestId("panel-vg").locator(".js-plotly-plot")).toBeVisible({ timeout: 15_000 });
    // before a second run the sub-line explains the number
    await expect(page.getByTestId("kpi-vlu")).toContainText("HRS→LRS");
    await expect(page.getByTestId("kpi-vlu-delta")).toHaveCount(0);
    const vgJobs = (await jobs(page)).filter((k) => k === "vg_curve").length;
    expect(vgJobs).toBe(1);
    await setVg(page, "-1.9");
    await expect(page.getByTestId("kpi-vlu-delta")).toContainText("▲", { timeout: 15_000 });
    await expect(page.getByTestId("kpi-vlu-delta")).toContainText(/\+8\d\.\d mV/);
    await expect(page.getByTestId("kpi-vlu-delta")).toContainText("이전 대비");
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.78");
    await expect(page.getByTestId("kpis")).not.toHaveClass(/stale/);
    // the previous run as a grey ghost with its own legend entry
    await expect(page.getByTestId("panel-iv").locator(".legend")).toContainText("이전");
    // the V_G curve does not depend on the sidebar V_G: no badge, no dot, no new job — only the marker moved
    await expect(page.getByTestId("panel-vg").locator(".badge")).toHaveCount(0);
    await expect(page.getByTestId("more-tab-vg")).not.toHaveAttribute("data-status");
    const after = await jobs(page);
    expect(after.filter((k) => k === "vg_curve").length).toBe(vgJobs);
    expect(after.filter((k) => k === "branches").length).toBeGreaterThanOrEqual(2);
    await page.waitForTimeout(700);
    await page.screenshot({ path: `${SHOTS}/desktop-deterministic.png` });
    // CSV and PNG live in the ⋯ menu; the ghost is never exported
    await page.getByTestId("panel-menu-iv").click();
    await expect(page.getByTestId("png-iv")).toBeVisible();
    const [dl] = await Promise.all([page.waitForEvent("download"), page.getByTestId("csv-iv").click()]);
    const csv = readFileSync((await dl.path())!, "utf8");
    expect(csv.split("\n")[0]).toContain("HRS");
    expect(csv).not.toContain("이전");
    await expect(page.getByRole("menu")).toHaveCount(0); // an action closes the menu
    await expect(page.getByTestId("panel-menu-iv")).toBeFocused();
  });

  test("analysis tabs render the original panels; a stale result in a hidden tab shows a dot", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704", { timeout: 15_000 });
    await expect(page.getByTestId("panel-vg").locator(".js-plotly-plot")).toBeVisible({ timeout: 15_000 });
    await page.getByTestId("autorun").click();
    await expect(page.getByTestId("autorun")).toHaveAttribute("aria-checked", "false");
    await setVg(page, "-1.9");
    await expect(page.getByTestId("kpis")).toHaveClass(/stale/);
    await expect(page.getByTestId("more-tab-components")).toHaveAttribute("data-status", "stale");
    await expect(page.getByTestId("more-tab-charge-balance")).toHaveAttribute("data-status", "stale");
    await expect(page.getByTestId("more-tab-vg")).not.toHaveAttribute("data-status");
    await page.getByTestId("more-tab-components").click();
    await expect(page.getByTestId("panel-components").locator(".js-plotly-plot")).toBeVisible();
    await expect(page.getByTestId("more-device-det").locator(".badge.stale")).toBeVisible();
    // components: 4 drawn, the rest one legend click away
    await expect(page.getByTestId("panel-components").locator(".legend .traces")).toHaveCount(8);
    await page.getByTestId("more-tab-charge-balance").click();
    await expect(page.getByTestId("cb-vd")).toBeVisible();
    await expect(page.getByTestId("panel-charge-balance").locator(".js-plotly-plot")).toBeVisible();
    // the chosen tab is remembered
    await page.reload();
    await expect(page.getByTestId("more-tab-charge-balance")).toHaveAttribute("aria-selected", "true");
    // ⚙ 범위 popover of the V_G curve: Esc closes it and returns focus
    await page.getByTestId("more-tab-vg").click();
    await page.getByTestId("vg-range").click();
    await expect(page.getByTestId("vg-range-min")).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("vg-range-min")).toHaveCount(0);
    await expect(page.getByTestId("vg-range")).toBeFocused();
  });

  test("CSV and PNG stay reachable for every panel through its ⋯ menu", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704", { timeout: 15_000 });
    const check = async (id: string) => {
      await expect(page.getByTestId(`panel-${id}`).locator(".js-plotly-plot")).toBeVisible({ timeout: 15_000 });
      await page.getByTestId(`panel-menu-${id}`).click();
      await expect(page.getByTestId(`csv-${id}`)).toBeVisible();
      await expect(page.getByTestId(`png-${id}`)).toBeVisible();
      await page.keyboard.press("Escape");
      await expect(page.getByTestId(`csv-${id}`)).toHaveCount(0);
    };
    await check("iv");
    for (const id of ["vg", "components", "charge-balance"]) {
      await page.getByTestId(`more-tab-${id}`).click();
      await check(id);
    }
    await page.getByTestId("mode-stochastic").click();
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 20_000 });
    await check("mc-iv");
    for (const id of ["dist", "hazard", "cycles", "design-map"]) {
      await page.getByTestId(`more-tab-${id}`).click();
      await check(id);
    }
    // the V_G (MC) tab computes on request, from inside the tab
    await page.getByTestId("more-tab-vg-sto").click();
    await page.getByTestId("vgs-compute").click();
    await check("vg-sto");
  });

  test("stochastic: hero + distribution, one statistics line; 표 보기 opens the compact table, 모든 열 the full one", async ({ page }) => {
    await fresh(page, "#tab=device&mode=stochastic");
    await expect(page.getByTestId("getting-started")).toHaveCount(0); // the hint is about auto-run (deterministic)
    await expect(page.getByTestId("empty-run-mc-iv")).toBeVisible();
    await expect(page.getByTestId("panel-stats")).toBeVisible();
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 20_000 });
    await expect(page.getByTestId("kpi-vlu")).toContainText("측정");
    const meta = page.getByTestId("stats-meta");
    await expect(meta).toContainText("시드");
    await expect(meta).toContainText("래치업 안 된 사이클 0");
    await expect(meta).toContainText("보정 조회표");
    await expect(meta).not.toContainText("calibrated_lookup");
    await expect(page.getByTestId("panel-mc-iv")).not.toContainText("calibrated_lookup");
    await expect(page.getByTestId("panel-dist").locator(".js-plotly-plot")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("dist-ks")).toContainText("KS D");
    await expect(page.getByTestId("stats-table")).toHaveCount(0);
    await page.waitForTimeout(600);
    expect(await plotsInView(page)).toEqual({ inView: 2, fully: 2 });
    const off = await heroOffset(page);
    expect(off.hero - off.main).toBeLessThanOrEqual(246);
    // the stats line sits between the answers and the plots
    const k = (await page.getByTestId("kpis").boundingBox())!;
    const st = (await page.getByTestId("panel-stats").boundingBox())!;
    const grid = (await page.getByTestId("panels-stochastic").boundingBox())!;
    expect(st.y).toBeGreaterThan(k.y);
    expect(grid.y).toBeGreaterThan(st.y);
    expect(st.height).toBeLessThanOrEqual(48);
    await page.screenshot({ path: `${SHOTS}/desktop-stochastic.png` });

    await page.getByTestId("stats-expand").click();
    await expect(page.getByTestId("stats-expand")).toHaveAttribute("aria-expanded", "true");
    const table = page.getByTestId("stats-table");
    await expect(table).toBeVisible();
    await expect(table.locator("thead th[data-col]")).toHaveCount(6);
    for (const c of ["mean", "sd", "p05", "p95", "dmean", "ks_p"]) await expect(table.locator(`thead th[data-col="${c}"]`)).toBeVisible();
    await expect(table.locator('tr[data-row="V_LU:measured"]')).toBeVisible();
    await expect(table.locator('tr[data-row="hazard"]')).toHaveCount(0);
    await expect(page.getByTestId("stats-table-all-cols")).toContainText("19");
    const fits = await page.evaluate(() => {
      const sc = document.querySelector(".stats-scroll") as HTMLElement;
      return sc.scrollWidth <= sc.clientWidth + 1;
    });
    expect(fits).toBe(true);
    await page.getByTestId("panel-stats").screenshot({ path: `${SHOTS}/stats-expanded.png` });
    await page.getByTestId("stats-table-all-cols").click();
    await expect(table.locator('th[data-col="lag1"]')).toBeVisible();
    await expect(table.locator('tr[data-row="hazard"]')).toBeVisible();
    await expect(page.getByTestId("stats-ci")).toBeVisible();
    await expect(page.getByTestId("stats-censored")).toContainText("중도절단");
    // every analysis tab renders its panel under the original test id
    for (const id of ["hazard", "cycles", "vg-sto", "design-map", "dist"]) {
      await page.getByTestId(`more-tab-${id}`).click();
      await expect(page.getByTestId(`panel-${id}`)).toBeVisible();
    }
    await expect(page.getByTestId("dmap-field")).toHaveCount(0);
    await page.getByTestId("more-tab-design-map").click();
    await expect(page.getByTestId("dmap-field")).toBeVisible({ timeout: 15_000 });
  });

  test("dark theme: stochastic default", async ({ page }) => {
    await fresh(page, "#tab=device&mode=stochastic");
    await page.getByTestId("theme-toggle").click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 20_000 });
    await expect(page.getByTestId("panel-dist").locator(".js-plotly-plot")).toBeVisible({ timeout: 15_000 });
    await page.waitForTimeout(700);
    await page.screenshot({ path: `${SHOTS}/dark-stochastic.png` });
  });

  test("모두 보기 restores the full grids and the expanded statistics table", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704", { timeout: 15_000 });
    await page.getByTestId("layout-all").click();
    await expect(page.getByTestId("panels-deterministic")).toHaveClass(/\bgrid\b/);
    for (const id of ["iv", "components", "charge-balance", "vg"]) await expect(page.getByTestId(`panel-${id}`)).toBeVisible();
    await expect(page.getByTestId("more-device-det")).toHaveCount(0);
    await page.getByTestId("mode-stochastic").click();
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("stats-table")).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId("stats-table").locator('th[data-col="lag1"]')).toBeVisible();
    for (const id of ["mc-iv", "dist", "hazard", "vg-sto", "cycles", "design-map"]) await expect(page.getByTestId(`panel-${id}`)).toBeVisible();
    // the choice persists; 간단히 comes back with one click
    await page.reload();
    await expect(page.getByTestId("layout-all")).toHaveAttribute("aria-checked", "true");
    await page.getByTestId("layout-simple").click();
    await expect(page.getByTestId("more-device-sto")).toBeVisible();
  });

  test("tablet 1024: the first load fills the page", async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await fresh(page);
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704", { timeout: 15_000 });
    await expect(page.getByTestId("panel-iv").locator(".js-plotly-plot")).toBeVisible();
    await page.waitForTimeout(700);
    await page.screenshot({ path: `${SHOTS}/tablet-1024.png` });
  });

  test("phone 390: answers in one row, the tab strip becomes a select, no horizontal scroll", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await fresh(page);
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704", { timeout: 15_000 });
    await expect(page.getByTestId("more-select-device-det")).toBeVisible();
    await expect(page.getByTestId("more-tab-vg")).toHaveCount(0);
    const tops = await Promise.all(["kpi-vlu", "kpi-vld", "kpi-window"].map(async (id) => (await page.getByTestId(id).boundingBox())!.y));
    expect(Math.max(...tops) - Math.min(...tops)).toBeLessThanOrEqual(1);
    const heroH = (await page.getByTestId("panel-iv").locator(".js-plotly-plot").boundingBox())!.height;
    expect(heroH).toBeGreaterThanOrEqual(280);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
    await page.getByTestId("more-select-device-det").selectOption("charge-balance");
    await expect(page.getByTestId("cb-vd")).toBeVisible();
    await page.getByTestId("more-select-device-det").selectOption("vg");
    await page.waitForTimeout(700);
    await page.screenshot({ path: `${SHOTS}/phone-device.png`, fullPage: true });
  });

  test("phone 390: the compact statistics table fits without scrolling", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 860 });
    await fresh(page, "#tab=device&mode=stochastic");
    await page.getByTestId("empty-run-mc-iv").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 20_000 });
    await page.getByTestId("stats-expand").click();
    await expect(page.getByTestId("stats-table")).toBeVisible();
    const m = await page.evaluate(() => {
      const sc = document.querySelector(".stats-scroll") as HTMLElement;
      const panel = document.querySelector('[data-testid="panel-stats"]') as HTMLElement;
      return { inner: sc.scrollWidth, outer: sc.clientWidth, right: panel.getBoundingClientRect().right, vw: window.innerWidth, page: document.documentElement.scrollWidth };
    });
    expect(m.inner).toBeLessThanOrEqual(m.outer + 1);
    expect(m.right).toBeLessThanOrEqual(m.vw + 1);
    expect(m.page).toBeLessThanOrEqual(390);
    await expect(page.getByTestId("stats-table").locator('td[data-col="dmean"]').first()).toBeHidden(); // Δ측정 hidden below 420 px
    await page.getByTestId("panel-stats").screenshot({ path: `${SHOTS}/stats-panel-mobile.png` });
  });

  test("English labels: readable engine name and plain wording", async ({ page }) => {
    await fresh(page, "#tab=device&mode=stochastic", { "stl-websim:v1": JSON.stringify({ v: 2, lang: "en" }) });
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 20_000 });
    const meta = page.getByTestId("stats-meta");
    await expect(meta).toContainText("Seed");
    await expect(meta).toContainText("no latch-up in 0 cycles");
    await expect(meta).toContainText("calibrated lookup table");
    await expect(page.getByTestId("kpi-vlu")).toContainText("Turn-on");
    await expect(page.getByTestId("panel-mc-iv")).toContainText("calibrated lookup table");
  });
});
