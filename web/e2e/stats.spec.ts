// Statistics summary (stochastic device tab) in mock mode: table rows, measured comparison, tooltips, CSV,
// language, narrow-screen scrolling, CDF KS line.
import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

const SHOTS = "e2e/screenshots";

async function freshStochastic(page: Page) {
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
  await page.goto("/?mock=1#tab=device&mode=stochastic");
  await expect(page.getByTestId("mode-toggle")).toBeVisible();
}

async function run(page: Page) {
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("stats-table")).toBeVisible({ timeout: 20_000 });
}

test.describe("statistics summary (mock mode)", () => {
  test("stochastic run shows the statistics panel under the KPI strip", async ({ page }) => {
    await freshStochastic(page);
    await expect(page.getByTestId("panel-stats")).toBeVisible();
    await run(page);
    const table = page.getByTestId("stats-table");
    for (const row of ["V_LU", "V_LD", "window"]) await expect(table.locator(`tr[data-row="${row}"]`)).toBeVisible();
    // hazard (carrier noise only) row once the hazard job is done
    await expect(table.locator('tr[data-row="hazard"]')).toBeVisible({ timeout: 15_000 });
    // numbers are formatted: mean in V (3 decimals), SD in mV
    const mean = await table.locator('tr[data-row="V_LU"] td[data-col="mean"]').innerText();
    expect(mean).toMatch(/^\d\.\d{3,4}$/);
    const sd = await table.locator('tr[data-row="V_LU"] td[data-col="sd"]').innerText();
    expect(Number(sd)).toBeGreaterThan(1); // mV
    await expect(page.getByTestId("stats-meta")).toContainText(/시드|Seed/);
    await expect(page.getByTestId("stats-ci")).toBeVisible();
    // the panel sits between the KPI strip and the plot grid
    const kpiBox = await page.getByTestId("kpis").boundingBox();
    const statsBox = await page.getByTestId("panel-stats").boundingBox();
    const gridBox = await page.getByTestId("panels-stochastic").boundingBox();
    expect(statsBox!.y).toBeGreaterThan(kpiBox!.y);
    expect(gridBox!.y).toBeGreaterThan(statsBox!.y);
    await page.getByTestId("panel-stats").screenshot({ path: `${SHOTS}/stats-panel-mock.png` });
  });

  test("measured record adds a sub-row and the KS comparison (photo preset)", async ({ page }) => {
    await freshStochastic(page);
    await page.getByTestId("preset-photo").click();
    await run(page);
    const table = page.getByTestId("stats-table");
    await expect(table.locator('tr[data-row="V_LU:measured"]')).toBeVisible();
    await expect(table.locator('tr[data-row="V_LU"] td[data-col="ks_d"]')).toHaveText(/^0\.\d{3}$/);
    await expect(table.locator('tr[data-row="V_LU"] td[data-col="ks_p"]')).toHaveText(/^(<0\.001|\d\.\d{2,3})$/);
    await expect(table.locator('tr[data-row="V_LU"] td[data-col="dmean"]')).toHaveText(/^[+−]?\d/);
    // comparison cells stay empty in the measured sub-row
    await expect(table.locator('tr[data-row="V_LU:measured"] td[data-col="ks_d"]')).toHaveText("");
  });

  test("header tooltips explain each statistic, in Korean and English", async ({ page }) => {
    await freshStochastic(page);
    await run(page);
    const head = page.getByTestId("stats-table").locator('th[data-col="lag1"] .hovertip');
    await head.hover();
    await expect(page.getByRole("tooltip")).toContainText("자기상관(lag-1)");
    await page.mouse.move(0, 0);
    await page.getByRole("button", { name: /English/ }).first().click();
    await expect(page.getByTestId("panel-stats")).toContainText("Statistics");
    await head.focus();
    await expect(page.getByRole("tooltip")).toContainText("Lag-1 autocorrelation");
    await page.keyboard.press("Escape");
    await expect(page.getByRole("tooltip")).toHaveCount(0);
  });

  test("CSV download holds every statistic in SI units", async ({ page }) => {
    await freshStochastic(page);
    await run(page);
    const [dl] = await Promise.all([page.waitForEvent("download"), page.getByTestId("stats-table-download").click()]);
    expect(dl.suggestedFilename()).toBe("statistics_vlu_vld.csv");
    const text = readFileSync((await dl.path())!, "utf8");
    const [header, first] = text.trim().split("\n");
    expect(header.split(",").slice(0, 8)).toEqual(["quantity", "series", "unit", "n_total", "n", "censored", "mean", "ci95_half"]);
    expect(header).toContain("kurtosis_excess");
    expect(header).toContain("ks_p");
    const cells = first.split(",");
    expect(cells[0]).toBe("V_LU");
    expect(cells[2]).toBe("V");
    expect(Number(cells[6])).toBeGreaterThan(1); // mean in volts, full precision
  });

  test("CDF view: KS line under the plot; narrow screens scroll the table inside its card", async ({ page }) => {
    await freshStochastic(page);
    await page.getByTestId("preset-photo").click();
    await run(page);
    const dist = page.getByTestId("panel-dist");
    await dist.getByRole("radio", { name: "CDF" }).click();
    await expect(dist.getByTestId("dist-ks")).toContainText("KS D");
    await page.setViewportSize({ width: 390, height: 860 });
    await page.waitForTimeout(400);
    const m = await page.evaluate(() => {
      const sc = document.querySelector(".stats-scroll") as HTMLElement;
      const panel = document.querySelector('[data-testid="panel-stats"]') as HTMLElement;
      return { inner: sc.scrollWidth, outer: sc.clientWidth, right: panel.getBoundingClientRect().right, vw: window.innerWidth };
    });
    expect(m.inner).toBeGreaterThan(m.outer); // the table scrolls inside the card …
    expect(m.right).toBeLessThanOrEqual(m.vw + 1); // … and the card itself stays within the viewport
    await page.getByTestId("panel-stats").screenshot({ path: `${SHOTS}/stats-panel-mobile.png` });
  });
});
