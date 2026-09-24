// Mock-mode end-to-end checks (no backend needed): rendering, mode toggle, Details window (KaTeX, drag,
// Esc, focus return), language toggle, presets, circuit and physics tabs, screenshots.
import { expect, test, type Page } from "@playwright/test";

const SHOTS = "e2e/screenshots";

async function fresh(page: Page, hash = "#tab=device&mode=deterministic") {
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
  await page.goto(`/?mock=1${hash}`);
  await expect(page.getByTestId("mode-toggle")).toBeVisible();
}

test.describe("STL simulator (mock mode)", () => {
  test("renders the shell with demo banner, tabs, mode toggle and grouped parameters", async ({ page }) => {
    await fresh(page);
    await expect(page.getByText("STL Simulator").first()).toBeVisible();
    await expect(page.getByTestId("offline-banner")).toBeVisible();
    for (const t of ["device", "circuit", "validation", "physics"]) await expect(page.getByTestId(`tab-${t}`)).toBeVisible();
    await expect(page.getByTestId("group-bias")).toBeVisible();
    await expect(page.getByTestId("group-light")).toBeVisible();
    await expect(page.getByTestId("group-calib")).toBeVisible();
    await expect(page.getByTestId("panels-deterministic")).toBeVisible();
    // KaTeX symbols in field labels
    await expect(page.getByTestId("field-vg").locator(".katex").first()).toBeVisible();
  });

  test("deterministic run shows folds in the KPI strip", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.70", { timeout: 15_000 });
    await expect(page.getByTestId("kpi-vld-value")).toContainText("2.59");
    await expect(page.getByTestId("panel-iv").locator(".js-plotly-plot")).toBeVisible();
    await page.screenshot({ path: `${SHOTS}/desktop-deterministic.png` });
  });

  test("mode toggle switches visible groups and panels", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("group-stoch")).toHaveCount(0);
    await expect(page.getByTestId("group-local")).toHaveCount(0);
    await page.getByTestId("mode-stochastic").click();
    await expect(page.getByTestId("mode-stochastic")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("group-stoch")).toBeVisible();
    await expect(page.getByTestId("group-local")).toBeVisible();
    await expect(page.getByTestId("panels-stochastic")).toBeVisible();
    await expect(page.getByTestId("panel-hazard")).toBeVisible();
    await expect(page.getByTestId("panel-iv")).toHaveCount(0);
    await expect(page).toHaveURL(/mode=stochastic/);
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 15_000 });
    await expect(page.getByTestId("stats-table")).toBeVisible();
    await expect(page.getByTestId("panel-hazard").locator(".js-plotly-plot")).toBeVisible({ timeout: 15_000 });
    await page.waitForTimeout(600);
    await page.screenshot({ path: `${SHOTS}/desktop-stochastic.png` });
    // experimental action point shows the uncalibrated warning
    await page.getByTestId("field-ls_action").locator("select").selectOption("junction");
    await expect(page.getByTestId("uncalibrated-warning")).toBeVisible();
    await page.getByTestId("mode-deterministic").click();
    await expect(page.getByTestId("group-stoch")).toHaveCount(0);
  });

  test("Details window: opens next to the button with KaTeX, drags, navigates, closes with Esc", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.70", { timeout: 15_000 });
    const btn = page.getByTestId("details-group-bias");
    await btn.click();
    const win = page.getByTestId("physics-window");
    await expect(win).toBeVisible();
    await expect(win.getByRole("heading", { level: 2 })).toBeFocused();
    await expect(win.locator(".katex").first()).toBeVisible();
    // the page is still usable (non-modal): no page navigation, tabs still there
    await expect(page.getByTestId("main-device")).toBeVisible();
    const box = await win.boundingBox();
    const bbox = await btn.boundingBox();
    expect(box && bbox).toBeTruthy();
    // opened next to the button (to its right) and inside the viewport
    expect(box!.x).toBeGreaterThan(bbox!.x);
    expect(box!.x + box!.width).toBeLessThanOrEqual(1440);
    await page.screenshot({ path: `${SHOTS}/details-window.png` });
    // drag by the header
    const head = page.getByTestId("physics-window-header");
    const hb = (await head.boundingBox())!;
    await page.mouse.move(hb.x + 60, hb.y + 20);
    await page.mouse.down();
    await page.mouse.move(hb.x + 60 - 150, hb.y + 20 + 60, { steps: 6 });
    await page.mouse.up();
    const box2 = (await win.boundingBox())!;
    expect(Math.round(box2.x)).toBe(Math.round(box!.x - 150));
    expect(Math.round(box2.y)).toBe(Math.round(box!.y + 60));
    // related topic replaces the content, back restores it
    const title1 = await win.getByRole("heading", { level: 2 }).textContent();
    const related = win.locator("[data-testid^=related-]").first();
    if (await related.count()) {
      await related.click();
      await expect(win.getByRole("heading", { level: 2 })).not.toHaveText(title1 ?? "");
      await page.getByTestId("physics-back").click();
      await expect(win.getByRole("heading", { level: 2 })).toHaveText(title1 ?? "");
    }
    // Esc closes and focus returns to the trigger
    await page.keyboard.press("Escape");
    await expect(win).toHaveCount(0);
    await expect(btn).toBeFocused();
  });

  test("language toggle switches UI strings", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("tab-device")).toContainText("소자");
    await expect(page.getByTestId("run-button")).toContainText("결정론");
    await page.getByTestId("lang-toggle").click();
    await expect(page.getByTestId("tab-device")).toHaveText("Device");
    await expect(page.getByTestId("run-button")).toContainText("Run deterministic");
    await expect(page.getByTestId("group-bias")).toContainText("Bias & sweep");
    await page.getByTestId("lang-toggle").click();
    await expect(page.getByTestId("tab-device")).toContainText("소자");
  });

  test("editing a value marks the preset as modified; reset restores it", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("preset-label")).not.toContainText("수정");
    const input = page.getByTestId("field-vg").locator("input.input");
    await input.fill("-1.8");
    await input.press("Enter");
    await expect(page.getByTestId("preset-label")).toContainText("논문 소자에서 수정");
    await expect(page.getByTestId("field-vg").locator(".field-changed")).toBeVisible();
    // out-of-range input shows a validation message and is not committed
    await input.fill("-9");
    await expect(page.getByTestId("field-vg").locator(".field-err")).toBeVisible();
    await input.press("Escape");
    await page.getByTestId("group-bias").getByRole("button", { name: /초기화|Reset/ }).click();
    await expect(page.getByTestId("preset-label")).not.toContainText("수정");
    // photo preset loads its defaults (power mode, 1200 V/s)
    await page.getByTestId("preset-photo").click();
    await expect(page.getByTestId("field-vg").locator("input.input")).toHaveValue("-1.8");
    await expect(page.getByTestId("light-conversion")).toContainText("pA/mW");
  });

  test("circuit tab: bench cards, generic result rendering", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=deterministic");
    await expect(page.getByTestId("bench-picker")).toBeVisible();
    await expect(page.getByTestId("schematic")).toBeVisible();
    await expect(page.getByTestId("group-bench")).toBeVisible();
    await expect(page.getByTestId("group-solver")).toBeVisible();
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("circuit-summary")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("panel-waves").locator(".js-plotly-plot")).toBeVisible();
    await expect(page.getByTestId("solver-stats")).toBeVisible();
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${SHOTS}/circuit.png` });
    await page.getByTestId("bench-pbit").click();
    await expect(page.getByTestId("bench-pbit")).toHaveAttribute("aria-checked", "true");
  });

  test("physics tab lists all topics with a working search", async ({ page }) => {
    await fresh(page, "#tab=physics&mode=deterministic");
    await expect(page.getByTestId("physics-tab")).toBeVisible();
    await expect(page.getByTestId("topic-overview")).toBeVisible();
    const before = await page.locator("[data-testid^=topic-]").count();
    expect(before).toBe(18);
    await page.getByTestId("physics-search").fill("hazard");
    await expect.poll(async () => page.locator("[data-testid^=topic-]").count()).toBeLessThan(before);
  });

  test("validation tab runs checks", async ({ page }) => {
    await fresh(page, "#tab=validation&mode=deterministic");
    await page.getByTestId("val-fast").click();
    await expect(page.getByTestId("validation-table")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("validation-table").locator(".pass.yes").first()).toBeVisible();
  });

  test("dark theme", async ({ page }) => {
    await fresh(page, "#tab=device&mode=stochastic");
    await page.getByTestId("theme-toggle").click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 15_000 });
    await expect(page.getByTestId("panel-hazard").locator(".js-plotly-plot")).toBeVisible({ timeout: 15_000 });
    await page.waitForTimeout(600);
    await page.screenshot({ path: `${SHOTS}/dark-stochastic.png` });
  });

  test("narrow screen: sidebar becomes a drawer", async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await fresh(page);
    const toggle = page.getByRole("button", { name: /파라미터 패널|Parameter panel/ });
    await expect(toggle).toBeVisible();
    await page.screenshot({ path: `${SHOTS}/tablet-1024.png` });
  });
});
