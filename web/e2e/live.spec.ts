// Live backend check: runs only when the FastAPI server answers on :8000 (through the Vite proxy).
import { expect, test } from "@playwright/test";

test("live: deterministic run shows the paper folds (3.70 / 2.60 V)", async ({ page, request }) => {
  let ok = false;
  try {
    const r = await request.get("http://127.0.0.1:8000/api/health", { timeout: 3000 });
    ok = r.ok() && (await r.json()).ok === true;
  } catch {
    ok = false;
  }
  test.skip(!ok, "backend not running on :8000");
  test.setTimeout(240_000);
  await page.addInitScript(() => {
    try {
      localStorage.clear();
    } catch {
      /* ignore */
    }
  });
  await page.goto("/#tab=device&mode=deterministic");
  await expect(page.getByTestId("backend-status")).toContainText("API", { timeout: 15_000 });
  await expect(page.getByTestId("offline-banner")).toHaveCount(0);
  await page.getByTestId("preset-paper").click();
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.70", { timeout: 180_000 });
  await expect(page.getByTestId("kpi-vld-value")).toContainText("2.59");
  await expect(page.getByTestId("panel-iv").locator(".js-plotly-plot")).toBeVisible();
  await expect(page.getByTestId("panel-charge-balance").locator(".js-plotly-plot")).toBeVisible({ timeout: 120_000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: "e2e/screenshots/live-deterministic.png" });
});

test("live: stochastic run (paper preset) shows V_LU mean ± σ", async ({ page, request }) => {
  let ok = false;
  try {
    const r = await request.get("http://127.0.0.1:8000/api/health", { timeout: 3000 });
    ok = r.ok() && (await r.json()).ok === true;
  } catch {
    ok = false;
  }
  test.skip(!ok, "backend not running on :8000");
  test.setTimeout(420_000);
  await page.addInitScript(() => {
    try {
      localStorage.clear();
    } catch {
      /* ignore */
    }
  });
  await page.goto("/#tab=device&mode=stochastic");
  await expect(page.getByTestId("backend-status")).toContainText("API", { timeout: 15_000 });
  await page.getByTestId("preset-paper").click();
  await page.getByTestId("run-button").click();
  // paper record: mean V_LU ≈ 3.63 V, σ ≈ 120 mV (engine/docs/VALIDATION.md)
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.6", { timeout: 400_000 });
  await expect(page.getByTestId("panel-hazard").locator(".js-plotly-plot")).toBeVisible({ timeout: 400_000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: "e2e/screenshots/live-stochastic.png" });
});

test("live: circuit load-line bench reproduces the folds", async ({ page, request }) => {
  let ok = false;
  try {
    const r = await request.get("http://127.0.0.1:8000/api/health", { timeout: 3000 });
    ok = r.ok() && (await r.json()).ok === true;
  } catch {
    ok = false;
  }
  test.skip(!ok, "backend not running on :8000");
  test.setTimeout(300_000);
  await page.addInitScript(() => {
    try {
      localStorage.clear();
    } catch {
      /* ignore */
    }
  });
  await page.goto("/#tab=circuit&mode=deterministic");
  await expect(page.getByTestId("backend-status")).toContainText("API", { timeout: 15_000 });
  await page.getByTestId("bench-load_line").click();
  await page.getByTestId("run-button").click();
  const summary = page.getByTestId("circuit-summary");
  await expect(summary).toBeVisible({ timeout: 240_000 });
  // the circuit result must pass the §4 shape guard (no error box) and show the latch-up voltage
  await expect(page.getByTestId("kpi-c-V_LU-value")).toContainText("3.70");
  await expect(page.getByTestId("panel-waves").locator(".js-plotly-plot")).toBeVisible();
  await page.waitForTimeout(800);
  await page.screenshot({ path: "e2e/screenshots/live-circuit.png", fullPage: true });
});

test("live: pulse bench uses the server's rise/fall default (auto → 10 µs)", async ({ page, request }) => {
  let ok = false;
  try {
    const r = await request.get("http://127.0.0.1:8000/api/health", { timeout: 3000 });
    ok = r.ok() && (await r.json()).ok === true;
  } catch {
    ok = false;
  }
  test.skip(!ok, "backend not running on :8000");
  test.setTimeout(300_000);
  await page.addInitScript(() => {
    try {
      localStorage.clear();
    } catch {
      /* ignore */
    }
  });
  await page.goto("/#tab=circuit&mode=deterministic");
  await expect(page.getByTestId("backend-status")).toContainText("API", { timeout: 15_000 });
  await page.getByTestId("bench-pulse").click();
  await page.getByTestId("run-button").click();
  const chips = page.getByTestId("resolved-params");
  await expect(chips).toContainText("rise_s = 1.000e-5", { timeout: 240_000 });
  await expect(chips).toContainText("fall_s = 1.000e-5");
});

test("live: HTTP 429 (queue full) shows 'server busy' and retries once after Retry-After", async ({ page, request }) => {
  let ok = false;
  try {
    const r = await request.get("http://127.0.0.1:8000/api/health", { timeout: 3000 });
    ok = r.ok() && (await r.json()).ok === true;
  } catch {
    ok = false;
  }
  test.skip(!ok, "backend not running on :8000");
  test.setTimeout(120_000);
  await page.addInitScript(() => {
    try {
      localStorage.clear();
    } catch {
      /* ignore */
    }
  });
  let rejected = 0;
  await page.route("**/api/compute/branches*", async (route) => {
    if (rejected === 0) {
      rejected++;
      await route.fulfill({ status: 429, headers: { "Retry-After": "2", "Content-Type": "application/json" }, body: JSON.stringify({ detail: "job queue full" }) });
    } else await route.continue();
  });
  await page.goto("/#tab=device&mode=deterministic");
  await expect(page.getByTestId("backend-status")).toContainText("API", { timeout: 15_000 });
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("panel-iv")).toContainText("서버가 바쁩니다", { timeout: 10_000 });
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.70", { timeout: 60_000 });
  expect(rejected).toBe(1);
  await expect(page.getByTestId("panel-iv").locator(".err-box")).toHaveCount(0);
});
