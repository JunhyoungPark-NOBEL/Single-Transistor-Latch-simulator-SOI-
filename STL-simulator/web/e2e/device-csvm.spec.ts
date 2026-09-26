// UI verification with authentic direct-solver captures (see fixtures/csvm/README.md).
// HTTP transport is routed; no waveform or derived metric is invented by the test.
import { expect, test, type Page } from "@playwright/test";
import { readFileSync, mkdirSync } from "node:fs";

const fixture = (name: string) => JSON.parse(readFileSync(`e2e/fixtures/csvm/${name}.json`, "utf8"));

async function prepare(page: Page) {
  await page.addInitScript(() => {
    if (!sessionStorage.getItem("csvm-test-init")) {
      localStorage.clear();
      localStorage.setItem("stl-websim:v1", JSON.stringify({ v: 2, lang: "en", mode: "deterministic", autoRun: false, autoRunChosen: true }));
      sessionStorage.setItem("csvm-test-init", "1");
    }
  });
  await page.route("**/api/health", (r) => r.fulfill({ json: { ok: true, version: "test-transport", workers: 1 } }));
  await page.route("**/api/meta", (r) => r.fulfill({ json: {} }));
  const requests: any[] = [];
  await page.route("**/api/compute/circuit?*", async (r) => {
    const payload = r.request().postDataJSON(); requests.push(payload);
    const current = payload.netlist.elements.find((e: any) => e.type === "I").wave.value;
    const cap = payload.netlist.elements.find((e: any) => e.type === "C").value;
    const name = current === 30e-9 ? "no-oscillation" : payload.tran.t_stop_s === .004 ? "short" : cap === 2e-12 ? "capacitance-2pf" : current === 2e-9 ? "current-2na" : "default";
    await r.fulfill({ json: { id: `csvm-${requests.length}`, kind: "circuit", status: "done", progress: 1, result: fixture(name), cached: false } });
  });
  await page.route("**/api/compute/branches?*", (r) => {
    const result = JSON.parse(readFileSync("src/devices/fixtures/export-reference.json", "utf8")).result;
    return r.fulfill({ json: { id: "branches", kind: "branches", status: "done", progress: 1, result } });
  });
  await page.goto("/#tab=device&mode=deterministic");
  await expect(page.getByTestId("backend-status")).toContainText("API");
  return requests;
}

async function setting(page: Page, name: string, value: string) {
  await page.getByTestId(`csvm-${name}`).fill(value);
  await page.getByTestId(`csvm-${name}`).press("Enter");
}
const frequency = async (page: Page) => {
  const text = await page.getByTestId("kpi-frequency-value").innerText();
  return Number.parseFloat(text) * (text.includes("kHz") ? 1000 : 1);
};

test("CSVM switch uses calibrated transient, updates metrics, and restores VSCM", async ({ page }) => {
  const requests = await prepare(page);
  await expect(page.getByTestId("field-vd_max")).toBeVisible();
  await page.getByRole("radio", { name: "Current · CSVM", exact: true }).click();
  await expect(page.getByTestId("field-vd_max")).toHaveCount(0);
  await expect(page.getByTestId("field-vg")).toBeVisible();
  await expect(page.getByTestId("kpi-vlu")).toHaveCount(0);
  await expect(page.getByTestId("kpi-frequency-value")).toContainText("—");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("csvm-status")).toContainText("startup excluded");
  expect(await frequency(page)).toBeGreaterThan(850);
  expect(await frequency(page)).toBeLessThan(875);
  await expect(page.getByTestId("kpi-vtop-value")).toContainText("3.7");
  await expect(page.getByTestId("kpi-vbottom-value")).toContainText("2.5");
  await expect(page.getByTestId("panel-csvm").locator(".js-plotly-plot")).toBeVisible();
  expect(requests[0].netlist.elements.find((e: any) => e.type === "STL").device.calib.beta).toBeGreaterThan(7);
  expect(requests[0].netlist.elements.find((e: any) => e.name === "VG").wave.value).toBe(-2);

  await setting(page, "current_A", "2");
  await expect(page.getByTestId("csvm-kpis")).toHaveClass(/stale/);
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("csvm-kpis")).not.toHaveClass(/stale/);
  expect(await frequency(page)).toBeGreaterThan(1500);
  expect(requests.at(-1).netlist.elements[0].wave.value).toBe(2e-9);
  await setting(page, "current_A", "1");
  await setting(page, "capacitance_F", "2");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("csvm-kpis")).not.toHaveClass(/stale/);
  expect(await frequency(page)).toBeLessThan(460);
  await page.reload();
  await expect(page.getByRole("radio", { name: "Current · CSVM", exact: true })).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("csvm-capacitance_F")).toHaveValue("2");
  await page.getByRole("radio", { name: "Voltage · VSCM", exact: true }).click();
  await expect(page.getByTestId("field-vd_max")).toBeVisible();
  await expect(page.getByTestId("kpi-vlu")).toBeVisible();
  await expect(page.getByTestId("csvm-kpis")).toHaveCount(0);
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704");
});

test("short runs and nonoscillating bias never fabricate frequency", async ({ page }) => {
  await prepare(page);
  await page.getByRole("radio", { name: "Current · CSVM", exact: true }).click();
  await setting(page, "duration_s", "4");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("csvm-status")).toContainText(/cycle|longer/i);
  await expect(page.getByTestId("kpi-frequency-value")).toContainText("—");
  await setting(page, "duration_s", ".4");
  await setting(page, "current_A", "30");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("csvm-kpis")).not.toHaveClass(/stale/);
  await expect(page.getByTestId("kpi-frequency-value")).toContainText("—");
  await expect(page.getByTestId("csvm-status")).toContainText(/oscillation|cycle/i);
});

test("offline CSVM rejects the demo and compact controls fit mobile", async ({ page }) => {
  await prepare(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/?mock=1#tab=device&mode=deterministic");
  await page.getByRole("radio", { name: "Current · CSVM", exact: true }).click();
  await page.getByTestId("panel-csvm").getByRole("button", { name: /Run/ }).click();
  await expect(page.getByTestId("panel-csvm")).toContainText("requires a live transient solver");
  await expect(page.getByTestId("kpi-frequency-value")).toContainText("—");
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
});

for (const width of [1440, 390]) {
  test(`${width}px CSVM screenshots show actual simulated trace`, async ({ page }) => {
    await prepare(page);
    await page.setViewportSize({ width, height: width === 1440 ? 1000 : 844 });
    await page.getByRole("radio", { name: "Current · CSVM", exact: true }).click();
    await page.getByTestId("panel-csvm").getByRole("button", { name: /Run/ }).click();
    await expect(page.getByTestId("csvm-status")).toContainText("startup excluded");
    await expect(page.getByTestId("panel-csvm").locator(".js-plotly-plot")).toBeVisible();
    await page.getByTestId("lang-toggle").click();
    await page.evaluate(() => document.fonts.ready);
    mkdirSync("review", { recursive: true });
    await page.evaluate(() => {
      const label = document.createElement("div");
      label.textContent = "실제 계산 결과 · 기록 보기";
      label.style.cssText = "position:fixed;bottom:6px;right:8px;z-index:9999;font:11px var(--font);background:#fff;color:#475569;border:1px solid #cbd5e1;border-radius:4px;padding:4px 7px";
      document.body.append(label);
    });
    await page.screenshot({ path: `review/${width === 1440 ? "desktop" : "mobile"}-csvm.png`, fullPage: true });
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
  });
}
