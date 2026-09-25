// UI tests route authentic direct-solver recordings; see fixtures/geometry/README.md.
// They verify the transport contract and UI, not the physical accuracy of a geometry fit.
import { expect, test, type Page } from "@playwright/test";
import { mkdirSync, readFileSync } from "node:fs";

const KEYS = ["Lg_nm", "W_nm", "Tsi_nm", "EOT_nm", "Tbox_nm", "Nbody_cm3"] as const;
const CHANGED = { Lg_nm: 400, W_nm: 400, Tsi_nm: 40, EOT_nm: 12, Tbox_nm: 100, Nbody_cm3: 1e17 };
const fixture = (name: string) => JSON.parse(readFileSync(`e2e/fixtures/geometry/${name}.json`, "utf8"));
const stored = (page: Page) => page.evaluate(() => JSON.parse(localStorage.getItem("stl-websim:devices") ?? '{"devices":[]}').devices);

async function prepare(page: Page, mock = false) {
  await page.addInitScript(() => {
    if (!sessionStorage.getItem("geometry-init")) {
      localStorage.clear();
      localStorage.setItem("stl-websim:v1", JSON.stringify({ v: 2, lang: "en", mode: "deterministic", autoRun: false, autoRunChosen: true }));
      sessionStorage.setItem("geometry-init", "1");
    }
  });
  await page.route("**/api/health", (r) => r.fulfill({ json: { ok: true, version: "recorded-geometry-transport", workers: 1 } }));
  await page.route("**/api/meta", (r) => r.fulfill({ json: {} }));
  const requests: any[] = [];
  await page.route("**/api/compute/branches?*", async (r) => {
    const payload = r.request().postDataJSON();
    requests.push(payload);
    const name = payload.device.geometry?.W_nm === 400 ? "width-400"
      : payload.device.geometry?.Lg_nm === 400 ? "length-400"
      : payload.device.geometry?.Tsi_nm === 30 ? "tsi-30" : "reference";
    const record = fixture(name);
    // A fixture may only answer its actual geometry, never silently relabel a reference curve.
    expect(payload.device.geometry).toEqual(record.payload.device.geometry);
    expect(payload.device.vbg ?? 0).toBe(record.payload.device.vbg ?? 0);
    await r.fulfill({ json: { id: `geometry-${requests.length}`, kind: "branches", status: "done", progress: 1, result: record.result, cached: false } });
  });
  await page.goto(`/${mock ? "?mock=1" : ""}#tab=device&mode=deterministic`);
  await expect(page.getByTestId("backend-status")).toContainText(mock ? "mock" : "API");
  return requests;
}

async function setGeometry(page: Page, key: typeof KEYS[number], value: string) {
  const input = page.getByTestId(`geometry-${key}`);
  await input.fill(value);
  await input.press("Enter");
}

async function save(page: Page, name: string) {
  await page.getByTestId("dev-save-open").click();
  await page.getByTestId("dev-save-name").fill(name);
  await page.getByTestId("dev-save-submit").click();
  await expect(page.getByTestId("save-device-dialog")).toHaveCount(0);
}

test("six geometry controls are first and persist through save, reload, load and circuit placement", async ({ page }) => {
  await prepare(page);
  const geometry = page.getByTestId("geometry-controls");
  await expect(page.locator(".sidebar-scroll > section").first()).toHaveAttribute("data-testid", "geometry-controls");
  expect(await geometry.locator("input").evaluateAll(inputs => inputs.map(input => input.getAttribute("data-testid"))))
    .toEqual(KEYS.map(key => `geometry-${key}`));
  expect((await geometry.boundingBox())!.y).toBeLessThan((await page.getByTestId("preset-card").boundingBox())!.y);
  for (const key of KEYS) await setGeometry(page, key, String(CHANGED[key]));
  await expect(page.getByTestId("geometry-reset")).toBeVisible();
  await save(page, "Geometry device");
  const first = (await stored(page))[0];
  expect(first.geometry).toEqual(CHANGED);
  expect(first.device.geometry).toEqual(CHANGED);
  await page.reload();
  for (const key of KEYS) expect(Number(await page.getByTestId(`geometry-${key}`).inputValue())).toBe(CHANGED[key]);
  await page.getByTestId("geometry-reset").click();
  await expect(page.getByTestId("geometry-Lg_nm")).toHaveValue("500");
  await page.getByTestId("device-slot-1").getByRole("button", { name: "Load", exact: true }).click();
  for (const key of KEYS) expect(Number(await page.getByTestId(`geometry-${key}`).inputValue())).toBe(CHANGED[key]);
  await page.getByTestId("tab-circuit").click();
  await page.getByTestId("menu-file").click();
  await page.getByTestId("file-new").click();
  await page.getByTestId("lib-Geometry device").getByTestId("lib-place").click();
  const canvas = (await page.getByTestId("sch-svg").boundingBox())!;
  await page.mouse.click(canvas.x + canvas.width / 2, canvas.y + canvas.height / 2);
  await expect(page.locator('[data-testid="sch-svg"] [data-kind="STL"]')).toHaveCount(1);
  await expect.poll(async () => page.evaluate(() => {
    const data = JSON.parse(localStorage.getItem("stl-websim:schematic") ?? "{}");
    return data.doc?.elements.find((e: any) => e.kind === "STL")?.stl?.device.geometry;
  })).toEqual(CHANGED);
  await page.reload();
  await expect(page.locator('[data-testid="sch-svg"] [data-kind="STL"]')).toHaveCount(1);
  expect((await stored(page))[0].device.geometry).toEqual(CHANGED);
});

test("invalid geometry edits do not replace the last valid dimensions", async ({ page }) => {
  await prepare(page);
  await setGeometry(page, "Lg_nm", "400");
  const input = page.getByTestId("geometry-Lg_nm");
  await input.fill("0");
  await expect(input).toHaveAttribute("aria-invalid", "true");
  await expect(page.getByTestId("geometry-controls").getByRole("alert")).toContainText("100");
  await input.press("Tab");
  await expect(input).toHaveValue("400");
  await save(page, "Valid geometry");
  expect((await stored(page))[0].device.geometry.Lg_nm).toBe(400);
});

test("actual resized branch result reaches IDVD and both commercial export metadata", async ({ page }) => {
  const requests = await prepare(page);
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704");
  await setGeometry(page, "W_nm", "400");
  await page.getByTestId("run-button").click();
  await expect.poll(() => requests.length).toBe(2);
  await expect(page.getByTestId("run-status")).toContainText(/done|completed/i);
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704");
  expect(requests[1].device.geometry.W_nm).toBe(400);
  await page.getByTestId("model-export-open").click();
  for (const format of ["ltspice", "verilog-a"]) {
    await page.getByTestId(`model-export-${format}`).check();
    const [download] = await Promise.all([page.waitForEvent("download"), page.getByTestId("model-export-download").click()]);
    const content = readFileSync((await download.path())!, "utf8");
    // the fixed geometry is recorded; the calibration descriptors and engine vector are left out by default
    expect(content).toContain('"fixed_geometry"');
    expect(content).toContain('"W_nm": 400');
    expect(content).toContain('"Tbox_nm": 140');
    expect(content).toContain('"calibration_included": false');
    expect(content).not.toContain("effective_engine_p");
    expect(content).not.toContain("submitted_device");
  }
  expect(requests).toHaveLength(4);
  for (const request of requests.slice(1)) expect(request.device.geometry.W_nm).toBe(400);
});

test("legacy library file receives reference geometry without changing calibration", async ({ page }) => {
  await prepare(page);
  await save(page, "Old device");
  const old = (await stored(page))[0];
  const calibration = old.device.calib;
  delete old.device.geometry;
  old.id = "legacy-geometry";
  old.name = "Legacy import";
  old.geometry = { Lg_nm: 500, W_nm: 200, Tsi_nm: 50, EOT_nm: 14.1 };
  await page.getByTestId("dev-manage").click();
  await page.getByTestId("dm-import-input").setInputFiles({ name: "legacy-device.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(old)) });
  await expect(page.getByTestId("dm-user-list").locator(".dm-row")).toHaveCount(2);
  const loaded = (await stored(page)).find((d: any) => d.name === "Legacy import");
  expect(loaded.device.calib).toEqual(calibration);
  expect(loaded.device.geometry).toMatchObject({ Lg_nm: 500, W_nm: 200, Tsi_nm: 50, EOT_nm: 14.1, Tbox_nm: 140 });
  expect(loaded.device.geometry.Nbody_cm3).toBeCloseTo(2.295773162796593e17, -3);
});

test("shorter L and thinner Tsi show their actual independently computed threshold changes", async ({ page }) => {
  const requests = await prepare(page);
  await setGeometry(page, "Lg_nm", "400");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.582");
  await expect(page.getByTestId("kpi-vld-value")).toContainText("2.242");
  await page.getByTestId("geometry-reset").click();
  await setGeometry(page, "Tsi_nm", "30");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.815");
  await expect(page.getByTestId("kpi-vld-value")).toContainText("2.635");
  expect(requests.map(p => [p.device.geometry.Lg_nm, p.device.geometry.Tsi_nm])).toEqual([[400, 50], [500, 30]]);
});

test("offline demo refuses resized simulation and never supplies reference thresholds", async ({ page }) => {
  await prepare(page, true);
  await setGeometry(page, "Lg_nm", "400");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("panel-iv")).toContainText("Connect to the live compute server to simulate changed geometry or V_BG.");
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("—");
  await expect(page.getByTestId("panel-iv").locator(".js-plotly-plot")).toHaveCount(0);
});

test("the live server refuses a too-short L for this Nbody and the message names the limit", async ({ page }) => {
  // no recorded transport here: the request reaches the real backend (STL_API), which answers HTTP 422
  await page.addInitScript(() => {
    if (!sessionStorage.getItem("geometry-init")) {
      localStorage.clear();
      localStorage.setItem("stl-websim:v1", JSON.stringify({ v: 2, lang: "en", mode: "deterministic", autoRun: false, autoRunChosen: true }));
      sessionStorage.setItem("geometry-init", "1");
    }
  });
  await page.goto("/#tab=device&mode=deterministic");
  await expect(page.getByTestId("backend-status")).toContainText("API");
  await setGeometry(page, "Lg_nm", "120");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("panel-iv")).toContainText("At this Nbody, L is too short to leave a neutral body. Make L longer than 153.6 nm or raise Nbody.");
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("—");
});

for (const width of [1440, 390]) {
  test(`${width}px geometry screenshot with actual resized model result`, async ({ page }) => {
    await page.setViewportSize({ width, height: width === 1440 ? 1000 : 844 });
    await prepare(page);
    await page.getByTestId("lang-toggle").click();
    if (width < 1100) await page.locator(".sidebar-btn").click();
    await setGeometry(page, "Lg_nm", "400");
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.582");
    // KPI updates precede lazy Plotly loading. Only capture once real trace paths have drawn.
    const traces = page.getByTestId("panel-iv").locator(".scatterlayer .trace path.js-line");
    await expect.poll(() => traces.count()).toBeGreaterThanOrEqual(2);
    await expect.poll(() => traces.evaluateAll(paths => paths.some(path => {
      const bounds = (path as SVGGraphicsElement).getBBox();
      return bounds.width > 30 && bounds.height > 30;
    }))).toBe(true);
    await page.evaluate(() => document.fonts.ready);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
    await expect(page.getByTestId("geometry-controls")).toBeInViewport();
    const box = (await page.getByTestId("geometry-controls").boundingBox())!;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(width);
    mkdirSync("review", { recursive: true });
    await page.evaluate(() => {
      const label = document.createElement("div");
      label.textContent = "실제 계산 결과 · 기록 보기";
      label.style.cssText = "position:fixed;bottom:6px;right:8px;z-index:9999;font:11px var(--font);background:#fff;color:#475569;border:1px solid #cbd5e1;border-radius:4px;padding:4px 7px";
      document.body.append(label);
    });
    await page.screenshot({ path: `review/${width === 1440 ? "desktop" : "mobile"}-geometry.png`, fullPage: true });
  });
}

test("Geometry guide opens its published model document and fits mobile", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await prepare(page);
  await page.locator(".sidebar-btn").click();
  await page.getByTestId("tip-geometry").click();
  await expect(page.getByTestId("guide-pop")).toBeVisible();
  const [document] = await Promise.all([page.waitForEvent("popup"), page.getByTestId("guide-pop-more").click()]);
  await expect(document).toHaveURL(/\/docs\/geometry-model\.html$/);
  await document.setViewportSize({ width: 390, height: 844 });
  await expect(document.locator("body")).toContainText("Geometry");
  await expect.poll(() => document.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
});
