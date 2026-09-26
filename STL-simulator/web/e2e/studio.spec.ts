// Studio release gates. Live tests make real HTTP requests through Vite's API proxy.
// Transport-contract tests are separately named and explicitly route their responses.
import { expect, test, type Page, type Locator } from "@playwright/test";
import { mkdirSync, writeFileSync, readFileSync } from "node:fs";

const review = "review/studio";
const live = process.env.STUDIO_LIVE === "1";

async function fresh(page: Page, tab = "device", extra = "") {
  await page.addInitScript(() => {
    if (!sessionStorage.getItem("studio-qa")) {
      localStorage.clear();
      localStorage.setItem("stl-websim:v1", JSON.stringify({ v: 2, lang: "en", theme: "light", mode: "deterministic", autoRun: false, autoRunChosen: true }));
      sessionStorage.setItem("studio-qa", "1");
    }
  });
  await page.goto(`/${extra}#tab=${tab}&mode=deterministic`);
}
async function plotted(graph: Locator) {
  await expect(graph).toBeVisible({ timeout: 180_000 });
  await expect.poll(() => graph.evaluate(el => {
    const d = (el as any).data as { x?: number[]; y?: number[] }[] | undefined;
    return (d ?? []).some(tr => (tr.x?.length ?? 0) > 10 && tr.y?.some(Number.isFinite));
  }), { timeout: 180_000 }).toBe(true);
  await expect(graph.locator(".scatterlayer .trace path").first()).toHaveAttribute("d", /.+/);
}
async function screenshot(page: Page, name: string) {
  mkdirSync(review, { recursive: true });
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => window.scrollTo(0, 0));
  if (await page.getByTestId("dev-toast").count()) await page.getByTestId("dev-toast").getByRole("button").click();
  await expect(page.getByTestId("sch-toast")).toHaveCount(0, { timeout: 5000 });
  for (const close of await page.locator(".plotly-notifier .notifier-close").all()) await close.click();
  await expect(page.locator(".plotly-notifier .notifier-note")).toHaveCount(0, { timeout: 5000 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
  if ((page.viewportSize()?.width ?? 1440) < 1100 && await page.getByTestId("sidebar").count()) { await expect(page.getByTestId("sidebar")).not.toHaveClass(/ open/); await expect(page.getByTestId("sidebar")).not.toBeInViewport(); }
  await page.screenshot({ path: `${review}/${name}.png`, fullPage: true });
}
function recordResponses(page: Page) {
  const jobs: any[] = [];
  page.on("response", async response => {
    if (/\/api\/(compute|jobs)\//.test(response.url()) && response.ok()) {
      try { const body = await response.json(); if (body.status === "done") jobs.push({ url: response.url(), body }); } catch { /* Navigation can abort completed response reads. */ }
    }
  });
  return jobs;
}
async function save(page: Page, name: string) {
  await page.getByTestId("dev-save-open").click();
  await page.getByTestId("dev-save-name").fill(name);
  await page.getByTestId("dev-save-submit").click();
  await expect(page.getByTestId("save-device-dialog")).toHaveCount(0);
}
async function geometry(page: Page, name: string, value: string) {
  await page.getByTestId(`geometry-${name}`).fill(value);
  await page.getByTestId(`geometry-${name}`).press("Enter");
}
async function world(page: Page) {
  await page.evaluate(() => window.scrollTo(0, 0));
  const box = (await page.getByTestId("sch-svg").boundingBox())!;
  const k = Number(await page.getByTestId("sch-svg").getAttribute("data-k"));
  return (x: number, y: number) => ({ x: box.x + box.width / 2 + x * k, y: box.y + box.height / 2 + y * k });
}
async function place(page: Page, tool: string, x: number, y: number) {
  await page.getByTestId(`tool-${tool}`).click();
  const at = await world(page); const p = at(x, y); await page.mouse.click(p.x, p.y);
}

test.describe("Studio live HTTP solver", () => {
  test.skip(!live, "Set STUDIO_LIVE=1 with the actual FastAPI solver running.");
  test.setTimeout(240_000);
  test.beforeEach(async ({ request }) => {
    const health = await request.get("/api/health");
    expect(health.ok()).toBe(true);
    expect((await health.json()).ok).toBe(true);
  });

  test("reference and L400 IDVD use live geometry model; zoomed log/linear remains valid", async ({ page }) => {
    const jobs = recordResponses(page);
    await fresh(page);
    await expect(page.getByTestId("geometry-Lg_nm")).toHaveValue("500");
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704", { timeout: 180_000 });
    await expect(page.getByTestId("kpi-vld-value")).toContainText("2.598");
    const graph = page.getByTestId("panel-iv").locator(".js-plotly-plot");
    await plotted(graph);
    const maxCurrent = await graph.evaluate(el => Math.max(...(el as any).data.filter((tr: any) => tr.mode === "lines").flatMap((tr: any) => tr.y ?? []).filter(Number.isFinite)));
    const panel = page.getByTestId("panel-iv");
    const drag = await panel.locator(".nsewdrag").boundingBox();
    if (drag) { await page.mouse.move(drag.x + drag.width * .2, drag.y + drag.height * .2); await page.mouse.down(); await page.mouse.move(drag.x + drag.width * .75, drag.y + drag.height * .7, { steps: 8 }); await page.mouse.up(); }
    for (let i = 0; i < 2; i++) {
      await panel.getByRole("radio").nth(1).click();
      await expect.poll(() => graph.evaluate(el => (el as any)._fullLayout.yaxis.type)).toBe("linear");
      const range = await graph.evaluate(el => (el as any)._fullLayout.yaxis.range as number[]);
      expect(range[0]).toBeLessThanOrEqual(0); expect(range[1]).toBeGreaterThanOrEqual(maxCurrent); expect(range[1]).toBeLessThan(maxCurrent * 2);
      await panel.getByRole("radio").nth(0).click();
      await expect.poll(() => graph.evaluate(el => (el as any)._fullLayout.yaxis.type)).toBe("log");
    }
    await panel.getByRole("button", { name: "Reset axes", exact: true }).click();
    if (await page.locator(".plotly-notifier .notifier-close").count()) await page.locator(".plotly-notifier .notifier-close").first().click();
    await save(page, "Reference");
    await page.getByTestId("lang-toggle").click();
    await screenshot(page, "device-desktop");
    await page.setViewportSize({ width: 390, height: 844 });
    await screenshot(page, "device-mobile");
    await page.setViewportSize({ width: 1440, height: 900 });
    await geometry(page, "Lg_nm", "400");
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.610", { timeout: 180_000 });
    await expect(page.getByTestId("kpi-vld-value")).toContainText("2.556");
    await expect.poll(() => jobs.filter(j => j.body.kind === "branches").length).toBeGreaterThanOrEqual(2);
    writeFileSync(`${review}/live-branches.json`, JSON.stringify(jobs, null, 2));
  });

  test("CSVM live current plus drain capacitance produces measured cycle extrema and frequency", async ({ page }) => {
    const jobs = recordResponses(page);
    await fresh(page);
    await page.getByRole("radio", { name: "Current · CSVM", exact: true }).click();
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("csvm-status")).toContainText("startup excluded", { timeout: 200_000 });
    const frequencyText = await page.getByTestId("kpi-frequency-value").innerText();
    const frequency = parseFloat(frequencyText) * (frequencyText.includes("kHz") ? 1000 : 1);
    expect(frequency).toBeGreaterThan(850); expect(frequency).toBeLessThan(875);
    await expect(page.getByTestId("kpi-vtop-value")).toContainText("3.7");
    await expect(page.getByTestId("kpi-vbottom-value")).toContainText("2.5");
    await plotted(page.getByTestId("panel-csvm").locator(".js-plotly-plot"));
    await page.getByTestId("lang-toggle").click();
    await screenshot(page, "csvm-desktop");
    await page.setViewportSize({ width: 390, height: 844 });
    await screenshot(page, "csvm-mobile");
    expect(jobs.some(j => j.body.kind === "circuit")).toBe(true);
    writeFileSync(`${review}/live-csvm.json`, JSON.stringify(jobs, null, 2));
  });

  test("free placement and wiring yields the actual 2 V / 2 mA resistor solution", async ({ page }) => {
    const jobs = recordResponses(page);
    await fresh(page, "circuit", "?view=all");
    await page.getByTestId("menu-file").click(); await page.getByTestId("file-new").click();
    await place(page, "V", 0, 0); await place(page, "GND", 0, 40); await place(page, "R", 120, 0); await place(page, "GND", 120, 40);
    await page.getByTestId("tool-wire").click();
    let at = await world(page); await page.mouse.click(at(0, -40).x, at(0, -40).y); await page.mouse.click(at(120, -40).x, at(120, -40).y);
    await expect(page.locator("[data-testid=sch-svg] [data-wire]")).toHaveCount(1);
    await expect(page.getByTestId("erc-pill")).toHaveClass(/ok/);
    await page.getByTestId("tool-select").click(); at = await world(page); await page.mouse.click(at(0, 0).x, at(0, 0).y);
    await page.getByTestId("insp-wave-value").fill("2"); await page.getByTestId("insp-wave-value").press("Enter");
    await page.getByTestId("run-button").click();
    await plotted(page.getByTestId("panel-sch-waves").locator(".js-plotly-plot"));
    at = await world(page); await page.mouse.click(at(60, -40).x, at(60, -40).y); await page.mouse.click(at(120, 0).x, at(120, 0).y);
    await expect(page.locator("[data-ann-node=N001]")).toContainText("2.000 V");
    await expect(page.locator("[data-ann-el=R1]")).toContainText("2 mA");
    await screenshot(page, "circuit-drawn");
    expect(jobs.some(j => j.body.kind === "circuit")).toBe(true);
    writeFileSync(`${review}/live-drawn-circuit.json`, JSON.stringify(jobs, null, 2));
  });

  test("optional STL load-line example runs with real latch event", async ({ page }) => {
    const jobs = recordResponses(page);
    await fresh(page, "circuit");
    await page.getByTestId("menu-examples").click(); await page.getByTestId("tpl-load_line").click();
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("sch-summary")).toBeVisible({ timeout: 180_000 });
    await expect(page.getByTestId("kpi-sch-X1.n_latch_up-value")).toHaveText("1");
    await expect(page.getByTestId("kpi-sch-X1.vd_first_lu-value")).toContainText("3.70");
    await plotted(page.getByTestId("panel-sch-waves").locator(".js-plotly-plot"));
    await page.getByTestId("lang-toggle").click();
    await screenshot(page, "circuit-desktop");
    writeFileSync(`${review}/live-loadline.json`, JSON.stringify(jobs, null, 2));
  });
});

test("five saved device slots preserve geometry across reload and loading", async ({ page }) => {
  await fresh(page);
  for (let i = 1; i <= 5; i++) { await geometry(page, "Lg_nm", String(200 + 50 * i)); await save(page, `Device ${i}`); }
  await expect(page.getByTestId("dev-save-open")).toBeDisabled();
  await expect(page.getByTestId("device-shelf").locator(".device-slot")).toHaveCount(5);
  await page.reload();
  await page.getByTestId("device-slot-1").getByRole("button", { name: "Load", exact: true }).click();
  await expect(page.getByTestId("geometry-Lg_nm")).toHaveValue("250");
  await page.getByTestId("device-slot-5").getByRole("button", { name: "Load", exact: true }).click();
  await expect(page.getByTestId("geometry-Lg_nm")).toHaveValue("450");
});

test("Reference presents fixed measured and calibrated IDVD records", async ({ page }) => {
  await fresh(page, "validation");
  await expect(page.getByTestId("reference-fixed-label")).toBeVisible();
  await expect(page.getByTestId("reference-metrics")).toBeVisible();
  const graph = page.getByTestId("panel-val-iv").locator(".js-plotly-plot");
  await plotted(graph);
  const traces = await graph.evaluate(el => (el as any).data.map((tr: any) => ({ n: tr.x.length, first: tr.y.slice(0, 3) })));
  expect(traces.map((tr: any) => tr.n)).toEqual([401, 401, 2003, 2003]);
  expect(traces[0].first[1]).toBeCloseTo(3.95e-13, 16);
  await page.getByTestId("lang-toggle").click();
  await screenshot(page, "reference-desktop");
});

test("connection contract: lab authentication is memory-only and retires an earlier endpoint response", async ({ page }) => {
  // These responses test endpoint/auth/staleness contracts only. The curve is an existing
  // authentic recording; it is NOT counted as a live numerical simulation.
  const reference = JSON.parse(readFileSync("e2e/fixtures/geometry/reference.json", "utf8")).result;
  const remote = "http://lab.example.test/studio";
  let held: any = null;
  const auth: string[] = [];
  let passwordPosted = false;
  await page.route(url => /^\/(?:studio\/)?api\//.test(url.pathname), async route => {
    const request = route.request(); const url = request.url(); const isRemote = url.startsWith(remote);
    if (url.includes("/api/health")) return route.fulfill({ json: { ok: true, version: "transport-contract-only", workers: 1, auth_required: isRemote } });
    if (url.includes("/api/session")) {
      if (request.method() === "POST") { passwordPosted = request.postDataJSON().password === "test-only-password"; return route.fulfill({ json: { authenticated: true, token: "qa-memory-token", expires_in: 900 } }); }
      return route.fulfill({ json: { authenticated: false } });
    }
    if (isRemote) auth.push(request.headers().authorization ?? "");
    if (url.includes("/api/compute/branches")) {
      if (!isRemote) { held = route; return; }
      return route.fulfill({ json: { job_id: "remote-recorded-contract", kind: "branches", status: "done", progress: 1, result: reference } });
    }
    return route.fulfill({ json: {} });
  });
  await fresh(page);
  await page.getByTestId("run-button").click();
  await expect.poll(() => held !== null).toBe(true);
  await page.getByTestId("connection-button").click();
  await page.getByTestId("connection-lab").click();
  await page.getByTestId("connection-url").fill(remote + "/");
  await page.getByTestId("connection-password").fill("test-only-password");
  await page.getByTestId("connection-test").click();
  await expect(page.getByTestId("connection-status")).toContainText(/connected|online/i);
  expect(passwordPosted).toBe(true);
  await held.fulfill({ json: { job_id: "retired-recorded-contract", kind: "branches", status: "done", progress: 1, result: reference } });
  await page.getByTestId("connection-dialog").getByRole("button", { name: /close/i }).click();
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("—");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704");
  expect(auth.length).toBeGreaterThan(0);
  expect(auth.every(value => value === "Bearer qa-memory-token")).toBe(true);
  const storage = await page.evaluate(() => ({ local: { ...localStorage }, session: { ...sessionStorage } }));
  expect(storage.local["biristor-studio:connection"]).toBe(JSON.stringify({ version: 1, endpoint: remote }));
  expect(JSON.stringify(storage)).not.toContain("qa-memory-token");
  expect(JSON.stringify(storage)).not.toContain("test-only-password");
  await page.getByTestId("connection-button").click();
  await screenshot(page, "connection-dialog");
  await page.getByTestId("connection-local").click();
  await page.getByTestId("connection-test").click();
  await expect(page.getByTestId("connection-status")).toContainText(/connected|online/i);
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("biristor-studio:connection")!).endpoint)).toBe("");
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("—");
});

test("unavailable solver refuses unrecorded curves without a synthetic fallback", async ({ page }) => {
  await page.route(url => /^\/(?:studio\/)?api\//.test(url.pathname), route => route.abort("connectionrefused"));
  await page.route("**/snapshot/**", route => route.fulfill({ status: 404, body: "missing" }));
  await fresh(page);
  await geometry(page, "Lg_nm", "400");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("panel-iv")).toContainText(/compute server|connection|connect/i);
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("—");
  await expect(page.getByTestId("panel-iv").locator(".js-plotly-plot")).toHaveCount(0);
  await page.getByRole("radio", { name: "Current · CSVM", exact: true }).click();
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("panel-csvm")).toContainText(/compute server|connection|connect/i);
  await expect(page.getByTestId("kpi-frequency-value")).toContainText("—");
  await expect(page.getByTestId("panel-csvm").locator(".js-plotly-plot")).toHaveCount(0);
});

test("live remote: real cross-origin password-protected API computes the reference model", async ({ page }) => {
  test.skip(!live || !process.env.STUDIO_REMOTE_PASSWORD, "A second actual API with an ephemeral test password is required.");
  test.setTimeout(180_000);
  const remote = process.env.STUDIO_REMOTE_URL ?? "http://127.0.0.1:8001";
  const jobs = recordResponses(page);
  const remoteCompute: { origin: string; hasBearer: boolean }[] = [];
  page.on("request", request => {
    if (request.url().startsWith(remote + "/api/compute/")) remoteCompute.push({ origin: new URL(request.url()).origin, hasBearer: request.headers().authorization?.startsWith("Bearer ") ?? false });
  });
  await fresh(page);
  await page.getByTestId("connection-button").click();
  await page.getByTestId("connection-lab").click();
  await page.getByTestId("connection-url").fill(remote);
  await page.getByTestId("connection-password").fill(process.env.STUDIO_REMOTE_PASSWORD!);
  await page.getByTestId("connection-test").click();
  await expect(page.getByTestId("connection-status")).toContainText("Connected");
  await expect(page.getByTestId("connection-password")).toHaveValue("");
  await page.getByTestId("connection-dialog").getByRole("button", { name: /close/i }).click();
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.704", { timeout: 160_000 });
  await expect(page.getByTestId("kpi-vld-value")).toContainText("2.598");
  await plotted(page.getByTestId("panel-iv").locator(".js-plotly-plot"));
  expect(remoteCompute.length).toBeGreaterThan(0);
  expect(remoteCompute.every(request => request.hasBearer && request.origin === remote)).toBe(true);
  await expect.poll(() => jobs.some(job => job.url.startsWith(remote) && job.body.kind === "branches" && job.body.cached !== true)).toBe(true);
  const storage = await page.evaluate(() => JSON.stringify({ ...localStorage, ...sessionStorage }));
  expect(storage).not.toContain(process.env.STUDIO_REMOTE_PASSWORD!);
  mkdirSync(review, { recursive: true });
  writeFileSync(`${review}/live-remote.json`, JSON.stringify(jobs, null, 2));
});
