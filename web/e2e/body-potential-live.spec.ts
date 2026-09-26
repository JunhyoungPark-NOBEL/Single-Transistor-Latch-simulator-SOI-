// Actual production frontend + Python solver; no numerical response interception.
import { expect, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { analyzeCsvm } from "../src/device/csvmMetrics";

test("CSVM retains the complete drain waveform and plots actual electrostatic body potential", async ({ page }) => {
  test.skip(process.env.STUDIO_LIVE !== "1", "Requires the actual Python API.");
  test.setTimeout(240_000);
  const review = "review/body-terminals";
  mkdirSync(review, { recursive: true });
  const completed: any[] = [];
  page.on("response", async response => {
    if (/\/api\/(?:compute|jobs)\//.test(response.url()) && response.ok()) {
      try { const body = await response.json(); if (body.status === "done" && body.kind === "circuit") completed.push(body); } catch { /* Aborted navigation. */ }
    }
  });
  await page.addInitScript(() => {
    localStorage.clear();
    localStorage.setItem("stl-websim:v1", JSON.stringify({ v: 2, lang: "ko", theme: "light", mode: "deterministic", autoRun: false, autoRunChosen: true }));
  });
  await page.goto("/#tab=device&mode=deterministic");
  await page.getByTestId("device-forcing").getByRole("radio").nth(1).click();
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("kpi-frequency-value")).not.toHaveText("—", { timeout: 200_000 });
  await expect.poll(() => completed.length, { timeout: 200_000 }).toBeGreaterThan(0);
  const result = completed.at(-1).result;
  const run = result.runs[0];
  const body = run.signals.find((s: any) => s.key === "X1.vb");
  const drain = run.signals.find((s: any) => s.key === "V(drain)");
  expect(body).toBeTruthy();
  expect(body.values.filter(Number.isFinite).length).toBeGreaterThan(100);
  expect(Math.max(...body.values) - Math.min(...body.values)).toBeGreaterThan(0.05);
  const graph = page.getByTestId("panel-csvm").locator(".js-plotly-plot");
  await expect.poll(() => graph.evaluate(el => (el as any).data?.length)).toBe(2);
  const drawn = await graph.evaluate(el => (el as any).data.map((trace: any) => ({ x: trace.x, y: trace.y, name: trace.name })));
  expect(drawn[0].y).toEqual(drain.values);
  expect(drawn[1].y).toEqual(body.values);
  expect(drawn[0].x).toEqual(drawn[1].x);
  expect(drawn[1].name).toContain("<i>V</i><sub>B</sub>");
  expect(await graph.evaluate(el => (el as any)._fullLayout.xaxis.anchor)).toBe("y2");
  const capture = async (name: string) => {
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(() => { (document.activeElement as HTMLElement | null)?.blur(); window.scrollTo(0, 0); });
    for (const close of await page.locator(".plotly-notifier .notifier-close").all()) await close.click();
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
    await page.screenshot({ path: `${review}/${name}.png`, fullPage: true });
  };
  await capture("csvm-body-light");
  await page.getByTestId("theme-toggle").click();
  await capture("csvm-body-dark");
  await page.getByTestId("theme-toggle").click();
  await page.setViewportSize({ width: 390, height: 844 });
  await capture("csvm-body-mobile");
  writeFileSync(`${review}/live-csvm-body.json`, JSON.stringify(completed.at(-1), null, 2));
});

test("real 1 fF startup overshoot remains visible while later cycles produce Vtop, Vbottom and frequency", async ({ page }) => {
  test.skip(process.env.STUDIO_LIVE !== "1", "Requires the actual Python API.");
  test.setTimeout(240_000);
  const review = "review/body-terminals";
  mkdirSync(review, { recursive: true });
  const completed: any[] = [];
  page.on("response", async response => {
    if (/\/api\/(?:compute|jobs)\//.test(response.url()) && response.ok()) {
      try { const body = await response.json(); if (body.status === "done" && body.kind === "circuit") completed.push(body); } catch { /* Navigation. */ }
    }
  });
  await page.addInitScript(() => {
    localStorage.clear();
    localStorage.setItem("stl-websim:v1", JSON.stringify({ v: 2, lang: "ko", theme: "light", mode: "deterministic", autoRun: false, autoRunChosen: true }));
  });
  await page.goto("/#tab=device&mode=deterministic");
  await page.getByTestId("device-forcing").getByRole("radio").nth(1).click();
  await page.getByTestId("csvm-capacitance_F").fill("0.001"); // pF = 1 fF
  await page.getByTestId("csvm-capacitance_F").press("Enter");
  await page.getByTestId("csvm-duration_s").fill("0.04"); // ms = 40 us
  await page.getByTestId("csvm-duration_s").press("Enter");
  await page.getByTestId("run-button").click();
  await expect.poll(() => completed.length, { timeout: 200_000 }).toBeGreaterThan(0);
  const result = completed.at(-1).result;
  const metrics = analyzeCsvm(result);
  expect(metrics.status).toBe("oscillating");
  expect(metrics.cycles).toBeGreaterThanOrEqual(2);
  expect(metrics.vTop_V).toBeGreaterThan(3);
  expect(metrics.vTop_V).toBeLessThan(4.2);
  expect(metrics.vBottom_V).not.toBeNull();
  expect(metrics.frequency_Hz).toBeGreaterThan(100_000);
  const drain = result.runs[0].signals.find((s: any) => s.key === "V(drain)");
  const firstOvershoot = Math.max(...drain.values.filter(Number.isFinite));
  expect(firstOvershoot).toBeGreaterThan(metrics.vTop_V! + 1);
  const graph = page.getByTestId("panel-csvm").locator(".js-plotly-plot");
  await expect.poll(() => graph.evaluate(el => Math.max(...(el as any).data[0].y))).toBe(firstOvershoot);
  await expect(page.getByTestId("kpi-vtop-value")).toContainText(metrics.vTop_V!.toFixed(3));
  await expect(page.getByTestId("kpi-vbottom-value")).toContainText(metrics.vBottom_V!.toFixed(3));
  await expect(page.getByTestId("kpi-frequency-value")).not.toHaveText("—");
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => { (document.activeElement as HTMLElement | null)?.blur(); window.scrollTo(0, 0); });
  await page.screenshot({ path: `${review}/csvm-startup-overshoot.png`, fullPage: true });
  writeFileSync(`${review}/live-csvm-overshoot.json`, JSON.stringify({ response: completed.at(-1), metrics, firstOvershoot }, null, 2));
});
