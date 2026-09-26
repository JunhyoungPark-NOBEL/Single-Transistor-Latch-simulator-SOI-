// Production frontend and live solver: the fit data below are synthetic solver points, not measurements.
import { expect, test, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { analyzeCsvm } from "../src/device/csvmMetrics";

const review = "review/simple-model";
async function capture(page: Page, name: string) {
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => { (document.activeElement as HTMLElement | null)?.blur(); window.scrollTo(0, 0); });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
  await page.screenshot({path: `${review}/${name}.png`, fullPage: true});
}

test("Simple selection, HRS fit, saved identity and actual CSVM", async ({ page, request }) => {
  test.skip(process.env.STUDIO_LIVE !== "1", "Requires the real Python API");
  test.setTimeout(120_000);
  page.setDefaultTimeout(10_000);
  mkdirSync(review, { recursive: true });
  expect((await request.get("/api/health")).ok()).toBe(true);
  const finished: any[] = [];
  const browserErrors: string[] = [];
  page.on("pageerror", e => browserErrors.push(e.message));
  page.on("response", async response => {
    if (/\/api\/(compute|jobs)\//.test(response.url()) && response.ok()) {
      try { const body = await response.json(); if (body.status === "done") finished.push(body); } catch { /* cancelled navigation */ }
    }
  });
  await page.addInitScript(() => {
    localStorage.clear();
    localStorage.setItem("stl-websim:v1", JSON.stringify({v:2,lang:"ko",theme:"light",mode:"deterministic",autoRun:false,autoRunChosen:true}));
  });
  await page.goto("/#tab=device&mode=deterministic");
  await expect(page.getByTestId("model-detailed")).toHaveAttribute("aria-checked", "true");
  await page.getByTestId("model-simple").click();
  await expect(page.getByTestId("model-simple")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("mode-stochastic")).toBeDisabled();
  await expect(page.getByTestId("field-vg").locator("input[type=text]")).toHaveValue("-3");
  await expect(page.getByTestId("model-controls")).toContainText("초기값 · HRS 보정 필요");
  await page.getByTestId("run-button").click();
  await expect.poll(() => finished.find(x => x.kind === "branches" && x.result?.model === "simple"), {timeout:180_000}).toBeTruthy();
  const branches = finished.find(x => x.kind === "branches" && x.result?.model === "simple");
  expect(branches.result.folds.V_LU).toBeGreaterThan(2);
  expect(branches.result.folds.V_LD).toBeLessThan(branches.result.folds.V_LU);
  await capture(page, "simple-idvd");
  // Synthetic HRS points are taken directly from a computed stable branch.
  const curve = branches.result.HRS;
  const points = [1.0,1.7,2.1,2.25].map(target => {
    let best = 0;
    for (let i=1;i<curve.vd.length;i++) if (Math.abs(curve.vd[i]-target)<Math.abs(curve.vd[best]-target)) best=i;
    return {vd_V:curve.vd[best],id_A:curve.id[best]};
  });
  await page.getByTestId("field-simple-is_ref_A").locator("input[type=text]").fill("0.08");
  await page.getByTestId("field-simple-is_ref_A").locator("input[type=text]").press("Enter");
  await page.getByTestId("simple-calibrate-open").click();
  await page.getByTestId("simple-hrs-points").fill("VD_V,ID_A\n"+points.map(p=>`${p.vd_V},${p.id_A}`).join("\n"));
  await page.getByTestId("simple-calibrate-run").click();
  await expect(page.getByTestId("simple-calibration-result")).toBeVisible({timeout:120_000});
  const fit = finished.find(x=>x.kind === "simple_calibrate");
  expect(fit.result.identifiable).toBe(true);
  expect(fit.result.rmse_log10).toBeLessThan(1e-4);
  expect(fit.result.device.simple.beta_ref).toBe(2.3);
  await capture(page, "simple-hrs-fit");
  await page.getByTestId("simple-calibrate-apply").click();
  await expect(page.getByTestId("simple-calibration-dialog")).toHaveCount(0);
  await page.getByTestId("dev-save-open").click();
  await page.getByTestId("dev-save-name").fill("Simple RC");
  await page.getByTestId("dev-save-submit").click();
  await expect(page.getByTestId("device-slot-1")).toContainText("Simple");
  const saved = await page.evaluate(()=>JSON.parse(localStorage.getItem("stl-websim:devices")!));
  expect(saved.devices[0].device.model).toBe("simple");
  expect(saved.devices[0].device.simple.beta_ref).toBe(2.3);
  await page.getByTestId("model-export-open").click();
  await expect(page.getByTestId("model-export-simple-info")).toBeVisible();
  await expect(page.getByTestId("model-export-download")).toBeDisabled();
  await page.getByTestId("model-export-dialog").getByTestId("modal-close").click();
  await page.getByTestId("device-forcing").getByRole("radio").nth(1).click();
  await page.getByTestId("run-button").click();
  await expect.poll(()=>finished.find(x=>x.kind==="circuit" && x.result?.elements?.some((e:any)=>e.device?.model==="simple")),{timeout:180_000}).toBeTruthy();
  const csvm = finished.find(x=>x.kind==="circuit" && x.result?.elements?.some((e:any)=>e.device?.model==="simple"));
  const metrics = analyzeCsvm(csvm.result);
  expect(metrics.status).toBe("oscillating");
  expect(metrics.frequency_Hz).toBeGreaterThan(0);
  const graph = page.getByTestId("panel-csvm").locator(".js-plotly-plot");
  await expect.poll(()=>graph.evaluate(el=>(el as any).data?.length)).toBe(2);
  const signals = csvm.result.runs[0].signals;
  const drawn = await graph.evaluate(el=>(el as any).data.map((t:any)=>t.y));
  expect(drawn[0]).toEqual(signals.find((s:any)=>s.key==="V(drain)").values);
  expect(drawn[1]).toEqual(signals.find((s:any)=>s.key==="X1.vb").values);
  await expect(page.getByTestId("kpi-frequency-value")).not.toHaveText("—");
  await page.getByTestId("model-controls").scrollIntoViewIfNeeded();
  await capture(page, "simple-csvm");
  console.log("step: switch back to Detailed");
  await page.getByTestId("model-detailed").click();
  console.log("step: check Detailed mode");
  await expect(page.getByTestId("mode-stochastic")).toBeEnabled();
  await expect(page.getByTestId("field-vg").locator("input[type=text]")).toHaveValue("-3");
  // The retained stale waveform still identifies its original Simple body-potential definition.
  console.log("step: check stale plot");
  const details = page.getByTestId("panel-csvm");
  await expect(details).toContainText("변경됨");
  console.log("step: reload saved Simple");
  await page.getByTestId("device-slot-1").getByRole("button", {name:"불러오기"}).click();
  await expect(page.getByTestId("model-simple")).toHaveAttribute("aria-checked", "true");
  console.log("step: mobile viewport");
  await page.setViewportSize({width:390,height:844});
  // The responsive layout closes the drawer on resize; its offscreen close button is still CSS-visible.
  await expect(page.getByTestId("sidebar")).not.toHaveClass(/\bopen\b/);
  console.log("step: mobile capture");
  await capture(page, "simple-csvm-mobile");
  expect(browserErrors).toEqual([]);
  writeFileSync(`${review}/live-simple-ui.json`,JSON.stringify({branches,fit,csvm,metrics,syntheticHrsPoints:points,savedDevice:saved.devices[0]},null,2));
});
