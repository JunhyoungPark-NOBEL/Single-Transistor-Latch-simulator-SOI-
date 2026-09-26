import { expect, test, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";

const review = "review/performance";
async function prepare(page: Page) {
  await page.addInitScript(() => {
    localStorage.clear();
    localStorage.setItem("stl-websim:v1", JSON.stringify({v:2,lang:"ko",theme:"light",mode:"deterministic",autoRun:false,autoRunChosen:true}));
  });
}
async function capture(page: Page, name: string) {
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => { (document.activeElement as HTMLElement | null)?.blur(); window.scrollTo(0, 0); });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
  await page.screenshot({path:`${review}/${name}.png`,fullPage:true});
}

test("live performance catalog and compact estimate, desktop and phone", async ({ page }) => {
  test.skip(process.env.STUDIO_LIVE !== "1", "Requires live Python API");
  mkdirSync(review,{recursive:true});
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  await prepare(page);
  await page.goto("/#tab=device");
  await expect(page.getByTestId("run-estimate")).toContainText(/기준 환경|이 PC|저장 결과/);
  await page.getByTestId("tab-performance").click();
  await expect(page.getByTestId("performance-page")).toContainText("기준 환경 측정");
  await expect(page.getByTestId("performance-page").locator(".perf-table tbody")).not.toHaveCount(0);
  await expect(page.getByTestId("performance-page")).toContainText("미지원");
  await expect(page.getByTestId("performance-calibrate")).toBeEnabled();
  await capture(page,"performance-desktop");
  await page.getByRole("button",{name:"전체",exact:true}).click();
  await expect(page.locator(".perf-table")).toContainText("Detailed + Simple");
  await expect(page.locator(".perf-table")).toContainText("공통");
  await page.getByRole("button",{name:"주요 해석",exact:true}).click();
  await page.setViewportSize({width:390,height:844});
  await capture(page,"performance-mobile");
  expect(errors).toEqual([]);
});

test("actual host calibration, reactive CSVM estimate and cached reuse", async ({ page }) => {
  test.skip(process.env.STUDIO_LIVE !== "1" || process.env.STUDIO_COMPUTE !== "1", "Requires live compute checks enabled");
  test.setTimeout(180_000);
  mkdirSync(review,{recursive:true});
  const estimates: any[] = [];
  const finished: any[] = [];
  const estimateInputs: any[] = [];
  page.on("request", r => { if (r.url().endsWith("/api/performance/estimate")) estimateInputs.push(r.postDataJSON()); });
  page.on("response", async r => { if (!r.ok()) return; try {
    if (r.url().endsWith("/api/performance/estimate")) estimates.push(await r.json());
    if (/\/api\/(compute|jobs)\//.test(r.url())) { const body = await r.json(); if(body.status==="done") finished.push(body); }
  } catch { /* navigation cancelled */ } });
  await prepare(page);
  await page.goto("/#tab=performance");
  await page.getByTestId("performance-calibrate").click();
  await expect(page.getByTestId("performance-page")).toContainText("측정 완료",{timeout:160_000});
  await page.locator(".perf-host-samples summary").click();
  await expect(page.locator(".perf-host-samples tbody tr")).toHaveCount(4);
  await capture(page,"performance-calibrated");
  await page.getByTestId("tab-device").click();
  await page.getByTestId("model-simple").click();
  await expect(page.getByTestId("run-estimate")).toContainText("이 PC");
  await page.getByTestId("device-forcing").getByRole("radio").nth(1).click();
  await expect.poll(()=>estimates.at(-1)?.items?.[0]?.family).toBe("circuit_csvm");
  await expect.poll(()=>estimates.at(-1)?.source).toBe("calibrated");
  await page.getByTestId("csvm-duration_s").fill("7.5");
  await page.getByTestId("csvm-duration_s").press("Enter");
  await expect.poll(()=>estimateInputs.at(-1)?.jobs?.[0]?.payload?.tran?.t_stop_s).toBe(.0075);
  await expect(page.getByTestId("run-estimate")).toContainText(/이 PC · 예상 [0-9]/);
  await capture(page,"performance-csvm-before-run");
  await page.getByTestId("run-button").click();
  await expect.poll(()=>finished.find(x=>x.kind==="circuit"),{timeout:80_000}).toBeTruthy();
  await expect(page.getByTestId("run-estimate")).toContainText("저장 결과 사용");
  await capture(page,"performance-csvm-estimate");
  await page.getByTestId("tab-circuit").click();
  await page.getByTestId("menu-examples").click();
  await page.getByTestId("tpl-oscillator").click();
  await expect.poll(()=>estimateInputs.at(-1)?.jobs?.[0]?.key).toBe("schematic");
  await expect(page.getByTestId("run-estimate")).toContainText(/(?:기준 환경|이 PC) · 예상 [0-9]/,{timeout:20_000});
  expect(estimateInputs.at(-1).jobs[0].payload.bench).toBe("custom");
  await capture(page,"performance-circuit-before-run");
  writeFileSync(`${review}/live-performance.json`, JSON.stringify({estimates,estimateInputs,finished},null,2));
});
