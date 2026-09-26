import { expect, test } from "@playwright/test";

test("voltage sweep keeps valid axes across repeated zoomed log/linear switches", async ({ page }) => {
  await page.addInitScript(() => localStorage.clear());
  await page.goto("/?mock=1#tab=device&mode=deterministic");
  const panel = page.getByTestId("panel-iv");
  const graph = panel.locator(".js-plotly-plot");
  await expect(graph).toBeVisible();
  await expect(panel.getByRole("radio")).toHaveCount(2);
  const original = await graph.evaluate(el => {
    const gd = el as unknown as { data: { name?: string; mode?: string; y?: number[] }[] };
    return gd.data.filter(tr => tr.mode === "lines").map(tr => ({ name: tr.name, y: tr.y }));
  });
  await page.screenshot({ path: "test-results/device-sweep.png", fullPage: true });
  expect(original).toHaveLength(2);
  expect(original.map(tr => tr.name).join(" ")).not.toMatch(/unstable|불안정|HRS|LRS/);
  const maxCurrent = Math.max(...original.flatMap(tr => tr.y ?? []).filter(v => typeof v === "number"));
  // Start from a real Plotly user zoom, not merely the initial programmatic range.
  const drag = await panel.locator(".nsewdrag").boundingBox();
  expect(drag).not.toBeNull();
  if (drag) {
    await page.mouse.move(drag.x + drag.width * 0.2, drag.y + drag.height * 0.2);
    await page.mouse.down();
    await page.mouse.move(drag.x + drag.width * 0.75, drag.y + drag.height * 0.7, { steps: 8 });
    await page.mouse.up();
  }
  for (let i = 0; i < 3; i++) {
    await panel.getByRole("radio").nth(1).click();
    await expect.poll(() => graph.evaluate(el => (el as unknown as { _fullLayout: { yaxis: { type: string } } })._fullLayout.yaxis.type)).toBe("linear");
    const linear = await graph.evaluate(el => (el as unknown as { _fullLayout: { yaxis: { range: number[] } } })._fullLayout.yaxis.range);
    expect(linear[1]).toBeGreaterThanOrEqual(maxCurrent);
    expect(linear[1]).toBeLessThan(maxCurrent * 2);
    expect(linear[0]).toBeLessThanOrEqual(0);
    await panel.getByRole("radio").nth(0).click();
    await expect.poll(() => graph.evaluate(el => (el as unknown as { _fullLayout: { yaxis: { type: string } } })._fullLayout.yaxis.type)).toBe("log");
    const log = await graph.evaluate(el => (el as unknown as { _fullLayout: { yaxis: { range: number[] } } })._fullLayout.yaxis.range);
    expect(log[1]).toBeGreaterThanOrEqual(Math.log10(maxCurrent));
    expect(log[1]).toBeLessThan(0);
    expect(log[0]).toBeLessThan(log[1]);
  }
});

test("reference page uses fixed authentic records even while simulations are in demo mode", async ({ page }) => {
  await page.goto("/?mock=1#tab=validation&mode=deterministic");
  await expect(page.getByTestId("reference-fixed-label")).toBeVisible();
  await expect(page.getByTestId("reference-metrics")).toBeVisible();
  const graph = page.getByTestId("panel-val-iv").locator(".js-plotly-plot");
  await expect(graph).toBeVisible();
  const result = await graph.evaluate(el => {
    const gd = el as unknown as { data: { mode: string; x: number[]; y: number[] }[] };
    return { counts: gd.data.map(tr => tr.x.length), firstMeasured: gd.data[0].y.slice(0, 3) };
  });
  expect(result.counts).toEqual([401, 401, 2003, 2003]);
  expect(result.firstMeasured[1]).toBeCloseTo(3.9500000000000003e-13, 16);
  await expect(page.getByTestId("validation-table")).toHaveCount(0);
  await page.screenshot({ path: "test-results/reference-benchmark.png", fullPage: true });
});
