// User-facing workspace shell: compact chrome, creator disclosure, device shelf,
// free-form circuit editor, reference benchmark, and accessible responsive navigation.
import { expect, test, type Page, type TestInfo } from "@playwright/test";

async function fresh(page: Page, hash = "#tab=device&mode=deterministic", query = "") {
  await page.addInitScript(() => {
    if (!sessionStorage.getItem("e2e-init")) {
      localStorage.clear();
      sessionStorage.setItem("e2e-init", "1");
    }
  });
  await page.goto(`/?mock=1${query}${hash}`);
  await expect(page.getByTestId("tab-device")).toBeVisible();
}

const noHorizontalScroll = (page: Page) => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth);
async function screenshot(page: Page, testInfo: TestInfo, name: string) {
  await page.evaluate(() => document.fonts.ready);
  if (await page.getByTestId("panel-iv").count()) {
    await expect(page.getByTestId("panel-iv").locator(".js-plotly-plot")).toBeVisible();
  }
  await page.screenshot({ path: testInfo.outputPath(name), fullPage: true });
}
async function loadExample(page: Page, id: string) {
  await page.getByTestId("menu-examples").click();
  await page.getByTestId(`tpl-${id}`).click();
}
async function runSchematic(page: Page) {
  await page.getByTestId("run-button").click();
  const confirm = page.getByTestId("sch-confirm-run");
  if (await confirm.isVisible({ timeout: 1500 }).catch(() => false)) await confirm.click();
  await expect(page.getByTestId("sch-summary")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("panel-sch-waves").locator(".js-plotly-plot")).toBeVisible();
}

test.describe("compact workspace shell", () => {
  test("compact chrome discloses lab and institution only when creator is opened", async ({ page }) => {
    await fresh(page);
    const header = (await page.locator("header.header").boundingBox())!;
    const strip = (await page.getByTestId("modebar").boundingBox())!;
    expect(header.height + strip.height).toBeLessThanOrEqual(86);
    expect(strip.y).toBeGreaterThanOrEqual(header.y + header.height - 1);
    await expect(page.locator(".banner")).toHaveCount(0);
    const pill = page.getByTestId("offline-banner");
    await expect(pill).toHaveCount(1);
    await expect(pill).toContainText("데모 데이터");
    await pill.getByRole("button").first().click();
    await expect(pill.getByRole("dialog")).toContainText("?mock=1");
    await page.keyboard.press("Escape");
    await expect(pill.getByRole("dialog")).toHaveCount(0);
    await expect(pill.getByRole("button").first()).toBeFocused();

    const credits = page.getByTestId("credits-chip");
    await expect(credits).toHaveText("제작자", { useInnerText: true });
    await expect(credits).toHaveAttribute("aria-haspopup", "dialog");
    await expect(credits).not.toContainText(/NOBEL|KAIST|최양규|박준형/);
    await credits.focus();
    await credits.press("Enter");
    const about = page.getByTestId("about");
    await expect(about).toBeVisible();
    await expect(about.getByRole("heading")).toBeFocused();
    await expect(about).toContainText("NOBEL 연구실");
    await expect(about).toContainText("KAIST");
    await expect(about).not.toContainText(/최양규|Choi|박준형|Jun.?Hyoung|advisor/i);
    await page.keyboard.press("Escape");
    await expect(about).toHaveCount(0);
    await expect(credits).toBeFocused();
    await expect(credits).toHaveAttribute("aria-expanded", "false");
    const bg = await page.getByTestId("modebar").evaluate((el) => getComputedStyle(el).backgroundColor);
    expect(bg).toMatch(/^rgb\(/);
    expect((await page.getByTestId("backend-status").boundingBox())!.width).toBeLessThanOrEqual(30);
  });

  test("navigation and mode controls support keyboard, concise labels, languages, and themes", async ({ page }, testInfo) => {
    await fresh(page);
    for (const [id, label] of [["device", "소자"], ["circuit", "회로"], ["validation", "레퍼런스"], ["physics", "문서"]]) {
      await expect(page.getByTestId(`tab-${id}`)).toHaveText(label);
    }
    const device = page.getByTestId("tab-device");
    await device.focus();
    await device.press("ArrowRight");
    await expect(page.getByTestId("tab-circuit")).toBeFocused();
    await expect(page.getByTestId("tab-circuit")).toHaveAttribute("aria-selected", "true");
    await expect(page.getByTestId("main-circuit")).toHaveAttribute("aria-labelledby", "tab-circuit");
    await expect(page.getByTestId("mode-hint")).toHaveText("회로");
    await expect(page.getByTestId("mode-hint")).not.toContainText(/MNA|BE|Eq/);
    await page.getByTestId("mode-deterministic").focus();
    await page.getByTestId("mode-deterministic").press("ArrowRight");
    await expect(page.getByTestId("mode-stochastic")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("mode-stochastic")).toBeFocused();
    for (const id of ["validation", "physics"]) {
      await page.getByTestId(`tab-${id}`).click();
      await expect(page.getByTestId("mode-toggle")).toHaveCount(0);
    }
    await page.getByTestId("lang-toggle").click();
    await expect(page.getByTestId("tab-validation")).toHaveText("Reference");
    await expect(page.getByTestId("tab-physics")).toHaveText("Docs");
    await expect(page.getByTestId("credits-chip")).toHaveText("Credits", { useInnerText: true });
    await device.click();
    await page.getByTestId("mode-deterministic").click();
    await expect(page.getByTestId("run-button")).toContainText("Simulate");
    await page.getByTestId("theme-toggle").click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem("stl-websim:v1") ?? "{}").theme)).toBe("dark");
    await page.reload();
    await expect(page.getByTestId("tab-device")).toHaveText("Device");
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await expect(page.getByTestId("panel-iv").locator(".js-plotly-plot")).toBeVisible();
    expect(await noHorizontalScroll(page)).toBe(true);
    await screenshot(page, testInfo, "english-dark-device.png");
  });

  test("compact parameters sit left of the sweep with five visible device slots on the right", async ({ page }, testInfo) => {
    await fresh(page);
    const sidebar = page.getByTestId("sidebar");
    const sweep = page.getByTestId("panel-iv");
    const shelf = page.getByTestId("device-shelf");
    await expect(sweep).toBeVisible();
    await expect(sidebar.locator(".guide-inline, .gi-text")).toHaveCount(0);
    await expect(shelf).toHaveAccessibleName("사용자 소자");
    await expect(shelf.getByTestId("dev-save-open")).toBeVisible();
    await expect(shelf.locator("[data-testid^=device-slot-]")).toHaveCount(5);
    for (let i = 1; i <= 5; i++) await expect(shelf.getByTestId(`device-slot-${i}`)).toBeVisible();
    const [sb, chart, saved] = await Promise.all([sidebar.boundingBox(), sweep.boundingBox(), shelf.boundingBox()]);
    expect(sb!.x + sb!.width).toBeLessThanOrEqual(chart!.x + 1);
    expect(saved!.x).toBeGreaterThan(chart!.x + chart!.width);
    expect(saved!.y).toBeLessThanOrEqual(chart!.y + 30);
    await expect(shelf).toContainText("0 / 5");
    await page.getByTestId("dev-save-open").click();
    await expect(page.getByTestId("save-device-dialog")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("save-device-dialog")).toHaveCount(0);
    await screenshot(page, testInfo, "compact-device-workspace.png");
  });
});

test.describe("circuit workspace", () => {
  test("free-form canvas is primary, basic components are visible, examples are a menu", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=deterministic");
    await expect(page.getByTestId("schematic-view")).toBeVisible();
    await expect(page.getByTestId("circuit-view-benches")).toHaveCount(0);
    await expect(page.getByTestId("bench-picker")).toHaveCount(0);
    for (const kind of ["R", "C", "V", "I", "STL", "MOS", "D", "BJT", "GND"]) {
      const tool = page.getByTestId(`tool-${kind}`);
      await expect(tool).toBeVisible();
      await expect(tool).toHaveAccessibleName(/.+/);
    }
    await expect(page.getByTestId("menu-examples")).toBeVisible();
    await page.getByTestId("menu-file").click();
    await page.getByTestId("file-new").click();
    await expect(page.locator("[data-testid^=empty-tpl-]")).toHaveCount(0);
    await page.getByTestId("sch-canvas").focus();
    await page.keyboard.press("r");
    const box = (await page.getByTestId("sch-svg").boundingBox())!;
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    await expect(page.locator("[data-testid=sch-svg] [data-kind=R]")).toHaveCount(1);
    await expect(page.getByTestId("netlist-text")).toContainText("R1");
    await page.keyboard.press("Escape");
    await page.keyboard.press("Control+z");
    await expect(page.locator("[data-testid=sch-svg] [data-kind=R]")).toHaveCount(0);
    await loadExample(page, "load_line");
    await expect(page.locator("[data-testid=sch-svg] [data-kind=STL]")).toHaveCount(1);
    await expect(page.getByTestId("netlist-text")).toContainText("Rs src d 1k");
    await expect(page.getByTestId("netlist-text")).toBeHidden();
    await page.getByTestId("netlist-toggle").click();
    await expect(page.getByTestId("netlist-text")).toBeVisible();
    expect(await noHorizontalScroll(page)).toBe(true);
  });

  test("a loaded circuit runs and exposes readable waveform and event controls", async ({ page }, testInfo) => {
    await fresh(page, "#tab=circuit&mode=deterministic");
    await loadExample(page, "load_line");
    await runSchematic(page);
    expect(await page.getByTestId("sch-summary").locator(".sum-cell").count()).toBeLessThanOrEqual(4);
    await expect(page.getByTestId("kpi-sch-X1.n_latch_up-value")).toHaveText("1");
    await expect(page.getByTestId("sch-events-table")).toBeHidden();
    await page.getByTestId("sch-events-toggle").click();
    await expect(page.getByTestId("sch-events-table")).toBeVisible();
    await expect(page.getByTestId("sch-events-table")).toContainText("X1");
    await expect(page.getByTestId("trace-bar")).toBeVisible();
    await screenshot(page, testInfo, "circuit-results.png");
  });

  test("show-all keeps detailed editor controls without restoring quick benches", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=deterministic", "&view=all");
    await expect(page.locator("html")).toHaveAttribute("data-layout", "all");
    await expect(page.getByTestId("tool-delete")).toBeVisible();
    await expect(page.getByTestId("netlist-text")).toBeVisible();
    await expect(page.getByTestId("sim-dtmax")).toBeVisible();
    await expect(page.getByTestId("circuit-view-benches")).toHaveCount(0);
    await page.getByTestId("tab-device").click();
    await page.getByTestId("layout-simple").click();
    await expect(page.getByTestId("layout-simple")).toHaveAttribute("aria-checked", "true");
    await page.getByTestId("tab-circuit").click();
    await expect(page.getByTestId("menu-edit")).toBeVisible();
    await expect(page.getByTestId("netlist-text")).toBeHidden();
  });
});

test("Reference opens directly to ID–VD benchmarking and a two-direction accuracy table", async ({ page }, testInfo) => {
  await fresh(page, "#tab=validation&mode=deterministic");
  await expect(page.getByTestId("mode-toggle")).toHaveCount(0);
  await expect(page.getByTestId("val-fast")).toHaveCount(0);
  await expect(page.getByTestId("reference-fixed-label")).toContainText("기준 데이터");
  await expect(page.getByTestId("panel-val-iv").locator(".js-plotly-plot")).toBeVisible();
  const metrics = page.getByTestId("reference-metrics");
  await expect(metrics.locator("thead th")).toHaveCount(4);
  await expect(metrics.locator("tbody tr")).toHaveCount(2);
  await expect(metrics).toContainText("RMSE");
  await expect(metrics.locator("details")).not.toHaveAttribute("open", "");
  await metrics.locator("summary").click();
  await expect(metrics).toContainText("모델 보정에 사용한 기록");
  expect(await page.locator("main .js-plotly-plot").count()).toBe(1);
  expect(await noHorizontalScroll(page)).toBe(true);
  await screenshot(page, testInfo, "reference-benchmark.png");
});

test.describe("phone shell (390 × 844)", () => {
  test.use({ viewport: { width: 390, height: 844 } });
  test("two-row navigation, creator disclosure and stacked device shelf fit the viewport", async ({ page }, testInfo) => {
    await fresh(page);
    const header = (await page.locator("header.header").boundingBox())!;
    const tabs = (await page.locator("nav.tabs").boundingBox())!;
    expect(tabs.y).toBeGreaterThan(header.y + 30);
    const drawer = page.getByRole("button", { name: /파라미터 패널/ });
    await expect(drawer).toContainText("파라미터");
    await expect(page.getByTestId("modebar").getByTestId("mode-toggle")).toBeVisible();
    const credits = page.getByTestId("credits-chip");
    await expect(credits).toHaveText("제작자", { useInnerText: true });
    await credits.click();
    const about = page.getByTestId("about");
    await expect(about).toContainText("KAIST");
    const aboutBox = (await about.boundingBox())!;
    expect(aboutBox.x).toBeGreaterThanOrEqual(0);
    expect(aboutBox.x + aboutBox.width).toBeLessThanOrEqual(391);
    await page.keyboard.press("Escape");
    await drawer.click();
    await expect(drawer).toHaveAttribute("aria-expanded", "true");
    await expect(page.getByTestId("sidebar")).toBeInViewport();
    await expect(page.getByTestId("sidebar").locator(".guide-inline")).toHaveCount(0);
    await page.getByTestId("sidebar").locator(".drawer-head button").click();
    await expect(drawer).toHaveAttribute("aria-expanded", "false");
    const shelf = (await page.getByTestId("device-shelf").boundingBox())!;
    expect(shelf.x).toBeGreaterThanOrEqual(0);
    expect(shelf.x + shelf.width).toBeLessThanOrEqual(390);
    expect(await noHorizontalScroll(page)).toBe(true);
    await screenshot(page, testInfo, "mobile-device-workspace.png");
    await page.getByTestId("tab-circuit").click();
    await expect(page.getByTestId("schematic-view")).toBeVisible();
    await expect(page.getByTestId("menu-examples")).toBeVisible();
    expect(await noHorizontalScroll(page)).toBe(true);
    await page.getByTestId("tab-validation").click();
    await expect(page.getByTestId("reference-metrics")).toBeVisible();
    expect(await noHorizontalScroll(page)).toBe(true);
    await page.getByTestId("lang-toggle").click();
    await page.getByTestId("tab-device").click();
    await expect(page.getByTestId("tab-validation")).toHaveText("Reference");
    expect(await noHorizontalScroll(page)).toBe(true);
  });
});
