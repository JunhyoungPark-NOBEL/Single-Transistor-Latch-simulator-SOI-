// The 간단히 (default) layout of the shell, the Circuit tab, the schematic results and the Validation tab
// (mock mode, no backend needed): 84 px chrome with the demo state announced once, single-language tabs, the
// mode toggle only where it applies, compact circuit / schematic / validation screens, the ?view=all escape
// hatch, and the phone layout. Writes the canonical screenshots listed in the UX spec (§6).
import { expect, test, type Page } from "@playwright/test";

const SHOTS = "e2e/screenshots";

async function fresh(page: Page, hash = "#tab=device&mode=deterministic", query = "") {
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
  await page.goto(`/?mock=1${query}${hash}`);
  await expect(page.getByTestId("tab-device")).toBeVisible();
}

/** Plotly charts that are rendered (non-zero box) inside the main column. */
const plotsInMain = (page: Page) =>
  page.evaluate(() => [...document.querySelectorAll("main .js-plotly-plot")].filter((el) => el.getBoundingClientRect().height > 0).length);

const noHorizontalScroll = (page: Page) => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth);

async function runBench(page: Page, bench = "load_line") {
  await page.getByTestId("circuit-view-benches").click();
  await page.getByTestId(`bench-${bench}`).click();
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("circuit-summary")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByTestId("panel-waves").locator(".js-plotly-plot")).toBeVisible();
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
}

test.describe("shell: header and context strip", () => {
  test("84 px chrome, one demo announcement, opaque strip with the credits", async ({ page }) => {
    await fresh(page);
    const header = (await page.locator("header.header").boundingBox())!;
    const strip = (await page.getByTestId("modebar").boundingBox())!;
    expect(header.height + strip.height).toBeLessThanOrEqual(86);
    expect(strip.y).toBeGreaterThanOrEqual(header.y + header.height - 1);
    // no full-width banner: the demo state is the pill in the strip, said once
    await expect(page.locator(".banner")).toHaveCount(0);
    const pill = page.getByTestId("offline-banner");
    await expect(pill).toHaveCount(1);
    await expect(pill).toBeVisible();
    await expect(pill).toContainText("데모 데이터");
    expect(await page.getByTestId("modebar").locator("[data-testid=offline-banner]").count()).toBe(1);
    // the pill explains itself on click (no hover-only UI) and closes with Esc
    await pill.getByRole("button").first().click();
    await expect(pill.getByRole("dialog")).toContainText("?mock=1");
    await page.keyboard.press("Escape");
    await expect(pill.getByRole("dialog")).toHaveCount(0);
    // the strip is opaque and holds the credits (full attribution at ≥ 1280 px)
    const bg = await page.getByTestId("modebar").evaluate((el) => getComputedStyle(el).backgroundColor);
    expect(bg).toMatch(/^rgb\(/);
    const credits = page.getByTestId("credits-chip");
    await expect(page.getByTestId("modebar").getByTestId("credits-chip")).toBeVisible();
    await expect(credits).toContainText("NOBEL");
    await expect(credits).toContainText("최양규");
    await expect(credits).toContainText("박준형");
    await expect(credits).toContainText("KAIST");
    // the status is a dot: its text is for screen readers only
    await expect(page.getByTestId("backend-status")).toContainText("mock");
    const statusBox = (await page.getByTestId("backend-status").boundingBox())!;
    expect(statusBox.width).toBeLessThanOrEqual(30);
  });

  test("single-language tabs; the mode toggle only on Device and Circuit; plain mode hints", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("tab-device")).toHaveText("소자");
    await expect(page.getByTestId("tab-physics")).toHaveText("물리 모델");
    await expect(page.getByTestId("mode-deterministic")).toHaveText("결정론");
    await expect(page.getByTestId("mode-stochastic")).toHaveText("확률");
    const hint = page.getByTestId("mode-hint");
    await expect(hint).toContainText("V_LU");
    await page.getByTestId("mode-stochastic").click();
    await expect(hint).toContainText("MC");
    await page.getByTestId("tab-circuit").click();
    await expect(page.getByTestId("mode-toggle")).toBeVisible();
    await expect(hint).toContainText("Eq. 2");
    await expect(hint).toContainText("Q_B");
    await page.getByTestId("mode-deterministic").click();
    await expect(hint).toContainText("MNA");
    await expect(hint).toContainText("BE");
    await expect(hint).not.toContainText("fold");
    await page.getByTestId("tab-validation").click();
    await expect(page.getByTestId("mode-toggle")).toHaveCount(0);
    await expect(hint).toContainText("측정");
    await page.getByTestId("tab-physics").click();
    await expect(page.getByTestId("mode-toggle")).toHaveCount(0);
    // English: one word per tab and per mode
    await page.getByTestId("tab-device").click();
    await page.getByTestId("lang-toggle").click();
    await expect(page.getByTestId("tab-device")).toHaveText("Device");
    await expect(page.getByTestId("tab-validation")).toHaveText("Validation");
    await expect(page.getByTestId("mode-deterministic")).toHaveText("Deterministic");
    await expect(page.getByTestId("offline-banner")).toContainText("Demo data");
    await expect(page.getByTestId("run-button")).toContainText("Run");
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.70", { timeout: 15_000 }).catch(async () => {
      await page.getByTestId("run-button").click();
      await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.70", { timeout: 15_000 });
    });
    await page.waitForTimeout(600);
    await page.screenshot({ path: `${SHOTS}/en-deterministic.png` });
  });

  test("below 1280 px the credits shorten to KAIST · NOBEL", async ({ page }) => {
    await page.setViewportSize({ width: 1200, height: 800 });
    await fresh(page);
    const credits = page.getByTestId("credits-chip");
    await expect(credits).toBeVisible();
    await expect(credits).toHaveText(/KAIST · NOBEL 연구실/, { useInnerText: true });
    await expect(credits).not.toContainText("박준형", { useInnerText: true });
    await credits.click();
    await expect(page.getByTestId("about")).toContainText("박준형");
  });
});

test.describe("circuit: quick benches", () => {
  test("load-line: ≤ 4 summary cells, schematic thumbnail, 2-subplot waveform with solver footnote, 2 plots", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=deterministic");
    await page.getByTestId("circuit-view-benches").click();
    await expect(page.getByTestId("schematic")).toBeVisible();
    const chip = (await page.getByTestId("bench-load_line").boundingBox())!;
    expect(chip.height).toBeLessThanOrEqual(48);
    await runBench(page);
    const cells = page.getByTestId("circuit-summary").locator(".sum-cell");
    expect(await cells.count()).toBeLessThanOrEqual(4);
    await expect(page.getByTestId("kpi-c-V_LU")).toBeVisible();
    await expect(page.getByTestId("kpi-c-V_LU-value")).toContainText("3.70");
    await expect(page.getByTestId("schematic")).toBeVisible();
    await expect(page.getByTestId("panel-waves").locator(".cartesianlayer .subplot")).toHaveCount(2);
    await expect(page.getByTestId("solver-stats")).toBeVisible();
    await expect(page.getByTestId("panel-trajectory").locator(".js-plotly-plot")).toBeVisible();
    expect(await plotsInMain(page)).toBe(2);
    // the other views are one click away
    await expect(page.getByTestId("more-tab-c-events")).toBeVisible();
    await page.getByTestId("more-tab-c-events").click();
    await expect(page.getByTestId("panel-c-events")).toBeVisible();
    await page.getByTestId("more-tab-trajectory").click();
    // extra signals from "신호 ▾"
    await page.getByTestId("signals-menu").click();
    await page.getByTestId("signal-q_b").click();
    await expect(page.getByTestId("panel-waves").locator(".cartesianlayer .subplot")).toHaveCount(3);
    await page.getByTestId("signal-q_b").click();
    await page.keyboard.press("Escape");
    // the thumbnail opens the full-size schematic
    await page.getByTestId("schematic-enlarge").click();
    await expect(page.getByTestId("schematic-full")).toBeVisible();
    await page.keyboard.press("Escape");
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${SHOTS}/circuit.png` });
  });
});

test.describe("circuit: schematic editor and results", () => {
  test("toolbar ≤ 15 controls; load-line results ≤ 4 KPIs; netlist folded but in the DOM", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=deterministic");
    await expect(page.getByTestId("schematic-view")).toBeVisible();
    const controls = page.locator("[data-testid=sch-toolbar] > .sch-tgroup > button, [data-testid=sch-toolbar] > .sch-tgroup > select, [data-testid=sch-toolbar] > .sch-menu > button");
    expect(await controls.count()).toBeLessThanOrEqual(15);
    const bar = (await page.getByTestId("sch-toolbar").boundingBox())!;
    expect(bar.height).toBeLessThan(60); // one row at 1440 px
    // edit actions are in "편집 ▾"; the shortcuts still work
    await expect(page.getByTestId("tool-delete")).toHaveCount(0);
    await page.getByTestId("menu-edit").click();
    await expect(page.getByTestId("tool-delete")).toBeVisible();
    await expect(page.getByTestId("tool-fit")).toBeVisible();
    await page.getByTestId("tool-fit").click();
    await loadExample(page, "load_line");
    await expect(page.getByTestId("netlist-toggle")).toBeVisible();
    await expect(page.getByTestId("netlist-text")).toBeHidden();
    await expect(page.getByTestId("netlist-text")).toContainText("Rs src d 1k");
    await runSchematic(page);
    const kpis = page.getByTestId("sch-summary").locator(".sum-cell");
    expect(await kpis.count()).toBeLessThanOrEqual(4);
    await expect(page.getByTestId("kpi-sch-X1.n_latch_up-value")).toHaveText("1");
    await expect(page.getByTestId("kpi-sch-X1.vd_first_lu")).toBeVisible();
    await expect(page.getByTestId("panel-sch-waves").locator(".js-plotly-plot")).toBeVisible();
    // one STL: the events table is folded (still in the DOM)
    await expect(page.getByTestId("sch-events-toggle")).toBeVisible();
    await expect(page.getByTestId("sch-events-table")).toContainText("X1");
    await expect(page.getByTestId("sch-events-table")).toBeHidden();
    await expect(page.getByTestId("more-tab-sch-traj")).toHaveAttribute("aria-selected", "true");
    expect(await plotsInMain(page)).toBe(2);
    await page.getByTestId("netlist-toggle").click();
    await expect(page.getByTestId("netlist-text")).toBeVisible();
    await page.getByTestId("netlist-toggle").click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${SHOTS}/schematic-run.png`, fullPage: true });
  });

  test("empty canvas offers the examples; a drawn resistor lands in the folded netlist", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=deterministic");
    await page.getByTestId("menu-file").click();
    await page.getByTestId("file-new").click();
    for (const id of ["load_line", "pulse", "pbit", "oscillator"]) await expect(page.getByTestId(`empty-tpl-${id}`)).toBeVisible();
    await page.getByTestId("sch-canvas").focus();
    await page.keyboard.press("r");
    await expect(page.getByTestId("empty-tpl-load_line")).toHaveCount(0); // the centre stays placeable while a tool is active
    const box = (await page.getByTestId("sch-svg").boundingBox())!;
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    await expect(page.locator("[data-testid=sch-svg] [data-kind=R]")).toHaveCount(1);
    await expect(page.getByTestId("netlist-text")).toContainText("R1");
    await page.keyboard.press("Escape");
    await page.keyboard.press("Control+z");
    await expect(page.getByTestId("empty-tpl-oscillator")).toBeVisible();
    await page.getByTestId("empty-tpl-oscillator").click();
    await expect(page.locator("[data-testid=sch-svg] [data-kind=I]")).toHaveCount(1);
  });

  test("oscillator: the sawtooth waveform is the hero (V(out) over I(X1.d))", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=deterministic");
    await loadExample(page, "oscillator");
    await runSchematic(page);
    await expect(page.getByTestId("trace-V(out)")).toBeVisible();
    await expect(page.getByTestId("trace-I(X1.d)")).toBeVisible();
    await expect(page.getByTestId("panel-sch-waves").locator(".cartesianlayer .subplot")).toHaveCount(2);
    const h = (await page.getByTestId("panel-sch-waves").locator(".js-plotly-plot").boundingBox())!.height;
    expect(h).toBeGreaterThanOrEqual(400);
    expect(Number(await page.getByTestId("kpi-sch-X1.n_latch_up-value").textContent())).toBeGreaterThan(1);
  });

  test("p-bit example (stochastic) opens the comparator tab; statistics one click away", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=stochastic");
    await loadExample(page, "pbit");
    await expect(page.getByTestId("sch-sto")).toBeVisible();
    await page.getByTestId("sto-runs").fill("4");
    await page.getByTestId("sto-runs").press("Enter");
    await runSchematic(page);
    await expect(page.getByTestId("more-tab-sch-cmp-CMP1")).toHaveAttribute("aria-selected", "true");
    await expect(page.getByTestId("panel-sch-cmp-CMP1").locator(".js-plotly-plot")).toBeVisible();
    await expect(page.getByTestId("sch-cmp-stats-CMP1")).toContainText("P(");
    expect(await page.getByTestId("sch-summary").locator(".sum-cell").count()).toBeLessThanOrEqual(4);
    await page.getByTestId("more-tab-sch-dist").click();
    await expect(page.getByTestId("sch-stats-table")).toBeVisible();
    // the run overlay / mean ± SD options moved into ⋯
    await page.getByTestId("panel-menu-sch-waves").click();
    await expect(page.getByTestId("toggle-band")).toBeVisible();
    await page.keyboard.press("Escape");
    await page.getByTestId("more-tab-sch-cmp-CMP1").click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${SHOTS}/schematic-stochastic.png`, fullPage: true });
  });
});

test.describe("validation", () => {
  test("after 빠른 검증: n / n 통과, a 4-column table, one figure", async ({ page }) => {
    await fresh(page, "#tab=validation&mode=deterministic");
    await expect(page.getByTestId("mode-toggle")).toHaveCount(0);
    await page.getByTestId("val-fast").click();
    await expect(page.getByTestId("validation-table")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("val-status")).toHaveText(/(\d+) \/ \1 통과/);
    await expect(page.getByTestId("validation-table").locator("thead th")).toHaveCount(4);
    await expect(page.getByTestId("validation-table").locator(".pass.yes").first()).toBeVisible();
    await expect(page.getByTestId("panel-val-iv").locator(".js-plotly-plot")).toBeVisible({ timeout: 15_000 });
    expect(await plotsInMain(page)).toBe(1);
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${SHOTS}/validation.png` });
    // every number one click away
    await page.getByTestId("val-cols-all").click();
    await expect(page.getByTestId("validation-table").locator("thead th")).toHaveCount(7);
    await page.getByTestId("more-tab-val-photo").click();
    await expect(page.getByTestId("val-photo-run")).toBeVisible();
  });
});

test.describe("모두 보기 (?view=all) restores the full layouts", () => {
  test("circuit benches, schematic results and validation show every panel", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=deterministic", "&view=all");
    await expect(page.getByTestId("layout-all")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("tool-delete")).toBeVisible();
    await expect(page.getByTestId("netlist-text")).toBeVisible();
    await expect(page.getByTestId("sch-sim-adv")).toBeVisible();
    await expect(page.getByTestId("sim-dtmax")).toBeVisible();
    await runBench(page);
    await expect(page.locator(".bench-card")).toHaveCount(4);
    for (const id of ["panel-schematic", "panel-waves", "panel-trajectory", "panel-c-events"]) await expect(page.getByTestId(id)).toBeVisible();
    await expect(page.getByTestId("summary-more")).toHaveCount(0);
    await page.getByTestId("tab-validation").click();
    await page.getByTestId("val-fast").click();
    await expect(page.getByTestId("validation-table").locator("thead th")).toHaveCount(7, { timeout: 15_000 });
    for (const id of ["panel-val-iv", "panel-val-photo", "panel-val-vg"]) await expect(page.getByTestId(id)).toBeVisible();
    // switching back to 간단히 keeps working
    await page.getByTestId("layout-simple").click();
    await expect(page.getByTestId("more-validation")).toBeVisible();
  });
});

test.describe("phone (390 × 844)", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("two-row header, mode toggle in the strip, no horizontal scroll", async ({ page }) => {
    await fresh(page);
    const header = (await page.locator("header.header").boundingBox())!;
    const tabs = (await page.locator("nav.tabs").boundingBox())!;
    expect(tabs.y).toBeGreaterThan(header.y + 30); // second row
    await expect(page.getByRole("button", { name: /파라미터 패널/ })).toContainText("파라미터");
    await expect(page.getByTestId("modebar").getByTestId("mode-toggle")).toBeVisible();
    await expect(page.getByTestId("modebar").getByTestId("credits-chip")).toBeVisible();
    await expect(page.getByTestId("offline-banner")).toBeVisible();
    expect(await noHorizontalScroll(page)).toBe(true);
    await page.getByTestId("tab-circuit").click();
    await expect(page.getByTestId("schematic-view")).toBeVisible();
    expect(await noHorizontalScroll(page)).toBe(true);
    await page.getByTestId("circuit-view-benches").click();
    await expect(page.getByTestId("bench-picker")).toBeVisible();
    expect(await noHorizontalScroll(page)).toBe(true);
    await page.getByTestId("tab-validation").click();
    await page.getByTestId("val-fast").click();
    await expect(page.getByTestId("validation-table")).toBeVisible({ timeout: 15_000 });
    expect(await noHorizontalScroll(page)).toBe(true);
    await page.getByTestId("lang-toggle").click();
    await page.getByTestId("tab-device").click();
    expect(await noHorizontalScroll(page)).toBe(true);
  });
});
