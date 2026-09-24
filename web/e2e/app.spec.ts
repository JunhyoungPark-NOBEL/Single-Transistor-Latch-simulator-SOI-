// Mock-mode end-to-end checks (no backend needed): rendering, mode toggle, Details window (KaTeX, drag,
// Esc, focus return), language toggle, presets, circuit and physics tabs, screenshots.
import { expect, test, type Page } from "@playwright/test";

const SHOTS = "e2e/screenshots";

async function fresh(page: Page, hash = "#tab=device&mode=deterministic") {
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
  await page.goto(`/?mock=1${hash}`);
  await expect(page.getByTestId("mode-toggle")).toBeVisible();
}

test.describe("STL simulator (mock mode)", () => {
  test("renders the shell with demo banner, tabs, mode toggle and grouped parameters", async ({ page }) => {
    await fresh(page);
    await expect(page.getByText("STL Simulator").first()).toBeVisible();
    await expect(page.getByTestId("offline-banner")).toBeVisible();
    for (const t of ["device", "circuit", "validation", "physics"]) await expect(page.getByTestId(`tab-${t}`)).toBeVisible();
    await expect(page.getByTestId("group-bias")).toBeVisible();
    await expect(page.getByTestId("group-light")).toBeVisible();
    await expect(page.getByTestId("group-calib")).toBeVisible();
    await expect(page.getByTestId("panels-deterministic")).toBeVisible();
    // KaTeX symbols in field labels
    await expect(page.getByTestId("field-vg").locator(".katex").first()).toBeVisible();
  });

  test("deterministic run shows folds in the KPI strip", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.70", { timeout: 15_000 });
    await expect(page.getByTestId("kpi-vld-value")).toContainText("2.59");
    await expect(page.getByTestId("panel-iv").locator(".js-plotly-plot")).toBeVisible();
    await page.screenshot({ path: `${SHOTS}/desktop-deterministic.png` });
  });

  test("mode toggle switches visible groups and panels", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("group-stoch")).toHaveCount(0);
    await expect(page.getByTestId("group-local")).toHaveCount(0);
    await page.getByTestId("mode-stochastic").click();
    await expect(page.getByTestId("mode-stochastic")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("group-stoch")).toBeVisible();
    await expect(page.getByTestId("group-local")).toBeVisible();
    await expect(page.getByTestId("panels-stochastic")).toBeVisible();
    await expect(page.getByTestId("panel-hazard")).toBeVisible();
    await expect(page.getByTestId("panel-iv")).toHaveCount(0);
    await expect(page).toHaveURL(/mode=stochastic/);
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 15_000 });
    await expect(page.getByTestId("stats-table")).toBeVisible();
    await expect(page.getByTestId("panel-hazard").locator(".js-plotly-plot")).toBeVisible({ timeout: 15_000 });
    await page.waitForTimeout(600);
    await page.screenshot({ path: `${SHOTS}/desktop-stochastic.png` });
    // experimental action point shows the uncalibrated warning
    await page.getByTestId("field-ls_action").locator("select").selectOption("junction");
    await expect(page.getByTestId("uncalibrated-warning")).toBeVisible();
    await page.getByTestId("mode-deterministic").click();
    await expect(page.getByTestId("group-stoch")).toHaveCount(0);
  });

  test("Details window: opens next to the button with KaTeX, drags, navigates, closes with Esc", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.70", { timeout: 15_000 });
    const btn = page.getByTestId("details-group-bias");
    await btn.click();
    const win = page.getByTestId("physics-window");
    await expect(win).toBeVisible();
    await expect(win.getByRole("heading", { level: 2 })).toBeFocused();
    await expect(win.locator(".katex").first()).toBeVisible();
    // the page is still usable (non-modal): no page navigation, tabs still there
    await expect(page.getByTestId("main-device")).toBeVisible();
    const box = await win.boundingBox();
    const bbox = await btn.boundingBox();
    expect(box && bbox).toBeTruthy();
    // opened next to the button (to its right) and inside the viewport
    expect(box!.x).toBeGreaterThan(bbox!.x);
    expect(box!.x + box!.width).toBeLessThanOrEqual(1440);
    await page.screenshot({ path: `${SHOTS}/details-window.png` });
    // drag by the header
    const head = page.getByTestId("physics-window-header");
    const hb = (await head.boundingBox())!;
    await page.mouse.move(hb.x + 60, hb.y + 20);
    await page.mouse.down();
    await page.mouse.move(hb.x + 60 - 150, hb.y + 20 + 60, { steps: 6 });
    await page.mouse.up();
    const box2 = (await win.boundingBox())!;
    // sub-pixel layout rounding: allow ±2 px
    expect(Math.abs(box2.x - (box!.x - 150))).toBeLessThanOrEqual(2);
    expect(Math.abs(box2.y - (box!.y + 60))).toBeLessThanOrEqual(2);
    // related topic replaces the content, back restores it
    const title1 = await win.getByRole("heading", { level: 2 }).textContent();
    const related = win.locator("[data-testid^=related-]").first();
    if (await related.count()) {
      await related.click();
      await expect(win.getByRole("heading", { level: 2 })).not.toHaveText(title1 ?? "");
      await page.getByTestId("physics-back").click();
      await expect(win.getByRole("heading", { level: 2 })).toHaveText(title1 ?? "");
    }
    // Esc closes and focus returns to the trigger
    await page.keyboard.press("Escape");
    await expect(win).toHaveCount(0);
    await expect(btn).toBeFocused();
  });

  test("language toggle switches UI strings", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("tab-device")).toContainText("소자");
    await expect(page.getByTestId("run-button")).toContainText("결정론");
    await page.getByTestId("lang-toggle").click();
    await expect(page.getByTestId("tab-device")).toHaveText("Device");
    await expect(page.getByTestId("run-button")).toContainText("Run deterministic");
    await expect(page.getByTestId("group-bias")).toContainText("Bias & sweep");
    await page.getByTestId("lang-toggle").click();
    await expect(page.getByTestId("tab-device")).toContainText("소자");
  });

  test("editing a value marks the preset as modified; reset restores it", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("preset-label")).not.toContainText("수정");
    const input = page.getByTestId("field-vg").locator("input.input");
    await input.fill("-1.8");
    await input.press("Enter");
    await expect(page.getByTestId("preset-label")).toContainText("기준 보정에서 수정");
    await expect(page.getByTestId("field-vg").locator(".field-changed")).toBeVisible();
    // out-of-range input shows a validation message and is not committed
    await input.fill("-9");
    await expect(page.getByTestId("field-vg").locator(".field-err")).toBeVisible();
    await input.press("Escape");
    await page.getByTestId("group-bias").getByRole("button", { name: /초기화|Reset/ }).click();
    await expect(page.getByTestId("preset-label")).not.toContainText("수정");
    // photo preset loads its defaults (power mode, 1200 V/s)
    await page.getByTestId("preset-photo").click();
    await expect(page.getByTestId("field-vg").locator("input.input")).toHaveValue("-1.8");
    await expect(page.getByTestId("light-conversion")).toContainText("pA/mW");
  });

  test("circuit tab: bench cards, generic result rendering", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=deterministic");
    await page.getByTestId("circuit-view-benches").click(); // the circuit tab opens on the schematic editor
    await expect(page.getByTestId("bench-picker")).toBeVisible();
    await expect(page.getByTestId("schematic")).toBeVisible();
    await expect(page.getByTestId("group-bench")).toBeVisible();
    await expect(page.getByTestId("group-solver")).toBeVisible();
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("circuit-summary")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("panel-waves").locator(".js-plotly-plot")).toBeVisible();
    await expect(page.getByTestId("solver-stats")).toBeVisible();
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${SHOTS}/circuit.png` });
    await page.getByTestId("bench-pbit").click();
    await expect(page.getByTestId("bench-pbit")).toHaveAttribute("aria-checked", "true");
  });

  test("physics tab lists all topics with a working search", async ({ page }) => {
    await fresh(page, "#tab=physics&mode=deterministic");
    await expect(page.getByTestId("physics-tab")).toBeVisible();
    await expect(page.getByTestId("topic-overview")).toBeVisible();
    const before = await page.locator("[data-testid^=topic-]").count();
    expect(before).toBe(18);
    await page.getByTestId("physics-search").fill("hazard");
    await expect.poll(async () => page.locator("[data-testid^=topic-]").count()).toBeLessThan(before);
  });

  test("validation tab runs checks", async ({ page }) => {
    await fresh(page, "#tab=validation&mode=deterministic");
    await page.getByTestId("val-fast").click();
    await expect(page.getByTestId("validation-table")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("validation-table").locator(".pass.yes").first()).toBeVisible();
  });

  test("dark theme", async ({ page }) => {
    await fresh(page, "#tab=device&mode=stochastic");
    await page.getByTestId("theme-toggle").click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("±", { timeout: 15_000 });
    await expect(page.getByTestId("panel-hazard").locator(".js-plotly-plot")).toBeVisible({ timeout: 15_000 });
    await page.waitForTimeout(600);
    await page.screenshot({ path: `${SHOTS}/dark-stochastic.png` });
  });

  test("narrow screen: sidebar becomes a drawer; empty panels still offer Run", async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await fresh(page);
    const toggle = page.getByRole("button", { name: /파라미터 패널|Parameter panel/ });
    await expect(toggle).toBeVisible();
    await page.screenshot({ path: `${SHOTS}/tablet-1024.png` });
    await page.getByTestId("empty-run-iv").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.70", { timeout: 15_000 });
  });
});

test.describe("UX regressions", () => {
  // alpha of a computed colour: rgba(r, g, b, a) or modern syntax such as color(srgb r g b / a)
  const alpha = (c: string) => {
    if (c === "transparent") return 0;
    const slash = c.match(/\/\s*([\d.]+%?)\s*\)/);
    if (slash) return slash[1].endsWith("%") ? Number(slash[1].slice(0, -1)) / 100 : Number(slash[1]);
    const m = c.match(/rgba?\(([^)]+)\)/);
    if (!m) return 1;
    const parts = m[1].split(",").map((x) => x.trim());
    return parts.length === 4 ? Number(parts[3]) : 1;
  };

  test("sticky header and mode strip are opaque and stay above scrolled plots", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("panel-iv").locator(".js-plotly-plot")).toBeVisible({ timeout: 15_000 });
    for (const sel of ["header.header", '[data-testid="modebar"]']) {
      const bg = await page.locator(sel).evaluate((el) => getComputedStyle(el).backgroundColor);
      expect(alpha(bg), `${sel} background ${bg}`).toBe(1);
    }
    // Plotly's own z-indices (modebar 1001) are contained in the plot's stacking context
    const iso = await page.locator(".plot").first().evaluate((el) => getComputedStyle(el).isolation);
    expect(iso).toBe("isolate");
    // scroll the I–V legend under the mode strip: the strip still wins the hit test
    const strip = (await page.getByTestId("modebar").boundingBox())!;
    const legend = (await page.getByTestId("panel-iv").locator(".legend").boundingBox())!;
    await page.mouse.wheel(0, legend.y - strip.y - 4);
    await page.waitForTimeout(300);
    const hit = await page.evaluate(({ x, y }) => !!document.elementFromPoint(x, y)?.closest('[data-testid="modebar"]'), { x: legend.x + 20, y: strip.y + strip.height / 2 });
    expect(hit).toBe(true);
  });

  test("Details window: display equations are never clipped (fit or scroll in their own box)", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("details-panel-charge-balance").click();
    const win = page.getByTestId("physics-window");
    await expect(win.locator(".katex").first()).toBeVisible();
    const check = () =>
      win.locator(".eq-tex").evaluateAll((els) =>
        els.map((e) => ({ fit: (e as HTMLElement).dataset.fit, sw: e.scrollWidth, cw: e.clientWidth, ox: getComputedStyle(e).overflowX, tab: (e as HTMLElement).tabIndex })),
      );
    await expect.poll(async () => (await check()).filter((x) => x.fit !== "scroll" && x.sw > x.cw + 1).length).toBe(0);
    for (const x of await check()) {
      expect(x.ox).toBe("auto");
      if (x.fit === "scroll") expect(x.tab).toBe(0);
    }
    // a narrow window forces the scroll path: still reachable, never cut off
    await page.setViewportSize({ width: 420, height: 800 });
    await page.getByTestId("physics-close").click();
    await page.getByTestId("details-panel-charge-balance").click();
    await expect.poll(async () => (await check()).filter((x) => x.fit !== "scroll" && x.sw > x.cw + 1).length).toBe(0);
    expect((await check()).some((x) => x.fit !== "fits")).toBe(true);
  });

  test("mode banner describes the circuit meaning of each mode on the Circuit tab", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=deterministic");
    await page.getByTestId("circuit-view-benches").click(); // the circuit tab opens on the schematic editor
    const hint = page.getByTestId("mode-hint");
    await expect(hint).toContainText("MNA");
    await expect(hint).toContainText("BE");
    await expect(hint).not.toContainText("branch와 fold");
    await page.getByTestId("field-method").getByRole("radio", { name: "TRAP" }).click();
    await expect(hint).toContainText("TRAP");
    await page.getByTestId("mode-stochastic").click();
    await expect(hint).toContainText("Eq. 2");
    await expect(hint).toContainText("Q_B");
    await page.getByTestId("tab-device").click();
    await expect(hint).toContainText("MC");
  });

  test("rise/fall edges default to auto (server default) on the pulse bench", async ({ page }) => {
    await fresh(page, "#tab=circuit&mode=deterministic");
    await page.getByTestId("circuit-view-benches").click(); // the circuit tab opens on the schematic editor
    await page.getByTestId("bench-pulse").click();
    await expect(page.getByTestId("auto-bench_rise_s")).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("auto-bench_fall_s")).toHaveAttribute("aria-pressed", "true");
    const rise = page.getByTestId("field-bench_rise_s").locator("input.input");
    await rise.fill("5");
    await rise.press("Enter");
    await expect(page.getByTestId("auto-bench_rise_s")).toHaveAttribute("aria-pressed", "false");
  });

  test("l_GIDL help describes the effective GIDL field length", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("group-calib").locator(".group-toggle").click();
    await page.getByTestId("tip-l_gidl").hover();
    await expect(page.getByRole("tooltip")).toContainText("E_G");
    await expect(page.getByRole("tooltip")).not.toContainText("BTBT 영역 길이");
  });

  test("run bar reports only runs of the current tab/mode; KPIs dim when parameters change", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("run-button").click();
    await expect(page.getByTestId("kpi-vlu-value")).toContainText("3.70", { timeout: 15_000 });
    await expect(page.getByTestId("run-status")).toContainText("완료");
    await page.getByTestId("mode-stochastic").click();
    await expect(page.getByTestId("run-status")).toHaveText("대기");
    await page.getByTestId("mode-deterministic").click();
    await expect(page.getByTestId("run-status")).toContainText("완료");
    const input = page.getByTestId("field-vg").locator("input.input");
    await input.fill("-1.9");
    await input.press("Enter");
    await expect(page.getByTestId("kpis")).toHaveClass(/stale/);
    await expect(page.getByTestId("panel-components").locator(".badge.stale")).toBeVisible();
  });

  test("credits corner opens the About card (lab, advisor, developer, version) and never says “paper”", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("tech-chip")).toHaveText("FDSOI");
    const chip = page.getByTestId("credits-chip");
    await expect(chip).toBeVisible();
    await chip.click();
    const about = page.getByTestId("about");
    await expect(about).toBeVisible();
    await expect(about).toContainText("NOBEL");
    await expect(about).toContainText("최양규");
    await expect(about).toContainText("박준형");
    await expect(about).toContainText("KAIST");
    await expect(about).toContainText("500 nm");
    await page.keyboard.press("Escape");
    await expect(about).toHaveCount(0);
    await expect(chip).toBeFocused();
    await page.getByTestId("lang-toggle").click();
    await chip.click();
    await expect(about).toContainText("Prof. Yang-Kyu Choi");
    await expect(about).toContainText("Junhyoung Park");
    // the credits must not cover the Run bar or the panels: they live in the sticky mode strip
    const box = await chip.boundingBox();
    const strip = await page.getByTestId("modebar").boundingBox();
    expect(box && strip && box.y >= strip.y && box.y + box.height <= strip.y + strip.height).toBe(true);
    await expect(page.locator("body")).not.toContainText(/\bpaper\b|논문/i);
  });

  for (const [name, raw] of [
    ["null", "null"],
    ["not JSON", "{oops"],
    ["wrong types", JSON.stringify({ tab: "zzz", mode: "foo", lang: "fr", theme: 7, preset: "weird", params: { device: "x", circuit: { bench: "nope", bench_params: 5 } } })],
  ] as const) {
    test(`junk persisted state (${name}) falls back to defaults`, async ({ page }) => {
      await page.addInitScript((v) => {
        try {
          if (!sessionStorage.getItem("e2e-junk")) {
            localStorage.clear();
            localStorage.setItem("stl-websim:v1", v);
            localStorage.setItem("stl-websim:groups", "null");
            sessionStorage.setItem("e2e-junk", "1");
          }
        } catch {
          /* ignore */
        }
      }, raw);
      await page.goto("/?mock=1");
      await expect(page.getByTestId("mode-toggle")).toBeVisible();
      await expect(page.getByTestId("tab-device")).toContainText("소자");
      await expect(page.getByTestId("tab-device")).toHaveAttribute("aria-selected", "true");
      await expect(page.getByTestId("group-bias")).toBeVisible();
      await page.getByTestId("tab-circuit").click();
      await page.getByTestId("circuit-view-benches").click(); // the circuit tab opens on the schematic editor
      await expect(page.getByTestId("bench-load_line")).toHaveAttribute("aria-checked", "true");
    });
  }
});
