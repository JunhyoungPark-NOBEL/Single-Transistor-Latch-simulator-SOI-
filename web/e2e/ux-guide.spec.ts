// Compact parameter controls and click-only design guides, including full documentation,
// keyboard access, changed calibration parameters, both languages, and mobile layout.
import { expect, test, type Locator, type Page } from "@playwright/test";

async function fresh(page: Page, query = "") {
  await page.addInitScript(() => {
    if (!sessionStorage.getItem("e2e-init")) {
      localStorage.clear();
      sessionStorage.setItem("e2e-init", "1");
    }
  });
  await page.goto(`/?mock=1${query}#tab=device&mode=deterministic`);
  await expect(page.getByTestId("tab-device")).toBeVisible();
  await expect(page.getByTestId("sidebar")).toBeAttached();
}

const settle = (loc: Locator) => loc.evaluate((el) => Promise.all(el.getAnimations({ subtree: true }).map((a) => a.finished)).then(() => undefined));

/** WCAG contrast of an element's text against its composited background. */
async function contrastOf(loc: Locator): Promise<number> {
  return loc.first().evaluate((el) => {
    const parse = (c: string): [number, number, number, number] => {
      const m = c.match(/rgba?\(([^)]+)\)/);
      if (m) {
        const p = m[1].split(/[\s,/]+/).filter(Boolean).map(Number);
        return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1];
      }
      const s = c.match(/color\(srgb ([\d.]+) ([\d.]+) ([\d.]+)(?: \/ ([\d.]+))?\)/);
      if (s) return [Number(s[1]) * 255, Number(s[2]) * 255, Number(s[3]) * 255, s[4] ? Number(s[4]) : 1];
      return [0, 0, 0, 0];
    };
    const layers: [number, number, number, number][] = [];
    for (let n: Element | null = el; n; n = n.parentElement) {
      const bg = parse(getComputedStyle(n).backgroundColor);
      if (bg[3] > 0) layers.push(bg);
      if (bg[3] >= 1) break;
    }
    let base: [number, number, number] = [255, 255, 255];
    const last = layers[layers.length - 1];
    if (!last || last[3] < 1) base = document.documentElement.dataset.theme === "dark" ? [14, 16, 21] : [244, 245, 247];
    for (const l of layers.reverse()) base = [0, 1, 2].map((i) => l[i] * l[3] + base[i] * (1 - l[3])) as [number, number, number];
    const fg = parse(getComputedStyle(el).color);
    const lum = (c: number[]) => {
      const [r, g, b] = c.map((v) => {
        const x = v / 255;
        return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const fgc = [0, 1, 2].map((i) => fg[i] * fg[3] + base[i] * (1 - fg[3]));
    const a = lum(fgc);
    const b = lum(base);
    return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  });
}

test.describe("compact parameter guides", () => {
  test("sidebar contains controls without inline explanations; extra fields remain accessible", async ({ page }) => {
    await fresh(page);
    const sidebar = page.getByTestId("sidebar");
    await expect(sidebar.locator(".guide-inline, .gi-text, .fa-names")).toHaveCount(0);
    for (const key of ["vg", "vd_max", "iph_pA"]) {
      await expect(page.getByTestId(`field-${key}`)).toBeVisible();
      await expect(page.getByTestId(`tip-${key}`)).toHaveAccessibleName(/설계 가이드/);
    }
    const extra = page.getByTestId("field-adv-bias");
    await expect(extra).toHaveText("고급 항목 2개");
    await expect(page.getByTestId("field-rate")).toHaveCount(0);
    await extra.click();
    for (const key of ["rate", "dv"]) await expect(page.getByTestId(`field-${key}`)).toBeVisible();
    await expect(sidebar.locator(".guide-inline")).toHaveCount(0);
    await page.getByTestId("light-mode").getByRole("radio").nth(1).click();
    await expect(page.getByTestId("field-power_mW")).toBeVisible();
    await page.getByTestId("field-adv-light").click();
    await expect(page.getByTestId("field-resp")).toBeVisible();
    await page.getByTestId("mode-stochastic").click();
    for (const key of ["rate", "n_cycles", "ls_mode", "ls_sigma"]) await expect(page.getByTestId(`tip-${key}`)).toBeVisible();
    await expect(sidebar.locator(".guide-inline")).toHaveCount(0);
  });

  test("info button opens a short guide on click; keyboard, outside click, and documentation work", async ({ page }) => {
    await fresh(page);
    const trigger = page.getByTestId("tip-vg");
    await trigger.hover();
    await page.waitForTimeout(400);
    await expect(page.getByTestId("guide-pop")).toHaveCount(0);
    await trigger.focus();
    await expect(page.getByTestId("guide-pop")).toHaveCount(0);
    await trigger.press("Enter");
    const pop = page.getByTestId("guide-pop");
    await expect(pop).toBeVisible();
    await expect(pop).toHaveAttribute("role", "dialog");
    await expect(pop.locator(".gp-intuitive")).toHaveText("게이트는 드레인 가장자리 '정공 수도꼭지'(GIDL)의 손잡이입니다.");
    await expect(pop.locator(".gchip.lu")).toHaveText("VLU ↑ 80 mV");
    await expect(pop.locator(".gchip.ld")).toHaveText("VLD → 그대로");
    await expect(pop.locator(".gchips-step")).toHaveText("(+0.1 V)");
    await expect(pop).toContainText("기준 소자에서의 변화");
    await expect(pop.locator(".gp-tech, .gp-caveat, .gp-effects, .gp-basis")).toHaveCount(0);
    await settle(pop);
    const box = (await pop.boundingBox())!;
    expect(box.height).toBeLessThan(270);
    expect(box.width).toBeLessThanOrEqual(321);
    await page.keyboard.press("Escape");
    await expect(pop).toHaveCount(0);
    await expect(trigger).toBeFocused();
    await trigger.press("Space");
    await expect(pop).toBeVisible();
    // The floating card can cover the next row; keyboard activation still replaces it.
    await page.getByTestId("tip-vd_max").focus();
    await page.getByTestId("tip-vd_max").press("Enter");
    await expect(pop).toHaveCount(1);
    await expect(pop).toHaveAccessibleName(/스윕 최대/);
    await page.mouse.click(900, 500);
    await expect(pop).toHaveCount(0);
    await trigger.click();
    await pop.getByTestId("guide-pop-more").click();
    const win = page.getByTestId("physics-window");
    await expect(win).toBeVisible();
    await expect(pop).toHaveCount(0);
    await expect(win.getByTestId("pw-guide-row-vg")).toHaveClass(/hl/);
    // The full explanation and caveat remain available in the document.
    await expect(win.getByTestId("pw-guide-row-vg")).toContainText("게이트–드레인 전계");
    await expect(win.getByTestId("pw-guide-row-vg").locator(".grow-cav")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(win).toHaveCount(0);
    await expect(trigger).toBeFocused();
  });

  test("calibration fields stay editable and changed extra fields remain visible", async ({ page }) => {
    await fresh(page);
    const adv = page.getByTestId("adv-groups").locator(".adv-head");
    await adv.click();
    const calib = page.getByTestId("group-calib");
    await calib.locator(".group-toggle").click();
    const beta = page.getByTestId("field-beta").locator("input.input");
    await beta.fill("10");
    await beta.press("Enter");
    await expect(beta).toHaveValue("10");
    await expect(calib.locator(".group-head")).toContainText("1개 수정");
    await page.getByTestId("tip-l_gidl").click();
    await expect(page.getByTestId("guide-pop").locator(".gp-intuitive")).toBeVisible();
    await expect(page.getByTestId("guide-pop").locator(".gp-tech")).toHaveCount(0);
    await page.keyboard.press("Escape");
    await adv.click();
    await expect(adv).toContainText("1개 수정");
    const extra = page.getByTestId("field-adv-bias");
    await extra.click();
    const rate = page.getByTestId("field-rate").locator("input.input");
    await rate.fill("2");
    await rate.press("Enter");
    await extra.click();
    await expect(rate).toBeVisible();
    await expect(rate).toHaveValue("2");
    await expect(page.getByTestId("field-dv")).toHaveCount(0);
    await expect(extra).toHaveText("고급 항목 1개");
  });

  test("English compact guides remain readable in both themes", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("lang-toggle").click();
    await expect(page.getByTestId("tip-vg")).toHaveAccessibleName(/Gate voltage design guide/);
    await expect(page.getByTestId("field-adv-bias")).toHaveText("2 more settings");
    for (const theme of ["light", "dark"] as const) {
      if (theme === "dark") await page.getByTestId("theme-toggle").click();
      await page.getByTestId("tip-vg").click();
      const pop = page.getByTestId("guide-pop");
      await expect(pop.locator(".gp-intuitive")).toHaveText("The gate is the handle of the hole tap at the drain edge (GIDL).");
      await expect(pop.getByTestId("guide-pop-more")).toHaveText("Documentation →");
      await expect(pop).not.toContainText(/\bguide\.[a-z]/);
      for (const sel of [".gp-intuitive", ".q-lu", ".q-ld", ".gchips-lead", ".gp-reference", ".gp-more"]) {
        const c = await contrastOf(pop.locator(sel));
        expect(c, `${theme} ${sel} contrast ${c.toFixed(2)}`).toBeGreaterThanOrEqual(4.5);
      }
      await page.keyboard.press("Escape");
    }
  });

  test("all-settings layout also keeps explanations behind info buttons", async ({ page }) => {
    await fresh(page, "&view=all");
    for (const key of ["rate", "dv", "dphiG0"]) await expect(page.getByTestId(`field-${key}`)).toBeVisible();
    await expect(page.getByTestId("sidebar").locator(".guide-inline")).toHaveCount(0);
    await page.getByTestId("tip-rate").click();
    await expect(page.getByTestId("guide-pop").locator(".gp-intuitive")).toHaveText("드레인 전압을 올리고 내리는 속도입니다.");
  });
});

test.describe("compact guides on phone", () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });
  test("tapping info opens a short bottom sheet without horizontal overflow", async ({ page }) => {
    await fresh(page);
    await page.getByRole("button", { name: /파라미터 패널|Parameter panel/ }).click();
    await expect(page.getByTestId("sidebar").locator(".guide-inline")).toHaveCount(0);
    await page.getByTestId("tip-vg").tap();
    const sheet = page.getByTestId("guide-pop");
    await expect(sheet).toHaveClass(/sheet/);
    await settle(sheet);
    const box = (await sheet.boundingBox())!;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(390.5);
    expect(Math.abs(box.y + box.height - 844)).toBeLessThanOrEqual(1);
    expect(box.height).toBeLessThan(280);
    expect(await sheet.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
    await expect(sheet.locator(".gchip")).toHaveCount(2);
    await expect(sheet.getByTestId("guide-pop-more")).toBeVisible();
    await sheet.getByTestId("guide-pop-close").tap();
    await expect(sheet).toHaveCount(0);
    await expect(page.getByTestId("tip-vg")).toBeFocused();
  });
});
