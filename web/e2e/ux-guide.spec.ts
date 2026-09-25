// Compact parameter controls and click-only design guides (D1): the ⓘ popover shows the guide first (picture,
// V_LU / V_LD effect lines, caveat), then the one-line definition, code, default and range, and the documentation
// link. Keyboard access, changed calibration parameters, both languages, geometry fields and the phone sheet.
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

  test("info button opens the guide on click, guide first; keyboard, outside click, and documentation work", async ({ page }) => {
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
    // guide first: the picture, then the three effect lines, the caveat, then the technical block
    await expect(pop.locator(".gp-intuitive")).toContainText("게이트는 드레인 가장자리 '정공 수도꼭지'(GIDL");
    const effects = pop.getByTestId("guide-pop-effects").locator("li");
    await expect(effects).toHaveCount(3);
    await expect(effects.nth(0)).toContainText(/^VLU ↑/);
    await expect(effects.nth(1)).toContainText(/^VLD/);
    await expect(pop.locator(".gp-caveat")).toBeVisible();
    await expect(pop.locator(".gp-basis")).toContainText("기준 보정");
    const tech = pop.getByTestId("guide-pop-tech");
    await expect(tech).toContainText("게이트-소스 전압");
    await expect(tech).toContainText("p[11]");
    await expect(tech).toContainText("기본값 −2 V");
    await expect(tech).toContainText("범위 −6 … 1 V");
    const order = await pop.evaluate((el) => [".gp-intuitive", ".gp-effects", ".gp-caveat", ".gp-tech"].map((s) => el.querySelector(s)!.getBoundingClientRect().top));
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    await settle(pop);
    const box = (await pop.boundingBox())!;
    expect(box.width).toBeLessThanOrEqual(321);
    expect(box.y).toBeGreaterThanOrEqual(0);
    expect(box.y + box.height).toBeLessThanOrEqual(900.5);
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
    // the one-line definition brings back the E_G formula of l_GIDL
    await expect(page.getByTestId("guide-pop-help")).toContainText("GIDL 전계를 정하는 유효 길이");
    await expect(page.getByTestId("guide-pop-help")).toContainText("l_GIDL");
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
      await expect(pop.locator(".gp-intuitive")).toContainText("The gate is the handle of the hole tap at the drain edge (GIDL");
      await expect(pop.getByTestId("guide-pop-tech")).toContainText("Definition · default");
      await expect(pop.getByTestId("guide-pop-more")).toHaveText("Documentation →");
      await expect(pop).not.toContainText(/\bguide\.[a-z]/);
      for (const sel of [".gp-intuitive", ".q-lu", ".q-ld", ".gp-h", ".gp-basis", ".gp-tech-help", ".gp-tech-meta", ".gp-more"]) {
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
    await expect(page.getByTestId("guide-pop").locator(".gp-intuitive")).toContainText("드레인 전압을 올리고 내리는 속도입니다.");
  });
});

test("geometry fields and V_BG have their own guides; a field without a guide entry still explains itself", async ({ page }) => {
  await fresh(page);
  for (const key of ["Lg_nm", "W_nm", "Tsi_nm", "EOT_nm", "Tbox_nm", "Nbody_cm3"]) await expect(page.getByTestId(`tip-geometry-${key}`)).toBeVisible();
  await page.getByTestId("tip-geometry-Nbody_cm3").click();
  const pop = page.getByTestId("guide-pop");
  await expect(pop).toHaveAccessibleName(/바디 도핑/);
  await expect(pop.getByTestId("guide-pop-help")).toContainText("p형 도핑");
  await expect(pop.getByTestId("guide-pop-tech")).toContainText("×10");
  await expect(pop.getByTestId("guide-pop-more")).toBeVisible();
  await page.keyboard.press("Escape");
  await page.getByTestId("tip-vbg").click();
  await expect(pop.getByTestId("guide-pop-help")).toContainText("전면 채널 전류");
  await expect(pop.getByTestId("guide-pop-help")).toContainText("바디 정공 저장 효과는 포함하지 않습니다");
  await expect(pop).not.toContainText("V_BG");
  await page.keyboard.press("Escape");
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
    expect(box.height).toBeLessThanOrEqual(844 * 0.7 + 1);
    expect(await sheet.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
    await expect(sheet.getByTestId("guide-pop-effects").locator("li")).toHaveCount(3);
    await expect(sheet.getByTestId("guide-pop-more")).toBeVisible();
    await sheet.getByTestId("guide-pop-close").tap();
    await expect(sheet).toHaveCount(0);
    await expect(page.getByTestId("tip-vg")).toBeFocused();
  });
});
