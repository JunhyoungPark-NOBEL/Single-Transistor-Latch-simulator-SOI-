// Parameter guide + simplified sidebar (간단히 default): device card, basic/advanced groups, inline guides on
// main fields, the ⓘ popover (hover preview, pinned dialog, phone bottom sheet), the Details window's
// "한눈에" block, the Physics-tab guide list, both languages and themes. Writes the canonical screenshots
// sidebar-guide.png, guide-popover.png, phone-guide-sheet.png, details-window.png, sidebar-advanced.png.
import { expect, test, type Locator, type Page } from "@playwright/test";
import { PARAM_GUIDE } from "../src/content/params/guide";
import { inlineLead, parseEffect } from "../src/params/guideUi";

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
  if (!hash.includes("tab=physics")) await expect(page.getByTestId("sidebar")).toBeAttached();
}

/** Wait for CSS animations (sheet slide-in) so boxes are measured at rest. */
const settle = (loc: Locator) => loc.evaluate((el) => Promise.all(el.getAnimations({ subtree: true }).map((a) => a.finished)).then(() => undefined));

/** Chip text the way the inline guide renders an effect line ("VLU ↑ 80 mV", "VLD → 그대로"). */
function chipText(line: string, lang: "ko" | "en"): RegExp {
  const e = parseEffect(line)!;
  const q = e.q === "V_LU" ? "VLU" : "VLD";
  const mean = e.mean ? (lang === "ko" ? " 평균" : " mean") : "";
  const arrow = e.dir === "up" ? "↑" : e.dir === "down" ? "↓" : "→";
  const mag = e.dir === "flat" ? (lang === "ko" ? "그대로" : "unchanged") : (e.mag ?? "");
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`^${esc(`${q}${mean} ${arrow} ${mag}`.trim())}`);
}

async function expectInline(page: Page, key: string, lang: "ko" | "en" = "ko") {
  const g = PARAM_GUIDE[key];
  const inline = page.getByTestId(`guide-inline-${key}`);
  await expect(inline).toBeVisible();
  // the whole intuitive picture with glosses collapsed ("(GIDL)"), so the plain "raise it → …" half is there
  // (subscripts render V_G as "VG")
  const lead = inlineLead(g.intuitive[lang]).replace(/([A-Za-z])_([A-Za-z0-9]+)/g, "$1$2");
  await expect(inline.locator(".gi-text")).toHaveText(lead);
  // ≤ 3 lines
  const lines = await inline.locator(".gi-text").evaluate((el) => Math.round(el.clientHeight / parseFloat(getComputedStyle(el).lineHeight)));
  expect(lines).toBeLessThanOrEqual(3);
  await expect(inline.locator(".gchip.lu")).toHaveText(chipText(g.effect[0][lang], lang));
  await expect(inline.locator(".gchip.ld")).toHaveText(chipText(g.effect[1][lang], lang));
}

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

test.describe("parameter guide and simplified sidebar (간단히)", () => {
  test("sidebar: compact device card, basic groups with main fields, advanced settings closed", async ({ page }) => {
    await fresh(page);
    const card = page.getByTestId("preset-card");
    await expect(card).toBeVisible();
    expect((await card.boundingBox())!.height).toBeLessThanOrEqual(150);
    await expect(page.getByTestId("dev-geometry")).toContainText("500 nm");
    await expect(page.getByTestId("preset-label")).toContainText("V");
    await expect(page.getByTestId("preset-label")).not.toContainText("수정");
    // 기술 · 저장 is closed but its controls stay in the DOM
    await expect(page.getByTestId("dev-more")).not.toHaveAttribute("open", "");
    await expect(page.getByTestId("tech-PDSOI")).toBeDisabled();
    await expect(page.getByTestId("dev-load")).toBeAttached();

    for (const id of ["bias", "light"]) await expect(page.getByTestId(`group-${id}`).locator(".group-toggle")).toHaveAttribute("aria-expanded", "true");
    const adv = page.getByTestId("adv-groups");
    await expect(adv).toBeVisible();
    await expect(adv.locator(".adv-head")).toHaveAttribute("aria-expanded", "false");
    await expect(adv).toContainText("고급 설정");
    for (const id of ["state", "calib", "ext", "numerics"]) await expect(page.getByTestId(`group-${id}`)).toHaveCount(0);
    // order: card, bias, light, 고급 설정
    const ys = await Promise.all(["preset-card", "group-bias", "group-light", "adv-groups"].map(async (id) => (await page.getByTestId(id).boundingBox())!.y));
    expect([...ys].sort((a, b) => a - b)).toEqual(ys);
    // only main fields in the basic groups; the rest fold behind "고급 항목 n개"
    await expect(page.getByTestId("field-vg")).toBeVisible();
    await expect(page.getByTestId("field-vd_max")).toBeVisible();
    await expect(page.getByTestId("field-rate")).toHaveCount(0);
    await expect(page.getByTestId("field-dv")).toHaveCount(0);
    await expect(page.getByTestId("field-adv-bias")).toContainText("고급 항목 2개");
    // sliders only on main fields
    await expect(page.getByTestId("field-vg").locator('input[type="range"]')).toBeVisible();
    // group heads: no description line, icon-only 📖 with an accessible name
    await expect(page.getByTestId("sidebar").locator(".group-desc")).toHaveCount(0);
    await expect(page.getByTestId("details-group-bias")).toHaveAccessibleName(/물리 설명/);
    // the first field label is high in the first viewport
    expect((await page.getByTestId("field-vg").locator(".field-label").boundingBox())!.y).toBeLessThanOrEqual(330);
    // Run bar: one row (Run + 자동 실행), shortcut only in the title
    const run = page.getByTestId("run-button");
    await expect(run).toHaveAttribute("title", "Ctrl/⌘ + Enter");
    await expect(page.getByTestId("runbar")).not.toContainText("Ctrl/⌘ + Enter");
    const rb = (await run.boundingBox())!;
    const ar = (await page.getByTestId("autorun").boundingBox())!;
    expect(Math.abs(rb.y + rb.height / 2 - (ar.y + ar.height / 2))).toBeLessThan(6);
    await page.waitForTimeout(300);
    await page.getByTestId("sidebar").screenshot({ path: `${SHOTS}/sidebar-guide.png` });
  });

  test("every main field shows the inline guide with V_LU / V_LD chips from the guide", async ({ page }) => {
    await fresh(page);
    for (const key of ["vg", "vd_max", "iph_pA"]) await expectInline(page, key);
    // the V_G example of the spec
    await expect(page.getByTestId("guide-inline-vg")).toContainText("키우면");
    await expect(page.getByTestId("guide-inline-vg").locator(".gchip.lu")).toHaveText("VLU ↑ 80 mV");
    await expect(page.getByTestId("guide-inline-vg").locator(".gchip.ld")).toHaveText("VLD → 그대로");
    // the step the numbers were measured with comes right after the verb; the plain sentence ends in its verb
    await expect(page.getByTestId("guide-inline-vg").locator(".gchips-step")).toHaveText("(+0.1 V)");
    await expect(page.getByTestId("guide-inline-vg").locator(".gi-text")).toContainText("손잡이입니다");
    // V_D,max moves neither voltage: the row says what does change
    await expect(page.getByTestId("guide-inline-vd_max").locator(".gchips-why")).toContainText("래치업되는 사이클 비율");
    // the line describes the input (aria-describedby)
    const inlineId = await page.getByTestId("guide-inline-vg").getAttribute("id");
    await expect(page.getByTestId("field-vg").locator("input.input")).toHaveAttribute("aria-describedby", new RegExp(inlineId!.replace(/[:]/g, "\\:")));
    // V_LU in --hrs, V_LD in --lrs
    const [hrs, lrs] = await page.evaluate(() => {
      const probe = document.createElement("span");
      document.body.appendChild(probe);
      probe.style.color = "var(--hrs)";
      const a = getComputedStyle(probe).color;
      probe.style.color = "var(--lrs)";
      const b = getComputedStyle(probe).color;
      probe.remove();
      return [a, b];
    });
    await expect(page.getByTestId("guide-inline-vg").locator(".q-lu")).toHaveCSS("color", hrs);
    await expect(page.getByTestId("guide-inline-vg").locator(".q-ld")).toHaveCSS("color", lrs);
    // advanced fields carry no inline text
    await page.getByTestId("field-adv-bias").click();
    await expect(page.getByTestId("field-rate")).toBeVisible();
    await expect(page.getByTestId("guide-inline-rate")).toHaveCount(0);
    await expect(page.getByTestId("field-rate").locator('input[type="range"]')).toHaveCount(0);
    // power mode: P is main, R folds behind "고급 항목 1개"
    await page.getByTestId("light-mode").getByRole("radio").nth(1).click();
    await expectInline(page, "power_mW");
    await expect(page.getByTestId("field-resp")).toHaveCount(0);
    await expect(page.getByTestId("field-adv-light")).toContainText("1");
    // stochastic mode: rate becomes main; the stochastic groups show their main fields
    await page.getByTestId("mode-stochastic").click();
    for (const key of ["rate", "n_cycles", "ls_mode", "ls_sigma"]) await expectInline(page, key);
    await expect(page.getByTestId("field-seed")).toHaveCount(0);
    await expect(page.getByTestId("field-adv-stoch")).toBeVisible();
    await expect(page.getByTestId("group-stoch")).toHaveClass(/sto-only/);
    await expect(page.getByTestId("group-stoch")).not.toContainText("Stochastic");
  });

  test("ⓘ popover: hover preview with the technical help, pinned dialog with the guide first, Esc and focus", async ({ page }) => {
    await fresh(page);
    // hover preview (role=tooltip) on an advanced field keeps the technical help (E_G)
    await page.getByTestId("adv-groups").locator(".adv-head").click();
    await page.getByTestId("group-calib").locator(".group-toggle").click();
    await page.getByTestId("tip-l_gidl").hover();
    const tip = page.getByRole("tooltip");
    await expect(tip).toContainText("E_G");
    await expect(tip.locator(".gp-block").first()).toHaveClass(/gp-easy/);
    await page.mouse.move(900, 600);
    await expect(tip).toHaveCount(0);

    // pinned dialog on V_G: block order 쉽게 말하면 → 키우면 (3) → ⚠ → 근거 → 기술 설명 → 물리 자세히 보기
    await page.getByTestId("tip-vg").click();
    const dlg = page.getByRole("dialog", { name: /게이트 전압/ });
    await expect(dlg).toBeVisible();
    await expect(dlg).toHaveAttribute("data-testid", "guide-pop");
    await settle(dlg);
    const order = await dlg.evaluate((el) => Array.from(el.children).map((c) => c.className.split(" ")[0]));
    expect(order).toEqual(["gp-head", "gp-block", "gp-block", "gp-caveat", "gp-basis", "gp-tech", "gp-foot"]);
    await expect(dlg.locator(".gp-intuitive")).toHaveText(PARAM_GUIDE.vg.intuitive.ko.replace(/([A-Za-z])_([A-Za-z0-9]+)/g, "$1$2"));
    await expect(dlg.locator(".gp-effects li")).toHaveCount(3);
    await expect(dlg.locator(".gp-effects li").nth(0).locator(".q-lu")).toBeVisible();
    await expect(dlg.locator(".gp-effects li").nth(1).locator(".q-ld")).toBeVisible();
    await expect(dlg.locator(".gp-tech")).toContainText("p[11]");
    await expect(dlg.locator(".gp-basis")).toContainText("기준 보정");
    const box = (await dlg.boundingBox())!;
    expect(box.width).toBeLessThanOrEqual(341);
    expect(box.y + box.height).toBeLessThanOrEqual(900);
    await page.waitForTimeout(200);
    await page.screenshot({ path: `${SHOTS}/guide-popover.png` });
    // caveat: 2-line clamp with 더 보기
    await dlg.locator(".gp-caveat-more").click();
    await expect(dlg.locator(".gp-caveat-text")).toHaveClass(/open/);
    // one open at a time: pinning another replaces it
    await page.getByTestId("tip-vd_max").click();
    await expect(page.getByTestId("guide-pop")).toHaveCount(1);
    await expect(page.getByRole("dialog", { name: /스윕 최대/ })).toBeVisible();
    // click outside closes
    await page.mouse.click(900, 500);
    await expect(page.getByTestId("guide-pop")).toHaveCount(0);
    // keyboard: Enter pins, Esc closes and returns focus to the ⓘ
    await page.getByTestId("tip-vg").focus();
    await page.keyboard.press("Enter");
    await expect(page.getByRole("dialog", { name: /게이트 전압/ })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("guide-pop")).toHaveCount(0);
    await expect(page.getByTestId("tip-vg")).toBeFocused();
    // clicking the inline line pins the same popover
    await page.getByTestId("guide-inline-vg").locator(".gi-text").click();
    await expect(page.getByRole("dialog", { name: /게이트 전압/ })).toBeVisible();
    // 물리 자세히 보기 → Details window with the guide and the V_G row focused
    await page.getByTestId("guide-pop-more").click();
    const win = page.getByTestId("physics-window");
    await expect(win).toBeVisible();
    await expect(page.getByTestId("guide-pop")).toHaveCount(0);
    await expect(win.getByTestId("pw-guide-row-vg")).toHaveClass(/hl/);
    // Esc precedence: a popover pinned over the window closes first, then the window
    await page.getByTestId("tip-vd_max").click();
    await expect(page.getByTestId("guide-pop")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("guide-pop")).toHaveCount(0);
    await expect(win).toBeVisible();
    await expect(page.getByTestId("tip-vd_max")).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(win).toHaveCount(0);
  });

  test("Details window from a group starts with “한눈에”; from a result panel it does not", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("light-mode").getByRole("radio").nth(1).click();
    const btn = page.getByTestId("details-group-light");
    await btn.click();
    const win = page.getByTestId("physics-window");
    await expect(win).toBeVisible();
    await expect(win.getByRole("heading", { level: 2 })).toBeFocused();
    await expect(win.getByRole("heading", { level: 2 })).toHaveCount(1);
    const guide = win.getByTestId("pw-guide");
    await expect(guide).toBeVisible();
    // first section of the body, above every equation
    expect(await win.locator(".pw-body").evaluate((b) => b.firstElementChild?.getAttribute("data-testid"))).toBe("pw-guide");
    const gy = (await guide.boundingBox())!.y;
    const eq = win.getByTestId("equation").first();
    if (await eq.count()) expect((await eq.boundingBox())!.y).toBeGreaterThan(gy);
    // first pill "한눈에", active on open
    const pill = win.locator(".pw-nav .pill").first();
    await expect(pill).toHaveText("한눈에");
    await expect(pill).toHaveClass(/active/);
    // P and R rows, 2 columns at ≥ 520 px window width
    await expect(guide.getByTestId("pw-guide-row-power_mW")).toBeVisible();
    await expect(guide.getByTestId("pw-guide-row-resp")).toBeVisible();
    expect((await win.boundingBox())!.width).toBeGreaterThanOrEqual(520);
    const cols = await guide.getByTestId("pw-guide-row-power_mW").evaluate((el) => getComputedStyle(el).gridTemplateColumns.split(" ").length);
    expect(cols).toBe(2);
    await expect(guide).toContainText("V");
    // no kicker line; tags live in the footer
    await expect(win.locator(".pw-kicker")).toHaveCount(0);
    await page.waitForTimeout(250);
    await page.screenshot({ path: `${SHOTS}/details-window-light.png` });
    // a related topic hides the block, Back restores it
    const related = win.locator("[data-testid^=related-]").first();
    if (await related.count()) {
      await related.click();
      await expect(win.getByTestId("pw-guide")).toHaveCount(0);
      await page.getByTestId("physics-back").click();
      await expect(win.getByTestId("pw-guide")).toBeVisible();
    }
    await page.keyboard.press("Escape");
    await expect(win).toHaveCount(0);
    await expect(btn).toBeFocused();

    // from the bias group (canonical screenshot)
    await page.getByTestId("details-group-bias").click();
    await expect(win.getByTestId("pw-guide-row-vg")).toBeVisible();
    await expect(win.getByTestId("pw-guide-row-vd_max")).toBeVisible();
    await page.waitForTimeout(250);
    await page.screenshot({ path: `${SHOTS}/details-window.png` });
    await page.keyboard.press("Escape");

    // from a result panel: no guide block
    await page.getByTestId("details-panel-iv").click();
    await expect(win).toBeVisible();
    await expect(win.getByTestId("pw-guide")).toHaveCount(0);
    await expect(win.locator(".pw-nav .pill").first()).not.toHaveText("한눈에");
  });

  test("a changed advanced value shows as “n개 수정” and is never hidden", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("adv-groups").locator(".adv-head").click();
    await expect(page.getByTestId("group-calib")).toBeVisible();
    await expect(page.getByTestId("group-calib").locator(".group-sum")).toHaveText("기본값");
    await page.getByTestId("group-calib").locator(".group-toggle").click();
    const beta = page.getByTestId("field-beta").locator("input.input");
    await beta.fill("10");
    await beta.press("Enter");
    await expect(page.getByTestId("field-beta")).toBeVisible();
    await expect(page.getByTestId("group-calib").locator(".group-head")).toContainText("1개 수정");
    await expect(page.getByTestId("adv-groups").locator(".adv-head")).toContainText("1개 수정");
    await expect(page.getByTestId("preset-label")).toContainText("기준 보정에서 수정");
    await page.getByTestId("sidebar").locator(".sidebar-scroll").evaluate((el) => {
      const adv = el.querySelector<HTMLElement>('[data-testid="adv-groups"]')!;
      el.scrollTo(0, adv.getBoundingClientRect().top - el.getBoundingClientRect().top + el.scrollTop - 12);
    });
    await page.waitForTimeout(250);
    await page.getByTestId("sidebar").screenshot({ path: `${SHOTS}/sidebar-advanced.png` });
    // collapsed group and closed disclosure keep the badge
    await page.getByTestId("group-calib").locator(".group-toggle").click();
    await expect(page.getByTestId("group-calib").locator(".group-head")).toContainText("1개 수정");
    await page.getByTestId("adv-groups").locator(".adv-head").click();
    await expect(page.getByTestId("group-calib")).toHaveCount(0);
    await expect(page.getByTestId("adv-groups")).toContainText("1개 수정");
    // a changed non-main field of a basic group stays visible with its disclosure closed
    await page.getByTestId("field-adv-bias").click();
    const rate = page.getByTestId("field-rate").locator("input.input");
    await rate.fill("2");
    await rate.press("Enter");
    await page.getByTestId("field-adv-bias").click();
    await expect(page.getByTestId("field-rate")).toBeVisible();
    await expect(page.getByTestId("field-dv")).toHaveCount(0);
    await expect(page.getByTestId("field-adv-bias")).toContainText("고급 항목 1개");
    // the disclosure state persists
    await page.getByTestId("adv-groups").locator(".adv-head").click();
    await page.reload();
    await expect(page.getByTestId("adv-groups").locator(".adv-head")).toHaveAttribute("aria-expanded", "true");
  });

  test("Physics tab: guide list at the top of the parameters topic, filtered by search", async ({ page }) => {
    await fresh(page, "#tab=physics&mode=deterministic");
    await expect(page.getByTestId("physics-tab")).toBeVisible();
    expect(await page.locator("[data-testid^=topic-]").count()).toBe(18);
    // the guide comes first, above topic 1, and the TOC starts with it
    const list = page.getByTestId("physics-guide").getByTestId("guide-list");
    await expect(list).toBeInViewport();
    const gy = (await page.getByTestId("physics-guide").boundingBox())!.y;
    expect((await page.getByTestId("topic-overview").boundingBox())!.y).toBeGreaterThan(gy);
    await page.getByTestId("guide-jump").click();
    await expect(list).toBeInViewport();
    await expect(list.getByTestId("guide-row-vg")).toBeVisible();
    // main fields first within a group
    const bias = await list.getByTestId("guide-group-bias").locator("article").evaluateAll((els) => els.map((e) => e.getAttribute("data-testid")));
    expect(bias.slice(0, 2)).toEqual(["guide-row-vg", "guide-row-vd_max"]);
    const before = await list.locator("[data-testid^=guide-row-]").count();
    expect(before).toBeGreaterThan(40);
    await page.getByTestId("physics-search").fill("GIDL");
    await expect.poll(async () => list.locator("[data-testid^=guide-row-]").count()).toBeLessThan(before);
    await expect(list.getByTestId("guide-row-l_gidl")).toBeVisible();
    await expect(list.getByTestId("guide-row-n_cycles")).toHaveCount(0);
    await expect(page.getByTestId("topic-parameters")).toBeVisible();
  });

  test("English, dark theme: no missing strings, readable token colours", async ({ page }) => {
    await fresh(page);
    await page.getByTestId("lang-toggle").click();
    await expect(page.getByTestId("guide-inline-vg")).toContainText("If raised");
    await expectInline(page, "vg", "en");
    await expectInline(page, "vd_max", "en");
    await expect(page.getByTestId("adv-groups")).toContainText("Advanced settings");
    await expect(page.getByTestId("field-adv-bias")).toContainText("2 more settings");
    await expect(page.getByTestId("dev-more")).toContainText("Tech · Save");
    const sidebar = page.getByTestId("sidebar");
    await expect(sidebar).not.toContainText(/\b(guide|adv|dev|group|pw|light)\.[a-z]+/);
    await page.getByTestId("tip-vg").click();
    const dlg = page.getByRole("dialog", { name: /Gate voltage/ });
    await expect(dlg).toContainText("In plain words");
    await expect(dlg.locator(".gp-intuitive")).toHaveText(PARAM_GUIDE.vg.intuitive.en.replace(/([A-Za-z])_([A-Za-z0-9]+)/g, "$1$2"));
    await expect(dlg).not.toContainText(/\bguide\.[a-z]/);
    await page.keyboard.press("Escape");

    for (const theme of ["light", "dark"] as const) {
      if (theme === "dark") {
        await page.getByTestId("theme-toggle").click();
        await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
      }
      const inline = page.getByTestId("guide-inline-vg");
      for (const sel of [".q-lu", ".q-ld", ".gi-text", ".gchips-lead", ".gchip-mag"]) {
        const c = await contrastOf(inline.locator(sel));
        expect(c, `${theme} ${sel} contrast ${c.toFixed(2)}`).toBeGreaterThanOrEqual(4.5);
      }
      await page.getByTestId("tip-vg").click();
      const pop = page.getByTestId("guide-pop");
      for (const sel of [".gp-intuitive", ".gp-effects .q-lu", ".gp-effects .q-ld", ".gp-caveat-text", ".gp-basis", ".gp-tech-help", ".gp-h"]) {
        const c = await contrastOf(pop.locator(sel));
        expect(c, `${theme} popover ${sel} contrast ${c.toFixed(2)}`).toBeGreaterThanOrEqual(4.5);
      }
      const bg = await pop.evaluate((el) => getComputedStyle(el).backgroundColor);
      expect(bg).not.toMatch(/rgba\(.*, 0(\.\d+)?\)$/); // opaque
      await page.keyboard.press("Escape");
    }
  });

  test("모두 보기 (?view=all) reproduces today's sidebar", async ({ page }) => {
    await fresh(page, "#tab=device&mode=deterministic", "&view=all");
    await expect(page.getByTestId("adv-groups").locator(".adv-head")).toHaveAttribute("aria-expanded", "true");
    await expect(page.getByTestId("group-calib")).toBeVisible();
    await expect(page.getByTestId("group-state")).toBeVisible();
    await expect(page.getByTestId("field-dphiG0")).toBeVisible(); // state group open (today's default)
    await expect(page.getByTestId("field-rate")).toBeVisible();
    await expect(page.getByTestId("field-dv")).toBeVisible();
    await expect(page.getByTestId("field-rate").locator('input[type="range"]')).toBeVisible();
    await expect(page.getByTestId("field-adv-bias")).toHaveCount(0);
    await expect(page.getByTestId("dev-more")).toHaveAttribute("open", "");
    await expect(page.getByTestId("tech-select")).toBeVisible();
  });
});

test.describe("phone (390 × 844, touch)", () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });

  test("tapping ⓘ opens a bottom sheet with the guide first; Esc closes it and focus returns", async ({ page }) => {
    await fresh(page);
    await page.getByRole("button", { name: /파라미터 패널|Parameter panel/ }).click();
    const sidebar = page.getByTestId("sidebar");
    await expect(sidebar).toBeInViewport();
    await expect(page.getByTestId("guide-inline-vg")).toBeVisible();
    // no horizontal scroll: neither the page nor the drawer
    expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
    expect(await sidebar.locator(".sidebar-scroll").evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
    await page.getByTestId("tip-vg").tap();
    const sheet = page.getByRole("dialog", { name: /게이트 전압/ });
    await expect(sheet).toBeVisible();
    await expect(sheet).toHaveClass(/sheet/);
    await settle(sheet);
    const box = (await sheet.boundingBox())!;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(390.5);
    expect(Math.abs(box.y + box.height - 844)).toBeLessThanOrEqual(1);
    expect(box.height).toBeLessThanOrEqual(844 * 0.7 + 1);
    expect(await sheet.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
    // first text block = the intuitive picture; second = the 3 effect lines
    const blocks = sheet.locator(".gp-block");
    await expect(blocks.nth(0).locator("p")).toHaveText(PARAM_GUIDE.vg.intuitive.ko.replace(/([A-Za-z])_([A-Za-z0-9]+)/g, "$1$2"));
    await expect(blocks.nth(1).locator("li")).toHaveCount(3);
    await expect(sheet.getByTestId("guide-pop-close")).toBeVisible();
    await page.waitForTimeout(250);
    await page.screenshot({ path: `${SHOTS}/phone-guide-sheet.png` });
    await page.keyboard.press("Escape");
    await expect(sheet).toHaveCount(0);
    await expect(page.getByTestId("tip-vg")).toBeFocused();
  });
});
