import { expect, test, type Page } from "@playwright/test";
import { mkdirSync, readFileSync } from "node:fs";
import { snapshotKey } from "../src/api/snapshot";

const SHOTS = "review";
const DEVICE_KEY = "stl-websim:devices";

async function fresh(page: Page) {
  await page.addInitScript(() => {
    if (!sessionStorage.getItem("workspace-review-init")) {
      localStorage.clear();
      sessionStorage.setItem("workspace-review-init", "1");
    }
  });
  await page.goto("/?mock=1#tab=device&mode=deterministic");
  await page.evaluate(() => document.fonts.ready);
  await expect(page.getByTestId("panel-iv").locator(".js-plotly-plot")).toBeVisible();
  await expect(page.getByTestId("offline-banner")).toContainText(/데모|demo/i);
}

async function save(page: Page, name: string) {
  await page.getByTestId("dev-save-open").click();
  await page.getByTestId("dev-save-name").fill(name);
  await page.getByTestId("dev-save-submit").click();
  await expect(page.getByTestId("save-device-dialog")).toHaveCount(0);
}

async function setVg(page: Page, value: string) {
  const input = page.getByTestId("field-vg").locator("input.input");
  await input.fill(value);
  await input.press("Enter");
}

async function stored(page: Page) {
  return page.evaluate(key => JSON.parse(localStorage.getItem(key) ?? '{"devices":[]}').devices, DEVICE_KEY);
}

async function noHorizontalOverflow(page: Page) {
  await page.evaluate(() => document.fonts.ready);
  await expect.poll(() => page.evaluate(() => document.fonts.check('14px "Noto Sans KR Variable"', "사용자 소자"))).toBe(true);
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
}

test("five visible shelf slots save, reload, load, and update the current calibration without renaming", async ({ page }) => {
  await fresh(page);
  const shelf = page.getByTestId("device-shelf");
  await expect(shelf.locator('[data-testid^="device-slot-"]')).toHaveCount(5);
  for (let i = 1; i <= 5; i++) {
    await expect(page.getByTestId(`device-slot-${i}`)).toBeInViewport();
    expect((await page.getByTestId(`device-slot-${i}`).boundingBox())!.height).toBeLessThanOrEqual(80);
  }
  const shelfBox = (await shelf.boundingBox())!;
  const plotBox = (await page.getByTestId("panel-iv").boundingBox())!;
  expect(shelfBox.x).toBeGreaterThan(plotBox.x + plotBox.width);
  expect(shelfBox.width).toBeLessThanOrEqual(240);
  await setVg(page, "-1.8");
  await save(page, "My calibrated FDSOI");
  const first = (await stored(page))[0];
  expect(first.device.vg).toBe(-1.8);
  await page.reload();
  await expect(page.getByTestId("device-slot-1")).toContainText("My calibrated FDSOI");
  await setVg(page, "-2.1");
  await page.getByTestId("device-slot-1").getByRole("button", { name: "불러오기", exact: true }).click();
  await expect(page.getByTestId("field-vg").locator("input.input")).toHaveValue("-1.8");
  await setVg(page, "-1.7");
  await page.getByTestId("device-slot-1").getByRole("button", { name: "업데이트", exact: true }).click();
  await page.getByTestId("device-slot-1").getByRole("button", { name: "저장", exact: true }).click();
  const updated = (await stored(page))[0];
  expect(updated.id).toBe(first.id);
  expect(updated.name).toBe(first.name);
  expect(updated.device.vg).toBe(-1.7);
  await page.reload();
  await expect(page.getByTestId("device-slot-1")).toContainText("-1.7");
  expect(await stored(page)).toHaveLength(1);
});

test("imports stop at five slots and full shelves disable save and duplicate", async ({ page }) => {
  await fresh(page);
  await save(page, "Original device");
  const source = (await stored(page))[0];
  const devices = Array.from({ length: 6 }, (_, i) => ({ ...source, id: `qa-import-${i}`, name: `Imported ${i + 1}` }));
  await page.getByTestId("dev-manage").click();
  await expect(page.getByTestId("device-manager")).toBeVisible();
  await page.getByTestId("dm-import-input").setInputFiles({ name: "six-devices.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ format: "stl-device-library", v: 1, devices })) });
  await expect(page.getByTestId("dm-user-list").locator(".dm-row")).toHaveCount(5);
  await expect(page.getByTestId("dm-msg")).toContainText("4");
  await expect(page.getByTestId("dm-msg")).toContainText("2");
  for (const duplicate of await page.getByTestId("dm-duplicate").all()) await expect(duplicate).toBeDisabled();
  await page.getByTestId("dm-import-input").setInputFiles({ name: "one-device.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ ...source, id: "sixth-device", name: "Rejected sixth" })) });
  await expect(page.getByTestId("dm-msg")).toContainText("0");
  await expect(page.getByTestId("dm-user-list").locator(".dm-row")).toHaveCount(5);
  await page.getByTestId("modal-close").click();
  await expect(page.getByTestId("dev-save-open")).toBeDisabled();
  await expect(page.locator(".shelf-count")).toHaveText("5 / 5");
  await page.reload();
  await expect(page.getByTestId("device-shelf").locator(".device-slot.filled")).toHaveCount(5);
  expect(await stored(page)).toHaveLength(5);
});

test("creator attribution is a compact disclosure with lab and institution only", async ({ page }) => {
  await fresh(page);
  const credits = page.getByTestId("credits-chip");
  expect((await credits.innerText()).trim()).toBe("제작자");
  await expect(page.getByTestId("about")).toHaveCount(0);
  await credits.click();
  const about = page.getByTestId("about");
  await expect(about).toBeVisible();
  await expect(about).toContainText("KAIST");
  await expect(about).toContainText(/NOBEL/i);
  await expect(about).not.toContainText(/Choi|최양규|Yang.?Kyu/i);
  await page.keyboard.press("Escape");
  await expect(about).toHaveCount(0);
  await expect(credits).toBeFocused();
});

test("Device 1 is the only default library entry; custom devices remain available after reload", async ({ page }) => {
  await fresh(page);
  await expect(page.getByTestId("preset-card")).toContainText("Device 1");
  await expect(page.getByTestId("preset-photo")).toHaveCount(0);
  await expect(page.getByTestId("preset-custom")).toHaveCount(0);
  await page.getByTestId("dev-manage").click();
  await expect(page.getByTestId("device-manager").locator(".dm-row.builtin")).toHaveCount(1);
  await expect(page.getByTestId("dm-row-builtin:paper")).toContainText("Device 1");
  await expect(page.getByTestId("dm-row-builtin:photo")).toHaveCount(0);
  await page.getByTestId("modal-close").click();
  await page.getByTestId("tab-circuit").click();
  await expect(page.getByTestId("lib-list").locator("li")).toHaveCount(1);
  await expect(page.getByTestId("lib-builtin:paper")).toContainText("Device 1");
  await expect(page.getByTestId("lib-current")).toHaveCount(0);
  await page.getByTestId("tab-device").click();
  await setVg(page, "-1.9");
  await save(page, "Custom calibration");
  await page.reload();
  await page.getByTestId("tab-circuit").click();
  await expect(page.getByTestId("lib-list").locator("li")).toHaveCount(2);
  await expect(page.getByTestId("lib-Custom calibration")).toBeVisible();
  await expect(page.getByTestId("lib-builtin:paper")).toBeVisible();
});

for (const width of [1440, 390]) {
  test(`${width}px: simulator export offers LTspice and Verilog-A; TCAD explains its limits`, async ({ page }) => {
    mkdirSync(SHOTS, { recursive: true });
    await page.setViewportSize({ width, height: width === 1440 ? 1000 : 844 });
    await fresh(page);
    await page.getByTestId("model-export-open").click();
    const dialog = page.getByTestId("model-export-dialog");
    await expect(dialog).toBeVisible();
    await dialog.evaluate(async el => {
      await Promise.all([...el.getAnimations(), ...(el.parentElement?.getAnimations() ?? [])].map(animation => animation.finished));
    });
    await expect(dialog).toHaveCSS("opacity", "1");
    await expect(page.locator(".modal-backdrop")).toHaveCSS("opacity", "1");
    expect(await dialog.evaluate(el => {
      const box = el.getBoundingClientRect();
      return el.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2));
    })).toBe(true);
    await expect(dialog.getByRole("radio")).toHaveCount(3);
    await expect(page.getByTestId("model-export-ltspice")).toBeChecked();
    await expect(page.getByTestId("model-export-download")).toHaveText(".cir 다운로드");
    await expect(page.getByTestId("model-export-download")).toBeDisabled();
    await expect(dialog).toContainText("계산 서버");
    await noHorizontalOverflow(page);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: `${SHOTS}/${width === 1440 ? "desktop" : "mobile"}-export-ltspice.png`, animations: "disabled" });
    await page.getByTestId("model-export-verilog-a").check();
    await expect(page.getByTestId("model-export-download")).toHaveText(".va 다운로드");
    await expect(page.getByTestId("model-export-download")).toBeDisabled();
    await expect(dialog).toContainText("CSVM");
    await page.getByTestId("model-export-sentaurus").check();
    await expect(page.getByTestId("model-export-sentaurus-info")).toContainText("격자·도핑·접촉");
    await expect(page.getByTestId("model-export-download")).toHaveText("내보내기 미지원");
    await expect(page.getByTestId("model-export-download")).toBeDisabled();
    await noHorizontalOverflow(page);
    const box = (await dialog.boundingBox())!;
    expect(box.y).toBeGreaterThanOrEqual(0);
    expect(box.y + box.height).toBeLessThanOrEqual(width === 1440 ? 1000 : 844);
    await page.screenshot({ path: `${SHOTS}/${width === 1440 ? "desktop" : "mobile"}-export-tcad.png`, animations: "disabled" });
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
  });
}

for (const width of [1440, 390]) {
  test(`${width}px: light/dark device, circuit and reference fit; save review screenshots`, async ({ page }) => {
    mkdirSync(SHOTS, { recursive: true });
    await page.setViewportSize({ width, height: width === 1440 ? 1000 : 844 });
    await fresh(page);
    for (const theme of ["light", "dark"]) {
      if (theme === "dark") await page.getByTestId("theme-toggle").click();
      await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
      for (const tab of ["device", "circuit", "validation"]) {
        await page.getByTestId(`tab-${tab}`).click();
        await expect(page.getByTestId(`main-${tab}`)).toBeVisible();
        if (tab === "circuit") {
          await expect(page.getByTestId("schematic-view")).toBeVisible();
          for (const kind of ["MOS", "D", "BJT"]) await expect(page.getByTestId(`tool-${kind}`)).toBeVisible();
          await expect(page.getByTestId("circuit-view-benches")).toHaveCount(0);
          if (theme === "light") {
            await page.getByTestId("menu-examples").click();
            await page.getByTestId("tpl-cmos_inverter").click();
            await expect(page.getByTestId("sch-toast")).toBeHidden();
          }
        }
        if (tab === "validation") await expect(page.getByTestId("panel-val-iv").locator(".js-plotly-plot")).toBeVisible();
        await noHorizontalOverflow(page);
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({ path: `${SHOTS}/${width === 1440 ? "desktop" : "mobile"}-${tab === "validation" ? "reference" : tab}-${theme}.png`, fullPage: true });
      }
    }
  });
}

test("authentic recorded device sweep is clearly labeled as a snapshot", async ({ page }) => {
  const fixture = JSON.parse(readFileSync("src/devices/fixtures/export-reference.json", "utf8"));
  const payload = { device: fixture.selection.device, sweep: fixture.selection.sweep };
  const data = JSON.stringify(fixture.result);
  const index = {
    format: 1, created: "2026-09-25T00:00:00Z", source: "Recorded local analytical engine result", data: {},
    health: { ok: true, workers: 0, version: "recorded-reference" },
    compute: [{ kind: "branches", key: snapshotKey("branches", payload), file: "reference-branches.json", bytes: Buffer.byteLength(data), label: "Reference device" }],
  };
  await page.addInitScript(selection => {
    localStorage.clear();
    localStorage.setItem("stl-websim:v1", JSON.stringify({ v: 2, mode: "deterministic", lang: "ko", theme: "light", params: { device: selection.device, sweep: selection.sweep } }));
  }, fixture.selection);
  await page.route(/\/\/[^/]+\/api\//, route => route.abort());
  await page.route("**/snapshot/index.json", route => route.fulfill({ json: index }));
  await page.route("**/snapshot/reference-branches.json", route => route.fulfill({ contentType: "application/json", body: data }));
  await page.goto("/#tab=device&mode=deterministic");
  const graph = page.getByTestId("panel-iv").locator(".js-plotly-plot");
  await expect(graph).toBeVisible();
  await expect(page.getByTestId("snapshot-banner")).toBeVisible();
  const actual = await graph.evaluate(el => (el as unknown as { data: { mode: string; y: number[] }[] }).data.filter(tr => tr.mode === "lines").map(tr => tr.y));
  expect(actual[0].filter(v => v !== null)).toEqual(fixture.result.double_sweep.up.id.filter((v: number) => v > 0));
  await noHorizontalOverflow(page);
  await page.screenshot({ path: `${SHOTS}/desktop-device-recorded.png`, fullPage: true });
});
