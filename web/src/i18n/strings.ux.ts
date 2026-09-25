// UI strings of the shared layout building blocks (LayoutToggle, MoreCard, Panel ⋯ menu / footnote).
// Not merged into STRINGS: render with `t.l(UX["…"])`, and interpolate with `fill(t.l(UX["…"]), { n })`.
// Work-package string files (strings.device.ts, strings.guide.ts) follow the same pattern.
import type { L10n } from "../content/physics/types";

export const UX = {
  // ---------------------------------------------------------------- layout toggle (간단히 | 모두 보기)
  "layout.label": { ko: "화면 구성", en: "Layout" },
  "layout.simple": { ko: "간단히", en: "Simple" },
  "layout.all": { ko: "모두 보기", en: "Show all" },
  "layout.simple.title": {
    ko: "핵심 결과와 그래프 2개만 보여 줍니다. 나머지 그래프와 설정은 탭과 ⋯ 메뉴에서 한 번에 열 수 있습니다.",
    en: "Shows the key numbers and two plots. The other plots and settings are one click away in tabs and ⋯ menus.",
  },
  "layout.all.title": { ko: "모든 그래프, 표, 설정을 한 화면에 펼칩니다.", en: "Shows every plot, table and setting at once." },

  // ---------------------------------------------------------------- analysis card (MoreCard)
  "more.tabs": { ko: "분석 그래프", en: "Analysis views" },
  "more.select": { ko: "분석 그래프 선택", en: "Choose an analysis view" },
  // tab status dots (screen-reader text and tooltip; the dot itself is aria-hidden)
  "status.running": { ko: "계산 중", en: "running" },
  "status.error": { ko: "오류", en: "error" },
  "status.stale": { ko: "변경됨", en: "changed" },

  // ---------------------------------------------------------------- panel ⋯ menu, footnote, placeholder
  "menu.open": { ko: "보기 · 내보내기 옵션", en: "View and export options" },
  "menu.view": { ko: "보기", en: "View" },
  "menu.export": { ko: "내보내기", en: "Export" },
  "menu.csv": { ko: "CSV 저장", en: "Save CSV" },
  "menu.png": { ko: "PNG 저장", en: "Save PNG" },
  "badge.changed": { ko: "변경됨", en: "Changed" },
  "panel.placeholder": { ko: "계산을 실행하면 여기에 표시됩니다.", en: "Run a calculation to see this." },
  "close": { ko: "닫기", en: "Close" },
} satisfies Record<string, L10n>;

export type UxKey = keyof typeof UX;

/** `{name}` interpolation, the same rule as `t(key, vars)`: fill("{n}개 더", { n: 3 }) → "3개 더". */
export function fill(s: string, vars: Record<string, string | number>): string {
  let out = s;
  for (const [k, v] of Object.entries(vars)) out = out.split(`{${k}}`).join(String(v));
  return out;
}
