// UI strings of the parameter guide (inline line, ⓘ popover, Details-window "한눈에" block, Physics-tab
// guide list) and of the simplified sidebar (고급 설정, 고급 항목, device card, run bar).
// Not merged into STRINGS: render with `t.l(GUIDE["…"])`, interpolate with `fill(t.l(GUIDE["…"]), { n })`.
// The guide texts themselves come from content/params/guide.ts (PARAM_GUIDE, GUIDE_LEGEND).
import type { L10n } from "../content/physics/types";

export const GUIDE = {
  // ---------------------------------------------------------------- guide blocks
  // lead of the effect lines, by field kind (number / switch / choice)
  "guide.raise": { ko: "키우면", en: "If raised" },
  "guide.on": { ko: "켜면", en: "If on" },
  "guide.change": { ko: "바꾸면", en: "If changed" },
  "guide.flat": { ko: "그대로", en: "unchanged" },
  "guide.mean": { ko: "평균", en: "mean" },
  "guide.caveat": { ko: "주의", en: "Caution" },
  "guide.caveat.more": { ko: "더 보기", en: "More" },
  "guide.caveat.less": { ko: "접기", en: "Less" },
  "guide.basis": { ko: "근거", en: "Basis" },
  "guide.legend.short": {
    ko: "화살표는 이 값을 키울 때의 변화, 괄호 안은 키운 폭입니다 (기준 보정: V_G = −2 V, 암조건).",
    en: "Arrows show what happens when you raise the value; the step is in parentheses (reference calibration: V_G = −2 V, dark).",
  },
  "guide.tech": { ko: "정의 · 기본값", en: "Definition · default" },
  // documentation link at the foot of the ⓘ popover
  "guide.doc": { ko: "상세 문서", en: "Documentation" },
  // after the inline chips when another preset is active: the numbers are those of the reference calibration
  "guide.refTag": { ko: "기준 보정 수치", en: "reference numbers" },
  "guide.inline.title": { ko: "눌러서 자세히 보기", en: "Click for the full explanation" },
  "guide.close": { ko: "설명 닫기", en: "Close explanation" },
  // Details window + Physics tab
  "guide.glance": { ko: "한눈에", en: "At a glance" },
  "guide.glance.title": { ko: "한눈에 · 키우면 V_LU·V_LD는?", en: "At a glance: effect on V_LU and V_LD" },
  "guide.list.title": { ko: "파라미터 가이드", en: "Parameter guide" },
  "guide.list.jump": { ko: "파라미터 가이드", en: "Parameter guide" },
  "guide.list.heading": { ko: "파라미터 가이드 · 키우면 V_LU·V_LD는?", en: "Parameter guide · what raising each value does to V_LU and V_LD" },
  "guide.list.lead": {
    ko: "사이드바와 같은 순서입니다. 각 줄의 왼쪽은 쉬운 설명, 오른쪽은 값을 키웠을 때 V_LU·V_LD의 변화입니다.",
    en: "Same order as the sidebar. Each row gives a plain explanation on the left and what raising the value does to V_LU and V_LD on the right.",
  },
  "guide.list.none": { ko: "검색어와 일치하는 파라미터가 없습니다", en: "No parameter matches the search" },
  "guide.list.circuit": { ko: "회로 탭", en: "Circuit tab" },
  "guide.list.sto": { ko: "확률 모드", en: "Stochastic mode" },
  "guide.main": { ko: "주요", en: "main" },

  // ---------------------------------------------------------------- sidebar tiers
  "adv.title": { ko: "고급 설정", en: "Advanced settings" },
  "adv.changed": { ko: "{n}개 수정", en: "{n} changed" },
  "adv.short.state": { ko: "국소 상태", en: "Local state" },
  "adv.short.calib": { ko: "보정", en: "Calibration" },
  "adv.short.ext": { ko: "확장", en: "Extensions" },
  "adv.short.numerics": { ko: "수치", en: "Numerics" },
  "adv.short.solver": { ko: "솔버", en: "Solver" },
  "adv.fields": { ko: "고급 항목 {n}개", en: "{n} more settings" },
  "group.defaults": { ko: "기본값", en: "default" },
  "group.changed.title": { ko: "기본값에서 바꾼 항목 수", en: "Values changed from the defaults" },
  "group.sto": { ko: "확률 모드 전용", en: "Stochastic mode only" },
  "light.dark": { ko: "암조건", en: "dark" },

  // ---------------------------------------------------------------- Details window
  "pw.codeRef": { ko: "코드 대응", en: "Code reference" },
  "pw.tags": { ko: "태그", en: "Tags" },
} satisfies Record<string, L10n>;

export type GuideKey = keyof typeof GUIDE;
