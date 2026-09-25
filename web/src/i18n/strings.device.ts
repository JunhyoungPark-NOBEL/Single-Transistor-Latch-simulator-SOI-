// UI strings of the Device tab results (answer bar, hero I–V, analysis card, first-visit hint). Not merged into
// STRINGS: render with `t.l(DEV["…"])` and interpolate with `fill()` from strings.ux.ts.
import type { L10n } from "../content/physics/types";

export const DEV = {
  // ---------------------------------------------------------------- first-visit hint (GettingStarted)
  "gs.label": { ko: "처음 사용 안내", en: "Getting started" },
  "gs.1": { ko: "왼쪽에서 V_G·빛을 바꾸면", en: "Change V_G or the light on the left," },
  "gs.2": { ko: "자동으로 다시 계산되고", en: "the model re-runs by itself," },
  "gs.3": { ko: "얼마나 변했는지 ▲▼로 보여 줍니다", en: "and ▲▼ show how much the answer moved" },
  "gs.ok": { ko: "알겠어요", en: "Got it" },

  // ---------------------------------------------------------------- answer bar (3 cells)
  "kpi.label": { ko: "핵심 결과", en: "Key results" },
  "kpi.vlu": { ko: "켜지는 전압", en: "Turn-on" },
  "kpi.vld": { ko: "꺼지는 전압", en: "Turn-off" },
  "kpi.window": { ko: "기억 창", en: "Window" },
  "kpi.mean": { ko: "평균", en: "mean" },
  "kpi.meaning.vlu": { ko: "올라갈 때 켜짐 (HRS→LRS)", en: "switches on going up (HRS→LRS)" },
  "kpi.meaning.vld": { ko: "내려올 때 꺼짐 (LRS→HRS)", en: "switches off coming down (LRS→HRS)" },
  "kpi.meaning.window": { ko: "V_LU − V_LD", en: "V_LU − V_LD" },
  "kpi.delta": { ko: "{d} · 이전 대비", en: "{d} vs previous" },
  "kpi.delta.aria": { ko: "이전 계산 대비 {d}", en: "{d} compared with the previous run" },
  "kpi.nochange": { ko: "변화 없음", en: "no change" },
  "kpi.nolatch": { ko: "래치 없음", en: "No latch" },
  "kpi.nolatch.hint": { ko: "V_{D,max}를 올리거나 V_G를 더 음(−)으로", en: "raise V_{D,max} or make V_G more negative" },
  "kpi.nolatch.window": { ko: "래치 창: V_G {lo} … {hi} V", en: "latch window: V_G {lo} … {hi} V" },
  "kpi.nolatch.vdmax": { ko: "V_{D,max}를 올려 보세요", en: "try a higher V_{D,max}" },
  "kpi.measured": { ko: "측정 {v}", en: "measured {v}" },
  "kpi.fold": { ko: "fold {v}", en: "fold {v}" },
  "kpi.tip.runtime": { ko: "계산 {t}", en: "computed in {t}" },
  "kpi.tip.cached": { ko: "캐시", en: "cached" },
  "kpi.tip.noLu": { ko: "래치업 안 된 사이클 {n}", en: "{n} cycles without latch-up" },

  // ---------------------------------------------------------------- panel titles (simple layout)
  "iv.title": { ko: "I–V 곡선", en: "I–V curve" },
  "iv.desc": {
    ko: "파란 선을 따라 올라가다 V_LU에서 켜지고, 빨간 선을 따라 내려오다 V_LD에서 꺼집니다.",
    en: "Going up along the blue line the device switches on at V_LU; coming down along the red line it switches off at V_LD.",
  },
  "hazard.title": { ko: "켜질 확률 (hazard)", en: "Turn-on probability (hazard)" },
  "cycles.title": { ko: "사이클마다 변화", en: "Cycle by cycle" },

  // ---------------------------------------------------------------- analysis-card tab labels
  "tab.vg": { ko: "V_G 의존성", en: "V_G dependence" },
  "tab.components": { ko: "전류 성분", en: "Currents" },
  "tab.cb": { ko: "바디 전하 균형", en: "Charge balance" },
  "tab.dist": { ko: "분포", en: "Distribution" },
  "tab.hazard": { ko: "켜질 확률", en: "Hazard" },
  "tab.cycles": { ko: "사이클", en: "Cycles" },
  "tab.vgs": { ko: "V_G (확률)", en: "V_G (MC)" },
  "tab.dmap": { ko: "설계 지도", en: "Design map" },

  // ---------------------------------------------------------------- plot legends and direct labels
  "leg.hrs": { ko: "HRS", en: "HRS" },
  "leg.lrs": { ko: "LRS", en: "LRS" },
  "leg.meas": { ko: "측정", en: "Measured" },
  "leg.measBand": { ko: "측정 10–90 %", en: "Measured 10–90 %" },
  "leg.prev": { ko: "이전", en: "Previous" },
  "leg.unstable": { ko: "불안정", en: "unstable" },
  "leg.full": { ko: "정상상태 곡선", en: "steady states" },
  "leg.traces": { ko: "MC 궤적 {n}개", en: "{n} MC traces" },
  "leg.up": { ko: "상향 스윕", en: "Up sweep" },
  "leg.down": { ko: "하향 스윕", en: "Down sweep" },
  "ann.jumpUp": { ko: "켜짐", en: "on" },
  "ann.jumpDown": { ko: "꺼짐", en: "off" },

  // ---------------------------------------------------------------- ⋯ menu items
  "menu.y": { ko: "y축", en: "y-axis" },
  "menu.x": { ko: "x축", en: "x-axis" },
  "menu.sweeps": { ko: "상향/하향 스윕", en: "Up/down sweep" },
  "menu.meas": { ko: "측정", en: "Measured" },
  "menu.band": { ko: "측정 10–90 % 띠", en: "Measured 10–90 % band" },
  "menu.prev": { ko: "이전 결과", en: "Previous result" },
  "menu.series": { ko: "표시할 전압", en: "Show" },
  "menu.cbAuto": { ko: "V_D 자동 (두 fold 사이)", en: "Auto V_D (between the folds)" },
  "menu.traces": { ko: "MC 궤적", en: "MC traces" },

  // ---------------------------------------------------------------- I–V click → charge balance at that V_D
  "cbOffer.label": { ko: "V_D = {v} V", en: "V_D = {v} V" },
  "cbOffer.go": { ko: "이 V_D에서 전하 균형 보기", en: "Charge balance at this V_D" },
  "cbOffer.aria": { ko: "I–V 곡선에서 고른 V_D", en: "V_D picked on the I–V curve" },

  // ---------------------------------------------------------------- footnotes
  "foot.ifold": { ko: "fold 전류 I_LU {lu} · I_LD {ld}", en: "fold currents I_LU {lu} · I_LD {ld}" },
  "foot.nolatch": { ko: "이 조건에서는 래치가 생기지 않습니다 (fold 쌍 없음).", en: "No latch at this condition (no pair of folds)." },
  "foot.components": {
    ko: "큰 성분 4개만 그렸습니다. 범례를 누르면 나머지도 켜집니다.",
    en: "The 4 largest components are drawn; click a legend entry to add the others.",
  },
  "foot.range": { ko: "범위 {min} … {max} V · {n}점", en: "range {min} … {max} V · {n} points" },
  "foot.engine": { ko: "{engine} · 저장 궤적 {n}개", en: "{engine} · {n} stored traces" },
  "foot.detFold": { ko: "점선 = 결정론 fold", en: "dotted = deterministic fold" },
  "foot.recompute": { ko: "다시 계산", en: "Recompute" },
  "foot.staleVgs": { ko: "파라미터가 바뀌었습니다.", en: "Parameters changed." },

  // ---------------------------------------------------------------- engine names (readable, never the raw id)
  "engine.calibrated_lookup": { ko: "보정 조회표 엔진", en: "calibrated lookup table" },
  "engine.general": { ko: "일반 엔진", en: "general engine" },

  // ---------------------------------------------------------------- V_G range popover (⚙)
  "range.button": { ko: "범위", en: "Range" },
  "range.title": { ko: "V_G 범위", en: "V_G range" },
  "range.min": { ko: "최소 (V)", en: "min (V)" },
  "range.max": { ko: "최대 (V)", en: "max (V)" },
  "range.n": { ko: "점 수", en: "points" },
  "range.run": { ko: "다시 계산", en: "Recompute" },
  "range.compute": { ko: "계산", en: "Compute" },

  // ---------------------------------------------------------------- empty states
  "empty.runBtn": { ko: "계산 실행", en: "Run" },
  "empty.runHint": { ko: "또는 Ctrl/⌘ + Enter", en: "or Ctrl/⌘ + Enter" },
} satisfies Record<string, L10n>;

export type DevKey = keyof typeof DEV;
