// UI strings of the Device tab results (answer bar, hero I–V, analysis card, first-visit hint). Not merged into
// STRINGS: render with `t.l(DEV["…"])` and interpolate with `fill()` from strings.ux.ts.
import type { L10n } from "../content/physics/types";

export const DEV = {
  // ---------------------------------------------------------------- first-visit hint (GettingStarted)
  "gs.label": { ko: "처음 사용 안내", en: "Getting started" },
  "gs.1": { ko: "왼쪽에서 V_G·빛을 바꾸면", en: "Change V_G or the light on the left," },
  // ≤ 1100 px: the parameters live in the ☰ drawer, not on the left
  "gs.1.drawer": { ko: "☰ 파라미터를 열어 V_G·빛을 바꾸면", en: "Open ☰ Parameters and change V_G or the light," },
  "gs.2": { ko: "자동으로 다시 계산되고", en: "the model re-runs by itself," },
  "gs.3": { ko: "얼마나 변했는지 ▲▼로 보여 줍니다", en: "and ▲▼ show how much the answer moved" },
  "gs.ok": { ko: "알겠어요", en: "Got it" },

  // ---------------------------------------------------------------- answer bar (3 cells)
  "kpi.label": { ko: "핵심 결과", en: "Key results" },
  // the glossary names (래치업/래치다운, 히스테리시스 창) everywhere; the sub-line gives the plain meaning
  "kpi.vlu": { ko: "래치업 전압", en: "Latch-up" },
  "kpi.vld": { ko: "래치다운 전압", en: "Latch-down" },
  "kpi.window": { ko: "히스테리시스 창", en: "Hysteresis window" },
  "kpi.mean": { ko: "평균", en: "mean" },
  "kpi.meaning.vlu": { ko: "올라갈 때 켜짐 (HRS→LRS)", en: "switches on going up (HRS→LRS)" },
  "kpi.meaning.vld": { ko: "내려올 때 꺼짐 (LRS→HRS)", en: "switches off coming down (LRS→HRS)" },
  "kpi.meaning.window": { ko: "V_LU − V_LD", en: "V_LU − V_LD" },
  "kpi.delta": { ko: "{d} · 이전 대비", en: "{d} vs previous" },
  "kpi.delta.aria": { ko: "이전 계산 대비 {d}", en: "{d} compared with the previous run" },
  "kpi.nochange": { ko: "변화 없음", en: "no change" },
  "kpi.nolatch": { ko: "래치 없음", en: "No latch" },
  "kpi.nolatch.hint": { ko: "V_{D,max}를 올리거나 V_G를 더 음(−)으로", en: "raise V_{D,max} or make V_G more negative" },
  "kpi.nolatch.window": { ko: "래치 가능 범위: V_G {lo} … {hi} V", en: "latching range: V_G {lo} … {hi} V" },
  // the fold exists but lies above the sweep peak: the quasi-static sweep never latches
  // the verdict first: the sub-line is one line on desktop and may be cut at its end
  "kpi.beyondVdmax": { ko: "래치업 안 됨: 스윕 최대 V_{D,max} {v} V보다 높음", en: "never latches: above V_{D,max} = {v} V" },
  "kpi.beyondVdmax.hint": { ko: "V_{D,max}를 올리면 래치업됩니다", en: "raise V_{D,max} to latch" },
  "kpi.nolatch.vdmax": { ko: "V_{D,max}를 올려 보세요", en: "try a higher V_{D,max}" },
  "kpi.measured": { ko: "측정 {v}", en: "measured {v}" },
  "kpi.fold": { ko: "평균 모델 {v}", en: "mean model {v}" },
  "kpi.tip.runtime": { ko: "계산 {t}", en: "computed in {t}" },
  "kpi.tip.cached": { ko: "캐시", en: "cached" },
  "kpi.tip.noLu": { ko: "래치업 안 된 사이클 {n}", en: "{n} cycles without latch-up" },

  // ---------------------------------------------------------------- panel titles (simple layout)
  "iv.title": { ko: "I–V 곡선", en: "I–V curve" },
  "iv.desc": {
    ko: "파란 선을 따라 올라가다 V_LU에서 켜지고, 빨간 선을 따라 내려오다 V_LD에서 꺼집니다.",
    en: "Going up along the blue line the device switches on at V_LU; coming down along the red line it switches off at V_LD.",
  },
  "hazard.title": { ko: "켜질 확률 (켜짐률 h · 생존 S)", en: "Turn-on hazard h and survival S" },
  // the hazard is carrier noise only, at the centre local state: its σ is much smaller than the MC answer's ±
  "hazard.foot": {
    ko: "캐리어 잡음만 넣은 첫 통과 계산입니다 (중심 국소 상태, 국소 상태 요동 제외). 그래서 σ {sd} mV가 위 답의 ± 값보다 작습니다. 사이클 간 전체 퍼짐은 분포 탭을 보세요.",
    en: "First passage with carrier noise only (centre local state, no local-state spread), so σ {sd} mV is smaller than the ± above. The full cycle-to-cycle spread is in the Distribution tab.",
  },
  "cycles.title": { ko: "사이클마다 변화", en: "Cycle by cycle" },

  // ---------------------------------------------------------------- analysis-card tab labels
  "tab.vg": { ko: "V_G 의존성", en: "V_G dependence" },
  "tab.components": { ko: "전류 성분", en: "Currents" },
  "tab.cb": { ko: "바디 전하 균형", en: "Charge balance" },
  "tab.dist": { ko: "분포", en: "Distribution" },
  // EN tab kept short so the five stochastic tabs fit one row at 1440 px ("Turn-on hazard h" is the panel title)
  "tab.hazard": { ko: "켜질 확률", en: "Hazard" },
  "tab.cycles": { ko: "사이클", en: "Cycles" },
  "tab.vgs": { ko: "V_G (확률)", en: "V_G (MC)" },
  // phone <select> options cannot show subscripts
  "tab.vg.plain": { ko: "게이트 전압 의존성", en: "Gate-voltage dependence" },
  "tab.vgs.plain": { ko: "게이트 전압 (확률)", en: "Gate voltage (MC)" },
  "tab.dmap": { ko: "설계 지도", en: "Design map" },

  // ---------------------------------------------------------------- plot legends and direct labels
  "leg.hrs": { ko: "HRS (저전류)", en: "HRS (low current)" },
  "leg.lrs": { ko: "LRS (고전류)", en: "LRS (high current)" },
  "leg.meas": { ko: "측정", en: "Measured" },
  "leg.measMedian": { ko: "측정 중앙값 (100회)", en: "Measured median (100 sweeps)" },
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
  "foot.ifold": { ko: "꺾임점(fold) 전류 I_LU {lu} · I_LD {ld}", en: "Fold currents I_LU {lu} · I_LD {ld}" },
  // the measured median switches before the folds: noise-driven early escape, reproduced by the stochastic mode
  "foot.measEarly": { ko: "측정은 잡음 때문에 fold보다 조금 일찍 스위칭합니다 (확률 모드에서 재현됨)", en: "the measured sweeps switch a little before the folds because of noise (the stochastic mode reproduces this)" },
  "foot.vdmax": { ko: "V_{D,max} {v} V", en: "V_{D,max} {v} V" },
  "foot.nolatch": { ko: "이 조건에서는 래치가 생기지 않습니다 (fold 쌍 없음).", en: "No latch at this condition (no pair of folds)." },
  "foot.components": {
    ko: "큰 성분 4개만 그렸습니다. 범례를 누르면 나머지도 켜집니다.",
    en: "The 4 largest components are drawn; click a legend entry to add the others.",
  },
  "foot.range": { ko: "범위 {min} … {max} V · {n}점", en: "range {min} … {max} V · {n} points" },
  "foot.engine": { ko: "{engine} · 저장 궤적 {n}개", en: "{engine} · {n} stored traces" },
  "foot.detFold": { ko: "점선 = 평균 모델 fold", en: "dotted = mean-model fold" },
  "foot.hist.scale": { ko: "모델 사이클 수에 맞춰 ×{s} 조정", en: "rescaled ×{s} to the model cycle count" },
  // V_G curve: which part of the latch range the current sweep peak actually reaches
  "foot.vgReach": { ko: "V_{D,max} {v} V 스윕으로 래치업되는 V_G: {r}", en: "V_G that latch within the {v} V sweep: {r}" },
  "foot.vgReach.none": { ko: "V_{D,max} {v} V 스윕으로는 어느 V_G에서도 래치업 안 됨", en: "no V_G latches within the {v} V sweep" },
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
