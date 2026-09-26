// UI strings owned by the stats work package (merged into STRINGS in strings.ts).
// Keys must be prefixed "stats." to avoid collisions. Natural Korean and natural English.
import type { L10n } from "../content/physics/types";

export const STATS_STRINGS = {
  // ---------------------------------------------------------------- statistics panel (device tab, stochastic)
  "stats.panel.title": { ko: "통계 요약", en: "Statistics" },
  "stats.panel.desc": {
    ko: "사이클마다 얻은 V_LU, V_LD와 창(V_LU − V_LD)의 기술통계입니다. 측정 기록이 있으면 바로 아래 줄에 함께 보여 줍니다.",
    en: "Descriptive statistics of the per-cycle V_LU, V_LD and window (V_LU − V_LD). When a measured record exists, it is shown right below for comparison.",
  },
  "stats.empty": { ko: "확률적 모드로 실행하면 여기에 통계가 표시됩니다.", en: "Run in Stochastic mode to see the statistics here." },
  "stats.meta.engine": { ko: "엔진", en: "Engine" },
  "stats.meta.cycles": { ko: "{n}회 사이클", en: "{n} cycles" },
  "stats.meta.seed": { ko: "시드", en: "Seed" },
  "stats.meta.sweep": { ko: "스윕 0 → {v} V · {rate} V/s", en: "Sweep 0 → {v} V · {rate} V/s" },
  "stats.meta.latched": { ko: "스윕 안에서 래치업 {pct}", en: "Latched up within the sweep: {pct}" },
  "stats.meta.censored": { ko: "중도절단 {n}회 ({pct})", en: "Censored: {n} ({pct})" },
  "stats.meta.ci": { ko: "평균 V_LU의 95 % 신뢰구간 {lo} … {hi} V", en: "95 % CI of mean V_LU: {lo} … {hi} V" },
  "stats.engine.calibrated_lookup": { ko: "보정 조회표 엔진", en: "calibrated lookup table" },
  "stats.engine.general": { ko: "일반 엔진", en: "general engine" },

  // ---------------------------------------------------------------- one-line summary (stats-meta) + disclosure
  "stats.line.title": { ko: "통계", en: "Statistics" },
  "stats.line.cycles": { ko: "{n} 사이클", en: "{n} cycles" },
  // plain words here; the table and its tooltips keep the technical term (중도절단 / censored)
  "stats.line.noLatch": { ko: "래치업 안 된 사이클(중도절단) {n} ({pct})", en: "no latch-up in {n} cycles ({pct})" },
  "stats.line.noLatch.tip": {
    ko: "스윕 최대 전압까지 래치업하지 않은 사이클입니다. 평균과 분포는 래치업한 사이클만으로 계산합니다 (표의 ‘중도절단’).",
    en: "Cycles that did not latch up before the sweep maximum. Means and distributions use only the cycles that did (‘censored’ in the table).",
  },
  "stats.line.ks": { ko: "측정과 분포 비교 KS p: {p}", en: "KS p vs measured: {p}" },
  "stats.line.ks.tip": { ko: "p < 0.05이면 모델 분포가 측정과 통계적으로 다릅니다", en: "p < 0.05: the model distribution differs significantly from the measurement" },
  "stats.line.seed": { ko: "시드 {s}", en: "Seed {s}" },
  "stats.line.running": { ko: "계산 중…", en: "Computing…" },
  "stats.line.empty": { ko: "계산을 실행하면 사이클 통계가 여기에 한 줄로 요약됩니다.", en: "Run the model to see a one-line summary of the cycle statistics here." },
  "stats.expand": { ko: "표 보기", en: "Show table" },
  "stats.collapse": { ko: "표 접기", en: "Hide table" },
  "stats.cols.all": { ko: "모든 통계 열 ({n})", en: "All columns ({n})" },
  "stats.cols.fewer": { ko: "주요 열만", en: "Key columns only" },

  // ---------------------------------------------------------------- compact table headers (one header row)
  "stats.cc.sd": { ko: "σ", en: "σ" },
  "stats.cc.dmean": { ko: "Δ측정", en: "Δmeas." },
  "stats.cc.ks_p": { ko: "KS p", en: "KS p" },

  "stats.row.vlu": { ko: "래치업", en: "latch-up" },
  "stats.row.vld": { ko: "래치다운", en: "latch-down" },
  "stats.row.window": { ko: "창 (사이클별)", en: "window (per cycle)" },
  "stats.row.window.tip": {
    ko: "각 사이클의 V_LU − V_LD입니다. 두 전이가 모두 관측된 사이클만 셉니다. 평균끼리 뺀 값과 달리 사이클별 히스테리시스 창의 산포를 보여 줍니다.",
    en: "V_LU − V_LD of each cycle, counting only cycles where both transitions were observed. Unlike the difference of the means, it shows how the hysteresis window itself varies from cycle to cycle.",
  },
  "stats.row.hazard": { ko: "캐리어 잡음만", en: "carrier noise only" },
  "stats.row.hazard.tip": {
    ko: "중심 국소 상태에서 켜짐률 h(V_D)로 계산한 V_LU 분포입니다. 캐리어(첫 통과) 잡음만 들어 있고 국소 상태 산포와 중도절단은 없습니다. MC 행과의 차이가 곧 국소 상태가 더하는 산포입니다.",
    en: "V_LU distribution from the hazard h(V_D) at the centre local state: carrier (first-passage) noise only, no state spread and no censoring. The difference from the MC row is the spread added by the local states.",
  },
  "stats.row.hazard.atom": { ko: "폴드 도달 {pct}", en: "{pct} reach the fold" },
  "stats.row.model": { ko: "모델", en: "model" },
  "stats.row.measured": { ko: "측정", en: "measured" },
  "stats.row.analytic": { ko: "해석 분포", en: "analytic" },

  // ---------------------------------------------------------------- column groups
  "stats.g.centre": { ko: "중심", en: "Centre" },
  "stats.g.spread": { ko: "퍼짐", en: "Spread" },
  "stats.g.tails": { ko: "꼬리", en: "Tails" },
  "stats.g.shape": { ko: "모양", en: "Shape" },
  "stats.g.sequence": { ko: "순서", en: "Order" },
  "stats.g.counts": { ko: "개수", en: "Counts" },
  "stats.g.compare": { ko: "측정과 비교", en: "vs measured" },
  "stats.quantity": { ko: "항목", en: "Quantity" },

  // ---------------------------------------------------------------- columns (short headers)
  "stats.c.mean": { ko: "평균", en: "mean" },
  "stats.c.ci95": { ko: "95% CI", en: "95% CI" },
  "stats.c.median": { ko: "중앙값", en: "median" },
  "stats.c.sd": { ko: "표준편차", en: "SD" },
  "stats.c.iqr": { ko: "IQR", en: "IQR" },
  "stats.c.cv": { ko: "CV", en: "CV" },
  "stats.c.p05": { ko: "p5", en: "p5" },
  "stats.c.p95": { ko: "p95", en: "p95" },
  "stats.c.min": { ko: "최소", en: "min" },
  "stats.c.max": { ko: "최대", en: "max" },
  "stats.c.skew": { ko: "왜도", en: "skew" },
  "stats.c.kurt": { ko: "첨도", en: "kurt." },
  "stats.c.lag1": { ko: "lag-1", en: "lag-1" },
  "stats.c.n": { ko: "n", en: "n" },
  "stats.c.censored": { ko: "중도절단", en: "cens." },
  "stats.c.m_mean": { ko: "측정 평균", en: "meas. mean" },
  "stats.c.m_sd": { ko: "측정 SD", en: "meas. SD" },
  "stats.c.dmean": { ko: "평균 차", en: "Δmean" },
  "stats.c.sd_ratio": { ko: "SD 비", en: "SD ratio" },
  "stats.c.ks_d": { ko: "KS D", en: "KS D" },
  "stats.c.ks_p": { ko: "p값", en: "p" },

  // ---------------------------------------------------------------- column tooltips
  "stats.tip.mean": {
    ko: "평균 — 값이 있는 사이클의 산술평균입니다. 스윕 안에서 전이가 없어 값이 없는(중도절단된) 사이클은 빠집니다.",
    en: "Mean — the arithmetic average over cycles that have a value. Censored cycles (no transition within the sweep) are left out.",
  },
  "stats.tip.ci95": {
    ko: "평균의 95 % 신뢰구간 반폭 — ± t(0.975, n−1) · SD/√n (t 분포). 사이클을 늘리면 좁아집니다. 사이클 사이 상관(lag-1)이 크면 실제 불확실성은 이보다 큽니다.",
    en: "Half-width of the 95 % confidence interval of the mean — ± t(0.975, n−1) · SD/√n (Student's t). It narrows with more cycles; with strong cycle-to-cycle correlation (lag-1) the true uncertainty is larger.",
  },
  "stats.tip.median": {
    ko: "중앙값 — 사이클의 절반이 이 값보다 낮습니다. 긴 꼬리나 튀는 값에 평균보다 덜 흔들립니다.",
    en: "Median — half of the cycles lie below this value. Less affected by long tails and outliers than the mean.",
  },
  "stats.tip.sd": {
    ko: "표준편차 — n−1로 나눈 표본 표준편차로, 사이클마다 값이 얼마나 흩어지는지를 나타냅니다.",
    en: "Standard deviation — the sample SD (divided by n − 1): how much the value scatters from cycle to cycle.",
  },
  "stats.tip.iqr": {
    ko: "사분위 범위(IQR) Q3 − Q1 — 가운데 50 % 사이클이 차지하는 폭입니다. 튀는 값에 강하며, 정규분포라면 SD의 약 1.35배입니다.",
    en: "Interquartile range Q3 − Q1 — the width of the middle 50 % of cycles. Robust to outliers; about 1.35 × SD for a normal distribution.",
  },
  "stats.tip.cv": {
    ko: "변동계수 SD ÷ |평균| — 평균에 견준 상대 산포(%)입니다.",
    en: "Coefficient of variation SD ÷ |mean| — the spread relative to the mean, in %.",
  },
  "stats.tip.p05": { ko: "5번째 백분위수 — 사이클의 5 %가 이 값보다 낮습니다.", en: "5th percentile — 5 % of the cycles lie below this value." },
  "stats.tip.p95": {
    ko: "95번째 백분위수 — 사이클의 95 %가 이 값보다 낮습니다. p5와 p95 사이에 90 %가 들어갑니다.",
    en: "95th percentile — 95 % of the cycles lie below this value; 90 % lie between p5 and p95.",
  },
  "stats.tip.min": { ko: "관측된 가장 작은 값 (중도절단된 사이클 제외).", en: "Smallest observed value (censored cycles excluded)." },
  "stats.tip.max": { ko: "관측된 가장 큰 값 (중도절단된 사이클 제외).", en: "Largest observed value (censored cycles excluded)." },
  "stats.tip.skew": {
    ko: "왜도(표본 보정) — 분포가 얼마나 한쪽으로 치우쳤는지. 0이면 좌우 대칭, 양수면 높은 쪽 꼬리가, 음수면 낮은 쪽 꼬리가 깁니다.",
    en: "Skewness (sample-adjusted) — how lopsided the distribution is. 0 is symmetric; positive means a longer tail towards high values, negative towards low values.",
  },
  "stats.tip.kurt": {
    ko: "초과 첨도(표본 보정) — 정규분포면 0입니다. 양수면 꼬리가 두꺼워 극단적인 사이클이 더 자주 나오고, 음수면 꼬리가 얇고 납작한 분포입니다.",
    en: "Excess kurtosis (sample-adjusted) — 0 for a normal distribution. Positive: heavier tails, extreme cycles are more common; negative: lighter tails, a flatter distribution.",
  },
  "stats.tip.lag1": {
    ko: "자기상관(lag-1) — 연속한 두 사이클 값 사이의 상관계수입니다. 0 근처면 사이클끼리 독립이고, 양수면 한 사이클의 값이 다음 사이클로 이어집니다(천천히 변하는 local state 등). 중도절단된 사이클을 사이에 둔 쌍은 뺍니다.",
    en: "Lag-1 autocorrelation — the correlation between consecutive cycles. Near 0: cycles are independent; positive: one cycle's value carries over to the next (e.g. a slowly varying local state). Pairs that straddle a censored cycle are skipped.",
  },
  "stats.tip.n": { ko: "값이 있는 사이클 수 (스윕 안에서 전이가 관측된 사이클).", en: "Number of cycles with a value (the transition was observed within the sweep)." },
  "stats.tip.censored": {
    ko: "중도절단 — 스윕 범위 안에서 전이가 일어나지 않아 값이 없는 사이클 수와, 전체 사이클에 대한 비율입니다.",
    en: "Censored — cycles without a value because the transition did not happen within the sweep, and their share of all cycles.",
  },
  "stats.tip.m_mean": { ko: "측정 기록의 평균.", en: "Mean of the measured record." },
  "stats.tip.m_sd": { ko: "측정 기록의 표준편차.", en: "Standard deviation of the measured record." },
  "stats.tip.dmean": {
    ko: "모델 평균 − 측정 평균. 양수면 모델이 더 높은 전압에서 전이합니다.",
    en: "Model mean − measured mean. Positive means the model switches at a higher voltage.",
  },
  "stats.tip.sd_ratio": {
    ko: "모델 SD ÷ 측정 SD. 1이면 산포가 같고, 1보다 작으면 모델의 산포가 더 좁습니다.",
    en: "Model SD ÷ measured SD. 1 means equal spread; below 1 the model is narrower.",
  },
  "stats.tip.ks_d": {
    ko: "2표본 Kolmogorov–Smirnov 통계량 D — 모델과 측정의 경험적 누적분포(CDF) 사이에서 가장 큰 세로 간격입니다. 0이면 똑같고, 1이면 전혀 겹치지 않습니다.",
    en: "Two-sample Kolmogorov–Smirnov statistic D — the largest vertical gap between the model and measured empirical CDFs. 0: identical; 1: no overlap at all.",
  },
  "stats.tip.ks_p": {
    ko: "KS 검정의 점근 p값 — 두 표본이 같은 분포에서 나왔다면 이만큼 큰 D가 나올 확률입니다. 0.05보다 작으면 분포가 다르다는 근거가 됩니다. 표본이 크면 작은 차이도 유의하게 나오고, 사이클 사이 상관이 있으면 p값이 실제보다 작게 나옵니다.",
    en: "Asymptotic p-value of the KS test — the chance of a D this large if both samples came from the same distribution. Below 0.05 is evidence that the distributions differ. Large samples make small differences significant, and cycle-to-cycle correlation makes p too small.",
  },

  // ---------------------------------------------------------------- table chrome
  "stats.copy": { ko: "CSV 복사", en: "Copy CSV" },
  "stats.copied": { ko: "복사했습니다", en: "Copied" },
  "stats.copy.fail": { ko: "복사하지 못했습니다", en: "Could not copy" },
  "stats.download": { ko: "CSV 저장", en: "Download CSV" },
  "stats.opt.show": { ko: "{cols} 열 보기", en: "Show {cols}" },
  "stats.opt.hide": { ko: "{cols} 열 숨기기", en: "Hide {cols}" },
  "stats.units": { ko: "값은 {level}, 산포(σ·Δ)는 {spread}", en: "values in {level}, spreads (σ, Δ) in {spread}" },
  "stats.foot.censoring": {
    ko: "통계와 KS 비교는 값이 있는 사이클만 씁니다. 중도절단된 사이클은 개수로만 셉니다.",
    en: "Statistics and the KS comparison use only cycles that have a value; censored cycles are counted but not averaged.",
  },
  "stats.foot.measured": { ko: "측정 기록: {label}", en: "Measured record: {label}" },
  "stats.aria.table": { ko: "기술통계 표", en: "Descriptive statistics table" },
  "stats.meas.caption": { ko: "측정 기록: V_G = {vg} V, {light}, 사이클 {n}회", en: "Measured record: {n} cycles at V_G = {vg} V, {light}" },
  "stats.meas.dark": { ko: "암조건", en: "dark" },

  // ---------------------------------------------------------------- distribution panel (CDF)
  "stats.dist.plateau": {
    ko: "CDF는 모든 사이클 기준이라 스윕 안에서 전이한 비율 {pct}에서 멈춥니다 (나머지는 중도절단).",
    en: "CDF over all cycles: it levels off at the {pct} that switched within the sweep (the rest are censored).",
  },
  "stats.dist.ks": { ko: "모델 vs 측정 KS D = {d}, p = {p}", en: "model vs measured KS D = {d}, p = {p}" },
  "stats.dist.ksHead": { ko: "측정 대비 KS D:", en: "KS D vs measured:" },
  "stats.dist.ksItem": { ko: "{d} (p {p})", en: "{d} (p {p})" },

  // ---------------------------------------------------------------- V_G curve (stochastic) censoring
  "stats.vgs.censored": { ko: "중도절단 비율", en: "Censored" },
  "stats.vgs.noLatch": { ko: "래치 없음 (폴드 없음)", en: "no latch (no fold)" },
  "stats.vgs.beyond": { ko: "스윕 최대 {v} V 너머", en: "beyond the {v} V sweep maximum" },
  "stats.vgs.axis": { ko: "중도절단 (%)", en: "censored (%)" },
  "stats.vgs.foot": {
    ko: "평균과 σ는 스윕 안에서 래치업하는 사이클만으로 계산합니다. 중도절단 비율은 V_G = {vg} V에서 최대 {max}입니다 (래치 없음 {nl}, 스윕 너머 {bs}).",
    en: "Mean and σ are over the cycles that latch up within the sweep. The censored share peaks at {max} at V_G = {vg} V (no latch {nl}, beyond the sweep {bs}).",
  },
  "stats.vgs.masked": {
    ko: "중도절단이 50 %를 넘는 V_G는 평균·σ가 스윕 한계에 눌려 작게 나오므로 비워 둡니다.",
    en: "V_G values with more than 50 % censored are left blank: their mean and σ would be squeezed by the sweep limit.",
  },

  // ---------------------------------------------------------------- validation (8 conditions)
  "stats.val.title": { ko: "조건별 분포 비교", en: "Per-condition comparison" },
  "stats.val.desc": {
    ko: "조건마다 모델 V_LU와 측정 V_LU(각 400 사이클)의 평균·표준편차와 KS 검정입니다.",
    en: "Model vs measured V_LU for each condition (400 cycles each): mean, SD and the KS test.",
  },
  "stats.val.cond": { ko: "V_G {vg} V · {p} mW", en: "V_G {vg} V · {p} mW" },
} satisfies Record<string, L10n>;
