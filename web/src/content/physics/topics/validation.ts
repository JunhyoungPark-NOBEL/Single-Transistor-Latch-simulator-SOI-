import type { PhysicsTopic } from "../types";
import { L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "validation",
  title: L("검증 수치", "Validation numbers"),
  summary: L(
    "`engine/docs/VALIDATION.md`의 기준값, 각 값이 검사하는 것, 실행으로 확인한 값을 정리한다. 측정 기준은 광조사 소자 8개 조건(400 cycle, 1200 V/s)과 논문 소자 기록(100 sweep, 0.4 V/s)이다.",
    "Reference values of `engine/docs/VALIDATION.md`, what each one checks and the value obtained by running the code. Measured references are the photo device's 8 conditions (400 cycles, 1200 V/s) and the paper device record (100 sweeps, 0.4 V/s).",
  ),
  tags: ["validation", "measured data"],
  sections: [
    {
      heading: L("결정론 모델 (fold)", "Deterministic model (folds)"),
      body: L(
        "fold는 `classify`가 추적한 정상상태 곡선의 첫 $V_D$ 극대($V_{LU}$)와 마지막 극소($V_{LD}$)에 포물선을 맞춘 꼭짓점이다(→ charge-balance). 값: 기대 / 실행.",
        "Folds are the vertices of parabolas fitted at the first $V_D$ maximum ($V_{LU}$) and the last minimum ($V_{LD}$) of the steady-state locus traced by `classify` (→ charge-balance). Values: expected / run.",
      ),
      variables: [
        { symbol: r`V_{LU},V_{LD}`, name: L("논문 모델, V_G = −2 V 암조건 — 실행: 동일", "paper model, V_G = −2 V dark — run: same"), value: "3.7037, 2.5979", unit: "V", code: "stl_api.folds(-2.0)" },
        { symbol: r`V_{LU},V_{LD}`, name: L("논문 모델, V_G = −1.8 V 암조건 — 실행: 동일", "paper model, V_G = −1.8 V dark — run: same"), value: "3.8644, 2.5979", unit: "V", code: "stl_api.folds(-1.8)" },
        { symbol: r`V_{LU},V_{LD}`, name: L("광 모델, −1.8 V, I_PH = 2.63 pA — 실행: 3.2913, 2.5962", "photo model, −1.8 V, I_PH = 2.63 pA — run: 3.2913, 2.5962"), value: "3.2913, 2.596", unit: "V", code: "stl_api.folds(-1.8, 2.63e-12)" },
        { symbol: r`\Delta V`, name: L("확장 = 0일 때 gate_mean과 차이", "difference to gate_mean with extensions = 0"), value: "≤ 1e-12", unit: "V", code: "photo_mean vs gate_mean" },
        { symbol: r`V_G`, name: L("latch 창 (논문, 고정 상태) — classify 이분법 실행: −3.906 … −0.810 V", "latch window (paper, fixed states) — classify bisection run: −3.906 … −0.810 V"), value: "−3.90 … −0.815", unit: "V", code: "validation.py · check_latch_window" },
        { symbol: r`I_{\mathrm{PH}}/P`, name: L("광 변환: 1.15/2.55/3.51 mW → 0.86/1.91/2.63 pA", "light conversion: 1.15/2.55/3.51 mW → 0.86/1.91/2.63 pA"), value: "0.75", unit: "pA/mW", code: "photo_conversion_fit.json" },
      ],
      notes: [
        L(
          "작은 불일치: 광 모델 $V_{LD}$는 실행값 2.5962 V, VALIDATION.md 2.596 V, `stl_api.py` 출력 문구 2.5959 V.",
          "Small inconsistency: photo-model $V_{LD}$ runs to 2.5962 V; VALIDATION.md says 2.596 V and the `stl_api.py` print text 2.5959 V.",
        ),
      ],
    },
    {
      heading: L("확률 모델", "Stochastic model"),
      variables: [
        { symbol: r`\overline{V_{LU}},\ \mathrm{SD}`, name: L("FPT 노드, −2 V 암조건, 중심 상태, 0.4 V/s (Eq. 2 잡음만) — 실행: 3.6442 V, 8.03 mV", "FPT node, −2 V dark, centre states, 0.4 V/s (Eq. 2 noise only) — run: 3.6442 V, 8.03 mV"), value: "≈3.644, ≈8", unit: "V, mV", code: "stl_api.hazard(-2.0, rate=.4)" },
        { symbol: r`V_{LU}`, name: L("동적 MC 100 sweep, seed 2026092920: 평균, SD — 실행: 3.6344 V, 119.0 mV", "dynamic MC 100 sweeps, seed 2026092920: mean, SD — run: 3.6344 V, 119.0 mV"), value: "≈3.63, ≈120", unit: "V, mV", code: "stl_api.sweeps(n=100)" },
        { symbol: r`V_{LD}`, name: L("같은 실행: 평균, SD — 실행: 2.6999 V, 21.4 mV", "same run: mean, SD — run: 2.6999 V, 21.4 mV"), value: "≈2.70, ≈20", unit: "V, mV", code: "stl_api.sweeps(n=100)" },
        { symbol: r`\sigma_{LU},\ \sigma_{LD}`, name: L("10 기록 × 100 sweep (seed 2026093000–09, 합동) — 실행: 125.8, 19.6 mV", "10 records × 100 sweeps (seeds 2026093000–09, pooled) — run: 125.8, 19.6 mV"), value: "125.8, 19.6", unit: "mV", code: "gate_dynamic_compare.simulate" },
        { symbol: r`\sigma_{\mathrm{carrier}}`, name: L("캐리어 잡음만: II, BTBT, REC, DIFF, 전체 (full 수준)", "carrier noise only: II, BTBT, REC, DIFF, all (full level)"), value: "4.6, 2.7, 4.3, 1.8, 7.8", unit: "mV", code: "validation.py · carrier_noise_breakdown" },
        { symbol: r`\max\sigma_{LU}`, name: L("V_G 곡선: σ 최대 −1.25 V, 평균 최대 4.354 V (−1.10 V)", "V_G curve: σ peak at −1.25 V, mean peak 4.354 V (−1.10 V)"), value: "129.8 (full level)", unit: "mV", code: "validation.py · check_vg_curve_stochastic" },
      ],
      notes: [
        L(
          "FPT 노드는 캐리어 잡음(II 클러스터 + 단위 사건)만 검사하고, 동적 MC는 여기에 국소 상태(OU + 추세)를 더한 전체 모델과 10 mV 판독을 검사한다. 10×100 합동 SD가 측정 123.1/19.5 mV와 비교되는 논문 값이다.",
          "The FPT node checks carrier noise only (II clusters + unit events); the dynamic MC checks the full model with local states (OU + trend) and the 10 mV readout. The pooled 10×100 SDs are the paper values compared with the measured 123.1/19.5 mV.",
        ),
        L(
          "'full level' 항목은 여기서 다시 실행하지 않았다. 캐리어 잡음 분해의 원 스크립트는 인수인계에 없어 검증 탭이 아래 방법으로 재구성한다.",
          "'full level' items were not re-run here. The original carrier-noise breakdown script is not in the handoff; the validation tab reconstructs it as described below.",
        ),
      ],
    },
    {
      heading: L("측정: 논문 소자 기록", "Measured: paper-device record"),
      body: L(
        "V_G = −2 V 암조건, 0 → 4 V → 0, 10 mV 스텝, 0.4 V/s, 100 up + 100 down. $V_{LU}$, $V_{LD}$는 10 mV 괄호(low/high)의 중점이다. 상승·하강은 짝지어지지 않는다.",
        "V_G = −2 V dark, 0 → 4 V → 0, 10 mV steps, 0.4 V/s, 100 up + 100 down. $V_{LU}$, $V_{LD}$ are midpoints of the 10 mV low/high bracket. Up and down records are unpaired.",
      ),
      variables: [
        { symbol: r`V_{LU}`, name: L("평균, SD (lag-1 0.68)", "mean, SD (lag-1 0.68)"), value: "3.6455 V, 123.1 mV", code: "measured_idvd_parsed.npz" },
        { symbol: r`V_{LD}`, name: L("평균, SD (lag-1 0.17)", "mean, SD (lag-1 0.17)"), value: "2.7001 V, 19.5 mV", code: "measured_idvd_parsed.npz" },
        { symbol: r`f_{tr}`, name: L("2차 획득 추세가 설명하는 V_LU 분산", "V_LU variance explained by the quadratic acquisition trend"), value: "0.660" },
      ],
      notes: [
        L(
          "이 주제의 측정 lag-1은 `measured_stats.json`·데이터 탭의 추정량 $\\sum(x_t-\\bar x)(x_{t+1}-\\bar x)/\\sum(x_t-\\bar x)^2$이다. 스윕 MC 결과의 `Stats.lag1`(측정 overlay 포함)은 연속한 유한 쌍의 Pearson 상관이라 값이 약간 다르다: 논문 기록 $V_{LU}$ 0.70, $V_{LD}$ 0.17; 광조사 소자 −1.8 V 2.55 mW 0.24, −1.1 V 3.51 mW 0.50(나머지는 같음).",
          "Measured lag-1 values in this topic use the estimator of `measured_stats.json` and the Data tab, $\\sum(x_t-\\bar x)(x_{t+1}-\\bar x)/\\sum(x_t-\\bar x)^2$. `Stats.lag1` of sweep-MC results (including the measured overlay) is the Pearson correlation of consecutive finite pairs and differs slightly: paper record $V_{LU}$ 0.70, $V_{LD}$ 0.17; photo device −1.8 V 2.55 mW 0.24, −1.1 V 3.51 mW 0.50 (others unchanged).",
        ),
      ],
    },
    {
      heading: L("측정: 광조사 소자 8개 조건", "Measured: photo device, 8 conditions"),
      body: L(
        "400 cycle, 삼각 1200 V/s, 0 → 5 V; $I_{\\mathrm{PH}} = 0.75\\,\\mathrm{pA/mW}\\times P$. 값: 평균, SD (`measured_stats.json`, `raw_VLU.npy`).",
        "400 cycles, triangular 1200 V/s, 0 → 5 V; $I_{\\mathrm{PH}} = 0.75\\,\\mathrm{pA/mW}\\times P$. Values: mean, SD (`measured_stats.json`, `raw_VLU.npy`).",
      ),
      variables: [
        { symbol: r`-1.8\,\mathrm{V},\ 0\,\mathrm{mW}`, name: L("암조건 (보정 목표), lag-1 0.20", "dark (calibration target), lag-1 0.20"), value: "3.806 V, 173.2 mV" },
        { symbol: r`-1.8\,\mathrm{V},\ 1.15\,\mathrm{mW}`, name: L("I_PH 0.86 pA, lag-1 0.00", "I_PH 0.86 pA, lag-1 0.00"), value: "3.500 V, 177.4 mV" },
        { symbol: r`-1.8\,\mathrm{V},\ 2.55\,\mathrm{mW}`, name: L("I_PH 1.91 pA, lag-1 0.23", "I_PH 1.91 pA, lag-1 0.23"), value: "3.336 V, 118.6 mV" },
        { symbol: r`-1.8\,\mathrm{V},\ 3.51\,\mathrm{mW}`, name: L("I_PH 2.63 pA, lag-1 0.31", "I_PH 2.63 pA, lag-1 0.31"), value: "3.073 V, 134.1 mV" },
        { symbol: r`-1.1\,\mathrm{V},\ 0\,\mathrm{mW}`, name: L("암조건 (γ 설정 목표), lag-1 0.02", "dark (target for γ), lag-1 0.02"), value: "3.408 V, 53.9 mV" },
        { symbol: r`-1.1\,\mathrm{V},\ 1.15\,\mathrm{mW}`, name: L("I_PH 0.86 pA, lag-1 0.13", "I_PH 0.86 pA, lag-1 0.13"), value: "3.273 V, 46.4 mV" },
        { symbol: r`-1.1\,\mathrm{V},\ 2.55\,\mathrm{mW}`, name: L("I_PH 1.91 pA, lag-1 0.06", "I_PH 1.91 pA, lag-1 0.06"), value: "3.098 V, 42.7 mV" },
        { symbol: r`-1.1\,\mathrm{V},\ 3.51\,\mathrm{mW}`, name: L("I_PH 2.63 pA, lag-1 0.49", "I_PH 2.63 pA, lag-1 0.49"), value: "2.933 V, 33.6 mV" },
      ],
      notes: [
        L(
          "보정에는 −1.8 V 암조건(δφ_G0 = +0.074 V, σ_φ = 0.215 V)과 −1.1 V 암조건 평균(γ)만 쓰이고, 나머지 6개 조건은 예측이다(`predict.py`, `mc_cycles.py`).",
          "Only −1.8 V dark (δφ_G0 = +0.074 V, σ_φ = 0.215 V) and the −1.1 V dark mean (γ) are used for calibration; the other six conditions are predictions (`predict.py`, `mc_cycles.py`).",
        ),
      ],
    },
    {
      heading: L("검증 탭", "Validation tab"),
      body: L(
        "검증 탭(`kind: validation`)은 같은 엔진 함수로 다시 계산해 기대값·계산값·허용오차·통과 여부를 보여준다.\n\n- **fast**: fold 3개(±1 mV), 확장 = 0 항등성(무작위 400개 $(u,r)$ + V_G −2, −1.8, −1.1 V fold, ≤ 1e-12 V), 광 변환(±0.005 pA), latch 창(이분법 0.5 mV, ±10 mV), 측정 기록 통계, FPT 노드(평균 ±3 mV, SD ±1.5 mV), 동적 MC 100 sweep(평균 ±10 mV, SD_LU ±10 mV, SD_LD ±3 mV)\n- **full**: + 10 기록 × 100 sweep(합동 σ, ±1 / ±0.5 mV), 캐리어 잡음 분해(각 ±1 mV), 광조사 소자 sweep_mc(photo 프리셋, −1.8 V 암조건, 평균·SD ±20 mV), V_G 곡선 최대(고정 상태; σ ±10 mV, 평균 ±20 mV, 위치 ±0.1 V), 회로 부하선이 fold를 재현하는지(±30 mV)",
        "The validation tab (`kind: validation`) recomputes the items with the same engine functions and shows expected, computed, tolerance and pass/fail.\n\n- **fast**: 3 fold checks (±1 mV), extensions = 0 identity (400 random $(u,r)$ points + folds at V_G −2, −1.8, −1.1 V, ≤ 1e-12 V), light conversion (±0.005 pA), latch window (0.5 mV bisection, ±10 mV), measured-record statistics, FPT node (mean ±3 mV, SD ±1.5 mV), dynamic MC 100 sweeps (means ±10 mV, SD_LU ±10 mV, SD_LD ±3 mV)\n- **full**: + 10 records × 100 sweeps (pooled σ, ±1 / ±0.5 mV), carrier-noise breakdown (±1 mV each), photo-device sweep_mc (photo preset, −1.8 V dark, mean and SD ±20 mV), V_G-curve peaks (frozen states; σ ±10 mV, mean ±20 mV, positions ±0.1 V), circuit load line reproducing the folds (±30 mV)",
      ),
      equations: [
        {
          id: "eq-val-breakdown",
          label: L("캐리어 잡음 분해 (검증 탭의 재구성)", "carrier-noise breakdown (validation-tab reconstruction)"),
          tex: r`\begin{aligned} &W_{n\to n\pm1}^{\mathrm{drift}} = N\,\max\!\big(\pm v_{\mathrm{det}},\,0\big),\qquad v_{\mathrm{det}} = \sum_{i\notin\mathcal S} s_i\lambda_i\\ &\ln h \approx 2\ln h_{N=8} - \ln h_{N=4}\end{aligned}`,
          note: L(
            "선택한 채널 집합 $\\mathcal S$만 점프(격자 간격 $1/N$에서 크기 $N s_i$), 나머지는 결정론적 drift(upwind)로 둔 조각별 결정론 과정의 MFPT; Richardson 외삽. BTBT = 모든 단위 생성(GIDL + 접합 BTBT), REC = bulk + 접합 SRH, DIFF = emitter 확산.",
            "MFPT of a piecewise-deterministic process in which only the channel set $\\mathcal S$ jumps (size $N s_i$ on a lattice of spacing $1/N$) and the rest is deterministic drift (upwind); Richardson extrapolation. BTBT = all unit generation (GIDL + junction BTBT), REC = bulk + junction SRH, DIFF = emitter diffusion.",
          ),
          code: "validation.py · carrier_noise_breakdown(), _subset_mfpt()",
        },
      ],
      notes: [
        L(
          "`check_paper_records`(full)는 `gate_dynamic_compare.main()`과 같은 seed 2026093000 … 2026093009로 100 sweep 기록 10개를 만들고 1000 sweep을 합쳐 SD를 계산한다(ddof 1). 실행값 $\\sigma_{LU}$ = 125.8 mV, $\\sigma_{LD}$ = 19.6 mV로 논문 값을 그대로 재현한다(허용오차 ±1 / ±0.5 mV; 합동 평균 3.6471 / 2.7000 V).",
          "`check_paper_records` (full) builds the ten 100-sweep records with the seeds of `gate_dynamic_compare.main()`, 2026093000 … 2026093009, and pools the 1000 sweeps (SD with ddof 1). It reproduces the paper values exactly: $\\sigma_{LU}$ = 125.8 mV, $\\sigma_{LD}$ = 19.6 mV (tolerance ±1 / ±0.5 mV; pooled means 3.6471 / 2.7000 V).",
        ),
      ],
    },
  ],
  related: ["sweep-mc", "first-passage", "local-states", "photo", "charge-balance", "design-map"],
  codeRefs: ["docs/VALIDATION.md", "stl_api.py", "data/measured_stats.json", "data/raw_VLU.npy", "server/compute/validation.py"],
};

export default topic;
