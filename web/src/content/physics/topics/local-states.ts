import type { PhysicsTopic } from "../types";
import { CODE, L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "local-states",
  title: L("국소 상태 (drain edge / source edge)", "Local states (drain edge / source edge)"),
  summary: L(
    "사이클 간 산포의 대부분은 두 국소 전위에서 온다: drain edge의 $\\delta\\phi_G$는 GIDL 전계(p[9])를, source edge의 $\\delta\\phi_E$는 emitter 확산 전치인자(p[10])를 바꾼다. 출력 전압에 잡음을 더하지 않고 사건률과 branch 전류를 바꾸며, 사이클마다 고정(frozen)하거나 OU 과정으로 진화(evolving)시킨다.",
    "Most of the cycle-to-cycle spread comes from two local potentials: the drain-edge $\\delta\\phi_G$ shifts the GIDL field (p[9]) and the source-edge $\\delta\\phi_E$ shifts the emitter diffusion prefactor (p[10]). They change rates and branch currents, never add output-voltage jitter, and are either frozen per cycle or evolve as OU processes.",
  ),
  tags: ["stochastic", "local state", "OU"],
  sections: [
    {
      heading: L("작용점 (논문 모델)", "Action points (paper model)"),
      body: L(
        "`params.build_p`는 $p_9=\\phi_{\\mathrm{GIDL},0}+\\delta\\phi_{G0}+\\delta\\phi_G$, $p_{10}=\\phi_{E,0}+\\delta\\phi_{E0}+\\delta\\phi_E$로 넣는다($\\delta\\phi_{G0}$, $\\delta\\phi_{E0}$: UI의 국소 상태 중심).",
        "`params.build_p` sets $p_9=\\phi_{\\mathrm{GIDL},0}+\\delta\\phi_{G0}+\\delta\\phi_G$ and $p_{10}=\\phi_{E,0}+\\delta\\phi_{E0}+\\delta\\phi_E$ ($\\delta\\phi_{G0}$, $\\delta\\phi_{E0}$: the local-state centre in the UI).",
      ),
      equations: [
        {
          id: "eq-ls-gidl-field",
          label: L("drain edge: GIDL 전계", "drain edge: GIDL field"),
          tex: r`\begin{aligned} E_{\mathrm{GIDL}} &= \max\!\Big(\frac{u + r - V_G - 0.3 - 1.12 + p_9}{l_{\mathrm{GIDL}}},\,0\Big)\\ I_{\mathrm{GIDL}} &= q\,\mathcal V\,A_{BB}\,E_{\mathrm{GIDL}}^{2.5}\,e^{-B_{BB}/\max(E_{\mathrm{GIDL}},1)}\,\big(1-e^{-r/V_T}\big)\end{aligned}`,
          note: L(
            "$A_{BB}=4\\times10^{14}$, $B_{BB}=1.9\\times10^7$ V/cm, $\\mathcal V = W\\cdot 5\\,\\mathrm{nm}\\cdot\\min(\\sqrt{2\\varepsilon_{\\mathrm{Si}}1.12/(q\\,7\\times10^{19})},T_{\\mathrm{Si}})$, $l_{\\mathrm{GIDL}}=p_4$ (nm → cm). $\\delta\\phi_G>0$는 전계를 키워 $V_{LU}$를 낮춘다.",
            "$A_{BB}=4\\times10^{14}$, $B_{BB}=1.9\\times10^7$ V/cm, $\\mathcal V = W\\cdot 5\\,\\mathrm{nm}\\cdot\\min(\\sqrt{2\\varepsilon_{\\mathrm{Si}}1.12/(q\\,7\\times10^{19})},T_{\\mathrm{Si}})$, $l_{\\mathrm{GIDL}}=p_4$ (nm → cm). $\\delta\\phi_G>0$ raises the field and lowers $V_{LU}$.",
          ),
          code: "photo_mean.py · components(): eg, gidl",
        },
        {
          id: "eq-ls-emitter",
          label: L("source edge: emitter 확산 손실", "source edge: emitter diffusion loss"),
          tex: r`I_{\mathrm{diff}} = I_{sp}\,e^{-p_{10}/V_T}\,\big(e^{u/V_T}-1\big),\qquad I_{sp} = \frac{q A D_n n_i^2}{N_A L_{\mathrm{ref}}\,\beta}`,
          note: L(
            "$D_n=450V_T$ cm²/s, $L_{\\mathrm{ref}} = L - 2\\sqrt{2\\varepsilon_{\\mathrm{Si}}V_{bi}/(qN_A)}$, $\\beta=p_0$.",
            "$D_n=450V_T$ cm²/s, $L_{\\mathrm{ref}} = L - 2\\sqrt{2\\varepsilon_{\\mathrm{Si}}V_{bi}/(qN_A)}$, $\\beta=p_0$.",
          ),
          code: "photo_mean.py · components(): isp, diff",
        },
      ],
      variables: [
        { symbol: r`\phi_{\mathrm{GIDL},0}`, name: L("drain-edge 평균 (보정)", "drain-edge mean (calibrated)"), value: "0.272", unit: "mV", code: "p[9]" },
        { symbol: r`\phi_{E,0}`, name: L("source-edge 평균 (보정)", "source-edge mean (calibrated)"), value: "0.0591", unit: "mV", code: "p[10]" },
        { symbol: r`\sigma_{\phi G}`, name: L("drain-edge 상태 SD", "drain-edge state SD"), value: "0.1534", unit: "V", code: "SIGMA_PHI_G_V" },
        { symbol: r`\sigma_{\phi E}`, name: L("source-edge 상태 SD", "source-edge state SD"), value: "0.437", unit: "mV", code: "SIGMA_PHI_E_V" },
      ],
      notes: [
        L(
          "네 값은 논문 기록(0.4 V/s, 100 sweep)의 평균·SD($V_{LU}$, $V_{LD}$)에 조회표 분위수로 맞춘 것이다(`calibrate()`, Sobol 2¹⁵ 표본, 10 mV 판독 분산 0.01²/12 포함). 결함 위치를 식별한 것이 아니다.",
          "The four values fit the paper record's means and SDs of $V_{LU}$, $V_{LD}$ (0.4 V/s, 100 sweeps) via the lookup quantiles (`calibrate()`, 2¹⁵ Sobol samples, 10 mV readout variance 0.01²/12 included). No defect is identified.",
        ),
      ],
    },
    {
      heading: L("fold 감도 (V_G = −2 V)", "Fold sensitivities (V_G = −2 V)"),
      equations: [
        {
          id: "eq-ls-sensitivity",
          label: L("감도 (실행 검증)", "sensitivities (verified by running)"),
          tex: r`\begin{aligned} &\frac{\partial V_{LU}}{\partial\phi_G}\approx -0.80,\qquad \frac{\partial V_{LD}}{\partial\phi_E}\approx -41\ \ \mathrm{(V/V)}\\ &\sigma_{LU}\approx 0.80\,\sigma_{\phi G}\approx 122\ \mathrm{mV},\qquad \sigma_{LD}\approx 41\,\sigma_{\phi E}\approx 18\ \mathrm{mV}\end{aligned}`,
          note: L(
            "중심 차분 fold: $\\partial V_{LU}/\\partial\\phi_G=-0.796$, $\\partial V_{LD}/\\partial\\phi_E=-39.8$, $\\partial V_{LU}/\\partial\\phi_E=-4.18$, $\\partial V_{LD}/\\partial\\phi_G\\approx0$. −41.2는 조회표 탈출 평균의 도함수(`LD_event_derivatives`)이다. 캐리어 잡음(≈ 8 mV)은 제곱합으로 더해진다.",
            "Central-difference folds: $\\partial V_{LU}/\\partial\\phi_G=-0.796$, $\\partial V_{LD}/\\partial\\phi_E=-39.8$, $\\partial V_{LU}/\\partial\\phi_E=-4.18$, $\\partial V_{LD}/\\partial\\phi_G\\approx0$. The −41.2 is the derivative of the lookup escape mean (`LD_event_derivatives`). Carrier noise (≈ 8 mV) adds in quadrature.",
          ),
          code: "stl_api.branches(dg=±h, de=±h); gate_dynamic_calibration.json",
        },
      ],
    },
    {
      heading: L("OU 동역학 (evolving 모드)", "OU kinetics (evolving mode)"),
      body: L(
        "정규화 상태 $x$(drain edge), $y$(source edge)는 단위 분산 OU 과정이며 정확한 이산형으로 적분한다. 스텝 $\\Delta t=\\Delta v/\\dot v$ (논문 5 ms). OU는 $n$개 스윕을 이은 한 줄의 시계열이고(스윕 간 경계 없음), 상승·하강 기록은 서로 독립인 경로를 쓴다(unpaired).",
        "The normalised states $x$ (drain edge) and $y$ (source edge) are unit-variance OU processes integrated with the exact discrete form. Step $\\Delta t=\\Delta v/\\dot v$ (paper: 5 ms). The OU runs as one series through the $n$ concatenated sweeps (no boundary between sweeps); up and down records use independent paths (unpaired).",
      ),
      equations: [
        {
          id: "eq-ls-ou",
          label: L("정확한 이산 OU", "exact discrete OU"),
          tex: r`x_{n+1} = \rho\,x_n + \sqrt{1-\rho^2}\,\xi_{n+1},\qquad \rho = e^{-\Delta t/\tau},\qquad \xi_n\sim\mathcal N(0,1),\ \ x_{-1}\sim\mathcal N(0,1)`,
          note: L(
            "`lfilter([√(1−ρ²)], [1, −ρ], ξ, zi=[ρ ξ₀])`: 정상 분포에서 시작한다.",
            "`lfilter([√(1−ρ²)], [1, −ρ], ξ, zi=[ρ ξ₀])`: started from the stationary distribution.",
          ),
          code: "gate_dynamic_compare.py · ou(n, dt, tau, rng)",
        },
        {
          id: "eq-ls-emitter-ou",
          label: L("source edge: 빠른 + 느린 성분", "source edge: fast + slow components"),
          tex: r`y = \sqrt{f_s}\;\mathrm{OU}_{1000\,\mathrm{s}} + \sqrt{1-f_s}\;\mathrm{OU}_{\tau_E},\qquad f_s = 5.6\times10^{-14}\approx 0,\ \ \tau_E = 1.62\ \mathrm{s}`,
          code: "gate_dynamic_compare.py · simulate()",
        },
        {
          id: "eq-ls-trend",
          label: L("drain edge: 획득 추세 (상승만)", "drain edge: acquisition trend (up only)"),
          tex: r`\begin{aligned} &x^{\mathrm{up}}_c = \sqrt{f_{tr}}\;\tilde T(c) + \sqrt{1-f_{tr}}\;\mathrm{OU}_{5\,\mathrm{s}}\\ &\tilde T = -\frac{T-\overline T}{\mathrm{SD}(T)},\qquad f_{tr} = \frac{\operatorname{Var}T}{\operatorname{Var}V_{LU}^{\mathrm{meas}}} = 0.660\end{aligned}`,
          note: L(
            "$T(k)$는 측정 $V_{LU}$ 기록(100 획득)을 $k\\in[-1,1]$에 대해 2차 다항식으로 맞춘 것이다. GIDL 전계 증가가 $V_{LU}$를 낮추므로 부호를 뒤집어 전위로 옮긴다. $n\\ne100$이면 $\\tilde T$를 $n$개 사이클로 선형 보간한다. frozen 모드는 추세를 쓰지 않는다.",
            "$T(k)$ is a quadratic fit of the measured $V_{LU}$ record (100 acquisitions) versus $k\\in[-1,1]$. Since a larger GIDL field lowers $V_{LU}$, the sign is flipped when mapping it to the potential. For $n\\ne100$, $\\tilde T$ is linearly interpolated over the $n$ cycles. Frozen mode does not use the trend.",
          ),
          code: "gate_dynamic_compare.py · simulate(): trend, frac",
        },
        {
          id: "eq-ls-state",
          label: L("작용점 상태", "action-point states"),
          tex: r`\big(p_9,\;p_{10}\big) = \big(\phi_{\mathrm{GIDL},0} + \sigma_{\phi G}\,x,\ \ \phi_{E,0} + \sigma_{\phi E}\,y\big)`,
          code: "gate_dynamic_compare.py · simulate(): state = c_[j0+sj*x, e0+se*y]",
        },
      ],
      variables: [
        { symbol: r`\tau_G`, name: L("drain-edge OU τ (스윕 내 'up residual', 조건부 선택)", "drain-edge OU τ (within a sweep, 'up residual', conditional choice)"), value: "5", unit: "s", code: "kinetic_fit.up_residual_tau_s" },
        { symbol: r`\tau_E`, name: L("source-edge OU τ (LD 공분산 3–3.5 V로 적합)", "source-edge OU τ (fit to LD covariance, 3–3.5 V)"), value: "1.62", unit: "s", code: "kinetic_fit.tau_fast_s" },
        { symbol: r`f_s`, name: L("느린 성분 비율 (τ 1000 s)", "slow-component fraction (τ 1000 s)"), value: "≈ 0", code: "kinetic_fit.fraction_slow" },
        { symbol: r`f_{tr}`, name: L("추세가 설명하는 V_LU 분산 비율", "V_LU variance fraction explained by the trend"), value: "0.660" },
      ],
    },
    {
      heading: L("모드: none / frozen / evolving", "Modes: none / frozen / evolving"),
      body: L(
        "- **none**: 상태를 중심값에 고정, 캐리어 잡음만.\n- **frozen**: 사이클마다 $x,y\\sim\\mathcal N(0,1)$ 한 번 추출, 사이클 동안 일정. 사이클 ≪ τ일 때 유효(광조사 소자: 1200 V/s, 사이클 8 ms).\n- **evolving**: 위 OU 과정(+ 논문 기록의 획득 추세 선택).\n\n논문 조건 100 sweep(seed 2026092920) 실행값 — evolving: $V_{LU}$ 3.6344 V / 119.0 mV, $V_{LD}$ 2.6999 V / 21.4 mV; frozen: 3.6602 / 122.4, 2.6998 / 19.7; 상태 없음: 3.6462 / 9.1, 2.7005 / 7.7; 추세 없음: 3.6302 / 112.8 mV.",
        "- **none**: states fixed at the centre, carrier noise only.\n- **frozen**: one draw $x,y\\sim\\mathcal N(0,1)$ per cycle, constant during the cycle. Valid when cycle ≪ τ (photo device: 1200 V/s, 8 ms cycle).\n- **evolving**: the OU processes above (+ optional acquisition trend of the paper record).\n\nPaper condition, 100 sweeps (seed 2026092920), as run — evolving: $V_{LU}$ 3.6344 V / 119.0 mV, $V_{LD}$ 2.6999 V / 21.4 mV; frozen: 3.6602 / 122.4, 2.6998 / 19.7; no states: 3.6462 / 9.1, 2.7005 / 7.7; no trend: 3.6302 / 112.8 mV.",
      ),
      notes: [
        L(
          "frozen 모드의 사이클 간 독립 추출은 사이클 간 상관을 무시한다: 광조사 소자의 측정 lag-1은 0.00–0.50(앱 `Stats.lag1` 추정량, 아래). 사이클 내 고정(τ ≫ 4 ms)과 사이클 간 독립(τ ≪ 8 ms)은 하나의 OU로 동시에 만족될 수 없다.",
          "Independent per-cycle draws in frozen mode ignore inter-cycle correlation: the photo device's measured lag-1 is 0.00–0.50 (the app's `Stats.lag1` estimator, below). Frozen within a cycle (τ ≫ 4 ms) and independent between cycles (τ ≪ 8 ms) cannot both hold for one OU process.",
        ),
        L(
          "evolving 모드 lag-1 (100 sweep, seed 2026092920, 추세 포함): 앱 `Stats.lag1`(연속한 유한 쌍의 Pearson 상관, `stoch_core.lag1`)으로 모의 $V_{LU}$ 0.68 vs 측정 0.70. `measured_stats.json`·데이터 탭의 추정량 $\\sum(x_t-\\bar x)(x_{t+1}-\\bar x)/\\sum(x_t-\\bar x)^2$로는 0.66 vs 0.68 (광조사 소자 측정 0.00–0.49).",
          "Evolving-mode lag-1 (100 sweeps, seed 2026092920, with trend): with the app's `Stats.lag1` (Pearson correlation of consecutive finite pairs, `stoch_core.lag1`) simulated $V_{LU}$ 0.68 vs measured 0.70. With the estimator of `measured_stats.json` and the Data tab, $\\sum(x_t-\\bar x)(x_{t+1}-\\bar x)/\\sum(x_t-\\bar x)^2$, 0.66 vs 0.68 (photo device measured 0.00–0.49).",
        ),
      ],
    },
    {
      heading: L("대안 작용점 (옵션)", "Alternative action points (options)"),
      body: L(
        "UI의 `action`은 상태가 작용하는 파라미터를 고른다: `gidl` → p[9] (논문), `local_avalanche` → p[23] (ln 단위; p[21] > 0 필요), `junction` → p[19] (V), `multiplication` → p[20] (ln 단위). 뒤의 셋은 실험적이다(`hypotheses.py`).",
        "The UI `action` chooses the parameter the state acts on: `gidl` → p[9] (paper), `local_avalanche` → p[23] (ln units; needs p[21] > 0), `junction` → p[19] (V), `multiplication` → p[20] (ln units). The last three are experimental (`hypotheses.py`).",
      ),
      equations: [
        {
          id: "eq-ls-iloc",
          label: L("국소 애벌랜치 경로, 표면 정의 (p[24] ≤ 0.5)", "local avalanche path, surface definition (p[24] ≤ 0.5)"),
          tex: r`\begin{aligned} I_{\mathrm{loc}} = \min\!\Big(&p_{21}\,e^{p_{23}}\,e^{p_{25}\,(u+r-V_G-5.6)}\,\max(M-1,0)\\ &\times\big(I_{\mathrm{GIDL}} + I_{\mathrm{BTBT},j} + I_{\mathrm{PH}} + I_{\mathrm{ch}}\big),\ p_{22}\Big)\end{aligned}`,
          code: "photo_mean.py · components(): iloc (loc_on and not bulk)",
        },
        {
          id: "eq-ls-iloc-bulk",
          label: L("벌크 정의 (p[24] > 0.5, p[24] = 2 포함)", "bulk definition (p[24] > 0.5, including p[24] = 2)"),
          tex: r`I_{\mathrm{loc}} = \min\!\Big(p_{21}\,e^{p_{23}}\,\max(M-1,0)\,\big(I_{\mathrm{seed}} + I_{\mathrm{PH}} + I_{\mathrm{BTBT},j}\big),\ p_{22}\Big)`,
          note: L(
            "SRH/전송 풀이와 3회 고정점 반복, $p_{25}$ 인자 없음. $I_{\\mathrm{loc}}$는 정공 공급·드레인 전류에 더해지고 확률 모델에서는 II 클러스터로 간주된다.",
            "Three fixed-point iterations with the SRH/transport solve, no $p_{25}$ factor. $I_{\\mathrm{loc}}$ adds to the hole supply and drain current and counts as II clusters in the stochastic model.",
          ),
          code: "photo_mean.py · components(): loc_on and bulk loop",
        },
        {
          id: "eq-ls-junction",
          label: L("접합 오프셋과 (M−1) 스케일", "junction offset and (M−1) scale"),
          tex: r`M = 1 + \big(M_0(r+p_{19}) - 1\big)\,e^{p_{20}},\qquad I_{\mathrm{BTBT},j} = G_0(r+p_{19})\,\big(1-e^{-r/V_T}\big)`,
          code: "photo_mean.py · components(): mult, bbj",
        },
      ],
      notes: [
        L(
          "불일치: 주석상 p[24] = 2는 '채널 제외 edge 정의'이지만 `bulk = p[24] > 0.5`가 먼저 참이 되어 `edge_only` 분기는 도달 불가하다. 실행 확인 결과 p[24] = 2는 p[24] = 1(벌크)과 같은 값을 준다.",
          "Inconsistency: p[24] = 2 is documented as 'edge definition without channel', but `bulk = p[24] > 0.5` is already true, so the `edge_only` branch is unreachable. Running it confirms p[24] = 2 gives exactly the p[24] = 1 (bulk) result.",
        ),
      ],
    },
    {
      heading: L("광조사 소자 보정", "Photo-device calibration"),
      body: L(
        "V_G = −1.8 V 암조건 400 cycle 기록(1200 V/s)의 평균 3.806 V, SD 173.2 mV에 fold 혼합의 두 모멘트를 맞춘다. fold 표 $f(\\delta)$는 $\\delta\\in[-1.5,2.0]$ V (0.05 V) PCHIP, 40점 Gauss–Hermite, latch 범위 밖 가중치는 제외 후 재정규화.",
        "The two moments of the fold mixture are matched to the V_G = −1.8 V dark 400-cycle record (1200 V/s): mean 3.806 V, SD 173.2 mV. The fold table $f(\\delta)$ is PCHIP over $\\delta\\in[-1.5,2.0]$ V (0.05 V), 40-point Gauss–Hermite, weights outside the latch range dropped and renormalised.",
      ),
      equations: [
        {
          id: "eq-ls-calibration",
          label: L("모멘트 조건", "moment conditions"),
          tex: r`\begin{aligned} &\bar f = \sum_i w_i\,f\big(\delta\phi_{G0}+\sqrt2\,\sigma_\phi x_i\big) = 3.806\ \mathrm{V}\\ &\Big[\sum_i w_i\,\big(f_i-\bar f\big)^2\Big]^{1/2} = 173.2\ \mathrm{mV}\end{aligned}`,
          code: "calibrate.py · moments(), fsolve",
        },
      ],
      variables: [
        { symbol: r`\delta\phi_{G0}`, name: L("광조사 소자 drain-edge 중심", "photo-device drain-edge centre"), value: "+0.0744", unit: "V", code: "c2c_calibration_m18_dark.json" },
        { symbol: r`\sigma_\phi`, name: L("광조사 소자 상태 SD", "photo-device state SD"), value: "0.2154", unit: "V", code: "c2c_calibration_m18_dark.json" },
      ],
      notes: [
        L(
          "보정은 fold만 사용(캐리어 잡음 제외); `predict.py`/`mc_cycles.py`는 그 위에 first-passage 잡음을 더한다. 논문 모델 레버를 그대로 쓴 값이다.",
          "The calibration uses folds only (carrier noise excluded); `predict.py`/`mc_cycles.py` add first-passage noise on top. The values use the paper-model levers unchanged.",
        ),
      ],
    },
  ],
  related: ["btbt-gidl", "bjt-transport", "sweep-mc", "open-problems", "first-passage", "photo"],
  codeRefs: [CODE.gateDyn, CODE.gateCal, CODE.params, CODE.photoMean, CODE.hypotheses, CODE.verifyHloc, CODE.calibrate],
};

export default topic;
