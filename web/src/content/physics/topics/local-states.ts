import type { PhysicsTopic } from "../types";
import { CODE, L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "local-states",
  title: L("국소 상태 (드레인 가장자리 / 소스 가장자리)", "Local states (drain edge / source edge)"),
  summary: L(
    "사이클 간 산포는 대부분 두 국소 전위에서 나온다. 드레인 가장자리의 $\\delta\\phi_G$는 GIDL 전계(p[9])를, 소스 가장자리의 $\\delta\\phi_E$는 이미터 확산 전치 인자(p[10])를 바꾼다. 국소 상태는 출력 전압에 잡음을 더하지 않고 사건률과 branch 전류를 바꾸며, 사이클마다 고정(frozen)하거나 OU 과정으로 진화(evolving)시킨다.",
    "Most of the cycle-to-cycle spread comes from two local potentials: the drain-edge $\\delta\\phi_G$ shifts the GIDL field (p[9]) and the source-edge $\\delta\\phi_E$ shifts the emitter-diffusion prefactor (p[10]). The local states change rates and branch currents, never add output-voltage jitter, and are either frozen per cycle or evolve as OU processes.",
  ),
  tags: ["stochastic", "local state", "OU"],
  sections: [
    {
      heading: L("작용점 (기본 모델)", "Action points (base model)"),
      body: L(
        "`params.build_p`는 $p_9=\\phi_{\\mathrm{GIDL},0}+\\delta\\phi_{G0}+\\delta\\phi_G$, $p_{10}=\\phi_{E,0}+\\delta\\phi_{E0}+\\delta\\phi_E$로 설정한다($\\delta\\phi_{G0}$, $\\delta\\phi_{E0}$는 UI에서 정하는 국소 상태의 중심).",
        "`params.build_p` sets $p_9=\\phi_{\\mathrm{GIDL},0}+\\delta\\phi_{G0}+\\delta\\phi_G$ and $p_{10}=\\phi_{E,0}+\\delta\\phi_{E0}+\\delta\\phi_E$ ($\\delta\\phi_{G0}$, $\\delta\\phi_{E0}$: the local-state centers set in the UI).",
      ),
      equations: [
        {
          id: "eq-ls-gidl-field",
          label: L("드레인 가장자리: GIDL 전계", "Drain edge: GIDL field"),
          tex: r`\begin{aligned} E_{\mathrm{GIDL}} &= \max\!\Big(\frac{u + r - V_G - 0.3 - 1.12 + p_9}{l_{\mathrm{GIDL}}},\,0\Big)\\ I_{\mathrm{GIDL}} &= q\,\mathcal V\,A_{BB}\,E_{\mathrm{GIDL}}^{2.5}\,e^{-B_{BB}/\max(E_{\mathrm{GIDL}},1)}\,\big(1-e^{-r/V_T}\big)\end{aligned}`,
          note: L(
            "$A_{BB}=4\\times10^{14}$, $B_{BB}=1.9\\times10^7$ V/cm, $\\mathcal V = W\\cdot 5\\,\\mathrm{nm}\\cdot\\min(\\sqrt{2\\varepsilon_{\\mathrm{Si}}1.12/(q\\,7\\times10^{19})},T_{\\mathrm{Si}})$, $l_{\\mathrm{GIDL}}=p_4$ (nm → cm). $\\delta\\phi_G>0$이면 전계가 커져 $V_{LU}$가 낮아진다.",
            "$A_{BB}=4\\times10^{14}$, $B_{BB}=1.9\\times10^7$ V/cm, $\\mathcal V = W\\cdot 5\\,\\mathrm{nm}\\cdot\\min(\\sqrt{2\\varepsilon_{\\mathrm{Si}}1.12/(q\\,7\\times10^{19})},T_{\\mathrm{Si}})$, $l_{\\mathrm{GIDL}}=p_4$ (nm → cm). $\\delta\\phi_G>0$ raises the field and lowers $V_{LU}$.",
          ),
          code: "photo_mean.py · components(): eg, gidl",
        },
        {
          id: "eq-ls-emitter",
          label: L("소스 가장자리: 이미터 확산 손실", "Source edge: emitter diffusion loss"),
          tex: r`I_{\mathrm{diff}} = I_{sp}\,e^{-p_{10}/V_T}\,\big(e^{u/V_T}-1\big),\qquad I_{sp} = \frac{q A D_n n_i^2}{N_A L_{\mathrm{ref}}\,\beta}`,
          note: L(
            "$D_n=450V_T$ cm²/s, $L_{\\mathrm{ref}} = L - 2\\sqrt{2\\varepsilon_{\\mathrm{Si}}V_{bi}/(qN_A)}$, $\\beta=p_0$.",
            "$D_n=450V_T$ cm²/s, $L_{\\mathrm{ref}} = L - 2\\sqrt{2\\varepsilon_{\\mathrm{Si}}V_{bi}/(qN_A)}$, $\\beta=p_0$.",
          ),
          code: "photo_mean.py · components(): isp, diff",
        },
      ],
      variables: [
        { symbol: r`\phi_{\mathrm{GIDL},0}`, name: L("드레인 가장자리 평균 (보정값)", "Drain-edge mean (calibrated)"), value: "0.272", unit: "mV", code: "p[9]" },
        { symbol: r`\phi_{E,0}`, name: L("소스 가장자리 평균 (보정값)", "Source-edge mean (calibrated)"), value: "0.0591", unit: "mV", code: "p[10]" },
        { symbol: r`\sigma_{\phi G}`, name: L("드레인 가장자리 상태 SD", "Drain-edge state SD"), value: "0.1534", unit: "V", code: "SIGMA_PHI_G_V" },
        { symbol: r`\sigma_{\phi E}`, name: L("소스 가장자리 상태 SD", "Source-edge state SD"), value: "0.437", unit: "mV", code: "SIGMA_PHI_E_V" },
      ],
      notes: [
        L(
          "네 값은 기준 측정 기록(0.4 V/s, 100회 스윕)의 $V_{LU}$, $V_{LD}$ 평균과 SD에 조회표 분위수로 맞춘 것이다(`calibrate()`, Sobol 표본 2¹⁵개, 10 mV 판독 분산 0.01²/12 포함). 특정 결함을 식별한 결과는 아니다.",
          "The four values are fitted to the means and SDs of $V_{LU}$ and $V_{LD}$ in the reference record (0.4 V/s, 100 sweeps) via the lookup quantiles (`calibrate()`, 2¹⁵ Sobol samples, 10 mV readout variance 0.01²/12 included). No specific defect is identified.",
        ),
      ],
    },
    {
      heading: L("fold 감도 (V_G = −2 V)", "Fold sensitivities (V_G = −2 V)"),
      equations: [
        {
          id: "eq-ls-sensitivity",
          label: L("감도 (실행으로 확인)", "Sensitivities (verified by running)"),
          tex: r`\begin{aligned} &\frac{\partial V_{LU}}{\partial\phi_G}\approx -0.80,\qquad \frac{\partial V_{LD}}{\partial\phi_E}\approx -41\ \ \mathrm{(V/V)}\\ &\sigma_{LU}\approx 0.80\,\sigma_{\phi G}\approx 122\ \mathrm{mV},\qquad \sigma_{LD}\approx 41\,\sigma_{\phi E}\approx 18\ \mathrm{mV}\end{aligned}`,
          note: L(
            "중앙 차분으로 구한 fold 감도: $\\partial V_{LU}/\\partial\\phi_G=-0.796$, $\\partial V_{LD}/\\partial\\phi_E=-39.8$, $\\partial V_{LU}/\\partial\\phi_E=-4.18$, $\\partial V_{LD}/\\partial\\phi_G\\approx0$. −41.2는 조회표 탈출 평균의 도함수(`LD_event_derivatives`)다. 캐리어 잡음(≈ 8 mV)은 제곱합으로 더해진다.",
            "Central-difference fold sensitivities: $\\partial V_{LU}/\\partial\\phi_G=-0.796$, $\\partial V_{LD}/\\partial\\phi_E=-39.8$, $\\partial V_{LU}/\\partial\\phi_E=-4.18$, $\\partial V_{LD}/\\partial\\phi_G\\approx0$. The −41.2 is the derivative of the lookup escape mean (`LD_event_derivatives`). Carrier noise (≈ 8 mV) adds in quadrature.",
          ),
          code: "stl_api.branches(dg=±h, de=±h); gate_dynamic_calibration.json",
        },
      ],
    },
    {
      heading: L("OU 동역학 (진화 모드)", "OU kinetics (evolving mode)"),
      body: L(
        "정규화된 상태 $x$(드레인 가장자리)와 $y$(소스 가장자리)는 분산이 1인 OU 과정이며, 정확한 이산형으로 적분한다. 스텝은 $\\Delta t=\\Delta v/\\dot v$이다(기준 보정: 5 ms). OU 과정은 $n$개의 스윕을 이어 붙인 하나의 시계열로 진행하고(스윕 사이에 경계가 없음), 상승 기록과 하강 기록은 서로 독립인 경로를 쓴다(짝짓지 않음, unpaired).",
        "The normalized states $x$ (drain edge) and $y$ (source edge) are unit-variance OU processes integrated with the exact discrete form. The step is $\\Delta t=\\Delta v/\\dot v$ (reference calibration: 5 ms). The OU process runs as one series through the $n$ concatenated sweeps (no boundary between sweeps); up and down records use independent paths (unpaired).",
      ),
      equations: [
        {
          id: "eq-ls-ou",
          label: L("정확한 이산 OU", "Exact discrete OU"),
          tex: r`x_{n+1} = \rho\,x_n + \sqrt{1-\rho^2}\,\xi_{n+1},\qquad \rho = e^{-\Delta t/\tau},\qquad \xi_n\sim\mathcal N(0,1),\ \ x_{-1}\sim\mathcal N(0,1)`,
          note: L(
            "`lfilter([√(1−ρ²)], [1, −ρ], ξ, zi=[ρ ξ₀])`: 정상 분포에서 시작한다.",
            "`lfilter([√(1−ρ²)], [1, −ρ], ξ, zi=[ρ ξ₀])`: starts from the stationary distribution.",
          ),
          code: "gate_dynamic_compare.py · ou(n, dt, tau, rng)",
        },
        {
          id: "eq-ls-emitter-ou",
          label: L("소스 가장자리: 빠른 성분 + 느린 성분", "Source edge: fast + slow components"),
          tex: r`y = \sqrt{f_s}\;\mathrm{OU}_{1000\,\mathrm{s}} + \sqrt{1-f_s}\;\mathrm{OU}_{\tau_E},\qquad f_s = 5.6\times10^{-14}\approx 0,\ \ \tau_E = 1.62\ \mathrm{s}`,
          code: "gate_dynamic_compare.py · simulate()",
        },
        {
          id: "eq-ls-trend",
          label: L("드레인 가장자리: 획득 추세 (상승 스윕만)", "Drain edge: acquisition trend (up sweeps only)"),
          tex: r`\begin{aligned} &x^{\mathrm{up}}_c = \sqrt{f_{tr}}\;\tilde T(c) + \sqrt{1-f_{tr}}\;\mathrm{OU}_{5\,\mathrm{s}}\\ &\tilde T = -\frac{T-\overline T}{\mathrm{SD}(T)},\qquad f_{tr} = \frac{\operatorname{Var}T}{\operatorname{Var}V_{LU}^{\mathrm{meas}}} = 0.660\end{aligned}`,
          note: L(
            "$T(k)$는 측정 $V_{LU}$ 기록(100회 획득)을 측정 순서 $k\\in[-1,1]$에 대한 2차 다항식으로 맞춘 것이다. GIDL 전계가 커지면 $V_{LU}$가 낮아지므로, 전위로 옮길 때 부호를 뒤집는다. $n\\ne100$이면 $\\tilde T$를 $n$개 사이클로 선형 보간한다. 고정(frozen) 모드는 추세를 쓰지 않는다.",
            "$T(k)$ is a quadratic fit of the measured $V_{LU}$ record (100 acquisitions) versus the acquisition index $k\\in[-1,1]$. Since a stronger GIDL field lowers $V_{LU}$, the sign is flipped when the trend is mapped onto the potential. For $n\\ne100$, $\\tilde T$ is linearly interpolated over the $n$ cycles. Frozen mode does not use the trend.",
          ),
          code: "gate_dynamic_compare.py · simulate(): trend, frac",
        },
        {
          id: "eq-ls-state",
          label: L("작용점 상태", "Action-point states"),
          tex: r`\big(p_9,\;p_{10}\big) = \big(\phi_{\mathrm{GIDL},0} + \sigma_{\phi G}\,x,\ \ \phi_{E,0} + \sigma_{\phi E}\,y\big)`,
          code: "gate_dynamic_compare.py · simulate(): state = c_[j0+sj*x, e0+se*y]",
        },
      ],
      variables: [
        { symbol: r`\tau_G`, name: L("드레인 가장자리 OU τ (스윕 내 'up residual', 조건부 선택)", "Drain-edge OU τ (within a sweep, 'up residual', conditional choice)"), value: "5", unit: "s", code: "kinetic_fit.up_residual_tau_s" },
        { symbol: r`\tau_E`, name: L("소스 가장자리 OU τ (3–3.5 V의 LD 공분산에 맞춤)", "Source-edge OU τ (fitted to the LD covariance, 3–3.5 V)"), value: "1.62", unit: "s", code: "kinetic_fit.tau_fast_s" },
        { symbol: r`f_s`, name: L("느린 성분의 비율 (τ 1000 s)", "Slow-component fraction (τ 1000 s)"), value: "≈ 0", code: "kinetic_fit.fraction_slow" },
        { symbol: r`f_{tr}`, name: L("추세가 설명하는 V_LU 분산 비율", "Fraction of the V_LU variance explained by the trend"), value: "0.660" },
      ],
    },
    {
      heading: L("모드: 없음 / 고정 / 진화", "Modes: none / frozen / evolving"),
      body: L(
        "- **없음(none)**: 상태를 중심값에 고정하고 캐리어 잡음만 넣는다.\n- **고정(frozen)**: 사이클마다 $x,y\\sim\\mathcal N(0,1)$을 한 번 뽑아 그 사이클 동안 유지한다. 사이클 길이가 τ보다 훨씬 짧을 때 유효하다(광조사 보정: 1200 V/s, 사이클 8 ms).\n- **진화(evolving)**: 위의 OU 과정을 쓴다(기준 측정 기록의 획득 추세는 선택 사항).\n\n기준 보정 조건, 100회 스윕(난수 시드 2026092920)의 실행 결과 — 진화: $V_{LU}$ 3.6344 V / 119.0 mV, $V_{LD}$ 2.6999 V / 21.4 mV; 고정: 3.6602 / 122.4, 2.6998 / 19.7; 상태 없음: 3.6462 / 9.1, 2.7005 / 7.7; 추세 없음: 3.6302 / 112.8 mV.",
        "- **none**: states fixed at the center, carrier noise only.\n- **frozen**: one draw $x,y\\sim\\mathcal N(0,1)$ per cycle, held constant during the cycle. Valid when the cycle is much shorter than τ (illumination calibration: 1200 V/s, 8 ms cycle).\n- **evolving**: the OU processes above (optionally with the acquisition trend of the reference record).\n\nReference-calibration condition, 100 sweeps (seed 2026092920), as run — evolving: $V_{LU}$ 3.6344 V / 119.0 mV, $V_{LD}$ 2.6999 V / 21.4 mV; frozen: 3.6602 / 122.4, 2.6998 / 19.7; no states: 3.6462 / 9.1, 2.7005 / 7.7; no trend: 3.6302 / 112.8 mV.",
      ),
      notes: [
        L(
          "고정 모드는 사이클마다 독립적으로 뽑으므로 사이클 간 상관을 무시한다. 광조사 측정 기록의 lag-1은 0.00–0.50이다(앱의 `Stats.lag1` 추정량, 아래 참조). 사이클 안에서 고정(τ ≫ 4 ms)되면서 사이클 사이에서는 독립(τ ≪ 8 ms)인 조건은 하나의 OU 과정으로 동시에 만족할 수 없다.",
          "Independent per-cycle draws in frozen mode ignore the correlation between cycles: the measured lag-1 of the illumination records is 0.00–0.50 (the app's `Stats.lag1` estimator, see below). Being frozen within a cycle (τ ≫ 4 ms) and independent between cycles (τ ≪ 8 ms) cannot both hold for a single OU process.",
        ),
        L(
          "진화 모드의 lag-1(100회 스윕, 난수 시드 2026092920, 추세 포함): 앱의 `Stats.lag1`(연속한 유한값 쌍의 Pearson 상관, `stoch_core.lag1`)로는 모의 $V_{LU}$ 0.68, 측정 0.70이다. `measured_stats.json`과 데이터 탭의 추정량 $\\sum(x_t-\\bar x)(x_{t+1}-\\bar x)/\\sum(x_t-\\bar x)^2$로는 0.66, 0.68이다(광조사 측정 기록 0.00–0.49).",
          "Evolving-mode lag-1 (100 sweeps, seed 2026092920, with trend): with the app's `Stats.lag1` (Pearson correlation of consecutive finite pairs, `stoch_core.lag1`), simulated $V_{LU}$ 0.68 vs measured 0.70. With the estimator of `measured_stats.json` and the Data tab, $\\sum(x_t-\\bar x)(x_{t+1}-\\bar x)/\\sum(x_t-\\bar x)^2$: 0.66 vs 0.68 (illumination records: 0.00–0.49).",
        ),
      ],
    },
    {
      heading: L("다른 작용점 (옵션)", "Alternative action points (options)"),
      body: L(
        "UI의 `action`은 상태가 작용할 파라미터를 고른다: `gidl` → p[9](기본 모델), `local_avalanche` → p[23](ln 단위, p[21] > 0 필요), `junction` → p[19](V), `multiplication` → p[20](ln 단위). 뒤의 세 가지는 실험적 옵션이다(`hypotheses.py`).",
        "The UI `action` selects the parameter the state acts on: `gidl` → p[9] (base model), `local_avalanche` → p[23] (ln units; needs p[21] > 0), `junction` → p[19] (V), `multiplication` → p[20] (ln units). The last three are experimental (`hypotheses.py`).",
      ),
      equations: [
        {
          id: "eq-ls-iloc",
          label: L("국소 애벌랜치 경로, 표면 정의 (p[24] ≤ 0.5)", "Local avalanche path, surface definition (p[24] ≤ 0.5)"),
          tex: r`\begin{aligned} I_{\mathrm{loc}} = \min\!\Big(&p_{21}\,e^{p_{23}}\,e^{p_{25}\,(u+r-V_G-5.6)}\,\max(M-1,0)\\ &\times\big(I_{\mathrm{GIDL}} + I_{\mathrm{BTBT},j} + I_{\mathrm{PH}} + I_{\mathrm{ch}}\big),\ p_{22}\Big)\end{aligned}`,
          code: "photo_mean.py · components(): iloc (loc_on and not bulk)",
        },
        {
          id: "eq-ls-iloc-bulk",
          label: L("벌크 정의 (p[24] > 0.5, p[24] = 2 포함)", "Bulk definition (p[24] > 0.5, including p[24] = 2)"),
          tex: r`I_{\mathrm{loc}} = \min\!\Big(p_{21}\,e^{p_{23}}\,\max(M-1,0)\,\big(I_{\mathrm{seed}} + I_{\mathrm{PH}} + I_{\mathrm{BTBT},j}\big),\ p_{22}\Big)`,
          note: L(
            "SRH/수송 풀이를 포함한 고정점 반복을 3회 하며, $p_{25}$ 인자는 없다. $I_{\\mathrm{loc}}$는 정공 공급과 드레인 전류에 더해지고, 확률 모델에서는 II 클러스터로 취급한다.",
            "Three fixed-point iterations including the SRH/transport solve, no $p_{25}$ factor. $I_{\\mathrm{loc}}$ adds to the hole supply and the drain current and counts as II clusters in the stochastic model.",
          ),
          code: "photo_mean.py · components(): loc_on and bulk loop",
        },
        {
          id: "eq-ls-junction",
          label: L("접합 오프셋과 (M−1) 스케일", "Junction offset and (M−1) scale"),
          tex: r`M = 1 + \big(M_0(r+p_{19}) - 1\big)\,e^{p_{20}},\qquad I_{\mathrm{BTBT},j} = G_0(r+p_{19})\,\big(1-e^{-r/V_T}\big)`,
          code: "photo_mean.py · components(): mult, bbj",
        },
      ],
      notes: [
        L(
          "불일치: 주석에는 p[24] = 2가 '채널을 뺀 가장자리 정의'로 적혀 있지만, `bulk = p[24] > 0.5`가 먼저 참이 되므로 `edge_only` 분기에는 도달할 수 없다. 실행해 보면 p[24] = 2는 p[24] = 1(벌크)과 똑같은 결과를 준다.",
          "Inconsistency: p[24] = 2 is documented as the 'edge definition without the channel', but `bulk = p[24] > 0.5` is already true, so the `edge_only` branch is unreachable. Running it confirms that p[24] = 2 gives exactly the p[24] = 1 (bulk) result.",
        ),
      ],
    },
    {
      heading: L("광조사 보정", "Illumination calibration"),
      body: L(
        "fold 혼합의 두 모멘트를 V_G = −1.8 V 암조건 400 사이클 기록(1200 V/s)의 평균 3.806 V, SD 173.2 mV에 맞춘다. fold 표 $f(\\delta)$는 $\\delta\\in[-1.5,2.0]$ V(0.05 V 간격)의 PCHIP이고, 40점 Gauss–Hermite 적분을 쓰며, 래치 범위 밖 노드의 가중치는 빼고 재정규화한다.",
        "The two moments of the fold mixture are matched to the V_G = −1.8 V dark 400-cycle record (1200 V/s): mean 3.806 V, SD 173.2 mV. The fold table $f(\\delta)$ is a PCHIP over $\\delta\\in[-1.5,2.0]$ V (0.05 V steps) with 40-point Gauss–Hermite quadrature; weights outside the latch range are dropped and the rest renormalized.",
      ),
      equations: [
        {
          id: "eq-ls-calibration",
          label: L("모멘트 조건", "Moment conditions"),
          tex: r`\begin{aligned} &\bar f = \sum_i w_i\,f\big(\delta\phi_{G0}+\sqrt2\,\sigma_\phi x_i\big) = 3.806\ \mathrm{V}\\ &\Big[\sum_i w_i\,\big(f_i-\bar f\big)^2\Big]^{1/2} = 173.2\ \mathrm{mV}\end{aligned}`,
          code: "calibrate.py · moments(), fsolve",
        },
      ],
      variables: [
        { symbol: r`\delta\phi_{G0}`, name: L("광조사 보정의 드레인 가장자리 중심", "Drain-edge center, illumination calibration"), value: "+0.0744", unit: "V", code: "c2c_calibration_m18_dark.json" },
        { symbol: r`\sigma_\phi`, name: L("광조사 보정의 상태 SD", "State SD, illumination calibration"), value: "0.2154", unit: "V", code: "c2c_calibration_m18_dark.json" },
      ],
      notes: [
        L(
          "보정에는 fold만 쓰며 캐리어 잡음은 제외한다. `predict.py`/`mc_cycles.py`는 그 위에 첫 통과 잡음을 더한다. 나머지 파라미터는 기본 모델의 값을 그대로 쓴다.",
          "The calibration uses the folds only (carrier noise excluded); `predict.py`/`mc_cycles.py` add first-passage noise on top. All other parameters keep their base-model values.",
        ),
      ],
    },
  ],
  related: ["btbt-gidl", "bjt-transport", "sweep-mc", "open-problems", "first-passage", "photo"],
  codeRefs: [CODE.gateDyn, CODE.gateCal, CODE.params, CODE.photoMean, CODE.hypotheses, CODE.verifyHloc, CODE.calibrate],
};

export default topic;
