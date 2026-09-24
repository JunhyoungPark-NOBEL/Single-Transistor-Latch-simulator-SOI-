import type { PhysicsTopic } from "../types";
import { CODE, L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "sweep-mc",
  title: L("스윕 Monte Carlo와 V_G 곡선", "Sweep Monte Carlo and the V_G curve"),
  summary: L(
    "삼각 스윕 0 → $V_{D,\\max}$ → 0의 각 사이클에서 국소 상태로 fold를 정하고, fold까지의 거리에 대한 first-passage hazard를 Exp(1) 문턱과 비교해 $V_{LU}$, $V_{LD}$를 뽑는다. 논문 소자는 보정 조회표 엔진, 그 밖의 조건은 일반 엔진을 쓴다.",
    "In each cycle of the triangular sweep 0 → $V_{D,\\max}$ → 0 the local state sets the fold, and the first-passage hazard versus distance to that fold is integrated against an Exp(1) threshold to draw $V_{LU}$, $V_{LD}$. The paper device uses the calibrated lookup engine; any other condition uses the general engine.",
  ),
  tags: ["stochastic", "Monte Carlo", "V_LU", "V_LD"],
  sections: [
    {
      heading: L("엔진 선택", "Engine selection"),
      body: L(
        "`engine: auto`는 (논문 기준 소자: V_G = −2 V, 암조건, 확장 없음, 상태 중심 0) ∧ `action = gidl` ∧ $V_{D,\\max}=4$ V ∧ 캐리어 잡음 on ∧ $\\Delta V = 10\\,\\mathrm{mV}/k$ ($k=1,2,\\dots$: 10, 5, 3.333, 2.5, 2, … mV)일 때 `calibrated_lookup`, 아니면 `general`을 고른다(ΔV 조건만 어긋나면 경고, `calibrated_lookup`을 직접 지정하면 오류). 모드 대응(보정 엔진): none → `fast_only`, frozen → `frozen`, evolving → `dynamic`(추세) 또는 `stationary`.",
        "`engine: auto` picks `calibrated_lookup` when (paper reference device: V_G = −2 V, dark, no extensions, zero state centre) ∧ `action = gidl` ∧ $V_{D,\\max}=4$ V ∧ carrier noise on ∧ $\\Delta V = 10\\,\\mathrm{mV}/k$ ($k=1,2,\\dots$: 10, 5, 3.333, 2.5, 2, … mV), otherwise `general` (a warning when only the ΔV condition fails; an explicit `calibrated_lookup` request is an error). Mode mapping (calibrated engine): none → `fast_only`, frozen → `frozen`, evolving → `dynamic` (trend) or `stationary`.",
      ),
      notes: [
        L(
          "보정 엔진은 σ 비율 $\\sigma/\\sigma_{\\phi G}$ 하나로 두 상태 진폭을 함께 스케일하며 τ_E는 1.62 s로 고정된다(stochastic.py 경고).",
          "The calibrated engine scales both state amplitudes by one ratio $\\sigma/\\sigma_{\\phi G}$ and keeps τ_E = 1.62 s (stochastic.py warnings).",
        ),
        L(
          "ΔV 조건: `gate_dynamic_compare.simulate`는 0 → 4 V를 round(4/ΔV) 스텝으로 가며 round(0.01/ΔV) 스텝마다 전류를 읽어 고정된 401점(10 mV) 축에 붙인다. 10 mV/k가 아닌 ΔV는 트레이스 길이가 어긋나고 ΔV > 20 mV는 0으로 나눈다(`stochastic.calibrated_dv_ok`).",
          "ΔV condition: `gate_dynamic_compare.simulate` steps 0 → 4 V in round(4/ΔV) steps and reads the current every round(0.01/ΔV) steps onto a fixed 401-point (10 mV) axis. A ΔV that is not 10 mV/k gives traces of the wrong length and ΔV > 20 mV divides by zero (`stochastic.calibrated_dv_ok`).",
        ),
      ],
    },
    {
      heading: L("보정 조회표 엔진 (논문 소자)", "Calibrated-lookup engine (paper device)"),
      body: L(
        "`gate_state_lookup.npz`: $J=\\phi_G$ 7점(−0.66…0.66 V) × $E=\\phi_E$ 7점(−1.8…1.8 mV)에서 fold, 거리 0…0.32 V(1 mV)의 hazard, 거리별 로그 전류, 521점 분위수. 다중선형 보간(RegularGridInterpolator); hazard는 격자 밖과 0.32 V 밖에서 0, fold는 외삽. 상태 경로는 local-states의 OU/추세.",
        "`gate_state_lookup.npz`: fold, hazard versus distance 0…0.32 V (1 mV), log current versus distance and 521 quantiles on 7 $J=\\phi_G$ nodes (−0.66…0.66 V) × 7 $E=\\phi_E$ nodes (−1.8…1.8 mV). Multilinear interpolation (RegularGridInterpolator); hazard is 0 outside the grid and beyond 0.32 V, the fold is extrapolated. State paths are the OU/trend of local-states.",
      ),
      equations: [
        {
          id: "eq-mc-distance",
          label: L("fold까지의 거리", "distance to the fold"),
          tex: r`d_n = V^{\mathrm{fold}}_{LU}(\phi_n) - V_n\ \ (\text{up}),\qquad d_n = V_n - V^{\mathrm{fold}}_{LD}(\phi_n)\ \ (\text{down})`,
          code: "gate_dynamic_compare.py · simulate(): distance",
        },
        {
          id: "eq-mc-lookup-integral",
          label: L("누적 hazard와 사건", "integrated hazard and event"),
          tex: r`\begin{aligned} &\Lambda_n = \sum_{k=1}^{n} h(\phi_k, d_k)\,\Delta t\\ &n^\ast = \min\{n:\ \Lambda_n \ge \mathcal E\ \ \text{or}\ \ d_n \le 0\},\qquad \mathcal E\sim\mathrm{Exp}(1)\end{aligned}`,
          note: L(
            "$\\Delta t=\\Delta v/\\dot v$ (논문 2 mV / 0.4 V/s = 5 ms), 스윕당 2001점. 스윕 끝까지 사건이 없으면 censored(NaN); 미전환 상승은 HRS, 미전환 하강은 LRS로 남는다.",
            "$\\Delta t=\\Delta v/\\dot v$ (paper: 2 mV / 0.4 V/s = 5 ms), 2001 points per sweep. No event by the end of the sweep → censored (NaN); an unswitched up sweep stays HRS, an unswitched down sweep stays LRS.",
          ),
          code: "gate_dynamic_compare.py · simulate(): integrated, event",
        },
        {
          id: "eq-mc-readout",
          label: L("10 mV 판독 중점", "10 mV readout midpoint"),
          tex: r`V_{LU} = \Big\lfloor \frac{V_{n^\ast}}{0.01} \Big\rfloor \cdot 0.01 + 0.005\ \ \mathrm{V}`,
          note: L(
            "측정(10 mV 스텝의 low/high 중점)과 같은 판독. 연속값은 `V_LU_continuous`로 따로 반환된다.",
            "Same readout as the measurement (midpoint of the 10 mV low/high bracket). Continuous values are returned separately as `V_LU_continuous`.",
          ),
          code: "gate_dynamic_compare.py · simulate(): mid",
        },
      ],
      variables: [
        { symbol: r`\overline{V_{LU}},\ \sigma_{LU}`, name: L("100 sweep, seed 2026092920 (실행 검증)", "100 sweeps, seed 2026092920 (verified)"), value: "3.6344 V, 119.0 mV" },
        { symbol: r`\overline{V_{LD}},\ \sigma_{LD}`, name: L("같은 실행", "same run"), value: "2.6999 V, 21.4 mV" },
        { symbol: r`n_{\mathrm{cens}}`, name: L("censored (V_LU)", "censored (V_LU)"), value: "1 / 100" },
      ],
    },
    {
      heading: L("일반 엔진", "General engine"),
      body: L(
        "임의의 V_G, 빛, 확장, 램프, $V_{D,\\max}$에 대해 표를 만든다(디스크 캐시).\n\n- **fold 표**: 상태 $X_j=X_0+\\sigma t_j$, $t_j\\in[-4.5,4.5]$ (`fold_nodes`, 기본 25)에서 `classify` → PCHIP(4001점); latch 없는 노드 근처의 상태를 받은 사이클은 censored(`mc_cycles.py`는 latch 범위로 자른다).\n- **emitter 상태**: fold를 선형 이동, 기울기는 $\\pm\\sigma_E$ 중심 차분.\n- **hazard 노드**: Gauss–Hermite $t_i$ (`hazard_nodes`, 기본 5)에서 compound-FPT 곡선; 창 0.28 V (앞 10 mV의 탈출 확률 ≥ 1e-3이면 0.5, 0.8 V로 확장).\n- **난수**: seed에서 $Z$, $U$, $Z_E$, $U_2$ 순으로 뽑는 공통 난수(CRN); OU는 별도 스트림.",
        "Tables are built for any V_G, light, extension, ramp and $V_{D,\\max}$ (disk-cached).\n\n- **fold table**: `classify` at states $X_j=X_0+\\sigma t_j$, $t_j\\in[-4.5,4.5]$ (`fold_nodes`, default 25) → PCHIP (4001 points); a cycle whose state lands near a non-latching node is censored (`mc_cycles.py` clips to the latch range instead).\n- **emitter state**: shifts the folds linearly, slopes from central differences at $\\pm\\sigma_E$.\n- **hazard nodes**: compound-FPT curves at Gauss–Hermite $t_i$ (`hazard_nodes`, default 5); window 0.28 V (extended to 0.5, 0.8 V when the escape probability in its first 10 mV is ≥ 1e-3).\n- **random numbers**: common random numbers $Z$, $U$, $Z_E$, $U_2$ drawn in that order from the seed; the OU uses separate streams.",
      ),
      equations: [
        {
          id: "eq-mc-fold-shift",
          label: L("사이클 fold", "cycle fold"),
          tex: r`\begin{aligned} &V^{\mathrm{fold}}_{LU,LD} = f_{LU,LD}(X) + s_{LU,LD}\,E\\ &s_{LU,LD} = \frac{V^{\mathrm{fold}}_{LU,LD}(+\sigma_E) - V^{\mathrm{fold}}_{LU,LD}(-\sigma_E)}{2\sigma_E}\end{aligned}`,
          code: "stoch_mc.py · FoldTable, simulate_general(); stochastic.py · _general_tables()",
        },
        {
          id: "eq-mc-hazard-field",
          label: L("hazard 장", "hazard field"),
          tex: r`\begin{aligned} &\ln h(X,d) = \text{bilinear}\big[\ln h_i(d_m)\big],\qquad d_m = m\cdot 1\,\mathrm{mV}\\ &h = 0\ \ \text{for}\ d\le 0\ \text{or beyond the window}\end{aligned}`,
          note: L(
            "각 노드 곡선은 자기 fold까지의 거리로 바뀌고, 마지막 계산 전압과 fold 사이는 상수로 유지된다(조회표와 같은 규칙).",
            "Each node curve is re-expressed versus the distance to its own fold; between the last computed voltage and the fold the hazard is held constant (same rule as the lookup table).",
          ),
          code: "stoch_core.py · hazard_vs_distance(); stoch_mc.py · HazardField",
        },
        {
          id: "eq-mc-general-event",
          label: L("사건 (≤ 2 mV 부분 스텝의 사다리꼴 + 스텝 내 보간)", "event (trapezoid on ≤ 2 mV sub-steps + in-step interpolation)"),
          tex: r`\begin{aligned} &I_k = \sum_{m=1}^{k}\tfrac12\big(h_m+h_{m-1}\big)\delta t,\qquad \mathcal E = -\ln(1-U)\\ &V_{LU} = V_{k-1} + \frac{\mathcal E - I_{k-1}}{I_k - I_{k-1}}\,\delta v,\qquad \delta v = \frac{\Delta v}{s},\ \ s = \Big\lceil \frac{\Delta v}{2\,\mathrm{mV}} \Big\rceil\end{aligned}`,
          note: L(
            "hazard 적분과 fold 교차는 ΔV와 무관하게 최대 2 mV의 부분 스텝($\\delta t=\\delta v/\\dot v$)에서 계산한다. hazard가 fold 근처에서 10 mV마다 몇 자릿수씩 커지므로 거친 사다리꼴은 치우친다(논문 소자 ΔV = 50 mV에서 $\\overline{V_{LU}}$ −34 mV, $\\sigma_{LD}$ 2.3배). ΔV는 상태 표본화(OU 스텝, 스텝 안에서는 유지)만 정하며 ΔV ≤ 2 mV이면 $s=1$. fold 교차($d\\le0$)는 $d$의 선형 영점으로 두고 둘 중 먼저 오는 것을 택한다. 일반 엔진은 10 mV 판독 양자화를 하지 않는다. LD hazard는 중심 상태 곡선 하나(`ld_carrier_noise`), 끄면 LD는 fold에서 일어난다.",
            "The hazard integral and the fold crossing run on sub-steps of at most 2 mV ($\\delta t=\\delta v/\\dot v$) whatever ΔV: near the fold the hazard grows by orders of magnitude per 10 mV, so a coarse trapezoid is biased (paper device, ΔV = 50 mV: $\\overline{V_{LU}}$ −34 mV, $\\sigma_{LD}$ ×2.3). ΔV only sets the state sampling (OU step, held within a step); ΔV ≤ 2 mV gives $s=1$. A fold crossing ($d\\le0$) is placed at the linear zero of $d$; the earlier of the two wins. The general engine does not quantise to 10 mV. The LD hazard is a single centre-state curve (`ld_carrier_noise`); when off, latch-down happens at the fold.",
          ),
          code: "stoch_mc.py · _first_event()",
        },
      ],
      variables: [
        { symbol: r`V_{LU}`, name: L("일반 엔진, 논문 소자, evolving(추세 없음), 100 cycle, seed 2026092920 (실행)", "general engine, paper device, evolving (no trend), 100 cycles, seed 2026092920 (run)"), value: "3.6457 V, 108.5 mV" },
        { symbol: r`V_{LU}`, name: L("같은 조건, 상태 없음(none)", "same, no states (none)"), value: "3.6450 V, 8.3 mV" },
        { symbol: r`V_{LD}`, name: L("evolving, ld_carrier_noise 끔 (fold에서 latch-down)", "evolving, ld_carrier_noise off (latch-down at the fold)"), value: "2.5990 V, 18.7 mV" },
        { symbol: r`V_{LD}`, name: L("evolving, ld_carrier_noise 켬 (측정 2.7001 V, 19.5 mV)", "evolving, ld_carrier_noise on (measured 2.7001 V, 19.5 mV)"), value: "2.7007 V, 19.9 mV" },
      ],
      notes: [
        L(
          "두 프리셋(paper, photo; custom은 paper 복사) 모두 `ld_carrier_noise = true`가 기본이다(`server/params.py`). 끄면 일반 엔진에서 latch-down이 (상태에 따른) fold에서 일어나 논문 소자 0.4 V/s에서 $V_{LD}$ 평균이 약 0.10 V 낮아진다(측정 2.7001 V와 비교할 때는 켜 둔다). 보정 엔진은 이 스위치와 무관하게 항상 LD first-passage를 포함한다(끄면 경고만). 회로 시뮬레이터의 같은 이름 키는 별도 기본값(false)을 쓴다(`circuit-element`).",
          "Both presets (paper, photo; custom copies paper) default to `ld_carrier_noise = true` (`server/params.py`). Turned off, the general engine places latch-down at the (state-dependent) fold and the mean $V_{LD}$ of the paper device at 0.4 V/s drops by about 0.10 V (keep it on to compare with the measured 2.7001 V). The calibrated engine always includes the LD first passage regardless of the switch (off only adds a warning). The circuit simulator's key of the same name has its own default (false; `circuit-element`).",
        ),
        L(
          "구현 차이: 보정 엔진은 우측 합 $\\sum_{k\\ge1}h_k\\Delta t$와 격자 전압, 상승·하강에 독립 상태 경로(unpaired)를 쓴다. 일반 엔진은 ≤ 2 mV 부분 스텝의 사다리꼴과 스텝 내 보간, 한 사이클 안에서 상승→하강이 이어진 상태 경로(frozen: 같은 추출)를 쓴다.",
          "Implementation difference: the calibrated engine uses the right sum $\\sum_{k\\ge1}h_k\\Delta t$, grid voltages and independent up/down state paths (unpaired). The general engine uses the trapezoid on ≤ 2 mV sub-steps with in-step interpolation and one state path running up → down within a cycle (frozen: the same draw).",
        ),
        L(
          "획득 추세는 논문 기록 전용이라 일반 엔진에서는 무시된다. 트레이스는 가장 가까운 fold 노드의 준정적 HRS/LRS branch에 그 사이클의 전환 전압을 붙인 근사이다.",
          "The acquisition trend belongs to the paper record and is ignored by the general engine. Traces are approximate: quasi-static HRS/LRS branches of the nearest fold node with that cycle's switching voltages.",
        ),
      ],
    },
    {
      heading: L("광조사 소자 참조 방법 (mc_cycles.py)", "Photo-device reference method (mc_cycles.py)"),
      equations: [
        {
          id: "eq-mc-photo-cycles",
          label: L("가장 가까운 노드의 분위수를 fold로 이동", "nearest-node quantiles shifted to the fold"),
          tex: r`\begin{aligned} &V_c = Q_{i^\ast}(U_c) + \big(f(\delta_c) - f_{i^\ast}\big)\\ &\delta_c = \delta\phi_{G0} + \sigma_\phi Z_c,\qquad i^\ast = \arg\min_i\,\big|f_i - f(\delta_c)\big|\end{aligned}`,
          note: L(
            "$Q_i$: 9점 Gauss–Hermite 상태에서의 1200 V/s 탈출 분위수, $f$: 91점 fold 표 PCHIP. $Z_c$, $U_c$는 8개 조건에 공통.",
            "$Q_i$: 1200 V/s escape quantiles at 9 Gauss–Hermite states, $f$: 91-point fold table (PCHIP). $Z_c$, $U_c$ are shared by all 8 conditions.",
          ),
          code: "mc_cycles.py",
        },
      ],
    },
    {
      heading: L("V_G 곡선 (vg_sweep.py)", "V_G curve (vg_sweep.py)"),
      body: L(
        "각 V_G에서 frozen 상태의 Gaussian fold 혼합에 first-passage 잡음을 제곱합으로 더한다. evolving 상태는 정상 Gaussian 분포로 취급; emitter 상태는 포함하지 않는다. `sweep_mc`처럼 $V_{LU} > V_{D,\\max}$인 사이클은 censored: 평균과 σ는 스윕 안에서 latch되는 사이클에 대한 조건부 값이고, 그 비중은 `beyond_sweep_weight`, `censored_weight` (= `no_latch_weight` + beyond)로 보고된다.",
        "At each V_G the Gaussian fold mixture of frozen states is combined in quadrature with the first-passage noise. Evolving states are replaced by their stationary Gaussian; the emitter state is not included. As in `sweep_mc`, cycles with $V_{LU} > V_{D,\\max}$ are censored: mean and σ are conditional on latching within the sweep, and that weight is reported as `beyond_sweep_weight` and `censored_weight` (= `no_latch_weight` + beyond).",
      ),
      equations: [
        {
          id: "eq-mc-vg-mixture",
          label: L("fold 혼합 (81점)", "fold mixture (81 points)"),
          tex: r`\begin{aligned} &\bar f = \sum_{j\in\mathcal L} \tilde w_j\,f(\delta_{G0}+\sigma t_j),\qquad \sigma_{\mathrm{state}}^2 = \sum_{j\in\mathcal L}\tilde w_j\big(f_j-\bar f\big)^2\\ &t_j\in[-4,4]\ (81),\qquad \tilde w_j = e^{-t_j^2/2}\Big/\sum_{\mathcal L} e^{-t^2/2}\end{aligned}`,
          note: L(
            "$\\mathcal L$: latch가 있는 노드. `no_latch_weight` $=1-\\sum_{\\mathcal L}w_j$ (정규화 전 가중치).",
            "$\\mathcal L$: nodes with a latch. `no_latch_weight` $=1-\\sum_{\\mathcal L}w_j$ (weights before renormalisation).",
          ),
          code: "vg_sweep.py; stochastic.py · _vg_point()",
        },
        {
          id: "eq-mc-vg-total",
          label: L("평균과 σ", "mean and σ"),
          tex: r`\overline{V_{LU}} = \bar f + \sum_i \hat w_i\big(\overline{Q_i} - f_i\big),\qquad \sigma_{V_{LU}} = \sqrt{\sigma_{\mathrm{state}}^2 + \sum_i \hat w_i\,\mathrm{SD}^2[Q_i]}`,
          note: L(
            "$Q_i$: GH 노드(`vg_sweep.py` 5점)의 탈출 분위수, $\\hat w_i$: 정규화 GH 가중치. 웹 구현은 fold 표 PCHIP으로 81점 fold를 얻는다(원본은 81점마다 `classify`).",
            "$Q_i$: escape quantiles at the GH nodes (5 in `vg_sweep.py`), $\\hat w_i$: normalised GH weights. The web implementation takes the 81 folds from the PCHIP fold table (the original calls `classify` at each of the 81 points).",
          ),
          code: "vg_sweep.py; stochastic.py · _vg_point()",
        },
        {
          id: "eq-mc-vg-censored",
          label: L("스윕 최대로 자른 혼합", "mixture censored at the sweep maximum"),
          tex: r`\begin{aligned} &V = F + \varepsilon,\qquad \overline{V_{LU}} = \mathrm E\big[V \mid V\le V_{D,\max}\big]\\ &\sigma^2_{\mathrm{state}} = \mathrm{Var}_F\,\mathrm E\big[V\mid F,\,V\le V_{D,\max}\big],\qquad \sigma^2_{\mathrm{noise}} = \mathrm E_F\,\mathrm{Var}\big[V\mid F,\,V\le V_{D,\max}\big]\end{aligned}`,
          note: L(
            "$F$: 81점 fold 혼합, $\\varepsilon$: GH 노드의 분위수 $Q_i-\\overline{Q_i}$를 공통 평균 이동 $\\sum_i\\hat w_i(\\overline{Q_i}-f_i)$에 맞춘 합동 분포. 잘리는 질량이 없으면 위 식과 정확히 같다.",
            "$F$: the 81-point fold mixture, $\\varepsilon$: pooled GH-node quantiles $Q_i-\\overline{Q_i}$ recentred on the common mean shift $\\sum_i\\hat w_i(\\overline{Q_i}-f_i)$. With nothing censored this equals the formula above exactly.",
          ),
          code: "stochastic.py · truncated_mixture()",
        },
      ],
    },
    {
      heading: L("보고 통계", "Reported statistics"),
      body: L(
        "- 평균, SD (ddof 1), 중앙값, 5/95 % 분위, min/max — 유한값만\n- censored: 스윕 안에서 전환하지 않은 사이클 수\n- lag-1: 연속 유한 쌍의 Pearson 상관\n- 히스토그램(Freedman–Diaconis, 10–80 bin), 경험적 CDF(모든 사이클로 정규화: censored가 있으면 LU CDF는 latch된 비율에서 멈추고, 0 V까지 latch-down하지 않은 사이클은 LD CDF의 시작값으로 들어간다), 사이클별 상태(상승 스윕 중간)",
        "- mean, SD (ddof 1), median, 5/95 % quantiles, min/max — finite values only\n- censored: cycles that did not switch within the sweep\n- lag-1: Pearson correlation of consecutive finite pairs\n- histogram (Freedman–Diaconis, 10–80 bins), empirical CDF (normalised to all cycles: with censoring the LU CDF plateaus at the latched fraction, and cycles that never latch down by 0 V enter the LD CDF as its starting value), per-cycle state (middle of the up sweep)",
      ),
      notes: [
        L(
          "10 mV 판독은 분산에 약 $0.01^2/12$를 더한다(보정 시 포함). 상승·하강 기록은 짝지어지지 않는다(unpaired).",
          "The 10 mV readout adds about $0.01^2/12$ to the variance (included in the calibration). Up and down records are unpaired.",
        ),
      ],
    },
  ],
  related: ["local-states", "first-passage", "stochastic-events", "validation", "design-map", "numerics"],
  codeRefs: [CODE.gateDyn, CODE.gateLookup, CODE.stochastic, "server/compute/stoch_mc.py", CODE.stochCore, CODE.mcCycles, CODE.vgSweep],
};

export default topic;
