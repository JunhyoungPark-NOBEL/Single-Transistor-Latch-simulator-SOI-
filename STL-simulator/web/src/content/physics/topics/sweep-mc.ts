import type { PhysicsTopic } from "../types";
import { CODE, L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "sweep-mc",
  title: L("스윕 몬테카를로와 V_G 곡선", "Sweep Monte Carlo and the V_G curve"),
  summary: L(
    "삼각 스윕 0 → $V_{D,\\max}$ → 0의 각 사이클에서 국소 상태로 fold를 정하고, fold까지의 거리에 대한 첫 통과 hazard를 적분해 Exp(1) 문턱과 비교함으로써 $V_{LU}$, $V_{LD}$를 뽑는다. 기준 보정 조건에는 보정 조회표 엔진을, 그 밖의 조건에는 일반 엔진을 쓴다.",
    "In each cycle of the triangular sweep 0 → $V_{D,\\max}$ → 0 the local state sets the fold, and the first-passage hazard versus the distance to that fold is integrated against an Exp(1) threshold to draw $V_{LU}$ and $V_{LD}$. The reference-calibration condition uses the calibrated-lookup engine; any other condition uses the general engine.",
  ),
  tags: ["stochastic", "Monte Carlo", "V_LU", "V_LD"],
  sections: [
    {
      heading: L("엔진 선택", "Engine selection"),
      body: L(
        "`engine: auto`는 다음 조건을 모두 만족하면 `calibrated_lookup`을, 아니면 `general`을 고른다: 기준 보정 소자(V_G = −2 V, 암조건, 확장 없음, 상태 중심 0), `action = gidl`, $V_{D,\\max}=4$ V, 캐리어 잡음 켬, $\\Delta V = 10\\,\\mathrm{mV}/k$ ($k=1,2,\\dots$: 10, 5, 3.333, 2.5, 2, … mV). ΔV 조건만 어긋나면 경고를 내고, `calibrated_lookup`을 직접 지정했는데 조건이 맞지 않으면 오류다. 보정 엔진의 모드 대응: 없음 → `fast_only`, 고정 → `frozen`, 진화 → `dynamic`(추세 포함) 또는 `stationary`.",
        "`engine: auto` picks `calibrated_lookup` when all of the following hold, and `general` otherwise: reference-calibration device (V_G = −2 V, dark, no extensions, zero state center), `action = gidl`, $V_{D,\\max}=4$ V, carrier noise on, and $\\Delta V = 10\\,\\mathrm{mV}/k$ ($k=1,2,\\dots$: 10, 5, 3.333, 2.5, 2, … mV). A warning is issued when only the ΔV condition fails; an explicit `calibrated_lookup` request that does not qualify is an error. Mode mapping in the calibrated engine: none → `fast_only`, frozen → `frozen`, evolving → `dynamic` (with trend) or `stationary`.",
      ),
      notes: [
        L(
          "보정 엔진은 하나의 비율 $\\sigma/\\sigma_{\\phi G}$로 두 상태의 진폭을 함께 조정하며, τ_E는 1.62 s로 고정된다(stochastic.py 경고).",
          "The calibrated engine scales both state amplitudes by one ratio $\\sigma/\\sigma_{\\phi G}$ and keeps τ_E = 1.62 s (stochastic.py warnings).",
        ),
        L(
          "ΔV 조건: `gate_dynamic_compare.simulate`는 0 → 4 V를 round(4/ΔV) 스텝으로 진행하면서 round(0.01/ΔV) 스텝마다 전류를 읽어, 401점(10 mV 간격)으로 고정된 축에 기록한다. ΔV가 10 mV/k가 아니면 트레이스 길이가 맞지 않고, ΔV > 20 mV이면 0으로 나누게 된다(`stochastic.calibrated_dv_ok`).",
          "ΔV condition: `gate_dynamic_compare.simulate` steps from 0 to 4 V in round(4/ΔV) steps and reads the current every round(0.01/ΔV) steps onto a fixed 401-point (10 mV) axis. A ΔV that is not 10 mV/k gives traces of the wrong length, and ΔV > 20 mV divides by zero (`stochastic.calibrated_dv_ok`).",
        ),
      ],
    },
    {
      heading: L("보정 조회표 엔진 (기준 보정)", "Calibrated-lookup engine (reference calibration)"),
      body: L(
        "`gate_state_lookup.npz`에는 $J=\\phi_G$ 7개 노드(−0.66…0.66 V) × $E=\\phi_E$ 7개 노드(−1.8…1.8 mV)에서의 fold, 거리 0…0.32 V(1 mV 간격)에 대한 hazard, 거리별 로그 전류, 521개 분위수가 들어 있다. 다중 선형 보간(RegularGridInterpolator)을 쓰며, 격자 밖과 0.32 V 너머에서는 hazard를 0으로 두고 fold는 외삽한다. 상태 경로는 local-states의 OU 과정과 추세를 따른다.",
        "`gate_state_lookup.npz` holds the fold, the hazard versus distance 0…0.32 V (1 mV steps), the log current versus distance and 521 quantiles on 7 $J=\\phi_G$ nodes (−0.66…0.66 V) × 7 $E=\\phi_E$ nodes (−1.8…1.8 mV). Interpolation is multilinear (RegularGridInterpolator); the hazard is 0 outside the grid and beyond 0.32 V, and the fold is extrapolated. State paths follow the OU processes and trend of local-states.",
      ),
      equations: [
        {
          id: "eq-mc-distance",
          label: L("fold까지의 거리", "Distance to the fold"),
          tex: r`d_n = V^{\mathrm{fold}}_{LU}(\phi_n) - V_n\ \ (\text{up}),\qquad d_n = V_n - V^{\mathrm{fold}}_{LD}(\phi_n)\ \ (\text{down})`,
          code: "gate_dynamic_compare.py · simulate(): distance",
        },
        {
          id: "eq-mc-lookup-integral",
          label: L("누적 hazard와 사건", "Integrated hazard and event"),
          tex: r`\begin{aligned} &\Lambda_n = \sum_{k=1}^{n} h(\phi_k, d_k)\,\Delta t\\ &n^\ast = \min\{n:\ \Lambda_n \ge \mathcal E\ \ \text{or}\ \ d_n \le 0\},\qquad \mathcal E\sim\mathrm{Exp}(1)\end{aligned}`,
          note: L(
            "$\\Delta t=\\Delta v/\\dot v$이다(기준 보정: 2 mV / 0.4 V/s = 5 ms). 스윕당 2001점이다. 스윕이 끝날 때까지 사건이 없으면 중도절단(censored, NaN)으로 처리하며, 전환하지 않은 상승 스윕은 HRS에, 전환하지 않은 하강 스윕은 LRS에 남는다.",
            "$\\Delta t=\\Delta v/\\dot v$ (reference calibration: 2 mV / 0.4 V/s = 5 ms), 2001 points per sweep. If no event occurs by the end of the sweep, the cycle is censored (NaN); an up sweep without a switch stays in the HRS, a down sweep without a switch stays in the LRS.",
          ),
          code: "gate_dynamic_compare.py · simulate(): integrated, event",
        },
        {
          id: "eq-mc-readout",
          label: L("10 mV 판독의 중점", "10 mV readout midpoint"),
          tex: r`V_{LU} = \Big\lfloor \frac{V_{n^\ast}}{0.01} \Big\rfloor \cdot 0.01 + 0.005\ \ \mathrm{V}`,
          note: L(
            "측정과 같은 판독 방식이다(10 mV 스텝의 low/high 중점). 연속값은 `V_LU_continuous`로 따로 반환한다.",
            "Same readout as the measurement (midpoint of the 10 mV low/high bracket). Continuous values are returned separately as `V_LU_continuous`.",
          ),
          code: "gate_dynamic_compare.py · simulate(): mid",
        },
      ],
      variables: [
        { symbol: r`\overline{V_{LU}},\ \sigma_{LU}`, name: L("100회 스윕, 난수 시드 2026092920 (실행으로 확인)", "100 sweeps, seed 2026092920 (verified by running)"), value: "3.6344 V, 119.0 mV" },
        { symbol: r`\overline{V_{LD}},\ \sigma_{LD}`, name: L("같은 실행", "Same run"), value: "2.6999 V, 21.4 mV" },
        { symbol: r`n_{\mathrm{cens}}`, name: L("중도절단 (V_LU)", "Censored (V_LU)"), value: "1 / 100" },
      ],
    },
    {
      heading: L("일반 엔진", "General engine"),
      body: L(
        "임의의 V_G, 빛, 확장, 램프, $V_{D,\\max}$에 대해 표를 만든다(디스크 캐시).\n\n- **fold 표**: 상태 $X_j=X_0+\\sigma t_j$, $t_j\\in[-4.5,4.5]$(`fold_nodes`, 기본 25)에서 `classify`를 실행해 PCHIP(4001점)으로 만든다. 래치가 없는 노드 근처의 상태를 받은 사이클은 중도절단된다(`mc_cycles.py`는 대신 래치 범위로 자른다).\n- **이미터 상태**: fold를 선형으로 이동시키며, 기울기는 $\\pm\\sigma_E$에서의 중앙 차분으로 구한다.\n- **hazard 노드**: Gauss–Hermite 노드 $t_i$(`hazard_nodes`, 기본 5)에서 복합 FPT 곡선을 계산한다. 창은 0.28 V이며, 창의 첫 10 mV에서의 탈출 확률이 1e-3 이상이면 0.5 V, 필요하면 0.8 V로 넓힌다.\n- **난수**: 시드에서 $Z$, $U$, $Z_E$, $U_2$ 순서로 뽑는 공통 난수(CRN)를 쓰며, OU 과정은 별도의 난수 스트림을 쓴다.",
        "Tables are built for any V_G, light, extension, ramp and $V_{D,\\max}$ (cached on disk).\n\n- **Fold table**: `classify` at the states $X_j=X_0+\\sigma t_j$, $t_j\\in[-4.5,4.5]$ (`fold_nodes`, default 25) → PCHIP (4001 points); a cycle whose state lands near a non-latching node is censored (`mc_cycles.py` clips to the latch range instead).\n- **Emitter state**: shifts the folds linearly, with slopes from central differences at $\\pm\\sigma_E$.\n- **Hazard nodes**: compound-FPT curves at the Gauss–Hermite nodes $t_i$ (`hazard_nodes`, default 5); window 0.28 V, widened to 0.5 V and, if needed, 0.8 V when the escape probability in its first 10 mV is ≥ 1e-3.\n- **Random numbers**: common random numbers $Z$, $U$, $Z_E$, $U_2$, drawn in that order from the seed; the OU processes use separate streams.",
      ),
      equations: [
        {
          id: "eq-mc-fold-shift",
          label: L("사이클 fold", "Cycle fold"),
          tex: r`\begin{aligned} &V^{\mathrm{fold}}_{LU,LD} = f_{LU,LD}(X) + s_{LU,LD}\,E\\ &s_{LU,LD} = \frac{V^{\mathrm{fold}}_{LU,LD}(+\sigma_E) - V^{\mathrm{fold}}_{LU,LD}(-\sigma_E)}{2\sigma_E}\end{aligned}`,
          code: "stoch_mc.py · FoldTable, simulate_general(); stochastic.py · _general_tables()",
        },
        {
          id: "eq-mc-hazard-field",
          label: L("hazard 장", "Hazard field"),
          tex: r`\begin{aligned} &\ln h(X,d) = \text{bilinear}\big[\ln h_i(d_m)\big],\qquad d_m = m\cdot 1\,\mathrm{mV}\\ &h = 0\ \ \text{for}\ d\le 0\ \text{or beyond the window}\end{aligned}`,
          note: L(
            "각 노드의 곡선은 그 노드 자신의 fold까지의 거리에 대한 함수로 바꾼다. 마지막으로 계산한 전압과 fold 사이에서는 hazard를 상수로 유지한다(조회표와 같은 규칙).",
            "Each node curve is re-expressed as a function of the distance to its own fold; between the last computed voltage and the fold the hazard is held constant (same rule as in the lookup table).",
          ),
          code: "stoch_core.py · hazard_vs_distance(); stoch_mc.py · HazardField",
        },
        {
          id: "eq-mc-general-event",
          label: L("사건 (≤ 2 mV 부분 스텝의 사다리꼴 적분 + 스텝 내 보간)", "Event (trapezoid on ≤ 2 mV sub-steps + in-step interpolation)"),
          tex: r`\begin{aligned} &I_k = \sum_{m=1}^{k}\tfrac12\big(h_m+h_{m-1}\big)\delta t,\qquad \mathcal E = -\ln(1-U)\\ &V_{LU} = V_{k-1} + \frac{\mathcal E - I_{k-1}}{I_k - I_{k-1}}\,\delta v,\qquad \delta v = \frac{\Delta v}{s},\ \ s = \Big\lceil \frac{\Delta v}{2\,\mathrm{mV}} \Big\rceil\end{aligned}`,
          note: L(
            "hazard 적분과 fold 교차는 ΔV와 상관없이 최대 2 mV의 부분 스텝($\\delta t=\\delta v/\\dot v$)에서 계산한다. fold 근처에서는 hazard가 10 mV마다 몇 자릿수씩 커지므로 거친 사다리꼴 적분은 편향된다(기준 보정, ΔV = 50 mV에서 $\\overline{V_{LU}}$ −34 mV, $\\sigma_{LD}$ 2.3배). ΔV는 상태 표본화(OU 스텝, 스텝 안에서는 유지)만 정하며, ΔV ≤ 2 mV이면 $s=1$이다. fold 교차($d\\le0$)는 $d$의 선형 영점에 두고, 두 사건 중 먼저 오는 것을 택한다. 일반 엔진은 10 mV 판독 양자화를 하지 않는다. LD hazard는 중심 상태의 곡선 하나만 쓰며(`ld_carrier_noise`), 이를 끄면 래치다운은 fold에서 일어난다.",
            "The hazard integral and the fold crossing are evaluated on sub-steps of at most 2 mV ($\\delta t=\\delta v/\\dot v$) regardless of ΔV: near the fold the hazard grows by orders of magnitude per 10 mV, so a coarse trapezoid is biased (reference calibration, ΔV = 50 mV: $\\overline{V_{LU}}$ −34 mV, $\\sigma_{LD}$ ×2.3). ΔV only sets the state sampling (OU step, held within a step); ΔV ≤ 2 mV gives $s=1$. A fold crossing ($d\\le0$) is placed at the linear zero of $d$, and the earlier of the two events wins. The general engine does not quantize to 10 mV. The LD hazard is a single center-state curve (`ld_carrier_noise`); when it is off, latch-down happens at the fold.",
          ),
          code: "stoch_mc.py · _first_event()",
        },
      ],
      variables: [
        { symbol: r`V_{LU}`, name: L("일반 엔진, 기준 보정, 진화(추세 없음), 100 사이클, 난수 시드 2026092920 (실행)", "General engine, reference calibration, evolving (no trend), 100 cycles, seed 2026092920 (run)"), value: "3.6457 V, 108.5 mV" },
        { symbol: r`V_{LU}`, name: L("같은 조건, 상태 없음(none)", "Same, no states (none)"), value: "3.6450 V, 8.3 mV" },
        { symbol: r`V_{LD}`, name: L("진화, ld_carrier_noise 끔 (fold에서 래치다운)", "Evolving, ld_carrier_noise off (latch-down at the fold)"), value: "2.5990 V, 18.7 mV" },
        { symbol: r`V_{LD}`, name: L("진화, ld_carrier_noise 켬 (측정 2.7001 V, 19.5 mV)", "Evolving, ld_carrier_noise on (measured 2.7001 V, 19.5 mV)"), value: "2.7007 V, 19.9 mV" },
      ],
      notes: [
        L(
          "두 프리셋(기준 보정, 광조사 보정; 사용자 정의는 기준 보정을 복사) 모두 `ld_carrier_noise = true`가 기본값이다(`server/params.py`). 이를 끄면 일반 엔진은 래치다운을 (상태에 따라 달라지는) fold에 두므로, 기준 보정 0.4 V/s에서 $V_{LD}$ 평균이 약 0.10 V 낮아진다(측정값 2.7001 V와 비교할 때는 켜 둔다). 보정 엔진은 이 스위치와 관계없이 항상 LD 첫 통과를 포함한다(끄면 경고만 낸다). 회로 시뮬레이터의 같은 이름의 키는 기본값이 따로 있다(false; `circuit-element`).",
          "Both presets (reference and illumination calibration; custom copies the reference calibration) default to `ld_carrier_noise = true` (`server/params.py`). When it is off, the general engine places latch-down at the (state-dependent) fold, and the mean $V_{LD}$ of the reference calibration at 0.4 V/s drops by about 0.10 V (keep it on when comparing with the measured 2.7001 V). The calibrated engine always includes the LD first passage regardless of the switch (turning it off only adds a warning). The circuit simulator's key of the same name has its own default (false; `circuit-element`).",
        ),
        L(
          "구현상의 차이: 보정 엔진은 우측 합 $\\sum_{k\\ge1}h_k\\Delta t$와 격자 전압을 쓰고, 상승과 하강에 서로 독립인 상태 경로를 쓴다(짝짓지 않음). 일반 엔진은 ≤ 2 mV 부분 스텝의 사다리꼴 적분과 스텝 내 보간을 쓰고, 한 사이클 안에서 상승에서 하강으로 이어지는 하나의 상태 경로를 쓴다(고정 모드는 같은 추출값).",
          "Implementation differences: the calibrated engine uses the right sum $\\sum_{k\\ge1}h_k\\Delta t$, grid voltages and independent up/down state paths (unpaired). The general engine uses the trapezoid on ≤ 2 mV sub-steps with in-step interpolation and a single state path running from up to down within a cycle (frozen mode: the same draw).",
        ),
        L(
          "획득 추세는 기준 측정 기록에만 해당하므로 일반 엔진은 이를 무시한다. 트레이스는 근사다: 가장 가까운 fold 노드의 준정적 HRS/LRS branch에 그 사이클의 전환 전압을 붙인 것이다.",
          "The acquisition trend belongs to the reference record and is ignored by the general engine. Traces are approximate: the quasi-static HRS/LRS branches of the nearest fold node, joined at that cycle's switching voltages.",
        ),
      ],
    },
    {
      heading: L("광조사 보정의 참조 방법 (mc_cycles.py)", "Illumination-calibration reference method (mc_cycles.py)"),
      equations: [
        {
          id: "eq-mc-photo-cycles",
          label: L("가장 가까운 노드의 분위수를 fold로 이동", "Nearest-node quantiles shifted to the fold"),
          tex: r`\begin{aligned} &V_c = Q_{i^\ast}(U_c) + \big(f(\delta_c) - f_{i^\ast}\big)\\ &\delta_c = \delta\phi_{G0} + \sigma_\phi Z_c,\qquad i^\ast = \arg\min_i\,\big|f_i - f(\delta_c)\big|\end{aligned}`,
          note: L(
            "$Q_i$는 9개 Gauss–Hermite 상태에서의 1200 V/s 탈출 분위수이고, $f$는 91점 fold 표의 PCHIP이다. $Z_c$, $U_c$는 8개 조건이 함께 쓴다.",
            "$Q_i$: 1200 V/s escape quantiles at 9 Gauss–Hermite states; $f$: 91-point fold table (PCHIP). $Z_c$ and $U_c$ are shared by all 8 conditions.",
          ),
          code: "mc_cycles.py",
        },
      ],
    },
    {
      heading: L("V_G 곡선 (vg_sweep.py)", "V_G curve (vg_sweep.py)"),
      body: L(
        "각 V_G에서 고정 상태의 가우스 fold 혼합에 첫 통과 잡음을 제곱합으로 더한다. 진화 상태는 정상 가우스 분포로 대신하며, 이미터 상태는 포함하지 않는다. `sweep_mc`와 마찬가지로 $V_{LU} > V_{D,\\max}$인 사이클은 중도절단된다. 즉 평균과 σ는 스윕 안에서 래치되는 사이클에 대한 조건부 값이고, 절단된 비중은 `beyond_sweep_weight`, `censored_weight`(= `no_latch_weight` + beyond)로 보고한다.",
        "At each V_G the Gaussian fold mixture of frozen states is combined in quadrature with the first-passage noise. Evolving states are replaced by their stationary Gaussian; the emitter state is not included. As in `sweep_mc`, cycles with $V_{LU} > V_{D,\\max}$ are censored: the mean and σ are conditional on latching within the sweep, and the censored weight is reported as `beyond_sweep_weight` and `censored_weight` (= `no_latch_weight` + beyond).",
      ),
      equations: [
        {
          id: "eq-mc-vg-mixture",
          label: L("fold 혼합 (81점)", "Fold mixture (81 points)"),
          tex: r`\begin{aligned} &\bar f = \sum_{j\in\mathcal L} \tilde w_j\,f(\delta_{G0}+\sigma t_j),\qquad \sigma_{\mathrm{state}}^2 = \sum_{j\in\mathcal L}\tilde w_j\big(f_j-\bar f\big)^2\\ &t_j\in[-4,4]\ (81),\qquad \tilde w_j = e^{-t_j^2/2}\Big/\sum_{\mathcal L} e^{-t^2/2}\end{aligned}`,
          note: L(
            "$\\mathcal L$은 래치가 있는 노드의 집합이다. `no_latch_weight` $=1-\\sum_{\\mathcal L}w_j$이다(정규화 전 가중치).",
            "$\\mathcal L$: the set of nodes with a latch. `no_latch_weight` $=1-\\sum_{\\mathcal L}w_j$ (weights before renormalization).",
          ),
          code: "vg_sweep.py; stochastic.py · _vg_point()",
        },
        {
          id: "eq-mc-vg-total",
          label: L("평균과 σ", "Mean and σ"),
          tex: r`\overline{V_{LU}} = \bar f + \sum_i \hat w_i\big(\overline{Q_i} - f_i\big),\qquad \sigma_{V_{LU}} = \sqrt{\sigma_{\mathrm{state}}^2 + \sum_i \hat w_i\,\mathrm{SD}^2[Q_i]}`,
          note: L(
            "$Q_i$는 GH 노드(`vg_sweep.py`에서 5개)의 탈출 분위수, $\\hat w_i$는 정규화한 GH 가중치다. 웹 구현은 81개 fold를 fold 표의 PCHIP에서 얻는다(원본은 81점마다 `classify`를 호출한다).",
            "$Q_i$: escape quantiles at the GH nodes (5 in `vg_sweep.py`); $\\hat w_i$: normalized GH weights. The web implementation takes the 81 folds from the PCHIP fold table (the original calls `classify` at each of the 81 points).",
          ),
          code: "vg_sweep.py; stochastic.py · _vg_point()",
        },
        {
          id: "eq-mc-vg-censored",
          label: L("스윕 최대 전압에서 절단한 혼합", "Mixture censored at the sweep maximum"),
          tex: r`\begin{aligned} &V = F + \varepsilon,\qquad \overline{V_{LU}} = \mathrm E\big[V \mid V\le V_{D,\max}\big]\\ &\sigma^2_{\mathrm{state}} = \mathrm{Var}_F\,\mathrm E\big[V\mid F,\,V\le V_{D,\max}\big],\qquad \sigma^2_{\mathrm{noise}} = \mathrm E_F\,\mathrm{Var}\big[V\mid F,\,V\le V_{D,\max}\big]\end{aligned}`,
          note: L(
            "$F$는 81점 fold 혼합이고, $\\varepsilon$은 GH 노드의 분위수 $Q_i-\\overline{Q_i}$를 합친 뒤 공통 평균 이동 $\\sum_i\\hat w_i(\\overline{Q_i}-f_i)$에 맞춰 다시 중심을 잡은 분포다. 절단되는 확률 질량이 없으면 위 식과 정확히 같다.",
            "$F$: the 81-point fold mixture; $\\varepsilon$: the pooled GH-node quantiles $Q_i-\\overline{Q_i}$, recentered on the common mean shift $\\sum_i\\hat w_i(\\overline{Q_i}-f_i)$. With no censored probability mass this equals the formula above exactly.",
          ),
          code: "stochastic.py · truncated_mixture()",
        },
      ],
    },
    {
      heading: L("보고하는 통계량", "Reported statistics"),
      body: L(
        "- 평균, SD(ddof 1), 중앙값, 5/95 % 분위수, 최솟값/최댓값 — 유한값만 사용\n- 중도절단: 스윕 안에서 전환하지 않은 사이클 수\n- lag-1: 연속한 유한값 쌍의 Pearson 상관\n- 히스토그램(Freedman–Diaconis, 10–80 bin), 경험적 CDF(전체 사이클 수로 정규화: 중도절단이 있으면 LU CDF는 래치된 비율에서 평탄해지고, 0 V까지 래치다운하지 않은 사이클은 LD CDF의 시작값으로 들어간다), 사이클별 상태(상승 스윕의 중간 시점)",
        "- Mean, SD (ddof 1), median, 5/95 % quantiles, min/max — finite values only\n- Censored: cycles that did not switch within the sweep\n- lag-1: Pearson correlation of consecutive finite pairs\n- Histogram (Freedman–Diaconis, 10–80 bins), empirical CDF (normalized to all cycles: with censoring the LU CDF plateaus at the latched fraction, and cycles that never latch down by 0 V enter the LD CDF as its starting value), per-cycle state (middle of the up sweep)",
      ),
      notes: [
        L(
          "10 mV 판독은 분산에 약 $0.01^2/12$를 더한다(보정 시 포함). 상승 기록과 하강 기록은 짝지어지지 않는다(unpaired).",
          "The 10 mV readout adds about $0.01^2/12$ to the variance (included in the calibration). Up and down records are unpaired.",
        ),
      ],
    },
  ],
  related: ["local-states", "first-passage", "stochastic-events", "validation", "design-map", "numerics"],
  codeRefs: [CODE.gateDyn, CODE.gateLookup, CODE.stochastic, "server/compute/stoch_mc.py", CODE.stochCore, CODE.mcCycles, CODE.vgSweep],
};

export default topic;
