import type { PhysicsTopic } from "../types";
import { CODE, L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "first-passage",
  title: L("First passage: 정공 격자, 후진 방정식, hazard", "First passage: hole lattice, backward equation, hazard"),
  summary: L(
    "고정 $V_D$에서 body 정공 수 $x=Q/q$ 위의 복합 점프 과정(Eq. 2)을 세우고, 평균 first-passage 시간 $T$를 후진 방정식으로 정확히 풀어 hazard $h(V_D)=1/T(x_w)$를 얻는다. 램프를 따라 $h/\\dot v$를 적분해 $V_{LU}$ 분포를 만든다.",
    "At fixed $V_D$ the compound-jump process of Eq. 2 is set up on the body hole count $x=Q/q$; the mean first-passage time $T$ is solved exactly from the backward equation, giving the hazard $h(V_D)=1/T(x_w)$. Integrating $h/\\dot v$ along the ramp gives the $V_{LU}$ distribution.",
  ),
  tags: ["FPT", "stochastic", "backward equation"],
  sections: [
    {
      heading: L("정공 수 격자", "Hole-count lattice"),
      body: L(
        "각 $u$(고정 $V_D$)에서 $r$을 풀고 전하 좌표를 계산한다. $F=q(G-L)$의 $+\\to-$ 영점이 well $u_w$, 그 위(LU)의 $-\\to+$ 영점이 saddle $u_s$이다. 흡수 경계는 $I_D$가 10 nA를 넘는 $u_{th}$이다. 격자는 정수 정공 수이며 $n\\ge N$ 상태는 흡수된다.",
        "At each $u$ (fixed $V_D$) $r$ is solved and the charge coordinate evaluated. The $+\\to-$ zero of $F=q(G-L)$ is the well $u_w$, the next $-\\to+$ zero above it (LU) the saddle $u_s$. The absorbing boundary is $u_{th}$ where $I_D$ exceeds 10 nA. The lattice is the integer hole count; states $n\\ge N$ are absorbing.",
      ),
      equations: [
        {
          id: "eq-fpt-charge",
          label: L("전하 좌표", "charge coordinate"),
          tex: r`\begin{aligned} &Q(u) = C_{ox}\,\psi + Q_{\mathrm{exc}} + qN_A A L_n,\qquad x = Q/q\\ &\psi = u - V_T\ln\!\big(1+\delta/N_A\big)\end{aligned}`,
          note: L(
            "$Q_{\\mathrm{exc}} = Q_{\\mathrm{pair}}+Q_{\\mathrm{acc}} = q\\bar p\\,(A L_n + W t_{\\mathrm{acc}}L_{\\mathrm{acc}})$ (= `z[13]` − $C_{ox}u$), $L_n$은 중성 body 길이 `z[11]`. 모두 $r(u)$에서 평가한다. 회로 소자의 $Q$와는 상수 $-C_{ox}V_G$만 다르다.",
            "$Q_{\\mathrm{exc}} = Q_{\\mathrm{pair}}+Q_{\\mathrm{acc}} = q\\bar p\\,(A L_n + W t_{\\mathrm{acc}}L_{\\mathrm{acc}})$ (= `z[13]` − $C_{ox}u$), $L_n$ is the neutral body length `z[11]`; all evaluated at $r(u)$. It differs from the circuit element's $Q$ only by the constant $-C_{ox}V_G$.",
          ),
          code: "setup_photo.py · state() cols [5],[6],[7]; compound_fpt.py · make_lattice() q",
        },
        {
          id: "eq-fpt-lattice",
          label: L("격자, well, 흡수 경계 (LU)", "lattice, well, absorbing boundary (LU)"),
          tex: r`\begin{aligned} &x_n = \lceil x(0.1\,\mathrm{V})\rceil + n,\quad n = 0,\dots,N-1\\ &N = \lceil x(u_{th})\rceil - \lceil x(0.1\,\mathrm{V})\rceil,\qquad I_D(u_{th}) = 10^{-8}\,\mathrm{A}\end{aligned}`,
          note: L(
            "시작 상태 $n_0=\\mathrm{round}(x(u_w)-x_0)$. 사건률과 $r$은 $x$에 대해 PCHIP(사건률은 $\\ln$) 보간. LD: $x=-Q/q$(뒤집은 행), 상한 $u=\\min(u_w+0.06,\\,u_{\\max}-0.002)$, 흡수 $u_a=\\min(u_{th},\\,u_s-0.05)$.",
            "Start state $n_0=\\mathrm{round}(x(u_w)-x_0)$. Rates and $r$ are PCHIP-interpolated in $x$ (rates in $\\ln$). LD: $x=-Q/q$ (reversed rows), upper bound $u=\\min(u_w+0.06,\\,u_{\\max}-0.002)$, absorption at $u_a=\\min(u_{th},\\,u_s-0.05)$.",
          ),
          code: "compound_fpt.py · make_lattice(rows, direction, upper_extra=.06)",
        },
      ],
      variables: [
        { symbol: r`C_{ox}`, name: L("게이트 산화막 용량 (W·L)", "gate-oxide capacitance (W·L)"), value: "0.2449", unit: "fF", code: "m.COX_F" },
        { symbol: r`A`, name: L("단면적 W·T_Si", "cross-section W·T_Si"), value: "1e-10", unit: "cm²", code: "m.AREA_CM2" },
        { symbol: r`qN_AA`, name: L("중성 길이당 전하", "charge per neutral length"), value: "3.678e-12", unit: "C/cm" },
        { symbol: r`u`, name: L("u 격자: [0.1, 0.9] 181점 ∪ u_fold ± 0.065 V 61점", "u grid: 181 pts in [0.1, 0.9] ∪ 61 pts in u_fold ± 0.065 V"), unit: "V", code: "photo_fpt.hazard_curve() ug" },
        { symbol: r`N`, name: L("상태 수 (V_G = −2 V, V_D = 3.64 V; x ≈ 766…1849, well x ≈ 1490)", "number of states (V_G = −2 V, V_D = 3.64 V; x ≈ 766…1849, well x ≈ 1490)"), value: "1084" },
        { symbol: r`C_{\mathrm{eff}}`, name: L("well에서 dQ/du", "dQ/du at the well"), value: "0.271", unit: "fF" },
      ],
    },
    {
      heading: L("복합 점프 생성자와 후진 방정식", "Compound-jump generator and backward equation"),
      body: L(
        "상태 $n$에서의 점프: 단위 사건 $n\\to n+s$ (률 $\\lambda_{\\mathrm{unit}}$), 손실 $n\\to n-s$ ($L$), 클러스터 $n\\to n+sk$ ($\\lambda_k$); LU는 $s=+1$, LD는 $s=-1$. 목표는 $\\max(\\cdot,0)$으로 잘리고 자기 자신으로의 점프는 제거된다(아래 경계 반사). $j\\ge N$은 흡수.",
        "Jumps from state $n$: unit events $n\\to n+s$ (rate $\\lambda_{\\mathrm{unit}}$), losses $n\\to n-s$ ($L$), clusters $n\\to n+sk$ ($\\lambda_k$); $s=+1$ for LU, $s=-1$ for LD. Targets are clipped by $\\max(\\cdot,0)$ and self-jumps removed (reflecting lower boundary); $j\\ge N$ is absorbing.",
      ),
      equations: [
        {
          id: "eq-fpt-backward",
          label: L("평균 first-passage 시간", "mean first-passage time"),
          tex: r`\sum_{j} W_{n\to j}\,\big(T_j - T_n\big) = -1,\qquad T_j = 0\ \ (j\ge N)`,
          code: "compound_fpt.py · backward()",
        },
        {
          id: "eq-fpt-normalised",
          label: L("코드가 푸는 정규화 형태", "normalised form solved by the code"),
          tex: r`T_n - \sum_{j<N}\frac{W_{n\to j}}{W_n}\,T_j = \frac{1}{W_n},\qquad W_n = \sum_{j\neq n} W_{n\to j}`,
          note: L(
            "희소 행렬 $A=I-P$ (COO→CSC)로 `spsolve(A, 1/total)`. 잔차 $\\max|AT-1/W|/\\max|T|$를 `normalized_linear_residual`로 보고(≈4e-16).",
            "Sparse matrix $A=I-P$ (COO→CSC), `spsolve(A, 1/total)`. The residual $\\max|AT-1/W|/\\max|T|$ is reported as `normalized_linear_residual` (≈ 4e-16).",
          ),
          code: "compound_fpt.py · backward(): A, t = spsolve(A, 1/total)",
        },
        {
          id: "eq-fpt-hazard",
          label: L("준정적 hazard", "quasi-static hazard"),
          tex: r`h(V_D) = \frac{1}{T_{n_0}(V_D)}`,
          code: "photo_fpt.py · hazard_curve(); gate_fpt.py · one()",
        },
      ],
    },
    {
      heading: L("램프 적분, 분위수, fold atom", "Ramp integration, quantiles, fold atom"),
      body: L(
        "노드 $V_D$ = $\\max(V_{LU}-w,\\,V_{LD}+0.001)$ … $V_{LU}-0.001$, 4 mV 간격, 창 $w=0.28$ V (`photo_fpt`; `gate_fpt`는 $V_G>-1.11$ V에서 0.5 V; 웹 서버는 창 앞 10 mV의 탈출 확률 $h_0\\cdot0.01/\\dot v\\ge10^{-3}$이면 0.5, 0.8 V로 넓힌다). 풀리지 않는 노드는 건너뛴다. 격자의 역바이어스 $r$이 애벌랜치 클러스터 커널 범위(0.7–5.0 V)를 벗어나는 노드도 건너뛴다($V_D \\gtrsim 5.1$ V, 예: $l_{\\mathrm{GIDL}} \\gtrsim 42$ nm로 fold가 5.1 V 위로 올라간 경우) — 그 구간의 탈출은 fold에 놓이고 경고가 원인을 밝힌다(`kernel_skipped`). 행은 `S.state`와 같되, 실패하는 곳에서는 $r$ 구간을 유한 영역으로 줄인다(`deterministic.state_row`). 먼 꼬리는 희소 풀이의 상쇄 오차 때문에 버리고, $h\\ge10^{-4}$ s⁻¹인 마지막 연결 구간만 남긴다.",
        "Nodes $V_D$ = $\\max(V_{LU}-w,\\,V_{LD}+0.001)$ … $V_{LU}-0.001$ in 4 mV steps, window $w=0.28$ V (`photo_fpt`; `gate_fpt` uses 0.5 V for $V_G>-1.11$ V; the web server widens it to 0.5, 0.8 V when the escape probability in the first 10 mV, $h_0\\cdot0.01/\\dot v$, is $\\ge10^{-3}$). Unresolved nodes are skipped. So are nodes whose lattice reverse bias $r$ leaves the avalanche cluster kernel range (0.7–5.0 V; $V_D \\gtrsim 5.1$ V, e.g. a fold pushed above 5.1 V by $l_{\\mathrm{GIDL}} \\gtrsim 42$ nm): escapes there are placed at the fold and a warning names the cause (`kernel_skipped`). Rows are `S.state`, with the $r$ bracket shrunk to its finite part where `S.state` fails (`deterministic.state_row`). The far tail is dropped (cancellation in the sparse solve) and only the final connected region with $h\\ge10^{-4}$ s⁻¹ is kept.",
      ),
      equations: [
        {
          id: "eq-fpt-retained",
          label: L("유지 영역", "retained domain"),
          tex: r`h_k \leftarrow 0\ \ (k<k_0),\qquad k_0 = 1 + \max\{k:\ h_k\ \text{not finite or}\ h_k < 10^{-4}\,\mathrm{s^{-1}}\}`,
          code: "photo_fpt.py · quantiles(); compound_fpt.py · run_direction()",
        },
        {
          id: "eq-fpt-survival",
          label: L("생존 확률", "survival along the ramp"),
          tex: r`S(V) = e^{-\Lambda(V)},\qquad \Lambda(V) = \int_{V_0}^{V}\frac{h(V')}{\dot v}\,dV'\ \ \text{(trapezoid on the nodes)}`,
          code: "cumulative_trapezoid(h/rate, V, initial=0)",
        },
        {
          id: "eq-fpt-quantile",
          label: L("탈출 분위수 (10001점)", "escape quantiles (10001 points)"),
          tex: r`V_m = \Lambda^{-1}\!\big(-\ln(1-P_m)\big),\qquad P_m = \frac{m+\tfrac12}{10001},\ m = 0,\dots,10000`,
          note: L(
            "$-\\ln(1-P_m)>\\Lambda(V_{\\mathrm{last}})$이면 $V_m=V_{LU}^{\\mathrm{fold}}$ (np.interp `right=fold`). 평균·SD는 10001개 분위수의 모멘트(SD는 ddof 0).",
            "If $-\\ln(1-P_m)>\\Lambda(V_{\\mathrm{last}})$ then $V_m=V_{LU}^{\\mathrm{fold}}$ (np.interp `right=fold`). Mean and SD are moments of the 10001 quantiles (SD with ddof 0).",
          ),
          code: "photo_fpt.py · quantiles(rec, rate), PROB",
        },
        {
          id: "eq-fpt-atom",
          label: L("fold atom", "fold atom"),
          tex: r`P_{\mathrm{atom}} = S(V_{\mathrm{last}}) = e^{-\Lambda(V_{\mathrm{last}})}`,
          note: L(
            "fold까지 살아남은 확률은 fold에 놓인 결정론적 atom으로 명시 보고된다(버리거나 재정규화하지 않음). 이 모델에는 fold 이후 지연이 없다.",
            "The probability of surviving to the fold is reported as a deterministic atom at the fold (neither dropped nor renormalised). The model has no post-fold delay.",
          ),
          code: "gate_fpt.py · one(): fold_atom",
        },
      ],
      variables: [
        { symbol: r`V_{LU}^{\mathrm{fold}}`, name: L("V_G = −2 V 암조건", "V_G = −2 V dark"), value: "3.7037", unit: "V" },
        { symbol: r`\overline{V_{LU}}`, name: L("FPT 평균 (0.4 V/s, 중심 상태)", "FPT mean (0.4 V/s, centre states)"), value: "3.6442", unit: "V" },
        { symbol: r`\mathrm{SD}`, name: L("FPT 표준편차 (검증: 실행값)", "FPT SD (verified by running)"), value: "8.03", unit: "mV" },
        { symbol: r`V_{k_0}`, name: L("유지 영역 시작 (70개 노드 중 인덱스 39)", "retained-domain start (index 39 of 70 nodes)"), value: "3.580", unit: "V" },
        { symbol: r`P_{\mathrm{atom}}`, name: L("fold atom (V_G = −2 V, 0.4 V/s; 1200 V/s에서는 0.90)", "fold atom (V_G = −2 V, 0.4 V/s; 0.90 at 1200 V/s)"), value: "2e-141" },
      ],
      notes: [
        L(
          "LD 방향: 노드 $V_{LD}+0.30$ … $V_{LD}+0.001$ V (하강), $u$ 격자 [0.55, 1.04] 181점 ∪ $u_{\\mathrm{fold}}\\pm0.065$ V, 적분 변수 $-V$ (`conditional_table.calculate`, `stoch_core.ld_hazard_curve`). V_G = −2 V에서 LD 탈출은 fold(2.598 V)보다 약 0.10 V 위(평균 ≈ 2.70 V)에서 일어난다.",
          "LD direction: nodes $V_{LD}+0.30$ … $V_{LD}+0.001$ V (descending), $u$ grid of 181 points in [0.55, 1.04] ∪ $u_{\\mathrm{fold}}\\pm0.065$ V, integration variable $-V$ (`conditional_table.calculate`, `stoch_core.ld_hazard_curve`). At V_G = −2 V the LD escape happens about 0.10 V above the fold (2.598 V), mean ≈ 2.70 V.",
        ),
      ],
    },
    {
      heading: L("준안정 근사와 주의사항", "Metastable approximation and caveats"),
      equations: [
        {
          id: "eq-fpt-metastable",
          label: L("준정적 조건", "quasi-static condition"),
          tex: r`\tau_{\mathrm{rel}} = -\frac{dQ/du}{\partial F/\partial u}\Big|_{u_w},\qquad \tau_{\mathrm{rel}}\,\dot v\,\Big|\frac{d\ln h}{dV}\Big| \ll 1,\quad \tau_{\mathrm{rel}}\,h \ll 1`,
          note: L(
            "$\\tau_{\\mathrm{rel}}$은 `lu_fpt.generator_for`와 같은 격자 정의($x=Q/q$, $F=G-L$ [1/s]; well에서 $-(dx/du)/(dF/du)$)로 계산한 값이다. V_G = −2 V 암조건: $V_D$ = 3.64 V에서 12 µs, fold 4 mV 아래의 마지막 노드(3.6997 V)에서 28 µs(fold 근처 임계 감속). 0.4 V/s에서는 탈출 구간에서 $\\tau_{\\mathrm{rel}}\\dot v|d\\ln h/dV| < 10^{-3}$, $\\tau_{\\mathrm{rel}}h \\lesssim 3\\times10^{-3}$(99 % 분위까지)으로 근사가 잘 맞는다(fold atom $2\\times10^{-141}$).\n\n1200 V/s: 유지 창의 hazard는 fold 직전에도 약 $10^4$ s⁻¹에 그친다(마지막 노드: −2 V 암조건 $1.0\\times10^4$, photo 프리셋(−1.8 V 암조건, $\\gamma$, $\\delta\\varphi_{G0}$) $9.1\\times10^3$, −1.8 V·2.63 pA(논문 레버) $2.0\\times10^4$ s⁻¹). 따라서 대부분의 cycle이 fold atom에 놓인다: $P_{\\mathrm{atom}}$ = 0.90, 0.91, 0.53. fold 50 mV 이내에서 $\\tau_{\\mathrm{rel}}\\dot v|d\\ln h/dV|$ ≈ 1.6–2.5(두 암조건) / 0.3–0.9(2.63 pA)이고 $\\tau_{\\mathrm{rel}}h$는 마지막 노드에서 0.27 / 0.27 / 0.54(τ_rel 28–30 µs)이므로 준정적 조건이 성립하지 않는다. fold atom(fold 이후 지연 없음)은 실제의 느린 fold 통과를 대신한 값이다: 회로 시뮬레이터는 이를 해상하며, −1.8 V·2.63 pA, 1200 V/s에서 fold를 넘는 비율은 0.55 ± 0.02(FPT atom 0.53)로 같고 평균 $V_{LU}$는 FPT보다 약 +23 mV 높다(`circuit-element`, docs/CIRCUIT_SIMULATOR.md V2).",
            "$\\tau_{\\mathrm{rel}}$ is computed with the lattice definition of `lu_fpt.generator_for` ($x=Q/q$, $F=G-L$ [1/s]; $-(dx/du)/(dF/du)$ at the well). V_G = −2 V dark: 12 µs at $V_D$ = 3.64 V, 28 µs at the last node 4 mV below the fold (3.6997 V; critical slowing near the fold). At 0.4 V/s, $\\tau_{\\mathrm{rel}}\\dot v|d\\ln h/dV| < 10^{-3}$ and $\\tau_{\\mathrm{rel}}h \\lesssim 3\\times10^{-3}$ up to the 99 % quantile, so the approximation holds well (fold atom $2\\times10^{-141}$).\n\n1200 V/s: the hazard in the retained window stays around $10^4$ s⁻¹ even next to the fold (last node: −2 V dark $1.0\\times10^4$; photo preset (−1.8 V dark, $\\gamma$, $\\delta\\varphi_{G0}$) $9.1\\times10^3$; −1.8 V, 2.63 pA (paper levers) $2.0\\times10^4$ s⁻¹). Most cycles therefore land in the fold atom: $P_{\\mathrm{atom}}$ = 0.90, 0.91, 0.53. Within 50 mV of the fold $\\tau_{\\mathrm{rel}}\\dot v|d\\ln h/dV|$ ≈ 1.6–2.5 (both dark cases) / 0.3–0.9 (2.63 pA), and $\\tau_{\\mathrm{rel}}h$ reaches 0.27 / 0.27 / 0.54 at the last node ($\\tau_{\\mathrm{rel}}$ 28–30 µs), so the quasi-static condition fails. The fold atom (no post-fold delay) stands in for the actual slow passage through the fold: the circuit simulator resolves it and, at −1.8 V, 2.63 pA, 1200 V/s, finds the same fraction beyond the fold (0.55 ± 0.02 vs the FPT atom 0.53) but a mean $V_{LU}$ about +23 mV above the FPT (`circuit-element`, docs/CIRCUIT_SIMULATOR.md V2).",
          ),
          code: "lu_fpt.py · generator_for() relaxation; main() checks",
        },
      ],
      notes: [
        L(
          "역-MFPT 램프 hazard는 준안정 근사이다(`compound_fpt.py` docstring): well 안에서 준정상 분포에 도달한다고 가정한다.",
          "The inverse-MFPT ramp hazard is a metastable approximation (`compound_fpt.py` docstring): quasi-stationarity inside the well is assumed.",
        ),
        L(
          "상승·하강 탈출은 독립으로 계산되며(unpaired), 스윕 사이 body 기억은 없다(→ open-problems).",
          "Up and down escapes are computed independently (unpaired); there is no body memory between sweeps (→ open-problems).",
        ),
        L(
          "불일치: `stl_api.py` 스모크 출력은 기대 SD를 '6.8 mV'로 적지만 VALIDATION.md는 ≈ 8 mV이며, 실제 실행값은 8.03 mV이다.",
          "Inconsistency: the `stl_api.py` smoke print states an expected SD of '6.8 mV', VALIDATION.md says ≈ 8 mV, and the code gives 8.03 mV.",
        ),
      ],
    },
  ],
  related: ["stochastic-events", "charge-balance", "sweep-mc", "local-states", "numerics"],
  codeRefs: [CODE.compound, CODE.photoFpt, CODE.gateFpt, CODE.conditional, CODE.luFpt, CODE.ldFpt, CODE.stochCore],
};

export default topic;
