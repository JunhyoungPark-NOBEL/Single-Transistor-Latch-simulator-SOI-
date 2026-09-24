// Physics topic "numerics" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "numerics",
  title: { ko: "수치 해법과 가드", en: "Numerics and guards" },
  summary: {
    ko: "결과의 해상도와 유효 범위를 정하는 수치적 선택을 정리한다: $u$ 격자, $u$마다 $r$을 찾는 로그 이분법, fold의 포물선 보정, 전계 표, 수송 슈팅법, NaN 가드, FPT 전압 창과 hazard 적분.",
    en: "The numerical choices that set the resolution and validity of the results: the $u$ grid, log bisection for $r$ at each $u$, parabolic fold refinement, field tables, transport shooting, NaN guards, and the FPT voltage window and hazard integral.",
  },
  tags: ["grid", "bisection", "FPT window"],
  sections: [
    {
      heading: { ko: "u 격자 (state_grid)", en: "u grid (state_grid)" },
      body: {
        ko: String.raw`수렴(계산값, $V_G = -2$ V): $N$ = 201 / 601 / 2001에서 $V_{\mathrm{LU}}$ = 3.703742 / 3.703689 / 3.703688 V, $V_{\mathrm{LD}}$ = 2.597867 / 2.597867 / 2.597866 V이고, 계산 시간은 0.2–0.8 s이다.`,
        en: String.raw`Convergence (computed, $V_G = -2$ V): with $N$ = 201 / 601 / 2001, $V_{\mathrm{LU}}$ = 3.703742 / 3.703689 / 3.703688 V and $V_{\mathrm{LD}}$ = 2.597867 / 2.597867 / 2.597866 V; run time 0.2–0.8 s.`,
      },
      equations: [
        {
          id: "eq-num-grid",
          label: { ko: "상태 격자", en: "State grid" },
          tex: String.raw`\mathcal{U}_N = \mathrm{unique}\Big(\{0\} \cup \mathrm{geom}\big(10^{-80},\,0.02,\,121\big) \cup \mathrm{lin}\big(0.02,\,1.12,\,N\big)\Big)\ \ [\mathrm{V}]`,
          note: {
            ko: "$N$은 UI의 `numerics.grid`(201–2001, 기본 601)이며, 격자는 $N + 121$점이 된다(601이면 722점). 선형 구간의 간격은 $\\Delta u = 1.1/(N-1)$ = 1.833 mV이다. 기하 구간은 평형에 가까운 깊은 branch를 주입 전위로 분해하기 위한 것이다(코드 주석).",
            en: "$N$ is the UI setting `numerics.grid` (201–2001, default 601), giving $N + 121$ points (722 for 601); the linear spacing is $\\Delta u = 1.1/(N-1)$ = 1.833 mV. The geometric part resolves the deep near-equilibrium branch in injection potential (code comment).",
          },
          code: "photo_mean.py · state_grid()",
        },
        {
          id: "eq-num-grid-vbi",
          label: { ko: "V_bi 근처 보강 (branch)", en: "Refinement near V_bi (branch)" },
          tex: String.raw`\mathcal{U} \leftarrow \mathcal{U} \cup \{V_{\mathrm{bi}}\} \cup \{V_{\mathrm{bi}} \pm g\},\quad g \in \mathrm{geom}\big(10^{-7},\,0.02,\,17\big),\quad 0 \le u \le 1.12`,
          note: {
            ko: "격자가 3점보다 많을 때만 적용한다(+35점). 호출하는 곳별 $N$: `stl_api.folds`, FPT hazard 곡선(`photo_fpt.hazard_curve`, `gate_fpt.one`; 서버 `stoch_core`의 LU/LD 곡선은 UI의 `grid`를 쓰며 기본 601), 보정 스크립트는 601; 기준 보정 조회표를 만드는 `conditional_table.calculate`는 401; `ld_fpt.py`는 801; `stl_api.branches`는 1201; `double_curve`는 131.",
            en: "Applied only when the grid has more than 3 points (+35 points). $N$ per caller: `stl_api.folds`, the FPT hazard curves (`photo_fpt.hazard_curve`, `gate_fpt.one`; the server's `stoch_core` LU/LD curves use the UI `grid`, default 601) and the calibration scripts: 601; the reference-calibration lookup-table builder `conditional_table.calculate`: 401; `ld_fpt.py`: 801; `stl_api.branches`: 1201; `double_curve`: 131.",
          },
          code: "photo_mean.py · FastModel.branch()",
        },
      ],
    },
    {
      heading: { ko: "u마다 r 풀기와 건너뛰기 규칙", en: "Solving for r at each u and skip rules" },
      body: {
        ko: "`curve_grid()`는 다음 경우에 그 $u$를 건너뛴다.\n\n- 소스 장벽 $V_{\\mathrm{bi}} - \\psi \\le 0$\n- $L - w_s - 1.01\\times10^{-7}$ cm $\\le 0$ 또는 $r_{\\mathrm{hi}} \\le 0$\n- $F(u, 0)$이 NaN이거나 $> 0$ (광조사 시 $u < u_0$, `photo`)\n- $F(u, r_{\\mathrm{hi}})$가 유한하고 $< 0$ (증배 표의 끝까지 균형점이 없음)\n- 최종 $|F| > 10^{-5}\\max(I_D, 10^{-25})$\n\n$u = 0$은 항상 $(0, 0)$으로 넣으며, 19번 열에는 $\\sqrt{2qN_AV_{\\mathrm{bi}}/\\varepsilon}$를 둔다.",
        en: "`curve_grid()` skips a $u$ when:\n\n- the source barrier $V_{\\mathrm{bi}} - \\psi \\le 0$\n- $L - w_s - 1.01\\times10^{-7}$ cm $\\le 0$ or $r_{\\mathrm{hi}} \\le 0$\n- $F(u, 0)$ is NaN or $> 0$ ($u < u_0$ under illumination, `photo`)\n- $F(u, r_{\\mathrm{hi}})$ is finite and $< 0$ (no balance up to the end of the multiplication table)\n- the final $|F| > 10^{-5}\\max(I_D, 10^{-25})$\n\n$u = 0$ is always inserted as $(0, 0)$, with $\\sqrt{2qN_AV_{\\mathrm{bi}}/\\varepsilon}$ in column 19.",
      },
      equations: [
        {
          id: "eq-num-bisect",
          label: { ko: "로그 이분법의 해상도", en: "Log-bisection resolution" },
          tex: String.raw`\ln r_{\mathrm{mid}} = \tfrac12\big(\ln r_{\mathrm{lo}} + \ln r_{\mathrm{hi}}\big),\quad r_{\mathrm{lo}}^{(0)} = 10^{-100},\quad 29\ \text{steps} \ \Rightarrow\ \frac{\delta r}{r} \approx \frac{\ln(r_{\mathrm{hi}}/10^{-100})}{2^{29}} \approx 4\times10^{-7}`,
          code: "photo_mean.py · curve_grid()",
        },
      ],
    },
    {
      heading: { ko: "fold 분류", en: "Fold classification" },
      body: {
        ko: "`classify()`는 $V_D$ 차분의 부호 변화로 극대와 극소를 찾고, 첫 극대 $i$와 그 뒤의 마지막 극소 $j$를 fold로 쓴다. 각 fold는 $(k-1, k, k+1)$ 세 점을 지나는 $V_D(u - u_k)$의 2차 다항식(`np.polyfit`) 꼭짓점 값으로 보정한다(식은 `charge-balance`). 점이 8개 미만이거나 극값이 없으면 `None`을 돌려준다(웹: `latch = false`). 웹 서버는 또 fold의 세 행이 $u$에서 연속하지 않으면(간격 > 0.05 V; 실제 fold는 격자 한 칸에 걸친다) 그 fold를 추적할 수 없는 것으로 보고 버리며, `latch = false`와 경고를 낸다. $V_G \\gtrsim 0$ V에서는 $10^{-40} < u < 0.84$ V에 정상상태가 없어서, `classify()`가 간격 너머의 첫 행을 극대로 잡고 포물선을 간격에 걸쳐 맞추기 때문이다(`deterministic.classify_checked`, 확률 모델의 fold 노드 포함). `sample_root()`는 고정된 $V_D$에서 $u$를 13회 이분해 한 상태를 찾는 보조 함수다.",
        en: "`classify()` finds maxima and minima from sign changes of the $V_D$ differences and takes the first maximum $i$ and the last minimum $j$ after it as the folds. Each fold is refined by the vertex of a quadratic `np.polyfit` of $V_D(u - u_k)$ through $(k-1, k, k+1)$ (formula in `charge-balance`). Fewer than 8 points or missing extrema give `None` (web: `latch = false`). The web server also rejects a fold whose three rows are not contiguous in $u$ (gap > 0.05 V; a real fold spans one grid step) as not traceable, returning `latch = false` with a warning: for $V_G \\gtrsim 0$ V there is no steady state for $10^{-40} < u < 0.84$ V, so `classify()` takes the first row after the gap as the maximum and fits its parabola across the gap (`deterministic.classify_checked`, stochastic fold nodes included). `sample_root()` is a helper that bisects $u$ 13 times to find the state at a fixed $V_D$.",
      },
    },
    {
      heading: { ko: "전계 표와 수송 해법", en: "Field tables and transport solver" },
      body: {
        ko: "- 전계 표: $r$ = `linspace(0, 5, 1501)`($\\Delta r$ = 3.333 mV), $z$ 방향 501점 사다리꼴 적분, 유효 조건 $1/M > 10^{-3}$, $E_{\\mathrm{pk}} \\le 1.2\\times10^{6}$ V/cm. PCHIP을 격자에서 샘플링한 뒤 선형 보간한다(외삽 허용).\n- 수송: RK4 64단계, Newton 슈팅 최대 32회, 상대 허용 오차 $10^{-9}$, 초기값 $\\max(2h_E - \\ln(1+h_E), 10^{-100})$, 갱신 $j_0 \\leftarrow \\max(0.1j_0,\\ j_0 - \\mathrm{res}/\\mathrm{deriv})$. 저주입 해석해 가드(`solve_voltage`, 식은 `bjt-transport`): $h_E < 10^{-8}$이고 $(1+\\tau_r)h_E < 10^{-8}$($\\tau_r = \\tau_p/\\tau_n$)이면 먼저 선형 해 $j_0 = h_E\\,k/\\sinh k$($k = \\sqrt{\\kappa/(1+\\theta_t)}$, $k < 10^{-6}$이면 급수)를 계산하고, 그 $j_0$가 $Mj_0 + b < 10^{-8}$도 만족할 때에만 해석해를 돌려준다. 둘 중 하나라도 어긋나면 전체 슈팅을 실행한다.\n- 국소 경로의 벌크 정의: 수송 재계산을 포함한 고정점 반복 3회.",
        en: "- Field tables: $r$ = `linspace(0, 5, 1501)` ($\\Delta r$ = 3.333 mV), 501-point trapezoid in $z$, validity $1/M > 10^{-3}$ and $E_{\\mathrm{pk}} \\le 1.2\\times10^{6}$ V/cm; the PCHIP is sampled on the grid and then interpolated linearly (extrapolation allowed).\n- Transport: RK4 with 64 steps, at most 32 Newton shooting iterations, relative tolerance $10^{-9}$, initial guess $\\max(2h_E - \\ln(1+h_E), 10^{-100})$, update $j_0 \\leftarrow \\max(0.1j_0,\\ j_0 - \\mathrm{res}/\\mathrm{deriv})$. Low-injection analytic guard (`solve_voltage`, formulas in `bjt-transport`): if $h_E < 10^{-8}$ and $(1+\\tau_r)h_E < 10^{-8}$ ($\\tau_r = \\tau_p/\\tau_n$), the linear solution $j_0 = h_E\\,k/\\sinh k$ ($k = \\sqrt{\\kappa/(1+\\theta_t)}$, series for $k < 10^{-6}$) is computed first, and the analytic result is returned only if that $j_0$ also gives $Mj_0 + b < 10^{-8}$; if either test fails, the full shooting runs.\n- Local path, bulk definition: 3 fixed-point iterations including transport re-solves.",
      },
    },
    {
      heading: { ko: "NaN 가드", en: "NaN guards" },
      body: {
        ko: "`components()`는 다음 경우에 NaN 19개를 돌려준다(상위 코드는 그 점을 버린다).\n\n- 소스 장벽 $\\le 0$\n- $L_n \\le 10^{-7}$ cm\n- 수송 해법 실패: $h < 0$, $h > 10^{80}$, 유한하지 않은 값, $h_s(1) \\le 0$, 32회 안에 수렴하지 않음\n\n`Field`는 유효한 $r$ 점이 5개 미만이면 `ValueError`를 내고, `state()`의 `brentq`는 $r \\in [0, V_D - u]$에서 부호 변화가 없으면 `ValueError`를 낸다(FPT에서는 그 노드를 건너뛴다).",
        en: "`components()` returns 19 NaNs when:\n\n- the source barrier $\\le 0$\n- $L_n \\le 10^{-7}$ cm\n- the transport solve fails: $h < 0$, $h > 10^{80}$, non-finite values, $h_s(1) \\le 0$, no convergence within 32 iterations\n\n`Field` raises `ValueError` with fewer than 5 valid $r$ points; `brentq` in `state()` raises `ValueError` without a sign change on $r \\in [0, V_D - u]$ (the FPT then skips that node).",
      },
    },
    {
      heading: { ko: "FPT 전압 창과 hazard 적분", en: "FPT voltage window and hazard integral" },
      equations: [
        {
          id: "eq-num-fpt-window",
          label: { ko: "hazard 노드 전압 (LU)", en: "Hazard node voltages (LU)" },
          tex: String.raw`V_k \in \big[\max(V_{\mathrm{LU}} - 0.28,\ V_{\mathrm{LD}} + 0.001),\ V_{\mathrm{LU}} - 0.001\big),\qquad \Delta V = 4\ \mathrm{mV}`,
          note: {
            ko: "각 $V_k$에서 $u$ 격자 lin(0.1, 0.9, 181) ∪ lin($u_f$ − 0.065, $u_f$ + 0.065, 61)로 `state()` 행을 만들고, $Q(u = 0.1)$부터 $I_D = 10^{-8}$ A인 흡수 경계까지 격자를 잡아 평균 첫 통과 시간 $T$를 풀면 $h(V_k) = 1/T(x_w)$가 된다. `ValueError`/`IndexError`/`AssertionError`가 나는 노드는 건너뛴다. 기준 보정의 조회표 경로(`conditional_table.calculate`)도 같은 0.28 V 창과 4 mV 간격(정밀 모드 2 mV)을 쓰며, LD는 fold + 0.30 V에서 시작한다.",
            en: "At each $V_k$, `state()` rows on the $u$ grid lin(0.1, 0.9, 181) ∪ lin($u_f$ − 0.065, $u_f$ + 0.065, 61) build the lattice from $Q(u = 0.1)$ to the absorbing boundary at $I_D = 10^{-8}$ A; the mean first-passage time $T$ then gives $h(V_k) = 1/T(x_w)$. Nodes raising `ValueError`/`IndexError`/`AssertionError` are skipped. The reference-calibration lookup path (`conditional_table.calculate`) uses the same 0.28 V window with 4 mV steps (fine mode: 2 mV), and for LD starts at fold + 0.30 V.",
          },
          code: "photo_fpt.py · hazard_curve()",
        },
        {
          id: "eq-num-quant",
          label: { ko: "램프에서의 생존 확률과 분위수", en: "Survival and quantiles on a ramp" },
          tex: String.raw`\begin{aligned} &h(V) \leftarrow 0 \ \text{ up to the last node with } h < 10^{-4}\ \mathrm{s^{-1}} \text{ or non-finite}\\ &\Lambda(V) = \int_{V_0}^{V} \frac{h(V')}{\dot V}\,dV',\qquad S(V) = e^{-\Lambda(V)}\\ &V_{P_k} = \Lambda^{-1}\big(-\ln(1 - P_k)\big),\qquad P_k = \frac{k + 1/2}{10001},\ \ k = 0, \dots, 10000 \end{aligned}`,
          note: {
            ko: "누적 적분은 사다리꼴(`cumulative_trapezoid`), 역함수는 선형 보간(`np.interp`)으로 구한다. 표의 범위를 넘는 확률 질량은 fold $V_{\\mathrm{LU}}$에 둔다(이 모델에는 fold 이후의 지연이 없다). $\\dot V$는 램프 속도(V/s)다.",
            en: "Cumulative trapezoid (`cumulative_trapezoid`), inverse by linear interpolation (`np.interp`); the probability mass beyond the table is placed at the fold $V_{\\mathrm{LU}}$ (the model has no post-fold delay). $\\dot V$ is the ramp rate (V/s).",
          },
          code: "photo_fpt.py · quantiles(), PROB",
        },
      ],
    },
    {
      heading: { ko: "UI의 수치 설정", en: "UI numerics controls" },
      body: {
        ko: "- `grid` (201–2001, 기본 601): 모든 결정론 계산에서 `state_grid`의 $N$. 크게 하면 branch가 매끄러워지고 fold 보정이 안정되지만, 계산 시간은 거의 선형으로 늘어난다(fold 변화는 0.1 mV 미만).\n- `fold_nodes` (≤ 61, 기본 25): 확률 스윕의 fold 표를 만들 때 fold를 다시 계산하는 국소 상태 값의 개수(`sweep-mc`).\n- `hazard_nodes` (≤ 9, 기본 5): 위의 FPT hazard 곡선을 계산해 보간하는 상태 값의 개수(`first-passage`).",
        en: "- `grid` (201–2001, default 601): the `state_grid` $N$ of every deterministic computation. Larger values give smoother branches and a more stable fold refinement at roughly linear cost (the folds change by less than 0.1 mV).\n- `fold_nodes` (≤ 61, default 25): the number of local-state values at which the folds are recomputed for the stochastic fold table (`sweep-mc`).\n- `hazard_nodes` (≤ 9, default 5): the number of state values at which the FPT hazard curve above is computed and then interpolated (`first-passage`).",
      },
    },
  ],
  related: ["charge-balance", "impact-ionization", "bjt-transport", "first-passage", "sweep-mc", "validation"],
  codeRefs: [
    "photo_extension/photo_mean.py",
    "photo_extension/photo_fpt.py",
    "model/janus_calibration_20260920/hypothesis_study_20260920/compound_fpt.py",
    "model/janus_calibration_20260920/idvd_model_v3/srh_transport.py",
  ],
};

export default topic;
