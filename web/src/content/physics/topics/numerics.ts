// Physics topic "numerics" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "numerics",
  title: { ko: "수치 해법과 가드", en: "Numerics and guards" },
  summary: {
    ko: "$u$ 격자, $u$마다 $r$의 로그 이분법, fold 포물선 보정, 전계 표, 수송 shooting, NaN 가드, FPT 전압창과 hazard 적분 — 결과의 해상도와 유효 범위를 정하는 수치 선택들.",
    en: "The $u$ grid, log bisection for $r$ at each $u$, parabolic fold refinement, field tables, transport shooting, NaN guards, and the FPT voltage window and hazard integral — the numerical choices that set resolution and validity.",
  },
  tags: ["grid", "bisection", "FPT window"],
  sections: [
    {
      heading: { ko: "u 격자 (state_grid)", en: "u grid (state_grid)" },
      body: {
        ko: String.raw`수렴(계산값, $V_G = -2$ V): $N$ = 201 / 601 / 2001에서 $V_{\mathrm{LU}}$ = 3.703742 / 3.703689 / 3.703688 V, $V_{\mathrm{LD}}$ = 2.597867 / 2.597867 / 2.597866 V; 계산 시간 0.2–0.8 s.`,
        en: String.raw`Convergence (computed, $V_G = -2$ V): with $N$ = 201 / 601 / 2001, $V_{\mathrm{LU}}$ = 3.703742 / 3.703689 / 3.703688 V and $V_{\mathrm{LD}}$ = 2.597867 / 2.597867 / 2.597866 V; run time 0.2–0.8 s.`,
      },
      equations: [
        {
          id: "eq-num-grid",
          label: { ko: "상태 격자", en: "State grid" },
          tex: String.raw`\mathcal{U}_N = \mathrm{unique}\Big(\{0\} \cup \mathrm{geom}\big(10^{-80},\,0.02,\,121\big) \cup \mathrm{lin}\big(0.02,\,1.12,\,N\big)\Big)\ \ [\mathrm{V}]`,
          note: {
            ko: "$N$ = UI `numerics.grid`(201–2001, 기본 601) → $N + 121$점(601이면 722점), 선형 구간 $\\Delta u = 1.1/(N-1)$ = 1.833 mV. 기하 구간은 깊은 평형 가지를 주입 전위로 분해하기 위한 것(코드 주석).",
            en: "$N$ = UI `numerics.grid` (201–2001, default 601) → $N + 121$ points (722 for 601), linear spacing $\\Delta u = 1.1/(N-1)$ = 1.833 mV. The geometric part resolves the deep-equilibrium branch in injection potential (code comment).",
          },
          code: "photo_mean.py · state_grid()",
        },
        {
          id: "eq-num-grid-vbi",
          label: { ko: "V_bi 근처 보강 (branch)", en: "Refinement near V_bi (branch)" },
          tex: String.raw`\mathcal{U} \leftarrow \mathcal{U} \cup \{V_{\mathrm{bi}}\} \cup \{V_{\mathrm{bi}} \pm g\},\quad g \in \mathrm{geom}\big(10^{-7},\,0.02,\,17\big),\quad 0 \le u \le 1.12`,
          note: {
            ko: "격자가 3점보다 많을 때만 적용(+35점). 호출별 $N$: `stl_api.folds`, FPT, 보정 601; `stl_api.branches` 1201; `double_curve` 131.",
            en: "Applied only when the grid has more than 3 points (+35 points). $N$ per caller: `stl_api.folds`, FPT and calibrations 601; `stl_api.branches` 1201; `double_curve` 131.",
          },
          code: "photo_mean.py · FastModel.branch()",
        },
      ],
    },
    {
      heading: { ko: "u마다 r 풀기와 건너뛰기 규칙", en: "Solving r per u and skip rules" },
      body: {
        ko: "`curve_grid()`가 $u$를 건너뛰는 경우:\n\n- source 장벽 $V_{\\mathrm{bi}} - \\psi \\le 0$\n- $L - w_s - 1.01\\times10^{-7}$ cm $\\le 0$ 또는 $r_{\\mathrm{hi}} \\le 0$\n- $F(u, 0)$이 NaN이거나 $> 0$ (조사 시 $u < u_0$, `photo`)\n- $F(u, r_{\\mathrm{hi}})$가 유한하고 $< 0$ (증배 표 끝까지 래치 조건 불충족)\n- 최종 $|F| > 10^{-5}\\max(I_D, 10^{-25})$\n\n$u = 0$은 $(0, 0)$으로 항상 넣고 19번 열에 $\\sqrt{2qN_AV_{\\mathrm{bi}}/\\varepsilon}$를 둔다.",
        en: "`curve_grid()` skips a $u$ when:\n\n- source barrier $V_{\\mathrm{bi}} - \\psi \\le 0$\n- $L - w_s - 1.01\\times10^{-7}$ cm $\\le 0$ or $r_{\\mathrm{hi}} \\le 0$\n- $F(u, 0)$ is NaN or $> 0$ (illuminated $u < u_0$, `photo`)\n- $F(u, r_{\\mathrm{hi}})$ is finite and $< 0$ (no balance up to the end of the multiplication table)\n- final $|F| > 10^{-5}\\max(I_D, 10^{-25})$\n\n$u = 0$ is always inserted as $(0, 0)$ with $\\sqrt{2qN_AV_{\\mathrm{bi}}/\\varepsilon}$ in column 19.",
      },
      equations: [
        {
          id: "eq-num-bisect",
          label: { ko: "로그 이분법 해상도", en: "Log-bisection resolution" },
          tex: String.raw`\ln r_{\mathrm{mid}} = \tfrac12\big(\ln r_{\mathrm{lo}} + \ln r_{\mathrm{hi}}\big),\quad r_{\mathrm{lo}}^{(0)} = 10^{-100},\quad 29\ \text{steps} \ \Rightarrow\ \frac{\delta r}{r} \approx \frac{\ln(r_{\mathrm{hi}}/10^{-100})}{2^{29}} \approx 4\times10^{-7}`,
          code: "photo_mean.py · curve_grid()",
        },
      ],
    },
    {
      heading: { ko: "Fold 분류", en: "Fold classification" },
      body: {
        ko: "`classify()`: $V_D$ 차분의 부호 변화로 극대·극소를 찾고, 첫 극대 $i$와 그 뒤 마지막 극소 $j$를 fold로 쓴다. 각 fold는 $(k-1, k, k+1)$ 세 점에서 $V_D(u - u_k)$ 2차 다항식(`np.polyfit`)의 꼭짓점 값으로 보정한다(식은 `charge-balance`). 8점 미만·극값 없음이면 `None`(웹: `latch = false`). `sample_root()`는 고정 $V_D$에서 $u$를 13회 이분해 한 상태를 찾는 보조 함수다.",
        en: "`classify()` finds maxima/minima from sign changes of the $V_D$ differences and takes the first maximum $i$ and the last minimum $j$ after it. Each fold is refined by the vertex of a quadratic `np.polyfit` of $V_D(u - u_k)$ through $(k-1, k, k+1)$ (formula in `charge-balance`). Fewer than 8 points or missing extrema give `None` (web: `latch = false`). `sample_root()` is a helper that bisects $u$ 13 times to find the state at a fixed $V_D$.",
      },
    },
    {
      heading: { ko: "전계 표와 수송 해법", en: "Field tables and transport solver" },
      body: {
        ko: "- 전계 표: $r$ = `linspace(0, 5, 1501)`($\\Delta r$ = 3.333 mV), $z$ 501점 사다리꼴, 유효 조건 $1/M > 10^{-3}$, $E_{\\mathrm{pk}} \\le 1.2\\times10^{6}$ V/cm; PCHIP을 격자에서 샘플한 뒤 선형 보간(외삽 허용).\n- 수송: RK4 64단계, Newton shooting 최대 32회, 상대 허용오차 $10^{-9}$, 초기값 $2h_E - \\ln(1+h_E)$, 저주입 해석 분기 $h_E < 10^{-8}$ 및 $Mj_0 + b < 10^{-8}$.\n- 국소 경로 bulk 정의: 수송 재해법을 포함한 고정점 반복 3회.",
        en: "- Field tables: $r$ = `linspace(0, 5, 1501)` ($\\Delta r$ = 3.333 mV), 501-point trapezoid in $z$, validity $1/M > 10^{-3}$ and $E_{\\mathrm{pk}} \\le 1.2\\times10^{6}$ V/cm; PCHIP sampled on the grid, then linear interpolation (extrapolation allowed).\n- Transport: RK4 with 64 steps, at most 32 Newton shooting iterations, relative tolerance $10^{-9}$, initial guess $2h_E - \\ln(1+h_E)$, low-injection analytic branch for $h_E < 10^{-8}$ and $Mj_0 + b < 10^{-8}$.\n- Local path, bulk definition: 3 fixed-point iterations including transport re-solves.",
      },
    },
    {
      heading: { ko: "NaN 가드", en: "NaN guards" },
      body: {
        ko: "`components()`는 다음에서 NaN 19개를 돌려준다(상위 코드는 그 점을 버린다):\n\n- source 장벽 $\\le 0$\n- $L_n \\le 10^{-7}$ cm\n- 수송 해법 실패: $h < 0$, $h > 10^{80}$, 비유한, $h_s(1) \\le 0$, 32회 미수렴\n\n`Field`는 유효 $r$ 점이 5개 미만이면 `ValueError`, `state()`의 `brentq`는 $r \\in [0, V_D - u]$에서 부호 변화가 없으면 `ValueError`를 낸다(FPT에서 그 노드는 건너뜀).",
        en: "`components()` returns 19 NaNs when:\n\n- source barrier $\\le 0$\n- $L_n \\le 10^{-7}$ cm\n- the transport solve fails: $h < 0$, $h > 10^{80}$, non-finite, $h_s(1) \\le 0$, no convergence in 32 iterations\n\n`Field` raises `ValueError` with fewer than 5 valid $r$ points; `brentq` in `state()` raises `ValueError` without a sign change on $r \\in [0, V_D - u]$ (the FPT node is then skipped).",
      },
    },
    {
      heading: { ko: "FPT 전압창과 hazard 적분", en: "FPT voltage window and hazard integral" },
      equations: [
        {
          id: "eq-num-fpt-window",
          label: { ko: "hazard 노드 전압 (LU)", en: "Hazard node voltages (LU)" },
          tex: String.raw`V_k \in \big[\max(V_{\mathrm{LU}} - 0.28,\ V_{\mathrm{LD}} + 0.001),\ V_{\mathrm{LU}} - 0.001\big),\qquad \Delta V = 4\ \mathrm{mV}`,
          note: {
            ko: "각 $V_k$에서 $u$ 격자 lin(0.1, 0.9, 181) ∪ lin($u_f$ − 0.065, $u_f$ + 0.065, 61)로 `state()` 행을 만들고, 격자를 $Q(u = 0.1)$부터 $I_D = 10^{-8}$ A인 흡수 경계까지 잡아 평균 첫 통과 시간 $T$를 풀고 $h(V_k) = 1/T(x_w)$. `ValueError`/`IndexError`/`AssertionError` 노드는 건너뛴다. 논문 조회표 경로(`conditional_table.calculate`)는 같은 0.28 V 창, 4 mV(정밀 2 mV), LD는 fold + 0.30 V부터.",
            en: "At each $V_k$, `state()` rows on the $u$ grid lin(0.1, 0.9, 181) ∪ lin($u_f$ − 0.065, $u_f$ + 0.065, 61) build the lattice from $Q(u = 0.1)$ to the absorbing boundary at $I_D = 10^{-8}$ A; the mean first-passage time $T$ gives $h(V_k) = 1/T(x_w)$. Nodes raising `ValueError`/`IndexError`/`AssertionError` are skipped. The paper lookup path (`conditional_table.calculate`) uses the same 0.28 V window, 4 mV (fine: 2 mV), and for LD starts at fold + 0.30 V.",
          },
          code: "photo_fpt.py · hazard_curve()",
        },
        {
          id: "eq-num-quant",
          label: { ko: "램프에서의 생존과 분위수", en: "Survival and quantiles on a ramp" },
          tex: String.raw`\begin{aligned} &h(V) \leftarrow 0 \ \text{ up to the last node with } h < 10^{-4}\ \mathrm{s^{-1}} \text{ or non-finite}\\ &\Lambda(V) = \int_{V_0}^{V} \frac{h(V')}{\dot V}\,dV',\qquad S(V) = e^{-\Lambda(V)}\\ &V_{P_k} = \Lambda^{-1}\big(-\ln(1 - P_k)\big),\qquad P_k = \frac{k + 1/2}{10001},\ \ k = 0, \dots, 10000 \end{aligned}`,
          note: {
            ko: "누적은 사다리꼴(`cumulative_trapezoid`), 역함수는 선형 보간(`np.interp`); 표 범위를 넘는 확률 질량은 fold $V_{\\mathrm{LU}}$에 둔다(이 모델에는 fold 이후 지연이 없음). $\\dot V$ = 램프 속도(V/s).",
            en: "Cumulative trapezoid (`cumulative_trapezoid`), inverse by linear interpolation (`np.interp`); probability mass beyond the table is placed at the fold $V_{\\mathrm{LU}}$ (no post-fold delay in this model). $\\dot V$ = ramp rate (V/s).",
          },
          code: "photo_fpt.py · quantiles(), PROB",
        },
      ],
    },
    {
      heading: { ko: "UI numerics 컨트롤", en: "UI numerics controls" },
      body: {
        ko: "- `grid` (201–2001, 기본 601): 모든 결정론 계산의 `state_grid` $N$. 크게 하면 가지가 매끄럽고 fold 보정이 안정되지만 시간은 거의 선형으로 는다(fold 변화는 0.1 mV 미만).\n- `fold_nodes` (≤ 61, 기본 25): 확률 스윕에서 local-state 값 몇 개에서 fold를 다시 계산해 fold 표를 만들지(`sweep-mc`).\n- `hazard_nodes` (≤ 9, 기본 5): 위 FPT hazard 곡선을 몇 개의 상태 값에서 계산해 보간할지(`first-passage`).",
        en: "- `grid` (201–2001, default 601): the `state_grid` $N$ of every deterministic computation. Larger values give smoother branches and a more stable fold refinement at roughly linear cost (fold changes are below 0.1 mV).\n- `fold_nodes` (≤ 61, default 25): number of local-state values at which the folds are recomputed for the stochastic fold table (`sweep-mc`).\n- `hazard_nodes` (≤ 9, default 5): number of state values at which the FPT hazard curve above is computed and interpolated (`first-passage`).",
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
