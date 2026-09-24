// Physics topic "charge-balance" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "charge-balance",
  title: { ko: "전하 균형 (Eq. 1), 정상상태, fold", en: "Charge balance (Eq. 1), steady state, folds" },
  summary: {
    ko: String.raw`body 정공 전하 보존식 $dQ_B/dt = F(u, r)$의 정상해 $F = 0$과 Kirchhoff 관계가 $V_D(u)$, $I_D(u)$ 궤적을 주고, $V_D(u)$의 첫 극대와 마지막 극소가 $V_{\mathrm{LU}}$, $V_{\mathrm{LD}}$다. 고정 $V_D$에서는 $F$의 부호가 body 전하 지형(안정·불안정 근)을 준다.`,
    en: String.raw`The steady state $F = 0$ of the body hole-charge balance $dQ_B/dt = F(u, r)$ plus Kirchhoff's law gives the locus $V_D(u)$, $I_D(u)$; its first maximum and last minimum are $V_{\mathrm{LU}}$ and $V_{\mathrm{LD}}$. At fixed $V_D$ the sign of $F$ gives the body-charge landscape (stable and unstable roots).`,
  },
  tags: ["Eq. 1", "deterministic", "folds"],
  sections: [
    {
      heading: { ko: "Eq. 1과 코드 대응", en: "Eq. 1 and its code mapping" },
      equations: [
        {
          id: "eq-cb-eq1",
          label: { ko: "Eq. 1", en: "Eq. 1" },
          tex: String.raw`\frac{dQ_B}{dt} = I_{\mathrm{II}} + I_{\mathrm{BTBT}} + I_{\mathrm{GIDL}} + I_{\mathrm{PH}} - I_{\mathrm{REC}} - I_{\mathrm{DIFF}} \equiv F(u, r;\, V_G, I_{\mathrm{PH}})`,
          code: "docs/MODEL_SPEC.md §2",
        },
        {
          id: "eq-cb-F",
          label: { ko: "코드의 순 정공 전류 F (z[2])", en: "Net hole current F in the code (z[2])" },
          tex: "F = \\underbrace{(M-1)I_{\\mathrm{seed}} + I_{\\mathrm{II,ch}} + I_{\\mathrm{II,PH}} + I_{\\mathrm{loc}}}_{I_{\\mathrm{II}}} + I_{\\mathrm{BTBT}} + I_{\\mathrm{GIDL}} + I_{\\mathrm{PH}} - \\underbrace{\\big(I_{\\mathrm{bulk}} + I_J\\big)}_{I_{\\mathrm{REC}}} - I_{\\mathrm{DIFF}}",
          note: {
            ko: "코드: `net = ii + ii_ch + bbj + gidl + iph + ii_ph + iloc - diff - bulk - junction`. 단위 A(양수 = body로 정공 유입).",
            en: "Code: `net = ii + ii_ch + bbj + gidl + iph + ii_ph + iloc - diff - bulk - junction`. Units A (positive = holes into the body).",
          },
          code: "photo_mean.py · components() (net)",
        },
      ],
      variables: [
        {
          symbol: String.raw`I_{\mathrm{II}}`,
          name: { ko: "증배 정공 (impact-ionization)", en: "Multiplication holes (impact-ionization)" },
          code: "ii + ii_ch + ii_ph + iloc",
        },
        {
          symbol: String.raw`I_{\mathrm{BTBT}}`,
          name: { ko: "접합 BTBT (btbt-gidl)", en: "Junction BTBT (btbt-gidl)" },
          code: "bbj = z[8]",
        },
        {
          symbol: String.raw`I_{\mathrm{GIDL}}`,
          name: { ko: "GIDL (btbt-gidl)", en: "GIDL (btbt-gidl)" },
          code: "gidl = z[9]",
        },
        {
          symbol: String.raw`I_{\mathrm{PH}}`,
          name: { ko: "광생성 (photo)", en: "Photogeneration (photo)" },
          code: "iph = z[18] = p[13]",
        },
        {
          symbol: String.raw`I_{\mathrm{REC}}`,
          name: { ko: "body SRH + 접합 SRH (bjt-transport)", en: "Body SRH + junction SRH (bjt-transport)" },
          code: "bulk + junction = z[5] + z[7]",
        },
        {
          symbol: String.raw`I_{\mathrm{DIFF}}`,
          name: { ko: "emitter 확산 (bjt-transport)", en: "Emitter diffusion (bjt-transport)" },
          code: "diff = z[6]",
        },
      ],
    },
    {
      heading: { ko: "정상상태와 Kirchhoff", en: "Steady state and Kirchhoff" },
      equations: [
        {
          id: "eq-cb-ss",
          label: { ko: "u로 매개화한 정상 궤적", en: "Steady locus parametrised by u" },
          tex: String.raw`F\big(u, r^{\star}(u)\big) = 0\ \ \Rightarrow\ \ V_D(u) = V_D\big(u, r^{\star}(u)\big),\qquad I_D(u) = I_D\big(u, r^{\star}(u)\big)`,
          note: {
            ko: "$u$가 독립 추적 변수다: 궤적은 $u$에 대해 단일값이지만 $V_D$에 대해서는 다중값(S자)이다. $F$는 $r$에 대해 증가(증배·BTBT·GIDL 증가)하므로 $r$을 구간 이분법으로 찾는다.",
            en: "$u$ is the independent tracing variable: the locus is single-valued in $u$ but multi-valued (S-shaped) in $V_D$. $F$ increases with $r$ (more multiplication, BTBT, GIDL), so $r$ is found by bracketed bisection.",
          },
          code: "photo_mean.py · curve_grid()",
        },
        {
          id: "eq-cb-bisect",
          label: { ko: "u마다 r 풀기", en: "Solving r for each u" },
          tex: String.raw`\begin{aligned} &r \in \big[10^{-100},\ r_{\mathrm{hi}}\big],\qquad r_{\mathrm{hi}} = \min\!\big(5\ \mathrm{V},\ r_{\mathrm{geo}}(u)\big)\\ &\text{require } F(u, 0) \le 0 \text{ (finite)},\quad F(u, r_{\mathrm{hi}}) \ge 0 \text{ or NaN}\\ &29 \text{ bisections in } \ln r:\ \ F(u, r_{\mathrm{mid}}) > 0 \text{ or NaN} \Rightarrow r_{\mathrm{hi}} \leftarrow r_{\mathrm{mid}},\ \text{else } r_{\mathrm{lo}} \leftarrow r_{\mathrm{mid}}\\ &\text{accept if } |F| \le 10^{-5}\max(I_D,\,10^{-25}) \end{aligned}`,
          note: {
            ko: String.raw`branch 배열 열: 0–16 = z[0..16], 17 = $u$, 18 = $r$, 19 = $E_{\mathrm{pk}}$, 20 = 정공 강하(V). $u = 0$은 $(u, r) = (0, 0)$으로 항상 넣는다.`,
            en: String.raw`Branch-array columns: 0–16 = z[0..16], 17 = $u$, 18 = $r$, 19 = $E_{\mathrm{pk}}$, 20 = hole drop (V). $u = 0$ is always inserted as $(u, r) = (0, 0)$.`,
          },
          code: "photo_mean.py · curve_grid()",
        },
      ],
    },
    {
      heading: { ko: "Fold 검출", en: "Fold detection" },
      body: {
        ko: "계산값 (grid 601, `VALIDATION.md`와 일치):\n\n- $V_G = -2$ V 암조건: $V_{\\mathrm{LU}} = 3.7037$ V, $V_{\\mathrm{LD}} = 2.5979$ V (창 1.106 V)\n- $V_G = -1.8$ V 암조건: 3.8644 V, 2.5979 V\n- $V_G = -1.8$ V, $I_{\\mathrm{PH}} = 2.63$ pA: 3.2913 V, 2.5962 V\n\n$V_G = -2$ V fold 상태(앱의 결정론 `folds` 표시와 같은 보정: $V_D(u)$ 3점 포물선의 꼭짓점 $u$, $I_D$는 $u$에 대해 로그 보간, $r$은 선형 보간): LU에서 $u = 0.5673$ V, $r \\approx 3.136$ V, $I_D = 14.89$ pA; LD에서 $u = 0.7601$ V, $r \\approx 1.831$ V, $I_D = 16.33$ nA. 가장 가까운 601 격자 행은 LU $u = 0.5682$ V, $r = 3.1355$ V, $I_D = 15.30$ pA, LD $u = 0.7607$ V, $r = 1.8301$ V, $I_D = 16.66$ nA이다(다른 주제의 fold 상태 값은 이 행 값).",
        en: "Computed values (grid 601, matching `VALIDATION.md`):\n\n- $V_G = -2$ V dark: $V_{\\mathrm{LU}} = 3.7037$ V, $V_{\\mathrm{LD}} = 2.5979$ V (window 1.106 V)\n- $V_G = -1.8$ V dark: 3.8644 V, 2.5979 V\n- $V_G = -1.8$ V, $I_{\\mathrm{PH}} = 2.63$ pA: 3.2913 V, 2.5962 V\n\nFold states at $V_G = -2$ V (refined as in the app's deterministic `folds` readout: vertex $u$ of the 3-point parabola $V_D(u)$, $I_D$ log-interpolated in $u$, $r$ linearly interpolated): at LU $u = 0.5673$ V, $r \\approx 3.136$ V, $I_D = 14.89$ pA; at LD $u = 0.7601$ V, $r \\approx 1.831$ V, $I_D = 16.33$ nA. The nearest 601-grid rows are LU $u = 0.5682$ V, $r = 3.1355$ V, $I_D = 15.30$ pA and LD $u = 0.7607$ V, $r = 1.8301$ V, $I_D = 16.66$ nA (fold-state values quoted in other topics are these rows).",
      },
      equations: [
        {
          id: "eq-cb-fold",
          label: { ko: "극값 탐색과 포물선 보정", en: "Extremum search and parabolic refinement" },
          tex: String.raw`\begin{aligned} &\Delta V_k = V_{D,k+1} - V_{D,k};\qquad \max\!: \Delta V_{k-1} > 0 > \Delta V_k,\qquad \min\!: \Delta V_{k-1} < 0 < \Delta V_k\\ &i = \text{first max},\qquad j = \text{last min with } j > i\\ &V_D \approx c_0 x^2 + c_1 x + c_2,\ \ x = u - u_k\ \ (k-1, k, k+1)\ \Rightarrow\ V_{\mathrm{fold}} = V_D\!\left(-\frac{c_1}{2c_0}\right) \end{aligned}`,
          note: {
            ko: "점이 8개 미만이거나 극대가 없거나 $i$ 뒤 극소가 없으면 `None`(래치 없음). 코드 주석: 최적화 중 중간 fold가 생길 수 있으므로 마지막 안정 가지를 쓴다.",
            en: "Returns `None` (no latch) with fewer than 8 points, no maximum, or no minimum after $i$. Code comment: intermediate folds may appear during optimisation, so the terminal stable branch is used.",
          },
          code: "photo_mean.py · FastModel.classify()",
        },
      ],
    },
    {
      heading: { ko: "가지와 준정적 스윕", en: "Branches and the quasi-static sweep" },
      body: {
        ko: "- HRS = `b[:i+1]`, 불안정 = `b[i:j+1]`, LRS = `b[j:]` (웹 API의 분할). 불안정 가지에서는 $dV_D/du < 0$.\n- 상승 스윕은 HRS를 따라 $V_{\\mathrm{LU}}$까지 간 뒤 LRS로 점프하고, 하강 스윕은 LRS를 따라 $V_{\\mathrm{LD}}$까지 간 뒤 HRS로 떨어진다.\n- `double_curve`는 각 단조 구간에서 $\\ln I_D$를 $V_D$에 대해 보간하고, fold 꼭짓점 상태를 한 점 더 계산해 붙이며, $V_D = 0$에서 $I_D = 0$(평형)으로 둔다.",
        en: "- HRS = `b[:i+1]`, unstable = `b[i:j+1]`, LRS = `b[j:]` (web API split). On the unstable branch $dV_D/du < 0$.\n- The up-sweep follows the HRS to $V_{\\mathrm{LU}}$ and jumps to the LRS; the down-sweep follows the LRS to $V_{\\mathrm{LD}}$ and drops to the HRS.\n- `double_curve` interpolates $\\ln I_D$ versus $V_D$ on each monotonic segment, appends one extra state at each fold vertex, and sets $I_D = 0$ at $V_D = 0$ (equilibrium).",
      },
    },
    {
      heading: { ko: "고정 V_D에서의 body 전하 지형", en: "Body-charge landscape at fixed V_D" },
      body: {
        ko: String.raw`예시 ($V_G = -2$ V, 암조건, $V_D = 3.5$ V, 계산값):

- HRS 우물: $u = 0.4953$ V, $I_D = 1.837$ pA, $G = 7.857\times10^{6}$ s⁻¹, $Q/q = 1432.4$
- 안장(불안정): $u = 0.6097$ V, $I_D = 62.2$ pA, $Q/q = 1626.3$
- LRS: $u = 0.9284$ V, $I_D = 7.62$ µA, $Q/q = 3162.7$

우물과 안장 사이 전하 장벽은 약 194개의 정공이다.`,
        en: String.raw`Example ($V_G = -2$ V, dark, $V_D = 3.5$ V, computed):

- HRS well: $u = 0.4953$ V, $I_D = 1.837$ pA, $G = 7.857\times10^{6}$ s⁻¹, $Q/q = 1432.4$
- Saddle (unstable): $u = 0.6097$ V, $I_D = 62.2$ pA, $Q/q = 1626.3$
- LRS: $u = 0.9284$ V, $I_D = 7.62$ µA, $Q/q = 3162.7$

The charge barrier between well and saddle is about 194 holes.`,
      },
      equations: [
        {
          id: "eq-cb-landscape",
          label: { ko: "u마다 V_D를 맞춘 생성·손실률", en: "Generation and loss rates at fixed V_D" },
          tex: String.raw`\begin{aligned} &V_D(u, r) = V_D^{\mathrm{set}}\ \Rightarrow\ r(u)\qquad (\text{brentq},\ r \in [0,\ V_D^{\mathrm{set}} - u],\ \mathrm{xtol} = 10^{-11})\\ &G = \frac{I_D - I_{\mathrm{seed}} - I_{\mathrm{ch}}}{q},\qquad L = \frac{I_{\mathrm{bulk}} + I_{\mathrm{DIFF}} + I_J}{q},\qquad F = q\,(G - L)\\ &G_{\mathrm{unit}} = \frac{I_{\mathrm{BTBT}} + I_{\mathrm{GIDL}} + I_{\mathrm{PH}}}{q},\qquad G_{\mathrm{II}} = \max\!\big(G - G_{\mathrm{unit}},\ 0\big) \end{aligned}`,
          note: {
            ko: "`state()` 열: 2 = $I_D$, 3 = $G$, 4 = $L$ [1/s], 8 = $F$ [A], 9 = seed, 10 = unit 사건 전류 [A]. 근: $u$가 커질 때 $F$가 +→−이면 안정(HRS 우물, LRS), −→+이면 불안정(안장).",
            en: "`state()` columns: 2 = $I_D$, 3 = $G$, 4 = $L$ [1/s], 8 = $F$ [A], 9 = seed, 10 = unit-event current [A]. Roots: $F$ going +→− with increasing $u$ is stable (HRS well, LRS); −→+ is unstable (saddle).",
          },
          code: "setup_photo.py · state(); compound_fpt.py · make_lattice()",
        },
      ],
    },
    {
      heading: { ko: "전하 좌표와 준전위", en: "Charge coordinate and quasi-potential" },
      equations: [
        {
          id: "eq-cb-Q",
          label: { ko: "FPT 격자·회로 소자의 전하 좌표", en: "Charge coordinate of the FPT lattice and circuit element" },
          tex: String.raw`\begin{aligned} Q &= C_{\mathrm{ox}}\,\psi + Q_{\mathrm{exc}} + q\,N_A\,A\,L_n,\qquad \psi = u - V_T\ln(1 + \delta/N_A)\\ Q_{\mathrm{exc}} &= q_b + q_a = z_{13} - C_{\mathrm{ox}}u = Q_{\mathrm{pair}} + Q_{\mathrm{acc}},\qquad q_a = \rho\,q_b,\quad \rho = \frac{t_{\mathrm{acc}}}{T_{\mathrm{Si}}}\,\frac{L_{\mathrm{acc}}}{L_n} \end{aligned}`,
          note: {
            ko: "격자 좌표는 $x = Q/q$(정수 정공 수, `arange(ceil(lo), ceil(end))`). $C_{\\mathrm{ox}}/q = 1529$ 정공/V. $qN_AAL_n$은 중성 영역의 억셉터 전하로, $w_d$가 커지면 줄어든다. 회로 소자는 여기서 상수 $-C_{\\mathrm{ox}}V_G$만큼 다른 좌표를 쓴다.",
            en: "Lattice coordinate $x = Q/q$ (integer hole counts, `arange(ceil(lo), ceil(end))`). $C_{\\mathrm{ox}}/q = 1529$ holes/V. $qN_AAL_n$ is the acceptor charge of the neutral region and shrinks as $w_d$ grows. The circuit element uses this coordinate up to the constant $-C_{\\mathrm{ox}}V_G$.",
          },
          code: "setup_photo.py · state() cols 5–7; compound_fpt.py · make_lattice()",
        },
        {
          id: "eq-cb-U",
          label: { ko: "준전위 (웹 표시용)", en: "Quasi-potential (web display)" },
          tex: String.raw`U(x) = -\sum_{x_w}^{x} \ln\frac{G(x')}{L(x')},\qquad U(x_w) = 0`,
          note: {
            ko: "`WEB_CONTRACT.md` §2의 정의(백엔드 계산): 모든 생성을 단위 계단으로 보는 birth–death 근사이므로 시각화용이다. FPT 자체는 II 클러스터를 포함한 compound-jump 격자를 쓴다(`first-passage`).",
            en: "Definition of `WEB_CONTRACT.md` §2 (computed by the backend): a birth–death approximation that treats all generation as unit steps, so it is a visual guide. The FPT itself uses the compound-jump lattice with II clusters (`first-passage`).",
          },
          code: "server compute kind charge_balance (WEB_CONTRACT §2)",
        },
      ],
      notes: [
        {
          ko: "코드에는 전하 정의가 세 가지 있다: (1) `components()`의 z[13] $= C_{\\mathrm{ox}}u + Q_{\\mathrm{exc}}$ ($u$ 사용, 공핍 항 없음), (2) `export_tables.py`와 MODEL_SPEC §1·§5의 $Q_B = Q_{\\mathrm{exc}}$, (3) FPT 격자·회로 소자의 $Q = C_{\\mathrm{ox}}\\psi + Q_{\\mathrm{exc}} + qN_AAL_n$. 동역학 좌표로는 (3)만 쓰인다.",
          en: "The code has three charge definitions: (1) z[13] of `components()` $= C_{\\mathrm{ox}}u + Q_{\\mathrm{exc}}$ (uses $u$, no depletion term); (2) $Q_B = Q_{\\mathrm{exc}}$ in `export_tables.py` and MODEL_SPEC §1/§5; (3) $Q = C_{\\mathrm{ox}}\\psi + Q_{\\mathrm{exc}} + qN_AAL_n$ for the FPT lattice and circuit element. Only (3) is used as a dynamic coordinate.",
        },
        {
          ko: "결정론 가지는 준정적이다: 고정 $u$에서 $r$로 $F = 0$을 맞추고, 변위·용량 전류는 없다(과도 응답은 `circuit-element`).",
          en: "Deterministic branches are quasi-static: $F = 0$ is imposed through $r$ at fixed $u$, with no displacement or capacitive current (transients: `circuit-element`).",
        },
        {
          ko: "$G$에는 모든 II(BJT, 채널, 광, 국소)가 들어가며, 격자의 II 클러스터율은 unit 사건을 뺀 나머지다.",
          en: "$G$ contains every II contribution (BJT, channel, photo, local); the lattice II cluster rate is the remainder after the unit events.",
        },
      ],
    },
  ],
  related: ["overview", "bjt-transport", "impact-ionization", "numerics", "first-passage", "stochastic-events", "circuit-element"],
  codeRefs: [
    "photo_extension/photo_mean.py",
    "photo_extension/setup_photo.py",
    "model/janus_calibration_20260920/hypothesis_study_20260920/compound_fpt.py",
    "docs/MODEL_SPEC.md",
  ],
};

export default topic;
