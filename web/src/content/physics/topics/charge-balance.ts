// Physics topic "charge-balance" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "charge-balance",
  title: { ko: "전하 균형(Eq. 1), 정상상태, fold", en: "Charge balance (Eq. 1), steady state and folds" },
  summary: {
    ko: String.raw`바디 정공 전하 균형식 $dQ_B/dt = F(u, r)$의 정상해 $F = 0$과 Kirchhoff 법칙으로 궤적 $V_D(u)$, $I_D(u)$를 얻는다. $V_D(u)$의 첫 극대와 마지막 극소가 각각 $V_{\mathrm{LU}}$, $V_{\mathrm{LD}}$이다. $V_D$를 고정하면 $F$의 부호로부터 바디 전하 지형(안정 근과 불안정 근)이 정해진다.`,
    en: String.raw`The steady state $F = 0$ of the body hole-charge balance $dQ_B/dt = F(u, r)$, together with Kirchhoff's law, gives the locus $V_D(u)$, $I_D(u)$; its first maximum and last minimum are $V_{\mathrm{LU}}$ and $V_{\mathrm{LD}}$. At fixed $V_D$ the sign of $F$ defines the body-charge landscape (stable and unstable roots).`,
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
            ko: "코드: `net = ii + ii_ch + bbj + gidl + iph + ii_ph + iloc - diff - bulk - junction`. 단위는 A이며, 양수는 정공이 바디로 들어오는 방향이다.",
            en: "Code: `net = ii + ii_ch + bbj + gidl + iph + ii_ph + iloc - diff - bulk - junction`. In A; positive means holes flowing into the body.",
          },
          code: "photo_mean.py · components() (net)",
        },
      ],
      variables: [
        {
          symbol: String.raw`I_{\mathrm{II}}`,
          name: { ko: "증배로 생긴 정공 (impact-ionization)", en: "Multiplication holes (impact-ionization)" },
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
          name: { ko: "바디 SRH + 접합 SRH (bjt-transport)", en: "Body SRH + junction SRH (bjt-transport)" },
          code: "bulk + junction = z[5] + z[7]",
        },
        {
          symbol: String.raw`I_{\mathrm{DIFF}}`,
          name: { ko: "이미터 확산 (bjt-transport)", en: "Emitter diffusion (bjt-transport)" },
          code: "diff = z[6]",
        },
      ],
    },
    {
      heading: { ko: "정상상태와 Kirchhoff 법칙", en: "Steady state and Kirchhoff's law" },
      equations: [
        {
          id: "eq-cb-ss",
          label: { ko: "u로 매개화한 정상상태 궤적", en: "Steady-state locus parameterized by u" },
          tex: String.raw`F\big(u, r^{\star}(u)\big) = 0\ \ \Rightarrow\ \ V_D(u) = V_D\big(u, r^{\star}(u)\big),\qquad I_D(u) = I_D\big(u, r^{\star}(u)\big)`,
          note: {
            ko: "$u$가 독립적인 추적 변수다. 궤적은 $u$에 대해서는 단일값이지만 $V_D$에 대해서는 다중값(S자)이다. $F$는 $r$에 대해 증가하므로(증배, BTBT, GIDL 증가) $r$은 구간 이분법으로 찾는다.",
            en: "$u$ is the independent tracing variable: the locus is single-valued in $u$ but multi-valued (S-shaped) in $V_D$. $F$ increases with $r$ (more multiplication, BTBT and GIDL), so $r$ is found by bracketed bisection.",
          },
          code: "photo_mean.py · curve_grid()",
        },
        {
          id: "eq-cb-bisect",
          label: { ko: "u마다 r 풀기", en: "Solving for r at each u" },
          tex: String.raw`\begin{aligned} &r \in \big[10^{-100},\ r_{\mathrm{hi}}\big],\qquad r_{\mathrm{hi}} = \min\!\big(5\ \mathrm{V},\ r_{\mathrm{geo}}(u)\big)\\ &\text{require } F(u, 0) \le 0 \text{ (finite)},\quad F(u, r_{\mathrm{hi}}) \ge 0 \text{ or NaN}\\ &29 \text{ bisections in } \ln r:\ \ F(u, r_{\mathrm{mid}}) > 0 \text{ or NaN} \Rightarrow r_{\mathrm{hi}} \leftarrow r_{\mathrm{mid}},\ \text{else } r_{\mathrm{lo}} \leftarrow r_{\mathrm{mid}}\\ &\text{accept if } |F| \le 10^{-5}\max(I_D,\,10^{-25}) \end{aligned}`,
          note: {
            ko: String.raw`branch 배열의 열 구성: 0–16 = z[0..16], 17 = $u$, 18 = $r$, 19 = $E_{\mathrm{pk}}$, 20 = 정공 강하(V). $u = 0$은 항상 $(u, r) = (0, 0)$으로 넣는다.`,
            en: String.raw`Branch-array columns: 0–16 = z[0..16], 17 = $u$, 18 = $r$, 19 = $E_{\mathrm{pk}}$, 20 = hole drop (V). $u = 0$ is always inserted as $(u, r) = (0, 0)$.`,
          },
          code: "photo_mean.py · curve_grid()",
        },
      ],
    },
    {
      heading: { ko: "fold 검출", en: "Fold detection" },
      body: {
        ko: "계산값(격자 601점, `VALIDATION.md`와 일치):\n\n- $V_G = -2$ V 암조건: $V_{\\mathrm{LU}} = 3.7037$ V, $V_{\\mathrm{LD}} = 2.5979$ V (창 1.106 V)\n- $V_G = -1.8$ V 암조건: 3.8644 V, 2.5979 V\n- $V_G = -1.8$ V, $I_{\\mathrm{PH}} = 2.63$ pA: 3.2913 V, 2.5962 V\n\n$V_G = -2$ V의 fold 상태는 앱의 결정론 `folds` 표시와 같은 방식으로 보정한 값이다($V_D(u)$ 3점 포물선의 꼭짓점 $u$, $I_D$는 $u$에 대한 로그 보간, $r$은 선형 보간). LU에서는 $u = 0.5673$ V, $r \\approx 3.136$ V, $I_D = 14.89$ pA이고, LD에서는 $u = 0.7601$ V, $r \\approx 1.831$ V, $I_D = 16.33$ nA이다. 가장 가까운 601 격자 행은 LU $u = 0.5682$ V, $r = 3.1355$ V, $I_D = 15.30$ pA, LD $u = 0.7607$ V, $r = 1.8301$ V, $I_D = 16.66$ nA이다(다른 주제에 나오는 fold 상태 값은 이 격자 행의 값이다).",
        en: "Computed values (601-point grid, matching `VALIDATION.md`):\n\n- $V_G = -2$ V dark: $V_{\\mathrm{LU}} = 3.7037$ V, $V_{\\mathrm{LD}} = 2.5979$ V (window 1.106 V)\n- $V_G = -1.8$ V dark: 3.8644 V, 2.5979 V\n- $V_G = -1.8$ V, $I_{\\mathrm{PH}} = 2.63$ pA: 3.2913 V, 2.5962 V\n\nThe fold states at $V_G = -2$ V are refined as in the app's deterministic `folds` readout (vertex $u$ of the 3-point parabola $V_D(u)$, $I_D$ log-interpolated in $u$, $r$ linearly interpolated): at LU $u = 0.5673$ V, $r \\approx 3.136$ V, $I_D = 14.89$ pA; at LD $u = 0.7601$ V, $r \\approx 1.831$ V, $I_D = 16.33$ nA. The nearest 601-grid rows are LU $u = 0.5682$ V, $r = 3.1355$ V, $I_D = 15.30$ pA and LD $u = 0.7607$ V, $r = 1.8301$ V, $I_D = 16.66$ nA (fold-state values quoted in other topics are these grid rows).",
      },
      equations: [
        {
          id: "eq-cb-fold",
          label: { ko: "극값 탐색과 포물선 보정", en: "Extremum search and parabolic refinement" },
          tex: String.raw`\begin{aligned} &\Delta V_k = V_{D,k+1} - V_{D,k};\qquad \max\!: \Delta V_{k-1} > 0 > \Delta V_k,\qquad \min\!: \Delta V_{k-1} < 0 < \Delta V_k\\ &i = \text{first max},\qquad j = \text{last min with } j > i\\ &V_D \approx c_0 x^2 + c_1 x + c_2,\ \ x = u - u_k\ \ (k-1, k, k+1)\ \Rightarrow\ V_{\mathrm{fold}} = V_D\!\left(-\frac{c_1}{2c_0}\right) \end{aligned}`,
          note: {
            ko: "점이 8개 미만이거나, 극대가 없거나, $i$ 뒤에 극소가 없으면 `None`(래치 없음)을 돌려준다. 코드 주석: 최적화 도중 중간 fold가 생길 수 있으므로 마지막 안정 branch를 쓴다.",
            en: "Returns `None` (no latch) with fewer than 8 points, no maximum, or no minimum after $i$. Code comment: intermediate folds may appear during optimization, so the terminal stable branch is used.",
          },
          code: "photo_mean.py · FastModel.classify()",
        },
      ],
    },
    {
      heading: { ko: "branch와 준정적 스윕", en: "Branches and the quasi-static sweep" },
      body: {
        ko: "- HRS = `b[:i+1]`, 불안정 branch = `b[i:j+1]`, LRS = `b[j:]`이다(웹 API의 분할). 불안정 branch에서는 $dV_D/du < 0$이다.\n- 상승 스윕은 HRS를 따라 $V_{\\mathrm{LU}}$까지 올라간 뒤 LRS로 점프하고, 하강 스윕은 LRS를 따라 $V_{\\mathrm{LD}}$까지 내려간 뒤 HRS로 떨어진다.\n- `double_curve`는 각 단조 구간에서 $\\ln I_D$를 $V_D$에 대해 보간하고, fold 꼭짓점마다 상태를 한 점 더 계산해 붙이며, $V_D = 0$에서는 $I_D = 0$(평형)으로 둔다.",
        en: "- HRS = `b[:i+1]`, unstable branch = `b[i:j+1]`, LRS = `b[j:]` (split used by the web API). On the unstable branch $dV_D/du < 0$.\n- The up-sweep follows the HRS up to $V_{\\mathrm{LU}}$ and jumps to the LRS; the down-sweep follows the LRS down to $V_{\\mathrm{LD}}$ and drops to the HRS.\n- `double_curve` interpolates $\\ln I_D$ versus $V_D$ on each monotonic segment, appends one extra state at each fold vertex, and sets $I_D = 0$ at $V_D = 0$ (equilibrium).",
      },
    },
    {
      heading: { ko: "고정 V_D에서의 바디 전하 지형", en: "Body-charge landscape at fixed V_D" },
      body: {
        ko: String.raw`예($V_G = -2$ V, 암조건, $V_D = 3.5$ V, 계산값):

- HRS 우물(well): $u = 0.4953$ V, $I_D = 1.837$ pA, $G = 7.857\times10^{6}$ s⁻¹, $Q/q = 1432.4$
- 안장점(saddle, 불안정): $u = 0.6097$ V, $I_D = 62.2$ pA, $Q/q = 1626.3$
- LRS: $u = 0.9284$ V, $I_D = 7.62$ µA, $Q/q = 3162.7$

우물과 안장점 사이의 전하 장벽은 정공 약 194개다.`,
        en: String.raw`Example ($V_G = -2$ V, dark, $V_D = 3.5$ V, computed):

- HRS well: $u = 0.4953$ V, $I_D = 1.837$ pA, $G = 7.857\times10^{6}$ s⁻¹, $Q/q = 1432.4$
- Saddle (unstable): $u = 0.6097$ V, $I_D = 62.2$ pA, $Q/q = 1626.3$
- LRS: $u = 0.9284$ V, $I_D = 7.62$ µA, $Q/q = 3162.7$

The charge barrier between the well and the saddle is about 194 holes.`,
      },
      equations: [
        {
          id: "eq-cb-landscape",
          label: { ko: "고정 V_D에서의 생성률과 손실률", en: "Generation and loss rates at fixed V_D" },
          tex: String.raw`\begin{aligned} &V_D(u, r) = V_D^{\mathrm{set}}\ \Rightarrow\ r(u)\qquad (\text{brentq},\ r \in [0,\ V_D^{\mathrm{set}} - u],\ \mathrm{xtol} = 10^{-11})\\ &G = \frac{I_D - I_{\mathrm{seed}} - I_{\mathrm{ch}}}{q},\qquad L = \frac{I_{\mathrm{bulk}} + I_{\mathrm{DIFF}} + I_J}{q},\qquad F = q\,(G - L)\\ &G_{\mathrm{unit}} = \frac{I_{\mathrm{BTBT}} + I_{\mathrm{GIDL}} + I_{\mathrm{PH}}}{q},\qquad G_{\mathrm{II}} = \max\!\big(G - G_{\mathrm{unit}},\ 0\big) \end{aligned}`,
          note: {
            ko: "`state()`의 열: 2 = $I_D$, 3 = $G$, 4 = $L$ [1/s], 8 = $F$ [A], 9 = 시드 전류, 10 = 단위 사건 전류 [A]. 근의 안정성: $u$가 커질 때 $F$가 +에서 −로 바뀌면 안정(HRS 우물, LRS), −에서 +로 바뀌면 불안정(안장점)이다.",
            en: "`state()` columns: 2 = $I_D$, 3 = $G$, 4 = $L$ [1/s], 8 = $F$ [A], 9 = seed current, 10 = unit-event current [A]. Roots: $F$ changing from + to − with increasing $u$ is stable (HRS well, LRS); from − to + it is unstable (saddle).",
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
          label: { ko: "FPT 격자와 회로 소자의 전하 좌표", en: "Charge coordinate of the FPT lattice and the circuit element" },
          tex: String.raw`\begin{aligned} Q &= C_{\mathrm{ox}}\,\psi + Q_{\mathrm{exc}} + q\,N_A\,A\,L_n,\qquad \psi = u - V_T\ln(1 + \delta/N_A)\\ Q_{\mathrm{exc}} &= q_b + q_a = z_{13} - C_{\mathrm{ox}}u = Q_{\mathrm{pair}} + Q_{\mathrm{acc}},\qquad q_a = \rho\,q_b,\quad \rho = \frac{t_{\mathrm{acc}}}{T_{\mathrm{Si}}}\,\frac{L_{\mathrm{acc}}}{L_n} \end{aligned}`,
          note: {
            ko: "격자 좌표는 $x = Q/q$(정수 정공 수, `arange(ceil(lo), ceil(end))`)이다. $C_{\\mathrm{ox}}/q = 1529$ 정공/V이다. $qN_AAL_n$은 중성 영역의 억셉터 전하로, $w_d$가 커지면 줄어든다. 회로 소자는 이 좌표에서 상수 $-C_{\\mathrm{ox}}V_G$만큼 다른 좌표를 쓴다.",
            en: "The lattice coordinate is $x = Q/q$ (integer hole counts, `arange(ceil(lo), ceil(end))`). $C_{\\mathrm{ox}}/q = 1529$ holes/V. $qN_AAL_n$ is the acceptor charge of the neutral region and shrinks as $w_d$ grows. The circuit element uses the same coordinate shifted by the constant $-C_{\\mathrm{ox}}V_G$.",
          },
          code: "setup_photo.py · state() cols 5–7; compound_fpt.py · make_lattice()",
        },
        {
          id: "eq-cb-U",
          label: { ko: "준전위 (웹 표시용)", en: "Quasi-potential (web display)" },
          tex: String.raw`U(x) = -\sum_{x_w}^{x} \ln\frac{G(x')}{L(x')},\qquad U(x_w) = 0`,
          note: {
            ko: "`WEB_CONTRACT.md` §2의 정의이며 백엔드가 계산한다. 모든 생성을 단위 계단으로 보는 출생–사멸(birth–death) 근사이므로 시각화용이다. FPT 계산 자체는 II 클러스터를 포함한 복합 점프(compound-jump) 격자를 쓴다(`first-passage`).",
            en: "Definition from `WEB_CONTRACT.md` §2, computed by the backend. It is a birth–death approximation that treats all generation as unit steps, so it serves as a visual guide only. The FPT itself uses the compound-jump lattice with II clusters (`first-passage`).",
          },
          code: "server compute kind charge_balance (WEB_CONTRACT §2)",
        },
      ],
      notes: [
        {
          ko: "코드에는 전하 정의가 세 가지 있다: (1) `components()`의 z[13] $= C_{\\mathrm{ox}}u + Q_{\\mathrm{exc}}$ ($u$ 사용, 공핍 항 없음), (2) `export_tables.py`와 MODEL_SPEC §1·§5의 $Q_B = Q_{\\mathrm{exc}}$, (3) FPT 격자와 회로 소자의 $Q = C_{\\mathrm{ox}}\\psi + Q_{\\mathrm{exc}} + qN_AAL_n$. 동역학 좌표로는 (3)만 쓴다.",
          en: "The code has three charge definitions: (1) z[13] of `components()` $= C_{\\mathrm{ox}}u + Q_{\\mathrm{exc}}$ (uses $u$, no depletion term); (2) $Q_B = Q_{\\mathrm{exc}}$ in `export_tables.py` and MODEL_SPEC §1/§5; (3) $Q = C_{\\mathrm{ox}}\\psi + Q_{\\mathrm{exc}} + qN_AAL_n$ for the FPT lattice and the circuit element. Only (3) is used as a dynamic coordinate.",
        },
        {
          ko: "결정론 branch는 준정적이다. 고정된 $u$에서 $r$로 $F = 0$을 맞추며, 변위 전류나 용량성 전류는 없다(과도 응답은 `circuit-element`).",
          en: "The deterministic branches are quasi-static: $F = 0$ is imposed through $r$ at fixed $u$, with no displacement or capacitive current (transients: `circuit-element`).",
        },
        {
          ko: "$G$에는 모든 II 기여(BJT, 채널, 광, 국소 경로)가 들어가며, 격자의 II 클러스터율은 $G$에서 단위 사건을 뺀 나머지다.",
          en: "$G$ contains every II contribution (BJT, channel, photo, local path); the II cluster rate of the lattice is what remains of $G$ after the unit events.",
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
