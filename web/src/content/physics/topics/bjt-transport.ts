// Physics topic "bjt-transport" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "bjt-transport",
  title: { ko: "BJT 수송: 전류가 흐르는 준중성 베이스와 SRH", en: "BJT transport: current-carrying quasi-neutral base with SRH" },
  summary: {
    ko: "중성 바디(베이스)의 전자·정공 전류식을 무차원 ODE로 바꿔 컬렉터에서 이미터까지 적분하고, 슈팅법(shooting)으로 $j_0$를 맞춘다. 그 결과로 시드 전류, SRH 재결합, 평균 과잉 농도, 정공 준페르미 준위 강하가 나온다. 여기에 이미터 확산, 접합 SRH, 접근 저항을 더해 $I_D$와 $V_D$를 완성한다.",
    en: "The electron and hole current equations of the neutral body (base) are integrated as a dimensionless ODE from collector to emitter, with $j_0$ found by shooting. This gives the seed current, the SRH recombination, the mean excess density and the hole quasi-Fermi-level drop. Emitter diffusion, junction SRH and the access resistance then complete $I_D$ and $V_D$.",
  },
  tags: ["I_seed", "I_REC", "I_DIFF", "R_acc", "shooting"],
  sections: [
    {
      heading: { ko: "정규화", en: "Normalization" },
      equations: [
        {
          id: "eq-bjt-norm",
          label: { ko: "무차원 변수와 총전류", en: "Dimensionless variables and total current" },
          tex: String.raw`\begin{aligned} h &= \delta/N_A,\qquad z \in [0,1]\ \ (0 = \text{collector},\ 1 = \text{emitter})\\ j &= I_n/I_0,\qquad I_0 = \frac{q\,A\,D_n N_A}{L_n}\\ r_\mu &= \frac{\mu_p}{\mu_n} = \frac{150}{450} = \frac{1}{3},\qquad D_n = 450\,V_T,\quad D_p = 150\,V_T\\ J &= M j_0 + b,\qquad b = \frac{I_{\mathrm{BTBT}} + I_{\mathrm{GIDL}} + I_{\mathrm{II,ch}} + I_{\mathrm{PH}} + I_{\mathrm{II,PH}} + I_{\mathrm{loc}}}{I_0} \end{aligned}`,
          note: {
            ko: "$J$는 베이스를 지나는 (전자 + 정공) 총전류로, $z$에 무관하다. 컬렉터에서의 정공 전류 $J - j_0 = (M-1)j_0 + b$가 바디로 들어오는 정공 공급이다. 코드는 `mult`, `b = (bbj+gidl+ii_ch+iph+ii_ph+iloc)/scale`, `kappa = length²/(DN·tau)`로 `srh_solve`를 호출한다.",
            en: "$J$ is the total (electron + hole) current through the base and does not depend on $z$. At the collector the hole current $J - j_0 = (M-1)j_0 + b$ is the hole supply entering the body. The code calls `srh_solve` with `mult`, `b = (bbj+gidl+ii_ch+iph+ii_ph+iloc)/scale` and `kappa = length²/(DN·tau)`.",
          },
          code: "photo_mean.py · components() (scale, srh_solve call)",
        },
      ],
      variables: [
        {
          symbol: "D_n",
          name: { ko: "전자 확산 계수 (450 V_T)", en: "Electron diffusivity (450 V_T)" },
          value: "11.633",
          unit: "cm²/s",
          code: "DN",
        },
        {
          symbol: "D_p",
          name: { ko: "정공 확산 계수 (150 V_T)", en: "Hole diffusivity (150 V_T)" },
          value: "3.8778",
          unit: "cm²/s",
          code: "DP",
        },
        {
          symbol: "I_0",
          name: { ko: "전류 규모 (L_n = 347.4 nm)", en: "Current scale (L_n = 347.4 nm)" },
          value: "1.232 × 10⁻⁶",
          unit: "A",
          code: "scale",
        },
        {
          symbol: "I_0",
          name: { ko: "HRS fold (V_G = −2 V, L_n = 295.6 nm)", en: "HRS fold (V_G = −2 V, L_n = 295.6 nm)" },
          value: "1.448 × 10⁻⁶",
          unit: "A",
          code: "scale",
        },
      ],
    },
    {
      heading: { ko: "ODE 계", en: "ODE system" },
      equations: [
        {
          id: "eq-bjt-ode",
          label: { ko: "컬렉터 → 이미터 적분", en: "Integration from collector to emitter" },
          tex: String.raw`\begin{aligned} \frac{dh}{dz} &= \frac{j\,\big(r_\mu + (1 + r_\mu)h\big) - h\,J}{r_\mu\,(1 + 2h)}\\ \frac{dj}{dz} &= R(h)\\ \frac{d\bar q}{dz} &= h\\ \frac{d\Delta_p}{dz} &= \frac{J - j}{r_\mu\,(1 + h)} \end{aligned}`,
          note: {
            ko: String.raw`코드에서는 $dh/dz = (j - f_n J)\,g$, $f_n = h/D$, $g = D/(r_\mu(1+2h))$, $D = r_\mu + (1+r_\mu)h$ 형태로 쓴다. $J = 0$, $h \ll 1$이면 $dh/dz = j$(순수 확산), $h \gg 1$이면 $dh/dz \to 2j$(양극성 확산, $D_a = D_n/2$)이다. $\bar q$는 정규화한 평균 과잉 농도, $\Delta_p$는 정공 준페르미 준위 강하를 $V_T$로 나눈 값이다(정전기 전계의 적분이 아님).`,
            en: String.raw`Code form: $dh/dz = (j - f_n J)\,g$ with $f_n = h/D$, $g = D/(r_\mu(1+2h))$, $D = r_\mu + (1+r_\mu)h$. For $J = 0$ and $h \ll 1$, $dh/dz = j$ (pure diffusion); for $h \gg 1$, $dh/dz \to 2j$ (ambipolar, $D_a = D_n/2$). $\bar q$ is the normalized mean excess density and $\Delta_p$ the hole quasi-Fermi-level drop divided by $V_T$ (not an electrostatic field integral).`,
          },
          code: "srh_transport.py · rhs(), integrate_voltage()",
        },
      ],
    },
    {
      heading: { ko: "SRH 재결합 항", en: "SRH reaction term" },
      equations: [
        {
          id: "eq-bjt-srh",
          label: { ko: "농도에 의존하는 비대칭 SRH", en: "Density-dependent asymmetric SRH" },
          tex: String.raw`R(h) = \kappa\,\frac{h\,(1 + h)}{1 + \theta_t + (1 + \tau_r)\,h},\qquad \kappa = \frac{L_n^2}{D_n\,\tau_{\mathrm{bulk}}},\quad \tau_r = \frac{\tau_p}{\tau_n} = p_8,\quad \theta_t = \frac{(1 + \tau_r)\,n_i}{N_A}`,
          note: {
            ko: "$U = (np - n_i^2)/[\\tau_p(n + n_1) + \\tau_n(p + p_1)]$에 $n = N_A h$, $p = N_A(1+h)$, $n_1 = p_1 = n_i$(midgap 트랩), $\\tau_n = \\tau_{\\mathrm{bulk}}$ = p[1]을 넣고 $dj/dz = L_n^2 U/(D_n N_A)$로 정리한 식이다. 코드 주석: `trap_offset` $= (\\tau_p n_1 + \\tau_n p_1)/(\\tau_n N_A)$. 고주입에서는 $R \\to \\kappa h/(1+\\tau_r)$, 즉 유효 수명이 $(1+\\tau_r)\\tau_n$이 된다.",
            en: "Obtained from $U = (np - n_i^2)/[\\tau_p(n + n_1) + \\tau_n(p + p_1)]$ with $n = N_A h$, $p = N_A(1+h)$, $n_1 = p_1 = n_i$ (midgap traps) and $\\tau_n = \\tau_{\\mathrm{bulk}}$ = p[1], written as $dj/dz = L_n^2 U/(D_n N_A)$. Code comment: `trap_offset` $= (\\tau_p n_1 + \\tau_n p_1)/(\\tau_n N_A)$. At high injection $R \\to \\kappa h/(1+\\tau_r)$, i.e. the effective lifetime is $(1+\\tau_r)\\tau_n$.",
          },
          code: "srh_transport.py · reaction()",
        },
      ],
      variables: [
        {
          symbol: String.raw`\kappa`,
          name: { ko: "L_n = 347.4 nm에서", en: "At L_n = 347.4 nm" },
          value: "1.120 × 10⁻⁴",
          unit: "–",
          code: "kappa",
        },
        {
          symbol: String.raw`\kappa`,
          name: { ko: "HRS fold (V_G = −2 V)", en: "HRS fold (V_G = −2 V)" },
          value: "8.104 × 10⁻⁵",
          unit: "–",
          code: "kappa",
        },
        {
          symbol: String.raw`\theta_t`,
          name: { ko: "트랩 오프셋", en: "Trap offset" },
          value: "5.158 × 10⁻⁶",
          unit: "–",
          code: "(1+tau_ratio)*NI_CM3/na",
        },
        {
          symbol: String.raw`\tau_r`,
          name: { ko: "τ_p/τ_n", en: "τ_p/τ_n" },
          value: "117.42",
          unit: "–",
          code: "p[8]",
        },
        {
          symbol: String.raw`(1+\tau_r)\tau_n`,
          name: { ko: "고주입 유효 수명", en: "High-injection effective lifetime" },
          value: "109.7",
          unit: "µs",
          code: "p[1]·(1+p[8])",
        },
      ],
    },
    {
      heading: { ko: "경계 조건과 슈팅법", en: "Boundary conditions and shooting" },
      equations: [
        {
          id: "eq-bjt-bc",
          label: { ko: "경계 조건", en: "Boundary conditions" },
          tex: String.raw`h(0) = 0,\qquad j(0) = j_0,\qquad \bar q(0) = \Delta_p(0) = 0,\qquad h(1) = h_E = \frac{\delta(u)}{N_A}`,
          note: {
            ko: "컬렉터($z = 0$)의 소수 캐리어 경계값은 0이고(공핍층 경계가 전자를 흡수), 이미터($z = 1$)의 $h_E$는 `electrostatics`의 $\\delta(u)$로 정해진다.",
            en: "The minority-carrier boundary value is zero at the collector ($z = 0$; the depletion edge absorbs electrons), and $h_E$ at the emitter ($z = 1$) is $\\delta(u)$ from `electrostatics`.",
          },
          code: "srh_transport.py · integrate_voltage()",
        },
        {
          id: "eq-bjt-newton",
          label: { ko: "j₀에 대한 Newton 슈팅", en: "Newton shooting on j₀" },
          tex: String.raw`\begin{aligned} j_0^{(0)} &= \max\!\big(2h_E - \ln(1 + h_E),\ 10^{-100}\big)\\ j_0^{(k+1)} &= \max\!\left(0.1\,j_0^{(k)},\ \ j_0^{(k)} - \frac{h(1) - h_E}{h_s(1)}\right),\qquad h_s = \frac{\partial h}{\partial j_0}\\ \text{stop}&:\ \ |h(1) - h_E| < 10^{-9}\max(h_E,\,10^{-100}),\qquad k < 32 \end{aligned}`,
          note: {
            ko: String.raw`초기값은 $M = 1$, $b = 0$, $\kappa = 0$일 때의 정확한 해다($dh/dz = j_0(1+h)/(1+2h)$의 적분). RK4 64단계로 적분한다. $h < 0$, $h > 10^{80}$, 유한하지 않은 값, $h_s(1) \le 0$이 나오거나 32회 안에 수렴하지 않으면 NaN을 돌려준다.`,
            en: String.raw`The initial guess is the exact solution for $M = 1$, $b = 0$, $\kappa = 0$ (integral of $dh/dz = j_0(1+h)/(1+2h)$). RK4 with 64 steps. NaN is returned for $h < 0$, $h > 10^{80}$, non-finite values, $h_s(1) \le 0$, or no convergence within 32 iterations.`,
          },
          code: "srh_transport.py · solve_full_voltage()",
        },
        {
          id: "eq-bjt-sens",
          label: { ko: "민감도 방정식 (함께 적분)", en: "Sensitivity equations (integrated alongside)" },
          tex: String.raw`\begin{aligned} \frac{dh_s}{dz} &= \big[-f_n'\,J\,g + (j - f_n J)\,g'\big]\,h_s + g\,j_s - f_n\,M\,g,\qquad \frac{dj_s}{dz} = R'(h)\,h_s\\ f_n' &= \frac{r_\mu}{D^2},\qquad g' = \frac{1 - r_\mu}{r_\mu\,(1 + 2h)^2},\qquad h_s(0) = 0,\ \ j_s(0) = 1 \end{aligned}`,
          code: "srh_transport.py · rhs() (dhs, dr·hs)",
        },
      ],
    },
    {
      heading: { ko: "저주입 해석해", en: "Low-injection analytic solution" },
      equations: [
        {
          id: "eq-bjt-lowinj",
          label: { ko: "h_E < 10⁻⁸일 때의 선형 해", en: "Linear solution for h_E < 10⁻⁸" },
          tex: String.raw`\begin{aligned} &\text{if } h_E < 10^{-8} \text{ and } (1 + \tau_r)h_E < 10^{-8}:\qquad \tilde\kappa = \frac{\kappa}{1 + \theta_t},\quad k = \sqrt{\tilde\kappa}\\ &j_0 = h_E\,\frac{k}{\sinh k},\qquad \bar q = h_E\,\frac{\tanh(k/2)}{k}\qquad (k \ge 10^{-6})\\ &j_0 = h_E\,(1 - \tilde\kappa/6),\qquad \bar q = h_E\,(1/2 - \tilde\kappa/24)\qquad (k < 10^{-6})\\ &\text{if } M j_0 + b < 10^{-8}:\ \ (\text{seed}, \text{loss}, \bar q, \Delta_p) = \big(j_0,\ \tilde\kappa\,\bar q,\ \bar q,\ (M j_0 + b - h_E)/r_\mu\big) \end{aligned}`,
          note: {
            ko: String.raw`선형화한 $h'' = \tilde\kappa h$를 $h(0) = 0$, $h(1) = h_E$로 풀면 $h = h_E\sinh(kz)/\sinh k$이다. 조건을 만족하지 않으면 전체 슈팅법을 쓴다(docstring: 작은 $O(h^2, hJ)$ 항만 버린다).`,
            en: String.raw`Solution of the linearized $h'' = \tilde\kappa h$ with $h(0) = 0$, $h(1) = h_E$: $h = h_E\sinh(kz)/\sinh k$. Otherwise the full shooting is used (docstring: only small $O(h^2, hJ)$ terms are dropped).`,
          },
          code: "srh_transport.py · solve_voltage()",
        },
      ],
    },
    {
      heading: { ko: "시드 전류, 재결합, 평균 과잉 농도", en: "Seed current, recombination and mean excess" },
      equations: [
        {
          id: "eq-bjt-currents",
          label: { ko: "수송 해 → 전류", en: "Transport solution → currents" },
          tex: String.raw`\begin{aligned} I_{\mathrm{seed}} &= I_0\,j_0,\qquad I_{\mathrm{bulk}} = I_0\,\big(j(1) - j_0\big) = I_0\!\int_0^1\! R(h)\,dz\\ I_E &= I_{\mathrm{seed}} + I_{\mathrm{bulk}},\qquad \bar n = N_A\,\bar q,\qquad \text{hole drop} = V_T\,\Delta_p(1) \end{aligned}`,
          note: {
            ko: String.raw`코드 주석: 비선형 SRH 적분이므로 $I_{\mathrm{bulk}}$는 $Q_{\mathrm{pair}}/\tau$와 같지 않다. 계산값($V_G = -2$ V): HRS fold에서 $I_{\mathrm{seed}} = 9.630$ pA, $I_{\mathrm{bulk}} = 0.390$ fA, LRS fold에서 각각 14.56 nA, 0.412 pA.`,
            en: String.raw`Code comment: the SRH integral is nonlinear, so $I_{\mathrm{bulk}}$ is not $Q_{\mathrm{pair}}/\tau$. Computed ($V_G = -2$ V): $I_{\mathrm{seed}} = 9.630$ pA and $I_{\mathrm{bulk}} = 0.390$ fA at the HRS fold; 14.56 nA and 0.412 pA at the LRS fold.`,
          },
          code: "photo_mean.py · components() (seed, bulk, emitter, avg)",
        },
      ],
    },
    {
      heading: { ko: "이미터 확산 손실과 접합 SRH", en: "Emitter diffusion loss and junction SRH" },
      body: {
        ko: String.raw`계산값($V_G = -2$ V): HRS fold에서 $I_{\mathrm{DIFF}} = 1.141$ pA, $I_J = 4.532$ pA로 HRS 손실은 접합 SRH가 주도한다. LRS fold에서는 $I_{\mathrm{DIFF}} = 1.954$ nA, $I_J = 0.1437$ nA이다.`,
        en: String.raw`Computed ($V_G = -2$ V): $I_{\mathrm{DIFF}} = 1.141$ pA and $I_J = 4.532$ pA at the HRS fold, so junction SRH dominates the HRS loss; $I_{\mathrm{DIFF}} = 1.954$ nA and $I_J = 0.1437$ nA at the LRS fold.`,
      },
      equations: [
        {
          id: "eq-bjt-diff",
          label: { ko: "이미터로의 정공 확산 (β 기준)", en: "Hole diffusion into the emitter (β reference)" },
          tex: String.raw`\begin{aligned} I_{\mathrm{DIFF}} &= I_{\mathrm{sp}}\,e^{-\varphi_E/V_T}\,\big(e^{u/V_T} - 1\big),\qquad I_{\mathrm{sp}} = \frac{q\,A\,D_n\,n_i^2}{N_A\,L_{\mathrm{ref}}\,\beta}\\ L_{\mathrm{ref}} &= L - 2\sqrt{\frac{2\varepsilon\,V_{\mathrm{bi}}}{q\,N_A}} = 347.4\ \mathrm{nm} \end{aligned}`,
          note: {
            ko: String.raw`$\beta$ = p[0]은 고정된 영바이어스 베이스 길이 $L_{\mathrm{ref}}$에서의 저주입 기준 비율($I_{e,\mathrm{injected}}/I_{h,\mathrm{out}}$)이고, $\varphi_E$ = p[10]은 소스 가장자리 국소 상태다. $I_{\mathrm{sp}} = 3.261\times10^{-22}$ A이며, $e^{-\varphi_{E0}/V_T}$를 곱하면 $3.253\times10^{-22}$ A가 된다. 고주입에서도 $e^{u/V_T}$ 스케일링을 그대로 유지한다.`,
            en: String.raw`$\beta$ = p[0] is the low-injection reference ratio ($I_{e,\mathrm{injected}}/I_{h,\mathrm{out}}$) at the fixed zero-bias base length $L_{\mathrm{ref}}$, and $\varphi_E$ = p[10] is the source-edge local state. $I_{\mathrm{sp}} = 3.261\times10^{-22}$ A, or $3.253\times10^{-22}$ A after multiplying by $e^{-\varphi_{E0}/V_T}$. The $e^{u/V_T}$ scaling is kept at high injection.`,
          },
          code: "photo_mean.py · components() (lref, isp, diff)",
        },
        {
          id: "eq-bjt-junction",
          label: { ko: "소스 공핍층 SRH", en: "Source depletion-region SRH" },
          tex: String.raw`I_J = \frac{q\,A\,w_s\,n_i}{2\,\tau_j}\,\big(e^{u/(2V_T)} - 1\big)`,
          note: {
            ko: String.raw`체적 $A\,w_s$에서의 대칭 midgap SRH이며 $\tau_j$ = p[2] = 5.358 ns이다. $u = 0$($w_s = 76.30$ nm)에서 앞 계수는 $1.141\times10^{-16}$ A이다.`,
            en: String.raw`Symmetric midgap SRH in the volume $A\,w_s$, with $\tau_j$ = p[2] = 5.358 ns. Prefactor at $u = 0$ ($w_s = 76.30$ nm): $1.141\times10^{-16}$ A.`,
          },
          code: "photo_mean.py · components() (junction)",
        },
      ],
    },
    {
      heading: { ko: "접근 저항", en: "Access resistance" },
      equations: [
        {
          id: "eq-bjt-racc",
          label: { ko: "독립된 접근 영역 slab", en: "Independent access slab" },
          tex: String.raw`R_{\mathrm{acc}} = \frac{L_{\mathrm{acc}}}{W\,t_{\mathrm{acc}}\,\sigma},\qquad \sigma = q\,\big[450\,\bar n + 150\,(N_{A,\mathrm{acc}} + \bar n)\big]\ \ [\mathrm{S/cm}]`,
          note: {
            ko: String.raw`길이는 cm 단위다(p[7], p[5]는 nm × 10⁻⁷). 코드 주석: 이 농도 공유 가정은 Poisson 방정식에서 유도한 것이 아니며, 접근 영역의 과잉 농도를 베이스 평균 과잉 농도 $\bar n$과 같다고 둔다. 계산값: $\bar n = 0$에서 439.5 kΩ, LRS fold($V_G = -2$ V)에서 417.9 kΩ.`,
            en: String.raw`Lengths in cm (p[7] and p[5] are in nm × 10⁻⁷). Code comment: this density-sharing assumption is not derived from Poisson's equation — the access excess is set equal to the mean base excess $\bar n$. Computed: 439.5 kΩ at $\bar n = 0$ and 417.9 kΩ at the LRS fold ($V_G = -2$ V).`,
          },
          code: "photo_mean.py · components() (sigma, racc)",
        },
      ],
    },
    {
      heading: { ko: "드레인 전류, 단자 전압, 전하 항", en: "Drain current, terminal voltage and charge terms" },
      equations: [
        {
          id: "eq-bjt-id",
          label: { ko: "드레인 전류", en: "Drain current" },
          tex: String.raw`I_D = I_{\mathrm{seed}} + (M-1)\,I_{\mathrm{seed}} + I_{\mathrm{BTBT}} + I_{\mathrm{GIDL}} + I_{\mathrm{ch}} + I_{\mathrm{II,ch}} + I_{\mathrm{PH}} + I_{\mathrm{II,PH}} + I_{\mathrm{loc}}`,
          code: "photo_mean.py · components() (drain)",
        },
        {
          id: "eq-bjt-vd",
          label: { ko: "단자 전압", en: "Terminal voltage" },
          tex: String.raw`V_D = u + r + V_T\,\Delta_p + (R_c + R_{\mathrm{acc}})\,I_D`,
          note: {
            ko: String.raw`LRS fold($V_G = -2$ V)에서의 전압 분해: $u + r = 2.5908$ V, $V_T\Delta_p = 0.126$ mV, $(R_c + R_{\mathrm{acc}})I_D = 6.96$ mV → $V_D = 2.5979$ V.`,
            en: String.raw`Voltage budget at the LRS fold ($V_G = -2$ V): $u + r = 2.5908$ V, $V_T\Delta_p = 0.126$ mV, $(R_c + R_{\mathrm{acc}})I_D = 6.96$ mV → $V_D = 2.5979$ V.`,
          },
          code: "photo_mean.py · components() (vd)",
        },
        {
          id: "eq-bjt-charge",
          label: { ko: "전하 출력 z[13]", en: "Charge output z[13]" },
          tex: String.raw`Q_{\mathrm{pair}} = q\,A\,L_n\,\bar n,\qquad Q_{\mathrm{acc}} = q\,W\,t_{\mathrm{acc}}\,L_{\mathrm{acc}}\,\bar n,\qquad z_{13} = C_{\mathrm{ox}}\,u + Q_{\mathrm{pair}} + Q_{\mathrm{acc}}`,
          note: {
            ko: "FPT 격자와 회로 소자는 이와 다른 전하 좌표를 쓴다(`charge-balance`).",
            en: "The FPT lattice and the circuit element use a different charge coordinate (`charge-balance`).",
          },
          code: "photo_mean.py · components() (qpair, qaccess, charge)",
        },
      ],
      notes: [
        {
          ko: "Auger 재결합은 없고, 이동도는 상수(450/150 cm²/V·s)이며 Einstein 관계를 쓴다. 내부는 준중성이고, 컬렉터의 소수 캐리어 경계값은 0이다.",
          en: "No Auger recombination; constant mobilities (450/150 cm²/V·s) with the Einstein relation; quasi-neutral interior; zero minority-carrier boundary value at the collector.",
        },
        {
          ko: "유효 수명: 횡방향 분포가 평탄하다고 보므로 벌크 SRH와 계면 SRH를 구분할 수 없다(docstring).",
          en: "Effective lifetime: with a flat transverse profile, bulk SRH cannot be distinguished from interface SRH (docstring).",
        },
        {
          ko: "완전한 Poisson/축퇴/전계 의존 이동도 모델이 아니다. '접근 영역 과잉 농도 = 베이스 평균 과잉 농도'는 검증되지 않은 준정적 closure다.",
          en: "Not a full Poisson/degenerate/field-dependent-mobility model. 'Access excess = mean base excess' is an unverified quasi-static closure.",
        },
        {
          ko: String.raw`LRS는 고주입 영역까지 올라간다($V_G = -2$ V, $V_D = 4$ V에서 $h_E \approx 5.0$, $I_D = 26.5$ µA).`,
          en: String.raw`The LRS reaches high injection ($h_E \approx 5.0$ and $I_D = 26.5$ µA at $V_G = -2$ V, $V_D = 4$ V).`,
        },
        {
          ko: "`photo_mean.py`의 `transport_solve()`(SRH 비대칭 없음)와 쌍 표(`make_table`, `lookup`)는 쓰이지 않는 이전 코드이며, `table` 인자는 자리만 채우는 값이다.",
          en: "`transport_solve()` (no SRH asymmetry) and the pair table (`make_table`, `lookup`) in `photo_mean.py` are unused legacy code; the `table` argument is a placeholder.",
        },
      ],
    },
  ],
  related: ["electrostatics", "impact-ionization", "charge-balance", "parameters", "local-states"],
  codeRefs: [
    "model/janus_calibration_20260920/idvd_model_v3/srh_transport.py",
    "photo_extension/photo_mean.py",
    "model/janus_calibration_20260920/high_injection_review/current_carrying_transport.py",
  ],
};

export default topic;
