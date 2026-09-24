// Physics topic "electrostatics" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "electrostatics",
  title: { ko: "정전기: 주입, 공핍, 중성 길이", en: "Electrostatics: injection, depletion, neutral length" },
  summary: {
    ko: String.raw`열전압, built-in 전위, source 접합 주입 수준 $\delta(u)$, 두 접합의 공핍폭, 중성 base 길이 $L_n$, 게이트 용량 — 모든 전류 항이 공유하는 1D abrupt-junction 정전기.`,
    en: String.raw`Thermal voltage, built-in potential, source-junction injection level $\delta(u)$, the two depletion widths, the neutral base length $L_n$ and the gate capacitance — the 1D abrupt-junction electrostatics shared by every current term.`,
  },
  tags: ["deterministic", "cm units"],
  sections: [
    {
      heading: { ko: "열전압과 built-in 전위", en: "Thermal voltage and built-in potential" },
      equations: [
        {
          id: "eq-es-vt",
          label: { ko: "열전압", en: "Thermal voltage" },
          tex: String.raw`V_T = \frac{k_B T}{q} = 25.852\ \mathrm{mV},\qquad T = 300\ \mathrm{K}`,
          code: "mean_model.py · VT = KB*T/Q",
        },
        {
          id: "eq-es-vbi",
          label: { ko: "built-in 전위 (source, drain 공통)", en: "Built-in potential (source and drain)" },
          tex: String.raw`V_{\mathrm{bi}} = V_T \ln\frac{N_D\,N_A}{n_i^2} = 1.0334\ \mathrm{V}`,
          code: "mean_model.py · Field.__init__ (self.vbi)",
        },
        {
          id: "eq-es-eps",
          label: { ko: "유전율 (cm 단위)", en: "Permittivity (cm units)" },
          tex: String.raw`\varepsilon = \frac{11.7\,\varepsilon_0}{100}\ \ [\mathrm{F/cm}],\qquad \varepsilon_0 = 8.8541878128\times10^{-12}\ \mathrm{F/m}`,
          note: {
            ko: "코드의 `eps = 11.7*EPS0/100`. 이 값으로 계산하는 모든 길이는 cm이다.",
            en: "Code: `eps = 11.7*EPS0/100`. Every length computed with it is in cm.",
          },
          code: "photo_mean.py · components() (eps)",
        },
      ],
      variables: [
        {
          symbol: "q",
          name: { ko: "기본 전하", en: "Elementary charge" },
          value: "1.602176634 × 10⁻¹⁹",
          unit: "C",
          code: "standard_mean.Q",
        },
        {
          symbol: "k_B",
          name: { ko: "볼츠만 상수", en: "Boltzmann constant" },
          value: "1.380649 × 10⁻²³",
          unit: "J/K",
          code: "standard_mean.KB",
        },
        {
          symbol: "V_T",
          name: { ko: "열전압", en: "Thermal voltage" },
          value: "25.852",
          unit: "mV",
          code: "VT",
        },
        {
          symbol: String.raw`V_{\mathrm{bi}}`,
          name: { ko: "built-in 전위", en: "Built-in potential" },
          value: "1.0334",
          unit: "V",
          code: "MODEL.vbi",
        },
        {
          symbol: String.raw`\varepsilon`,
          name: { ko: "Si 유전율", en: "Si permittivity" },
          value: "1.0359 × 10⁻¹²",
          unit: "F/cm",
          code: "eps",
        },
      ],
    },
    {
      heading: { ko: "주입 수준 δ(u)", en: "Injection level δ(u)" },
      body: {
        ko: String.raw`계산값(논문 소자, $V_G = -2$ V):

- HRS fold ($u = 0.5682$ V): $\delta/N_A = 6.651\times10^{-6}$ (저주입)
- LRS fold ($u = 0.7607$ V): $\delta/N_A = 0.01127$
- LRS의 $V_D = 4$ V ($u = 0.964$ V): $\delta/N_A \approx 5.0$ (고주입)`,
        en: String.raw`Computed values (paper device, $V_G = -2$ V):

- HRS fold ($u = 0.5682$ V): $\delta/N_A = 6.651\times10^{-6}$ (low injection)
- LRS fold ($u = 0.7607$ V): $\delta/N_A = 0.01127$
- LRS at $V_D = 4$ V ($u = 0.964$ V): $\delta/N_A \approx 5.0$ (high injection)`,
      },
      equations: [
        {
          id: "eq-es-delta",
          label: { ko: "중성 body 과잉 전자 농도", en: "Excess electron density in the neutral body" },
          tex: String.raw`\delta(u) = \frac{2\,n_i^2\left(e^{u/V_T}-1\right)}{N_A + \sqrt{N_A^2 + 4\,n_i^2\left(e^{u/V_T}-1\right)}}`,
          note: {
            ko: String.raw`$\delta\,(N_A + \delta) = n_i^2\,(e^{u/V_T}-1)$의 양의 근이다(준중성 body에서 $np = n_i^2 e^{u/V_T}$, $p = N_A + \delta$, 평형 전자 $n_0 = n_i^2/N_A \approx 436$ cm⁻³의 $n_0\delta$ 항은 무시). $h_E = \delta/N_A$가 수송 해법의 emitter 경계값이고 출력 z[10]이다.`,
            en: String.raw`Positive root of $\delta\,(N_A + \delta) = n_i^2\,(e^{u/V_T}-1)$ (quasi-neutral body with $np = n_i^2 e^{u/V_T}$, $p = N_A + \delta$; the $n_0\delta$ term with $n_0 = n_i^2/N_A \approx 436$ cm⁻³ is dropped). $h_E = \delta/N_A$ is the emitter boundary value of the transport solver and output z[10].`,
          },
          code: "photo_mean.py · components() (prod, delta)",
        },
      ],
    },
    {
      heading: { ko: "Source 장벽, ψ, source 공핍폭", en: "Source barrier, ψ and source depletion width" },
      body: {
        ko: "계산값: $w_s(u=0) = 76.30$ nm, HRS fold 51.19 nm, LRS fold 39.22 nm ($V_G = -2$ V).",
        en: "Computed: $w_s(u=0) = 76.30$ nm, 51.19 nm at the HRS fold, 39.22 nm at the LRS fold ($V_G = -2$ V).",
      },
      equations: [
        {
          id: "eq-es-psi",
          label: { ko: "정전기적 body 전위", en: "Electrostatic body potential" },
          tex: String.raw`\psi = u - V_T\ln\!\left(1+\frac{\delta}{N_A}\right)`,
          note: {
            ko: String.raw`코드 주석: 접합의 정전기적 감소량은 $u$ 자체가 아니라 $u - V_T\ln(p_{\mathrm{source}}/N_A)$이다(준중성 근사). $\psi$는 FPT 격자와 회로 소자의 전하 좌표에 쓰인다.`,
            en: String.raw`Code comment: the junction electrostatic reduction is $u - V_T\ln(p_{\mathrm{source}}/N_A)$, not $u$ itself (neutrality approximation). $\psi$ is used in the charge coordinate of the FPT lattice and the circuit element.`,
          },
          code: "photo_mean.py · components(); setup_photo.py · state() (psi)",
        },
        {
          id: "eq-es-ws",
          label: { ko: "source 장벽과 공핍폭", en: "Source barrier and depletion width" },
          tex: String.raw`\begin{aligned} V_{\mathrm{bi}} - \psi &= V_{\mathrm{bi}} - u + V_T\ln\!\left(1+\frac{\delta}{N_A}\right)\\ w_s &= \sqrt{\frac{2\varepsilon\,(V_{\mathrm{bi}} - \psi)}{q\,N_A}}\quad [\mathrm{cm}] \end{aligned}`,
          note: {
            ko: "장벽 $V_{\\mathrm{bi}} - \\psi \\le 0$(flat-band 이상)이면 `components()`가 NaN 19개를 반환하고 `curve_grid()`는 그 $u$를 건너뛴다.",
            en: "If the barrier $V_{\\mathrm{bi}} - \\psi \\le 0$ (flat band or beyond), `components()` returns 19 NaNs and `curve_grid()` skips that $u$.",
          },
          code: "photo_mean.py · components() (source_barrier, ws)",
        },
      ],
    },
    {
      heading: { ko: "Drain 공핍과 최대 전계", en: "Drain depletion and peak field" },
      body: {
        ko: String.raw`계산값 ($w_d$, $E_{\mathrm{pk}}$):

- $r = 0$: 76.30 nm, $2.709\times10^{5}$ V/cm
- $r = 1$ V: 107.0 nm, $3.800\times10^{5}$ V/cm
- $r = 2$ V: 130.7 nm, $4.641\times10^{5}$ V/cm
- $r = 3$ V: 150.7 nm, $5.352\times10^{5}$ V/cm
- $r = 4$ V: 168.4 nm, $5.979\times10^{5}$ V/cm
- $r = 5$ V (표 끝): $6.546\times10^{5}$ V/cm`,
        en: String.raw`Computed values ($w_d$, $E_{\mathrm{pk}}$):

- $r = 0$: 76.30 nm, $2.709\times10^{5}$ V/cm
- $r = 1$ V: 107.0 nm, $3.800\times10^{5}$ V/cm
- $r = 2$ V: 130.7 nm, $4.641\times10^{5}$ V/cm
- $r = 3$ V: 150.7 nm, $5.352\times10^{5}$ V/cm
- $r = 4$ V: 168.4 nm, $5.979\times10^{5}$ V/cm
- $r = 5$ V (end of table): $6.546\times10^{5}$ V/cm`,
      },
      equations: [
        {
          id: "eq-es-wd",
          label: { ko: "drain 공핍폭", en: "Drain depletion width" },
          tex: String.raw`w_d = \sqrt{\frac{2\varepsilon\,(V_{\mathrm{bi}} + r)}{q\,N_A}}\quad [\mathrm{cm}]`,
          code: "photo_mean.py · components() (wd); mean_model.py · Field (width)",
        },
        {
          id: "eq-es-epk",
          label: { ko: "최대 전계 (삼각 분포)", en: "Peak field (triangular profile)" },
          tex: String.raw`E_{\mathrm{pk}} = \frac{2\,(V_{\mathrm{bi}} + r)}{w_d} = \sqrt{\frac{2\,q\,N_A\,(V_{\mathrm{bi}} + r)}{\varepsilon}}\quad [\mathrm{V/cm}]`,
          note: {
            ko: String.raw`한쪽 abrupt n⁺/p 접합($N_D \gg N_A$): 공핍은 모두 body 쪽. branch 배열의 19번 열이 $E_{\mathrm{pk}}$다.`,
            en: String.raw`One-sided abrupt n⁺/p junction ($N_D \gg N_A$): all depletion is on the body side. Column 19 of the branch array stores $E_{\mathrm{pk}}$.`,
          },
          code: "mean_model.py · Field (peak); photo_mean.py · curve_grid() col 19",
        },
      ],
    },
    {
      heading: { ko: "중성 base 길이 L_n", en: "Neutral base length L_n" },
      body: {
        ko: "계산값: $L_n(u=0, r=0) = 347.4$ nm ($= L_{\\mathrm{ref}}$, `bjt-transport`), HRS fold 295.6 nm, LRS fold 333.8 nm ($V_G = -2$ V). $L_n$은 수송의 규모 $I_0 = qAD_nN_A/L_n$와 $\\kappa = L_n^2/(D_n\\tau)$, 전하 항 $qAL_n\\bar n$, $qN_AAL_n$에 들어간다.",
        en: "Computed: $L_n(u=0, r=0) = 347.4$ nm ($= L_{\\mathrm{ref}}$, `bjt-transport`), 295.6 nm at the HRS fold, 333.8 nm at the LRS fold ($V_G = -2$ V). $L_n$ sets the transport scale $I_0 = qAD_nN_A/L_n$, $\\kappa = L_n^2/(D_n\\tau)$ and the charge terms $qAL_n\\bar n$, $qN_AAL_n$.",
      },
      equations: [
        {
          id: "eq-es-ln",
          label: { ko: "중성 길이와 유효 조건", en: "Neutral length and validity" },
          tex: String.raw`L_n = L - w_d - w_s\quad [\mathrm{cm}],\qquad L = 5\times10^{-5}\ \mathrm{cm},\qquad L_n \le 10^{-7}\ \mathrm{cm} \Rightarrow \mathrm{NaN}`,
          code: "photo_mean.py · components() (length)",
        },
        {
          id: "eq-es-rgeo",
          label: { ko: "기하학적 r 상한 (curve_grid)", en: "Geometric upper bound on r (curve_grid)" },
          tex: String.raw`r_{\mathrm{geo}}(u) = \frac{q\,N_A\,\big(L - w_s - 1.01\times10^{-7}\big)^2}{2\varepsilon} - V_{\mathrm{bi}}`,
          note: {
            ko: String.raw`$r \le r_{\mathrm{geo}}$이면 $L_n \ge 1.01$ nm가 보장되어 NaN 가드(1 nm)에 걸리지 않는다. 이분법 상한은 $\min(5\ \mathrm{V}, r_{\mathrm{geo}})$.`,
            en: String.raw`$r \le r_{\mathrm{geo}}$ guarantees $L_n \ge 1.01$ nm, i.e. above the 1 nm NaN guard. The bisection upper bound is $\min(5\ \mathrm{V}, r_{\mathrm{geo}})$.`,
          },
          code: "photo_mean.py · curve_grid() (lavail, rgeo)",
        },
      ],
    },
    {
      heading: { ko: "게이트 용량", en: "Gate capacitance" },
      equations: [
        {
          id: "eq-es-cox",
          label: { ko: "게이트 산화막 용량 (SI)", en: "Gate-oxide capacitance (SI)" },
          tex: String.raw`C_{\mathrm{ox}} = \frac{3.9\,\varepsilon_0}{\mathrm{EOT}}\,W L = 2.449\times10^{-16}\ \mathrm{F},\qquad \frac{C_{\mathrm{ox}}}{q} = 1529\ \mathrm{holes/V}`,
          note: {
            ko: String.raw`SI 단위($\varepsilon_0$ F/m, EOT·W·L m). 면적당 $2.449\times10^{-3}$ F/m² $= 2.449\times10^{-7}$ F/cm². 전하 항에만 쓰인다(z[13]의 $C_{\mathrm{ox}}u$, 격자의 $C_{\mathrm{ox}}\psi$); $w_s$, $w_d$에는 게이트 결합이 없다.`,
            en: String.raw`SI units ($\varepsilon_0$ in F/m, EOT, W, L in m). Per area $2.449\times10^{-3}$ F/m² $= 2.449\times10^{-7}$ F/cm². Used only in charge terms ($C_{\mathrm{ox}}u$ in z[13], $C_{\mathrm{ox}}\psi$ in the lattice); $w_s$ and $w_d$ have no gate coupling.`,
          },
          code: "mean_model.py · COX_F",
        },
      ],
      notes: [
        {
          ko: "abrupt 한쪽 접합, $N_D = 10^{20}$ cm⁻³는 가정값(추출 아님). 두 접합 모두 같은 $N_A$를 쓴다.",
          en: "Abrupt one-sided junctions; $N_D = 10^{20}$ cm⁻³ is assumed (not extracted). Both junctions use the same $N_A$.",
        },
        {
          ko: "코드 docstring: source/drain 공핍폭은 abrupt-junction 저주입 정전기를 유지한다 — 공핍층 내 이동 전하, 게이트/2D 결합, Poisson 풀이가 없다.",
          en: "Code docstring: source/drain depletion widths retain abrupt-junction low-injection electrostatics — no mobile charge in the depletion regions, no gate/2D coupling, no Poisson solve.",
        },
        {
          ko: "단위: 기하 상수는 m(`LENGTH_M` …), 정전기·수송은 cm(×100), $\\varepsilon$은 F/cm, 전계는 V/cm, $C_{\\mathrm{ox}}$만 SI로 계산된다.",
          en: "Units: geometry constants in m (`LENGTH_M` …), electrostatics and transport in cm (×100), $\\varepsilon$ in F/cm, fields in V/cm; only $C_{\\mathrm{ox}}$ is computed in SI.",
        },
      ],
    },
  ],
  related: ["overview", "impact-ionization", "bjt-transport", "charge-balance"],
  codeRefs: ["photo_extension/photo_mean.py", "model/janus_calibration_20260920/idvd_model/mean_model.py"],
};

export default topic;
