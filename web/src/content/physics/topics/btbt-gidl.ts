// Physics topic "btbt-gidl" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "btbt-gidl",
  title: { ko: "BTBT와 GIDL", en: "Band-to-band tunnelling and GIDL" },
  summary: {
    ko: "Kane형 국소 BTBT 생성률을 drain 공핍층 전체(접합 BTBT)와 게이트–drain 가장자리의 고정 체적(GIDL)에 적용하고, 평형($r=0$)에서 0이 되는 detailed-balance 인자 $(1-e^{-r/V_T})$를 곱한다.",
    en: "A Kane-type local BTBT generation rate is applied to the whole drain depletion region (junction BTBT) and to a fixed volume at the gate–drain edge (GIDL), multiplied by the detailed-balance factor $(1-e^{-r/V_T})$ that vanishes at equilibrium ($r=0$).",
  },
  tags: ["Eq. 1", "I_BTBT", "I_GIDL", "local state φ_G"],
  sections: [
    {
      heading: { ko: "생성률", en: "Generation rate" },
      equations: [
        {
          id: "eq-bt-rate",
          label: { ko: "국소 BTBT 생성률", en: "Local BTBT generation rate" },
          tex: String.raw`G(E) = A_{\mathrm{BB}}\,E^{2.5}\exp\!\left(-\frac{B_{\mathrm{BB}}}{\max(E,\,1)}\right),\qquad A_{\mathrm{BB}} = 4\times10^{14},\quad B_{\mathrm{BB}} = 19\times10^{6}\ \mathrm{V/cm}`,
          note: {
            ko: String.raw`$E$ [V/cm], $G$ [cm⁻³ s⁻¹] → $A_{\mathrm{BB}}$ [cm⁻³ s⁻¹ (V/cm)⁻²·⁵]. 두 BTBT 항이 같은 상수를 쓴다.`,
            en: String.raw`$E$ in V/cm, $G$ in cm⁻³ s⁻¹ → $A_{\mathrm{BB}}$ in cm⁻³ s⁻¹ (V/cm)⁻²·⁵. Both BTBT terms use the same constants.`,
          },
          code: "mean_model.py · BB_A, BB_B",
        },
      ],
    },
    {
      heading: { ko: "접합 BTBT", en: "Junction BTBT" },
      body: {
        ko: String.raw`표 값: $2.166\times10^{-29}$ A (1 V), $4.515\times10^{-25}$ A (2 V), $1.938\times10^{-22}$ A (3 V), $1.302\times10^{-20}$ A (4 V). 논문 소자에서는 GIDL에 비해 무시할 수준이다(HRS fold에서 $3.7\times10^{-22}$ A 대 2.285 pA).`,
        en: String.raw`Table values: $2.166\times10^{-29}$ A (1 V), $4.515\times10^{-25}$ A (2 V), $1.938\times10^{-22}$ A (3 V), $1.302\times10^{-20}$ A (4 V). In the paper device it is negligible next to GIDL ($3.7\times10^{-22}$ A vs 2.285 pA at the HRS fold).`,
      },
      equations: [
        {
          id: "eq-bt-junction",
          label: { ko: "공핍층 적분 (표)", en: "Depletion-region integral (table)" },
          tex: String.raw`I_{\mathrm{BTBT}}^{\mathrm{tab}}(r) = q\,A\,W\!\int_0^1 G\big(E_{\mathrm{pk}}\,z\big)\,dz,\qquad W = w_d(r),\quad A = W_{\mathrm{ch}}T_{\mathrm{Si}} = 10^{-10}\ \mathrm{cm^2}`,
          note: {
            ko: "`impact-ionization`과 같은 삼각 전계, 501점 사다리꼴 적분, 1501점 $r$ 표에서 선형 보간. $W_{\\mathrm{ch}}$는 채널 폭 200 nm.",
            en: "Same triangular field as `impact-ionization`, 501-point trapezoid, linear interpolation on the 1501-point $r$ table. $W_{\\mathrm{ch}}$ is the 200 nm channel width.",
          },
          code: "mean_model.py · Field.__init__ (generation, bb)",
        },
        {
          id: "eq-bt-junction-net",
          label: { ko: "모델에 쓰는 순 접합 BTBT", en: "Net junction BTBT used by the model" },
          tex: String.raw`I_{\mathrm{BTBT}} = I_{\mathrm{BTBT}}^{\mathrm{tab}}(r + p_{19})\,\big(1 - e^{-r/V_T}\big)`,
          note: { ko: "평형 인자는 $r + p_{19}$가 아니라 $r$을 쓴다.", en: "The balance factor uses $r$, not $r + p_{19}$." },
          code: "photo_mean.py · components() (bbj)",
        },
      ],
    },
    {
      heading: { ko: "Detailed-balance 인자", en: "Detailed-balance factor" },
      equations: [
        {
          id: "eq-bt-balance",
          label: { ko: "순 역방향 BTBT 가정", en: "Net reverse-BTBT ansatz" },
          tex: String.raw`b_{\mathrm{db}}(r) = 1 - e^{-r/V_T}`,
          note: {
            ko: String.raw`코드 주석: 순 역방향 BTBT detailed-balance 가정, 평형 $r = 0$에서 0. 접합 BTBT와 GIDL 모두에 곱하며 $r \gg V_T$에서 1이다.`,
            en: String.raw`Code comment: net reverse BTBT detailed-balance ansatz, zero at equilibrium $r = 0$. Multiplies both junction BTBT and GIDL; equals 1 for $r \gg V_T$.`,
          },
          code: "photo_mean.py · components() (balance = -expm1(-r/VT))",
        },
      ],
    },
    {
      heading: { ko: "GIDL 전계", en: "GIDL field" },
      equations: [
        {
          id: "eq-bt-eg",
          label: { ko: "drain 가장자리 터널링 전계", en: "Drain-edge tunnelling field" },
          tex: String.raw`E_G = \max\!\left(\frac{u + r - V_G - 0.3 - 1.12 + \varphi_{\mathrm{GIDL}}}{l_{\mathrm{GIDL}}},\ 0\right),\qquad l_{\mathrm{GIDL}} = p_4\,[\mathrm{nm}]\times10^{-7}\ \mathrm{cm}`,
          note: {
            ko: String.raw`$u + r$는 내부 drain–source 전압(정공 강하·직렬저항 제외)이므로 $u + r - V_G$는 내부 drain–gate 전압이다. 1.12 V = Si 밴드갭 $E_g/q$, 0.3 V는 코드에 고정된 오프셋, $\varphi_{\mathrm{GIDL}}$ = p[9]. HRS fold($V_G = -2$ V, $u + r = 3.7037$ V)에서 $E_G = 1.490\times10^{6}$ V/cm.`,
            en: String.raw`$u + r$ is the internal drain–source voltage (without hole drop and series resistance), so $u + r - V_G$ is the internal drain–gate voltage. 1.12 V = Si band gap $E_g/q$; 0.3 V is a hard-coded offset; $\varphi_{\mathrm{GIDL}}$ = p[9]. At the HRS fold ($V_G = -2$ V, $u + r = 3.7037$ V), $E_G = 1.490\times10^{6}$ V/cm.`,
          },
          code: "photo_mean.py · components() (eg)",
        },
      ],
    },
    {
      heading: { ko: "GIDL 전류", en: "GIDL current" },
      body: {
        ko: String.raw`계산값($V_G = -2$ V): HRS fold에서 $I_{\mathrm{GIDL}} = 2.285$ pA(같은 점의 $I_{\mathrm{II}} = 3.388$ pA), LRS fold에서는 $r$이 작아 $E_G = 1.103\times10^{6}$ V/cm, $I_{\mathrm{GIDL}} = 12.26$ fA.`,
        en: String.raw`Computed ($V_G = -2$ V): $I_{\mathrm{GIDL}} = 2.285$ pA at the HRS fold (vs $I_{\mathrm{II}} = 3.388$ pA there); at the LRS fold $r$ is smaller, $E_G = 1.103\times10^{6}$ V/cm and $I_{\mathrm{GIDL}} = 12.26$ fA.`,
      },
      equations: [
        {
          id: "eq-bt-gidl",
          label: { ko: "고정 체적 GIDL", en: "Fixed-volume GIDL" },
          tex: String.raw`I_{\mathrm{GIDL}} = q\,\mathcal{V}_G\,A_{\mathrm{BB}}\,E_G^{2.5}\exp\!\left(-\frac{B_{\mathrm{BB}}}{\max(E_G,\,1)}\right)\big(1 - e^{-r/V_T}\big)`,
          code: "photo_mean.py · components() (gidl)",
        },
        {
          id: "eq-bt-vol",
          label: { ko: "GIDL 체적 [cm³]", en: "GIDL volume [cm³]" },
          tex: String.raw`\mathcal{V}_G = W_{\mathrm{ch}}\cdot\big(5\times10^{-7}\ \mathrm{cm}\big)\cdot d,\qquad d = \min\!\left(\sqrt{\frac{2\varepsilon\cdot 1.12}{q\cdot 7\times10^{19}}},\ T_{\mathrm{Si}}\right)`,
          note: {
            ko: "$d = 4.549$ nm ($< T_{\\mathrm{Si}} = 50$ nm) → $\\mathcal{V}_G = 4.549\\times10^{-18}$ cm³. 7×10¹⁹ cm⁻³는 drain 쪽 고농도 터널링 영역의 가정값(body $N_A$가 아님), 5 nm는 활성 길이(`gidl_active_length_nm`).",
            en: "$d = 4.549$ nm ($< T_{\\mathrm{Si}} = 50$ nm) → $\\mathcal{V}_G = 4.549\\times10^{-18}$ cm³. 7×10¹⁹ cm⁻³ is an assumed highly-doped drain-side tunnelling region (not body $N_A$); 5 nm is the active length (`gidl_active_length_nm`).",
          },
          code: "photo_mean.py · components() (depth, volume); mean_model.py · Model",
        },
      ],
    },
    {
      heading: { ko: "φ_GIDL: drain-edge local state", en: "φ_GIDL: the drain-edge local state" },
      body: {
        ko: String.raw`$\varphi_{\mathrm{GIDL}}$는 drain 가장자리 터널링 전계에 걸리는 추가 국소 전압 강하라는 가설이다(출력 전압 지터가 아니고, 2D Poisson 해나 결함 위치도 아님). $\delta\varphi_G > 0$이면 $E_G$와 GIDL 정공이 늘어 더 일찍 래치된다: $V_G = -2$ V에서 $dV_{\mathrm{LU}}/d\varphi_G = -0.795$ V/V(±10 mV 중앙차분, 계산값; MODEL_SPEC ≈ −0.80).`,
        en: String.raw`$\varphi_{\mathrm{GIDL}}$ is the hypothesis of an additional local voltage drop across the drain-edge tunnelling field (not output-voltage jitter, not a 2D Poisson solution or an identified defect position). $\delta\varphi_G > 0$ raises $E_G$ and the GIDL hole supply, so the device latches earlier: at $V_G = -2$ V, $dV_{\mathrm{LU}}/d\varphi_G = -0.795$ V/V (±10 mV central difference, computed; MODEL_SPEC ≈ −0.80).`,
      },
      equations: [
        {
          id: "eq-bt-phi",
          label: { ko: "p[9]의 구성", en: "Composition of p[9]" },
          tex: String.raw`p_9 = \varphi_{\mathrm{GIDL},0} + \delta\varphi_{G0} + \delta\varphi_G`,
          note: {
            ko: String.raw`$\varphi_{\mathrm{GIDL},0} = 0.2721$ mV(보정 평균), $\delta\varphi_{G0}$: 소자별 중심(photo 소자 +0.0744 V), $\delta\varphi_G$: cycle마다의 요동(논문 $\sigma_{\varphi G} = 0.1534$ V, photo 0.2154 V).`,
            en: String.raw`$\varphi_{\mathrm{GIDL},0} = 0.2721$ mV (calibrated mean), $\delta\varphi_{G0}$: device centre (photo device +0.0744 V), $\delta\varphi_G$: cycle-to-cycle deviation (paper $\sigma_{\varphi G} = 0.1534$ V, photo 0.2154 V).`,
          },
          code: "server/params.py · build_p()",
        },
      ],
      notes: [
        {
          ko: "$l_{\\mathrm{GIDL}}$은 유효 전계 길이이며 설계 지도의 hotspot 길이가 아니다(`MODEL_PARAMETERS.json` scope).",
          en: "$l_{\\mathrm{GIDL}}$ is an effective field length, not the hotspot length of the design map (`MODEL_PARAMETERS.json` scope).",
        },
        {
          ko: "국소(Kane형) 전계 모델: 비국소 터널링 경로 적분, trap-assisted tunnelling 없음.",
          en: "Local (Kane-type) field model: no non-local tunnelling path integral, no trap-assisted tunnelling.",
        },
        {
          ko: "확률 모델에서 GIDL·BTBT 정공은 Poisson unit 사건이다(`stochastic-events`).",
          en: "In the stochastic model GIDL and BTBT holes are Poisson unit events (`stochastic-events`).",
        },
      ],
    },
  ],
  related: ["impact-ionization", "electrostatics", "charge-balance", "local-states", "parameters"],
  codeRefs: [
    "model/janus_calibration_20260920/idvd_model/mean_model.py",
    "photo_extension/photo_mean.py",
    "server/params.py",
  ],
};

export default topic;
