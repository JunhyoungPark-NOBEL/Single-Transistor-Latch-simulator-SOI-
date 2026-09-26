// Physics topic "btbt-gidl" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "btbt-gidl",
  title: { ko: "밴드 간 터널링(BTBT)과 GIDL", en: "Band-to-band tunneling and GIDL" },
  summary: {
    ko: "Kane형 국소 BTBT 생성률을 드레인 공핍층 전체(접합 BTBT)와 게이트–드레인 가장자리의 고정 체적(GIDL)에 적용한다. 두 항 모두 평형($r=0$)에서 0이 되는 상세 균형(detailed balance) 인자 $(1-e^{-r/V_T})$를 곱한다.",
    en: "A Kane-type local BTBT generation rate is applied to the whole drain depletion region (junction BTBT) and to a fixed volume at the gate–drain edge (GIDL). Both terms are multiplied by the detailed-balance factor $(1-e^{-r/V_T})$, which vanishes at equilibrium ($r=0$).",
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
            ko: String.raw`$E$는 V/cm, $G$는 cm⁻³ s⁻¹ 단위이므로 $A_{\mathrm{BB}}$의 단위는 cm⁻³ s⁻¹ (V/cm)⁻²·⁵이다. 두 BTBT 항은 같은 상수를 쓴다.`,
            en: String.raw`$E$ in V/cm and $G$ in cm⁻³ s⁻¹, so $A_{\mathrm{BB}}$ is in cm⁻³ s⁻¹ (V/cm)⁻²·⁵. Both BTBT terms use the same constants.`,
          },
          code: "mean_model.py · BB_A, BB_B",
        },
      ],
    },
    {
      heading: { ko: "접합 BTBT", en: "Junction BTBT" },
      body: {
        ko: String.raw`표 값: $2.166\times10^{-29}$ A (1 V), $4.515\times10^{-25}$ A (2 V), $1.938\times10^{-22}$ A (3 V), $1.302\times10^{-20}$ A (4 V). 기준 보정에서는 GIDL에 비해 무시할 만큼 작다(HRS fold에서 $3.7\times10^{-22}$ A 대 2.285 pA).`,
        en: String.raw`Table values: $2.166\times10^{-29}$ A (1 V), $4.515\times10^{-25}$ A (2 V), $1.938\times10^{-22}$ A (3 V), $1.302\times10^{-20}$ A (4 V). In the reference calibration it is negligible next to GIDL ($3.7\times10^{-22}$ A vs 2.285 pA at the HRS fold).`,
      },
      equations: [
        {
          id: "eq-bt-junction",
          label: { ko: "공핍층 적분 (표)", en: "Depletion-region integral (table)" },
          tex: String.raw`I_{\mathrm{BTBT}}^{\mathrm{tab}}(r) = q\,A\,W\!\int_0^1 G\big(E_{\mathrm{pk}}\,z\big)\,dz,\qquad W = w_d(r),\quad A = W_{\mathrm{ch}}T_{\mathrm{Si}} = 10^{-10}\ \mathrm{cm^2}`,
          note: {
            ko: "`impact-ionization`과 같은 삼각형 전계를 501점 사다리꼴 적분으로 계산하고, 1501점 $r$ 표에서 선형 보간한다. $W_{\\mathrm{ch}}$는 채널 폭 200 nm이다.",
            en: "Same triangular field as in `impact-ionization`, 501-point trapezoid, linear interpolation on the 1501-point $r$ table. $W_{\\mathrm{ch}}$ is the 200 nm channel width.",
          },
          code: "mean_model.py · Field.__init__ (generation, bb)",
        },
        {
          id: "eq-bt-junction-net",
          label: { ko: "모델에 쓰는 순 접합 BTBT", en: "Net junction BTBT used by the model" },
          tex: String.raw`I_{\mathrm{BTBT}} = I_{\mathrm{BTBT}}^{\mathrm{tab}}(r + p_{19})\,\big(1 - e^{-r/V_T}\big)`,
          note: { ko: "평형 인자에는 $r + p_{19}$가 아니라 $r$을 쓴다.", en: "The balance factor uses $r$, not $r + p_{19}$." },
          code: "photo_mean.py · components() (bbj)",
        },
      ],
    },
    {
      heading: { ko: "상세 균형 인자", en: "Detailed-balance factor" },
      equations: [
        {
          id: "eq-bt-balance",
          label: { ko: "순 역방향 BTBT 가정", en: "Net reverse-BTBT ansatz" },
          tex: String.raw`b_{\mathrm{db}}(r) = 1 - e^{-r/V_T}`,
          note: {
            ko: String.raw`코드 주석: 순 역방향 BTBT에 대한 상세 균형 가정으로, 평형($r = 0$)에서 0이 된다. 접합 BTBT와 GIDL에 모두 곱하며, $r \gg V_T$에서는 1이다.`,
            en: String.raw`Code comment: detailed-balance ansatz for net reverse BTBT, zero at equilibrium ($r = 0$). It multiplies both junction BTBT and GIDL and equals 1 for $r \gg V_T$.`,
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
          label: { ko: "드레인 가장자리의 터널링 전계", en: "Drain-edge tunneling field" },
          tex: String.raw`E_G = \max\!\left(\frac{u + r - V_G - 0.3 - 1.12 + \varphi_{\mathrm{GIDL}}}{l_{\mathrm{GIDL}}},\ 0\right),\qquad l_{\mathrm{GIDL}} = p_4\,[\mathrm{nm}]\times10^{-7}\ \mathrm{cm}`,
          note: {
            ko: String.raw`$u + r$는 내부 드레인–소스 전압(정공 강하와 직렬 저항 제외)이므로 $u + r - V_G$는 내부 드레인–게이트 전압이다. 1.12 V는 Si 밴드갭 $E_g/q$, 0.3 V는 코드에 고정된 오프셋, $\varphi_{\mathrm{GIDL}}$은 p[9]이다. HRS fold($V_G = -2$ V, $u + r = 3.7037$ V)에서 $E_G = 1.490\times10^{6}$ V/cm이다.`,
            en: String.raw`$u + r$ is the internal drain–source voltage (without the hole drop and the series resistance), so $u + r - V_G$ is the internal drain–gate voltage. 1.12 V is the Si band gap $E_g/q$, 0.3 V a hard-coded offset and $\varphi_{\mathrm{GIDL}}$ = p[9]. At the HRS fold ($V_G = -2$ V, $u + r = 3.7037$ V), $E_G = 1.490\times10^{6}$ V/cm.`,
          },
          code: "photo_mean.py · components() (eg)",
        },
      ],
    },
    {
      heading: { ko: "GIDL 전류", en: "GIDL current" },
      body: {
        ko: String.raw`계산값($V_G = -2$ V): HRS fold에서 $I_{\mathrm{GIDL}} = 2.285$ pA(같은 점의 $I_{\mathrm{II}} = 3.388$ pA)이다. LRS fold에서는 $r$이 작아 $E_G = 1.103\times10^{6}$ V/cm, $I_{\mathrm{GIDL}} = 12.26$ fA이다.`,
        en: String.raw`Computed ($V_G = -2$ V): $I_{\mathrm{GIDL}} = 2.285$ pA at the HRS fold (vs $I_{\mathrm{II}} = 3.388$ pA at the same point); at the LRS fold $r$ is smaller, so $E_G = 1.103\times10^{6}$ V/cm and $I_{\mathrm{GIDL}} = 12.26$ fA.`,
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
            ko: "$d = 4.549$ nm ($< T_{\\mathrm{Si}} = 50$ nm)이므로 $\\mathcal{V}_G = 4.549\\times10^{-18}$ cm³이다. 7×10¹⁹ cm⁻³는 드레인 쪽 고농도 터널링 영역에 대한 가정값(바디의 $N_A$가 아님)이고, 5 nm는 활성 길이(`gidl_active_length_nm`)다.",
            en: "$d = 4.549$ nm ($< T_{\\mathrm{Si}} = 50$ nm), so $\\mathcal{V}_G = 4.549\\times10^{-18}$ cm³. 7×10¹⁹ cm⁻³ is an assumed doping of the heavily doped drain-side tunneling region (not the body $N_A$); 5 nm is the active length (`gidl_active_length_nm`).",
          },
          code: "photo_mean.py · components() (depth, volume); mean_model.py · Model",
        },
      ],
    },
    {
      heading: { ko: "φ_GIDL: 드레인 가장자리 국소 상태", en: "φ_GIDL: the drain-edge local state" },
      body: {
        ko: String.raw`$\varphi_{\mathrm{GIDL}}$은 드레인 가장자리의 터널링 전계에 추가 국소 전압 강하가 걸린다는 가설이다(출력 전압의 지터가 아니며, 2차원 Poisson 해나 특정된 결함 위치도 아니다). $\delta\varphi_G > 0$이면 $E_G$와 GIDL 정공 공급이 늘어 소자가 더 일찍 래치된다. $V_G = -2$ V에서 $dV_{\mathrm{LU}}/d\varphi_G = -0.795$ V/V이다(±10 mV 중앙 차분 계산값, MODEL_SPEC ≈ −0.80).`,
        en: String.raw`$\varphi_{\mathrm{GIDL}}$ is the hypothesis of an additional local voltage drop across the drain-edge tunneling field (not output-voltage jitter, and neither a 2D Poisson solution nor an identified defect position). $\delta\varphi_G > 0$ raises $E_G$ and the GIDL hole supply, so the device latches earlier: at $V_G = -2$ V, $dV_{\mathrm{LU}}/d\varphi_G = -0.795$ V/V (computed by a ±10 mV central difference; MODEL_SPEC ≈ −0.80).`,
      },
      equations: [
        {
          id: "eq-bt-phi",
          label: { ko: "p[9]의 구성", en: "Composition of p[9]" },
          tex: String.raw`p_9 = \varphi_{\mathrm{GIDL},0} + \delta\varphi_{G0} + \delta\varphi_G`,
          note: {
            ko: String.raw`$\varphi_{\mathrm{GIDL},0} = 0.2721$ mV는 보정된 평균, $\delta\varphi_{G0}$는 소자별 중심(광조사 보정 +0.0744 V), $\delta\varphi_G$는 사이클마다의 요동이다(기준 보정 $\sigma_{\varphi G} = 0.1534$ V, 광조사 보정 0.2154 V).`,
            en: String.raw`$\varphi_{\mathrm{GIDL},0} = 0.2721$ mV is the calibrated mean, $\delta\varphi_{G0}$ the device center (illumination calibration: +0.0744 V) and $\delta\varphi_G$ the cycle-to-cycle deviation (reference calibration $\sigma_{\varphi G} = 0.1534$ V, illumination calibration 0.2154 V).`,
          },
          code: "server/params.py · build_p()",
        },
      ],
      notes: [
        {
          ko: "$l_{\\mathrm{GIDL}}$은 유효 전계 길이이며, 설계 지도의 핫스팟 길이와는 다르다(`MODEL_PARAMETERS.json` scope).",
          en: "$l_{\\mathrm{GIDL}}$ is an effective field length, not the hotspot length of the design map (`MODEL_PARAMETERS.json` scope).",
        },
        {
          ko: "국소(Kane형) 전계 모델이다. 비국소 터널링 경로 적분과 트랩 보조 터널링(trap-assisted tunneling)은 포함하지 않는다.",
          en: "Local (Kane-type) field model: no non-local tunneling-path integral and no trap-assisted tunneling.",
        },
        {
          ko: "확률 모델에서 GIDL과 BTBT 정공은 Poisson 단위 사건이다(`stochastic-events`).",
          en: "In the stochastic model GIDL and BTBT holes are Poisson unit events (`stochastic-events`).",
        },
      ],
    },
    {
      heading: { ko: "Simple Model(논문)의 BTBT와 GIDL", en: "BTBT and GIDL in the Simple Model (paper)" },
      body: {
        ko: String.raw`Simple Model은 논문 Table I의 $I_{\mathrm{BTBT}}=q\int A E^{2.5}\exp(-B/E)\,dv$ ($A=4\times10^{14}$, $B=19$ MV/cm)를 논문 참고 스크립트처럼 "한 전계에서의 Kane 생성률 × 체적"으로 평가한다. 접합 BTBT는 급격 드레인 접합의 최대 전계 $E_{\mathrm j}=2(V_{\mathrm{bi}}+r)/W_{\mathrm d}$와 체적 $W T_{\mathrm{Si}} W_{\mathrm d}$를 쓰고, GIDL은 수직 전계 $E_{\mathrm g}=(r-V_{\mathrm G}+1.2-1.12)/(3\,\mathrm{EOT})$와 입력 체적 $V_{\mathrm{GIDL}}\,(W/W_{\mathrm{ref}})$ (기준 폭 $W_{\mathrm{ref}}=200$ nm에서 기본 $4.55\times10^{5}\,\mathrm{nm^3}$, 스크립트의 유효 체적 $W\cdot5\,\mathrm{nm}\cdot W_{\mathrm t}\times100$)를 쓴다. $V_{\mathrm G}$가 평탄대 $V_{\mathrm{FB}}$ 아래로 내려가면 바디 바이어스는 더 변하지 않지만 $E_{\mathrm g}$는 계속 커지므로 GIDL이 정공 생성을 지배하게 되고 $V_{\mathrm{LU}}$가 다시 내려간다(논문 Fig. 5(a), (b)). 위의 국소 전계 적분과 전계 테이블은 Detailed Model의 정의이다.`,
        en: String.raw`The Simple Model evaluates the paper's Table I integral $I_{\mathrm{BTBT}}=q\int A E^{2.5}\exp(-B/E)\,dv$ ($A=4\times10^{14}$, $B=19$ MV/cm) as "Kane rate at one field × volume", as the paper's reference script does. Junction BTBT uses the peak field of the abrupt drain junction, $E_{\mathrm j}=2(V_{\mathrm{bi}}+r)/W_{\mathrm d}$, over the volume $W T_{\mathrm{Si}} W_{\mathrm d}$; GIDL uses the vertical field $E_{\mathrm g}=(r-V_{\mathrm G}+1.2-1.12)/(3\,\mathrm{EOT})$ over the input volume $V_{\mathrm{GIDL}}\,(W/W_{\mathrm{ref}})$ (default $4.55\times10^{5}\,\mathrm{nm^3}$ at $W_{\mathrm{ref}}=200$ nm, the script's effective volume $W\cdot5\,\mathrm{nm}\cdot W_{\mathrm t}\times100$). Once $V_{\mathrm G}$ drops below the flat band $V_{\mathrm{FB}}$ the body bias no longer changes while $E_{\mathrm g}$ keeps growing, so GIDL takes over hole generation and $V_{\mathrm{LU}}$ comes back down (paper Fig. 5(a), (b)). The local field integral and field tables above belong to the Detailed Model.`,
      },
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
