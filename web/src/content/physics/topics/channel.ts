// Physics topic "channel" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "channel",
  title: { ko: "채널 전류", en: "Channel current" },
  summary: {
    ko: String.raw`저 $V_D$(0.05 V) $I_D$–$V_G$ 곡선에 맞춰 고정한 EKV형 채널 식을 내부 전압 $u + r$에 적용한다. 광조사 확장은 DIBL $\eta$, 바디 결합 $\gamma$, 기울기 감쇠 $\kappa$, 고 $V_D$ 시드 항을 선택적으로 더한다. 채널 전자는 $I_D$와 II에만 기여하며 BJT 수송에는 들어가지 않는다.`,
    en: String.raw`An EKV-type channel expression, frozen from a fit to the low-$V_D$ (0.05 V) $I_D$–$V_G$ curve, is applied at the internal voltage $u + r$. The illumination extensions optionally add DIBL $\eta$, body coupling $\gamma$, slope degradation $\kappa$ and a high-$V_D$ seed term. Channel electrons contribute to $I_D$ and II only, not to the BJT transport.`,
  },
  tags: ["I_ch", "frozen fit", "open problem −1.1 V"],
  sections: [
    {
      heading: { ko: "고정된 저 V_D 채널 fit", en: "Frozen low-V_D channel fit" },
      body: {
        ko: "`fit_results.json`에서 선택된 후보(`mobility_degradation`)이다. $V_D = 0.05$ V 한 조건의 $I_D$–$V_G$ 곡선에 대해 로그 잔차(0.05 dec)와 강전류 잔차를 같은 가중치로 최소화했다. 5–10 pA 미만의 전류는 중도절단(censored)으로 제외했고, 누설 바닥값은 맞추지 않았으며 $R_{sd} = 0$이다.",
        en: "Selected candidate in `fit_results.json` (`mobility_degradation`): a single $I_D$–$V_G$ curve at $V_D = 0.05$ V, fitted with equal weights on the log residual (0.05 dec) and the strong-current residual. Currents below 5–10 pA were censored, no leakage floor was fitted, and $R_{sd} = 0$.",
      },
      equations: [
        {
          id: "eq-ch-beta0",
          label: { ko: "β₀ 확인", en: "β₀ check" },
          tex: String.raw`\beta_0 = \mu_0\,C_{\mathrm{ox}}'\,\frac{W}{L} = 767.79\ \mathrm{cm^2/(V\,s)}\times 2.449\times10^{-3}\ \mathrm{F/m^2}\times 0.4 = 7.521\times10^{-5}\ \mathrm{A/V^2}`,
          code: "fit_results.json · selected_candidate.params",
        },
      ],
      variables: [
        {
          symbol: "V_{T0}",
          name: { ko: "문턱 전압", en: "Threshold voltage" },
          value: "−0.4903252",
          unit: "V",
          code: "Vth0_V (hard-coded −.49032524444873615)",
        },
        {
          symbol: "n",
          name: { ko: "문턱 전압 이하 기울기 계수", en: "Subthreshold slope factor" },
          value: "1.778668",
          unit: "–",
          code: "n (1.7786684648788609)",
        },
        {
          symbol: String.raw`\beta_0`,
          name: { ko: "전달 계수 μ₀C_ox'W/L", en: "Transfer coefficient μ₀C_ox'W/L" },
          value: "7.521352 × 10⁻⁵",
          unit: "A/V²",
          code: "beta0_A_V2 (7.52135238967614e-5)",
        },
        {
          symbol: String.raw`\theta`,
          name: { ko: "이동도 감쇠", en: "Mobility degradation" },
          value: "0.6335606",
          unit: "1/V",
          code: "theta_per_V (.6335606399651017)",
        },
        {
          symbol: String.raw`\mu_0`,
          name: { ko: "저전계 이동도 (fit)", en: "Low-field mobility (fit)" },
          value: "767.79",
          unit: "cm²/(V·s)",
          code: "mu0_cm2_Vs",
        },
        {
          symbol: String.raw`C_{\mathrm{ox}}'`,
          name: { ko: "단위 면적당 산화막 용량", en: "Oxide capacitance per unit area" },
          value: "2.449 × 10⁻³",
          unit: "F/m²",
          code: "Cox_F_m2",
        },
      ],
    },
    {
      heading: { ko: "채널 식 (확장 = 0)", en: "Channel expression (extensions = 0)" },
      body: {
        ko: String.raw`계산값(포화 영역, $u = 0.3$ V, $r = 3$ V):

- $V_G = -2$ V: $9.857\times10^{-22}$ A
- $V_G = -1.8$ V: $7.634\times10^{-20}$ A
- $V_G = -1.1$ V: $3.116\times10^{-13}$ A
- $V_G = -0.5$ V: $7.243\times10^{-8}$ A`,
        en: String.raw`Computed values (saturation, $u = 0.3$ V, $r = 3$ V):

- $V_G = -2$ V: $9.857\times10^{-22}$ A
- $V_G = -1.8$ V: $7.634\times10^{-20}$ A
- $V_G = -1.1$ V: $3.116\times10^{-13}$ A
- $V_G = -0.5$ V: $7.243\times10^{-8}$ A`,
      },
      equations: [
        {
          id: "eq-ch-base",
          label: { ko: "EKV형 순방향/역방향 보간", en: "EKV-type forward/reverse interpolation" },
          tex: String.raw`\begin{aligned} V_{\mathrm{ov}} &= V_G - V_{T0},\qquad V_P = V_{\mathrm{ov}}/n\\ s_f &= \ln\!\big(1 + e^{V_P/(2V_T)}\big),\qquad s_r = \ln\!\big(1 + e^{(V_P - u - r)/(2V_T)}\big)\\ I_{\mathrm{ch}} &= \frac{2\,n\,\beta_0\,V_T^2\,(s_f - s_r)(s_f + s_r)}{1 + \theta\,n V_T \ln\!\big(1 + e^{V_{\mathrm{ov}}/(n V_T)}\big)} \end{aligned}`,
          note: {
            ko: String.raw`채널의 $V_{DS}$는 내부 전압 $u + r$이다(정공 강하, $R_c$, $R_{\mathrm{acc}}$ 제외). 분모는 평활화한 오버드라이브에 대한 이동도 감쇠다.`,
            en: String.raw`The channel $V_{DS}$ is the internal voltage $u + r$ (without the hole drop, $R_c$ and $R_{\mathrm{acc}}$). The denominator is the mobility degradation on the smoothed overdrive.`,
          },
          code: "photo_mean.py · components() (n, ov, pp, sf, sr, ch); mean_model.py · channel_current()",
        },
      ],
    },
    {
      heading: { ko: "광조사 확장 항 p[14]–p[18]", en: "Illumination-extension terms p[14]–p[18]" },
      body: {
        ko: "UI의 채널 시드 옵션(`params.CHANNEL_SEED_OPTIONS`)과 $V_G = -1.1$ V, $u = 0.55$ V, $r = 2.9$ V에서의 계산값:\n\n- none: $I_{\\mathrm{ch}} = 0.312$ pA\n- body_coupling ($\\gamma = 0.2794$, 광조사 보정 프리셋의 기본값): 8.76 pA\n- high_vd_seed ($I_p = 1.33$ pA, $S = 0.8$ V/dec): $0.312 + 9.97 = 10.29$ pA",
        en: "Channel-seed options of the UI (`params.CHANNEL_SEED_OPTIONS`), computed at $V_G = -1.1$ V, $u = 0.55$ V, $r = 2.9$ V:\n\n- none: $I_{\\mathrm{ch}} = 0.312$ pA\n- body_coupling ($\\gamma = 0.2794$, default of the illumination-calibration preset): 8.76 pA\n- high_vd_seed ($I_p = 1.33$ pA, $S = 0.8$ V/dec): $0.312 + 9.97 = 10.29$ pA",
      },
      equations: [
        {
          id: "eq-ch-ext",
          label: { ko: "확장된 채널", en: "Extended channel" },
          tex: String.raw`\begin{aligned} n' &= n\,\big[1 + p_{16}\,(u + r)\big]\\ V_{\mathrm{ov}} &= V_G - V_{T0} + p_{14}\,(u + r) + p_{15}\,u,\qquad V_P = V_{\mathrm{ov}}/n'\\ I_{\mathrm{ch}} &\leftarrow I_{\mathrm{ch}}(n', V_{\mathrm{ov}}) + p_{17}\,10^{(V_G + 1.8)/p_{18}}\qquad (p_{17} > 0) \end{aligned}`,
          note: {
            ko: String.raw`p[14]는 DIBL $\eta$ (V/V), p[15]는 바디 결합 $\gamma$ (V/V, $u$에 곱함), p[16]은 기울기 감쇠 $\kappa$ (1/V), p[17]은 $V_G = -1.8$ V에서의 시드 전류 $I_p$ (A), p[18]은 기울기 $S$ (V/dec)이다. $n'$은 앞 계수, $V_P$, $\theta$ 항에서 모두 $n$을 대신한다.`,
            en: String.raw`p[14]: DIBL $\eta$ (V/V); p[15]: body coupling $\gamma$ (V/V, multiplies $u$); p[16]: slope degradation $\kappa$ (1/V); p[17]: seed current $I_p$ at $V_G = -1.8$ V (A); p[18]: slope $S$ (V/dec). $n'$ replaces $n$ in the prefactor, in $V_P$ and in the $\theta$ term.`,
          },
          code: "photo_mean.py · components() (n, ov, ch += p[17]·10^((vg+1.8)/p[18]))",
        },
      ],
    },
    {
      heading: { ko: "모델 안에서의 역할", en: "Role in the model" },
      equations: [
        {
          id: "eq-ch-role",
          label: { ko: "채널 전류가 들어가는 곳", en: "Where the channel current enters" },
          tex: String.raw`\begin{aligned} I_D &\ni I_{\mathrm{ch}} + \max(M-1,0)\,p_{12}\,I_{\mathrm{ch}}\\ F &\ni \max(M-1,0)\,p_{12}\,I_{\mathrm{ch}},\qquad b \ni \max(M-1,0)\,p_{12}\,I_{\mathrm{ch}}/I_0 \end{aligned}`,
          note: {
            ko: String.raw`코드 주석: 채널 전자의 애벌랜치로 생긴 정공은 중성 바디의 경계로 들어간다. 채널 전자 자체는 중성 베이스의 BJT 해법을 거치지 않는다. 확률 모델의 생성률 $G = (I_D - I_{\mathrm{seed}} - I_{\mathrm{ch}})/q$에서도 $I_{\mathrm{ch}}$ 자체는 뺀다.`,
            en: String.raw`Code comment: holes from channel-electron avalanches enter at the neutral-body boundary; the channel electrons themselves do not pass through the neutral-base BJT solver. The stochastic generation rate $G = (I_D - I_{\mathrm{seed}} - I_{\mathrm{ch}})/q$ likewise excludes $I_{\mathrm{ch}}$ itself.`,
          },
          code: "photo_mean.py · components() (ii_ch, drain, net); setup_photo.py · state()",
        },
      ],
      notes: [
        {
          ko: "$V_D$ 한 조건(0.05 V)에서만 맞췄으므로 DIBL, 포화, II, 바디 결합은 추출되지 않았다(코드 주석: 'not extracted and not invented'). 따라서 기본 모델의 채널에는 바디 효과가 없다.",
          en: "Fitted at a single $V_D$ (0.05 V): DIBL, saturation, II and body coupling were not extracted (code comment: 'not extracted and not invented'). The channel of the base model therefore has no body effect.",
        },
        {
          ko: "300 K를 가정하며, 이동도는 공칭 W/L과 EOT를 전제로 한 값이다. 채널 II는 벌크 드레인 공핍층의 $M$을 함께 쓴다(표면 전계는 알 수 없음).",
          en: "300 K is assumed, and the mobility is conditional on the nominal W/L and EOT. Channel II shares the bulk drain-depletion $M$ (the surface field is unknown).",
        },
        {
          ko: "광조사 보정에서 $V_G = -1.1$ V의 래치를 설명하려면 채널 시드가 필요하다. 그 시드가 $\\gamma$(p[15])인지 고 $V_D$ 시드(p[17], p[18])인지는 열린 문제다(`open-problems`).",
          en: "Latching at $V_G = -1.1$ V in the illumination calibration needs a channel seed; whether it is $\\gamma$ (p[15]) or the high-$V_D$ seed (p[17], p[18]) is an open problem (`open-problems`).",
        },
      ],
    },
  ],
  related: ["impact-ionization", "bjt-transport", "open-problems", "parameters", "photo"],
  codeRefs: [
    "photo_extension/photo_mean.py",
    "model/janus_calibration_20260920/model_review/channel_fit/fit_results.json",
    "server/params.py",
  ],
};

export default topic;
