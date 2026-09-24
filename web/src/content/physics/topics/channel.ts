// Physics topic "channel" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "channel",
  title: { ko: "채널 전류", en: "Channel current" },
  summary: {
    ko: String.raw`저 $V_D$(0.05 V) $I_D$–$V_G$에서 고정한 EKV형 채널 식을 내부 전압 $u + r$에 적용한다. photo 확장은 DIBL $\eta$, body 결합 $\gamma$, 기울기 $\kappa$, 고 $V_D$ seed 항을 옵션으로 더한다. 채널 전자는 $I_D$와 II에만 들어가고 BJT 수송에는 들어가지 않는다.`,
    en: String.raw`An EKV-type channel expression frozen from the low-$V_D$ (0.05 V) $I_D$–$V_G$ fit is applied at the internal voltage $u + r$. The photo extension optionally adds DIBL $\eta$, body coupling $\gamma$, slope degradation $\kappa$ and a high-$V_D$ seed. Channel electrons enter $I_D$ and II only, not the BJT transport.`,
  },
  tags: ["I_ch", "frozen fit", "open problem −1.1 V"],
  sections: [
    {
      heading: { ko: "고정된 저 V_D 채널 fit", en: "Frozen low-V_D channel fit" },
      body: {
        ko: "`fit_results.json`의 선택 후보(`mobility_degradation`): $V_D = 0.05$ V 한 조건의 $I_D$–$V_G$에 대해 로그 잔차(0.05 dec)와 강전류 잔차를 같은 가중으로 최소화했다. 5–10 pA 미만 전류는 제외(검열), 누설 floor는 넣지 않았고 $R_{sd} = 0$이다.",
        en: "Selected candidate in `fit_results.json` (`mobility_degradation`): one $I_D$–$V_G$ curve at $V_D = 0.05$ V, fitted with equal weights on the log residual (0.05 dec) and the strong-current residual. Currents below 5–10 pA were censored, no leakage floor was fitted, $R_{sd} = 0$.",
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
          name: { ko: "subthreshold 기울기 인자", en: "Subthreshold slope factor" },
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
          name: { ko: "면적당 산화막 용량", en: "Oxide capacitance per area" },
          value: "2.449 × 10⁻³",
          unit: "F/m²",
          code: "Cox_F_m2",
        },
      ],
    },
    {
      heading: { ko: "채널 식 (확장 = 0)", en: "Channel expression (extensions = 0)" },
      body: {
        ko: String.raw`계산값 (포화 영역, $u = 0.3$ V, $r = 3$ V):

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
          label: { ko: "EKV형 순/역 보간", en: "EKV-type forward/reverse interpolation" },
          tex: String.raw`\begin{aligned} V_{\mathrm{ov}} &= V_G - V_{T0},\qquad V_P = V_{\mathrm{ov}}/n\\ s_f &= \ln\!\big(1 + e^{V_P/(2V_T)}\big),\qquad s_r = \ln\!\big(1 + e^{(V_P - u - r)/(2V_T)}\big)\\ I_{\mathrm{ch}} &= \frac{2\,n\,\beta_0\,V_T^2\,(s_f - s_r)(s_f + s_r)}{1 + \theta\,n V_T \ln\!\big(1 + e^{V_{\mathrm{ov}}/(n V_T)}\big)} \end{aligned}`,
          note: {
            ko: String.raw`채널의 $V_{DS}$는 내부 전압 $u + r$(정공 강하·$R_c$·$R_{\mathrm{acc}}$ 제외)이다. 분모는 평활화한 overdrive에 대한 이동도 감쇠.`,
            en: String.raw`The channel $V_{DS}$ is the internal voltage $u + r$ (without hole drop, $R_c$, $R_{\mathrm{acc}}$). The denominator is mobility degradation on the smoothed overdrive.`,
          },
          code: "photo_mean.py · components() (n, ov, pp, sf, sr, ch); mean_model.py · channel_current()",
        },
      ],
    },
    {
      heading: { ko: "photo 확장항 p[14]–p[18]", en: "Photo-extension terms p[14]–p[18]" },
      body: {
        ko: "UI의 채널 seed 옵션(`params.CHANNEL_SEED_OPTIONS`)과 $V_G = -1.1$ V, $u = 0.55$ V, $r = 2.9$ V에서의 계산값:\n\n- none: $I_{\\mathrm{ch}} = 0.312$ pA\n- body_coupling ($\\gamma = 0.2794$, photo preset 기본): 8.76 pA\n- high_vd_seed ($I_p = 1.33$ pA, $S = 0.8$ V/dec): $0.312 + 9.97 = 10.29$ pA",
        en: "Channel-seed options of the UI (`params.CHANNEL_SEED_OPTIONS`), computed at $V_G = -1.1$ V, $u = 0.55$ V, $r = 2.9$ V:\n\n- none: $I_{\\mathrm{ch}} = 0.312$ pA\n- body_coupling ($\\gamma = 0.2794$, photo-preset default): 8.76 pA\n- high_vd_seed ($I_p = 1.33$ pA, $S = 0.8$ V/dec): $0.312 + 9.97 = 10.29$ pA",
      },
      equations: [
        {
          id: "eq-ch-ext",
          label: { ko: "확장된 채널", en: "Extended channel" },
          tex: String.raw`\begin{aligned} n' &= n\,\big[1 + p_{16}\,(u + r)\big]\\ V_{\mathrm{ov}} &= V_G - V_{T0} + p_{14}\,(u + r) + p_{15}\,u,\qquad V_P = V_{\mathrm{ov}}/n'\\ I_{\mathrm{ch}} &\leftarrow I_{\mathrm{ch}}(n', V_{\mathrm{ov}}) + p_{17}\,10^{(V_G + 1.8)/p_{18}}\qquad (p_{17} > 0) \end{aligned}`,
          note: {
            ko: String.raw`p[14] DIBL $\eta$ (V/V), p[15] body 결합 $\gamma$ (V/V, $u$에 곱함), p[16] 기울기 감쇠 $\kappa$ (1/V), p[17] $V_G = -1.8$ V에서의 seed $I_p$ (A), p[18] 기울기 $S$ (V/dec). $n'$은 앞 계수, $V_P$, $\theta$ 항 모두에서 $n$을 대신한다.`,
            en: String.raw`p[14] DIBL $\eta$ (V/V), p[15] body coupling $\gamma$ (V/V, multiplies $u$), p[16] slope degradation $\kappa$ (1/V), p[17] seed $I_p$ at $V_G = -1.8$ V (A), p[18] slope $S$ (V/dec). $n'$ replaces $n$ in the prefactor, in $V_P$ and in the $\theta$ term.`,
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
            ko: String.raw`코드 주석: 채널 전자 avalanche의 정공은 중성 body 경계로 들어간다. 채널 전자 자체는 중성 base BJT 해법을 지나지 않는다. 확률 모델의 생성률 $G = (I_D - I_{\mathrm{seed}} - I_{\mathrm{ch}})/q$도 $I_{\mathrm{ch}}$ 자체는 뺀다.`,
            en: String.raw`Code comment: holes from channel-electron avalanches enter the neutral-body boundary; direct channel electrons do not flow through the neutral-base BJT solver. The stochastic generation rate $G = (I_D - I_{\mathrm{seed}} - I_{\mathrm{ch}})/q$ likewise excludes $I_{\mathrm{ch}}$ itself.`,
          },
          code: "photo_mean.py · components() (ii_ch, drain, net); setup_photo.py · state()",
        },
      ],
      notes: [
        {
          ko: "한 $V_D$(0.05 V)만 fit: DIBL, 포화, II, body 결합은 추출되지 않았다(코드 주석: 'not extracted and not invented'). 따라서 논문 모델의 채널에는 body 효과가 없다.",
          en: "Fitted at one $V_D$ (0.05 V) only: DIBL, saturation, II and body coupling were not extracted (code comment: 'not extracted and not invented'). The paper-model channel therefore has no body effect.",
        },
        {
          ko: "300 K 가정, 이동도는 명목 W/L·EOT 조건부. 채널 II는 bulk drain 공핍층의 $M$을 공유한다(표면 전계 미상).",
          en: "300 K assumed; mobility conditional on nominal W/L and EOT. Channel II shares the bulk drain-depletion $M$ (surface field unknown).",
        },
        {
          ko: "photo 소자의 $V_G = -1.1$ V 래치에는 채널 seed가 필요하며 $\\gamma$(p[15])와 고 $V_D$ seed(p[17], p[18]) 중 어느 쪽인지는 열린 문제다(`open-problems`).",
          en: "Latching of the photo device at $V_G = -1.1$ V needs a channel seed; whether it is $\\gamma$ (p[15]) or the high-$V_D$ seed (p[17], p[18]) is an open problem (`open-problems`).",
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
