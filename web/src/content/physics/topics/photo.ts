// Physics topic "photo" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "photo",
  title: { ko: "광조사: 광생성 정공 공급", en: "Illumination: photogenerated hole supply" },
  summary: {
    ko: String.raw`빛은 바디에 균일하게 공급되는 정공 전류 $I_{\mathrm{PH}}$(p[13])로 들어간다. 짝을 이루는 전자는 드레인에서 수집되며 $(M-1)$배로 증배된다. 광세기 변환 $I_{\mathrm{PH}} = R\,P$의 $R = 0.75$ pA/mW는 래치 전 plateau에 맞춘 값이다.`,
    en: String.raw`Light enters as a uniform hole supply $I_{\mathrm{PH}}$ (p[13]) into the body; the partner electron is collected at the drain and multiplied by $(M-1)$. The conversion $I_{\mathrm{PH}} = R\,P$ with $R = 0.75$ pA/mW is fitted to the pre-latch plateau.`,
  },
  tags: ["I_PH", "p[13]", "illumination calibration"],
  sections: [
    {
      heading: { ko: "모델 속의 I_PH", en: "I_PH in the model" },
      equations: [
        {
          id: "eq-ph-terms",
          label: { ko: "광생성 항이 들어가는 곳", en: "Where the photogeneration terms enter" },
          tex: String.raw`\begin{aligned} F &\ni I_{\mathrm{PH}} + \max(M-1,0)\,I_{\mathrm{PH}}\\ I_D &\ni I_{\mathrm{PH}} + \max(M-1,0)\,I_{\mathrm{PH}}\\ b &\ni \big[I_{\mathrm{PH}} + \max(M-1,0)\,I_{\mathrm{PH}}\big]/I_0 \end{aligned}`,
          note: {
            ko: "광생성 전자–정공 쌍마다 정공 하나는 바디로 들어가고(수송 해법에서 컬렉터 쪽 경계 정공원 $b$), 전자 하나는 드레인에서 수집되어 다른 드레인 접합 전자처럼 증배된다. 채널 II와 달리 p[12] 배율은 적용하지 않는다. 확률 모델에서 광생성은 단위(Poisson) 사건이다(`state()` 10번 열 $= I_{\\mathrm{BTBT}} + I_{\\mathrm{GIDL}} + I_{\\mathrm{PH}}$).",
            en: "Each photogenerated pair puts one hole into the body (the collector-side boundary hole source $b$ of the transport solver) and one electron into the drain, where it multiplies like any other drain-junction electron. Unlike channel II, the p[12] scale is not applied. In the stochastic model photogeneration is a unit (Poisson) event (`state()` column 10 $= I_{\\mathrm{BTBT}} + I_{\\mathrm{GIDL}} + I_{\\mathrm{PH}}$).",
          },
          code: "photo_mean.py · components() (iph, ii_ph); setup_photo.py · state()",
        },
      ],
    },
    {
      heading: { ko: "광세기 → I_PH 변환", en: "Optical power → I_PH conversion" },
      body: {
        ko: String.raw`fit 점검(측정 plateau − 암전류 / 모델 $I_D$ / 이득 $I_D/I_{\mathrm{PH}}$, 계산값):

- 0.799 mW: 0.450 / 0.758 pA / 1.26
- 1.42 mW: 1.733 / 1.529 pA / 1.44
- 2.55 mW: 3.046 / 3.318 pA / 1.74
- 2.80 mW: 4.693 / 3.777 pA / 1.80
- 3.31 mW: 6.303 / 4.777 pA / 1.92

$I_{\mathrm{PH}}$ = 0.5–5 pA에서 바이폴라 이득은 1.23–2.64다. 저주입 HRS에서는 광생성 정공이 주로 접합 SRH로 사라지고, $u$가 오를수록 시드 전류의 비중이 커져 이득이 증가한다.`,
        en: String.raw`Fit check (measured plateau − dark / model $I_D$ / gain $I_D/I_{\mathrm{PH}}$, computed):

- 0.799 mW: 0.450 / 0.758 pA / 1.26
- 1.42 mW: 1.733 / 1.529 pA / 1.44
- 2.55 mW: 3.046 / 3.318 pA / 1.74
- 2.80 mW: 4.693 / 3.777 pA / 1.80
- 3.31 mW: 6.303 / 4.777 pA / 1.92

For $I_{\mathrm{PH}}$ = 0.5–5 pA the bipolar gain is 1.23–2.64. In the low-injection HRS the photogenerated holes are lost mainly through junction SRH; as $u$ rises, the seed fraction and hence the gain increase.`,
      },
      equations: [
        {
          id: "eq-ph-conv",
          label: { ko: "선형 변환 (이 소자)", en: "Linear conversion (this device)" },
          tex: String.raw`I_{\mathrm{PH}} = R\,P,\qquad R = 0.75\ \mathrm{pA/mW}`,
          code: "photo_conversion_fit.json · R_A_per_mW; server/params.py · iph_A()",
        },
        {
          id: "eq-ph-fit",
          label: { ko: "R의 로그 최소제곱 fit", en: "Log least-squares fit of R" },
          tex: String.raw`R = \arg\min_{R}\ \sum_{k}\Big[\ln I_D^{\mathrm{model}}\big(R P_k;\ V_D = 1.25\ \mathrm{V},\ V_G = -1.8\ \mathrm{V}\big) - \ln \Delta I_k^{\mathrm{meas}}\Big]^2`,
          note: {
            ko: String.raw`$P_k$ = 0.799, 1.42, 2.55, 2.80, 3.31 mW이고, $\Delta I_k$는 0.5–2 V 구간에서 평균한 광조사 $I_D$에서 암전류 $I_D$를 뺀 값이다. 모델 $I_D$는 HRS 정상점(고정 $V_D$에서 $F$의 부호가 처음 바뀌는 점)에서 구하며, $R \in [0.1, 5]$ pA/mW 범위에서 제한 최소화를 한다. I–V 스윕의 $V_G$는 −1.8 V로 가정했다.`,
            en: String.raw`$P_k$ = 0.799, 1.42, 2.55, 2.80, 3.31 mW; $\Delta I_k$ is the illuminated $I_D$ averaged over 0.5–2 V minus the dark $I_D$. The model $I_D$ is taken at the HRS steady state (first sign change of $F$ at fixed $V_D$), with bounded minimization over $R \in [0.1, 5]$ pA/mW. The $V_G$ of the I–V sweeps is assumed to be −1.8 V.`,
          },
          code: "photo_extension/fit_conversion.py",
        },
      ],
      variables: [
        {
          symbol: String.raw`P = 1.15\ \mathrm{mW}`,
          name: { ko: "측정 조건 1", en: "Measured condition 1" },
          value: "0.8625",
          unit: "pA",
          code: "I_PH_pA['1.15']",
        },
        {
          symbol: String.raw`P = 2.55\ \mathrm{mW}`,
          name: { ko: "측정 조건 2", en: "Measured condition 2" },
          value: "1.9125",
          unit: "pA",
          code: "I_PH_pA['2.55']",
        },
        {
          symbol: String.raw`P = 3.51\ \mathrm{mW}`,
          name: { ko: "측정 조건 3", en: "Measured condition 3" },
          value: "2.6325",
          unit: "pA",
          code: "I_PH_pA['3.51']",
        },
      ],
    },
    {
      heading: { ko: "fold에 미치는 영향", en: "Effect on the folds" },
      body: {
        ko: "$V_G = -1.8$ V, $I_{\\mathrm{PH}}$ = 0 / 0.8625 / 1.9125 / 2.6325 pA (계산값):\n\n- 기본 모델($\\gamma = 0$, $\\delta\\varphi_{G0} = 0$): $V_{\\mathrm{LU}}$ = 3.8644 / 3.7088 / 3.4633 / 3.2907 V, $V_{\\mathrm{LD}}$ = 2.5979 → 2.5962 V. 정확히 2.63 pA에서는 3.2913 V (`VALIDATION.md`).\n- 광조사 보정 프리셋($\\gamma = 0.2794$, $\\delta\\varphi_{G0} = +0.0744$ V): 3.8042 / 3.6598 / 3.4371 / 3.2781 V; $V_G = -1.1$ V에서는 3.4079 / 3.1930 / 3.0231 / 2.9417 V.\n- 광조사 측정 기록의 평균(400 사이클, 1200 V/s): −1.8 V 3.806 / 3.500 / 3.336 / 3.073 V; −1.1 V 3.408 / 3.273 / 3.098 / 2.933 V. fold는 결정론적 중심값이고, 측정 평균에는 확률적 탈출과 상태 산포가 함께 들어 있다.\n\n기전: $I_{\\mathrm{PH}}$는 $V_D$와 무관한 정공 공급이므로, 각 $V_D$에서 HRS 균형이 더 큰 $u$(더 큰 시드 전류)에서 이루어지고 필요한 증배가 줄어든다. 따라서 fold가 더 낮은 $V_D$로 이동한다. 2.63 pA에서의 fold는 $u = 0.592$ V, $r = 2.699$ V, $M = 1.270$이다(암조건: $u = 0.559$ V, $r = 3.305$ V, $M = 1.387$). LRS는 nA–µA 수준의 BJT 전류가 지배하므로 $V_{\\mathrm{LD}}$는 거의 변하지 않는다.",
        en: "$V_G = -1.8$ V, $I_{\\mathrm{PH}}$ = 0 / 0.8625 / 1.9125 / 2.6325 pA (computed):\n\n- Base model ($\\gamma = 0$, $\\delta\\varphi_{G0} = 0$): $V_{\\mathrm{LU}}$ = 3.8644 / 3.7088 / 3.4633 / 3.2907 V, $V_{\\mathrm{LD}}$ = 2.5979 → 2.5962 V. At exactly 2.63 pA: 3.2913 V (`VALIDATION.md`).\n- Illumination-calibration preset ($\\gamma = 0.2794$, $\\delta\\varphi_{G0} = +0.0744$ V): 3.8042 / 3.6598 / 3.4371 / 3.2781 V; at $V_G = -1.1$ V: 3.4079 / 3.1930 / 3.0231 / 2.9417 V.\n- Means of the illumination records (400 cycles, 1200 V/s): −1.8 V 3.806 / 3.500 / 3.336 / 3.073 V; −1.1 V 3.408 / 3.273 / 3.098 / 2.933 V. The fold is the deterministic center; the measured means also include stochastic escape and state spread.\n\nMechanism: $I_{\\mathrm{PH}}$ is a $V_D$-independent hole supply, so at every $V_D$ the HRS balance sits at a larger $u$ (larger seed current) and less multiplication is needed; the fold therefore moves to a lower $V_D$. Fold at 2.63 pA: $u = 0.592$ V, $r = 2.699$ V, $M = 1.270$ (dark: $u = 0.559$ V, $r = 3.305$ V, $M = 1.387$). The LRS is dominated by BJT currents of nA to µA, so $V_{\\mathrm{LD}}$ barely moves.",
      },
    },
    {
      heading: { ko: "광조사 시의 저 V_D branch", en: "The illuminated low-V_D branch" },
      equations: [
        {
          id: "eq-ph-u0",
          label: { ko: "r = 0에서의 균형", en: "Balance at r = 0" },
          tex: String.raw`F(u_0, 0) = 0\ \Rightarrow\ u_0 = 0.533\ \mathrm{V}\qquad (I_{\mathrm{PH}} = 2.63\ \mathrm{pA},\ V_G = -1.8\ \mathrm{V})`,
          note: {
            ko: "$u < u_0$에서는 $F(u, 0) > 0$이므로 `curve_grid()`가 그 점을 건너뛴다($r \\ge 0$인 해가 없고, 모델에는 드레인 순방향 branch $r < 0$이 없다). $u = 0$ 점은 조건 없이 넣으므로(이때 $V_D \\approx 1.3$ µV, $I_D = 2.65$ pA) 궤적은 $(0, 0)$에서 $u \\approx 0.533$ V($V_D = 0.667$ V)로 건너뛴다. `double_curve`는 $V_D = 0$에서 $I_D = 0$으로 둔다.",
            en: "For $u < u_0$, $F(u, 0) > 0$ and `curve_grid()` skips the point (there is no root with $r \\ge 0$, and the model has no forward-biased drain branch $r < 0$). The $u = 0$ point is inserted unconditionally ($V_D \\approx 1.3$ µV, $I_D = 2.65$ pA there), so the locus jumps from $(0, 0)$ to $u \\approx 0.533$ V ($V_D = 0.667$ V). `double_curve` sets $I_D = 0$ at $V_D = 0$.",
          },
          code: "photo_mean.py · curve_grid()",
        },
      ],
      notes: [
        {
          ko: String.raw`코드 주석은 '균일한 정공 공급'이라고 하지만, 수송 해법에서 $I_{\mathrm{PH}}$는 $dj/dz$의 분포 생성 항이 아니라 $b$(일정한 총전류 $J$)로만 들어간다. 즉 컬렉터 쪽에서 들어오는 정공 전류로 다룬다(GIDL, BTBT, 채널 II와 같음).`,
          en: String.raw`The code comment says 'uniform hole supply', but in the transport solver $I_{\mathrm{PH}}$ enters only through $b$ (the constant total current $J$), not as a distributed generation term in $dj/dz$; it is treated as hole current entering at the collector side (like GIDL, BTBT and channel II).`,
        },
        {
          ko: "균일 공급 모델이다. 생성 분포, 파장·흡수 모델, 빛에 의한 수명이나 국소 상태의 변화는 포함하지 않는다.",
          en: "Uniform supply: no generation profile, no wavelength or absorption model, and no light-induced change of lifetimes or local states.",
        },
        {
          ko: String.raw`$R$은 기준 보정 파라미터($\gamma = 0$)와 가정한 $V_G = -1.8$ V로 맞췄다. 광세기별 plateau 점은 최대 약 70 %까지 벗어난다(0.799 mW에서 0.758 대 0.450 pA).`,
          en: String.raw`$R$ was fitted with the reference-calibration parameters ($\gamma = 0$) at the assumed $V_G = -1.8$ V. Individual plateau points deviate by up to about 70 % (0.758 vs 0.450 pA at 0.799 mW).`,
        },
        {
          ko: String.raw`MODEL_SPEC §2에는 광전자가 '채널 전자처럼' 증배된다고 적혀 있지만, 코드는 채널 II 배율 p[12] 없이 $\max(M-1, 0)$만 곱한다.`,
          en: String.raw`MODEL_SPEC §2 states that photo-electrons multiply 'like channel electrons', but the code multiplies them by $\max(M-1, 0)$ without the channel-II scale p[12].`,
        },
      ],
    },
  ],
  related: ["impact-ionization", "charge-balance", "channel", "local-states", "sweep-mc", "validation"],
  codeRefs: [
    "photo_extension/photo_mean.py",
    "photo_extension/fit_conversion.py",
    "photo_extension/photo_conversion_fit.json",
    "server/params.py",
  ],
};

export default topic;
