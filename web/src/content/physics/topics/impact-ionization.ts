// Physics topic "impact-ionization" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "impact-ionization",
  title: { ko: "Impact ionization(충돌 이온화): α, M(r), II 전류", en: "Impact ionisation: α, M(r), II currents" },
  summary: {
    ko: "drain 공핍층의 선형 전계에서 van Overstraeten–de Man 계수로 국소장 증배 $M(r)$을 계산해 표로 두고, BJT seed·채널·광전자에 $(M-1)$을 곱해 body로 돌아오는 정공 전류를 만든다.",
    en: "The local-field multiplication $M(r)$ is computed from the van Overstraeten–de Man coefficients on the linear field of the drain depletion region and tabulated; $(M-1)$ times the BJT seed, channel and photo-electron currents gives the holes returned to the body.",
  },
  tags: ["Eq. 1", "I_II", "table M(r)"],
  sections: [
    {
      heading: {
        ko: "이온화 계수 (van Overstraeten–de Man, 300 K)",
        en: "Ionisation coefficients (van Overstraeten–de Man, 300 K)",
      },
      equations: [
        {
          id: "eq-ii-alpha",
          label: { ko: "전자 α_n, 정공 α_p [cm⁻¹], E [V/cm]", en: "Electron α_n, hole α_p [cm⁻¹], E [V/cm]" },
          tex: String.raw`\begin{aligned} \alpha_n(E) &= 7.03\times10^{5}\,\exp\!\left(-\frac{1.231\times10^{6}}{E_{\mathrm{eff}}}\right)\\ \alpha_p(E) &= \begin{cases} 1.582\times10^{6}\,\exp\!\left(-\dfrac{2.036\times10^{6}}{E_{\mathrm{eff}}}\right), & E < 4\times10^{5}\\[6pt] 6.71\times10^{5}\,\exp\!\left(-\dfrac{1.693\times10^{6}}{E_{\mathrm{eff}}}\right), & E \ge 4\times10^{5} \end{cases}\\ E_{\mathrm{eff}} &= \max(E,\,1),\qquad \alpha_n(0) = \alpha_p(0) = 0 \end{aligned}`,
          note: {
            ko: "음수·비유한 전계는 `ValueError`. $E = 0$에서는 두 계수 모두 정확히 0으로 둔다.",
            en: "Negative or non-finite fields raise `ValueError`. At $E = 0$ both coefficients are set exactly to 0.",
          },
          code: "standard_mean.py · van_overstraeten_300k()",
        },
      ],
    },
    {
      heading: { ko: "공핍층 전계 분포", en: "Field profile in the depletion region" },
      equations: [
        {
          id: "eq-ii-field",
          label: { ko: "삼각 전계 (정규화 좌표 z)", en: "Triangular field (normalised coordinate z)" },
          tex: String.raw`E(z) = E_{\mathrm{pk}}\,z,\quad z \in [0, 1],\qquad E_{\mathrm{pk}} = \frac{2\,(V_{\mathrm{bi}} + r)}{W},\qquad W = w_d(r) = \sqrt{\frac{2\varepsilon (V_{\mathrm{bi}} + r)}{q N_A}}`,
          note: {
            ko: String.raw`$z = 0$: body 쪽 공핍 경계($E = 0$), $z = 1$: 야금학적 접합(최대 전계). 전자는 $z = 0$에서 들어와 $+z$(n⁺ drain)로, 정공은 $-z$로 움직인다. 전계는 $V_{\mathrm{bi}} + r$만 쓰며 정공 강하·직렬저항은 포함하지 않는다.`,
            en: String.raw`$z = 0$: body-side depletion edge ($E = 0$); $z = 1$: metallurgical junction (peak field). Electrons enter at $z = 0$ and move to $+z$ (n⁺ drain), holes move to $-z$. The field uses only $V_{\mathrm{bi}} + r$ (no hole drop, no series resistance).`,
          },
          code: "mean_model.py · Field.__init__ (width, peak, field = peak·zz)",
        },
      ],
    },
    {
      heading: { ko: "국소장 증배 M(r)", en: "Local-field multiplication M(r)" },
      equations: [
        {
          id: "eq-ii-mult",
          label: { ko: "전자 주입 증배 인자", en: "Electron-initiated multiplication factor" },
          tex: String.raw`\frac{1}{M(r)} = 1 - W\!\int_0^1 \alpha_n\big(E(z)\big)\,\exp\!\left[-W\!\int_0^{z}\big(\alpha_n - \alpha_p\big)\,dz'\right]dz`,
          note: {
            ko: "$z = 0$에 주입된 전자의 국소장 이론(`spatial_mean_gain`과 같은 식). 코드: 안쪽 적분은 `cumulative_trapezoid`, 바깥 적분은 `trapezoid`, $z$는 501점. `den` $= 1/M$.",
            en: "Local-field theory for an electron injected at $z = 0$ (same form as `spatial_mean_gain`). Code: inner integral by `cumulative_trapezoid`, outer by `trapezoid`, 501 points in $z$. `den` $= 1/M$.",
          },
          code: "mean_model.py · Field.__init__ (expint, den)",
        },
        {
          id: "eq-ii-valid",
          label: { ko: "유효 영역 (subcritical)", en: "Validity cut (subcritical)" },
          tex: String.raw`\text{valid}(r)\!:\quad \frac{1}{M} > 10^{-3}\quad\text{and}\quad E_{\mathrm{pk}} \le 1.2\times10^{6}\ \mathrm{V/cm}`,
          note: {
            ko: "$r$ 격자 `linspace(0, 5, 1501)` ($\\Delta r = 3.333$ mV)는 첫 무효 점에서 잘리며, 유효 점이 5개 미만이면 `ValueError`. $N_A = 2.2958\\times10^{17}$ cm⁻³에서는 1501점이 모두 유효: $r_{\\max} = 5$ V, $E_{\\mathrm{pk}}(5\\ \\mathrm{V}) = 6.546\\times10^{5}$ V/cm, $M(5\\ \\mathrm{V}) = 1.8996$ ($1/M = 0.5264$).",
            en: "The $r$ grid `linspace(0, 5, 1501)` ($\\Delta r = 3.333$ mV) is truncated at the first invalid point; fewer than 5 valid points raise `ValueError`. For $N_A = 2.2958\\times10^{17}$ cm⁻³ all 1501 points are valid: $r_{\\max} = 5$ V, $E_{\\mathrm{pk}}(5\\ \\mathrm{V}) = 6.546\\times10^{5}$ V/cm, $M(5\\ \\mathrm{V}) = 1.8996$ ($1/M = 0.5264$).",
          },
          code: "mean_model.py · Field.__init__ (valid, end)",
        },
      ],
      variables: [
        {
          symbol: "M(0)",
          name: { ko: "r = 0에서의 증배", en: "Multiplication at r = 0" },
          value: "1.00911",
          unit: "–",
          code: "MODEL.fg[0]",
        },
        {
          symbol: String.raw`M(1\,\mathrm{V})`,
          name: { ko: "r = 1 V", en: "r = 1 V" },
          value: "1.06190",
          unit: "–",
          code: "MODEL.fg[0]",
        },
        {
          symbol: String.raw`M(2\,\mathrm{V})`,
          name: { ko: "r = 2 V", en: "r = 2 V" },
          value: "1.16536",
          unit: "–",
          code: "MODEL.fg[0]",
        },
        {
          symbol: String.raw`M(3\,\mathrm{V})`,
          name: { ko: "r = 3 V", en: "r = 3 V" },
          value: "1.32508",
          unit: "–",
          code: "MODEL.fg[0]",
        },
        {
          symbol: String.raw`M(4\,\mathrm{V})`,
          name: { ko: "r = 4 V", en: "r = 4 V" },
          value: "1.55792",
          unit: "–",
          code: "MODEL.fg[0]",
        },
        {
          symbol: String.raw`M(5\,\mathrm{V})`,
          name: { ko: "표 끝 (최대)", en: "End of table (maximum)" },
          value: "1.89959",
          unit: "–",
          code: "Field.max_gain",
        },
      ],
    },
    {
      heading: { ko: "표, 보간, 확장 p[19], p[20]", en: "Table, interpolation, extensions p[19], p[20]" },
      body: {
        ko: "`Field.gain`은 PCHIP 보간기지만 `FastModel`은 같은 $r$ 격자에서 샘플링한 값(`fg = [gain(rr), btbt(rr)]`)만 저장하므로, 모델은 실제로 아래의 선형 보간 `f_interp`를 쓴다. $a$를 자르지 않으므로 $[0, 5]$ V 밖에서는 선형 외삽이 된다(`int()`는 0 쪽으로 버림).",
        en: "`Field.gain` is a PCHIP interpolant, but `FastModel` stores only its samples on the same $r$ grid (`fg = [gain(rr), btbt(rr)]`), so the model effectively uses the linear interpolation `f_interp` below. $a$ is not clamped, so outside $[0, 5]$ V it extrapolates linearly (`int()` truncates toward zero).",
      },
      equations: [
        {
          id: "eq-ii-interp",
          label: { ko: "표 보간 f_interp", en: "Table interpolation f_interp" },
          tex: String.raw`f(r) = f_i\,(1-a) + f_{i+1}\,a,\qquad s = \frac{r}{\Delta r},\quad i = \min\!\big(\max(\operatorname{trunc}(s),\,0),\,N-2\big),\quad a = s - i`,
          code: "photo_mean.py · f_interp()",
        },
        {
          id: "eq-ii-mext",
          label: { ko: "확장된 증배 (상태 가설)", en: "Extended multiplication (state hypotheses)" },
          tex: String.raw`M = 1 + \big[M_{\mathrm{tab}}(r + p_{19}) - 1\big]\,e^{\,p_{20}}`,
          note: {
            ko: "p[19]: 국소 접합 전위 오프셋(V, 접합 BTBT 표에도 같이 적용), p[20]: $(M-1)$의 로그 스케일. 논문 모델에서는 둘 다 0.",
            en: "p[19]: local junction potential offset (V, also applied to the junction-BTBT table); p[20]: log-scale of $(M-1)$. Both are 0 in the paper model.",
          },
          code: "photo_mean.py · components() (mult)",
        },
      ],
    },
    {
      heading: { ko: "II 정공 전류", en: "II hole currents" },
      body: {
        ko: String.raw`계산값 ($V_G = -2$ V, 암조건):

- HRS fold ($r = 3.1355$ V): $M = 1.3518$, $I_{\mathrm{seed}} = 9.630$ pA → $I_{\mathrm{II}} = 3.388$ pA (GIDL 2.285 pA와 합쳐 손실 5.673 pA와 균형)
- LRS fold ($r = 1.8301$ V): $M = 1.1441$, $I_{\mathrm{seed}} = 14.56$ nA → $I_{\mathrm{II}} = 2.098$ nA`,
        en: String.raw`Computed values ($V_G = -2$ V, dark):

- HRS fold ($r = 3.1355$ V): $M = 1.3518$, $I_{\mathrm{seed}} = 9.630$ pA → $I_{\mathrm{II}} = 3.388$ pA (with GIDL 2.285 pA it balances the 5.673 pA loss)
- LRS fold ($r = 1.8301$ V): $M = 1.1441$, $I_{\mathrm{seed}} = 14.56$ nA → $I_{\mathrm{II}} = 2.098$ nA`,
      },
      equations: [
        {
          id: "eq-ii-currents",
          label: { ko: "증배로 생긴 정공 (→ body)", en: "Holes created by multiplication (→ body)" },
          tex: String.raw`\begin{aligned} I_{\mathrm{II}} &= (M - 1)\,I_{\mathrm{seed}}\\ I_{\mathrm{II,ch}} &= \max(M-1,\,0)\;I_{\mathrm{ch}}\;p_{12}\\ I_{\mathrm{II,PH}} &= \max(M-1,\,0)\;I_{\mathrm{PH}} \end{aligned}`,
          note: {
            ko: String.raw`$(M-1)I_{\mathrm{seed}}$는 수송 해법의 총전류 $J = Mj_0 + b$로, $I_{\mathrm{II,ch}}$, $I_{\mathrm{II,PH}}$, $I_{\mathrm{loc}}$은 경계 정공원 $b$로 들어간다. 세 항 모두 $I_D$와 $F$에 더해진다. $I_{\mathrm{seed}}$ 항만 max(·, 0) 클램프가 없다.`,
            en: String.raw`$(M-1)I_{\mathrm{seed}}$ enters the transport solver through the total current $J = Mj_0 + b$; $I_{\mathrm{II,ch}}$, $I_{\mathrm{II,PH}}$ and $I_{\mathrm{loc}}$ enter as the boundary hole source $b$. All of them add to $I_D$ and $F$. Only the $I_{\mathrm{seed}}$ term has no max(·, 0) clamp.`,
          },
          code: "photo_mean.py · components() (ii, ii_ch, ii_ph)",
        },
      ],
    },
    {
      heading: { ko: "국소 avalanche 경로 (실험적, p[21]–p[25])", en: "Local avalanche path (experimental, p[21]–p[25])" },
      equations: [
        {
          id: "eq-ii-loc",
          label: { ko: "포화하는 국소 경로 (p[21] > 0일 때만)", en: "Saturating local path (only if p[21] > 0)" },
          tex: String.raw`\begin{aligned} I_{\mathrm{loc}}\big|_{p_{24}\le 0.5} &= \min\!\Big(p_{21}\,e^{p_{23}}\,f_F\,\max(M-1,0)\,\big[I_{\mathrm{GIDL}} + I_{\mathrm{BTBT}} + I_{\mathrm{PH}} + I_{\mathrm{ch}}\big],\ p_{22}\Big)\\ f_F &= \exp\!\big[p_{25}\,(u + r - V_G - 5.6)\big]\\ I_{\mathrm{loc}}\big|_{p_{24}>0.5} &= \min\!\Big(p_{21}\,e^{p_{23}}\,\max(M-1,0)\,\big[I_{\mathrm{seed}} + I_{\mathrm{PH}} + I_{\mathrm{BTBT}}\big],\ p_{22}\Big) \end{aligned}`,
          note: {
            ko: "bulk 정의($p_{24} > 0.5$)는 자기무모순 seed를 쓰도록 3회 고정점 반복하며(매번 수송 재해법), $f_F$를 적용하지 않는다. edge 정의에서 `edge_only`($1.5 < p_{24} < 2.5$)이면 $I_{\\mathrm{ch}}$를 뺀다.",
            en: "The bulk definition ($p_{24} > 0.5$) runs 3 fixed-point iterations with the self-consistent seed (transport re-solved each time) and does not apply $f_F$. In the edge definition, `edge_only` ($1.5 < p_{24} < 2.5$) removes $I_{\\mathrm{ch}}$.",
          },
          code: "photo_mean.py · components() (loc_on, bulk, edge_only, iloc)",
        },
      ],
      notes: [
        {
          ko: "**코드 주의**: `bulk = p[24] > 0.5`가 $p_{24} = 2$에서도 참이므로 bulk 분기가 실행되고 `edge_only` 분기(채널 제외 edge 정의)에는 도달하지 않는다. $u = 0.55$, $r = 3.0$ V, $V_G = -1.8$ V, $I_{\\mathrm{PH}} = 1$ pA에서 $p_{24} = 2$와 $p_{24} = 1$의 출력이 동일함을 확인했다.",
          en: "**Code caveat**: `bulk = p[24] > 0.5` is also true for $p_{24} = 2$, so the bulk branch runs and the `edge_only` branch (edge definition excluding the channel) is never reached. Verified: $p_{24} = 2$ and $p_{24} = 1$ give identical outputs at $u = 0.55$, $r = 3.0$ V, $V_G = -1.8$ V, $I_{\\mathrm{PH}} = 1$ pA.",
        },
      ],
    },
    {
      heading: { ko: "가정과 한계", en: "Assumptions and limits" },
      notes: [
        {
          ko: "국소장(local-field) 모델: dead space·이력 효과 없음, 1D 삼각 전계, 전자 주입만 고려.",
          en: "Local-field model: no dead space or history effects, 1D triangular field, electron injection only.",
        },
        {
          ko: "채널 전자와 광전자도 bulk drain 공핍층의 같은 $M$을 쓴다(코드 주석: 표면 전계는 따로 알 수 없음).",
          en: "Channel electrons and photo-electrons use the same bulk drain-depletion $M$ (code comment: the separate surface field is unknown).",
        },
        {
          ko: String.raw`접합 BTBT와 GIDL 캐리어는 증배하지 않는다($I_{\mathrm{loc}}$ 제외).`,
          en: String.raw`Junction-BTBT and GIDL carriers are not multiplied (except through $I_{\mathrm{loc}}$).`,
        },
        {
          ko: "확률 모델의 II 클러스터 크기 분포는 별도 표다(`stochastic-events`); 여기의 $M$은 평균값이다.",
          en: "The II cluster-size distribution of the stochastic model is a separate table (`stochastic-events`); $M$ here is the mean.",
        },
      ],
    },
  ],
  related: ["electrostatics", "btbt-gidl", "bjt-transport", "charge-balance", "stochastic-events", "open-problems"],
  codeRefs: [
    "model/stl_stochastic_research/process_randomness/standard_mean.py",
    "model/janus_calibration_20260920/idvd_model/mean_model.py",
    "photo_extension/photo_mean.py",
  ],
};

export default topic;
