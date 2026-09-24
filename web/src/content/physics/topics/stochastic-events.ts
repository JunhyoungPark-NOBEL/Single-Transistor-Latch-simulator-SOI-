import type { PhysicsTopic } from "../types";
import { CODE, L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "stochastic-events",
  title: L("확률 사건 (Eq. 2): 단위 사건과 II 클러스터", "Stochastic events (Eq. 2): unit events and II clusters"),
  summary: L(
    "body 전하는 정공 사건 단위로 변한다: ±1 단위 사건(GIDL, 접합 BTBT, 광생성 / 재결합·확산 손실)과 국소 전계 애벌랜치 pmf $p_k(r)$에서 뽑은 크기 $k$의 충돌이온화(II) 클러스터. 모든 사건률은 Eq. 1과 같은 모델 행(row)에서 오므로 평균 drift는 정확히 $F/q$이다.",
    "The body charge changes in discrete hole events: ±1 unit events (GIDL, junction BTBT, photogeneration / recombination and diffusion losses) and impact-ionisation (II) clusters of $k$ holes drawn from a local-field avalanche pmf $p_k(r)$. All rates come from the same model rows as Eq. 1, so the mean drift is exactly $F/q$.",
  ),
  tags: ["Eq. 2", "stochastic", "compound Poisson"],
  sections: [
    {
      heading: L("Eq. 2 — 전하 증분", "Eq. 2 — charge increment"),
      body: L(
        "시간 간격 $\\Delta t$ 동안 body 전하 증분은 사건 종류 $i$별 개수 $\\Delta N_i$와 부호·크기 $s_i$의 합이다.\n\n- **단위 생성** $s=+1$: 접합 BTBT, GIDL, 광생성 $I_{\\mathrm{PH}}$\n- **손실** $s=-1$: bulk SRH 재결합, 소스로의 확산(emitter), 소스 접합 SRH\n- **II 클러스터** $s=+k$, $k\\ge 1$: 시드 전자 하나가 공핍층에서 만든 정공 수",
        "Over a step $\\Delta t$ the body-charge increment is the signed sum of event counts $\\Delta N_i$ with size $s_i$.\n\n- **unit generation** $s=+1$: junction BTBT, GIDL, photogeneration $I_{\\mathrm{PH}}$\n- **losses** $s=-1$: bulk SRH recombination, out-diffusion into the source (emitter), source-junction SRH\n- **II clusters** $s=+k$, $k\\ge 1$: holes created in the depletion region by one seed electron",
      ),
      equations: [
        {
          id: "eq-ev-increment",
          label: L("Eq. 2", "Eq. 2"),
          tex: r`\Delta Q_B = q\sum_i s_i\,\Delta N_i,\qquad s_i\in\{+1,\,-1,\,+k\}`,
          note: L(
            "논문 스윕(0.4 V/s, 2 mV)에서 한 스텝은 $\\Delta t = 5$ ms이다. 소자 시뮬레이터는 이 증분을 직접 적분하지 않고 같은 생성자로 first-passage를 푼다(→ first-passage). 회로 시뮬레이터는 tau-leap 증분으로 쓴다(→ circuit-element).",
            "In the paper sweep (0.4 V/s, 2 mV) one step is $\\Delta t = 5$ ms. The device simulator does not integrate this increment directly; it solves the first passage of the same generator (→ first-passage). The circuit simulator uses it as tau-leap increments (→ circuit-element).",
          ),
          code: "MODEL_SPEC.md §4",
        },
      ],
      variables: [
        { symbol: "q", name: L("기본 전하", "elementary charge"), value: "1.602176634e-19", unit: "C", code: "m.Q" },
        { symbol: r`\Delta N_i`, name: L("사건 종류 i의 개수", "count of event class i"), unit: "1" },
        { symbol: r`s_i`, name: L("사건당 정공 수(부호 포함)", "holes per event (signed)"), unit: "1" },
      ],
    },
    {
      heading: L("모델 행에서 얻는 사건률", "Event rates from the model rows"),
      body: L(
        "고정 $V_D$에서 각 $u$에 대해 $V_D(u,r)=V_D$로 $r$을 풀고 `components()`의 전류로 행을 만든다(`state()`의 열 [3], [4], [10]). 격자에서는 이 값들을 $\\ln$ 공간 PCHIP으로 보간한다.",
        "At fixed $V_D$ each $u$ gets $r$ from $V_D(u,r)=V_D$ and a row built from the `components()` currents (`state()` columns [3], [4], [10]). On the lattice these are interpolated with PCHIP in $\\ln$ space.",
      ),
      equations: [
        {
          id: "eq-ev-generation",
          label: L("총 생성률", "total generation rate"),
          tex: r`G = \frac{I_D - I_{\mathrm{seed}} - I_{\mathrm{ch}}}{q}`,
          note: L(
            "$I_D - I_{\\mathrm{seed}} - I_{\\mathrm{ch}} = I_{\\mathrm{II}} + I_{\\mathrm{II,ch}} + I_{\\mathrm{II,ph}} + I_{\\mathrm{loc}} + I_{\\mathrm{BTBT},j} + I_{\\mathrm{GIDL}} + I_{\\mathrm{PH}}$: 드레인 전류 중 body로 정공을 넣는 모든 성분.",
            "$I_D - I_{\\mathrm{seed}} - I_{\\mathrm{ch}} = I_{\\mathrm{II}} + I_{\\mathrm{II,ch}} + I_{\\mathrm{II,ph}} + I_{\\mathrm{loc}} + I_{\\mathrm{BTBT},j} + I_{\\mathrm{GIDL}} + I_{\\mathrm{PH}}$: every drain-current component that delivers a hole to the body.",
          ),
          code: "setup_photo.py · state() [3] = (z[1]−z[3]−z[16])/Q",
        },
        {
          id: "eq-ev-unit",
          label: L("단위 사건률 (+1)", "unit-event rate (+1)"),
          tex: r`\lambda_{\mathrm{unit}} = \frac{I_{\mathrm{BTBT},j} + I_{\mathrm{GIDL}} + I_{\mathrm{PH}}}{q}`,
          code: "setup_photo.py · state() [10] = z[8]+z[9]+z[18]; compound_fpt.py · make_lattice() bt",
        },
        {
          id: "eq-ev-ii",
          label: L("II 정공 생성률", "II hole-generation rate"),
          tex: r`\begin{aligned}\lambda_{\mathrm{II}} &= \max\!\big(G - \lambda_{\mathrm{unit}},\,0\big)\\ &= \frac{(M-1)\,I_{\mathrm{seed}} + \max(M-1,0)\,\big(p_{12}I_{\mathrm{ch}} + I_{\mathrm{PH}}\big) + I_{\mathrm{loc}}}{q}\end{aligned}`,
          note: L(
            "두 번째 등호는 `photo_mean.components`의 `ii + ii_ch + ii_ph + iloc`이다. 국소 애벌랜치 경로 $I_{\\mathrm{loc}}$(p[21]–p[25])도 II 클러스터로 취급된다.",
            "The second equality is `ii + ii_ch + ii_ph + iloc` of `photo_mean.components`. The local avalanche path $I_{\\mathrm{loc}}$ (p[21]–p[25]) is also treated as II clusters.",
          ),
          code: "compound_fpt.py · make_lattice(): ii_rate = max(rates[:,0] − bt, 0)",
        },
        {
          id: "eq-ev-loss",
          label: L("손실률 (−1)", "loss rate (−1)"),
          tex: r`L = \frac{I_{\mathrm{bulk}} + I_{\mathrm{diff}} + I_{\mathrm{junc}}}{q},\qquad F = q\,(G - L)`,
          code: "setup_photo.py · state() [4] = (z[5]+z[6]+z[7])/Q",
        },
      ],
      notes: [
        L(
          "$F=q(G-L)$는 Eq. 1의 순 정공 전류 `z[2]`와 같다(같은 행). 따라서 확률 모델의 평균은 결정론 모델과 일치한다.",
          "$F=q(G-L)$ equals the net hole current `z[2]` of Eq. 1 (same row), so the mean of the stochastic model coincides with the deterministic model.",
        ),
      ],
    },
    {
      heading: L("II 클러스터 크기 법칙 (평균 보존)", "II cluster size law (mean-preserving)"),
      body: L(
        "크기 $k$ 클러스터의 사건률은 II 정공률을 커널의 평균 클러스터 크기로 나눈 시드 전자율에 $p_k$를 곱한 것이다. $p_k(r)$은 역바이어스 $r$에 대해 커널 행 사이를 선형 보간한다(`np.interp`). $k=0$(이온화 없는 시드)은 점프를 만들지 않는다.",
        "The rate of size-$k$ clusters is the seed-electron rate (II hole rate divided by the kernel's mean cluster size) times $p_k$. $p_k(r)$ is linearly interpolated between kernel rows in the reverse bias $r$ (`np.interp`). $k=0$ (a seed that does not ionise) makes no jump.",
      ),
      equations: [
        {
          id: "eq-ev-cluster-rate",
          label: L("클러스터 사건률", "cluster rates"),
          tex: r`\lambda_k(x) = \lambda_{\mathrm{II}}(x)\,\frac{p_k\big(r(x)\big)}{\sum_{j=1}^{K} j\,p_j\big(r(x)\big)},\qquad k = 1,\dots,K`,
          code: "compound_fpt.py · backward(): jumps=[(k, ii*probs[:,k-1]/mk)]",
        },
        {
          id: "eq-ev-mean-preserve",
          label: L("평균 보존", "mean preservation"),
          tex: r`\sum_{k=1}^{K} k\,\lambda_k = \lambda_{\mathrm{II}},\qquad \sum_{k=1}^{K}\lambda_k = \lambda_{\mathrm{II}}\,\frac{1-p_0}{\mathbb{E}[K]}`,
          note: L(
            "코드는 잔차 $\\max|\\sum_k k\\lambda_k-\\lambda_{\\mathrm{II}}|$를 `mean_drift_error_rate`로 보고한다(부동소수 수준).",
            "The code reports the residual $\\max|\\sum_k k\\lambda_k-\\lambda_{\\mathrm{II}}|$ as `mean_drift_error_rate` (round-off level).",
          ),
          code: "compound_fpt.py · backward(): meanpreserve",
        },
      ],
      variables: [
        { symbol: r`p_k(r)`, name: L("클러스터 크기 pmf", "cluster-size pmf"), code: "ct.cf.pmf, ct.cf.rv" },
        { symbol: "K", name: L("최대 클러스터 크기(확장 커널)", "largest cluster size (extended kernel)"), value: "27", code: "ct.cf.K" },
        { symbol: r`p_0`, name: L("V_G = −2 V, V_D = 3.64 V well에서", "at the well, V_G = −2 V, V_D = 3.64 V"), value: "0.754 (r = 3.11 V)" },
        { symbol: r`\mathbb{E}[K]`, name: L("평균 클러스터 크기 (DC M−1 = 0.346)", "mean cluster size (DC M−1 = 0.346)"), value: "0.344" },
        { symbol: r`\mathbb{E}[K\mid K\ge1]`, name: L("이온화가 일어난 시드당 정공", "holes per ionising seed"), value: "1.40" },
        { symbol: r`\mathbb{E}[K^2]/\mathbb{E}[K]`, name: L("클러스터 Fano 인자", "cluster Fano factor"), value: "1.82" },
      ],
    },
    {
      heading: L("애벌랜치 MC 커널", "Avalanche Monte Carlo kernel"),
      body: L(
        "$p_k(r)$은 결정론 모델과 같은 삼각형(abrupt-junction) 공핍 전계에서 두 캐리어 국소 전계 연쇄를 MC로 세어 얻는다. 전자는 $z=0$(중성 body 가장자리)에서 주입되어 $+z$로, 정공은 $-z$로 움직인다. 이온화마다 $k\\leftarrow k+1$, 원래 캐리어는 계속 진행하고 새 전자·정공이 그 위치에서 출발한다.",
        "$p_k(r)$ is counted by Monte Carlo of the two-carrier local-field cascade in the same triangular (abrupt-junction) depletion field as the deterministic model. The electron is injected at $z=0$ (neutral-body edge) and moves to $+z$, holes move to $-z$. Each ionisation sets $k\\leftarrow k+1$; the parent continues and a new electron and hole start at that position.",
      ),
      equations: [
        {
          id: "eq-ev-field",
          label: L("선형 공핍 전계", "linear depletion field"),
          tex: r`W = \sqrt{\frac{2\varepsilon_{\mathrm{Si}}\,(r+V_{bi})}{qN_A}},\qquad E(z) = \frac{2\,(r+V_{bi})}{W}\,z,\quad z\in[0,1]`,
          code: "avalanche_clusters.py · main(); check_escape.py · install_kernel()",
        },
        {
          id: "eq-ev-ionisation-integral",
          label: L("이온화 적분과 자유 행정", "ionisation integral and free path"),
          tex: r`A_{n,p}(z) = W\!\int_0^{z}\alpha_{n,p}\big(E_{pk}z'\big)\,dz',\qquad \begin{aligned} &A_n(z_{\mathrm{new}}) = A_n(z) + \xi\ \ (e^-)\\ &A_p(z_{\mathrm{new}}) = A_p(z) - \xi\ \ (h^+)\end{aligned}\quad \xi\sim\mathrm{Exp}(1)`,
          note: L(
            "$A_n(z_{\\mathrm{new}})\\ge A_n(1)$ 또는 $A_p(z_{\\mathrm{new}})\\le 0$이면 캐리어는 공핍층을 벗어난다. 계수는 van Overstraeten–de Man 300 K: $\\alpha_n = 7.03\\times10^5 e^{-1.231\\times10^6/E}$, $\\alpha_p = 1.582\\times10^6 e^{-2.036\\times10^6/E}$ ($E<4\\times10^5$ V/cm), 그 이상 $6.71\\times10^5 e^{-1.693\\times10^6/E}$ (cm⁻¹).",
            "A carrier leaves the depletion region when $A_n(z_{\\mathrm{new}})\\ge A_n(1)$ or $A_p(z_{\\mathrm{new}})\\le 0$. Coefficients are van Overstraeten–de Man 300 K: $\\alpha_n = 7.03\\times10^5 e^{-1.231\\times10^6/E}$, $\\alpha_p = 1.582\\times10^6 e^{-2.036\\times10^6/E}$ ($E<4\\times10^5$ V/cm), else $6.71\\times10^5 e^{-1.693\\times10^6/E}$ (cm⁻¹).",
          ),
          code: "avalanche_clusters.py · simulate() (numba)",
        },
        {
          id: "eq-ev-pmf",
          label: L("경험적 pmf", "empirical pmf"),
          tex: r`p_k(r) = \frac{1}{N}\,\#\{\text{cascades with } k \text{ ionisations}\},\qquad \mathbb{E}[K](r)\approx M(r)-1`,
          note: L(
            "평균은 DC 증배 $M(r)-1$과 z-score로 검사된다(평균 폐합). 웹 시뮬레이터는 `gate_fpt.py`/`setup_photo.py`처럼 확장 커널 `gate_avalanche_extended.npz`를 설치한다.",
            "The mean is checked against the DC multiplication $M(r)-1$ by a z-score (mean closure). The web simulator installs the extended kernel `gate_avalanche_extended.npz`, as `gate_fpt.py`/`setup_photo.py` do.",
          ),
          code: "check_escape.py · install_kernel(); setup_photo.py (extended kernel override)",
        },
      ],
      variables: [
        { symbol: r`p_k^{\mathrm{orig}}`, name: L("원 커널: r 0.7–4.5 V, 77행, N=400000", "original kernel: r 0.7–4.5 V, 77 rows, N = 400000"), code: "avalanche/cluster_pmf.npz" },
        { symbol: r`p_k^{N_A}`, name: L("보정 N_A용 재계산: 같은 격자, N=200000", "recomputed for the calibrated N_A: same grid, N = 200000"), code: "avalanche_2.29577316e+17.npz" },
        { symbol: r`p_k^{\mathrm{ext}}`, name: L("확장 커널(사용됨): r 0.7–5.0 V, 87행, N=200000, K=27", "extended kernel (used): r 0.7–5.0 V, 87 rows, N = 200000, K = 27"), code: "gate_avalanche_extended.npz" },
        { symbol: r`V_{bi}`, name: L("소스/드레인 내부 전위", "built-in potential"), value: "1.0334", unit: "V", code: "MODEL.vbi" },
        { symbol: r`N_A`, name: L("body 도핑(보정)", "body doping (calibrated)"), value: "2.2958e17", unit: "cm⁻³", code: "refit_3.json NA_cm3" },
      ],
      notes: [
        L(
          "국소 계수(에너지 이력·dead space 없음). BTBT/GIDL 시드는 평균 모델과 같이 추가 증배되지 않는다(단위 사건). 확장 커널 생성 스크립트는 인수인계에 없다(같은 MC를 5 V까지 연장한 것으로 보임).",
          "Local coefficients (no energy history, no dead space). BTBT/GIDL seeds are not additionally multiplied, consistent with the mean model (unit events). The script that generated the extended kernel is not in the handoff (it appears to be the same MC extended to 5 V).",
        ),
      ],
    },
    {
      heading: L("Poisson / Gaussian 극한", "Poisson and Gaussian limits"),
      body: L(
        "복합 Poisson 생성자의 처음 두 모멘트는 다음과 같다(정확한 결과). 클러스터가 없으면($K\\equiv1$) Fano 인자는 1이 되고(`backward(unit=True)`, `lu_fpt.py`의 단일 정공 Poisson 대조), 스텝당 사건 수가 크면(≈100 이상) 각 $\\Delta N_i$를 평균·분산 $\\lambda_i\\Delta t$의 Gaussian으로 바꿀 수 있다.",
        "The first two moments of the compound-Poisson generator are exact. Without clusters ($K\\equiv1$) the Fano factor is 1 (`backward(unit=True)`, the single-hole Poisson control of `lu_fpt.py`); with many events per step (≳ 100) each $\\Delta N_i$ can be replaced by a Gaussian with mean and variance $\\lambda_i\\Delta t$.",
      ),
      equations: [
        {
          id: "eq-ev-moments",
          label: L("증분의 평균과 분산", "mean and variance of the increment"),
          tex: r`\begin{aligned}\mathbb{E}[\Delta Q_B] &= F\,\Delta t\\ \operatorname{Var}[\Delta Q_B] &= q^2\Big(\lambda_{\mathrm{unit}} + L + \lambda_{\mathrm{II}}\,\frac{\mathbb{E}[K^2]}{\mathbb{E}[K]}\Big)\Delta t\end{aligned}`,
          note: L(
            "$\\operatorname{Var}=q^2\\sum_i s_i^2\\lambda_i\\Delta t$에서 유도. 클러스터링은 II 분산을 약 1.8배 키운다.",
            "Follows from $\\operatorname{Var}=q^2\\sum_i s_i^2\\lambda_i\\Delta t$. Clustering raises the II variance by about 1.8×.",
          ),
          code: "derived from compound_fpt.py · backward()",
        },
      ],
      notes: [
        L(
          "VALIDATION.md의 캐리어 잡음만의 기여(논문, V_G = −2 V): II 4.6, BTBT 2.7, REC 4.3, DIFF 1.8, 네 가지 모두 7.8 mV. 이를 만든 스크립트는 인수인계에 없어 검증 탭(full)이 재구성한다(→ validation); 네 가지 모두를 쓰는 현재 FPT 노드는 8.03 mV를 준다(→ first-passage).",
          "Carrier-noise-only contributions in VALIDATION.md (paper, V_G = −2 V): II 4.6, BTBT 2.7, REC 4.3, DIFF 1.8, all four 7.8 mV. The script that produced them is not in the handoff; the validation tab (full) reconstructs it (→ validation). The current FPT node with all four gives 8.03 mV (→ first-passage).",
        ),
        L(
          "빠른 SRH 계수 요동은 Poisson 계수로 두며, 느린 변조는 SRH 계수 평균을 명시적으로 바꾼다(`conditional_table.py` 요약).",
          "Fast SRH count fluctuations are kept as Poisson counts; slow modulation changes the mean SRH coefficient explicitly (`conditional_table.py` summary).",
        ),
      ],
    },
  ],
  related: ["impact-ionization", "btbt-gidl", "charge-balance", "photo", "first-passage", "circuit-element"],
  codeRefs: [CODE.compound, CODE.avalanche, CODE.checkEscape, CODE.conditional, CODE.setupPhoto, CODE.photoMean, CODE.gateFpt],
};

export default topic;
