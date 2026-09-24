import type { PhysicsTopic } from "../types";
import { CODE, L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "circuit-element",
  title: L("회로 소자로서의 STL (MNA 과도해석)", "STL as a circuit element (MNA transient)"),
  summary: L(
    "STL은 body 전하 $Q_B$를 상태로 갖는 3단자(D, G, S) 비선형 소자이다. 내부 미지수 $(u,r)$가 단자 전압식과 전하식을 만족하도록 MNA 행렬에 넣고, 음함수 BE/TRAP + Newton으로 적분한다. 확률 모드에서는 Eq. 2 사건 증분을 tau-leap으로 더한다. latch는 명시적 스위치 없이 ODE에서 나온다.",
    "The STL is a three-terminal (D, G, S) nonlinear element whose state is the body charge $Q_B$. Internal unknowns $(u,r)$ enter the MNA system through a terminal-voltage equation and a charge equation, integrated with implicit BE/TRAP + Newton. In stochastic mode the Eq. 2 event increments are added as tau-leaps. Latching emerges from the ODE without an explicit switch.",
  ),
  tags: ["circuit", "MNA", "transient", "tau-leap"],
  sections: [
    {
      heading: L("소자 방정식", "Element equations"),
      body: L(
        "단자 D, G, S (+ 선택적 광입력 $I_{\\mathrm{PH}}(t)$ → p[13]). 요소마다 미지수 $(u_k,r_k)$ 두 개와 식 두 개(E1, E2)가 추가된다. $V_{GS}$는 매 평가에서 p[11]에 들어간다. 전하 좌표는 compound-FPT 격자와 같다(상수 $-C_{ox}V_{GS}$ 차이). $V_{GS}$가 시간에 따라 바뀌면 같은 $Q$에서 $\\psi$가 따라 움직인다(게이트–body 용량 결합).",
        "Terminals D, G, S (+ optional light input $I_{\\mathrm{PH}}(t)$ → p[13]). Each element adds two unknowns $(u_k,r_k)$ and two equations (E1, E2). $V_{GS}$ is written to p[11] at every evaluation. The charge coordinate is that of the compound-FPT lattice (up to the constant $-C_{ox}V_{GS}$). When $V_{GS}$ changes in time, $\\psi$ follows at fixed $Q$ (gate–body capacitive coupling).",
      ),
      equations: [
        {
          id: "eq-ckt-charge",
          label: L("body 전하 좌표", "body-charge coordinate"),
          tex: r`\begin{aligned} &Q(u,r;V_{GS}) = C_{ox}\big(\psi(u) - V_{GS}\big) + Q_{\mathrm{exc}}(u,r) + qN_A A\,L_n(u,r)\\ &\psi = u - V_T\ln\!\big(1+\delta/N_A\big)\end{aligned}`,
          note: L(
            "$Q_{\\mathrm{exc}}=$ `z[13]` $-C_{ox}u$, $L_n=$ `z[11]`. 주의: `data/tables`의 `Q_B_C`와 MODEL_SPEC §1의 $Q_B$는 $Q_{\\mathrm{exc}}$만이다.",
            "$Q_{\\mathrm{exc}}=$ `z[13]` $-C_{ox}u$, $L_n=$ `z[11]`. Note: `Q_B_C` in `data/tables` and the $Q_B$ of MODEL_SPEC §1 are $Q_{\\mathrm{exc}}$ only.",
          ),
          code: "circuit/element.py · stl_eval() out[3]",
        },
        {
          id: "eq-ckt-e1",
          label: L("E1: 단자 전압", "E1: terminal voltage"),
          tex: r`\begin{aligned} &V_D(u,r) - (v_D - v_S) = 0\\ &V_D(u,r) = u + r + V_{hd}(u,r) + (R_c + R_{acc})\,I_D(u,r)\end{aligned}`,
          code: "circuit/mna.py · assemble() row ku",
        },
        {
          id: "eq-ckt-e2",
          label: L("E2: 전하식 (C_ox로 정규화)", "E2: charge equation (normalised by C_ox)"),
          tex: r`\frac{Q(u,r;V_{GS}) - Q_c - \theta\,h\,F(u,r)}{C_{ox}} = 0`,
          note: L(
            "결정론 BE: $Q_c=Q_n$, $\\theta=1$. TRAP: $Q_c=Q_n+\\tfrac h2F_n$, $\\theta=\\tfrac12$ ($h<2\\tau_{\\min}$일 때만, 아니면 BE). 확률(잡음 분해): $Q_c=Q_n+\\Delta Q_{\\mathrm{ev}}$, $\\theta=0$. 확률(빠른 완화): $Q_c=Q_n+\\Delta Q_{\\mathrm{ev}}-hF_n$, $\\theta=1$ (drift-implicit).",
            "Deterministic BE: $Q_c=Q_n$, $\\theta=1$. TRAP: $Q_c=Q_n+\\tfrac h2F_n$, $\\theta=\\tfrac12$ (only when $h<2\\tau_{\\min}$, else BE). Stochastic (noise resolved): $Q_c=Q_n+\\Delta Q_{\\mathrm{ev}}$, $\\theta=0$. Stochastic (fast relaxation): $Q_c=Q_n+\\Delta Q_{\\mathrm{ev}}-hF_n$, $\\theta=1$ (drift-implicit).",
          ),
          code: "circuit/mna.py · assemble() row kr; run_chunk()",
        },
      ],
      variables: [
        { symbol: r`I_D`, name: L("드레인 전류: D 노드에서 나가 S 노드로 들어감", "drain current: leaves node D, enters node S"), unit: "A", code: "z[1]" },
        { symbol: r`F`, name: L("순 정공 전류 G − L", "net hole current G − L"), unit: "A", code: "z[2]" },
        { symbol: r`C_{ox}`, name: L("E2 정규화 용량", "E2 normalisation capacitance"), value: "0.2449", unit: "fF", code: "COX_F" },
      ],
      notes: [
        L(
          "엔진 영역 밖 연속화: $u<0$(빠른 하강 후 소스 역바이어스)은 $u=0$의 전류에 $X'(0^+)V_T(e^{u/V_T}-1)$을 이어 붙이고, $r<0$(드레인 순바이어스)은 소스와 같은 포화전류의 대칭 순방향 드레인 다이오드 + 공핍 SRH를 손실로 더한다. 엔진은 이 영역을 평가하지 않는다.",
          "Continuation outside the engine domain: $u<0$ (source reverse biased after fast down-ramps) extends the $u=0$ currents by $X'(0^+)V_T(e^{u/V_T}-1)$; $r<0$ (drain forward biased) adds a symmetric forward drain diode (source saturation current) plus depletion SRH as a loss. The engine itself never evaluates these regions.",
        ),
      ],
    },
    {
      heading: L("MNA 조립과 Newton", "MNA assembly and Newton"),
      body: L(
        "미지수 $x=[v_1..v_{N-1},\\ i_V,\\ (u_k,r_k)]$. 각 비접지 노드의 KCL(나가는 전류 합 = 0), 전압원 행 $v_a-v_b-V(t)=0$, STL의 E1/E2. 노드마다 $G_{\\min}=10^{-18}$ S. STL 블록 Jacobian은 `stl_eval`의 전진 유한차분($10^{-6}$ V; 영역 밖이면 후진). 반복당 $|\\Delta u|\\le50$ mV, $|\\Delta r|\\le1$ V, 영역을 벗어나면 반감 backtracking.",
        "Unknowns $x=[v_1..v_{N-1},\\ i_V,\\ (u_k,r_k)]$. KCL at each non-ground node (sum of leaving currents = 0), source rows $v_a-v_b-V(t)=0$, STL rows E1/E2. $G_{\\min}=10^{-18}$ S at every node. The STL Jacobian block is a forward finite difference of `stl_eval` ($10^{-6}$ V; backward outside the domain). Per iteration $|\\Delta u|\\le50$ mV, $|\\Delta r|\\le1$ V, halving backtracking when leaving the domain.",
      ),
      equations: [
        {
          id: "eq-ckt-companion",
          label: L("커패시터 companion 모델", "capacitor companion models"),
          tex: r`\text{BE: } i^{n+1} = \frac{C}{h}\big(v^{n+1}-v^{n}\big),\qquad \text{TRAP: } i^{n+1} = \frac{2C}{h}\big(v^{n+1}-v^{n}\big) - i^{n}`,
          code: "circuit/mna.py · cap_companion()",
        },
        {
          id: "eq-ckt-kcl",
          label: L("KCL 스탬프", "KCL stamps"),
          tex: r`\sum_{R}\frac{v_a-v_b}{R} + \sum_{C} i_C + \sum_{V} i_V + \sum_{I} I(t) \pm \sum_{\mathrm{STL}} I_D(u,r) + G_{\min}v = 0`,
          code: "circuit/mna.py · assemble()",
        },
      ],
      notes: [
        L(
          "초기 동작점: $u=0$(빈 body)으로 고정해 풀고, 의사과도 BE($h$: $10^{-12}$ s에서 ×3, 최대 $10^8$ s)로 빈 body에서 도달하는 저전류 상태를 찾는다.",
          "Initial operating point: solve with $u=0$ (empty body), then pseudo-transient BE ($h$ from $10^{-12}$ s, ×3 per step, up to $10^8$ s) to the low-current state reachable from an empty body.",
        ),
      ],
    },
    {
      heading: L("확률 증분 (tau-leap)", "Stochastic increments (tau-leap)"),
      body: L(
        "스텝 시작 상태의 사건률로 $\\Delta Q_{\\mathrm{ev}}$를 뽑는다: 단위 사건 $N_\\uparrow$, 손실 $N_\\downarrow$는 Poisson, II는 클러스터 수 Poisson × 크기 $k\\sim p_k(r)/P_1$. 평균이 $\\lambda$ > 문턱이면 Gaussian 극한. 연속화 영역의 음의 전류는 부호를 바꿔 반대 방향 사건으로 넣으므로 $\\mathbb E[\\Delta Q]=hF$가 정확히 유지된다.",
        "$\\Delta Q_{\\mathrm{ev}}$ is drawn from the rates at the start of the step: unit events $N_\\uparrow$ and losses $N_\\downarrow$ are Poisson, II is a Poisson number of clusters with sizes $k\\sim p_k(r)/P_1$. Above a mean threshold the Gaussian limit is used. Negative currents of the continuation regions are folded into the opposite event class, so $\\mathbb E[\\Delta Q]=hF$ holds exactly.",
      ),
      equations: [
        {
          id: "eq-ckt-dq",
          label: L("사건 증분", "event increment"),
          tex: r`\begin{aligned} &\Delta Q_{\mathrm{ev}} = q\Big(N_\uparrow + \sum_{c=1}^{N_c} k_c - N_\downarrow\Big)\\ &N_\uparrow\sim\mathcal P\big(\lambda_{\mathrm{unit}}h\big),\quad N_\downarrow\sim\mathcal P(Lh),\quad N_c\sim\mathcal P\Big(\lambda_{\mathrm{II}}h\,\frac{P_1}{M_1}\Big)\end{aligned}`,
          note: L(
            "$P_1=\\sum_{k\\ge1}p_k$, $M_1=\\sum k p_k$, $M_2=\\sum k^2p_k$. Gaussian 극한: $\\sum k_c\\approx\\lambda_{\\mathrm{II}}h+\\sqrt{\\lambda_{\\mathrm{II}}hM_2/M_1}\\,\\xi$ (0에서 절단).",
            "$P_1=\\sum_{k\\ge1}p_k$, $M_1=\\sum k p_k$, $M_2=\\sum k^2p_k$. Gaussian limit: $\\sum k_c\\approx\\lambda_{\\mathrm{II}}h+\\sqrt{\\lambda_{\\mathrm{II}}hM_2/M_1}\\,\\xi$ (clipped at 0).",
          ),
          code: "circuit/mna.py · draw_dq(), poisson_or_gauss(); element.py · pmf_at()",
        },
        {
          id: "eq-ckt-ou",
          label: L("가변 스텝 OU (evolving 국소 상태)", "variable-step OU (evolving local states)"),
          tex: r`\delta_{n+1} = a\,\delta_n + \sigma\sqrt{1-a^2}\,\xi,\qquad a = e^{-h/\tau}`,
          note: L(
            "작용점 상태(p[9], p[23], p[19] 또는 p[20])와 emitter 상태(p[10])에 각각 적용; frozen 모드는 실행마다 한 번 추출.",
            "Applied to the action-point state (p[9], p[23], p[19] or p[20]) and to the emitter state (p[10]); frozen mode draws once per run.",
          ),
          code: "circuit/mna.py · run_chunk() (lsmode 2)",
        },
      ],
    },
    {
      heading: L("적응 Δt와 latch 검출", "Adaptive Δt and latch detection"),
      body: L(
        "- **결정론**: $\\mathrm{err}=\\max(|\\Delta u|/\\Delta u_{\\max},\\ |\\Delta\\ln I_D|/\\Delta_{\\ln I},\\ |\\Delta v|/\\Delta v_{\\max},\\ \\mathrm{LTE}/\\mathrm{LTE}_u)$, $\\mathrm{LTE}=\\tfrac12h|F-F_0|/|\\partial Q/\\partial u|$. err > 1.5이면 거부, $h\\leftarrow h\\max(0.1,0.7/\\mathrm{err})$; 다음 스텝 $h\\cdot\\min(2.5,\\max(0.3,0.8/\\mathrm{err}))$. 파형 breakpoint에 맞춘다.\n- **확률**: $\\tau_k=1/|dF_k/dQ_k|$ (회로 제약을 따른 감도). $\\tau_{\\mathrm{frac}}\\tau_{\\min}\\ge h_{\\mathrm{noise,min}}$이면 잡음 분해 모드: $h\\le\\tau_{\\mathrm{frac}}\\tau_{\\min}$, $h\\le\\Delta u_{\\max}|\\partial Q/\\partial u|/|F|$, $h\\le N_{\\mathrm{ev,max}}/\\lambda_{\\mathrm{tot}}$. 아니면 drift-implicit.\n- **latch 검출**: $I_D$가 $I_{th}$ (기본 10 nA)를 위로 지나면 latch-up, 아래로 지나면 latch-down; 시각은 $\\ln I$ 선형 보간, 그때의 $V_{DS}$와 소스 전압을 기록.",
        "- **deterministic**: $\\mathrm{err}=\\max(|\\Delta u|/\\Delta u_{\\max},\\ |\\Delta\\ln I_D|/\\Delta_{\\ln I},\\ |\\Delta v|/\\Delta v_{\\max},\\ \\mathrm{LTE}/\\mathrm{LTE}_u)$, $\\mathrm{LTE}=\\tfrac12h|F-F_0|/|\\partial Q/\\partial u|$. Reject if err > 1.5 with $h\\leftarrow h\\max(0.1,0.7/\\mathrm{err})$; next step $h\\cdot\\min(2.5,\\max(0.3,0.8/\\mathrm{err}))$. Steps land on waveform breakpoints.\n- **stochastic**: $\\tau_k=1/|dF_k/dQ_k|$ (sensitivity along the circuit constraints). If $\\tau_{\\mathrm{frac}}\\tau_{\\min}\\ge h_{\\mathrm{noise,min}}$ the noise-resolved mode applies: $h\\le\\tau_{\\mathrm{frac}}\\tau_{\\min}$, $h\\le\\Delta u_{\\max}|\\partial Q/\\partial u|/|F|$, $h\\le N_{\\mathrm{ev,max}}/\\lambda_{\\mathrm{tot}}$; otherwise drift-implicit.\n- **latch detection**: $I_D$ crossing $I_{th}$ (default 10 nA) upwards is latch-up, downwards latch-down; the time is interpolated linearly in $\\ln I$, and $V_{DS}$ and the source voltage at that time are recorded.",
      ),
      variables: [
        { symbol: r`\Delta u_{\max}`, name: L("스텝당 u 변화 (reltol/1e-3 배, [1 mV, 50 mV])", "u change per step (× reltol/1e-3, [1 mV, 50 mV])"), value: "10", unit: "mV", code: "CF_DUMAX" },
        { symbol: r`\Delta_{\ln I}`, name: L("스텝당 ln I_D 변화", "ln I_D change per step"), value: "0.2", code: "CF_DLNIMAX" },
        { symbol: r`\Delta v_{\max}`, name: L("스텝당 노드 전압 변화", "node-voltage change per step"), value: "20", unit: "mV", code: "CF_DVMAX" },
        { symbol: r`\mathrm{LTE}_u`, name: L("u 국소 절단오차", "local truncation error in u"), value: "1", unit: "mV", code: "CF_LTEU" },
        { symbol: r`\tau_{\mathrm{frac}},\ h_{\mathrm{noise,min}}`, name: L("잡음 분해 스텝 비율, 최소 스텝", "noise-resolved step fraction, minimum step"), value: "0.05, 2 ns", code: "SolverConfig" },
        { symbol: r`N_{\mathrm{ev,max}}`, name: L("스텝당 최대 기대 사건 수; Gaussian 문턱", "max expected events per step; Gaussian threshold"), value: "200; 100", code: "SolverConfig" },
        { symbol: r`h`, name: L("최소/최대/초기 스텝", "min / max / initial step"), value: "1e-15 / 1e-3 / 1e-9", unit: "s", code: "SolverConfig" },
        { symbol: r`I_{th}`, name: L("latch 검출 문턱", "latch detection threshold"), value: "10", unit: "nA", code: "detect.i_threshold_A" },
      ],
      notes: [
        L(
          "10 nA 문턱은 FPT 격자의 흡수 경계와 같은 값이다(→ first-passage).",
          "The 10 nA threshold equals the absorbing boundary of the FPT lattice (→ first-passage).",
        ),
        L(
          "구현 세부: docs/CIRCUIT_SIMULATOR.md 참조 (허용오차 기본값, 기록 간격).",
          "Implementation detail: see docs/CIRCUIT_SIMULATOR.md (default tolerances, recording intervals).",
        ),
      ],
    },
    {
      heading: L("테스트 벤치", "Test benches"),
      body: L(
        "- **load line**: 램프 전원 → 직렬 저항 → STL. $V_{LU}/V_{LD}$와 `branches`의 I–V를 재현하는지 확인.\n- **pulse train**: 진폭·폭·간격을 바꾼 펄스열에서 전환 확률 $P_{\\mathrm{sw}}$; 잔류 $Q_B$가 간격 기억을 만든다.\n- **p-bit**: STL + 부하 저항 + 비교기; V_G·빛에 따른 출력 비트 통계.\n- **coupled pair**: 저항으로 결합한 두 STL (Ising/p-bit 네트워크의 첫 요소).",
        "- **load line**: ramp source → series resistor → STL. Checks that $V_{LU}/V_{LD}$ and the I–V of `branches` are reproduced.\n- **pulse train**: switching probability $P_{\\mathrm{sw}}$ versus pulse amplitude, width and interval; residual $Q_B$ gives interval memory.\n- **p-bit**: STL + load resistor + comparator; output bit statistics versus V_G and light.\n- **coupled pair**: two STLs coupled through a resistor (first Ising/p-bit network element).",
      ),
      notes: [
        L("구현 세부: docs/CIRCUIT_SIMULATOR.md 참조 (벤치 기본값).", "Implementation detail: see docs/CIRCUIT_SIMULATOR.md (bench defaults)."),
      ],
    },
  ],
  related: ["charge-balance", "stochastic-events", "first-passage", "local-states", "open-problems", "numerics"],
  codeRefs: [CODE.circuit + "element.py", CODE.circuit + "mna.py", "engine/docs/CIRCUIT_ELEMENT_DESIGN.md", "docs/CIRCUIT_SIMULATOR.md"],
};

export default topic;
