import type { PhysicsTopic } from "../types";
import { CODE, L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "circuit-element",
  title: L("회로 소자로서의 STL (MNA 과도 해석)", "STL as a circuit element (MNA transient)"),
  summary: L(
    "STL은 바디 전하 $Q_B$를 상태로 갖는 3단자(D, G, S) 비선형 소자다. 내부 미지수 $(u,r)$은 단자 전압식과 전하식을 통해 MNA 연립 방정식에 들어가고, 음적분 BE/TRAP과 Newton 반복으로 적분된다. 확률 모드에서는 셀마다 매 스텝 전에 적분 계층을 고른다: Eq. 2 사건 수준 tau-leap, 분산 보정 가우스, 또는 잡음 없는 드리프트(빠른 완화, 래치된 셀, 잡음 대역 밖). 래치는 별도의 스위치 없이 ODE에서 저절로 나타난다.",
    "The STL is a three-terminal (D, G, S) nonlinear element whose state is the body charge $Q_B$. Its internal unknowns $(u,r)$ enter the MNA system through a terminal-voltage equation and a charge equation and are integrated with implicit BE/TRAP and Newton iteration. In stochastic mode each cell picks an integration tier before every step: event-level Eq. 2 tau-leap, variance-corrected Gaussian, or noise-free drift (fast relaxation, latched cell, outside the noise band). Latching emerges from the ODE without an explicit switch.",
  ),
  tags: ["circuit", "MNA", "transient", "tau-leap"],
  sections: [
    {
      heading: L("소자 방정식", "Element equations"),
      body: L(
        "단자는 D, G, S이다(선택적으로 광 입력 $I_{\\mathrm{PH}}(t)$ → p[13]). 소자마다 미지수 두 개 $(u_k,r_k)$와 식 두 개(E1, E2)가 추가된다. $V_{GS}$는 평가할 때마다 p[11]에 들어간다(모든 벤치에서 DC). 전하 좌표는 복합 FPT 격자와 같다(상수 $-C_{ox}V_{GS}$만큼 차이). $V_{GS}$가 시간에 따라 바뀌면 같은 $Q$에서 $\\psi$가 따라 움직인다(게이트–바디 용량 결합).",
        "Terminals D, G, S (plus an optional light input $I_{\\mathrm{PH}}(t)$ → p[13]). Each element adds two unknowns $(u_k,r_k)$ and two equations (E1, E2). $V_{GS}$ is written to p[11] at every evaluation (DC in every bench). The charge coordinate is that of the compound-FPT lattice (up to the constant $-C_{ox}V_{GS}$). When $V_{GS}$ changes in time, $\\psi$ follows at fixed $Q$ (gate–body capacitive coupling).",
      ),
      equations: [
        {
          id: "eq-ckt-charge",
          label: L("바디 전하 좌표", "Body-charge coordinate"),
          tex: r`\begin{aligned} &Q(u,r;V_{GS}) = C_{ox}\big(\psi(u) - V_{GS}\big) + Q_{\mathrm{exc}}(u,r) + qN_A A\,L_n(u,r)\\ &\psi = u - V_T\ln\!\big(1+\delta/N_A\big)\end{aligned}`,
          note: L(
            "$Q_{\\mathrm{exc}}=$ `z[13]` $-C_{ox}u$, $L_n=$ `z[11]`. 주의: `data/tables`의 `Q_B_C`와 MODEL_SPEC §1의 $Q_B$는 $Q_{\\mathrm{exc}}$만을 가리킨다.",
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
          label: L("E2: 전하식 (C_ox로 정규화)", "E2: charge equation (normalized by C_ox)"),
          tex: r`\frac{Q(u,r;V_{GS}) - Q_c - \theta\,h\,F(u,r)}{C_{ox}} = 0,\qquad Q_{n+1} = Q_c + \theta\,h\,F_{n+1}`,
          note: L(
            "- 결정론 BE: $Q_c=Q_n$, $\\theta=1$. TRAP: $Q_c=Q_n+\\tfrac h2F_n$, $\\theta=\\tfrac12$ (모든 셀에서 $h<2\\tau_{\\mathrm{rel}}$일 때만 쓰고, 아니면 BE).\n- 확률, 계층 1 (사건 수준 tau-leap): $Q_c=Q_n+\\Delta Q_{\\mathrm{ev}}$, $\\theta=0$ (양적분).\n- 확률, 계층 2 (가우스): $Q_c=Q_n+\\eta$, $\\theta=1$ (드리프트는 음적분).\n- 확률, 계층 3–5 (드리프트만): $Q_c=Q_n$, $\\theta=1$ (BE, 잡음 없음).\n\n캐리어 잡음을 켠 실행에서는 TRAP을 쓰지 않으며, 커패시터도 BE 등가(companion) 모델을 쓴다. 저장하는 전하는 적분 공식의 값 $Q_c+\\theta hF_{n+1}$이므로 사건 전하의 장부가 정확히 맞는다.",
            "- Deterministic BE: $Q_c=Q_n$, $\\theta=1$. TRAP: $Q_c=Q_n+\\tfrac h2F_n$, $\\theta=\\tfrac12$ (only when $h<2\\tau_{\\mathrm{rel}}$ for every cell, BE otherwise).\n- Stochastic tier 1 (event-level tau-leap): $Q_c=Q_n+\\Delta Q_{\\mathrm{ev}}$, $\\theta=0$ (explicit).\n- Stochastic tier 2 (Gaussian): $Q_c=Q_n+\\eta$, $\\theta=1$ (drift-implicit).\n- Stochastic tiers 3–5 (drift only): $Q_c=Q_n$, $\\theta=1$ (BE, no noise).\n\nRuns with carrier noise never use TRAP, and their capacitors use the BE companion model. The stored charge is the value of the integration formula, $Q_c+\\theta hF_{n+1}$, so the event bookkeeping stays exact.",
          ),
          code: "circuit/mna.py · assemble() row kr; run_chunk()",
        },
      ],
      variables: [
        { symbol: r`I_D`, name: L("드레인 전류: D 노드에서 나와 S 노드로 들어감", "Drain current: leaves node D, enters node S"), unit: "A", code: "z[1]" },
        { symbol: r`F`, name: L("순 정공 전류 G − L", "Net hole current G − L"), unit: "A", code: "z[2]" },
        { symbol: r`C_{ox}`, name: L("E2 정규화 용량", "E2 normalization capacitance"), value: "0.2449", unit: "fF", code: "COX_F" },
      ],
      notes: [
        L(
          "엔진 영역 밖으로의 연속화: $u<0$(빠른 하강 램프 뒤 소스 접합이 역방향 바이어스된 경우)에서는 $u=0$의 전류에 $X'(0^+)V_T(e^{u/V_T}-1)$을 이어 붙인다. $r<0$(드레인 접합이 순방향 바이어스된 경우)에서는 소스와 같은 포화 전류를 갖는 대칭 순방향 드레인 다이오드와 공핍 SRH를 손실로 더한다. 엔진 자체는 이 영역을 평가하지 않는다(결과의 `warnings`에 보고).",
          "Continuation outside the engine domain: for $u<0$ (source junction reverse-biased after fast down-ramps) the $u=0$ currents are extended by $X'(0^+)V_T(e^{u/V_T}-1)$; for $r<0$ (drain junction forward-biased) a symmetric forward drain diode with the source saturation current plus depletion SRH is added as a loss. The engine itself never evaluates these regions (reported in the result `warnings`).",
        ),
        L(
          "단자 전류는 준정적 $I_D$이다. $dQ_B/dt$에 따른 변위 전류(게이트–바디, 공핍 전하)는 단자에 넣지 않으며, 게이트에는 전류가 흐르지 않는다.",
          "The terminal currents are the quasi-static $I_D$: the displacement currents associated with $dQ_B/dt$ (gate–body, depletion charges) are not stamped into the terminals, and the gate draws no current.",
        ),
      ],
    },
    {
      heading: L("MNA 조립과 Newton 반복", "MNA assembly and Newton iteration"),
      body: L(
        "미지수는 $x=[v_1..v_{N-1},\\ i_V,\\ (u_k,r_k)]$이다. 방정식은 접지가 아닌 각 노드의 KCL(나가는 전류의 합 = 0), 전압원 행 $v_a-v_b-V(t)=0$, STL의 E1/E2 행으로 이루어진다. 모든 노드에 $G_{\\min}=10^{-18}$ S를 둔다. STL 블록의 Jacobian은 `stl_eval`의 전진 유한 차분($10^{-6}$ V; 영역 밖에서는 후진 차분)으로 구한다. 반복마다 $|\\Delta u|\\le50$ mV, $|\\Delta r|\\le1$ V로 제한하고, 영역을 벗어나면 스텝을 반으로 줄이는 백트래킹을 한다(최대 14회). 수렴 조건은 $|\\Delta u|<10^{-7}$ V, $|\\Delta r|<10^{-6}$ V, $|\\Delta v|<10^{-6}(1+|v|)$ V, $|E_1|,|E_2|<10^{-5}$ V이고, 반복은 최대 14회다(계층 1–2에서는 허용 오차를 20배 완화).",
        "Unknowns: $x=[v_1..v_{N-1},\\ i_V,\\ (u_k,r_k)]$. The equations are the KCL at each non-ground node (sum of leaving currents = 0), the source rows $v_a-v_b-V(t)=0$ and the STL rows E1/E2. $G_{\\min}=10^{-18}$ S is placed at every node. The STL Jacobian block is a forward finite difference of `stl_eval` ($10^{-6}$ V; backward outside the domain). Per iteration $|\\Delta u|\\le50$ mV and $|\\Delta r|\\le1$ V, with step-halving backtracking (at most 14×) when an iterate leaves the domain. Convergence: $|\\Delta u|<10^{-7}$ V, $|\\Delta r|<10^{-6}$ V, $|\\Delta v|<10^{-6}(1+|v|)$ V, $|E_1|,|E_2|<10^{-5}$ V, at most 14 iterations (tolerances relaxed 20× in tiers 1–2).",
      ),
      equations: [
        {
          id: "eq-ckt-companion",
          label: L("커패시터 등가(companion) 모델", "Capacitor companion models"),
          tex: r`\text{BE: } i^{n+1} = \frac{C}{h}\big(v^{n+1}-v^{n}\big),\qquad \text{TRAP: } i^{n+1} = \frac{2C}{h}\big(v^{n+1}-v^{n}\big) - i^{n}`,
          code: "circuit/mna.py · cap_companion()",
        },
        {
          id: "eq-ckt-kcl",
          label: L("KCL 스탬프", "KCL stamps"),
          tex: r`\sum_{R}\frac{v_a-v_b}{R} + \sum_{C} i_C + \sum_{V} i_V + \sum_{I} I(t) \pm \sum_{\mathrm{STL}} I_D(u,r) + G_{\min}v = 0`,
          code: "circuit/mna.py · assemble()",
        },
        {
          id: "eq-ckt-tau",
          label: L("회로 제약을 따른 국소 완화 시간", "Local relaxation time along the circuit constraints"),
          tex: r`J_{\theta=0}\,s_k = \frac{e_{E2,k}}{C_{ox}},\qquad \frac{dF_k}{dQ_k} = \frac{\partial F_k}{\partial u}s_{u,k} + \frac{\partial F_k}{\partial r}s_{r,k},\qquad \tau_{\mathrm{rel},k} = \frac{1}{|dF_k/dQ_k|}`,
          note: L(
            "매 스텝 뒤에 전하를 고정한 Jacobian으로 $dx/dQ_k$를 구한다($R_s$, $C_d$ 등가 모델, 다른 셀 포함). 전형적인 값: fold 근처의 HRS는 수 µs–수백 µs(fold에서 임계 감속), 암조건의 저 $V_D$ HRS는 ms–0.1 s, LRS는 0.03–50 ns이다(LD fold에서는 µs 이상으로 커진다).",
            "After every step the charge-fixed Jacobian gives $dx/dQ_k$ (including $R_s$, the $C_d$ companion and the other cells). Typical values: a few µs to hundreds of µs in the HRS near the fold (critical slowing down at the fold), ms to 0.1 s in the dark HRS at low $V_D$, and 0.03–50 ns in the LRS (rising to µs and beyond at the LD fold).",
          ),
          code: "circuit/mna.py · sensitivities()",
        },
      ],
      notes: [
        L(
          "초기 동작점: $u=0$(빈 바디)으로 고정해 풀고(커패시터 개방), 의사 과도(pseudo-transient) BE($h$는 $10^{-12}$ s에서 시작해 스텝마다 ×3, 최대 $10^8$ s)로 빈 바디에서 도달하는 저전류 상태를 찾는다(히스테리시스 창 안에서는 HRS). 초기 래치 상태는 바디가 어느 branch에 있는지로 정한다($u \\ge u_j$, 아래 래치 검출 참조). 사용자 회로(`tran.initial` = auto)에서는 이 동작점이 셀을 음저항 branch($u_i<u<u_j$) 위에 두거나(전류원 등으로 전류 바이어스된 셀의 불안정 평형점) 전류 바이어스가 HRS를 넘으면, 커패시터를 방전된 상태로 두고 시작한다(전원이 $t=0$에 켜진 것처럼; SPICE UIC). 그래서 전류 구동 STL은 첫 주기부터 이완 발진한다.",
          "Initial operating point: solve with $u=0$ (empty body, capacitors open), then run a pseudo-transient BE ($h$ from $10^{-12}$ s, ×3 per step, up to $10^8$ s) to the low-current state reachable from an empty body (the HRS inside the hysteresis window). The initial latch state is given by the body's branch ($u \\ge u_j$; see latch detection below). In user-drawn circuits (`tran.initial` = auto), when this operating point puts a cell on the negative-resistance branch ($u_i<u<u_j$: the unstable equilibrium of a current-biased cell) or the current bias exceeds what the HRS can carry, the run starts from discharged capacitors instead (as if the sources were switched on at $t=0$; SPICE UIC), so a current-driven STL relaxation-oscillates from the first cycle.",
        ),
      ],
    },
    {
      heading: L("확률 모드: 적분 계층", "Stochastic mode: integration tiers"),
      body: L(
        "각 셀은 매 스텝 전에 상태 $n$에서 계층을 고른다(아래 순서로 검사).\n\n- **4 드리프트만**: 셀이 래치 상태이고(바디가 LRS branch에 있음, 전류 문턱과 무관; 아래 래치 검출 참조) `ld_carrier_noise` = false(회로 기본값)인 경우.\n- **5 드리프트만**: $V_{DS}$가 그 셀의 잡음 대역 밖에 있고(아래), 구동 파형이 $4\\tau_{\\mathrm{rel}}$ 안에 대역으로 들어오지 않는 경우(선행 감지, look-ahead).\n- **3 드리프트만**: $\\tau_{\\mathrm{rel}} < 2$ ns (`gauss_tau_min`)이고 $\\tau_{\\mathrm{frac}}\\tau_{\\mathrm{rel}} < h_{\\mathrm{noise,min}}$인 경우.\n- **1 사건 수준 tau-leap**: $\\tau_{\\mathrm{frac}}\\tau_{\\mathrm{rel}} \\ge h_{\\mathrm{noise,min}}$인 경우(기본값에서는 $\\tau_{\\mathrm{rel}} \\ge 40$ ns).\n- **2 가우스**: 나머지, 즉 $2\\ \\mathrm{ns} \\le \\tau_{\\mathrm{rel}} < 40$ ns인 경우(예: 래치다운 fold 근처의 LRS).\n\n`carrier_noise` = false이면 결정론 적분에 국소 상태만 더한다(국소 상태도 없으면 난수 입력이 없으므로 한 번만 실행하고 경고한다). 계층 1은 스텝 시작 상태의 사건률로 $\\Delta Q_{\\mathrm{ev}}$를 뽑는다. 단위 사건 $N_\\uparrow$와 손실 $N_\\downarrow$는 Poisson 분포를 따르고, II는 Poisson 분포를 따르는 개수의 클러스터로 이루어지며 각 크기는 $k\\sim p_k(r)/P_1$이다(현재 $r$에서 선형 보간한 확장 애벌랜치 커널). 평균이 100(`gauss_threshold`)을 넘는 개수에는 가우스 극한을 쓴다. 연속화 영역의 음의 전류는 반대 방향의 사건으로 넣으므로 $\\mathbb E[\\Delta Q]=hF$가 성립한다.",
        "Each cell picks its tier from the state $n$ before every step (checked in this order):\n\n- **4 drift only**: the cell is latched (its body is on the LRS branch, independent of the current thresholds; see latch detection) and `ld_carrier_noise` = false (circuit default).\n- **5 drift only**: $V_{DS}$ lies outside the cell's noise band (below), and the drive does not enter the band within $4\\tau_{\\mathrm{rel}}$ (look-ahead).\n- **3 drift only**: $\\tau_{\\mathrm{rel}} < 2$ ns (`gauss_tau_min`) and $\\tau_{\\mathrm{frac}}\\tau_{\\mathrm{rel}} < h_{\\mathrm{noise,min}}$.\n- **1 event-level tau-leap**: $\\tau_{\\mathrm{frac}}\\tau_{\\mathrm{rel}} \\ge h_{\\mathrm{noise,min}}$ ($\\tau_{\\mathrm{rel}} \\ge 40$ ns with the defaults).\n- **2 Gaussian**: the rest, i.e. $2\\ \\mathrm{ns} \\le \\tau_{\\mathrm{rel}} < 40$ ns (e.g. the LRS near the latch-down fold).\n\nWith `carrier_noise` = false the integration is deterministic plus the local states (without local states there is no random input, so the run is done once, with a warning). Tier 1 draws $\\Delta Q_{\\mathrm{ev}}$ from the rates at the start of the step: unit events $N_\\uparrow$ and losses $N_\\downarrow$ are Poisson, and II is a Poisson number of clusters with sizes $k\\sim p_k(r)/P_1$ (extended avalanche kernel, linearly interpolated at the current $r$). Counts whose mean exceeds 100 (`gauss_threshold`) use the Gaussian limit. Negative currents in the continuation regions are folded into the opposite event class, so $\\mathbb E[\\Delta Q]=hF$ holds.",
      ),
      equations: [
        {
          id: "eq-ckt-dq",
          label: L("계층 1: 사건 증분", "Tier 1: event increment"),
          tex: r`\begin{aligned} &\Delta Q_{\mathrm{ev}} = q\Big(N_\uparrow + \sum_{c=1}^{N_c} k_c - N_\downarrow\Big)\\ &N_\uparrow\sim\mathcal P\big(\lambda_{\mathrm{unit}}h\big),\quad N_\downarrow\sim\mathcal P(Lh),\quad N_c\sim\mathcal P\Big(\lambda_{\mathrm{II}}h\,\frac{P_1}{M_1}\Big)\end{aligned}`,
          note: L(
            "$P_1=\\sum_{k\\ge1}p_k$, $M_1=\\sum k p_k$, $M_2=\\sum k^2p_k$이다. 크기 $k$의 점프율 $\\lambda_{\\mathrm{II}}p_k/M_1$은 `compound_fpt.backward`와 같다. 가우스 극한에서는 $\\sum k_c\\approx\\lambda_{\\mathrm{II}}h+\\sqrt{\\lambda_{\\mathrm{II}}hM_2/M_1}\\,\\xi$, Poisson 개수는 $\\lambda+\\sqrt\\lambda\\,\\xi$로 둔다(모두 0에서 자름).",
            "$P_1=\\sum_{k\\ge1}p_k$, $M_1=\\sum k p_k$, $M_2=\\sum k^2p_k$; the jump rate for size $k$, $\\lambda_{\\mathrm{II}}p_k/M_1$, is that of `compound_fpt.backward`. Gaussian limit: $\\sum k_c\\approx\\lambda_{\\mathrm{II}}h+\\sqrt{\\lambda_{\\mathrm{II}}hM_2/M_1}\\,\\xi$, and Poisson counts $\\lambda+\\sqrt\\lambda\\,\\xi$ (all clipped at 0).",
          ),
          code: "circuit/mna.py · draw_dq(), poisson_or_gauss(); element.py · pmf_at()",
        },
        {
          id: "eq-ckt-gauss",
          label: L("계층 2: 분산 보정 가우스", "Tier 2: variance-corrected Gaussian"),
          tex: r`\begin{aligned} &\eta\sim\mathcal N\Big(0,\ D\,h\big(1 + \tfrac{h}{2\tau_{\mathrm{rel}}}\big)\Big),\qquad D = q^2\Big(R_\uparrow + R_\downarrow + \lambda_{\mathrm{II}}\frac{M_2}{M_1}\Big)\\ &\delta_{n+1} = \frac{\delta_n + \eta}{1 + h/\tau_{\mathrm{rel}}}\ \Rightarrow\ \mathrm{Var}[\delta] = \frac{\mathrm{Var}[\eta]}{(1+h/\tau_{\mathrm{rel}})^2 - 1} = \frac{D\,\tau_{\mathrm{rel}}}{2}\end{aligned}`,
          note: L(
            "선형화한 음적분 드리프트에서는 어떤 $h$에서도 정상 분산이 정확한 OU 분산 $D\\tau_{\\mathrm{rel}}/2$와 같아진다. 다만 탈출 동역학은 $h\\lesssim\\tau_{\\mathrm{rel}}/2$에서만 분해되므로 $h\\le0.5\\,\\tau_{\\mathrm{rel}}$로 제한한다(`gauss_tau_frac`).",
            "For the linearized implicit drift the stationary variance equals the exact OU variance $D\\tau_{\\mathrm{rel}}/2$ for any $h$. The escape dynamics, however, are resolved only for $h\\lesssim\\tau_{\\mathrm{rel}}/2$, hence $h\\le0.5\\,\\tau_{\\mathrm{rel}}$ (`gauss_tau_frac`).",
          ),
          code: "circuit/mna.py · run_chunk() (reg 2), noise_var_rate()",
        },
        {
          id: "eq-ckt-band",
          label: L("잡음 대역 (계층 5의 경계)", "Noise bands (boundary of tier 5)"),
          tex: r`\begin{aligned} &z(V_D) = \frac{|Q - Q_{\mathrm{saddle}}(V_D)|}{\sqrt{D\,\tau_{\mathrm{rel}}/2}}\quad\text{(HRS and LRS branches, 301-point grid)}\\ &\text{unlatched: } \big[\min V_{\mathrm{HRS}}(z<z_{\max}) - e_{LU},\ \infty\big)\\ &\text{latched: } \big(-\infty,\ \max V_{\mathrm{LRS}}(z<z_{\max}) + e_{LD}\big]\\ &e_{LU} = 4(\sigma + 5\sigma_E),\qquad e_{LD} = 4(0.05\,\sigma + 50\,\sigma_E)\end{aligned}`,
          note: L(
            "$z_{\\max}=12$(`noise_z_max`)는 정확한 복합 점프 hazard에 맞춰 보정한 값으로, 대역 밖의 탈출률은 ≲ $10^{-3}$ s⁻¹이다. 대역 밖에서는 드리프트만 적분한다(수 mV의 정상 요동은 계산하지 않음). fold 너머 쪽에는 경계가 없다. 래치되지 않은 셀이 $V_{LU}$ 위에 있으면 fold 이후의 통과 과정에 있는 것이고, 그 잡음이 빠른 램프에서의 $V_{LU}$와 fold 위 펄스의 지연을 정하기 때문이다. 구동이 대역에 들어가기 $4\\tau_{\\mathrm{rel}}$ 전부터 잡음을 켜므로(선행 감지), 빠른 램프나 에지에서도 정상 요동이 미리 쌓인다. 확장 폭 $e$는 GIDL 작용점에서 국소 상태에 의한 fold 이동의 4σ이다. 다른 작용점이거나 $\\sigma_E>2$ mV이면 대역 제한을 두지 않고, 래치 창이 없는 소자는 국소 상태가 꺼져 있으면 잡음 없이(드리프트만) 적분한다. 예(국소 상태 없음): 기준 보정 −2 V 암조건에서 HRS 3.59 V 이상, LRS 2.82 V 이하; 광조사 보정 프리셋 HRS 3.70 V 이상; −1.8 V·2.63 pA HRS 2.93 V 이상. 이 대역 덕분에 기준 보정의 0.4 V/s 램프도 사건 수준으로 계산할 수 있다(사이클당 약 2×10⁵ 스텝).",
            "$z_{\\max}=12$ (`noise_z_max`) was calibrated against the exact compound-jump hazard: the escape rate outside the bands is ≲ $10^{-3}$ s⁻¹. Outside the bands only the drift is integrated (the stationary fluctuation of a few mV is not simulated). There is no edge beyond the fold: an unlatched cell above $V_{LU}$ is in the post-fold passage, whose noise sets $V_{LU}$ on fast ramps and the delay of supra-fold pulses. The noise is switched on $4\\tau_{\\mathrm{rel}}$ before the drive enters a band (look-ahead), so the stationary fluctuation is built up on fast ramps and edges as well. The widening $e$ is 4σ of the local-state fold shifts for the GIDL action point; other action points or $\\sigma_E>2$ mV get no band restriction, and a device without a latch window is integrated drift-only unless local states are on. Examples (no local states): reference calibration, −2 V dark: HRS from 3.59 V, LRS up to 2.82 V; illumination-calibration preset: HRS from 3.70 V; −1.8 V, 2.63 pA: HRS from 2.93 V. The bands make event-level runs of the reference calibration at 0.4 V/s affordable (about 2×10⁵ steps per cycle).",
          ),
          code: "circuit/stochastic.py · branch_profile(), noise_bands()",
        },
        {
          id: "eq-ckt-ou",
          label: L("가변 스텝 OU (진화하는 국소 상태)", "Variable-step OU (evolving local states)"),
          tex: r`\delta_{n+1} = a\,\delta_n + \sigma\sqrt{1-a^2}\,\xi,\qquad a = e^{-h/\tau}`,
          note: L(
            "작용점 상태(p[9], p[23], p[19] 또는 p[20])와 이미터 상태(p[10])에 각각 적용한다. 초기값은 실행마다 $\\mathcal N(0,\\sigma^2)$에서 뽑는다(고정 모드는 실행 내내 유지). 획득 추세는 회로에 적용하지 않는다. 실행 $r$의 난수: 사건은 seed + 1000003 r, 국소 상태 추출은 `default_rng([seed, 7349, r])`을 쓴다.",
            "Applied to the action-point state (p[9], p[23], p[19] or p[20]) and to the emitter state (p[10]); the initial value is drawn from $\\mathcal N(0,\\sigma^2)$ for each run (frozen: kept for the whole run). The acquisition trend is not applied in the circuit. Random numbers of run $r$: events use seed + 1000003 r, the local-state draws `default_rng([seed, 7349, r])`.",
          ),
          code: "circuit/mna.py · run_chunk() (lsmode 2); circuit/stochastic.py · draw_local_states()",
        },
      ],
      notes: [
        L(
          "대역 밖에서 큰 스텝의 가우스 적분을 쓰지 않는 이유: 쌍안정 구간에서 $h\\gg\\tau_{\\mathrm{rel}}$이면 음적분식이 약 $h/\\tau$ SD 크기의 충격을 받아 Newton 반복이 다른 basin의 근에 떨어질 수 있다(시험에서 $V_{LD}$가 2.656 V에서 2.84 V로 바뀜). 드리프트만 적분하면 평균은 정확하다.",
          "Why there is no large-step Gaussian outside the bands: with $h\\gg\\tau_{\\mathrm{rel}}$ in a bistable region, the implicit equation receives kicks of about $h/\\tau$ SDs and Newton can land on the root of the other basin (in a test, $V_{LD}$ moved from 2.656 to 2.84 V). Drift only is exact in the mean.",
        ),
        L(
          "회로의 `ld_carrier_noise` 기본값은 false다(계층 4: LRS는 드리프트만 적분하고, $V_{LD}$는 LRS fold에서의 결정론적 탈출에 국소 상태의 효과를 더한 값). 소자 수준 스윕 MC의 프리셋은 true를 쓴다(`sweep-mc`). 회로에서 이를 켜면 LRS에도 계층 1/2가 적용되어, 잡음에 의한 이른 래치다운이 $V_{LD}$를 높인다(1200 V/s, −1.8 V·2.63 pA: 2.593 V → 2.657 V, 계산 비용 약 20배).",
          "The circuit's `ld_carrier_noise` defaults to false (tier 4: the LRS is integrated drift-only, and $V_{LD}$ is the deterministic escape at the LRS fold plus local-state effects); the device-level sweep-MC presets use true (`sweep-mc`). When it is turned on in the circuit, tiers 1/2 also apply in the LRS, and noise-induced early latch-down raises $V_{LD}$ (1200 V/s, −1.8 V, 2.63 pA: 2.593 → 2.657 V at about 20× the cost).",
        ),
      ],
    },
    {
      heading: L("적응 Δt와 래치 검출", "Adaptive Δt and latch detection"),
      body: L(
        "- **결정론 (그리고 확률 모드의 드리프트 계층 3–5)**: $\\mathrm{err}=\\max(|\\Delta u|/\\Delta u_{\\max},\\ |\\Delta\\ln I_D|/\\Delta_{\\ln I},\\ |\\Delta v|/\\Delta v_{\\max},\\ \\mathrm{LTE}/\\mathrm{LTE}_u)$, $\\mathrm{LTE}=\\tfrac12h|F-F_0|/|\\partial Q/\\partial u|$. err > 1.5이면 스텝을 거부하고 $h\\leftarrow h\\max(0.1,0.7/\\mathrm{err})$로 줄인다. 다음 스텝은 $h\\cdot\\min(2.5,\\max(0.3,0.8/\\mathrm{err}))$이다. Newton 실패 시 $h/4$. 스텝은 파형의 꺾임점과 샘플 시각에 정확히 맞춘다.\n- **계층 1**: $h\\le\\tau_{\\mathrm{frac}}\\tau_{\\mathrm{rel}}$, $h\\le N_{\\mathrm{ev,max}}/\\lambda_{\\mathrm{tot}}$, $h\\le\\Delta u_{\\max}|\\partial Q/\\partial u|/|F|$. **계층 2**: $h\\le0.5\\,\\tau_{\\mathrm{rel}}$와 같은 드리프트 한계. 스텝은 모든 셀의 최솟값이다. 잡음에 의한 움직임으로는 스텝을 거부하지 않는다(뽑힌 결과를 보고 거부하면 통계가 편향된다). 예외는 $|\\Delta u| > 20\\Delta u_{\\max}$ = 0.2 V라는 강제 한계뿐이다.\n- **래치 검출**: 래치 상태는 바디가 어느 branch에 있는지로 정한다. 준정적 branch는 $u$로 매개화되며 HRS는 $u<u_i$(LU fold의 $u$), 불안정 branch는 $u_i<u<u_j$, LRS는 $u>u_j$(LD fold의 $u$)이다. 따라서 셀은 $u$가 $u_j$에 닿으면 래치되고, $u_i$까지 내려오면 래치가 풀린다(히스테리시스 = 불안정 branch의 범위; 래치 창이 없으면 래치되지 않음). 전환 시각은 전환 과도 안에서의 전류 문턱 교차로 잰다: 래치업은 $I_D \\ge I_{th}$ = 10 nA이면서 $u\\ge u_i$, 래치다운은 $I_D < I_{th}/\\mathrm{hysteresis}$ = 1 nA이면서 $u\\le u_j$인 시점이다(스텝 안에서 $\\ln I$로 선형 보간하고, 그 시각의 $V_{DS}$와 전원 전압을 기록). 과도 안에 교차가 없으면 바디가 새 branch에 닿은 시각을 쓴다. 바디가 HRS에 머무는 동안의 $I_D$ 문턱 교차(채널/HRS 전도, 예: $V_G$가 채널 문턱 위일 때)는 래치로 세지 않고 경고한다. 사이클마다 $V_{LU}$는 첫 래치업, $V_{LD}$는 그 뒤의 첫 래치다운이다. 실행이 중간에 잘려 도달하지 못한 사이클·펄스·클록은 통계에서 제외한다(중도절단).",
        "- **Deterministic (and the drift tiers 3–5 of stochastic runs)**: $\\mathrm{err}=\\max(|\\Delta u|/\\Delta u_{\\max},\\ |\\Delta\\ln I_D|/\\Delta_{\\ln I},\\ |\\Delta v|/\\Delta v_{\\max},\\ \\mathrm{LTE}/\\mathrm{LTE}_u)$, $\\mathrm{LTE}=\\tfrac12h|F-F_0|/|\\partial Q/\\partial u|$. A step with err > 1.5 is rejected with $h\\leftarrow h\\max(0.1,0.7/\\mathrm{err})$; the next step is $h\\cdot\\min(2.5,\\max(0.3,0.8/\\mathrm{err}))$. Newton failure → $h/4$. Steps land exactly on waveform breakpoints and sample times.\n- **Tier 1**: $h\\le\\tau_{\\mathrm{frac}}\\tau_{\\mathrm{rel}}$, $h\\le N_{\\mathrm{ev,max}}/\\lambda_{\\mathrm{tot}}$, $h\\le\\Delta u_{\\max}|\\partial Q/\\partial u|/|F|$. **Tier 2**: $h\\le0.5\\,\\tau_{\\mathrm{rel}}$ and the same drift limit. The step is the minimum over all cells. Noise-driven moves are not step-rejected (rejecting on the drawn outcome would bias the statistics), except for a hard guard at $|\\Delta u| > 20\\Delta u_{\\max}$ = 0.2 V.\n- **Latch detection**: the latch state is the body's branch. The quasi-static branch is parameterized by $u$: HRS for $u<u_i$ (the $u$ of the LU fold), the unstable branch for $u_i<u<u_j$ and LRS for $u>u_j$ (the $u$ of the LD fold). A cell therefore latches when $u$ reaches $u_j$ and unlatches when it falls back to $u_i$ (hysteresis = the range of the unstable branch; without a latch window a cell never latches). The switch is timed at the current-threshold crossing inside the switching transient: latch-up at $I_D \\ge I_{th}$ = 10 nA with $u\\ge u_i$, latch-down at $I_D < I_{th}/\\mathrm{hysteresis}$ = 1 nA with $u\\le u_j$ (linear in $\\ln I$ within the step; $V_{DS}$ and the supply voltage at that time are recorded), or, without such a crossing, when the body reaches the new branch. $I_D$ threshold crossings while the body stays on the HRS (channel/HRS conduction, e.g. with $V_G$ above the channel threshold) are not counted as latch events and are reported as a warning. In each cycle $V_{LU}$ is the first latch-up and $V_{LD}$ the first latch-down after it; cycles, pulses and clocks that a truncated run did not reach are censored.",
      ),
      equations: [
        {
          id: "eq-ckt-step",
          label: L("스텝 한계와 초기 스텝 (자동값)", "Step bounds and initial step (automatic values)"),
          tex: r`h_{\min} = \max\big(10^{-15}\,\mathrm{s},\ 10^{-13}\,t_{\mathrm{end}}\big),\qquad h_{\max} = \frac{t_{\mathrm{end}}}{2000},\qquad h_0 = \min\!\big(h_{\max},\ \max(10\,h_{\min},\ 10^{-7}\,t_{\mathrm{end}})\big)`,
          note: L(
            "`solver.dt_min_s`/`dt_max_s`를 지정하면 그 값이 자동값을 대신한다. 예: 기준 보정의 부하선 0 → 4 → 0 V, 0.4 V/s($t_{\\mathrm{end}}$ = 20 s)에서 $h_{\\min}$ = 2 ps, $h_{\\max}$ = 10 ms, $h_0$ = 2 µs.",
            "`solver.dt_min_s`/`dt_max_s` override the automatic values. Example: the reference-calibration load line 0 → 4 → 0 V at 0.4 V/s ($t_{\\mathrm{end}}$ = 20 s) gives $h_{\\min}$ = 2 ps, $h_{\\max}$ = 10 ms, $h_0$ = 2 µs.",
          ),
          code: "circuit/runner.py · _solver(), SolverConfig(h_init=…)",
        },
      ],
      variables: [
        { symbol: r`\Delta u_{\max}`, name: L("스텝당 u 변화 (reltol/1e-3 배, [1 mV, 50 mV])", "u change per step (× reltol/1e-3, [1 mV, 50 mV])"), value: "10", unit: "mV", code: "CF_DUMAX" },
        { symbol: r`\Delta_{\ln I}`, name: L("스텝당 ln I_D 변화", "ln I_D change per step"), value: "0.2", code: "CF_DLNIMAX" },
        { symbol: r`\Delta v_{\max}`, name: L("스텝당 노드 전압 변화", "Node-voltage change per step"), value: "20", unit: "mV", code: "CF_DVMAX" },
        { symbol: r`\mathrm{LTE}_u`, name: L("u의 국소 절단 오차 (램프 지연을 약 1 %까지 수렴시킴)", "Local truncation error in u (converges the ramp lag to about 1 %)"), value: "0.03", unit: "mV", code: "CF_LTEU" },
        { symbol: r`\tau_{\mathrm{frac}},\ h_{\mathrm{noise,min}}`, name: L("계층 1 스텝 비율, 계층 1 최소 스텝", "Tier-1 step fraction, tier-1 minimum step"), value: "0.05, 2 ns", code: "solver.tau_frac, noise_dt_min_s" },
        { symbol: r`\tau_{G,\min},\ f_G`, name: L("계층 2의 τ_rel 하한, 계층 2 스텝 비율", "Tier-2 lower τ_rel bound, tier-2 step fraction"), value: "2 ns, 0.5", code: "solver.gauss_tau_min_s, gauss_tau_frac" },
        { symbol: r`N_{\mathrm{ev,max}}`, name: L("스텝당 최대 기대 사건 수; 가우스 문턱", "Max expected events per step; Gaussian threshold"), value: "200; 100", code: "solver.max_events_per_step, gauss_threshold" },
        { symbol: r`z_{\max}`, name: L("잡음 대역의 장벽 한계 (fold 너머 쪽은 경계 없음)", "Noise-band barrier limit (no edge beyond the fold)"), value: "12", code: "solver.noise_z_max" },
        { symbol: r`I_{th}`, name: L("래치업 시각 문턱", "Latch-up timing threshold"), value: "10", unit: "nA", code: "detect.i_threshold_A" },
        { symbol: r`I_{th}/\mathrm{hyst}`, name: L("래치다운 시각 문턱 (hysteresis 10)", "Latch-down timing threshold (hysteresis 10)"), value: "1", unit: "nA", code: "detect.hysteresis" },
      ],
      notes: [
        L(
          "10 nA 래치업 문턱은 FPT 격자의 흡수 경계와 같다(→ first-passage). LD fold의 LRS 전류가 약 16 nA($V_G=-2$ V에서 16.7 nA)에 불과하므로 래치다운 시각은 더 낮은 1 nA에서 잰다. 문턱은 사건 시각만 정하며 래치 상태나 잡음 계층은 바꾸지 않는다. 문턱이 해당 fold의 branch 전류(HRS 약 15 pA, LRS 약 17 nA) 사이에 있지 않으면 경고한다.",
          "The 10 nA latch-up threshold equals the absorbing boundary of the FPT lattice (→ first-passage). The LRS current at the LD fold is only about 16 nA (16.7 nA at $V_G=-2$ V), so latch-down is timed at the lower threshold of 1 nA. The thresholds only time the events; they change neither the latch state nor the noise tiers, and a warning is given when a threshold does not lie between the branch currents at its fold (HRS about 15 pA, LRS about 17 nA).",
        ),
        L(
          "실행 가능성: 실행 전에 준정적 branch의 $\\tau_{\\mathrm{rel}}(V_D)$와 사건률, 위의 스텝 규칙으로 실행당 스텝 수를 추정한다. 잡음에 의한 fold 이전 탈출(경험적 hazard)도 반영한다(부하선은 보통 ±20 % 이내, 확률적 펄스열과 p-bit는 최대 약 1.6배 과대 추정). 전류원이나 100 MΩ 이상의 저항으로 구동되는 셀은 개방 전압 대신 준정적 부하선 보행 $C\\,dV/dt = I_N - G V - I_{\\mathrm{HRS/LRS}}(V)$(히스테리시스)로 $V_{DS}(t)$를 만들어 이완 발진의 주기 수까지 추정한다. 추정치가 $2\\times$`max_steps`(기본 $10^6$)를 넘으면 실행을 거부하고, $0.5\\times$를 넘으면 경고하며, 요청 전체가 $4\\times10^7$ 스텝을 넘어도 거부한다. 구현 세부는 docs/CIRCUIT_SIMULATOR.md에 있다.",
          "Feasibility: before running, the number of steps per run is estimated from $\\tau_{\\mathrm{rel}}(V_D)$ and the event rate along the quasi-static branches with the step rules above, including noise-induced escape before the fold (empirical hazard); load lines are typically within about ±20 %, probabilistic pulse trains and p-bits are overestimated by up to about 1.6×. Cells driven by a current source or a resistor of 100 MΩ or more use, instead of the open-circuit voltage, a quasi-static load-line walk $C\\,dV/dt = I_N - G V - I_{\\mathrm{HRS/LRS}}(V)$ (hysteretic), so the cycles of a relaxation oscillator are counted. Above $2\\times$`max_steps` (default $10^6$) the run is refused, above $0.5\\times$ a warning is issued, and a request above $4\\times10^7$ steps in total is refused. Implementation details: docs/CIRCUIT_SIMULATOR.md.",
        ),
      ],
    },
    {
      heading: L("테스트 벤치", "Test benches"),
      body: L(
        "모든 벤치 공통: 게이트는 DC $V_G$, 드레인 노드에 $C_d=2$ fF, 소스는 접지(p-bit만 소스 저항). `None`인 기본값은 소자에 맞춰 자동으로 정해지며, 결과의 `bench_params`에 돌려준다.\n\n- **부하선(load line)**: 삼각파 $v_{\\min}\\to v_{\\max}\\to v_{\\min}$ → $R_s=1$ kΩ → STL, 기본 1 사이클. $v_{\\max}$와 램프 속도는 프리셋을 따른다(기준 보정 4 V, 0.4 V/s; 광조사 보정 5 V, 1200 V/s). 캐리어 잡음 실행에서 속도를 직접 지정하지 않았고 추정 스텝 수가 $0.5\\times$`max_steps`를 넘으면 1200 V/s로 바꾼다(암조건 기준 보정의 0.4 V/s는 잡음 대역 덕분에 그대로 계산할 수 있다). $V_{LU}/V_{LD}$는 드레인 노드($V_{DS}$)와 전원 쪽에서 모두 기록한다. $Q_B$가 사이클 사이에 이어지므로 잔류 바디 기억이 자동으로 포함된다.\n- **펄스(pulse)**: 펄스열(기본 진폭 = fold $V_{LU}$ + 0.10 V, 평탄부 200 µs, 주기 1 ms, 상승/하강 10 µs, 10개) → $R_s=1$ kΩ. 평탄부 끝의 전환 확률, 주기 끝의 유지율(여전히 래치 상태인 비율), 전환 지연을 구하며, 진폭 목록을 주면 $P_{\\mathrm{sw}}$ 곡선을 얻는다.\n- **p-bit**: 드레인에 일정 간격의 전압 펄스(0 / fold $V_{LU}-15$ mV, 주기 1 ms, 평탄부 200 µs, 상승/하강 20 µs, 50개)를 직접 걸고, 소스 → $R_S=100$ kΩ → 접지, 비교기는 소스 노드 전압 $V(R_S)=R_S I_D$를 기준 전압 $V_{\\mathrm{ref}}$(기본 $R_S\\times1$ µA $=0.1$ V)와 비교한다. 래치되면 LRS 전류로 $V(R_S)\\approx0.44$ V, 래치되지 않으면 µV 수준이므로, fold 바로 아래의 펄스는 캐리어 잡음에 따라 무작위로 비교기를 발화시킨다(기준 소자 $P(1)\\approx0.5$; 결정론 모드에서는 0, 3.72 V 이상에서는 1). $P(1)$, 래치 비율 $P_{\\mathrm{latched}}$, 비트 lag-1을 구하며, V_G나 광세기 목록을 주면 $P(1)$ 곡선을 얻는다.\n- **결합 쌍(coupled)**: 공통 램프 또는 펄스 전원(상승/하강 10 µs)에서 각각 $R_{s1}=R_{s2}=100$ kΩ을 거쳐 연결하고, 두 드레인 사이에 $R_c=1$ MΩ을 둔다. 두 번째 셀은 V_G와 광세기를 따로 지정할 수 있다.",
        "All benches: DC gate $V_G$, $C_d=2$ fF on the drain node, grounded source (except the p-bit's source resistor). `None` defaults are resolved from the device and returned in the result's `bench_params`.\n\n- **Load line**: triangle $v_{\\min}\\to v_{\\max}\\to v_{\\min}$ → $R_s=1$ kΩ → STL, 1 cycle by default. $v_{\\max}$ and the ramp rate follow the preset (reference calibration 4 V, 0.4 V/s; illumination calibration 5 V, 1200 V/s). In carrier-noise runs without an explicit rate, the rate falls back to 1200 V/s when the estimated step count exceeds $0.5\\times$`max_steps` (the dark reference calibration at 0.4 V/s stays feasible thanks to the noise bands). $V_{LU}/V_{LD}$ are recorded both at the drain node ($V_{DS}$) and at the supply. $Q_B$ is continuous across cycles, so residual body memory is included automatically.\n- **Pulse**: pulse train (default amplitude = fold $V_{LU}$ + 0.10 V, 200 µs flat top, 1 ms period, 10 µs rise/fall, 10 pulses) → $R_s=1$ kΩ. Outputs: the switching probability at the end of the flat top, the retention at the end of the period (still latched) and the switching delay; a list of amplitudes gives a $P_{\\mathrm{sw}}$ curve.\n- **p-bit**: regular voltage pulses applied directly to the drain (0 / fold $V_{LU}-15$ mV, 1 ms period, 200 µs flat top, 20 µs rise/fall, 50 pulses), source → $R_S=100$ kΩ → ground, and a comparator that compares the source-node voltage $V(R_S)=R_S I_D$ with a reference $V_{\\mathrm{ref}}$ (default $R_S\\times1$ µA $=0.1$ V). A latched cell drives $V(R_S)\\approx0.44$ V through its LRS current, an unlatched one only µV, so pulses just below the fold fire the comparator at random through the carrier noise ($P(1)\\approx0.5$ for the reference device; 0 deterministically, 1 from 3.72 V). Outputs: $P(1)$, the latched fraction $P_{\\mathrm{latched}}$ and the bit lag-1; a list of V_G or light values gives a $P(1)$ curve.\n- **Coupled**: a common ramp or pulse source (10 µs rise/fall) feeds each cell through $R_{s1}=R_{s2}=100$ kΩ, with $R_c=1$ MΩ between the drains; the second cell may use its own V_G and light.",
      ),
      equations: [
        {
          id: "eq-ckt-psw",
          label: L("펄스 전환 확률", "Pulse switching probability"),
          tex: r`P_{\mathrm{sw}} = \frac{1}{N}\sum_{n=1}^{N}\mathbf 1\big[\text{latched at } t^{\mathrm{top}}_n\big]`,
          note: L("$t^{\\mathrm{top}}_n$은 $n$번째 펄스 평탄부의 끝이다. 래치 상태는 바디가 있는 branch로 정한다(위의 래치 검출). $N$은 실행이 도달한 펄스의 수다.", "$t^{\\mathrm{top}}_n$: end of the flat top of pulse $n$. Latched = the body's branch (latch detection above). $N$ counts the pulses the run reached."),
          code: "circuit/benches.py · build_pulse(); runner.py · _pulse_bits()",
        },
        {
          id: "eq-ckt-pbit",
          label: L("p-bit 비교기", "p-bit comparator"),
          tex: r`b_n = \mathbf 1\big[R_S\, I_D(t^{\mathrm{top}}_n) > V_{\mathrm{ref}}\big],\qquad V_{\mathrm{ref}} = R_S\cdot 1\,\mu\mathrm{A},\qquad P(1) = \frac{1}{N}\sum_n b_n`,
          note: L("$b_n$은 펄스 평탄부 끝에서의 비교기 출력이다. 소스 저항에는 드레인 전류가 그대로 흐르므로 $V(R_S)=R_S I_D$이다. 보통 $b_n=1$은 셀이 래치되어 LRS 전류(µA)가 흐르는 상태를 뜻한다. 그러나 채널이 켜지면($V_G$가 채널 문턱 위) 래치 없이도 전류가 커져 1이 되므로, 이때는 경고하고 $P_{\\mathrm{latched}}$를 따로 보고한다.", "$b_n$ is the comparator output at the end of the pulse's flat top. The source resistor carries the drain current, so $V(R_S)=R_S I_D$. Normally $b_n=1$ means the cell latched and carries its LRS current (µA); with the channel on ($V_G$ above the channel threshold) the current rises without a latch, which triggers a warning, and $P_{\\mathrm{latched}}$ is reported separately."),
          code: "circuit/benches.py · build_pbit()",
        },
      ],
      notes: [
        L(
          "바디 완화(µs)보다 빠른 전원 에지는 드레인 공핍 전하를 통해 떠 있는 바디를 흔든다(고정 $Q_B$에서 $\\Delta u\\approx0.03\\,\\Delta r$). 그래서 정적 fold 아래에서도 래치업이 일어날 수 있다. 이는 전하 좌표 모델의 실제 예측이며, 기본 에지 10–20 µs에서는 이 효과가 작다.",
          "Supply edges faster than the body relaxation (µs) kick the floating body through the drain-depletion charge ($\\Delta u\\approx0.03\\,\\Delta r$ at fixed $Q_B$) and can trigger latch-up below the static fold. This is a genuine prediction of the charge-coordinate model; the default 10–20 µs edges keep the effect small.",
        ),
        L("구현 세부(분석 출력, 요약 항목)는 docs/CIRCUIT_SIMULATOR.md를 참조한다.", "Implementation details (analysis outputs, summary items): see docs/CIRCUIT_SIMULATOR.md."),
      ],
    },
  ],
  related: ["charge-balance", "stochastic-events", "first-passage", "local-states", "open-problems", "numerics"],
  codeRefs: [CODE.circuit + "element.py", CODE.circuit + "mna.py", CODE.circuit + "benches.py", CODE.circuit + "stochastic.py", CODE.circuit + "sim.py", CODE.circuit + "runner.py", "engine/docs/CIRCUIT_ELEMENT_DESIGN.md", "docs/CIRCUIT_SIMULATOR.md"],
};

export default topic;
