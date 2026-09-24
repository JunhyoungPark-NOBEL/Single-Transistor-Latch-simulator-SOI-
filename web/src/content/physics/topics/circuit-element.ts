import type { PhysicsTopic } from "../types";
import { CODE, L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "circuit-element",
  title: L("회로 소자로서의 STL (MNA 과도해석)", "STL as a circuit element (MNA transient)"),
  summary: L(
    "STL은 body 전하 $Q_B$를 상태로 갖는 3단자(D, G, S) 비선형 소자이다. 내부 미지수 $(u,r)$가 단자 전압식과 전하식을 만족하도록 MNA 행렬에 넣고, 음함수 BE/TRAP + Newton으로 적분한다. 확률 모드에서는 셀마다 매 스텝 적분 계층을 고른다: Eq. 2 사건 수준 tau-leap, 분산 보정 Gaussian, 또는 잡음 없는 drift(빠른 완화, latch된 셀, 잡음 대역 밖). latch는 명시적 스위치 없이 ODE에서 나온다.",
    "The STL is a three-terminal (D, G, S) nonlinear element whose state is the body charge $Q_B$. Internal unknowns $(u,r)$ enter the MNA system through a terminal-voltage equation and a charge equation, integrated with implicit BE/TRAP + Newton. In stochastic mode each cell picks an integration tier before every step: event-level Eq. 2 tau-leap, variance-corrected Gaussian, or drift only without noise (fast relaxation, latched cell, outside the noise band). Latching emerges from the ODE without an explicit switch.",
  ),
  tags: ["circuit", "MNA", "transient", "tau-leap"],
  sections: [
    {
      heading: L("소자 방정식", "Element equations"),
      body: L(
        "단자 D, G, S (+ 선택적 광입력 $I_{\\mathrm{PH}}(t)$ → p[13]). 요소마다 미지수 $(u_k,r_k)$ 두 개와 식 두 개(E1, E2)가 추가된다. $V_{GS}$는 매 평가에서 p[11]에 들어간다(모든 벤치에서 DC). 전하 좌표는 compound-FPT 격자와 같다(상수 $-C_{ox}V_{GS}$ 차이). $V_{GS}$가 시간에 따라 바뀌면 같은 $Q$에서 $\\psi$가 따라 움직인다(게이트–body 용량 결합).",
        "Terminals D, G, S (+ optional light input $I_{\\mathrm{PH}}(t)$ → p[13]). Each element adds two unknowns $(u_k,r_k)$ and two equations (E1, E2). $V_{GS}$ is written to p[11] at every evaluation (DC in every bench). The charge coordinate is that of the compound-FPT lattice (up to the constant $-C_{ox}V_{GS}$). When $V_{GS}$ changes in time, $\\psi$ follows at fixed $Q$ (gate–body capacitive coupling).",
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
          tex: r`\frac{Q(u,r;V_{GS}) - Q_c - \theta\,h\,F(u,r)}{C_{ox}} = 0,\qquad Q_{n+1} = Q_c + \theta\,h\,F_{n+1}`,
          note: L(
            "- 결정론 BE: $Q_c=Q_n$, $\\theta=1$. TRAP: $Q_c=Q_n+\\tfrac h2F_n$, $\\theta=\\tfrac12$ (모든 셀에서 $h<2\\tau_{\\mathrm{rel}}$일 때만, 아니면 BE).\n- 확률, 계층 1 (사건 수준 tau-leap): $Q_c=Q_n+\\Delta Q_{\\mathrm{ev}}$, $\\theta=0$ (명시적).\n- 확률, 계층 2 (Gaussian): $Q_c=Q_n+\\eta$, $\\theta=1$ (drift 음함수).\n- 확률, 계층 3–5 (drift만): $Q_c=Q_n$, $\\theta=1$ (BE, 잡음 없음).\n\n캐리어 잡음이 켜진 실행에서는 TRAP을 쓰지 않고 커패시터도 BE companion이다. 저장 전하는 적분식 값 $Q_c+\\theta hF_{n+1}$이라 사건 장부가 정확히 유지된다.",
            "- Deterministic BE: $Q_c=Q_n$, $\\theta=1$. TRAP: $Q_c=Q_n+\\tfrac h2F_n$, $\\theta=\\tfrac12$ (only when $h<2\\tau_{\\mathrm{rel}}$ for every cell, else BE).\n- Stochastic tier 1 (event-level tau-leap): $Q_c=Q_n+\\Delta Q_{\\mathrm{ev}}$, $\\theta=0$ (explicit).\n- Stochastic tier 2 (Gaussian): $Q_c=Q_n+\\eta$, $\\theta=1$ (drift-implicit).\n- Stochastic tiers 3–5 (drift only): $Q_c=Q_n$, $\\theta=1$ (BE, no noise).\n\nRuns with carrier noise never use TRAP, and their capacitors use the BE companion. The stored charge is the integration-formula value $Q_c+\\theta hF_{n+1}$, so the event bookkeeping stays exact.",
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
          "엔진 영역 밖 연속화: $u<0$(빠른 하강 후 소스 역바이어스)은 $u=0$의 전류에 $X'(0^+)V_T(e^{u/V_T}-1)$을 이어 붙이고, $r<0$(드레인 순바이어스)은 소스와 같은 포화전류의 대칭 순방향 드레인 다이오드 + 공핍 SRH를 손실로 더한다. 엔진은 이 영역을 평가하지 않는다(결과 `warnings`에 보고).",
          "Continuation outside the engine domain: $u<0$ (source reverse biased after fast down-ramps) extends the $u=0$ currents by $X'(0^+)V_T(e^{u/V_T}-1)$; $r<0$ (drain forward biased) adds a symmetric forward drain diode (source saturation current) plus depletion SRH as a loss. The engine itself never evaluates these regions (reported in the result `warnings`).",
        ),
        L(
          "단자 전류는 준정적 $I_D$이다: $dQ_B/dt$에 따른 변위 전류(게이트–body, 공핍 전하)는 단자에 넣지 않으며 게이트는 전류를 흘리지 않는다.",
          "Terminal currents are the quasi-static $I_D$: the displacement currents tied to $dQ_B/dt$ (gate–body, depletion charges) are not stamped into the terminals, and the gate draws no current.",
        ),
      ],
    },
    {
      heading: L("MNA 조립과 Newton", "MNA assembly and Newton"),
      body: L(
        "미지수 $x=[v_1..v_{N-1},\\ i_V,\\ (u_k,r_k)]$. 각 비접지 노드의 KCL(나가는 전류 합 = 0), 전압원 행 $v_a-v_b-V(t)=0$, STL의 E1/E2. 노드마다 $G_{\\min}=10^{-18}$ S. STL 블록 Jacobian은 `stl_eval`의 전진 유한차분($10^{-6}$ V; 영역 밖이면 후진). 반복당 $|\\Delta u|\\le50$ mV, $|\\Delta r|\\le1$ V, 영역을 벗어나면 반감 backtracking(최대 14회). 수렴: $|\\Delta u|<10^{-7}$ V, $|\\Delta r|<10^{-6}$ V, $|\\Delta v|<10^{-6}(1+|v|)$ V, $|E_1|,|E_2|<10^{-5}$ V, 최대 14회(계층 1–2에서는 허용오차 20배 완화).",
        "Unknowns $x=[v_1..v_{N-1},\\ i_V,\\ (u_k,r_k)]$. KCL at each non-ground node (sum of leaving currents = 0), source rows $v_a-v_b-V(t)=0$, STL rows E1/E2. $G_{\\min}=10^{-18}$ S at every node. The STL Jacobian block is a forward finite difference of `stl_eval` ($10^{-6}$ V; backward outside the domain). Per iteration $|\\Delta u|\\le50$ mV, $|\\Delta r|\\le1$ V, halving backtracking (at most 14×) when leaving the domain. Convergence: $|\\Delta u|<10^{-7}$ V, $|\\Delta r|<10^{-6}$ V, $|\\Delta v|<10^{-6}(1+|v|)$ V, $|E_1|,|E_2|<10^{-5}$ V, at most 14 iterations (tolerance relaxed 20× in tiers 1–2).",
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
        {
          id: "eq-ckt-tau",
          label: L("회로 제약을 따른 국소 완화시간", "local relaxation time along the circuit constraints"),
          tex: r`J_{\theta=0}\,s_k = \frac{e_{E2,k}}{C_{ox}},\qquad \frac{dF_k}{dQ_k} = \frac{\partial F_k}{\partial u}s_{u,k} + \frac{\partial F_k}{\partial r}s_{r,k},\qquad \tau_{\mathrm{rel},k} = \frac{1}{|dF_k/dQ_k|}`,
          note: L(
            "매 스텝 뒤 전하 고정 Jacobian으로 $dx/dQ_k$를 구한다($R_s$, $C_d$ companion, 다른 셀 포함). 전형값: fold 근처 HRS 수 µs–수백 µs(fold에서 임계 감속), 암조건 저 $V_D$ HRS ms–0.1 s, LRS 0.03–50 ns(LD fold에서 µs 이상으로 증가).",
            "After every step the charge-fixed Jacobian gives $dx/dQ_k$ (including $R_s$, the $C_d$ companion and the other cells). Typical values: HRS near the fold a few µs to hundreds of µs (critical slowing at the fold), HRS at low $V_D$ in the dark ms–0.1 s, LRS 0.03–50 ns (rising to µs and beyond at the LD fold).",
          ),
          code: "circuit/mna.py · sensitivities()",
        },
      ],
      notes: [
        L(
          "초기 동작점: $u=0$(빈 body)으로 고정해 풀고(커패시터 개방), 의사과도 BE($h$: $10^{-12}$ s에서 ×3, 최대 $10^8$ s)로 빈 body에서 도달하는 저전류 상태를 찾는다(쌍안정 구간이면 HRS). 초기 latch 표시는 $I_D \\ge I_{th}$.",
          "Initial operating point: solve with $u=0$ (empty body, capacitors open), then pseudo-transient BE ($h$ from $10^{-12}$ s, ×3 per step, up to $10^8$ s) to the low-current state reachable from an empty body (the HRS inside the hysteresis window). The initial latch flag is $I_D \\ge I_{th}$.",
        ),
      ],
    },
    {
      heading: L("확률 모드: 적분 계층", "Stochastic mode: integration tiers"),
      body: L(
        "셀마다 스텝 전에 상태 $n$에서 계층을 고른다(이 순서로 검사):\n\n- **4 drift만**: 셀이 latch 상태(latch-up과 latch-down 사이)이고 `ld_carrier_noise` = false (회로 기본값).\n- **5 drift만**: $V_{DS}$가 그 셀의 잡음 대역 밖(아래).\n- **3 drift만**: $\\tau_{\\mathrm{rel}} < 2$ ns (`gauss_tau_min`)이고 $\\tau_{\\mathrm{frac}}\\tau_{\\mathrm{rel}} < h_{\\mathrm{noise,min}}$.\n- **1 사건 수준 tau-leap**: $\\tau_{\\mathrm{frac}}\\tau_{\\mathrm{rel}} \\ge h_{\\mathrm{noise,min}}$ (기본 $\\tau_{\\mathrm{rel}} \\ge 40$ ns).\n- **2 Gaussian**: 나머지, 즉 $2\\ \\mathrm{ns} \\le \\tau_{\\mathrm{rel}} < 40$ ns (예: latch-down fold 근처 LRS).\n\n`carrier_noise` = false이면 결정론 적분에 국소 상태만 더한다. 계층 1은 스텝 시작 상태의 사건률로 $\\Delta Q_{\\mathrm{ev}}$를 뽑는다: 단위 사건 $N_\\uparrow$, 손실 $N_\\downarrow$는 Poisson, II는 클러스터 수 Poisson × 크기 $k\\sim p_k(r)/P_1$(현재 $r$에서 선형 보간한 확장 avalanche 커널). 평균이 100 (`gauss_threshold`)을 넘는 개수는 Gaussian 극한. 연속화 영역의 음의 전류는 반대 방향 사건으로 넣으므로 $\\mathbb E[\\Delta Q]=hF$가 유지된다.",
        "Each cell picks its tier from the state $n$ before every step (checked in this order):\n\n- **4 drift only**: the cell is latched (between its latch-up and latch-down) and `ld_carrier_noise` = false (circuit default).\n- **5 drift only**: $V_{DS}$ lies outside the cell's noise band (below).\n- **3 drift only**: $\\tau_{\\mathrm{rel}} < 2$ ns (`gauss_tau_min`) and $\\tau_{\\mathrm{frac}}\\tau_{\\mathrm{rel}} < h_{\\mathrm{noise,min}}$.\n- **1 event-level tau-leap**: $\\tau_{\\mathrm{frac}}\\tau_{\\mathrm{rel}} \\ge h_{\\mathrm{noise,min}}$ ($\\tau_{\\mathrm{rel}} \\ge 40$ ns by default).\n- **2 Gaussian**: the rest, i.e. $2\\ \\mathrm{ns} \\le \\tau_{\\mathrm{rel}} < 40$ ns (e.g. the LRS near the latch-down fold).\n\nWith `carrier_noise` = false the integration is deterministic plus the local states. Tier 1 draws $\\Delta Q_{\\mathrm{ev}}$ from the rates at the start of the step: unit events $N_\\uparrow$ and losses $N_\\downarrow$ are Poisson, II is a Poisson number of clusters with sizes $k\\sim p_k(r)/P_1$ (extended avalanche kernel, linearly interpolated at the current $r$). Counts whose mean exceeds 100 (`gauss_threshold`) use the Gaussian limit. Negative currents of the continuation regions are folded into the opposite event class, so $\\mathbb E[\\Delta Q]=hF$ holds.",
      ),
      equations: [
        {
          id: "eq-ckt-dq",
          label: L("계층 1: 사건 증분", "tier 1: event increment"),
          tex: r`\begin{aligned} &\Delta Q_{\mathrm{ev}} = q\Big(N_\uparrow + \sum_{c=1}^{N_c} k_c - N_\downarrow\Big)\\ &N_\uparrow\sim\mathcal P\big(\lambda_{\mathrm{unit}}h\big),\quad N_\downarrow\sim\mathcal P(Lh),\quad N_c\sim\mathcal P\Big(\lambda_{\mathrm{II}}h\,\frac{P_1}{M_1}\Big)\end{aligned}`,
          note: L(
            "$P_1=\\sum_{k\\ge1}p_k$, $M_1=\\sum k p_k$, $M_2=\\sum k^2p_k$; 크기 $k$의 도약률 $\\lambda_{\\mathrm{II}}p_k/M_1$은 `compound_fpt.backward`와 같다. Gaussian 극한: $\\sum k_c\\approx\\lambda_{\\mathrm{II}}h+\\sqrt{\\lambda_{\\mathrm{II}}hM_2/M_1}\\,\\xi$, Poisson 개수는 $\\lambda+\\sqrt\\lambda\\,\\xi$ (모두 0에서 절단).",
            "$P_1=\\sum_{k\\ge1}p_k$, $M_1=\\sum k p_k$, $M_2=\\sum k^2p_k$; the jump rate for size $k$, $\\lambda_{\\mathrm{II}}p_k/M_1$, is that of `compound_fpt.backward`. Gaussian limit: $\\sum k_c\\approx\\lambda_{\\mathrm{II}}h+\\sqrt{\\lambda_{\\mathrm{II}}hM_2/M_1}\\,\\xi$, Poisson counts $\\lambda+\\sqrt\\lambda\\,\\xi$ (all clipped at 0).",
          ),
          code: "circuit/mna.py · draw_dq(), poisson_or_gauss(); element.py · pmf_at()",
        },
        {
          id: "eq-ckt-gauss",
          label: L("계층 2: 분산 보정 Gaussian", "tier 2: variance-corrected Gaussian"),
          tex: r`\begin{aligned} &\eta\sim\mathcal N\Big(0,\ D\,h\big(1 + \tfrac{h}{2\tau_{\mathrm{rel}}}\big)\Big),\qquad D = q^2\Big(R_\uparrow + R_\downarrow + \lambda_{\mathrm{II}}\frac{M_2}{M_1}\Big)\\ &\delta_{n+1} = \frac{\delta_n + \eta}{1 + h/\tau_{\mathrm{rel}}}\ \Rightarrow\ \mathrm{Var}[\delta] = \frac{\mathrm{Var}[\eta]}{(1+h/\tau_{\mathrm{rel}})^2 - 1} = \frac{D\,\tau_{\mathrm{rel}}}{2}\end{aligned}`,
          note: L(
            "선형화한 음함수 drift에서 정상 분산이 모든 $h$에서 정확한 OU 분산 $D\\tau_{\\mathrm{rel}}/2$가 된다. 탈출 동역학은 $h\\lesssim\\tau_{\\mathrm{rel}}/2$에서만 해상되므로 $h\\le0.5\\,\\tau_{\\mathrm{rel}}$ (`gauss_tau_frac`).",
            "For the linearised implicit drift the stationary variance equals the exact OU variance $D\\tau_{\\mathrm{rel}}/2$ for any $h$. The escape dynamics are resolved only for $h\\lesssim\\tau_{\\mathrm{rel}}/2$, hence $h\\le0.5\\,\\tau_{\\mathrm{rel}}$ (`gauss_tau_frac`).",
          ),
          code: "circuit/mna.py · run_chunk() (reg 2), noise_var_rate()",
        },
        {
          id: "eq-ckt-band",
          label: L("잡음 대역 (계층 5의 경계)", "noise bands (boundary of tier 5)"),
          tex: r`\begin{aligned} &z(V_D) = \frac{|Q - Q_{\mathrm{saddle}}(V_D)|}{\sqrt{D\,\tau_{\mathrm{rel}}/2}}\quad\text{(HRS and LRS branches, 301-point grid)}\\ &\text{unlatched: } \big[\min V_{\mathrm{HRS}}(z<z_{\max}) - e_{LU},\ V_{LU} + 0.25 + e_{LU}\big]\\ &\text{latched: } \big[V_{LD} - 0.25 - e_{LD},\ \max V_{\mathrm{LRS}}(z<z_{\max}) + e_{LD}\big]\\ &e_{LU} = 4(\sigma + 5\sigma_E),\qquad e_{LD} = 4(0.05\,\sigma + 50\,\sigma_E)\end{aligned}`,
          note: L(
            "$z_{\\max}=12$ (`noise_z_max`)는 정확한 compound-jump hazard로 보정했다: 대역 밖 탈출률 ≲ $10^{-3}$ s⁻¹. 대역 밖은 drift만 적분한다(정상 요동 몇 mV는 계산하지 않음). 확장 $e$는 GIDL 작용점의 국소 상태 fold 이동 4σ; 다른 작용점이나 $\\sigma_E>2$ mV, 또는 latch가 없는 소자는 대역 제한 없음. 예(국소 상태 없음): 논문 −2 V 암조건 HRS 3.59–3.95 V, LRS 2.35–2.83 V; photo 프리셋 HRS 3.70–4.05 V; −1.8 V·2.63 pA HRS 2.93–3.54 V. 이 대역 덕분에 0.4 V/s 논문 소자도 사건 수준으로 계산 가능하다(사이클당 약 2×10⁵ 스텝).",
            "$z_{\\max}=12$ (`noise_z_max`) was calibrated against the exact compound-jump hazard: escape rate ≲ $10^{-3}$ s⁻¹ outside the bands. Outside the bands only the drift is integrated (the stationary fluctuation of a few mV is not simulated). The widening $e$ is 4σ of the local-state fold shifts for the GIDL action point; other action points, $\\sigma_E>2$ mV, or a device without latch get no band restriction. Examples (no local states): paper −2 V dark HRS 3.59–3.95 V, LRS 2.35–2.83 V; photo preset HRS 3.70–4.05 V; −1.8 V, 2.63 pA HRS 2.93–3.54 V. The bands make event-level runs of the paper device at 0.4 V/s affordable (about 2×10⁵ steps per cycle).",
          ),
          code: "circuit/stochastic.py · branch_profile(), noise_bands()",
        },
        {
          id: "eq-ckt-ou",
          label: L("가변 스텝 OU (evolving 국소 상태)", "variable-step OU (evolving local states)"),
          tex: r`\delta_{n+1} = a\,\delta_n + \sigma\sqrt{1-a^2}\,\xi,\qquad a = e^{-h/\tau}`,
          note: L(
            "작용점 상태(p[9], p[23], p[19] 또는 p[20])와 emitter 상태(p[10])에 각각 적용; 초기값은 실행마다 $\\mathcal N(0,\\sigma^2)$ 추출(frozen은 실행 내내 고정). 획득 추세는 회로에 적용하지 않는다. 실행 $r$의 난수: 사건은 seed + 1000003 r, 국소 상태 초기값은 `default_rng([seed, 7349, r])`.",
            "Applied to the action-point state (p[9], p[23], p[19] or p[20]) and to the emitter state (p[10]); the initial value is drawn from $\\mathcal N(0,\\sigma^2)$ per run (frozen: kept for the whole run). The acquisition trend is not applied in the circuit. Random numbers of run $r$: events use seed + 1000003 r, the local-state draws `default_rng([seed, 7349, r])`.",
          ),
          code: "circuit/mna.py · run_chunk() (lsmode 2); circuit/stochastic.py · draw_local_states()",
        },
      ],
      notes: [
        L(
          "왜 대역 밖에서 큰 스텝 Gaussian을 쓰지 않나: 쌍안정 구간에서 $h\\gg\\tau_{\\mathrm{rel}}$이면 음함수식이 약 $h/\\tau$ SD의 충격을 받아 Newton이 다른 basin의 근에 떨어질 수 있다(시험에서 $V_{LD}$가 2.656 → 2.84 V). drift만은 평균이 정확하다.",
          "Why no large-step Gaussian outside the bands: with $h\\gg\\tau_{\\mathrm{rel}}$ in a bistable region the implicit equation receives kicks of about $h/\\tau$ SDs and Newton can land on the root of the other basin (in a test $V_{LD}$ moved from 2.656 to 2.84 V). Drift only is exact in the mean.",
        ),
        L(
          "회로의 `ld_carrier_noise` 기본값은 false(계층 4: LRS는 drift만, $V_{LD}$는 LRS fold의 결정론적 탈출 + 국소 상태 효과)이다. 소자 수준 스윕 MC 프리셋은 true를 쓴다(`sweep-mc`). 회로에서 켜면 LRS에도 계층 1/2가 적용되어 잡음에 의한 이른 latch-down이 $V_{LD}$를 올린다(1200 V/s, −1.8 V·2.63 pA: 2.593 → 2.657 V, 비용 약 20배).",
          "The circuit's `ld_carrier_noise` defaults to false (tier 4: the LRS is drift only and $V_{LD}$ is the deterministic escape at the LRS fold plus local-state effects); the device-level sweep-MC presets use true (`sweep-mc`). Turned on in the circuit, tiers 1/2 also apply in the LRS and noise-induced early latch-down raises $V_{LD}$ (1200 V/s, −1.8 V, 2.63 pA: 2.593 → 2.657 V at about 20× the cost).",
        ),
      ],
    },
    {
      heading: L("적응 Δt와 latch 검출", "Adaptive Δt and latch detection"),
      body: L(
        "- **결정론 (그리고 확률 모드의 drift 계층 3–5)**: $\\mathrm{err}=\\max(|\\Delta u|/\\Delta u_{\\max},\\ |\\Delta\\ln I_D|/\\Delta_{\\ln I},\\ |\\Delta v|/\\Delta v_{\\max},\\ \\mathrm{LTE}/\\mathrm{LTE}_u)$, $\\mathrm{LTE}=\\tfrac12h|F-F_0|/|\\partial Q/\\partial u|$. err > 1.5이면 거부, $h\\leftarrow h\\max(0.1,0.7/\\mathrm{err})$; 다음 스텝 $h\\cdot\\min(2.5,\\max(0.3,0.8/\\mathrm{err}))$. Newton 실패 → $h/4$. 파형 꺾임점·샘플 시각에 정확히 착지.\n- **계층 1**: $h\\le\\tau_{\\mathrm{frac}}\\tau_{\\mathrm{rel}}$, $h\\le N_{\\mathrm{ev,max}}/\\lambda_{\\mathrm{tot}}$, $h\\le\\Delta u_{\\max}|\\partial Q/\\partial u|/|F|$. **계층 2**: $h\\le0.5\\,\\tau_{\\mathrm{rel}}$, 같은 drift 한계. 스텝은 모든 셀의 최솟값. 잡음에 의한 움직임은 거부하지 않는다(뽑힌 결과로 거부하면 통계가 치우침); 예외는 $|\\Delta u| > 20\\Delta u_{\\max}$ = 0.2 V.\n- **latch 검출**: latch 안 된 셀의 $I_D$가 $I_{th}$ = 10 nA를 위로 지나면 latch-up, latch된 셀의 $I_D$가 $I_{th}/\\mathrm{hysteresis}$ = 1 nA를 아래로 지나면 latch-down. 시각은 스텝 안에서 $\\ln I$ 선형 보간, 그때의 $V_{DS}$와 전원 전압을 기록. 사이클마다 $V_{LU}$는 첫 latch-up, $V_{LD}$는 그 뒤 첫 latch-down.",
        "- **deterministic (and the drift tiers 3–5 of stochastic runs)**: $\\mathrm{err}=\\max(|\\Delta u|/\\Delta u_{\\max},\\ |\\Delta\\ln I_D|/\\Delta_{\\ln I},\\ |\\Delta v|/\\Delta v_{\\max},\\ \\mathrm{LTE}/\\mathrm{LTE}_u)$, $\\mathrm{LTE}=\\tfrac12h|F-F_0|/|\\partial Q/\\partial u|$. Reject if err > 1.5 with $h\\leftarrow h\\max(0.1,0.7/\\mathrm{err})$; next step $h\\cdot\\min(2.5,\\max(0.3,0.8/\\mathrm{err}))$. Newton failure → $h/4$. Steps land exactly on waveform breakpoints and sample times.\n- **tier 1**: $h\\le\\tau_{\\mathrm{frac}}\\tau_{\\mathrm{rel}}$, $h\\le N_{\\mathrm{ev,max}}/\\lambda_{\\mathrm{tot}}$, $h\\le\\Delta u_{\\max}|\\partial Q/\\partial u|/|F|$. **tier 2**: $h\\le0.5\\,\\tau_{\\mathrm{rel}}$ and the same drift limit. The step is the minimum over the cells. Noise-driven moves are not step-rejected (rejecting on the drawn outcome would bias the statistics), except a hard guard $|\\Delta u| > 20\\Delta u_{\\max}$ = 0.2 V.\n- **latch detection**: latch-up when the $I_D$ of an unlatched cell crosses $I_{th}$ = 10 nA upwards, latch-down when the $I_D$ of a latched cell crosses $I_{th}/\\mathrm{hysteresis}$ = 1 nA downwards. The time is interpolated linearly in $\\ln I$ within the step, and $V_{DS}$ and the supply voltage at that time are recorded. Per cycle $V_{LU}$ is the first latch-up and $V_{LD}$ the first latch-down after it.",
      ),
      equations: [
        {
          id: "eq-ckt-step",
          label: L("스텝 한계와 초기 스텝 (자동값)", "step bounds and initial step (automatic values)"),
          tex: r`h_{\min} = \max\big(10^{-15}\,\mathrm{s},\ 10^{-13}\,t_{\mathrm{end}}\big),\qquad h_{\max} = \frac{t_{\mathrm{end}}}{2000},\qquad h_0 = \min\!\big(h_{\max},\ \max(10\,h_{\min},\ 10^{-7}\,t_{\mathrm{end}})\big)`,
          note: L(
            "`solver.dt_min_s`/`dt_max_s`를 주면 그 값을 쓴다. 예: 논문 부하선 0 → 4 → 0 V, 0.4 V/s ($t_{\\mathrm{end}}$ = 20 s): $h_{\\min}$ = 2 ps, $h_{\\max}$ = 10 ms, $h_0$ = 2 µs.",
            "`solver.dt_min_s`/`dt_max_s` override the automatic values. Example: paper load line 0 → 4 → 0 V at 0.4 V/s ($t_{\\mathrm{end}}$ = 20 s): $h_{\\min}$ = 2 ps, $h_{\\max}$ = 10 ms, $h_0$ = 2 µs.",
          ),
          code: "circuit/runner.py · _solver(), SolverConfig(h_init=…)",
        },
      ],
      variables: [
        { symbol: r`\Delta u_{\max}`, name: L("스텝당 u 변화 (reltol/1e-3 배, [1 mV, 50 mV])", "u change per step (× reltol/1e-3, [1 mV, 50 mV])"), value: "10", unit: "mV", code: "CF_DUMAX" },
        { symbol: r`\Delta_{\ln I}`, name: L("스텝당 ln I_D 변화", "ln I_D change per step"), value: "0.2", code: "CF_DLNIMAX" },
        { symbol: r`\Delta v_{\max}`, name: L("스텝당 노드 전압 변화", "node-voltage change per step"), value: "20", unit: "mV", code: "CF_DVMAX" },
        { symbol: r`\mathrm{LTE}_u`, name: L("u 국소 절단오차", "local truncation error in u"), value: "1", unit: "mV", code: "CF_LTEU" },
        { symbol: r`\tau_{\mathrm{frac}},\ h_{\mathrm{noise,min}}`, name: L("계층 1 스텝 비율, 계층 1 최소 스텝", "tier-1 step fraction, tier-1 minimum step"), value: "0.05, 2 ns", code: "solver.tau_frac, noise_dt_min_s" },
        { symbol: r`\tau_{G,\min},\ f_G`, name: L("계층 2 하한 τ_rel, 계층 2 스텝 비율", "tier-2 lower τ_rel bound, tier-2 step fraction"), value: "2 ns, 0.5", code: "solver.gauss_tau_min_s, gauss_tau_frac" },
        { symbol: r`N_{\mathrm{ev,max}}`, name: L("스텝당 최대 기대 사건 수; Gaussian 문턱", "max expected events per step; Gaussian threshold"), value: "200; 100", code: "solver.max_events_per_step, gauss_threshold" },
        { symbol: r`z_{\max}`, name: L("잡음 대역 장벽 한계 (fold 너머 여유 0.25 V)", "noise-band barrier limit (margin 0.25 V beyond the folds)"), value: "12", code: "solver.noise_z_max" },
        { symbol: r`I_{th}`, name: L("latch-up 검출 문턱", "latch-up detection threshold"), value: "10", unit: "nA", code: "detect.i_threshold_A" },
        { symbol: r`I_{th}/\mathrm{hyst}`, name: L("latch-down 검출 문턱 (hysteresis 10)", "latch-down detection threshold (hysteresis 10)"), value: "1", unit: "nA", code: "detect.hysteresis" },
      ],
      notes: [
        L(
          "10 nA latch-up 문턱은 FPT 격자의 흡수 경계와 같다(→ first-passage). latch-down에 히스테리시스가 필요한 이유: LD fold의 LRS 전류가 약 16 nA($V_G=-2$ V: 16.3 nA)뿐이라 10 nA 한 문턱은 캐리어 잡음에서 떨린다.",
          "The 10 nA latch-up threshold equals the absorbing boundary of the FPT lattice (→ first-passage). Latch-down needs the hysteresis because the LRS current at the LD fold is only about 16 nA (16.3 nA at $V_G=-2$ V), so a single 10 nA threshold chatters under carrier noise.",
        ),
        L(
          "실행 가능성: 실행 전에 준정적 가지의 $\\tau_{\\mathrm{rel}}(V_D)$·사건률과 위 스텝 규칙으로 실행당 스텝 수를 추정한다. $2\\times$`max_steps`(기본 $10^6$)를 넘으면 거부, $0.5\\times$를 넘으면 경고, 요청 전체가 $4\\times10^7$ 스텝을 넘으면 거부. 구현 세부: docs/CIRCUIT_SIMULATOR.md.",
          "Feasibility: before running, the steps per run are estimated from $\\tau_{\\mathrm{rel}}(V_D)$ and the event rate of the quasi-static branches with the step rules above. Above $2\\times$`max_steps` (default $10^6$) the run is refused, above $0.5\\times$ it is warned, and a request above $4\\times10^7$ steps in total is refused. Implementation detail: docs/CIRCUIT_SIMULATOR.md.",
        ),
      ],
    },
    {
      heading: L("테스트 벤치", "Test benches"),
      body: L(
        "모든 벤치: 게이트는 DC $V_G$, 드레인 노드에 $C_d=2$ fF, 소스 접지. `None` 기본값은 소자에서 자동으로 정해지고 결과의 `bench_params`에 돌려준다.\n\n- **load line**: 삼각파 $v_{\\min}\\to v_{\\max}\\to v_{\\min}$ → $R_s=1$ kΩ → STL, 기본 1 cycle. $v_{\\max}$·램프 속도는 프리셋(논문 4 V, 0.4 V/s; 광 5 V, 1200 V/s). 캐리어 잡음 실행에서 속도를 직접 주지 않았고 추정 스텝이 $0.5\\times$`max_steps`를 넘으면 1200 V/s로 바꾼다(암조건 논문 소자 0.4 V/s는 잡음 대역 덕분에 가능). $V_{LU}/V_{LD}$는 드레인 노드($V_{DS}$)와 전원 쪽에서 모두 기록. $Q_B$가 cycle 사이에 연속이므로 잔류 body 기억이 자동 포함.\n- **pulse**: 펄스열(기본 진폭 = fold $V_{LU}$ + 0.10 V, 평탄부 200 µs, 주기 1 ms, 상승/하강 10 µs, 10개) → $R_s=1$ kΩ. 평탄부 끝의 전환 확률, 주기 끝의 유지 비율($I_D \\ge$ 1 nA), 전환 지연; 진폭 목록으로 $P_{\\mathrm{sw}}$ 곡선.\n- **p-bit**: 클럭(0 / fold $V_{LU}-0.02$ V, 주기 1 ms, high 200 µs, 상승/하강 20 µs, 50 clock) → $R_L=100$ kΩ → STL, 비교기는 $v_D$. $P(1)$와 비트 lag-1; V_G 또는 빛 목록으로 $P(1)$ 곡선.\n- **coupled**: 공통 램프 또는 펄스(상승/하강 10 µs) 전원에서 각자 $R_{s1}=R_{s2}=100$ kΩ, 드레인 사이 $R_c=1$ MΩ; 두 번째 셀은 V_G·빛을 따로 지정 가능.",
        "All benches: DC gate $V_G$, $C_d=2$ fF on the drain node, grounded source. `None` defaults are resolved from the device and returned in the result's `bench_params`.\n\n- **load line**: triangle $v_{\\min}\\to v_{\\max}\\to v_{\\min}$ → $R_s=1$ kΩ → STL, 1 cycle by default. $v_{\\max}$ and ramp rate follow the preset (paper 4 V, 0.4 V/s; photo 5 V, 1200 V/s). In carrier-noise runs without an explicit rate, the rate falls back to 1200 V/s when the estimated steps exceed $0.5\\times$`max_steps` (the dark paper device at 0.4 V/s stays feasible thanks to the noise bands). $V_{LU}/V_{LD}$ are recorded at the drain node ($V_{DS}$) and at the supply. $Q_B$ is continuous across cycles, so residual body memory is included automatically.\n- **pulse**: pulse train (default amplitude = fold $V_{LU}$ + 0.10 V, 200 µs flat top, 1 ms period, 10 µs rise/fall, 10 pulses) → $R_s=1$ kΩ. Switching probability at the end of the flat top, retention at the end of the period ($I_D \\ge$ 1 nA), switching delay; an amplitude list gives a $P_{\\mathrm{sw}}$ curve.\n- **p-bit**: clock (0 / fold $V_{LU}-0.02$ V, 1 ms period, 200 µs high, 20 µs rise/fall, 50 clocks) → $R_L=100$ kΩ → STL, comparator on $v_D$. $P(1)$ and bit lag-1; a V_G or light list gives a $P(1)$ curve.\n- **coupled**: common ramp or pulse source (10 µs rise/fall) through $R_{s1}=R_{s2}=100$ kΩ each, $R_c=1$ MΩ between the drains; the second cell may override V_G and light.",
      ),
      equations: [
        {
          id: "eq-ckt-psw",
          label: L("펄스 전환 확률", "pulse switching probability"),
          tex: r`P_{\mathrm{sw}} = \frac{1}{N}\sum_{n=1}^{N}\mathbf 1\big[I_D(t^{\mathrm{top}}_n) \ge I_{th}\big]`,
          note: L("$t^{\\mathrm{top}}_n$: $n$번째 펄스 평탄부의 끝(latch된 펄스의 비율).", "$t^{\\mathrm{top}}_n$: end of the flat top of pulse $n$ (fraction of pulses latched)."),
          code: "circuit/benches.py · build_pulse(); runner.py · _pulse_bits()",
        },
        {
          id: "eq-ckt-pbit",
          label: L("p-bit 비교기", "p-bit comparator"),
          tex: r`b_n = \mathbf 1\big[v_D(t^{\mathrm{top}}_n) < v_{th}\big],\qquad v_{th} = v_{\mathrm{high}} - R_L\cdot 100\,\mathrm{nA},\qquad P(1) = \frac{1}{N}\sum_n b_n`,
          note: L("$b_n=1$은 셀이 latch되어 $v_D$가 부하선을 따라 내려간 상태.", "$b_n=1$ means the cell latched and $v_D$ dropped along the load line."),
          code: "circuit/benches.py · build_pbit()",
        },
      ],
      notes: [
        L(
          "body 완화(µs)보다 빠른 전원 에지는 드레인 공핍 전하를 통해 떠 있는 body를 친다(고정 $Q_B$에서 $\\Delta u\\approx0.03\\,\\Delta r$). 그래서 정적 fold 아래에서도 latch-up이 일어날 수 있다 — 전하 좌표 모델의 실제 예측이며, 기본 에지 10–20 µs는 이 효과를 작게 유지한다.",
          "Supply edges faster than the body relaxation (µs) kick the floating body through the drain-depletion charge ($\\Delta u\\approx0.03\\,\\Delta r$ at fixed $Q_B$) and can trigger latch-up below the static fold — a genuine prediction of the charge-coordinate model; the default 10–20 µs edges keep it small.",
        ),
        L("구현 세부: docs/CIRCUIT_SIMULATOR.md 참조 (분석 출력, 요약 항목).", "Implementation detail: see docs/CIRCUIT_SIMULATOR.md (analysis outputs, summary items)."),
      ],
    },
  ],
  related: ["charge-balance", "stochastic-events", "first-passage", "local-states", "open-problems", "numerics"],
  codeRefs: [CODE.circuit + "element.py", CODE.circuit + "mna.py", CODE.circuit + "benches.py", CODE.circuit + "stochastic.py", CODE.circuit + "sim.py", CODE.circuit + "runner.py", "engine/docs/CIRCUIT_ELEMENT_DESIGN.md", "docs/CIRCUIT_SIMULATOR.md"],
};

export default topic;
