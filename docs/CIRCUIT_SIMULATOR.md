# STL 회로 시뮬레이터 / STL circuit simulator

Package `server/compute/circuit/` (compute kind `"circuit"`, WEB_CONTRACT §4). Tests
`server/tests/test_circuit.py`; validation script `python -m server.compute.circuit.validate`.

**요약 (KO).** 단일 트랜지스터 래치(SOI n-MOSFET, STL)를 내부 상태 (u, r)와 바디 전하 Q_B를 갖는
회로 소자로 만들고, 소형 넷리스트 MNA 과도해석기(numba)로 적분한다. 결정론 모드는 암시적 BE/TRAP +
뉴턴, 확률 모드는 Eq. 2의 사건 증분(포아송 단위 사건 + 전리충돌 클러스터 + 손실)을 명시적 tau-leap으로
적용한다. 준정적 fold(V_LU = 3.7037 V, V_LD = 2.5979 V)를 0.4 V/s 램프에서 0.25 mV 이내로 재현하고,
확률 모드의 V_LU 분포는 정확한 compound-jump 역방향 방정식(FPT)과 통계 오차 내에서 일치한다.

**Summary (EN).** The single-transistor latch is a circuit element with internal unknowns (u, r)
and state Q_B, integrated by a small-netlist MNA transient solver written in numba. Deterministic
mode: implicit BE/TRAP + Newton. Stochastic mode: the Eq. 2 event increments (Poisson unit events +
impact-ionisation clusters + losses) applied as an explicit tau-leap. The quasi-static folds are
reproduced within 0.25 mV on a 0.4 V/s ramp, and the stochastic V_LU distribution agrees with the
exact compound-jump backward equation (FPT) within the Monte-Carlo error.

---

## 1. 소자 수식 / Element formulation

> **KO.** STL 소자는 단자 D, G, S와 내부 미지수 u(소스–바디 준페르미 분리), r(드레인 접합 역바이어스)를
> 갖는다. 매 시점 두 식을 푼다: E1 V_D(u,r) = v_D − v_S, E2 전하식 Q(u,r) − Q_c − θhF(u,r) = 0.
> 전하 좌표 Q는 compound-FPT 격자 좌표(`S.state` 5+6+7열)에서 C_ox·V_GS를 뺀 것과 정확히 같다.
> 엔진 영역 밖(r < 0: 드레인 순방향, u < 0: 소스 역방향)은 대칭 순방향 드레인 다이오드와
> 저주입 다이오드 연장으로 처리하고 경고로 알린다.

Terminals D, G, S; light input I_PH(t) (A) → p[13]; gate voltage V_GS = v_G − v_S → p[11] (DC in all
benches). Internal unknowns: **u** (source–body quasi-Fermi splitting, V) and **r** (drain-junction
reverse bias, V). With z = `photo_mean.components(u, r, p, N_A, V_bi, rg, fg, table)` (19 numbers):

| quantity | expression | code |
|---|---|---|
| drain voltage | V_D(u,r) = u + r + V_T h_d + (R_c + R_acc) I_D | z[0] |
| drain current | I_D(u,r) | z[1] |
| net hole current into the body | F = G − L | z[2] |
| unit-event current (junction BTBT + GIDL + photo) | I_unit = z[8] + z[9] + z[18] | |
| hole generation | q G = z[1] − z[3] − z[16] | |
| hole loss (bulk SRH + out-diffusion + junction SRH) | q L = z[5] + z[6] + z[7] | |
| II hole current (clusters) | I_II = q G − I_unit | |

Body-charge coordinate (the compound-FPT lattice coordinate `make_lattice` uses, i.e. `S.state`
columns 5 + 6 + 7, minus the constant C_ox V_GS; checked to 1e-12 relative in the tests):

  Q(u, r; V_GS) = C_ox (ψ − V_GS) + Q_exc + q N_A A L_n,
  ψ = u − V_T ln(1 + δ/N_A),  δ/N_A = z[10],  Q_exc = z[13] − C_ox u,  L_n = z[11] (cm),

with C_ox = `m.COX_F` = 2.449e-16 F, A = `m.AREA_CM2` = 1e-10 cm², N_A = `MODEL.na` = 2.2958e17 cm⁻³,
V_T = 25.852 mV.

Element equations at every time point (E1 in V, E2 scaled by 1/C_ox to V):

  E1: V_D(u, r) − (v_D − v_S) = 0
  E2: Q(u, r) − Q_c − θ h F(u, r) = 0

| mode | Q_c | θ |
|---|---|---|
| deterministic BE | Q_n | 1 |
| deterministic TRAP (h ≤ 2 τ_rel) | Q_n + h/2 F_n | 1/2 |
| stochastic, event-level tau-leap | Q_n + q (N_unit + Σ k_i − N_loss) | 0 |
| stochastic, Gaussian tier | Q_n + η, η ~ N(0, D h (1 + h/2τ_rel)) | 1 |
| drift only | Q_n | 1 |

The element stamps I_D into the drain node and −I_D into the source node (the model's quasi-static
terminal current). The gate draws no current (the body–gate and depletion displacement currents,
≤ |dQ_B/dt|, are not stamped; see §9).

### 1.1 도메인 밖 확장 / Extensions outside the engine's domain

The engine's steady state never uses u < 0 or r < 0, but transients do:

* **r < 0 — drain junction forward biased** (fast down-ramps from the LRS, and the photovoltaic
  body at V_D ≈ 0 under light). The core is evaluated at r = 0 and a **symmetric forward drain
  diode** is added: the same n+ emitter hole-diffusion saturation current as the source
  (I_sd = q A D_n n_i² / (N_A L_ref β), without the source-edge state φ_E) plus depletion SRH
  q A w_d(r) n_i /(2 τ_j) expm1(−r/2V_T):
  I_fwd = I_sd expm1(−r/V_T) + q A w_d n_i/(2τ_j) expm1(−r/2V_T); L += I_fwd, F −= I_fwd,
  I_D −= I_fwd; the channel current is re-evaluated with the true u + r (same closed form as
  `photo_mean`); V_D = V_D(u,0) + r + (R_c + R_acc) ΔI_D; L_n grows by w_d(0) − w_d(r).
  Continuous at r = 0 (the Jacobian has a kink). Reported in `warnings` with min r and duration.
* **u < 0 — source junction reverse biased** (`components` returns NaN for any u < 0; happens after
  fast down-ramps in the dark because of the drain-depletion coupling du/dr|_Q ≈ −∂Q/∂r / ∂Q/∂u ≈ 0.03).
  The core is evaluated at u = 0 and every current X ∈ {I_D − I_ch, G, L, I_unit} is continued with the
  low-injection diode law X(0) + X′(0⁺) V_T expm1(u/V_T) (C¹ at u = 0, saturating at the reverse
  generation current); the channel current uses the true u; Q uses the exact ψ(u) and source
  depletion width w_s(u). Reported in `warnings`.

Both are extensions outside the calibrated model and are documented as such.

## 2. MNA 및 뉴턴 / MNA and Newton

> **KO.** 미지수는 노드 전압, 전압원 전류, 셀별 (u, r). STL 야코비안은 u, r에 대한 유한차분,
> 반복당 |Δu| ≤ 50 mV·|Δr| ≤ 1 V 제한과 역추적(유효 영역 유지)을 쓴다. 초기점은 빈 바디(u = 0)에서
> 의사-과도(pseudo-transient) BE로 찾으므로 쌍안정 영역에서는 HRS가 선택된다. 매 스텝 회로 제약을
> 포함한 dF/dQ로 국소 완화시간 τ_rel을 계산한다.

Unknowns x = [node voltages (node 0 = ground), V-source branch currents, (u_k, r_k) per STL].
Residuals: KCL at every node (R: (v_a − v_b)/R; C: companion BE C/h Δv or TRAP 2C/h Δv − i_n;
V source: branch current + v_a − v_b − V(t) = 0; STL: ±I_D), E1/E2 per STL, plus g_min = 1e-18 S to
ground. Dense `np.linalg.solve` (the benches have ≤ 12 unknowns).

* STL Jacobian block: forward finite differences of the element in u and r (δ = 1 µV; backward
  difference if the forward point is outside the domain). `components` costs 10–25 µs per call.
  Cells whose V_GS = v_G − v_S can move (source not grounded, or gate not held by a constant voltage source
  to ground; custom circuits) also get ∂/∂V_GS columns (δ = 1 µV) at the gate and source nodes of the
  KCL, E1 and E2 rows — V_GS enters I_D, V_D, F and the charge coordinate (−C_ox V_GS). Without them Newton
  failed at the latch-up of a cell with a 100 kΩ source resistor (V_S jumps by 0.44 V; time-step underflow).
  The benches (grounded source, DC gate) do not get them and are unchanged.
* Damping/limiting: |Δu| ≤ 50 mV and |Δr| ≤ 1 V per iteration (whole step scaled); backtracking
  (halving, ≤ 14×) until every element evaluates finitely (the source barrier must exist and the
  neutral base must not collapse).
* Convergence: |Δu| < 1e-7 V, |Δr| < 1e-6 V, |Δv| < 1e-6 (1 + |v|) V and |E1|, |E2| < 1e-5 V; ≤ 14
  iterations; partial derivatives are reused from the previous step and refreshed when the
  previous step needed > 2 iterations, after a rejection, or when convergence slows.
* Warm start: linear extrapolation (deterministic) or the linear response of the charge-fixed
  solution x_n + (dx/dQ) ΔQ_pred with ΔQ_pred = (Q_c − Q_n + θ h F_n)/(1 − θ h dF/dQ) (stochastic);
  |Δu_pred| ≤ 20 mV.
* **DC operating point**: u = 0 (empty body) with capacitors open, then a pseudo-transient BE
  continuation (h = 1e-12 s × 3^k up to 1e8 s) — it finds the state reachable from an empty body,
  i.e. the HRS when the bias is inside the hysteresis window.
* **Local relaxation time**: after each step, J(θ = 0) s = e_E2 / C_ox gives dx/dQ along the circuit
  constraints (R_s, C_d companion, other cells); dF/dQ = ∂F/∂u s_u + ∂F/∂r s_r and
  τ_rel = 1/|dF/dQ|. Typical values: HRS near the fold 5–50 µs, HRS at low V_D in the dark ms–0.1 s,
  LRS 0.03–50 ns (→ µs at the latch-down fold).

## 3. 결정론 시간 적분 / Deterministic time stepping

> **KO.** BE(기본) 또는 TRAP(τ_rel보다 긴 스텝에서는 자동으로 BE). |Δu|, |Δln I_D|, |Δv|, 전하 LTE로
> 스텝을 조절하고 파형 꺾임점에 정확히 착지한다. 느린 램프에서는 큰 스텝으로 준정적 가지를 따라가고,
> fold에서는 스텝 거절과 축소로 빠른 전이를 해상한다.

BE (default) or TRAP (θ = ½; automatic BE on stiff steps h > 2 τ_rel to avoid trapezoidal ringing).
Error measure per accepted step (reltol = 1e-3 values, scaled linearly with reltol and clipped):

  ε = max(|Δu|/10 mV, |Δ ln I_D|/0.2, |Δv_node|/20 mV, LTE_u/30 µV),
  LTE_u = h/2 |F_{n+1} − F_n| / (∂Q/∂u),

LTE_u = 30 µV (it was 1 mV) converges the first-order BE error of the slow passage through the fold:
the ramp lag is within ≈ 1 % of the converged value (BE at reltol 1e-5 / TRAP) at 40 and 1200 V/s
(1 mV left 13 % / 6 %), for +20 % steps on a load line and +50 % on a pulse train.

reject when ε > 1.5 (h ← h max(0.1, 0.7/ε)); otherwise h_next = h min(2.5, max(0.3, 0.8/ε)).
Breakpoints at every waveform corner and sample time (steps land exactly on them), dt_max =
t_end/2000, dt_min = max(1e-15, 1e-13 t_end); Newton failure → h/4. Because BE solves
F(Q_{n+1}) = ΔQ/h, large steps on a slow ramp stay on the quasi-static branch with the correct
adiabatic lag; at a fold the missing solution forces step rejection until the fast transition is
resolved by the |Δu| and |Δ ln I_D| limits.

## 4. 확률 모드 / Stochastic mode (Eq. 2)

> **KO.** 셀마다 스텝 전에 계층을 고른다. (1) 사건 수준 명시적 tau-leap: ΔQ = q(N_unit + Σ클러스터
> 크기 − N_loss), 포아송/복합 포아송(현재 r에서의 클러스터 pmf, `compound_fpt.backward`와 같은 평균
> 보존 도약률), 평균 > 100이면 가우스 극한, h ≤ 0.05 τ_rel·사건 ≤ 200개. (2) 빠르게 완화하는 상태
> (2 ns ≤ τ_rel)는 분산 보정 가우스 + 암시적 드리프트, h ≤ τ_rel/2. (3–5) 잡음이 전이를 일으킬 수
> 없는 곳 — 장벽이 정상 요동 SD의 12배 이상인 '잡음 대역' 밖(fold 너머 쪽은 경계 없음, 대역 진입
> 4 τ_rel 전부터 잡음 적용), τ_rel < 2 ns, 또는 래치된 셀(`ld_carrier_noise` = false) — 은 드리프트만
> 적분한다. 잡음 대역 덕분에 0.4 V/s 논문 소자도 사건 수준으로 계산 가능하다(사이클당 약 5 s).
> 래치 상태는 전류 문턱이 아니라 바디의 가지(u가 LD fold의 u_j에 닿으면 래치, LU fold의 u_i로
> 내려오면 해제)이고, I_D 문턱(10 nA / 1 nA)은 전환 시각만 정한다. 채널 전도로 I_D가 문턱을 넘는 것은
> 래치업이 아니다.

Each cell is advanced in one of the following tiers, chosen before each step from its state n:

| tier | condition (checked in this order) | charge update | step limit |
|---|---|---|---|
| 4 drift only | cell latched (body on the LRS branch, see **Latch state**) and `ld_carrier_noise` = false | BE, no noise | deterministic error control |
| 5 drift only | V_DS outside the cell's **noise band** (below) and the drive does not enter the band within 4 τ_rel (look-ahead) | BE, no noise | deterministic error control |
| 3 drift only | τ_rel < gauss_tau_min (2 ns) and tau_frac·τ_rel < noise_dt_min | BE, no noise | deterministic error control |
| 1 event-level tau-leap | tau_frac·τ_rel ≥ noise_dt_min (2 ns) | explicit: Q_{n+1} = Q_n + q(N_unit + Σ k_i − N_loss) | h ≤ tau_frac τ_rel (0.05), ≤ 200 expected events, ≤ 10 mV drift (h ≤ Δu_max (∂Q/∂u)/\|F\|) |
| 2 Gaussian | gauss_tau_min ≤ τ_rel (fast-relaxing, e.g. the LRS near the latch-down fold) | implicit drift + η ~ N(0, D h (1 + h/2τ_rel)) | h ≤ τ_rel/2 |

**Noise bands.** Carrier noise can only cause a transition where the barrier between the current
stable state and the saddle (unstable branch at the same V_D) is a few fluctuation SDs. From the
quasi-static branches (`MODEL.classify`) the runner computes along the HRS and the LRS
z(V_D) = |Q − Q_saddle| / √(D τ_rel/2) and the bands: unlatched cell [V_HRS(z = z_max), ∞),
latched cell (−∞, V_LRS(z = z_max)], widened by 4σ of the local-state fold shifts
(|dV_LU/dφ_G| ≤ 1, |dV_LU/dφ_E| ≤ 5, |dV_LD/dφ_G| ≤ 0.05, |dV_LD/dφ_E| ≤ 50 V/V; other actions or
σ_E > 2 mV → no band restriction). Outside the bands the element is integrated drift-only.
There is no edge beyond the fold (an earlier version cut the noise at V_LU + 0.25 V / V_LD − 0.25 V):
an unlatched cell above V_LU is in the post-fold passage, where the noise shapes V_LU on fast ramps
and the delay of supra-fold pulses. **Look-ahead:** a cell below its band is also noise-resolved when
the drive (the main PWL source) reaches the band within 4 τ_rel of the cell (`SolverConfig.
noise_lookahead`), so the stationary fluctuation is built up before the escape region even when the
band is crossed faster than τ_rel (fast ramps, pulse edges); on slow ramps this moves the effective
edge by < 0.1 mV (0.4 V/s × 4 × 50 µs). Measured (paper −2 V dark, 3e4 V/s to 6 V, 40 runs, same
seeds): previous bands 4.2508 V / SD 5.4 mV; now 4.2594 V / 13.1 mV, identical to no band restriction
at all, and 4.2593 V / 10.7 mV with tau_frac 0.01 and 20 events per step. Supra-fold pulses (3.98 V,
12 runs): the delay SD went from 0.1 µs to 1.2 µs (+0.9 µs mean). A device **without latch window** is
integrated drift-only (the noise cannot cause a transition) unless local states are on.
z_max = 12 (`solver.noise_z_max`) was calibrated against the exact compound-jump hazard (the
Gaussian z underestimates the heavy-tailed cluster noise, so a large margin is needed):

| device | z = 7.4–7.6 | z = 8.9 | z = 10.4–10.6 | z = 13.4–13.6 |
|---|---|---|---|---|
| paper, V_G = −2 V dark | 3.3 /s | – | 2.1e-3 /s | 1.1e-7 /s |
| V_G = −1.8 V dark | 3.8 /s | – | 3.3e-3 /s | 2.2e-7 /s |
| V_G = −1.8 V, 2.63 pA | – | 0.6 /s (z = 6.7: 40 /s) | – | – |

so outside the bands the escape hazard is ≲ 1e-3 /s (photo condition: extrapolated), negligible for
every waveform the simulator accepts. Resulting band edges (no local states): paper −2 V dark HRS
from 3.59 V, LRS up to 2.82 V; photo condition HRS from 2.93 V, LRS up to 2.81 V; photo preset
(γ = 0.2794, δφ_G0 = +74 mV) HRS from 3.70 V.
The bands make event-level runs affordable: the dark paper device at 0.4 V/s needs ≈ 2.3e5 steps
(5 s) per cycle instead of ≈ 4e8.

Why not a large-step Gaussian tier outside the bands? With h ≫ τ_rel inside a bistable region the
implicit equation Q − hF(Q) = Q_c receives kicks of ≈ h/τ SDs and Newton can land on the root of the
other basin: V_LD moved from 2.656 V (resolved) to 2.84 V in a test. Drift-only is exact in the mean
and loses only the stationary fluctuation outside the bands; the look-ahead switches the noise on
4 τ_rel before the band is entered, so that fluctuation is present when it matters.

**Event draws (tier 1)**, rates from the state n (all mean-preserving, E[ΔQ] = h F exactly):
N_unit ~ Poisson(I_unit h/q) (+ reverse-generation terms of the u < 0 extension), N_loss ~
Poisson(q L h/q); II clusters: event rate λ_II = (I_II/q) P₁/M₁ with P₁ = Σ_{k≥1} p_k(r),
M₁ = Σ k p_k(r) — i.e. jump rate for size k = (I_II/q) p_k / M₁, exactly as `compound_fpt.backward`;
sizes k ~ p_k(r)/P₁ from the extended avalanche kernel `ct.cf.pmf` (installed by `setup_photo`,
reverse bias 0.7–5 V, K = 27, linear interpolation in r, clamped like `np.interp`). Any count whose
mean exceeds 100 uses the Gaussian limit (cluster sum ~ N(I_II h/q, (I_II h/q) M₂/M₁)). Node voltages
and (u, r) are then solved with Q fixed (θ = 0), capacitors BE.

**Gaussian tier.** D = q² (R_up + R_down + (I_II/q) M₂/M₁). For the linearised dynamics
δ_{n+1} = (δ_n + η)/(1 + h/τ) the stationary variance is var(η)/((1 + h/τ)² − 1) = D τ/2 — the exact
Ornstein–Uhlenbeck variance for any h; the escape dynamics are resolved only for h ≲ τ/2 (with
h = 2τ or 5τ the latch-down voltage was biased by +140…+190 mV), hence gauss_tau_frac = 0.5. Newton
tolerance is relaxed 20× in tiers 1–2 (the stored charge is the integration-formula value
Q_c + θhF_{n+1}, so the event bookkeeping stays exact).

**Why the latched state is drift-only by default.** The LRS relaxes in 0.03–50 ns. Resolving its
carrier noise near the latch-down fold (τ_rel 2–40 ns, fluctuation SD of u 10–20 mV, thousands of
events per τ_rel) costs ≈ 1e4 (no local states) to 5e4 (with the band widened by local states) steps
per down-sweep. With `stochastic.ld_carrier_noise = true` (the
same key as the device MC) tiers 1/2 are applied in the LRS too: noise-induced early latch-down then
raises V_LD (§7, V5). By default V_LD is the deterministic escape at the LRS fold (plus local-state
effects), matching the device-level MC default.

**Local states** (`stochastic.local_state`, same block as §1 of the contract): `frozen` — one draw
δ ~ N(0, σ) per run and cell (the whole waveform, all cycles); `evolving` — Ornstein–Uhlenbeck in
time, δ(t+h) = δ e^{−h/τ} + σ √(1 − e^{−2h/τ}) ξ, continuous across cycles. Action points: `gidl` →
p[9] (added to the calibrated mean + device centre δφ_G0), `local_avalanche` → p[23] (with
ext.aloc = p[21] = 0 the local avalanche path does not exist, so aloc = 1.0 is substituted with a
warning — the same rule as the device-level MC, `compute/stochastic.StateMap`), `junction` → p[19],
`multiplication` → p[20]; the emitter state σ_E → p[10]. `carrier_noise = false` integrates
deterministically with the local states only (without local states nothing is random: one run and a
warning instead of n identical runs).
`acquisition_trend` is not applied (warning). Seeds: run r uses seed + 1000003 r (numba RNG) and
`default_rng([seed, 7349, r])` for the local-state draws; the same seeds are used at every sweep
point (common random numbers).

**Latch state and detection.** The latch state is the body's branch, not a current threshold. The
quasi-static branch (`MODEL.classify`) is parameterised by u: HRS u < u_i (u at the latch-up fold),
unstable branch u_i < u < u_j, LRS u > u_j (u at the latch-down fold); paper −2 V: u_i = 0.567 V,
u_j = 0.761 V. A cell latches when u reaches u_j and unlatches when u falls to u_i (the hysteresis is
the unstable-branch range, so the state does not chatter under noise; no latch window → never
latched). This state selects the noise tier (tier 4) and band, and is what P_sw, P_retained and
`P_latched` report. An event is the switch of that state, **timed** at the current-threshold crossing
inside the switching transient — latch-up at I_D ≥ `detect.i_threshold_A` (10 nA) with u ≥ u_i,
latch-down at I_D < i_threshold / `detect.hysteresis` (default 10 → 1 nA) with u ≤ u_j (log-linear
interpolation inside the step; v_DS and v_src at that time) — or, without such a crossing, when the
body reaches the new branch. For the default thresholds and the STL regime this is exactly the
I_D-crossing time (the 10 nA crossing lies between u_i and u_j), so V_LU / V_LD are unchanged.
Thresholds that do not lie between the branch currents at their fold (HRS 15 pA / LRS 13 µA at the
LU fold, HRS 1e-14 A / LRS 16.7 nA at the LD fold for the paper device) only move the event time to
the body crossing and are warned about. I_D crossings of i_threshold while the body stays on the HRS
— channel conduction when V_G is above the channel threshold (≈ −0.5 V), or HRS leakage — are **not
latch events**; their count is reported as a warning. The branches come from
`deterministic.classify_checked` (the deterministic engine's check): a fold that `MODEL.classify`
fitted across an untraced gap of the locus (channel-on regime, V_G ≳ −0.2…0 V; at V_G = +1 V the
fold parabola spans the rows u = 0 → 0.885 and extrapolates to 9.3 V) is rejected as "no latch",
so both tabs agree. Example (1200 V/s): V_G = −0.6 V (no window) and V_G = +1 V (locus gap) give no
latch-up (they used to report V_LU = 45 mV and 3.6 mV).
Per cycle, V_LU is the first latch-up of the cycle and V_LD the first latch-down after it.

**Residual body memory.** Q_B is integrated continuously across cycles and pulses; any charge left
from the previous cycle (e.g. short periods, v_base inside the window) is included automatically.

## 5. 벤치 / Benches (defaults in `benches.py`, resolved values returned in `result.bench_params`)

> **KO.** load_line(직렬 R + 삼각파, 사이클별 V_LU/V_LD 분포), pulse(펄스 열, 펄스 끝 래치 비율 P_sw,
> 진폭 스윕), pbit(드레인 펄스 + 소스 저항 + 비교기, P(1)·자기상관, V_G/광 스윕), coupled(저항 결합 두 셀).
> 바디 전하는 사이클 사이에 연속 적분되므로 잔류 바디 기억 효과가 자동으로 포함된다.

`None` = automatic. All voltages V, times s, R Ω, C F.

| bench | circuit | defaults |
|---|---|---|
| `load_line` | V_src triangle v_min → v_max → v_min, series R_s, C_d to ground, DC gate | v_min 0, v_max = preset vd_max (paper 4 V, photo 5 V), rate = preset (0.4 / 1200 V/s; stochastic falls back to 1200 V/s when infeasible), n_cycles 1 (≤ 50), R_s 1 kΩ, C_d 2 fF, vg_V = device.vg |
| `pulse` | trapezoidal pulses through R_s, C_d | v_base 0, v_amp = fold V_LU + 0.10 V, width (flat top) 200 µs, period 1 ms, rise/fall 10 µs, n_pulses 10 (≤ 2000), delay 0, R_s 1 kΩ, C_d 2 fF, amplitudes_V [] (sweep) |
| `pbit` | drain pulses (V_clk directly on the drain) → STL → source → R_S → ground; comparator (CMP element) on the source node | v_low 0, v_high = fold V_LU − 0.015 V, period 1 ms, width 200 µs, rise/fall 20 µs, n_clocks 50 (≤ 5000), R_S 100 kΩ, v_ref = R_S · 1 µA (0.1 V), vg_list_V [], light_list_pA [] |
| `coupled` | common ramp or pulse source; R_s1, R_s2 to the drains d1, d2; R_c between d1 and d2 | source "ramp", ramp and pulse keys as above, R_s1 = R_s2 = 100 kΩ, R_c 1 MΩ, C_d 2 fF each, vg2_V = vg_V, iph2_pA = device light |

* `pulse`: P_sw = fraction of pulses in which the cell is latched (§4 latch state) at the end of the
  flat top; P_retained = still latched at the end of the period; switching delay from the
  pulse start (first latch-up in the period). `amplitudes_V` → sweep `P_sw_vs_amplitude`
  (n_runs per point, binomial error bars).
* `pbit`: bit = comparator output [V(R_S) = R_S I_D > v_ref] at the end of the pulse's flat top (a latched
  cell carries its LRS current: V(R_S) ≈ 0.44 V at 100 kΩ; unlatched: µV). P(1) (comparator output),
  `P_latched` (fraction of clocks with the cell latched), pooled lag-1 autocorrelation. With the channel on
  (V_G above ≈ −0.5 V) the comparator reads
  1 without a latch; this is warned about per configuration. `vg_list_V` (or, if empty, `light_list_pA`)
  → sweep `P1_vs_vg` / `P1_vs_iph` (comparator P(1)).
* Truncated runs (step budget or step failure): pulses, clocks and cycles the run did not reach are
  censored — excluded from P_sw, P_retained, P(1), lag1, the sweep points and the n_latch_up
  denominator (a warning gives the count); `n_latch_up` = latch-ups / observed cycles.
* `coupled`: per-cell V_LU/V_LD (ramp) or P_sw (pulse), correlation between the cells, P(both),
  mean latch-up time difference.
* Physics note: supply edges faster than the body relaxation (µs) kick the floating body through
  the drain-depletion charge (Δu ≈ 0.03 Δr at fixed Q_B) and can trigger latch-up below the static
  fold — a genuine prediction of the charge-coordinate model (default edges 10–20 µs keep this small).

## 6. 매개변수 / Request parameters

```jsonc
"solver": { "method": "BE" | "TRAP", "dt_min_s": null, "dt_max_s": null, "reltol": 1e-3,
            "max_steps": 1000000,          // per run, cap 2e6
            "tau_frac": 0.05, "max_events_per_step": 200, "noise_dt_min_s": 2e-9,
            "gauss_tau_min_s": 2e-9, "gauss_tau_frac": 0.5, "noise_z_max": 12, "gauss_threshold": 100 },
"stochastic": { "seed": 2026092920, "n_runs": 20 /* cap 200 */, "carrier_noise": true,
                "ld_carrier_noise": false, "local_state": { "mode": "none" | "frozen" | "evolving", ... } },
"detect": { "i_threshold_A": 1e-8, "hysteresis": 10 }
```

**Feasibility.** Before running, the quasi-static HRS/LRS branches of each cell (`MODEL.classify`,
301-point grid) are evaluated for τ_rel(V_D), the event rate and the barrier z; the drive is sampled
on a voltage-resolved grid (≤ 2 mV per point on ramps) and walked with the hysteretic HRS/LRS state
and the step rules above (noise bands with the 4 τ_rel look-ahead). With carrier noise the HRS → LRS
switch is placed at the median of the noise-induced escape, from an empirical compound-noise hazard
λ(z) = 3.3 s⁻¹ · 10^(−1.2 (z − 7.5)) fitted to the z_max calibration table (likewise LRS → HRS with
`ld_carrier_noise`), or at the fold. Fixed costs: 250 steps per latch transition, 60 per waveform
corner, 6 per other breakpoint. Measured ratio estimate/actual (16 cases): load lines 0.80–1.08
(paper 0.4 V/s stochastic 0.98, previously 1.77; `ld_carrier_noise` 0.89, previously 0.45),
deterministic pulse / p-bit 0.98–0.99, stochastic pulse / p-bit trains with probabilistic switching
1.2–1.6 (the estimate assumes a switch in every pulse; conservative). A run is refused (ValueError)
when the estimate exceeds 2 × max_steps (the message names the cause: event-level noise, or for
deterministic runs the waveform length), warned when above 0.5 × max_steps, and a request is refused
above 4e7 estimated steps in total.
**Event-level stochastic sweeps at 0.4 V/s are expensive**: the step is bounded by 0.05 τ_rel inside the
noise band until the cell escapes. Paper device V_G = −2 V dark: band edge 3.59 V, escape near
3.645 V, τ_rel ≈ 12–50 µs → ≈ 2.3e5 steps (estimate 2.24e5), ≈ 5 s per cycle (feasible); photo
condition (V_G = −1.8 V, 2.63 pA): band edge 2.93 V, escape near 3.2 V, τ_rel ≈ 6 µs → estimate
1.4e6 steps per cycle → above the default max_steps = 1e6 (warned, the run would be truncated;
refused when max_steps < 7e5); without an explicit rate the ramp falls back to 1200 V/s (use ≥ 10 V/s, the deterministic mode, `carrier_noise = false`, or the device-level hazard MC
`sweep_mc`). `ld_carrier_noise` at slow ramps is refused for the same reason (τ_rel of the LRS ≈ 5–40 ns).
A running run stops at max_steps (warning, partial results; the unreached part is censored, §5).

## 7. 검증 결과 / Validation results

> **KO.** (V1) 결정론 0.4 V/s 부하선: V_LU 3.70393 V(fold 3.70369, +0.24 mV 램프 지연), V_LD 2.597855 V
> (fold 2.597866). (V2) 확률 모드, V_G −1.8 V·2.63 pA: 120 V/s에서 평균 3.2049 V(표준오차 2.9 mV) 대
> FPT 3.2077 V, SD 35.5 대 34.5 mV. 1200 V/s에서는 FPT가 fold에 두는 53 % 질량을 회로가 fold 이후
> 느린 통과로 해상하므로 평균이 +29 mV 높지만, fold 도달 비율(0.58 ± 0.035 대 0.53)과 fold 아래
> 분포는 일치한다. (V3) 고정 바이어스 MFPT 비 0.93 ± 0.08. (V6) 엔진 검증 노드(−2 V, 0.4 V/s)
> 3.6453 V·7.6 mV 대 3.6442 V·8.0 mV.

Reproduce with `python -m server.compute.circuit.validate --out results.json` (~12 min; `--quick`
≈ 3 min; `--only v1,v6` for single checks). Paper calibration throughout; "photo condition" = V_G = −1.8 V, I_PH = 2.63 pA, paper
calibration (the condition of the handoff FPT node, folds 3.2913 / 2.5962 V).

**V1 — deterministic load line**, paper device V_G = −2 V dark, R_s = 100 Ω, C_d = 1 fF:

| ramp | V_LU (drain) | fold (MODEL.classify) | lag | V_LD (drain) | fold | lag | median \|Δlog₁₀ I_D\| HRS / LRS | steps | time |
|---|---|---|---|---|---|---|---|---|---|
| 0.4 V/s | 3.70393 V | 3.70369 V | +0.24 mV | 2.597855 V | 2.597866 V | −0.01 mV | 9e-6 / 4e-5 dec | 2895 | 0.7 s |
| 40 V/s | 3.70869 V | | +5.00 mV | 2.59755 V | | −0.32 mV | 1e-3 / 4e-5 dec | 2771 | 0.5 s |
| 1200 V/s | 3.74771 V | | +44.0 mV | 2.59427 V | | −3.59 mV | 0.03 / 4e-5 dec | 2660 | 0.5 s |

The lag is physical (slow passage through a saddle-node) and converged to ≈ 1 %: BE at reltol 1e-5
(≈ 12 000 steps) gives +0.245 / +5.07 / +44.4 mV (LU) and −0.32 / −3.62 mV (LD), TRAP at reltol 1e-3
+0.245 / +4.79 / +44.4 mV. (With the former LTE_u = 1 mV the default BE values were +0.21 / +4.42 /
+41.8 mV, 6–13 % short.) The lag grows close to rate^{2/3} at low rates (×100 in rate → ×21.0,
rate^{2/3}: ×21.5) and more slowly at 1200 V/s (×30 → ×8.8 vs 9.7), where 44 mV is no longer small.
The trajectory lies on the quasi-static HRS/LRS branches (`A.branches(-2.0)`) to 1e-5 decades in I_D
at 0.4 V/s (the default R_s = 1 kΩ / C_d = 2 fF gives V_LU 3.70393 V as well).

**V2 — stochastic load line, carrier noise only, photo condition, vs the compound-jump FPT**
(`A.hazard(-1.8, 2.63e-12, rate=…)`, `photo_fpt.quantiles`):

| ramp | runs | circuit mean / SD | FPT mean / SD | circuit q05 / q25 / q50 / q75 / q95 | FPT q05 / q25 / q50 / q75 / q95 | notes |
|---|---|---|---|---|---|---|
| 120 V/s | 150 | 3.2049 V (SE 2.9 mV) / 35.5 mV | 3.2077 V / 34.5 mV | 3.1356 / 3.1824 / 3.2122 / 3.2293 / 3.2532 | 3.1458 / 3.1870 / 3.2111 / 3.2321 / 3.2584 | FPT fold atom 0.2 %; KS distance 0.066 (5 % critical value 0.11) |
| 1200 V/s | 200 | 3.2991 V / 47.9 mV | 3.2703 V / 31.7 mV | 3.2156 / 3.2711 / 3.3017 / 3.3350 / 3.3695 | 3.2026 / 3.2569 / 3.2913 / 3.2913 / 3.2913 | fraction beyond the fold 0.580 ± 0.035 vs FPT fold atom 0.529; below the fold: mean 3.2548 vs 3.2467 V, SD 32.6 vs 32.9 mV, KS 0.106 (5 % critical ≈ 0.15 for 84 values) |

Numbers of the current code (latch state from the body branch, noise bands without an edge beyond the
fold and with the 4 τ_rel look-ahead, LTE_u = 30 µV); the earlier version gave 3.2091 V / 34.6 mV at
120 V/s and 0.552 ± 0.025 / 3.2521 V (400 runs) at 1200 V/s — the same within the Monte-Carlo error.

At 120 V/s (no fold atom) the two independent methods agree within the Monte-Carlo error in mean,
SD and every quantile — the key cross-check of the event-increment implementation (unit events,
cluster sizes from the pmf at the current r, mean-preserving rates) against the exact backward
equation. At 1200 V/s the FPT (quasi-static hazard) assigns the 53 % of cycles that survive up to
the fold exactly *to* the fold, whereas the circuit resolves the post-fold slow passage (the same
mechanism as the deterministic +44 mV lag in V1), so the mean differs by +23…29 mV while the fraction
reaching the fold and the distribution below it agree. Across all 1200 V/s experiments (n = 1000:
different seeds, with and without noise bands, tau_frac 0.05/0.02/0.01, 200/50/12 events per step)
the fraction beyond the fold was 0.55 ± 0.02 and the below-fold mean 3.251–3.258 V; individual
100–200-run samples scattered by ±0.05 (one 100-run sample reached 0.69 on an intermediate code
version). Cost: 3 500 steps (0.37 s) per 1200 V/s cycle, 10 600 steps (0.56 s) per 120 V/s cycle.

**V3 — fixed bias** V_D = 3.20 V (20 µs step, hold 3 ms), escape time of the event-level circuit
simulation vs the exact MFPT of `compound_fpt.backward` on the `make_lattice` lattice (same
generator): MFPT(circuit) / MFPT(exact) = 0.93 ± 0.08 at 3.20 V (150 runs, exact 0.491 ms),
1.12 ± 0.08 (default steps) and 1.03 ± 0.08 (tau_frac 0.02, 25 events/step) at 3.27 V (exact 77.8 µs).

**V4 — step-size convergence** (1200 V/s, same seed): tau_frac/events-per-step 0.05/200 (200 runs)
→ mean 3.2991 V, fraction beyond fold 0.580, below-fold mean 3.2548 V (3 500 steps/run); 0.02/50
(100 runs) → 3.2948 V, 0.530, 3.2551 V (5 300 steps). Earlier version (200 runs each): 0.05/200 →
3.2932 V, 0.530, 3.2555 V; 0.02/50 → 3.2962 V, 0.525, 3.2568 V; 0.01/12 → 3.2943 V, 0.535, 3.2578 V.
Converged within the Monte-Carlo error (±3–5 mV, ±0.035); the default is the cheapest setting.

**V5 — latch-down with LRS carrier noise** (1200 V/s, photo condition, 20 runs): default (latched
cell drift-only) V_LD = 2.5924 V (SD 0: deterministic escape at the LRS fold 2.5962 V minus the ramp
lag); `ld_carrier_noise = true` V_LD = 2.6528 V, SD 12.2 mV (61 000 steps, 7.5 s per cycle; the earlier
version gave 2.657 V / 19.6 mV): carrier noise triggers the latch-down ≈ 55 mV above the fold. An intermediate fully event-level resolution
(noise_dt_min 1.5e-10 s, 2.8e5 steps, 29 s per cycle) gave the same ≈ 2.65 V.

**V6 — engine VALIDATION.md FPT node reproduced by the circuit** (paper device V_G = −2 V dark,
centre states, carrier noise only, 0 → 4 V at 0.4 V/s, 40 runs): V_LU = 3.6453 V (SE 1.2 mV),
SD 7.6 mV vs FPT 3.6442 V, 8.0 mV (VALIDATION.md "≈ 3.644 V, SD ≈ 8 mV"), KS 0.064; 2.1e5 steps,
4.3 s per cycle thanks to the noise bands.

## 8. 성능 / Performance (4-CPU shared container, warm numba cache)

| case | steps / run | time / run |
|---|---|---|
| deterministic load line, 0.4 V/s, 0 → 4 → 0 V | 2 900 | 0.2–0.7 s (1–2 s incl. fold/branch evaluation) |
| deterministic pulse train, 10 pulses | ~13 300 | 1.4 s |
| deterministic p-bit, 20 clocks | ~14 300 | 1.0 s |
| stochastic load line 1200 V/s, photo condition | ~3 500 | 0.37 s |
| stochastic load line 120 V/s | ~10 600 | 0.56 s |
| stochastic load line 0.4 V/s, paper device (dark) | ~2.3e5 | ~4.5 s |
| stochastic, `ld_carrier_noise` on, 1200 V/s | ~6e4 | ~7 s |
| stochastic pulse (5 pulses, 1 ms period) | ~6 400 | 0.5 s |
| stochastic p-bit, 20 clocks | ~17 700 | 1.2 s |

The first call in a fresh environment compiles the numba kernels (~30 s, cached in
`server/compute/circuit/__pycache__`). About 35–40 µs per accepted step (1 cell), dominated by the
2–4 `components` calls per step.

## 9. 한계 / Known limitations

> **KO.** 단자 전류는 준정적 I_D(변위 전류 미포함, 게이트는 DC). r < 0, u < 0 연장은 보정 모델 밖.
> LRS 잡음은 기본적으로 꺼져 있음(`ld_carrier_noise`로 켬, 약 20배 비용). FPT와의 1200 V/s 평균 차이는
> fold 이후 지연 때문. 잡음 대역 밖의 정상 요동은 계산하지 않음. 느린 램프의 사건 수준 계산은 소자에
> 따라 거부될 수 있음.

* The STL regime is V_G below the channel threshold (≈ −0.5 V). Above it the HRS carries channel
  current (I_D ≫ 10 nA at mV); the circuit reports no latch there unless the body switches branch of
  a latch window that `classify_checked` accepts, and warns.
* Terminal currents are the model's quasi-static I_D; the displacement currents associated with
  dQ_B/dt (gate–body C_ox coupling, depletion charges) are not stamped into the terminals; the gate
  must be DC (it is driven by an ideal source).
* The r < 0 and u < 0 extensions (§1.1) are not part of the calibrated model; the reverse-BJT
  electron current with a forward-biased drain is neglected.
* LRS carrier noise is off by default (§4); `ld_carrier_noise` makes V_LD include noise-induced
  early latch-down at ~20× the cost (1200 V/s: 0.35 → 6.7 s per cycle).
* The FPT reference places the probability mass that survives up to the fold exactly at the fold
  (no post-fold delay); the circuit resolves the post-fold slow passage, so at 1200 V/s its mean V_LU
  is ≈ 20–30 mV higher while the part below the fold agrees (§7 V2).
* The noise bands and the fold u values u_i / u_j of the latch state use the quasi-static branches of
  the centre device (z_max = 12, no edge beyond the fold, +4σ of the local-state fold shifts for the
  GIDL action; for other actions, or σ_E > 2 mV, the bands are disabled and the noise is resolved
  everywhere). Outside the bands (and more than 4 τ_rel before the drive enters them) the stationary
  fluctuations of u (a few mV) are not simulated. Local states shift u_i by a few 10 mV, small against
  u_j − u_i ≈ 0.19 V.
* Event-level stochastic runs are warned above 0.5 × and refused above 2 × max_steps (§6): e.g. the
  photo condition at 0.4 V/s (≈ 1.4e6 steps: warned and truncated at the default max_steps) or any
  `ld_carrier_noise` run at 0.4 V/s; the dark paper
  device at 0.4 V/s is feasible (≈ 5 s per cycle).
* The stochastic time step is bounded a priori; noise-driven moves are not step-rejected (rejecting
  on the drawn outcome would bias the statistics), except a hard 0.2 V |Δu| guard.

## 10. 결과 형식 / Result format (for the frontend)

WEB_CONTRACT §4 `CircuitResult`, plus: `bench_params` (resolved), `solver`, `detect`, `stochastic`,
`folds {V_LU, V_LD}` (quasi-static, cell 1), `feasibility {estimated_steps_per_run,
estimated_total_steps, estimated_runtime_s, total_runs}`, `regimes {t_total, t_drift_fast, t_gauss,
t_lrs_drift, t_band_drift, t_neg_u, t_neg_r, steps_by_tier[6], newton_by_tier[6],
rejected_by_tier[6]}` (seconds summed over runs; the tier times are cell-averaged — each cell adds
h / n_cells — so they never exceed t_total; `steps_by_tier` counts a step under the lowest tier of its
cells; tier index as in §4, 0 = deterministic).

**Signals** (`runs[i].signals`, decimated to ≤ 4000 points with a time + arc-length sampler; cells of
the `coupled` bench carry the suffix 1 / 2):

| key | unit | axis | meaning |
|---|---|---|---|
| `v_src` (`v_clk` in pbit) | V | voltage | supply (pbit: the drain pulses) |
| `v_d` | V | voltage | device V_DS (= drain node voltage when the source is grounded) |
| `v_s`, `v_cmp` | V | voltage | pbit: source node V(R_S) and comparator output |
| `i_d` | A | current | drain current (log axis recommended) |
| `u`, `r` | V | state | internal unknowns |
| `q_b` | C | charge | Q_B(t) − Q_B(0) |
| `f_body` | A | current | net hole current into the body F (signed) |
| `dphi`, `dphi_E` | V (or 1 for ln-scale actions) | state | local-state deviations (stochastic, local states on) |
| `bit` | 1 | logic | comparator output, sample-and-hold (pbit) |

`trajectory` = run 0 {vd = v_d, id = i_d} of cell 1 for the I–V overlay.

**Summary keys**: load_line/coupled-ramp: `V_LU`, `V_LD`, `window`, `V_LU_src`, `V_LD_src`,
`n_latch_up` ("k/n" string, n = observed cycles), `fold_V_LU`, `fold_V_LD`, `lag_LU`, `lag_LD` (V; spread = SD over runs
and cycles), deterministic load_line also `hrs_branch_dev`, `lrs_branch_dev` (decades); coupled
adds suffix `_1`/`_2`, `corr_LU`, `dt_LU` (s). pulse: `P_sw` (spread = SD over runs of per-run
P_sw), `P_retained`, `delay` (s), `amplitude`, `fold_V_LU`; coupled-pulse `P_sw_1/2`, `delay_1/2`,
`P_both`, `corr_sw`. pbit: `P1` (comparator output), `P_latched`, `lag1`, `n_bits` (observed clocks),
`v_th` (= the comparator reference V_ref on V(R_S)), `R_S`, `v_high`, `fold_V_LU`. Always: `runs`, `steps_per_run`; stochastic with carrier noise:
`t_noise_resolved_frac` (cell-averaged, in [0, 1]).

**Distributions**: `V_LU`, `V_LD`, `V_LU_src`, `V_LD_src` (V; one value per run × cycle, `null` =
censored or no event), `delay` (s). **Sweeps**: `P_sw_vs_amplitude` (x V), `P1_vs_vg` (x V),
`P1_vs_iph` (x pA), y in [0, 1] with binomial `y_err` over the observed pulses/clocks. **Events**:
`latch_up`, `latch_down` (value = v_d at the event time, `v_src`, `v_d`, `cell`, and `cycle` or
`pulse`; sorted by time), `pulse` (value 1/0 = latched at the end of the flat top, first 8 runs,
reached pulses only), `bit` (comparator value 0/1 per clock, first 8 runs, reached clocks only).

## 11. 코드 구조 / Code map

| file | content |
|---|---|
| `element.py` | numba: `stl_eval` (components + charge coordinate + u<0 / r<0 extensions), cluster pmf interpolation |
| `mna.py` | numba: MNA assembly, Newton, DC point, sensitivities/τ_rel, event draws, the time-stepping loop `run_chunk` |
| `sim.py` | Python driver for one run (chunks, progress/cancel, buffers) |
| `netlist.py` | netlist builder (R, C, V, I, STL, CMP = comparator: behavioural voltage source), PWL waveforms, compilation to arrays, schematic |
| `benches.py` | bench defaults, builders, statistics helpers (no numba import) |
| `stochastic.py` | local states, branch profiles (τ_rel, rates, barrier z), noise bands, feasibility estimate |
| `runner.py` | `run_circuit`: parsing, feasibility, run loop, analysis, result (bench `custom` → `custom.py`) |
| `custom.py` | user-drawn circuits (§12): netlist/wave parsing, ERC, linear DC estimate, per-cell profiles, streaming signal reduction, statistics |
| `oscillator.py` | quasi-static load-line walk of high-impedance (current-driven) cells: relaxation-oscillator prediction and feasibility drive (§13) |
| `validate.py` | validation V1–V6 |

## 12. 사용자 회로 / Custom circuits (`bench: "custom"`, WEB_CONTRACT §6)

> **KO.** 회로 탭에서 사용자가 그린 회로(STL, 저항, 커패시터, 전압원·전류원)를 벤치와 **같은 커널**로
> 과도해석한다. 넷리스트(JSON)를 검증하고 ERC(접지 없음, 접지까지 DC 경로가 없는 노드, 전압원 루프,
> 개방 노드로 흐르는 전류원 등)를 통과해야 실행된다. 파형은 dc / pulse(SPICE PULSE 의미) / pwl / sine을
> 조각선형(PWL)으로 바꾸고 모든 꺾임점에 시간 스텝을 정확히 맞춘다. 시간 스텝·종료 시각은 `tran`에서
> 사용자가 정한다. 모든 노드 전압과 모든 소자 전류(STL은 단자별)를 매 스텝 기록하고, 모서리·래치 전이를
> 보존하며 4000점 이하로 줄여 보낸다. 확률 모드에서는 STL마다 소자 라이브러리의 국소 상태 설정을 쓸 수
> 있고, 실행별 파형(최대 8개), 공통 시간축의 평균·SD·p05·p95 포락선, 실행별 스칼라 분포(첫 래치업 시각,
> 그때의 V_DS, t_stop에서의 각 신호 값)와 확률 요약(종료 시 래치 확률, 1회 이상 래치업 확률)을 준다.
> 부하선 벤치를 사용자 회로로 그리면 벤치와 **비트 단위로 같은 결과**(결정론·확률 모두)를 얻는다.

### 12.1 Request

```jsonc
{ "bench": "custom", "mode": "deterministic" | "stochastic",
  "netlist": { "elements": [
    { "type": "R",   "name": "R1", "nodes": ["n1", "n2"], "value": 1000 },          // Ω, 1e-3 … 1e15
    { "type": "C",   "name": "C1", "nodes": ["d", "0"],   "value": 2e-15 },         // F, 0 … 1 (0 = open)
    { "type": "V",   "name": "V1", "nodes": ["n+", "n-"], "wave": Wave },           // V, |v| ≤ 1000
    { "type": "I",   "name": "I1", "nodes": ["n+", "n-"], "wave": Wave },           // A, |i| ≤ 1
    { "type": "STL", "name": "X1", "nodes": { "d": "d", "g": "g", "s": "0" },
      "device": { ...device block (§1)... },          // each STL may use a different device
      "light_pA": Wave | null,                         // I_PH(t) in pA (≥ 0); null = the device block's light
      "local_state": { ...§1 local_state... } },       // optional (extension): this cell's local states
    { "type": "CMP", "name": "CMP1", "nodes": { "in": "s", "out": "q" /* , "inm": "ref" */ },
      "v_ref": 0.1, "v_high": 1, "v_low": 0, "hysteresis": 0, "width": 1e-3 }   // comparator (§12.9)
  ] },
  "tran": { "t_stop_s": 5e-3, "t_start_save_s": 0, "dt_max_s": 1e-5, "dt_min_s": 1e-15,
            "method": "BE" | "TRAP", "reltol": 1e-4,   // dt_max default t_stop/2000, dt_min max(1e-15, 1e-13 t_stop)
            "initial": "auto" | "op" | "zero" },      // initial state (§13.2), default "auto"
  "stochastic": { "seed", "n_runs" (≤ 200), "carrier_noise", "ld_carrier_noise", "local_state",
                  "local_state_override": false },       // true: stochastic.local_state for every STL
  "solver": { "max_steps", "tau_frac", "max_events_per_step", "noise_dt_min_s", ... },  // optional, §6 of this doc
  "detect": { "i_threshold_A": 1e-8, "hysteresis": 10 },
  "probes": null | ["V(d)", "I(R1)", "I(X1.d)", "X1.u"] }  // null: every signal
```

* **Nodes**: any string of 1–32 characters without spaces/parentheses; `0`, `gnd`, `GND` (any case of
  "gnd") are ground. Element names: 1–32 characters, no spaces, dots, commas or brackets; unique
  (case-insensitive).
* **Waves** (all converted to piecewise-linear; every point is a solver breakpoint, so steps land exactly
  on the corners):

| kind | parameters | semantics |
|---|---|---|
| `dc` | `value` | constant |
| `pulse` | `v1, v2, td, tr, tf, pw, per, ncycles` | SPICE PULSE: v1 until td, rise tr to v2, flat pw, fall tf, repeat every per (`per` ≤ 0: one pulse; `ncycles` 0 = until t_stop). tr or tf = 0 → a finite edge min(dt_max, 1e-3 t_stop, 0.1 pw, 0.1 per) (warned); pw default t_stop; tr + pw + tf ≤ per |
| `pwl` | `t[], v[]` | ≤ 2000 points, t ≥ 0 non-decreasing; v[0] before t[0], v[-1] after the last point; a vertical step (repeated time) gets a finite edge ≤ that of `pulse` (warned) |
| `sine` | `vo, va, freq, td, theta` (+ `phase` in degrees, extension) | SPICE SIN: vo before td, then vo + va e^(−θ(t−td)) sin(2πf(t−td) + phase); sampled with 128 points per period (≥ 16; fewer than 48 → warning with the PWL error 1 − cos(π/N)) |

  Generated waves (pulse corners, sine samples) may have up to 20 000 points (≤ 5000 pulse periods,
  ≤ 1250 sine periods); user PWL lists up to 2000.
* **STL**: the device block is resolved exactly like the device tab (`payloads.normalize_device` +
  `params.build_p`). The gate is driven by the circuit: V_GS = v(g) − v(s) is re-evaluated at every Newton
  iteration; the block's `vg` is not used (a warning names both values when they differ). `light_pA` drives
  p[13] = I_PH(t). The quasi-static branches used for the latch state (fold values u_i/u_j) and the noise
  bands are computed at the V_GS and light the cell sees — from a linear DC estimate of the network with
  the STLs open (1 pS drain–source) and the capacitors open; when V_GS or the light varies in time, at both
  extremes (u_i = min, u_j = max over the variants with a latch window, bands = their union; warned).

### 12.2 Sign conventions and signals

| key | unit | axis | meaning |
|---|---|---|---|
| `V(n)` | V | voltage | node voltage to ground (`V(0)`/`V(gnd)` may be probed: zero) |
| `I(R1)`, `I(C1)` | A | current | through the element from its first to its second node; `I(C1)` is the companion current of the integration formula (BE: C Δv/h, TRAP: 2C Δv/h − i_n) |
| `I(V1)`, `I(I1)` | A | current | through the source from its + (first) to its − (second) node (SPICE: a source delivering power has I(V1) < 0); `I(I1)` = the wave value |
| `I(X1.d)`, `I(X1.s)`, `I(X1.g)` | A | current | into the STL terminals: I(X1.d) = I_D, I(X1.s) = −I_D, I(X1.g) = 0 (ideal gate: the gate–body displacement current is not stamped, §9) |
| `X1.u`, `X1.r` | V | state | internal unknowns |
| `X1.q_b` | C | charge | ΔQ_B = Q(t) − Q(0) |
| `X1.dphi`, `X1.dphi_E` | V or 1 | state | local-state deviations (stochastic, cell with local states) |
| `I(CMP1)` | A | current | comparator output current through its source, out → ground (SPICE sign: delivering power < 0) |
| `CMP1.bit` | 1 | logic | comparator digital output (1 = output at v_high) |

Kirchhoff's current law holds for the reported currents at every node to the Newton tolerance
(≤ 1e-5 of the largest current at the node; the tests check it) plus the numerical GMIN = 1e-18 S from
every node to ground (≈ 1e-17 A, not an element).
The latch **events** are those of §4 (body branch, timed at the I_D thresholds) with `cell` = element name,
`v_d` = V_DS of the cell at the event (= `value`), `i_d`; events before `t_start_save_s` are reported too.

### 12.3 ERC and limits (ValueError, message names the element / node)

* no element on node 0 → "no ground reference";
* every node needs a **DC path to ground** through resistors, voltage sources or an STL drain–source path
  (capacitors, current sources and STL gates do not conduct DC) → "node 'c' has no DC path to ground: it is
  connected only through capacitors C1, C2" (SPICE practice; covers capacitor-only islands and floating
  gates); a node reached only by a current source → "current source I1 drives node 'x', which has no other
  connection (open circuit)";
* voltage-source loops (incl. parallel sources): "voltage source V2 is in parallel with V1" / "… in a loop
  with V1, V3"; a source with both terminals on one node: "short-circuited";
* warnings: nodes with a single connection, R/C/I with both terminals on one node, STL terminals on the
  same node (d = s, d = g, g = s);
* limits: ≤ 40 elements, ≤ 8 STL cells, ≤ 30 nodes besides ground, PWL ≤ 2000 points, generated waves
  ≤ 20 000 points, `n_runs` ≤ 200, `max_steps` ≤ 2e6; unknown element types / wave kinds / probes and
  duplicate names are errors ("unknown probe 'V(zz)' (no node 'zz'; nodes: …)").

### 12.4 Integration and feasibility

The kernel is the bench kernel (§2–§4) with four additions, all off for the benches (bit-identical bench
results, checked): (1) the noise look-ahead drive is per cell — the source that moves the cell's V_DS most,
with V_DS(t') ≈ V_DS(t) + g (w(t') − w(t)), g = ∂V_DS/∂w from the linear DC estimate; (2) per-cell
local-state configurations; (3) capacitor currents are recorded; (4) a node-voltage LTE control (custom
circuits only; deterministic integration): LTE_i = |v_i − v_i,pred| h/(h + h_prev) ≤ 7 (reltol |v_i| + 1 µV)
(SPICE's trtol = 7) in addition to |Δv| ≤ 20 mV × reltol/1e-3, |Δu|, |Δ ln I_D| and LTE_u.
`tran` maps onto the solver (`method`, `dt_min_s`, `dt_max_s`, `reltol`); `solver.max_steps` and the
stochastic tier parameters may still be given in `solver`.

**Feasibility** before running: (a) the linear network's node voltages over the union of all source
breakpoints (with the STLs open, clipped to ±max(10 V, 1.5 max|V source|)) give the Δv/dt_max steps of
every node plus 60 steps per sharp corner (corner sharpness |Δslope|/(|s−| + |s+|), so a sampled sine
costs ≈ 1 step per point); (b) per cell, the bench estimator of §6 is walked along the cell's drive
V_DS(t) at every V_GS/light variant — the open-circuit (Thevenin) voltage for low-impedance drives, and for
**high-impedance cells** (d–s port resistance > 100 MΩ: current sources, large load resistors) with a
capacitance across the cell the quasi-static load-line walk of §13.3, which predicts relaxation
oscillations and counts their cycles (the drain/source nodes of such cells are left out of part (a)); the
cells' own extra steps (transitions, resolved noise) are added to the shared part. Refused above
2 × max_steps per run (the message names the predicted oscillation cycles when that is the cause) or 4e7
steps per request, warned above 0.5 × max_steps.

### 12.5 Output reduction and statistics

Every accepted step is recorded (node voltages, source currents, per cell u, r, Q, I_D, F, local states,
capacitor currents); a streaming reducer converts each 4000-step block to the output signals, applies
`t_start_save_s` (a row interpolated at t_start_save) and keeps ≤ 30 000–60 000 rows in memory. The stored
waveforms are selected by 60 % arc length over the normalised signals (+ 0.3 × log10|I_D| of the STL
terminals, deterministic mode), 40 % uniform in time, plus every source corner and the rows around each
latch event: run 0 ≤ 4000 points (≤ 400 000 values over all signals), runs 1–7 ≤ 1500 (≤ 100 000 values).
Plotted values are rounded to 7 significant digits (time axes 10, envelopes 6); summary, events and `op`
keep full precision.

Stochastic (`n_runs` ≤ 200, per-run seeds as §4): the first ≤ 8 runs' waveforms; `envelopes` (mean, SD
(ddof 1), p05, p95 over the runs that reached each time) of every probed signal on a common grid of
≤ 1000 points (half uniform, half the arc-length points of run 0); `distributions` per run:
`X1.t_first_lu`, `X1.vd_first_lu`, `X1.n_latch_up` and `end:<signal>` (value at t_stop; truncated runs
are null); summary per cell `X1.p_any_lu` (P(≥ 1 latch-up)), `X1.p_latched_end` (P(latched at t_stop)),
`X1.t_first_lu`, `X1.vd_first_lu`, `X1.n_latch_up`, `X1.n_latch_down` (mean, spread = SD over runs).
Deterministic summary per cell: `X1.n_latch_up`, `X1.n_latch_down`, `X1.t_first_lu`, `X1.vd_first_lu`,
`X1.latched_end` (0/1), `X1.final_state` ("LRS" | "HRS"); with ≥ 2 latch-ups in a run (oscillator, pulse
trains) also `X1.period`, `X1.f_osc`, `X1.isi_cv`, `X1.vd_lu_mean`, `X1.vd_ld_mean` (+ stochastic
`X1.period_cv_runs` and the distribution `X1.isi`; §13.4); always `X1.fold_V_LU`, `X1.fold_V_LD`
(quasi-static at the cell's V_GS), `runs`, `steps_per_run` (+ `t_noise_resolved_frac` with carrier noise,
`truncated_runs`). Further result keys: `nodes`, `elements` (resolved echo: values, resolved waves, per STL
V_GS and its range, folds, latch window, u_i/u_j, noise band, local state, estimated steps), `op` (flat
{signal key: value} at t = 0, every signal), `probes`, `trajectory` (first STL: V_DS, I_D of run 0),
`tran` (+ `initial`, `initial_used`), `solver`, `detect`, `stochastic`, `feasibility`, `regimes`, `schematic`
(minimal); per STL in `elements` also `oscillator` (null for low-impedance drives, §13.3).

### 12.6 Validation (server/tests/test_circuit_custom.py, 66 tests, ~40 s warm; + 1 slow)

| check | result |
|---|---|
| RC charging (1 V step, τ = 1 ms, 6τ) vs 1 − e^(−t/τ) | BE max error 0.055 % (reltol 1e-3), 0.047 % (1e-4), 0.031 % (1e-5); TRAP 1e-4 % |
| current source into R, V source power sign | V(n) = ±1 V, I(I1) = wave, I(V1) = −2 mA for a 2 V source on 1 kΩ |
| PULSE (td, tr ≠ tf, ncycles 3), SIN with θ | pulse exact at the recorded points (< 1e-6 V), every corner present in the output; sine within the PWL error (< 1e-3) |
| KCL at every node (STL with source resistor, 2 capacitors, current source) | ≤ 1e-5 relative (+ GMIN floor) |
| load-line template (V_src → 1 kΩ → d, 2 fF, V_G −2 V, 1200 / 40 / 0.4 V/s) vs `load_line` bench | V_LU, V_LD **identical** (Δ = 0), same step count (2659 / 2773 / 2928) |
| stochastic load line (photo condition, 1200 V/s, same seeds) vs bench | identical events of every run |
| two STLs with different devices (−1.8 V + 2.63 pA, −2 V dark) | each latches at its own fold + ramp lag (3.337 / 3.748 V) |
| n_runs = 5, seed | envelopes/distributions present, p05 ≤ p95, reproducible with the seed, different with another seed |
| per-cell frozen local states (carrier noise off) | V_LU spread over runs, distinct δ per run |
| ERC / limits / invalid requests | 28 error messages checked; warnings (single connection, shorted element); ground aliases |
| feasibility | refused with the cause (dt_max too small for t_stop; event-level noise on a 0.4 V/s ramp with a small max_steps; six 1000-period sines; ~8600 predicted oscillation cycles); cancellation between chunks |
| comparator (§12.9) | switching at v_ref ± hysteresis/2 within 4 µs on a 1 V/ms ramp, output levels exact, SPICE sign of I(CMP1), KCL; differential input and inverted levels; 6 ERC errors |
| source-degenerated STL, p-bit (§14) | 100 kΩ source resistor: latches without convergence failure, V(s) = R_S I_D, KCL; drain pulses 3.69 / 3.72 V fire never / always (deterministic); stochastic P(fire) in (0.15, 0.85) with the latch-ups = fired pulses, seeded |
| current-driven oscillator (§13) | inside the window: ten alternating latch-up/latch-down events, period within 1 % of the converged value, peak/valley within the fold lags, KCL, estimate within 0.6–2×; outside (5 pA, 30 nA): no oscillation, settles on the HRS / LRS; initial states op / zero / auto; stochastic jitter > 0, reproducible with the seed; light raises the frequency > 1.4×; (slow) period vs BE/TRAP at reltol 1e-5 within 0.3 % |
| API (TestClient, POST /api/compute/circuit) | result with the §6 keys; ERC error → job status "error" with the message |

### 12.7 Performance (4-CPU shared container, warm numba cache)

| case | steps / run | time |
|---|---|---|
| RC, 1 V step, 6τ | 2 100–3 800 | 7–12 ms |
| RC driven by a 100-period sine (12 800 PWL points), BE reltol 1e-4 | 2.9e5 | 1.0 s |
| load-line template, 1 STL, 1200 V/s | 2 600 | 0.2–0.35 s |
| two STLs on one ramp | 3 080 | 0.7 s |
| eight STLs (4 V_G, 8 series R) on one ramp | 3 155 | 1.9 s (≈ 530 µs/step) |
| stochastic load line, photo condition, 1200 V/s, 20 runs | 3 400 | 5.9 s (20 runs) |
| stochastic pulse train through 10 kΩ / 5 fF, 5 pulses, 10 runs | 5 800 | 4.1 s (10 runs) |
| current-biased STL + 10 fF (relaxation oscillator, ~450 cycles in 5 ms) | 2.2e5 | 18 s |
| oscillator template (1 nA, 1 pF, 15 ms: 10 cycles), deterministic | 1.05e4 | 0.6–1.1 s |
| same, stochastic (carrier noise + evolving local states), per run | 1.1e4 | 0.9 s |

Cost per step ≈ 3 µs without STL, 60–90 µs per STL cell (Newton with finite-difference partials +
τ_rel sensitivities). The custom path uses the same compiled kernels as the benches (one signature per
kernel function for every circuit size and mode — no recompilation per request); a changed kernel source
recompiles once (~40 s, cached in `server/compute/circuit/__pycache__`); `scripts/warmup.py` runs a small
RC + STL netlist deterministically and stochastically. Post-processing (conversion, reduction, envelopes)
is ≲ 2 % of the run time. Result size: 1 STL deterministic (13 signals) 0.45 MB JSON (0.13 MB gzip);
20 stochastic runs 2.4 MB (0.57 MB gzip); 8 STLs, 58 signals 2.8 MB.

### 12.8 Limitations

* Linear elements are ideal R and C and independent sources (no inductors, controlled sources or
  switches); the STL gate is ideal (no gate current; gate–body coupling acts through V_GS in the charge
  coordinate only).
* TRAP can ring on stiff RC nodes (h ≫ RC) like any trapezoidal integrator; use BE there (the STL charge
  equation already falls back to BE on stiff steps).
* The noise look-ahead uses one drive source per cell and the open-cell gain; cells whose V_DS is set by
  several sources that move simultaneously get the look-ahead of the dominant one only.

### 12.9 Comparator (`CMP`)

> **KO.** 비교기는 이상적인 입력(전류 0)과, 출력 노드에서 접지로 가는 행동 모델 전압원을 갖는다.
> 출력 = V(in) − V(inm) > V_ref(± 히스테리시스/2)이면 v_high, 아니면 v_low이며, 뉴턴 수렴을 위해 폭 w = 1 mV의
> tanh로 부드럽게 바뀐다. 출력은 다른 소자를 구동할 수 있다. 펄스 전원이 있으면 그 주기마다 출력이 high였는지
> (발화)를 집계해 run × 펄스 래스터, 펄스별 발화 확률, 전체 P(발화), 비트열 lag-1을 준다.

* **Element**: nodes `{in, out}` (+ optional `inm`, default ground; also `[in, out]` / `[in, inm, out]`),
  `v_ref` (V, required), `v_high` (1), `v_low` (0), `hysteresis` (0; turn-on at v_ref + h/2, turn-off at
  v_ref − h/2), `width` (1 mV, 1 µV … 0.1 V). Output y = v_low + (v_high − v_low)(1 + tanh((d − thr)/w))/2,
  d = V(in) − V(inm): stamped as a voltage source (branch current `I(CMP1)`) whose Jacobian row carries
  −∂y/∂d at the inputs; the hysteresis state is updated after every accepted step. The output node is
  algebraic and excluded from the node-voltage step control (a switching comparator does not shrink the step).
* **ERC**: the output is a voltage source to ground: tied to another voltage source or comparator output →
  "the output of comparator CMP1 … must not be driven by another source"; output on ground → error; an input
  node needs its own DC path (inputs are ideal); input = output → warning. Limit 8 comparators.
* **Firing statistics** (`comparators` result key, one entry per comparator): windows = the periods of the
  periodic `pulse` source (V or I) with the most complete periods (a period counts when its flat top ends
  before t_stop; ≥ 2 periods); fired = the output was high at any recorded step within the period (tracked at
  full resolution in the streaming sink). Entry: `name, nodes, v_ref, v_high, v_low, hysteresis, width,
  window_source, t_windows, bits` (runs × windows, 0/1/null = not reached; capped at 1e5 cells),
  `p_fire_window, p_fire_window_err` (binomial SE over runs), `p_fire`, `lag1` (pooled over runs, consecutive
  pulses), `n_bits`, `p_fire_run`. Summary: `CMP1.p_fire` (spread = SD of the per-run fractions), `CMP1.lag1`,
  `CMP1.n_bits`, and always `CMP1.n_rise` (rising edges per run), `CMP1.duty` (fraction of time high);
  stochastic distribution `CMP1.p_fire_run`. Events `cmp_rise` / `cmp_fall` (`cell` = comparator name,
  mid-level crossing time interpolated linearly).
* The linear DC estimate (V_GS ranges, drives, feasibility) treats a comparator output as fixed at 0 V.
* Self-oscillation is foreseen by the quasi-static walk only for high-impedance cells (> 100 MΩ port
  resistance) with a capacitance on the drain; other self-oscillating configurations are still bounded by
  `solver.max_steps` (warning, truncated run, censored statistics). The oscillator step estimate is within
  0.8–1.9× of the actual count (conservative at high ramp rates, §13.5).

## 13. 전류 구동 발진기 / Current-driven oscillator (integrate-and-fire)

> **KO.** 직류 전류원 I_in이 노드 V_out(STL 드레인)을 충전하고, 기생 커패시터 C_par가 V_out과 접지 사이에,
> STL이 V_out과 접지(소스) 사이에 있으며 게이트는 직류 전압원(V_G = −2 V)에 묶인 회로다. I_in이 HRS fold
> 전류 I_LU(V_G −2 V에서 약 15 pA)와 LRS fold 전류 I_LD(약 16 nA) 사이에 있으면 전류원의 부하선은 음저항
> branch만 지나므로 이완 발진(적분-발화)이 일어난다: V_out은 (I_in − I_HRS)/C_par의 기울기로 V_LU까지 오르고,
> 바디가 래치되면 LRS 전류(µA)가 C_par를 V_LD까지 빠르게 방전시키고, 바디가 풀리면 다시 충전한다. 주기는
> T ≈ C_par(V_LU − V_LD)/I_in에 fold 통과 지연을 더한 값이다. 창 밖(I_in < I_LU: HRS에 정착, I_in > I_LD: LRS에
> 정착)에서는 발진하지 않는다. 확률 모드에서는 운반자 잡음 때문에 래치업 전압이 흔들려 스파이크 간격에 지터가
> 생기고(CV 약 1–2 %), 느린 램프에서는 fold 이전 탈출로 주기가 2–6 % 짧아진다. 빛(광조사 보정 소자)은 V_LU를 낮춰
> 발진 주파수를 올린다(2.63 pA에서 1.7배). 이 회로는 회로도 편집기의 예제 "전류 구동 발진기 (integrate-and-fire)"
> 로 제공된다(1 nA, 1 pF, 15 ms, 스파이크 10개).

**Summary (EN).** A DC current source I_in charges V_out (the STL drain), C_par sits between V_out and ground,
the STL between V_out and ground (source), the gate at a DC source. For I_LU < I_in < I_LD the load line of the
current source crosses only the negative-resistance branch and the cell relaxation-oscillates (integrate and
fire); outside that window it settles on the HRS or the LRS. Verified quantitatively below (period, sawtooth
extremes, events, KCL, convergence, jitter, light), with the schematic example "Current-driven oscillator
(integrate-and-fire)".

### 13.1 Physics and period

Quasi-static cycle (body on its steady-state branches, hysteretic switch at the folds; I_HRS/I_LRS the branch
currents of `MODEL.classify`, a Norton load I_N − G V in general):

  T_qs = ∫_{V_LD}^{V_LU} C_par dV / (I_in − I_HRS(V))  +  ∫_{V_LD}^{V_LU} C_par dV / (I_LRS(V) − I_in)
       ≈ C_par (V_LU − V_LD) / I_in          (I_LU ≪ I_in ≪ I_LD; the discharge term is ≲ 2 µs per pF),

V_LU − V_LD = 1.1058 V (reference device, V_G = −2 V; I_LU = 14.9 pA, I_LD = 16.3 nA at the 601-point grid).
The simulated period adds the slow passage through both folds (§7 V1: the lag grows with the ramp rate
I_in/C_par): with the measured peak V_pk and valley V_vl,

  T ≈ T_qs + C_par [(V_pk − V_LU) + (V_LD − V_vl)] / I_in,

e.g. 1 nA / 1 pF: T_qs = 1.1093 ms, lags +38.5 / −12.2 mV → 1.1600 ms vs 1.1628 ms simulated. The first
latch-up after switching on comes at ≈ C_par V_LU / I_in (charging from 0 V).

Stability of the equilibrium on the NDR branch (two state variables V and Q_B): with g_V = ∂I_D/∂V at fixed
body charge and F_Q = ∂F/∂Q > 0 (the body is unstable at fixed V on that branch) the equilibrium is unstable —
and the circuit oscillates — when C_par > g_V/F_Q (Hopf condition, trace of the Jacobian). For pF capacitors this
always holds; at C_par ≈ 1–30 fF (comparable to the body's own charges, whose terminal displacement currents are
not stamped, §9) the circuit can sit on a stable NDR equilibrium or oscillate around it without unlatching
(measured at 1 nA: 1 and 3 fF one latch-up then a small-amplitude oscillation of u between 0.62 and 0.84 V;
10 fF regular cycles). Treat the fF regime as qualitative.

### 13.2 Initial state (`tran.initial`)

The DC operating point of a current-biased cell with I_LU < I_in < I_LD is the equilibrium on the NDR branch
(I_D = I_in, u_i < u < u_j; e.g. V_out = 3.342 V at 100 pA). It is a legitimate DC solution (SPICE finds it too),
but an unstable one: the deterministic transient stayed there until round-off grew (first latch-up at 63 ms
instead of ≈ 37 ms at 100 pA / 1 pF, 821 ms instead of 370 ms at 100 pA / 10 pF), so the sawtooth started at an
arbitrary time; for I_in ≳ 5 nA (beyond what the HRS
reached from the empty body can carry) the operating point did not converge at all (error). Now:

| `tran.initial` | t = 0 state |
|---|---|
| `"op"` | the DC operating point (§2; capacitors open, bodies from the empty state) — the benches always use it |
| `"zero"` | discharged capacitors (SPICE UIC with IC = 0): every capacitor held at 0 V by 1 MS while the bodies relax (capacitors whose terminals are fixed by voltage sources alone start at that voltage); the recorded t = 0 capacitor current is the current the circuit pushes into it (KCL) |
| `"auto"` (default) | `"op"`, except (i) a high-impedance cell (§13.3) whose load line at t = 0 misses the HRS (I_N − G V_LU > I_LU), (ii) an operating point that puts a cell on the NDR branch, (iii) no operating point found: then `"zero"`, as if the sources were switched on at t = 0 (warning) |

The result echoes `tran.initial` and `tran.initial_used`; `op` is the t = 0 state actually used.

### 13.3 Quasi-static walk and feasibility (`oscillator.py`)

For each STL whose d–s port resistance in the linear network (STLs open) exceeds 100 MΩ and with a capacitance
C_eff on the drain (capacitors from the drain node or nodes tied to it by < 1 MΩ; far ends taken as AC ground),
the Norton equivalent (I_N(t) = V_oc(t)/R_th, G = 1/R_th − G_OFF) drives the hysteretic walk
C_eff dV/dt = I_N(t) − G V − I_b(V), b = HRS until V_LU, then LRS until V_LD (semi-implicit Euler, 5 mV per
step, ≤ 3e5 points, extrapolated beyond). Its V_DS(t) replaces the open-circuit drive in the §6 estimator
(400 steps per latch transition for these cells, calibrated below), it predicts oscillation (≥ 2 latch-ups) and
it yields a warning that explains the regime ("relaxation oscillator — fed by 1 nA (current source) with 1 pF
…, quasi-static period ≈ 1.109 ms", or "crosses the HRS … settles near V_DS ≈ 3.64 V", or "crosses the LRS …
stays latched"). Echo per STL: `oscillator {predicted, period_qs_s, latch_ups_expected, c_eff_F, i_norton_A
[min, max], r_ext_ohm}`.

### 13.4 Result keys (additive)

Per STL with ≥ 2 latch-ups in a run (any circuit, e.g. also pulse trains): `X1.period` (s; mean interval
between consecutive latch-ups, pooled over runs; spread = SD of the intervals), `X1.f_osc` (Hz, 1/period),
`X1.isi_cv` (CV of the intervals within a run, averaged over the runs: the spike-timing jitter of one
oscillator; ≈ 1e-5 deterministic), `X1.vd_lu_mean` / `X1.vd_ld_mean` (V_DS at all latch-ups / latch-downs,
mean ± SD: the sawtooth extremes), stochastic with ≥ 2 runs `X1.period_cv_runs` (CV of the per-run mean periods:
run-to-run spread, e.g. frozen or slowly evolving local states) and the distribution `X1.isi` (all intervals).

### 13.5 Measurements (reference device, V_G = −2 V; DC current source into V_out; t_stop = C V_LU/I + 10 T_qs inside the window, 3 C V_LU/I outside; default BE, reltol 1e-3, dt_max = t_stop/2000)

**Before** (previous code, same DC source): the run started at the NDR equilibrium (V_out 3.661 / 3.342 /
2.759 V at 30 pA / 100 pA / 1 nA); first latch-up at 132 ms instead of 124 ms (30 pA, 1 pF), 63 vs 37 ms
(100 pA, 1 pF), 821 vs 370 ms (100 pA, 10 pF), so only 6–8 of the expected 10 cycles; 5 nA and 30 nA: "DC
operating point did not converge" (every C_par); the step estimate was 2 256 for every case (0.18–0.30 × the
actual 7 400–12 300 steps; a 10 fF, 5 ms case: 2.6e3 estimated, 2.2e5 actual). Once running, the sawtooth itself
(period, extremes, events) was the same as now.

**After** (deterministic; stochastic column: carrier noise only, 4 runs, seed 11):

| C_par | I_in | regime | osc. | latch-ups up/down | 1st latch-up | T (sim) | T_qs | T/T_qs | V_pk − V_LU | V_vl − V_LD | steps | est/actual | time | KCL | stochastic: ISI CV / T_sto/T_det / steps per run / s per run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 pF | 5 pA | < I_LU | no | 0/0 | – | – | – | – | – | – | 2 009 | 1.00 | 0.3 s | 0 | settles at 3.6365 V (HRS) |
| 0.1 pF | 30 pA | inside | yes | 10/10 | 12.59 ms | 4.1349 ms | 3.8556 ms | 1.072 | +9.9 mV | −60.2 mV | 10 102 | 1.05 | 0.54 s | 3e-6 | 1.59 % / 0.979 / 1.6e4 / 0.86 s |
| 0.1 pF | 100 pA | inside | yes | 10/10 | 3.757 ms | 1.2213 ms | 1.1193 ms | 1.091 | +33.2 mV | −58.8 mV | 9 678 | 1.09 | 0.58 s | 3e-6 | 1.45 % / 0.998 / 1.0e4 / 0.72 s |
| 0.1 pF | 1 nA | inside | yes | 9/9 | 0.3959 ms | 134.53 µs | 110.93 µs | 1.213 | +169.7 mV | −51.5 mV | 8 213 | 1.29 | 0.55 s | 3e-6 | 1.52 % / 1.006 / 6.5e3 / 0.56 s |
| 0.1 pF | 5 nA | inside | yes | 7/7 | 88.2 µs | 31.49 µs | 22.36 µs | 1.408 | +386.6 mV | −37.0 mV | 6 308 | 1.67 | 0.42 s | 3e-6 | 1.69 % / 1.004 / 5.5e3 / 0.53 s |
| 0.1 pF | 30 nA | > I_LD | no | 1/0 | 16.9 µs | – | – | – | – | – | 2 575 | 0.95 | 0.08 s | 4e-6 | stays on the LRS at 2.6037 V |
| 1 pF | 5 pA | < I_LU | no | 0/0 | – | – | – | – | – | – | 2 009 | 1.00 | 0.04 s | 0 | 3.6365 V (HRS) |
| 1 pF | 30 pA | inside | yes | 10/10 | 125.4 ms | 39.228 ms | 38.556 ms | 1.017 | +2.3 mV | −13.7 mV | 10 884 | 0.97 | 0.62 s | 3e-6 | 1.20 % / 0.960 / 6.9e4 / 1.9 s |
| 1 pF | 100 pA | inside | yes | 10/10 | 37.27 ms | 11.428 ms | 11.193 ms | 1.021 | +7.8 mV | −13.5 mV | 10 486 | 1.01 | 0.61 s | 3e-6 | 1.12 % / 0.980 / 2.7e4 / 1.2 s |
| 1 pF | 1 nA | inside | yes | 10/10 | 3.744 ms | 1.1628 ms | 1.1093 ms | 1.048 | +38.5 mV | −12.2 mV | 9 812 | 1.08 | 0.60 s | 4e-6 | 1.62 % / 0.995 / 1.0e4 / 0.79 s |
| 1 pF | 5 nA | inside | yes | 9/9 | 0.7657 ms | 248.58 µs | 223.62 µs | 1.112 | +108.9 mV | −8.7 mV | 8 488 | 1.24 | 0.60 s | 3e-6 | 1.33 % / 1.003 / 7.3e3 / 0.67 s |
| 1 pF | 30 nA | > I_LD | no | 1/0 | 0.142 ms | – | – | – | – | – | 2 566 | 0.96 | 0.08 s | 4e-6 | LRS at 2.6037 V |
| 10 pF | 5 pA | < I_LU | no | 0/0 | – | – | – | – | – | – | 2 009 | 1.00 | 0.04 s | 0 | 3.6365 V (HRS) |
| 10 pF | 30 pA | inside | yes | 10/10 | 1.252 s | 387.43 ms | 385.56 ms | 1.005 | +0.5 mV | −3.1 mV | 11 855 | 0.89 | 0.82 s | 3e-6 | 1.14 % / 0.942 / 4.6e5 / 8.4 s |
| 10 pF | 100 pA | inside | yes | 10/10 | 371.9 ms | 112.48 ms | 111.93 ms | 1.005 | +1.8 mV | −3.1 mV | 11 600 | 0.91 | 0.76 s | 4e-6 | 1.11 % / 0.962 / 1.4e5 / 3.2 s |
| 10 pF | 1 nA | inside | yes | 10/10 | 37.14 ms | 11.217 ms | 11.093 ms | 1.011 | +8.9 mV | −2.9 mV | 10 897 | 0.97 | 0.65 s | 4e-6 | 1.29 % / 0.981 / 2.7e4 / 1.4 s |
| 10 pF | 5 nA | inside | yes | 10/10 | 7.459 ms | 2.2942 ms | 2.2362 ms | 1.026 | +25.2 mV | −2.1 mV | 10 370 | 1.02 | 0.64 s | 3e-6 | 1.45 % / 0.993 / 1.3e4 / 0.96 s |
| 10 pF | 30 nA | > I_LD | no | 1/0 | 1.261 ms | – | – | – | – | – | 2 646 | 0.93 | 0.11 s | 4e-6 | LRS at 2.6037 V |

KCL = max relative residual at V_out over the run (Newton tolerance, §12.2). Stochastic est/actual 0.82–1.90.
Observations: (1) the window is sharp — no oscillation at 5 pA or 30 nA, a regular sawtooth at every current in
between; (2) T/T_qs − 1 is the fold lag, which grows with the ramp rate I_in/C_par (0.5 % at 3 V/s, 41 % at
5e4 V/s where the body cannot latch before V_out overshoots V_LU by 0.39 V); the formula of §13.1 with the
measured extremes reproduces T within 0.3 %; (3) with carrier noise the latch-up happens at a random V_DS below
the deterministic value (noise-induced escape inside the noise band): ISI CV 1.1–1.7 %, and on slower ramps
(≤ 100 V/s) the period is 2–6 % shorter; (4) runs of 10 cycles take 0.4–0.8 s deterministic, 0.5–2 s per
stochastic run (8 s per run at 3 V/s, where event-level noise is resolved for tens of ms per cycle).

**Convergence** (1 nA / 1 pF unless noted; period, peak V_out): default BE 1.16276 ms / 3.74220 V; BE reltol 1e-4
1.16353 / 3.74267; BE 1e-5 1.16363 / 3.74276; TRAP 1e-3 1.16383 / 3.74286; TRAP 1e-5 1.16381 / 3.74287 → the
default period is within 0.09 % and its peak 0.6–0.7 mV below the converged value (no numerical overshoot).
100 pA / 1 pF: −0.03 %; 30 pA / 10 pF: +0.07 %; 5 nA / 0.1 pF (fastest edges): −0.24…−0.28 %, peak −3.4 mV. Events: every
latch-up/latch-down is reported (strictly alternating), its time equals the interpolated I_D threshold crossing
of the recorded steps to < 1e-17 s, and V_DS at the latch-up is within 0.22 mV of the sawtooth peak (the peak
comes 4 ns – 7 µs earlier, when I_D passes I_in).

**Where the steps go** (100 pA / 1 pF, ≈ 1 000 steps per cycle, 60–90 µs per step): HRS ramp ≈ 380 (dt_max and the
LTE_u = 30 µV control near the fold), latch-up passage ≈ 195, LRS discharge and latch-down ≈ 420 — LTE_u is the
binding limit in ≈ 90 % of the steps. Loosening it would cost accuracy at the folds (it is what makes the lag
converge, §3) and the runs already take < 1 s per 10 cycles, so the step control is unchanged.

**Stochastic template run** (1 nA / 1 pF, 15 ms, the reference device's evolving GIDL local states, 6 runs):
within-run ISI CV 1.1–2.1 %, per-run mean periods 1.07–1.24 ms (the frozen-over-15-ms local state shifts V_LU
by up to ±0.1 V from run to run: `X1.period_cv_runs` ≈ 4–5 %), 1.2e4 steps and ≈ 1 s per run. Carrier noise
only: ISI CV 1–2.5 %, periods 1.152–1.162 ms.

**Light** (illumination calibration device, V_G = −1.8 V, 100 pA / 1 pF): I_PH = 0 / 1 / 2.63 / 5 pA → V_LU
3.804 / 3.633 / 3.279 / 2.934 V, oscillation 80.5 / 91.0 / 123 / 148 Hz (T/T_qs 1.02–1.06); at 10 pA
I_LU = 0.40 nA exceeds I_in and the cell settles on the HRS (no firing). A light step 0 → 2.63 pA at 8 ms (1 nA / 1 pF) shortens the
interval from 1.266 ms to 0.748 ms within one cycle (light-to-frequency conversion).

**Template** (schematic example, `web/src/schematic/templates.ts`): I_in = 1 nA DC from ground into `out`,
C_par = 1 pF, STL (library device) drain `out`, source ground, gate at V_G1 = device V_G (−2 V), t_stop 15 ms,
dt_max 5 µs → first latch-up 3.74 ms, ten latch-ups, period 1.163 ms (860 Hz), ≈ 1.05e4 steps, ≈ 1 s;
default traces V(out) and I(X1.d) (the only labelled net is `out`).

### 13.6 Limitations

* The quasi-static walk (estimate, warnings) ignores the fold lags and the body dynamics: its period is 1–40 %
  short at high ramp rates, its step estimate 0.8–1.9× the actual count; it treats the far end of every drain
  capacitor as AC ground and uses the nominal V_GS / light variant for the warning.
* Only cells with > 100 MΩ port resistance and a drain capacitance use the walk; oscillators built otherwise are
  estimated along the open-circuit drive (bounded by `solver.max_steps` as before).
* fF-scale C_par: see §13.1 (Hopf regime, displacement currents not stamped).
* `tran.initial = "zero"` holds capacitors with a 1 MS conductance (≈ 1 µV error per A pushed into the node at
  t = 0); the schematic editor does not expose `tran.initial` (the server default "auto" applies).

## 14. p-비트: 드레인 펄스 · 소스 저항 · 비교기 / p-bit: drain pulses, source resistor, comparator

> **KO.** 드레인에 일정 간격의 전압 펄스를 걸고, 소스 → R_S → 접지, 소스 저항 양단 전압 V(R_S) = R_S·I_D를
> 사용자가 정한 V_ref의 비교기에 넣는다. 래치되면 LRS 전류(µA)로 V(R_S) ≈ 0.44 V(100 kΩ), 래치되지 않으면 µV이므로
> V_ref = 0.1 V가 두 상태를 가른다. 펄스 높이를 latch-up fold 바로 아래(V_LU − 15 mV)로 두면 운반자 잡음에 따라
> 펄스마다 무작위로 래치되어 비교기가 무작위로 발화한다(P ≈ 0.5, 연속 비트 사이 상관 거의 없음). 결정론 모드에서는
> 같은 펄스가 전혀 발화하지 않고, 3.72 V 이상에서는 매 펄스 발화한다. 빠른 벤치 `pbit`와 회로도 예제 "p-비트"가
> 이 구성을 쓴다.

**Summary (EN).** Regular voltage pulses on the drain, source → R_S → ground, and a comparator with a
user-set V_ref on the source-resistor voltage V(R_S) = R_S I_D. A latched pulse carries the LRS current and
lifts V(R_S) to ≈ 0.44 V (R_S = 100 kΩ), an unlatched one leaves micro-volts, so V_ref = 0.1 V separates them.
With the pulse height just below the latch-up fold the carrier noise decides in each pulse whether the cell
latches: the comparator fires at random.

**Circuit physics.** The latched operating point is the LRS point on the source-degenerated load line
I = (V_pulse − V_DS)/R_S: at 3.75 V pulses V_S = 0.437 V, V_DS = 3.31 V, I_D = 4.37 µA (V_GS = V_G − V_S =
−2.44 V while latched); the latched point must keep V_DS above V_LD (≈ 2.6 V) on that load line (source
degeneration limits the LRS current). The gate is referenced to ground, so the source rise also lowers V_GS; both enter the
element through the new ∂/∂V_GS Jacobian columns (§2). When the pulse falls (20 µs) the drain passes below
the source briefly (r < 0: forward drain-junction extension of §1.1, warned).

**Measured** (reference device, V_G = −2 V, 200 µs flat top, 20 µs edges, 1 ms period, R_S = 100 kΩ,
V_ref = 0.1 V; carrier noise only; 4 runs × 20 pulses unless noted):

| pulse high | 3.60 V | 3.64 V | 3.66 V | 3.68 V | 3.69 V | 3.70 V | 3.72 V | 3.75 V |
|---|---|---|---|---|---|---|---|---|
| P(fire), stochastic | 0 | 0 | 0.075 | 0.34 | 0.495 (10 runs; bench auto 3.6887 V: 0.505, 10 × 20) | 0.75 | – | – |
| lag-1 of the bits | – | – | −0.08 | −0.07 | +0.07 (bench +0.03) | 0.00 | – | – |
| deterministic | never | never | never | never | never | never | every pulse | every pulse |
| steps per run / time | 1.5e4 / 1.2 s | 1.5e4 | 1.6e4 | 1.8e4 | 1.9e4 / 1.5 s | 2.0e4 / 1.7 s | 7.5e3 (5 pulses) | 7.5e3 |

The bit stream is practically uncorrelated at 1 ms period (|lag-1| ≤ 0.08 within the sampling error of ±0.1
for 80–200 bits). Sweeps of the bench (6 runs × 20 pulses at the auto amplitude): P1 vs V_G = −2.02 / −2.00 /
−1.98 V → 0.90 / 0.53 / 0.17 (the fold moves with V_G), P1 vs I_PH = 0 / 0.1 / 0.3 pA → 0.53 / 0.78 / 1.0.
Live UI check (schematic example, 8 runs): P(fire) = 0.487, lag-1 −0.03, 1.4 s per run.

**Schematic example** "p-bit (drain pulses, source resistor, comparator)": V_pulse 0 → 3.69 V, 20 pulses (tr = tf
= 20 µs, pw = 200 µs, per = 1 ms) on the drain node `d`; STL (library device, gate at V_G1 = −2 V); source
node `s` → R_S = 100 kΩ → ground; comparator CMP1 in = `s`, V_ref = 0.1 V, out `q`; t_stop = 20 ms;
stochastic settings: carrier noise with the local states overridden to "none" (the library's slowly evolving
GIDL states would shift V_LU by up to ±0.1 V from run to run and make each run fire almost always or never).
Results: the comparator panel shows the run × pulse raster and P(fire) per pulse; the default traces include
`CMP1.bit` as a logic trace.

**Bench `pbit`** uses the same topology (the comparator is a CMP element in the netlist; the bit is sampled at the
end of each flat top from V(R_S) = R_S I_D); the previous load-resistor variant (clock → R_L → drain,
comparator on v_D) was replaced — old `R_L_ohm`, `C_d_F` and `cmp_threshold_V` bench parameters are ignored with
a warning.
