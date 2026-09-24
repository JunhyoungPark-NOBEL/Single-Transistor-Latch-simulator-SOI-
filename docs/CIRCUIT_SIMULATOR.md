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
> 진폭 스윕), pbit(부하 저항 + 클럭 + 비교기, P(1)·자기상관, V_G/광 스윕), coupled(저항 결합 두 셀).
> 바디 전하는 사이클 사이에 연속 적분되므로 잔류 바디 기억 효과가 자동으로 포함된다.

`None` = automatic. All voltages V, times s, R Ω, C F.

| bench | circuit | defaults |
|---|---|---|
| `load_line` | V_src triangle v_min → v_max → v_min, series R_s, C_d to ground, DC gate | v_min 0, v_max = preset vd_max (paper 4 V, photo 5 V), rate = preset (0.4 / 1200 V/s; stochastic falls back to 1200 V/s when infeasible), n_cycles 1 (≤ 50), R_s 1 kΩ, C_d 2 fF, vg_V = device.vg |
| `pulse` | trapezoidal pulses through R_s, C_d | v_base 0, v_amp = fold V_LU + 0.10 V, width (flat top) 200 µs, period 1 ms, rise/fall 10 µs, n_pulses 10 (≤ 2000), delay 0, R_s 1 kΩ, C_d 2 fF, amplitudes_V [] (sweep) |
| `pbit` | clocked supply → R_L → drain (C_d), comparator on v_D | v_low 0, v_high = fold V_LU − 0.02 V, period 1 ms, width 200 µs, rise/fall 20 µs, n_clocks 50 (≤ 5000), R_L 100 kΩ, C_d 2 fF, cmp_threshold = v_high − R_L·100 nA, vg_list_V [], light_list_pA [] |
| `coupled` | common ramp or pulse source; R_s1, R_s2 to the drains d1, d2; R_c between d1 and d2 | source "ramp", ramp and pulse keys as above, R_s1 = R_s2 = 100 kΩ, R_c 1 MΩ, C_d 2 fF each, vg2_V = vg_V, iph2_pA = device light |

* `pulse`: P_sw = fraction of pulses in which the cell is latched (§4 latch state) at the end of the
  flat top; P_retained = still latched at the end of the period; switching delay from the
  pulse start (first latch-up in the period). `amplitudes_V` → sweep `P_sw_vs_amplitude`
  (n_runs per point, binomial error bars).
* `pbit`: bit = comparator output [v_D < v_th] at the end of the clock-high phase (the latched cell
  pulls v_D down by R_L I_LRS). P(1) (comparator output), `P_latched` (fraction of clocks with the cell
  latched), pooled lag-1 autocorrelation. With the channel on (V_G above ≈ −0.5 V) the comparator reads
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
| `v_src` (`v_clk` in pbit) | V | voltage | supply |
| `v_d` | V | voltage | drain node voltage (= device V_DS, source grounded) |
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
`v_th`, `v_high`, `fold_V_LU`. Always: `runs`, `steps_per_run`; stochastic with carrier noise:
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
| `netlist.py` | netlist builder (R, C, V, I, STL, CMP), PWL waveforms, compilation to arrays, schematic |
| `benches.py` | bench defaults, builders, statistics helpers (no numba import) |
| `stochastic.py` | local states, branch profiles (τ_rel, rates, barrier z), noise bands, feasibility estimate |
| `runner.py` | `run_circuit`: parsing, feasibility, run loop, analysis, result |
| `validate.py` | validation V1–V6 |
