# STL deterministic + stochastic model — specification for implementation

## 1. Device and state variables
- SOI n-MOSFET, L = 500 nm, W = 200 nm, T_Si = 50 nm, EOT 14.1 nm, body N_A = 2.30e17 cm^-3 (paper device). Constants in `model/.../idvd_model/mean_model.py`.
- Internal state: `u` = source–body forward bias (V), `r` = drain–junction reverse bias (V). Terminal: V_D = u + r + hole drop + R_acc·I_D; V_G enters the channel current and the gate-edge (GIDL) field.
- `components(u, r, p, na, vbi, rg, fg, table)` (numba) returns 19 numbers: [0] V_D, [1] I_D, [2] net hole current into the body F (A), [3] BJT seed electron current, [4] emitter, [5] bulk SRH loss, [6] out-diffusion loss, [7] junction SRH loss, [8] junction BTBT, [9] GIDL, [10] injection level δ/N_A, [11] length, [12] R_acc, [13] charge term, [14] w_s, [15] w_d, [16] channel current, [17] hole drop (V), [18] I_PH.
- Body charge Q_B(u, r) = (z[13] − C_ox·u)/(1 + ratio)·(1 + ratio) — see `export_tables.py` for the exact expression (qb + qa).

## 2. Deterministic core (Eq. 1)
dQ_B/dt = I_II + I_BTBT + I_GIDL + I_PH − I_REC − I_DIFF ≡ F(u, r; V_G, I_PH). Steady state F = 0 with Kirchhoff’s current law gives, for each u, the r and therefore V_D(u), I_D(u). The locus folds: low-current solutions exist up to V_LU (first V_D maximum along u), high-current solutions down to V_LD (last minimum). `FastModel.classify(p, state_grid(n))` → (branch array, i_fold_up, j_fold_down, [V_LU, V_LD]). Branch columns: 0 V_D, 1 I_D, 2 F, 3 seed, 5–7 losses, 8 junction BTBT, 9 GIDL, 16 channel, 17 u (index 17 in branch = z[17]? no: branch[:,17] = u, branch[:,18] = r, branch[:,20] = hole drop — see `curve_grid`).
- Multiplication M(r) and junction BTBT vs r: tables `MODEL.rg`, `MODEL.fg[0]`, `MODEL.fg[1]` (van Overstraeten–de Man 300 K, local-field kernel). Exported in `data/tables/multiplication_and_clusters.npz`.
- Channel: paper fit V_T0 = −0.4903 V, n = 1.7787, β0, θ (in `gate_mean.py`), V_D = 0.05 V I_D–V_G. Photo-extension adds optional DIBL (p[14]), body coupling (p[15]), slope factor (p[16]), high-V_D seed I_p·10^((V_G+1.8)/S) (p[17], p[18]).
- Photocurrent (p[13]): uniform hole supply I_PH, photo-electrons collected at the drain multiply like channel electrons. Conversion this device: I_PH = 0.75 pA/mW × P.

## 3. Parameter vector p (calibrated values in `data/tables/parameters.json`)
0 β (diffusion ratio) 7.17 · 1 τ_bulk 0.93 µs · 2 τ_junction 5.4 ns · 3 R_contact 1 Ω · 4 l_GIDL 28.8 nm · 5 t_access 3.3 nm · 6 N_A,access 1e17 · 7 L_access 70 nm · 8 τ_p/τ_n 117 · 9 φ_GIDL offset (drain-edge state mean) 0.27 mV · 10 φ_emitter offset 0.059 mV · 11 V_G · 12 channel-II scale 1 · 13 I_PH · 14–25 extensions (0 = paper model).
Local-state amplitudes: σ_φG = 0.1534 V (drain edge, acts on the GIDL field through p[9]), σ_φE = 0.437 mV (source edge, p[10]). Kinetics (paper device, 0.4 V/s sweeps): drain-edge OU τ = 5 s within a sweep with an acquisition trend (66 % of variance), emitter OU τ = 1.62 s.

## 4. Stochastic part (Eq. 2)
ΔQ_B = q Σ_i s_i ΔN_i over a step Δt (paper: 5 ms per 2 mV at 0.4 V/s).
- Unit events (Poisson): GIDL, junction BTBT, photo-generation, recombination, diffusion — rates from the rows (columns of `conditional_table.state`: [3] total generation rate, [4] loss rate, [10] unit-event current).
- Impact ionization: cluster events. Cluster size pmf vs reverse voltage in `hypothesis_study_20260920/avalanche/cluster_pmf.npz` (`cluster_reverse_V`, `cluster_pmf`); rate = (generation − unit)/mean cluster size.
- Escape from the metastable low-current state during a ramp: compound-jump first passage on the charge lattice (`compound_fpt.make_lattice/backward`), giving hazard h(V_D) below the fold; V_LU quantiles = ∫h dV/rate. `photo_fpt.hazard_curve(vg, iph, dg, de, ...)` and `quantiles(rec, rate)`.
- Local states (frozen or evolving): drain-edge potential δφ_G ~ N(δφ_G0, σ_φG) shifts the GIDL field (p[9]); source-edge δφ_E shifts p[10]. Fold sensitivities at V_G = −2 V: dV_LU/dφ_G ≈ −0.80 V/V, dV_LD/dφ_E ≈ −41 V/V.
- Sweep MC with evolving states: `gate_dynamic_compare.simulate(n, seed, mode, dv, rate)`; lookup `gate_state_lookup.npz` (folds and hazards vs the two states).

## 5. Circuit-level formulation
State ODE: dQ_B/dt = F(u, r) + noise; u = U(Q_B) from the inverse of Q_B(u, r) (r enters weakly through the depletion term); r from V_D − u − hole drop − R_acc·I_D. Current I_D(u, r). Tables on a 101×101 (u, r) grid per (V_G, I_PH) in `data/tables/*.npz` (u 0–1 V, r 0–5 V; invalid cells NaN). Time step: Δt ≤ 1 ms for sweeps; for pulses use Δt ≤ τ_junction-scale only inside the LRS transition (stiff: F changes by decades).
Stochastic increment per Δt: ΔN_unit ~ Poisson(I_unit·Δt/q) with sign; ΔN_II = Σ over Poisson(rate_II·Δt) clusters of size drawn from the pmf at the current r; Gaussian limit when counts > ~100.

## 6. Sweep protocols used
- Paper device: V_D 0 → 4 V and back, 10 mV steps, ~10 s (0.4 V/s), 100 up + 100 down, V_G = −2 V.
- Photo device (data/): triangular 1200 V/s, 0 → 5 V, 400 cycles, V_G = −1.8 / −1.1 V, light 0 / 1.15 / 2.55 / 3.51 mW; frozen-state limit valid (cycle 8 ms ≪ state τ); measured lag-1 autocorrelation 0.0–0.5.
