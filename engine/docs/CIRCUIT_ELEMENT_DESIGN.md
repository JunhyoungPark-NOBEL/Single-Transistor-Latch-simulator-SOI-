# STL as a circuit element

## Element interface
Terminals: drain (D), source (S), gate (G). Optional light input I_PH (A) and back-gate. Internal state: Q_B (C). Outputs: I_D (A), and for diagnostics u, r, F.

## Evaluation (per Newton iteration, per time step)
1. Given V_DS, V_GS, Q_B: solve u from Q_B = Q_B(u, r) and r from V_DS = u + r + hole_drop(u,r) + R_acc·I_D(u,r) (2 unknowns, tables or `components()`; Newton with the table gradients; warm start from the previous step).
2. I_D = I_D(u, r) → stamp as a nonlinear current source with conductances ∂I_D/∂V_DS, ∂I_D/∂V_GS (finite differences on the table).
3. State update: Q_B(t+Δt) = Q_B(t) + F(u, r)·Δt + ΔQ_noise (backward Euler: solve implicitly for the deterministic part; noise added explicitly).
4. Latch-up/latch-down emerge from the ODE: no explicit switch. The hysteresis is the fold structure of F = 0.

## Noise options (menu in the UI)
- none (deterministic); carrier noise (Eq. 2 events); + frozen local states (draw δφ_G, δφ_E per cycle); + evolving states (OU with τ); + local avalanche path (p[21]–p[25], experimental).
- Reproducibility: seeded RNG; common random numbers across parameter variants.

## Suggested minimal circuit set
Voltage source (DC, pulse, ramp), resistor, capacitor, STL element, node voltage probe, current probe. MNA with backward Euler / trapezoidal; adaptive Δt: shrink when |ΔQ_B|/Q_B or |ΔI_D|/I_D exceeds thresholds (the switching takes ~ns–µs; the pre-latch phase ms–s).

## Test benches
1. Drain sweep through a series resistor (load line): reproduce V_LU/V_LD and the I–V of `stl_api.branches`.
2. Pulse train: P_sw vs pulse amplitude/width/interval (see project notes on pulse schemes; residual Q_B gives interval memory).
3. p-bit: STL + load resistor + comparator; output bit statistics vs V_G / light.
4. Two coupled STLs (resistive coupling) as the first Ising/p-bit network element.
