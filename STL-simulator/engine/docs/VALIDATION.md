# Validation numbers (must reproduce; `python3 stl_api.py`)
| Check | Expected |
|---|---|
| Paper model, VG = −2 V dark, folds (V_LU, V_LD) | 3.7037 V, 2.5979 V |
| Paper model, VG = −1.8 V dark, folds | 3.8644 V, 2.5979 V |
| Photo model, VG = −1.8 V, I_PH = 2.63 pA, folds | 3.2913 V, 2.596 V |
| Paper model at all extension parameters = 0 | identical to gate_mean (verified to 1e-12 V) |
| FPT node VG = −2 V dark, centre states, ramp 0.4 V/s | mean V_LU ≈ 3.644 V, SD ≈ 8 mV |
| Dynamic MC, 100 sweeps, seed 2026092920, 0.4 V/s | V_LU mean ≈ 3.63 V, SD ≈ 120 mV; V_LD mean ≈ 2.70 V, SD ≈ 20 mV |
| Paper values (10 records × 100 sweeps) | σ_LU 125.8 mV, σ_LD 19.6 mV (measured 123.1 / 19.5) |
| Carrier noise only (paper, VG = −2 V) | II 4.6, BTBT 2.7, REC 4.3, DIFF 1.8, all four 7.8 mV |
| Latch window (paper, fixed states) | VG = −3.90 … −0.815 V; σ_LU peak 129.8 mV at −1.25 V; mean peak 4.354 V at −1.10 V |
| Light conversion (this device) | I_PH = 0.75 pA/mW × P; 0.86 / 1.91 / 2.63 pA at 1.15 / 2.55 / 3.51 mW |
| This device, VG = −1.8 V dark, 1200 V/s | measured mean 3.806 V, SD 173 mV (400 cycles); calibration δφ_G0 = +0.074 V, σ_φ = 0.215 V (paper-model levers) |
