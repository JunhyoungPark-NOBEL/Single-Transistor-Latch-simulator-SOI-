# STL simulator HTTP API

Backend: FastAPI (`server/main.py`), job pool `server/jobs.py`, payload validation `server/payloads.py`.
The contract for compute kinds and result shapes is `docs/WEB_CONTRACT.md` (§1–§4); this file documents the
HTTP surface, the backend-core result details and the data endpoints.

## Conventions

* JSON everywhere (orjson). `NaN`, `+inf`, `-inf` are sent as `null`. Arrays are plain JSON arrays
  (2-D arrays are arrays of rows).
* Units: voltage V, current A, charge C, time s, optical power mW, photocurrent pA (payload) / A (results),
  resistance Ω. Where a key carries a unit suffix (`_mV`, `_pA`, `_nm`, `_s`) that unit wins.
* Responses larger than 2 kB are gzip-compressed when the client sends `Accept-Encoding: gzip`.
* Errors: `{"detail": "message"}` with HTTP 404 (unknown kind / job / path) or 422 (invalid payload).
  A `ValueError` raised *inside* a job gives `status: "error"` with `error: "message"` (HTTP 200).

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | `{ok, version, app_version, engine_version, workers, jobs:{queued,running,done,error,cancelled}}` — `version` = first 12 hex digits of `engine_version` |
| GET | `/api/meta` | `params.meta()` (`presets`, `constants`, `channel_seed_options`, `measured_photo_conditions`) + `kinds` (contract kinds), `extra_kinds` (`["folds"]`), `kinds_available` (module present?), `caps`, `engine_version`, `app_version`, `workers` |
| POST | `/api/compute/{kind}?wait=2.0` | body = payload (JSON object). Validates/clamps, submits, waits up to `wait` s (0–60). Returns `JobStatus`. |
| GET | `/api/jobs/{job_id}?wait=0` | `JobStatus`; optional long-poll `wait` (s, ≤ 60) |
| DELETE | `/api/jobs/{job_id}` | cancel → `JobStatus` with `status: "cancelled"` |
| GET | `/api/jobs` | recent jobs (`JobStatus` without `result`, oldest first; the last ~200 are kept) |
| GET | `/api/data/measured` | measured data, see below |
| GET | `/api/data/design_map` | design map, see below |
| GET/POST | `/api/branches`, `/api/folds`, `/api/hazard`, `/api/sweeps`, `/api/vg_curve` | handoff-name aliases of `/api/compute/{branches, folds, hazard, sweep_mc, vg_curve}`; POST takes the payload, GET takes query parameters (below) |
| GET | `/api/design_map` | alias of `/api/data/design_map` |
| GET | `/`, `/{path}` | the built frontend `web/dist` (SPA fallback to `index.html`; missing assets → 404). Without a build: a small HTML note. |
| GET | `/docs`, `/openapi.json` | FastAPI's interactive docs |

```ts
interface JobStatus { job_id: string; kind: string;
  status: "queued" | "running" | "done" | "error" | "cancelled";
  progress: number /* 0..1 */; message: string; result?: any /* only when done */;
  error?: string /* only when error */; cached: boolean; elapsed_s: number }
```

Typical client flow: `POST /api/compute/branches?wait=1.5` → if `status` is `done` use `result`; otherwise
poll `GET /api/jobs/{id}` (e.g. every 0.3–1 s, or long-poll with `?wait=5`) until the status is final;
`DELETE /api/jobs/{id}` to cancel (reported immediately; the worker stops at its next progress call).

**Kinds:** `branches`, `charge_balance`, `vg_curve`, `hazard`, `sweep_mc`, `vg_curve_stochastic`,
`circuit`, `validation` (contract) and `folds` (backend-core extra, folds only). Unknown kind → 404.

**Deduplication and cache.** An identical payload that is still queued/running returns the same job.
Finished results are cached in memory (LRU, 256 entries / 256 MB) and on disk (`server/.cache/results/*.json.gz`,
pruned to `STL_DISK_CACHE_MB`, default 1024) under
`sha256(kind, canonical JSON of the normalised payload, clamp warnings, ENGINE_VERSION)`;
`ENGINE_VERSION` hashes `server/compute/**/*.py`, `server/params.py`, `server/engine_bridge.py`, so a code
edit invalidates old entries. Canonical JSON sorts keys and treats `2` and `2.0` alike. A cache hit is a new
job with `cached: true`, `elapsed_s: 0`.

### GET alias query parameters

`preset`, `vg` (V), `iph_pA` (→ light mode `iph`) or `power_mW` (→ mode `power`), `grid`, `dg`, `de`
(→ `state.delta_phi_G0_V/E0_V`), `vd_max`, `rate`, `dv` (→ `sweep`), `n`, `seed` (`/api/sweeps`: `n_cycles`,
`seed`; `/api/vg_curve`: points), `vg_min`, `vg_max`, `wait` (default 2 s). Example: `GET /api/folds?vg=-1.8`.

## Payload normalisation and caps

`server/payloads.py` runs in the API process before submission:

* `device` is resolved against its preset (`params.resolve_device`); every numeric field must be finite
  (`vg` within ±10 V, light values ≥ 0, `light.mode` ∈ {iph, power}, `ext.loc_carriers` ∈ {0,1,2}).
* `sweep` (kinds branches, hazard, sweep_mc, vg_curve_stochastic) and `stochastic` (sweep_mc,
  vg_curve_stochastic) are filled from the preset defaults.
* Clamped with a warning (prepended to `result.warnings`): `device.numerics.grid` 201–2001,
  `sweep.vd_max_V` 0.1–8, `sweep.dv_V` 1e-4–0.1, `sweep.rate_V_per_s` 1e-4–1e8, `stochastic.n_cycles` 1–2000,
  `fold_nodes` 3–61, `hazard_nodes` 1–9, `n_traces` 0–50, `n` (V_G points) 2–61, `charge_balance.vd` 0.05–8,
  `n_u` 21–2001, `circuit.solver.max_steps` ≤ 2e6, `circuit.stochastic.n_runs` ≤ 200.
* Structural errors (unknown preset, non-numeric values, `vg_min ≥ vg_max`, unknown `local_state.mode`,
  `validation.level` not fast/full, …) → HTTP 422.
* `vg_curve` defaults (`vg_min` −4.2, `vg_max` −0.6, `n` 37) are filled here; `vg_curve_stochastic` keeps the
  stochastic package's own defaults.

`caps` in `/api/meta`: `{grid:[201,2001], n_cycles:2000, vg_curve_points:61, fold_nodes:61, hazard_nodes:9,
n_traces:50, circuit_max_steps:2000000, circuit_n_runs:200, vd_max_V:8, sweep_points_per_direction:2001}`.

## Backend-core results (additions to the contract shapes)

### `branches` — payload `{device, sweep?}`
Contract `BranchesResult` plus `grid` (state-grid points used) and `vd_max_V`.
* `HRS` = b[:i+1], `unstable` = b[i:j+1], `LRS` = b[j:] cut at the first row with V_D > vd_max + 1 V,
  `full` = the whole traced locus (up to V_D ≈ 12.7 V at the paper parameters).
* `comp.ii_total = I_D − seed − BTBT_j − GIDL − channel − I_PH` (channel-, photo- and BJT-seeded
  avalanche plus the optional local path); `comp.photo` = I_PH (constant); `injection` = δ/N_A;
  `hole_drop_V`, `r_access_ohm` as in `photo_mean.components`.
* `folds`: `V_LU`, `V_LD` from `FastModel.classify` (quadratic fold of V_D(u)); `u_LU/u_LD` = the vertex of that
  quadratic; `I_LU/I_LD` = log-interpolated along u at that vertex; `window_V = V_LU − V_LD`. All `null`
  when `latch` is false.
* `double_sweep`: quasi-static triangular sweep, `up` 0 → vd_max (HRS until V_LU, then LRS), `down`
  vd_max → 0 (LRS until V_LD, then HRS); currents log-interpolated along the branch; the fold jumps are two
  points at the same V_D (fold current, then the other branch). ≤ 2001 grid points per direction
  (+2 fold points; a finer `dv` is coarsened with a warning). If V_LU > vd_max the sweep stays on the HRS
  (warning). If there is no latch, both directions follow the monotone low-u part of `full`.

### `folds` (extra kind) — payload `{device}`
`{latch, folds, iph_A, p, grid, runtime_s, warnings}` (no arrays; ~0.3 s).

### `charge_balance` — payload `{device, vd, u_min?, u_max?, n_u?}`
Rows of `setup_photo.state(u, vd, p)` over `u` = 40 log-spaced points on [1e-9, 0.02] V (when `u_min` < 0.02,
default `u_min` = 0) plus `n_u` (401) linear points on [0.02, min(u_max, vd − 1 mV)], `u_max` default 1.10 V.
When the engine's r-bracket [0, vd−u] ends where `components()` is NaN (depletion reaches the neutral-length
limit), the bracket is shrunk to the finite part (same row formulas). Rows without a solution are skipped
(warning with the count).
* `Q_C = C_ox ψ + Q_exc + q N_A A L_n` (C), `holes = Q_C / q`.
* `generation_A = gen·q`, `loss_A = loss·q`, `unit_A` (GIDL + junction BTBT + I_PH), `ii_A = generation_A −
  unit_A`, `F_A` = net body current (= generation − loss).
* `potential`: `U(x) = −∫ ln(G/L) dx` on x = hole count (trapezoid), shifted to 0 at the first stable root
  (HRS well); dimensionless (log-probability units). With no stable root it is zeroed at its minimum.
* `roots[]`: `{u, Q_C, kind, id, r}` — sign changes of F refined with `brentq`; `stable` where F goes + → −
  with increasing u. Paper device, V_G = −2 V, V_D = 3.2 V: stable 0.4227 / unstable 0.6348 / stable 0.9014 V.
* also `iph_A`.

### `vg_curve` — payload `{device, vg_min, vg_max, n, refine?}`
Contract `VgCurveResult` plus `I_LD` and `grid`. `window.vg_low/vg_high` = latch-existence edges refined by
bisection to 1 mV between the last scanned no-latch point and the first latch point (`null` + warning when
the window reaches the scan boundary or there is no latch). `refine: false` skips the bisection. Paper device:
−3.906 … −0.810 V (paper: −3.90 … −0.815 V).

### `validation` — payload `{level: "fast" | "full"}`
Contract `ValidationResult` plus `level` and `summary {n, passed, failed, skipped}`. Check ids:

| id | level | expected (VALIDATION.md) |
|---|---|---|
| `fold_paper_vg-2`, `fold_paper_vg-1.8`, `fold_photo_2.63pA` | fast | 3.7037/2.5979, 3.8644/2.5979, 3.2913/2.596 V ± 1 mV |
| `extension_identity` | fast | photo_mean (extensions 0) ≡ gate_mean: 400 random (u, r), folds at 3 V_G, ≤ 1e-12 V |
| `light_conversion` | fast | 0.86 / 1.91 / 2.63 pA |
| `latch_window` | fast | V_G −3.90 … −0.815 V ± 10 mV |
| `measured_records` | fast | paper σ_LU 123.1 / σ_LD 19.5 mV; photo −1.8 V dark 3.806 V / 173.2 mV |
| `fpt_node` | fast | mean 3.644 V ± 3 mV, SD 8 ± 1.5 mV |
| `dynamic_mc` | fast | 3.63 V / 120 mV, 2.70 V / 20 mV |
| `paper_records` | full | σ_LU 125.8 ± 8, σ_LD 19.6 ± 1.5 mV (10 seeds × 100 sweeps) |
| `carrier_noise_breakdown` | full | II 4.6, BTBT 2.7, REC 4.3, DIFF 1.8, all 7.8 mV (± 1 mV) |
| `sweep_mc_photo_dark` | full | 3.806 V / 173 mV (± 20 mV) via `server.compute.stochastic.run_sweep_mc` |
| `vg_curve_stochastic_paper` | full | σ peak 129.8 mV at −1.25 V, mean peak 4.354 V at −1.10 V |
| `circuit_load_line` | full | latch-up/down events at the folds (± 30 mV) via `server.compute.circuit.run_circuit` |

Checks that depend on another package report `pass: null` with the reason when the module is missing or
raises. `carrier_noise_breakdown` is derived here (the paper's own code is not in the handoff): the compound
first-passage lattice with only one channel class as jumps (II clusters, unit generation = GIDL + junction
BTBT, REC = bulk + junction SRH loss, DIFF = emitter out-diffusion) and the others as deterministic drift,
discretised upwind on a lattice refined N = 4 and 8 and Richardson-extrapolated in log h. Result: II 5.1,
BTBT 3.0, REC 4.2, DIFF 1.9, all 8.0 mV.

## Data endpoints

### `GET /api/data/measured`

```jsonc
{
  "units": {"voltage": "V", "current": "A", "power": "mW", "photocurrent": "pA", "rate": "V/s"},
  "photo": {                                 // this device, 0→5 V triangular sweeps at 1200 V/s
    "description": "...", "rate_V_per_s": 1200, "vd_max_V": 5, "n_cycles": 400, "responsivity_pA_per_mW": 0.75,
    "conditions": [                          // 8 entries, order = params.MEASURED_PHOTO_CONDITIONS
      {"index": 0, "label": "-1.8V 0.00mW", "vg": -1.8, "power_mW": 0, "iph_pA": 0,
       "stats": Stats,                       // computed from the raw column (sd in V, lag1 as in the file)
       "file_stats": {...}}                  // measured_stats.json record as provided (sd_mV, trend_fraction, ...)
    ],
    "V_LU": number[8][400]                   // V_LU[k][c]: condition k, cycle c (chronological), V
  },
  "light_iv": {                              // idvd_light.npy: I_D–V_D under light (V_G −1.8 V assumed)
    "vg": -1.8, "vd": number[101], "power_mW": [0, 0.799, 1.42, 2.55, 2.80, 3.31],
    "iph_pA": number[6] /* 0.75 pA/mW × P */, "id": number[6][101], "plateau_vd_V": 1.25 },
  "idvg_dark": {                             // idvg_dark.npy: dark I_D–V_G at V_D = 0.05 V
    "vd_V": 0.05, "vg": number[161], "id": number[2][161] },          // two traces (identity unconfirmed)
  "paper_idvd": {                            // paper device, V_G −2 V, 0.4 V/s, 100 up + 100 down sweeps
    "vg": -2, "rate_V_per_s": 0.4, "n_sweeps": 100,
    "up":   {"vd": number[401] /* 0→4 V */, "median": number[401], "p10": number[401], "p90": number[401],
             "sample_index": number[10], "samples": number[10][401]},
    "down": {"vd": number[401] /* 4→0 V */, ...same keys},
    "V_LU": number[100], "V_LD": number[100],                        // (low + high)/2 of the 10 mV bracket
    "V_LU_bracket": {"low": number[100], "high": number[100]}, "V_LD_bracket": {...},
    "stats": {"LU": Stats, "LD": Stats} }                            // σ_LU 123.1 mV, σ_LD 19.5 mV
}
```

`Stats = {n, mean, sd, median, p05, p95, min, max, censored, lag1}` (V; `sd` with ddof = 1;
`lag1 = Σ dx_t dx_{t+1} / Σ dx_t²`).

### `GET /api/data/design_map`

```jsonc
{ "description": "...", "axes": {"x": "length_nm", "y": "depth_fraction"},
  "arrays": {
    "length_nm": number[81],            // nm, lateral size L of the local (trap) region, 3–100 (log)
    "depth_fraction": number[61],       // 1, depth as a fraction of the film, 0–0.85
    "sigma_phi_mV": number[61][81],     // mV, SD of the local drain-edge state   [depth][length]
    "sigma_VLU_mV": number[61][81],     // mV, SD of V_LU, 0–4 V sweep
    "latched_fraction": number[61][81], // 1, fraction of cycles latching within 0–4 V
    "sigma_VLU_sweep5p2V_mV": number[61][81],  // mV, SD of V_LU, 0–5.2 V sweep
    "expected_trap_count": number[61][81],     // 1, N_t L²
    "line_Nt": number[4],               // cm⁻², reference-line trap densities
    "line_L0_device": number[4],        // nm, L where σ_φ reaches the device value (153.4 mV) per line_Nt
    "line_L0_50": number[4] },          // nm, second reference contour per line_Nt
  "scalars": {"Nt_cm2": 1e12, "device_sigma_phi_mV": 153.39, "phi_50mV": 65.82},
  "shapes": {"<key>": [..]}, "doc": {"<key>": {"unit": "...", "meaning": "..."}} }
```

The meaning of the reference lines is inferred from the stored numbers (the npz has no metadata);
`doc` repeats units and meanings per key.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `STL_WORKERS` | CPUs − 1 (affinity/cgroup aware), ≥ 1 | compute worker processes |
| `STL_CACHE_DIR` | `server/.cache/results` | disk result cache |
| `STL_DISK_CACHE_MB` | 1024 | disk cache budget (oldest pruned at start-up) |
| `STL_PREWARM` | 1 | start all workers at start-up (numba cache load ≈ 1–2 s each) |
| `STL_MP_CONTEXT` | `spawn` | multiprocessing start method (`spawn` or `forkserver`) |
| `STL_CORS_ORIGINS` | localhost:5173/4173 | comma-separated allowed origins |
| `STL_WEB_DIST` | `web/dist` | built frontend directory |
| `PORT` | 8000 | (Docker `CMD`) listen port |
