# STL Web Simulator — implementation contract

This document is the single source of truth shared by all work packages (backend core, stochastic
engine, circuit engine, physics content, frontend). If something here is ambiguous, pick the
simplest reading that keeps the other packages working and record the decision in
`docs/DECISIONS.md` (append-only, one bullet per decision, prefixed by the package name).

User requirements (from the owner, KAIST NOBEL lab):
1. User-friendly web UI for the deterministic + stochastic STL (single-transistor latch, SOI n-MOSFET) model.
2. The user chooses **Deterministic** or **Stochastic** mode explicitly (global, prominent toggle).
3. Parameters are **grouped** (collapsible groups) and configurable.
4. The applied physics and equations are written as **exact expressions** (matching the code), and a
   **“상세 (Details)” button** opens a **compact floating window on the same screen** (no page change)
   with the detailed theory, equations, variables and assumptions.
5. Handoff plan (`engine/docs/00_START_HERE_websim_KO.md`): device simulator (I–V branches + MC sweeps,
   V_LU/V_LD histograms/CDF, V_G curve, design map) and circuit simulator (STL as a state-space
   element with MNA transient, deterministic first, then event increments), validation numbers
   (`engine/docs/VALIDATION.md`). Open problems must stay *options*, not forced answers.

## 0. Layout and ownership

```
engine/          handoff package, verbatim (numerical model + data). Do not edit; if a genuine bug
                 blocks you, write a wrapper in server/ instead and note it in docs/DECISIONS.md.
server/          FastAPI service (Python ≥ 3.11)
  params.py        [foundation, shared — do not change signatures] device schema, presets, build_p()
  engine_bridge.py [foundation] imports engine (A, S, m, MODEL, ct); worker processes only
  progress.py      [foundation] Progress protocol, JobCancelled
  main.py, jobs.py, jsonutil.py, compute/__init__.py (registry), compute/deterministic.py,
  compute/data.py, compute/validation.py, tests/test_api.py, tests/test_deterministic.py
                   [backend-core]
  compute/stochastic.py, tests/test_stochastic.py            [stochastic]
  compute/circuit/ (package), tests/test_circuit.py          [circuit]
  requirements.txt, ../Dockerfile, ../.dockerignore, ../scripts/*  [backend-core]
web/             Vite + React 18 + TypeScript frontend        [frontend]
  src/content/physics/   types.ts [foundation], all other files     [physics-content]
docs/WEB_CONTRACT.md [foundation], docs/DECISIONS.md [everyone, append-only]
app.py, config.toml, requirements.txt (repo root)  legacy Streamlit app — leave untouched.
```

Rules for every package: stay inside your files; never `git commit`/`git push` (the orchestrator
commits); never delete other packages' files; don't name any folder `lib` (root .gitignore ignores it);
Python deps are already installed globally (numpy, scipy, numba, fastapi, uvicorn); install others
with `pip install` and list them in `server/requirements.txt` (backend-core owns the file — other
packages append a line, don't rewrite it). Node 22 + npm are available; Chromium for Playwright is
at `/opt/pw-browsers` (never run `playwright install`). The machine has 4 CPUs shared by all
packages — keep test runs modest.

## 1. Compute function protocol

Every compute kind is a Python function

```python
def run(payload: dict, progress: Progress) -> dict   # progress(fraction 0..1, message)
```

registered in `server/compute/__init__.py`:

```python
KINDS = {  # kind -> "module:function" (imported lazily inside worker processes)
  "branches":            "server.compute.deterministic:run_branches",
  "charge_balance":      "server.compute.deterministic:run_charge_balance",
  "vg_curve":            "server.compute.deterministic:run_vg_curve",
  "hazard":              "server.compute.stochastic:run_hazard",
  "sweep_mc":            "server.compute.stochastic:run_sweep_mc",
  "vg_curve_stochastic": "server.compute.stochastic:run_vg_curve_stochastic",
  "circuit":             "server.compute.circuit:run_circuit",
  "validation":          "server.compute.validation:run_validation",
}
```

Return values may contain numpy arrays/scalars; the server serialises them with NaN/±inf → `null`.
Raise `ValueError("human readable message")` for invalid input (→ HTTP 422 / job error shown in UI).
Every result dict contains `runtime_s` (float) and `warnings` (list[str], possibly empty).
Payloads are validated/clamped server-side (caps below) to keep the service responsive.

### Shared payload blocks

`device` (resolved with `params.resolve_device`; every field optional in requests):

```jsonc
{
  "preset": "paper" | "photo" | "custom",
  "vg": -2.0,                                   // V_G (V)
  "light": { "mode": "iph" | "power", "iph_pA": 0, "power_mW": 0, "responsivity_pA_per_mW": 0.75 },
  "calib": { "beta", "tau_bulk_s", "tau_junction_s", "r_contact_ohm", "l_gidl_nm", "t_access_nm",
             "na_access_cm3", "l_access_nm", "tau_ratio", "phi_gidl0_V", "phi_emitter0_V",
             "channel_ii_scale" },             // p[0..10], p[12]
  "ext":   { "dibl", "gamma", "kappa", "seed_ip_pA", "seed_S", "dj", "dm", "aloc", "isat_pA",
             "dloc", "loc_carriers" (0|1|2), "kappaF" },   // p[14..25]
  "state": { "delta_phi_G0_V", "delta_phi_E0_V" },       // local-state centre, added to p[9], p[10]
  "numerics": { "grid": 601 }                   // state_grid points for classify (201..2001)
}
```

`sweep`: `{ "vd_max_V": 4.0, "rate_V_per_s": 0.4, "dv_V": 0.002 }` (triangular 0 → vd_max → 0).

`stochastic`:

```jsonc
{
  "n_cycles": 100, "seed": 2026092920,
  "carrier_noise": true,        // Eq. 2 first-passage noise (II clusters + unit events). false → escape at the fold
  "ld_carrier_noise": false,    // also compute the latch-down FPT (slower)
  "local_state": {
    "mode": "none" | "frozen" | "evolving",   // frozen: one draw per cycle; evolving: OU in time
    "action": "gidl" | "local_avalanche" | "junction" | "multiplication",
       // gidl → p[9] (paper), local_avalanche → p[23] with p[21],p[22],p[24],p[25] from ext (photo, experimental),
       // junction → p[19], multiplication → p[20]  (hypotheses.py levers, experimental)
    "sigma": 0.1534,            // SD of the action-point state (V for gidl/junction, ln-units otherwise)
    "tau_s": 5.0,               // OU correlation time of that state (evolving mode)
    "sigma_E_V": 0.000437,      // emitter state SD (p[10]); 0 disables
    "tau_E_s": 1.62,
    "acquisition_trend": true   // paper record only: calibrated up-sweep trend (calibrated_lookup engine)
  },
  "engine": "auto" | "general" | "calibrated_lookup",
      // auto → calibrated_lookup iff params.is_paper_reference(device) and action == "gidl"
  "n_traces": 12, "fold_nodes": 25, "hazard_nodes": 5
}
```

Presets (`params.PRESETS`, served by `GET /api/meta`): **paper** (V_G −2 V, dark, 0–4 V, 0.4 V/s,
evolving states σ_φG 0.1534 V τ 5 s, σ_φE 0.437 mV τ 1.62 s, 100 cycles, seed 2026092920) and
**photo** (V_G −1.8 V, power mode 0/1.15/2.55/3.51 mW with 0.75 pA/mW, γ = 0.2794, 0–5 V,
1200 V/s, frozen states δφ_G0 = +0.0744 V σ = 0.2154 V, 400 cycles, seed 20260922), and **custom**.

Caps (server clamps and adds a warning): grid ≤ 2001; n_cycles ≤ 2000; vg_curve points ≤ 61;
fold_nodes ≤ 61; hazard_nodes ≤ 9; circuit max_steps ≤ 2e6, n_runs ≤ 200; vd_max_V ≤ 8.

## 2. Result shapes (TypeScript notation; Python returns the same keys)

```ts
type Arr = (number | null)[];
interface Stats { n: number; mean: number|null; sd: number|null; median: number|null;
                  p05: number|null; p95: number|null; min: number|null; max: number|null;
                  censored: number; lag1: number|null }            // voltages in V, sd in V

// kind "branches"   payload {device, sweep?}
interface Components {  // A unless noted, aligned with vd
  channel: Arr; seed: Arr; ii_total: Arr; btbt_junction: Arr; gidl: Arr; photo: Arr;
  loss_bulk_srh: Arr; loss_diffusion: Arr; loss_junction_srh: Arr; net_F: Arr;
  hole_drop_V: Arr; injection: Arr /* δ/N_A */; r_access_ohm: Arr }
interface Curve { vd: Arr; id: Arr; u: Arr; r: Arr; comp: Components }
interface BranchesResult {
  latch: boolean;                                   // false when classify() finds no two-fold branch
  HRS: Curve; unstable: Curve; LRS: Curve;          // b[:i+1], b[i:j+1], b[j:]  (LRS truncated at vd_max+1 V)
  full: Curve;                                      // whole traced locus (for "no latch" cases)
  folds: { V_LU: number|null; V_LD: number|null; I_LU: number|null; I_LD: number|null;
           u_LU: number|null; u_LD: number|null; window_V: number|null };
  double_sweep: { up: {vd: Arr; id: Arr}; down: {vd: Arr; id: Arr} };  // quasi-static 0→vd_max→0
  iph_A: number; p: number[]; runtime_s: number; warnings: string[] }

// kind "charge_balance"   payload {device, vd: number}  — body-charge landscape at fixed V_D
interface ChargeBalanceResult {
  vd: number; u: Arr; r: Arr; id: Arr; Q_C: Arr /* C_ox ψ + Q_exc + qN_A A L_n */;
  generation_A: Arr; loss_A: Arr; unit_A: Arr; ii_A: Arr; F_A: Arr;
  potential: Arr   /* U(x) = −Σ ln(G/L) along the hole-count lattice, 0 at the HRS well */;
  roots: { u: number; Q_C: number; kind: "stable" | "unstable"; id: number }[];
  runtime_s: number; warnings: string[] }

// kind "vg_curve"   payload {device, vg_min, vg_max, n}
interface VgCurveResult { vg: Arr; V_LU: Arr; V_LD: Arr; I_LU: Arr; latch: boolean[];
  window: { vg_low: number|null; vg_high: number|null }; runtime_s: number; warnings: string[] }

// kind "hazard"   payload {device, sweep, dg?: number, de?: number}
interface HazardResult { fold_V: number|null; VLD_fold_V: number|null; voltage: Arr; hazard: Arr /* 1/s */;
  survival: Arr /* S(V) for sweep.rate */; quantiles: { prob: Arr; v: Arr }; stats: Stats;
  rate_V_per_s: number; runtime_s: number; warnings: string[] }

// kind "sweep_mc"   payload {device, sweep, stochastic}
interface SweepMCResult {
  engine: "calibrated_lookup" | "general";
  V_LU: Arr; V_LD: Arr;                             // one per cycle, null = censored
  stats: { LU: Stats; LD: Stats };
  hist: { LU: {edges: number[]; counts: number[]}; LD: {edges: number[]; counts: number[]} };
  cdf:  { LU: {v: number[]; p: number[]}; LD: {v: number[]; p: number[]} };
  traces: { cycle: number; V_LU: number|null; V_LD: number|null;
            up: {vd: Arr; id: Arr}; down: {vd: Arr; id: Arr} }[];
  cycle_state: Arr;                                 // action-point state per cycle (mid up-sweep)
  fold_table: { delta: Arr; V_LU: Arr; V_LD: Arr } | null;
  centre: { V_LU: number|null; V_LD: number|null; HRS: {vd: Arr; id: Arr}; LRS: {vd: Arr; id: Arr} };
  measured: { label: string; V_LU: Arr; V_LD: Arr | null; stats: { LU: Stats; LD: Stats | null } } | null;
  runtime_s: number; warnings: string[] }

// kind "vg_curve_stochastic"   payload {device, sweep, stochastic, vg_min, vg_max, n}
interface VgCurveStochasticResult { vg: Arr; mean_VLU: Arr; sd_VLU_mV: Arr; state_sd_mV: Arr;
  noise_sd_mV: Arr; fold_centre_V: Arr; VLD_fold_V: Arr; no_latch_weight: Arr;
  measured: { vg: number; power_mW: number; mean_V: number; sd_mV: number }[];
  runtime_s: number; warnings: string[] }

// kind "circuit"  — see §4
// kind "validation"   payload {level: "fast" | "full"}
interface ValidationResult { checks: { id: string; label: {ko: string; en: string}; expected: string;
  computed: string; pass: boolean | null; tolerance: string; note?: string; seconds: number }[];
  runtime_s: number; warnings: string[] }
```

Measured data (`GET /api/data/measured`), design map (`GET /api/data/design_map`): raw arrays with
units, see backend implementation; keys documented in the endpoint docstring and `docs/API.md`.

## 3. HTTP API (backend-core)

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | `{ok, version, workers}` |
| GET | `/api/meta` | `params.meta()` + `kinds` list + caps |
| POST | `/api/compute/{kind}?wait=2.0` | body = payload. Submits a job; waits up to `wait` s. Returns `JobStatus`. Identical payloads hit the cache. |
| GET | `/api/jobs/{job_id}` | `JobStatus` |
| DELETE | `/api/jobs/{job_id}` | cancel |
| GET | `/api/data/measured` | photo raw V_LU (400×8) + stats, light I–V, dark I_D–V_G, paper-device 100-sweep IDVD (subsampled) + V_LU/V_LD |
| GET | `/api/data/design_map` | design_map_filled.npz arrays |
| aliases | `/api/branches`, `/api/folds`, `/api/hazard`, `/api/sweeps`, `/api/vg_curve`, `/api/design_map` | handoff names, thin wrappers |

```ts
interface JobStatus { job_id: string; kind: string; status: "queued"|"running"|"done"|"error"|"cancelled";
  progress: number; message: string; result?: any; error?: string; cached: boolean; elapsed_s: number }
```

Jobs run in a `ProcessPoolExecutor` (workers import `server.engine_bridge` once; env `STL_WORKERS`,
default `max(1, cpu_count-1)`); results cached in memory (LRU) and on disk under `server/.cache/`
keyed by sha256(kind + canonical JSON of the payload + `ENGINE_VERSION`). The API process never
imports numba. The server serves `web/dist` at `/` when it exists (SPA fallback to index.html).
Dev: `uvicorn server.main:app --port 8000`, Vite on 5173 proxies `/api` → 8000.

## 4. Circuit simulator (circuit package)

Request `kind: "circuit"`:

```jsonc
{ "bench": "load_line" | "pulse" | "pbit" | "coupled",
  "mode": "deterministic" | "stochastic",
  "device": { ... },                 // as §1; bench may override vg / light per element
  "bench_params": { ... },           // bench specific, defaults documented in compute/circuit/benches.py
  "solver": { "method": "BE" | "TRAP", "dt_min_s", "dt_max_s", "reltol", "max_steps" },
  "stochastic": { "seed", "n_runs", "carrier_noise", "local_state": { ...as §1... } },
  "detect": { "i_threshold_A": 1e-8 } }
```

Response (generic so the UI can render any bench without special cases):

```ts
interface Signal { key: string; label: {ko: string; en: string}; unit: "V"|"A"|"C"|"s"|"1"|"A/V"|string;
                   values: Arr; axis?: "voltage"|"current"|"charge"|"state"|"logic" }
interface CircuitRun { run: number; t: Arr; signals: Signal[] }      // decimated to ≤ 4000 points
interface SummaryItem { key: string; label: {ko: string; en: string}; value: number|string|null; unit?: string;
                        spread?: number|null /* SD over runs */ }
interface CircuitResult {
  bench: string; mode: string;
  runs: CircuitRun[];                         // run 0 always present; stochastic: first ≤ 8 runs
  events: { run: number; kind: "latch_up"|"latch_down"|"bit"|string; t: number; value?: number; v_src?: number; v_d?: number }[];
  summary: SummaryItem[];
  distributions?: { key: string; label: {ko: string; en: string}; unit: string; values: Arr }[];  // e.g. V_LU per cycle over runs
  sweeps?: { key: string; label: {ko: string; en: string}; x: Arr; x_label: string; x_unit: string;
             y: Arr; y_label: string; y_unit: string; y_err?: Arr }[];  // e.g. P_sw vs amplitude
  trajectory?: { vd: Arr; id: Arr };          // run 0 drain trajectory for the I–V overlay
  schematic: { nodes: string[]; elements: { kind: "V"|"R"|"C"|"STL"|"I"|"CMP"; name: string; nodes: string[]; value?: string }[] };
  solver_stats: { steps: number; rejected: number; newton_iters: number; runtime_s: number };
  runtime_s: number; warnings: string[] }
```

STL element model (see `engine/docs/CIRCUIT_ELEMENT_DESIGN.md`): terminals D, G, S; state Q_B;
internal unknowns (u, r) satisfy V_D(u,r) = v_D − v_S and the charge equation
Q(u,r;V_GS) − Q_prev = ∫F dt (+ ΔQ_noise), with
Q(u,r) = C_ox(ψ(u) − V_GS) + Q_exc(u,r) + q N_A A L_n(u,r),  ψ = u − V_T ln(1+δ/N_A)
(the same charge coordinate as the compound-FPT lattice, `compound_fpt.make_lattice`, up to the
constant −C_ox V_G). Deterministic: implicit BE/TRAP + Newton, adaptive Δt. Stochastic: Eq. 2 event
increments (Poisson unit events + II clusters from the pmf at the current r + Poisson losses; Gaussian
limit for large counts), local states frozen/OU as in §1.

## 5. Frontend (frontend package)

Stack: Vite + React 18 + TypeScript (strict), Plotly (`plotly.js-dist-min` via `react-plotly.js/factory`),
KaTeX, Zustand (state), light custom CSS with CSS variables (light + dark theme). Optional: Radix UI
primitives for tooltip/tabs/accordion. No UI framework that fights custom styling.

Global layout: header (title, primary tabs **소자 Device · 회로 Circuit · 검증 Validation · 물리 모델
Physics**, the **Deterministic | Stochastic** segmented toggle — large and always visible, KO/EN toggle,
theme toggle); left sidebar with the preset selector, grouped parameter cards and a sticky **Run**
button (+ progress, cancel, elapsed); main area with a KPI strip and a responsive grid of plot panels.

Parameter groups (collapsible cards, each with a **상세** button → physics topic; modified values
marked with a dot and per-group "reset"; every input shows symbol, unit, code index tooltip, range
validation):
1. Bias & sweep — V_G, V_D max, ramp rate, ΔV  (topic `charge-balance`)
2. Illumination — mode (I_PH pA | optical power mW), value, responsivity  (topic `photo`)
3. Local-state centre — δφ_G0, δφ_E0  (topic `local-states`)
4. Calibrated device parameters p[0..10], p[12] (advanced, collapsed by default)  (topic `parameters`)
5. Model extensions / open problems — channel-seed option (none | body coupling γ p[15] | high-V_D seed
   p[17],p[18]), DIBL p[14], κ p[16], junction offset p[19], (M−1) scale p[20], local avalanche path
   p[21]–p[25] (experimental)  (topic `open-problems`)
6. Stochastic (visible only in Stochastic mode) — cycles, seed, carrier noise, local-state mode,
   action point, σ, τ, σ_E, τ_E, engine  (topics `stochastic-events`, `local-states`)
7. Numerics (collapsed) — grid, fold/hazard nodes  (topic `numerics`)
Circuit tab: bench selector (cards with mini SVG schematics) + bench groups + solver group
(topic `circuit-element`).

Device tab panels — Deterministic: (a) I–V branches HRS/unstable/LRS + folds + quasi-static
double-sweep arrows (+ measured overlay toggle); (b) current components along the branch;
(c) body-charge balance at a chosen V_D (G, L, F and quasi-potential, roots); (d) V_G curve of the
folds (latch window). Stochastic: (a) I–V + MC sweep traces + V_LU/V_LD markers; (b) V_LU/V_LD
histogram ↔ CDF (+ measured overlay); (c) hazard h(V_D) and survival; (d) V_G curve mean ± σ;
(e) cycle series; (f) design map heatmap. Every panel has a **상세** button, CSV/PNG export and
log/linear toggles where meaningful.

**Details window** (`PhysicsWindow`): non-modal, compact (≈ 560×620 px, max 90vh), draggable by
its header, resizable, closable with Esc, stays on the same screen; content = topic title, summary,
section tabs or scroll list with KaTeX display equations (numbered), variable tables (symbol, meaning,
value, unit, code), notes/assumptions callouts, code references, related-topic chips, and a link
"open in Physics tab". Opening another topic replaces the content (with back navigation).
The Physics tab renders all topics as a long document with a TOC, from the same content.

Physics content API (`web/src/content/physics/index.ts`, owned by physics-content): exports
`PHYSICS_TOPICS: Record<TopicId, PhysicsTopic>`, `TOPIC_ORDER: TopicId[]` and re-exports `types.ts`.
Rich-text format is defined in `types.ts`. Every equation must be exact with respect to the code in
`engine/` (constants included) and render with KaTeX `throwOnError: true`.
