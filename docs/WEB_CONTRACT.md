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

---

# Phase 2 addendum (owner requests, 2026-09-24)

Owner's new requirements (verbatim intent): (1) logo built around the **biristor** symbol, colours
harmonised with the page background, modern/trendy; (2) NOBEL lab info (Prof. Yang-Kyu Choi's lab;
developed by Junhyoung Park) and KAIST info in a corner; (3) natural Korean and natural English
everywhere; (4) the model is **not yet published** — never say "paper device/paper parameters";
describe devices by technology and geometry, e.g. "FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm ·
EOT 14.1 nm"; technology presets FDSOI (now), PDSOI and Bulk (later, shown as "coming soon");
(5) LTspice-like circuit simulation: devices built in the Device tab are saved to a device library
and "implanted" into user-drawn circuits; free voltage/current sources incl. pulses; user-set time
step/stop time; node voltages and element terminal currents displayed; elements picked from the
drawing; balance accuracy, run time and page weight; (6) stochastic mode shows statistics alongside
results; stochastic devices usable in circuit simulation.

Internal ids stay (`preset: "paper" | "photo" | "custom"`); only user-facing text changes.

## 6. Custom circuits — `kind: "circuit"`, `bench: "custom"` (circuit-custom package)

Request:
```jsonc
{ "bench": "custom", "mode": "deterministic" | "stochastic",
  "netlist": {
    "elements": [
      { "type": "R", "name": "R1", "nodes": ["n1", "n2"], "value": 1000 },             // ohm
      { "type": "C", "name": "C1", "nodes": ["d", "0"], "value": 2e-15 },              // F
      { "type": "V", "name": "V1", "nodes": ["n+", "n-"], "wave": Wave },              // volts
      { "type": "I", "name": "I1", "nodes": ["n+", "n-"], "wave": Wave },              // amperes; flows n+ → (through source) → n-,
                                                                                       //   i.e. it is pushed OUT of n- into the circuit … see §6.1
      { "type": "STL", "name": "X1", "nodes": { "d": "d", "g": "g", "s": "0" },
        "device": { ...device block §1... }, "light_pA": Wave | null },                 // light waveform in pA (null = device light)
      { "type": "CMP", "name": "CMP1", "nodes": { "in": "s", "out": "q" }, "v_ref": 0.1,
        "v_high": 1, "v_low": 0, "hysteresis": 0 }                                     // comparator (§6.3), extension
    ] },
  "tran": { "t_stop_s": 5e-3, "t_start_save_s": 0, "dt_max_s": 1e-5, "dt_min_s": 1e-12,
            "method": "BE" | "TRAP", "reltol": 1e-4 },
  "stochastic": { "seed", "n_runs", "carrier_noise", "ld_carrier_noise", "local_state": {...§1...} },
  "detect": { "i_threshold_A": 1e-8, "hysteresis": 10 },
  "probes": null | ["V(d)", "I(R1)", "I(X1.d)"] }        // null → all node voltages + all element currents
```
Node "0" (aliases "gnd", "GND") is ground; every other string is a node name.
`Wave` = `{ "kind": "dc", "value" }` | `{ "kind": "pulse", "v1", "v2", "td", "tr", "tf", "pw", "per", "ncycles" }`
(SPICE PULSE semantics; ncycles 0 = until t_stop) | `{ "kind": "pwl", "t": [...], "v": [...] }` |
`{ "kind": "sine", "vo", "va", "freq", "td", "theta" }` (sampled to PWL server-side with a bounded point count).

### 6.1 Sign conventions (document them in the UI)
- `V(node)`: node voltage w.r.t. ground.
- `I(R1)`, `I(C1)`: current through the element from its first node to its second node.
- `I(V1)`, `I(I1)`: current through the source from its first (+) node to its second (−) node
  (SPICE convention: a source delivering power has negative `I(V1)`). For `I` sources the `wave`
  value is exactly that current.
- `I(X1.d)`, `I(X1.s)`, `I(X1.g)`: currents INTO the STL terminals (drain, source, gate); gate current is 0 (ideal gate, documented).

### 6.2 Response
Same `CircuitResult` as §4 (generic signals/summary/events/distributions), with:
- signals for every probe: key `V(n)` (unit V, axis voltage), `I(name)` / `I(X1.d)` (unit A, axis current), plus per STL `X1.u`, `X1.r` (V, axis state), `X1.q_b` (C, axis charge);
- `nodes`: list of node names; `elements`: echo of the resolved netlist (names, nodes, values, resolved waves);
- `op`: operating point at t = 0 (node voltages, element currents) and `at(t)` is done client-side by interpolation;
- events per STL cell (`latch_up`/`latch_down`, with `cell` = element name, `t`, `v_d`);
- summary: per STL: number of latch-ups/downs, first latch-up time and V_D, final state; stochastic: P(latched at end), P(≥1 latch-up), mean ± SD of first latch-up time and V_D;
- stochastic `envelopes`: `{ key, t: Arr, mean: Arr, sd: Arr, p05: Arr, p95: Arr }[]` for every probed signal on a common time grid (≤ 1000 points), plus `distributions` of per-run scalar metrics (first latch-up time, V_D at latch-up, value of each probed signal at t_stop).
Limits: ≤ 40 elements, ≤ 8 STL cells, ≤ 30 nodes, waves ≤ 2000 points, t_stop and steps checked by the feasibility estimate (refuse with an explanation instead of running for minutes).
ERC errors (floating node, no ground reference, voltage-source loop, current source in series with nothing, unknown node in STL) → ValueError with a precise message naming the element/node.

### 6.3 Implementation notes (circuit-custom, as implemented in `server/compute/circuit/custom.py`)
Details: `docs/CIRCUIT_SIMULATOR.md` §12. Everything above holds; these are the precisions / deviations:
- **Wave points**: user `pwl` lists ≤ 2000 points; server-generated waves may have up to **20 000** points
  (`pulse` ≤ 5000 periods, `sine` 128 samples per period, ≥ 16, ≤ 1250 periods) — deviation from "waves ≤ 2000
  points", which applies to what the client sends. `tr`/`tf` = 0 and vertical PWL steps (repeated time) get a finite
  edge min(dt_max, 1e-3·t_stop, 0.1·pw, 0.1·per) (warning). `pulse`: `per` ≤ 0 = single pulse, `pw` default t_stop.
  `sine` accepts an optional `phase` (degrees, SPICE PHASE).
- **Limits**: 30 nodes = nodes besides ground. Node names 1–32 chars without spaces/parentheses (`gnd` any case =
  ground); element names 1–32 chars without spaces, dots, commas, brackets, unique case-insensitively.
  Request bodies of bench `custom` may hold up to 40 000 JSON values (other kinds 5000; byte cap unchanged).
- **ERC**: "floating" = no DC path to ground through R, V or an STL drain–source path (C, I sources and STL gates do not
  conduct DC) — SPICE practice, so a capacitor-only island or an undriven gate is an error; nodes with one connection,
  shorted R/C/I and STL terminals sharing a node are warnings.
- **STL**: V_GS comes from the circuit (the device block's `vg` is ignored, warned when different). Optional
  per-element **`local_state`** (§1 block, the library device's setting) is used for that STL in stochastic mode;
  **`stochastic.local_state_override: true`** applies `stochastic.local_state` to every STL instead (frontend: send it
  when the user chose "override").
- **Signals**: `X1.q_b` = Q_B(t) − Q_B(0); `X1.dphi` / `X1.dphi_E` added for cells with local states. `probes`: unknown
  keys → ValueError; `V(0)` allowed (zeros); `[]` → ValueError (use null). Plotted values rounded to 7 significant
  digits (t: 10, envelopes: 6). Stored points: run 0 ≤ 4000 (and ≤ 400 000 values over all signals), runs 1–7 ≤ 1500
  (≤ 100 000 values); envelopes ≤ 1000 points (fewer for very many signals).
- **Events**: `{run, kind, t, cell, v_d, value (= v_d), i_d}`; `v_d` = V_DS of the cell (= V(d) − V(s)); no `v_src`.
- **Summary keys** (aligned with the frontend mock): per STL `X1.n_latch_up`, `X1.n_latch_down`, `X1.t_first_lu` (s),
  `X1.vd_first_lu` (V), `X1.fold_V_LU`, `X1.fold_V_LD`; deterministic `X1.latched_end` (0/1), `X1.final_state`
  ("LRS" | "HRS"); stochastic `X1.p_any_lu`, `X1.p_latched_end` (spread = SD over runs). Global `runs`,
  `steps_per_run` (+ `t_noise_resolved_frac`, `truncated_runs`). Distributions: `X1.t_first_lu`, `X1.vd_first_lu`,
  `X1.n_latch_up`, `end:<signal key>` (every probed signal at t_stop; null for truncated runs).
- **`op`**: flat `{signal key: value}` at t = 0 for every signal (not only the probes). `elements`: resolved echo
  (`value`, `value_label`, `wave` with resolved parameters; STL: `nodes {d,g,s}`, `device {preset, vg_device_V,
  iph_pA, label}`, `light_pA`, `vgs_V`, `vgs_range_V`, `folds`, `latch_window`, `u_fold`, `local_state`,
  `noise_band_V`, `estimated_steps`). Extra keys: `probes`, `trajectory {vd, id, cell}` (first STL), `tran`, `solver`,
  `detect`, `stochastic`, `feasibility`, `regimes`; `sweeps` = [], `distributions`/`envelopes` = [] when deterministic.
- **Initial state and oscillators** (additive; `docs/CIRCUIT_SIMULATOR.md` §13): optional `tran.initial` =
  `"auto"` (default) | `"op"` (DC operating point) | `"zero"` (discharged capacitors, SPICE UIC); "auto" uses the
  operating point unless a current-biased cell would sit on its negative-resistance branch (or none is found), then
  "zero"; echoed as `tran.initial` / `tran.initial_used`. Per STL echo `oscillator` ({predicted, period_qs_s,
  latch_ups_expected, c_eff_F, i_norton_A, r_ext_ohm} for high-impedance drives, else null). Per STL with ≥ 2
  latch-ups in a run: summary `X1.period` (s), `X1.f_osc` (Hz), `X1.isi_cv`, `X1.vd_lu_mean`, `X1.vd_ld_mean` (V);
  stochastic also `X1.period_cv_runs` and the distribution `X1.isi` (s, all intervals).
- **Comparator `CMP`** (additive; `docs/CIRCUIT_SIMULATOR.md` §12.9): nodes `{in, out}` (+ optional `inm`, default
  ground), `v_ref` (V), `v_high` (1 V), `v_low` (0 V), `hysteresis` (0), `width` (1 mV smoothing). Ideal inputs; the
  output is a behavioural voltage source out → ground (it can drive other elements; ERC: not tied to another
  source or ground). Signals `I(CMP1)` (output current, SPICE sign) and `CMP1.bit` (unit "1", axis logic); events
  `cmp_rise` / `cmp_fall` (`cell` = comparator name). Result key `comparators`: per comparator `{name, nodes, v_ref,
  v_high, v_low, hysteresis, width, window_source, t_windows, bits (runs × pulse periods, 0/1/null), p_fire_window,
  p_fire_window_err, p_fire, lag1, n_bits, p_fire_run}` — windows = periods of the periodic pulse source with the
  most periods (fired = output high within the period); summary `CMP1.p_fire`, `CMP1.lag1`, `CMP1.n_bits`,
  `CMP1.n_rise`, `CMP1.duty`; stochastic distribution `CMP1.p_fire_run`. Limit 8 comparators.

## 7. Device library (frontend, device-library package)

A device = `{ id, name, technology: "FDSOI" | "PDSOI" | "Bulk", geometry: { Lg_nm, W_nm, Tsi_nm, EOT_nm },
calibration_label, device: <device block §1>, created, notes }`. Built-in (read-only): FDSOI reference
calibration (dark, V_G −2 V) and FDSOI illumination calibration (1200 V/s) — both L_g 500 nm, W 200 nm,
T_Si 50 nm, EOT 14.1 nm (geometry is fixed by the model; editable only when a future model supports it).
User devices: "Save as device" in the Device tab stores the current device block (+ stochastic local-state
settings used when the device is simulated stochastically in a circuit) in localStorage; rename, delete,
duplicate, export/import JSON. The circuit editor's STL palette lists library devices; each STL instance
stores a snapshot of the device block (so later library edits don't silently change saved circuits; offer
"update from library"). Technology selector shows FDSOI active, PDSOI/Bulk disabled "coming soon".

## 8. Branding and wording (brand package)
Logo: SVG built around a biristor-style two-terminal symbol (stylised: terminals + body with an
S-shaped/hysteresis motif), gradient using the app's accent tokens so it matches both themes; used in
the header, favicon and About. Credits corner: small unobtrusive block (e.g. bottom-left of the sidebar
footer or a corner chip) — "NOBEL Lab · Prof. Yang-Kyu Choi · School of Electrical Engineering, KAIST ·
Developed by Junhyoung Park" / "KAIST 전기및전자공학부 · NOBEL 연구실 (지도교수 최양규) · 개발 박준형",
opening an About popover. No email addresses. Wording: replace every user-facing "논문/paper" with
device/record descriptions; keep figure references descriptive ("V_G dependence of V_LU", not "Fig. 3(b)").

## 9. Statistics module (stats package) — shared by device MC and circuit runs
`web/src/stats/describe.ts`: `describe(values: (number|null|undefined)[]): Describe` with
`{ n, n_total, censored, mean, sd, se, cv, median, q1, q3, iqr, p05, p95, min, max, skewness, kurtosis_excess, lag1 }`
(null-safe; sd with ddof = 1; lag1 over consecutive finite pairs), `ks2(a, b)` (two-sample KS D and
asymptotic p-value), `histogram(values, bins?)`.
`web/src/stats/StatsTable.tsx`: `<StatsTable rows={{ key, label, unit, values, scale? }[]} measured?={...} />`
compact, copyable (CSV), bilingual; used in the Device tab (stochastic) and by the circuit editor.
