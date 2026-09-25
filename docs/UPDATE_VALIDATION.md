# Update verification — 2026-09-25

## Latest: geometry, surface recombination and back-gate coupling

- Frontend: **33 files / 514 tests passed**. TypeScript and production build passed.
- Backend: **163 passed, 7 deselected** across geometry, geometry-noise guards, deterministic and circuit regression suites. HTTP-worker tests and slow tests remain excluded for the existing environment limitation below.
- **14 targeted browser tests passed** (`geometry.spec.ts` + `device-csvm.spec.ts`). The six geometry controls are topmost. Save/load/import, circuit snapshots, fixed-geometry LTspice/Verilog-A provenance, legacy reference migration and refusal of offline resized results are covered. Desktop 1440px and mobile 390px previews show actual resized-model curves after Plotly trace readiness; the Geometry guide opens the responsive model document.
- Exact original reference folds are preserved: VLU=3.703688985 V / VLD=2.597866939 V. Across L=200/300/400/500/700nm, both folds increase with L. Across Tsi=5/10/20/30/50/70nm, both decrease with Tsi; no explicit fold correction is applied. Shorter lengths can lose the latch or the assumed lateral neutral base, so this is not a global monotonic guarantee.
- W doubled at dark bias gives the same branch voltages and twice the currents/charges. Explicit Iph and external Iin/Cdrain are not silently scaled.
- Final reference-anchored body capacitance at Tbox=70/140/280nm: 2.627056 / 2.449031 / 2.345016 ×10^-16 F. At VBG=1V, VG=-0.8V, the channel coupling contribution changes with Tbox; measured channel currents at u=.6V,r=2.7V are 6.0597/1.3775/0.56482nA. VBG=0 yields no invented DC fold shift from Tbox.
- Actual geometry CSVM run: L400nm, VG=-2V, VBG=0, Iin=1nA, Cdrain=1pF, 8ms completed with 5976 accepted steps / 3062 saved samples, four LU and four LD events. LU spacing gives about 899.230Hz; event voltages are about 3.6510/2.54527V. All sampled terminal currents and body charge are finite. Saved-output KCL residual is 30pA (about 2.47ppm of peak current), limited by seven-significant-digit output rounding; this is not the unrounded Newton residual.
- The surface-dominated interpretation of the existing effective junction lifetime is an explicit assumption, not an extracted surface recombination fraction. Tbox140nm is nominal. No arbitrary quantum knee, BJT barrier shift, or fold-voltage offset was introduced. Domain errors and nonreference noise requests are rejected explicitly.
- These are dimensional/numerical consistency checks, **not validation against multi-geometry measurements or TCAD**. See `../GEOMETRY_MODEL_KO.md`, `web/e2e/fixtures/geometry/numerical-report.json`, and `csvm-check.json` (the latter two relative to the repository root).
- All **66 original engine files** match the input source ZIP byte for byte. Numerical extensions live in server-owned code. No deployment was performed.

Final server command:

```bash
python3 -m pytest server/tests/test_geometry_model.py server/tests/test_geometry_stochastic.py server/tests/test_circuit_geometry.py server/tests/test_deterministic.py server/tests/test_circuit.py server/tests/test_circuit_custom.py server/tests/test_circuit_basic.py -q -m 'not slow' -k 'not api_custom_circuit'
```

## Follow-up: supplied symbol, CSVM, export selector

- Final frontend suite: **31 files / 494 tests passed**; TypeScript and production build passed.
- **19 targeted browser tests passed**: 9 workspace/library/export cases, 5 migrated
  device/statistics/schematic cases, and 5 new CSVM cases. Desktop/mobile previews were
  visually inspected. Modal opacity and stacking were checked after entry animations.
- The CSVM frontend request matches the request accepted by the actual numerical solver.
  Tests use authentic direct-solver recordings through the HTTP job contract because the
  worker-pool limitation below still applies. Production does not use these fixtures.
- Device 1 at VG = −2 V, Iin = 1 nA, Cdrain = 1 pF, 15 ms produced
  Vtop ≈ 3.7422 V, Vbottom ≈ 2.5857 V, f ≈ 860.025 Hz after startup exclusion.
  Iin = 2 nA gave ≈ 1685.994 Hz; Cdrain = 2 pF gave ≈ 437.152 Hz.
  A 4 ms run and the tested nonoscillating bias do not display a fabricated frequency.
  Insufficient waveform resolution also suppresses unsupported metrics.
- Verilog-A source interpolation is parsed independently and compared with actual engine
  currents. The selector, offline protection, canceled generation and input validation are
  tested. Commercial simulator executables were not run. Sentaurus export is explicitly
  unavailable; see `../SIMULATOR_EXPORTS_KO.md`.
- All **66 original engine files** still match the supplied source archive byte for byte.

## Passed

The following table records the initial redesign verification; the frontend suite above
supersedes its earlier test count. Numerical engine/backend code was unchanged by the follow-up.

| Check | Result |
|---|---|
| `npm run typecheck` | Pass |
| `npm test` | 28 files, 464 tests passed |
| `npm run build` | Pass; compiled frontend included in delivery |
| `python3 engine/stl_api.py` | Reference folds 3.7037/2.5979 V; FPT 3.6442 V / 8.03 mV; dynamic MC 3.6344 V / 119.0 mV |
| Relevant server numerical tests | 127 passed, 7 deselected |
| `ux-device.spec.ts` + `reference-sweep.spec.ts` | 12 passed |
| `ux-shell.spec.ts` + `ux-guide.spec.ts` | 14 passed |
| `workspace-redesign.spec.ts` | 6 passed |
| Original `engine/` comparison | All original files byte-identical |

Server test command:

```bash
python3 -m pytest server/tests/test_deterministic.py server/tests/test_circuit.py server/tests/test_circuit_custom.py server/tests/test_circuit_basic.py -q -m 'not slow' -k 'not api_custom_circuit'
```

Front-end-generated example netlists were evaluated by the actual backend numerical code:

- CMOS inverter: output approximately 1.08 nV–1.8 V, low output for high input.
- Diode clipper: output −1.5–0.65057 V.
- BJT switch: output 0.09923–3.3 V.
- Output-node KCL residuals ≤1e−11 A for these checks.

LTspice export tests parse generated branch tables independently and compare roughly 4,000
real-engine voltage-sweep points; relative errors <1e−9 away from switching discontinuities.
This checks table fidelity, metadata, threshold mapping and state selection, not LTspice execution.

## Important limits

- This environment did not run the production HTTP worker pool: SyncManager's Unix socket failed
  with `PermissionError: Operation not permitted`. Direct numerical calculations and browser
  tests succeeded. Tests of the deployed HTTP pool, full API suite and slow tests were not run.
- LTspice and commercial Verilog-A executable integration were not run. Export is a fixed-bias quasi-static model and
  does not preserve physical body-charge dynamics or stochastic behavior.
- New generic MOS/diode/BJT models are educational 300 K memoryless models; see
  `BASIC_CIRCUIT_DEVICES.md` for unsupported effects.
- The full historical end-to-end suite was not claimed as passing. Updated suites above cover
  the changed interfaces. Five preset/library tests were migrated in the follow-up; some
  remaining legacy assertions still target removed bench UI.
- No deployment or git push was performed. Cloudflare prevented inspection of the existing
  Claude artifact in the cloud browser. Full source and original handoff were available locally.
- Locked-artifact recording/rebuild was not run. Existing snapshots need regeneration for the
  new example circuits and reference workflow. The included `web/dist` is the ordinary server build.

## Reference benchmark interpretation

The fixed reference uses authentic measured forward/reverse median currents and actual
calibrated engine sweeps at VG=−2 V, dark, 0–4 V. The two sets of measured sweeps are not
paired cycles. The current comparison includes the measurement floor: full-range log RMSE
is about 5.49 decades because the model predicts much smaller low currents, while linear
range-normalized RMSE is approximately 5.083% up and 3.451% down. These metrics are
reported as errors, not an artificial overall accuracy percentage or a self-test pass rate.
The expandable method explains the formulas and valid points.

## Preview images

`PREVIEW/device.png` uses an authentic recorded reference branch result, labeled saved results.
`PREVIEW/circuit.png` shows an editable circuit; its demo-mode badge does not claim live
simulation. `PREVIEW/reference.png` shows the genuine fixed benchmark. Mobile/theme
behavior was also checked.

`PREVIEW/csvm.png` and `PREVIEW/csvm-mobile.png` show the actual direct-solver recording
with a visible recording label. The transport was routed for the screenshot; these images
do not claim a deployed live server. Export selector previews show its desktop/mobile layout.

The latest `PREVIEW/device.png` and `PREVIEW/mobile.png` now show the actual L400nm
geometry result and topmost controls. `PREVIEW/geometry-mobile.png` duplicates the dedicated
mobile control view for convenient review. Earlier reference/circuit/export views are retained.
