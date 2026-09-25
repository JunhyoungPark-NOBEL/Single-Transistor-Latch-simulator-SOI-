# CSVM direct-solver recordings

These test-only fixtures were captured from the unchanged Python run_circuit solver with the full PRESETS.paper.device and the request in web/src/device/fixtures/csvm-payload.json (kept under src/ because the unit test forcing.test.ts imports it and the installer kit excludes web/e2e). The calibrated Device 1 uses VG = -2 V. Default: Iin=1nA, Cdrain=1pF, Time=15ms, BE, reltol=1e-3, dt_max=Time/3000; event detection1e-8A / hysteresis10. The physical waveform and event list are not synthesized by the UI tests.

Variants change one parameter: current-2na (Iin2nA), capacitance-2pf (Cdrain2pF), short (Time4ms), no-oscillation (Iin30nA, Time0.4ms). Derived post-startup frequencies are approximately860.025,1685.994,437.152Hz; the latter two cases must show no frequency.

The local test environment prevents the API process pool from creating its Unix-domain manager socket. Playwright routes these direct-solver results through the existing HTTP job contract to verify UI dispatch, stale states, plot rendering, settings, and derived metrics. These files are not used by the production application as a simulation fallback.
