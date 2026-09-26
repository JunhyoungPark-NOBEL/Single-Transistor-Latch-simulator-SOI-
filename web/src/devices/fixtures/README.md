# Simulator export reference fixture

`export-reference.json` is a live `server.compute.deterministic.run_branches` result
generated on 2026-09-25 from the repository's `paper` preset, grid 601, 0–4 V,
0.4 V/s, 2 mV display steps. It is not mock data. Unused per-component arrays and
runtime are omitted; current/voltage arrays, parameter vector and folds are intact.

The export test independently parses the generated LTspice `exp(table(...))`
expressions and checks them against every non-discontinuous point of the engine's
forward/reverse ID–VD sweep. It checks switch thresholds and parameter provenance.
The Verilog-A export tests independently interpret the emitted analog current functions and compare the same full up/down sweep.

These tests do not execute LTspice or a commercial Verilog-A simulator, and do not validate physical transient behavior.

Regenerate after an intentional change to the reference physics/calibration by
calling `run_branches` with `params.resolve_device({"preset": "paper"})` and
`params.PRESETS["paper"]["sweep"]`. Do not update this fixture merely to make a
failing test pass.
