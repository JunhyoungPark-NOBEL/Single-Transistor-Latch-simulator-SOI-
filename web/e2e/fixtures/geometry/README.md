# Geometry browser recordings

These records are the unmodified output of `server.compute.deterministic.run_branches` for the payload included in each file. Regenerate from the repository root with:

```
python3 web/e2e/fixtures/geometry/generate.py
```

The browser test substitutes only HTTP transport because the execution environment cannot start the server's multiprocessing manager. It verifies payloads match the recorded geometry exactly, including Nbody. Screenshots are marked “실제 계산 결과 · 기록 보기” (computed result, recording).

`reference.json` is the calibrated reference geometry; the other records change one dimension at a time. `numerical-report.json` checks width scaling and native length/thickness trends with the production solver. These numerical checks do not establish agreement with new geometry measurement or TCAD data.
