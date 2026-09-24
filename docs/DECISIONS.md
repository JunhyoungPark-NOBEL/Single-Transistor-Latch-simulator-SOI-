# Implementation decisions (append-only)

- [foundation] Engine copied verbatim from the handoff zip into `engine/`; runtime caches are gitignored.
- [foundation] `server/params.py` reads every calibrated number from the engine JSON files (single source of truth); `build_p()` reproduces `setup_photo.BASE` exactly for the paper preset and `S.params(...)` for the photo preset (checked to 1e-27).
- [foundation] Photo preset applies the body-coupling option γ = 0.2794 (p[15]) as `mc_cycles.py` does; the high-V_D seed (p[17], p[18]) stays an alternative option in the UI (open problem).
