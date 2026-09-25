#!/usr/bin/env python3
"""Warm the numba JIT caches and the engine's run-time caches (used at Docker build time and after install).

    python scripts/warmup.py            # compile + run each deterministic kind once, FPT node, MC sweeps, circuit
    python scripts/warmup.py --quick    # compile only (import + one branch)
    python scripts/warmup.py --validate # additionally run the fast validation level and print the checks

numba caches land next to the engine sources (engine/**/__pycache__, or NUMBA_CACHE_DIR if set);
FPT hazard nodes land in engine/photo_extension/photo_nodes/.  Both directories must be writable at run time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def step(name: str, fn):
    t = time.perf_counter()
    try:
        out = fn()
        print(f"[warmup] {name:<44s} {time.perf_counter() - t:7.1f} s", flush=True)
        return out
    except Exception as exc:  # noqa: BLE001 - warm-up must not break the build for optional parts
        print(f"[warmup] {name:<44s} FAILED: {type(exc).__name__}: {exc}", flush=True)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    step("import engine (numba cache load/compile)", lambda: __import__("server.engine_bridge"))
    from server.compute import deterministic as D

    res = step("branches (paper, V_G = -2 V)", lambda: D.run_branches({"device": {"preset": "paper"}}))
    if res is None or res["folds"]["V_LU"] is None or abs(res["folds"]["V_LU"] - 3.7037) > 1e-3:
        print("[warmup] ERROR: fold check failed", res and res["folds"], flush=True)
        return 1
    if args.quick:
        print(f"[warmup] done in {time.perf_counter() - t0:.1f} s")
        return 0

    step("branches, changed geometry (geometry kernels)",
         lambda: D.run_branches({"device": {"preset": "paper", "geometry": {"Lg_nm": 400.0}}}))
    step("charge_balance (V_D = 3.2 V)", lambda: D.run_charge_balance({"device": {"preset": "paper"}, "vd": 3.2}))
    step("vg_curve (5 points)", lambda: D.run_vg_curve({"device": {"preset": "paper"}, "n": 5, "refine": False}))

    def gate_mean():
        import gate_mean as G  # paper model original, used by the validation identity check
        from server.engine_bridge import MODEL, S
        gm = G.FastModel(MODEL.na, G.load_transport_table()[0])
        return gm.classify(S.params(-2.0, 0.0)[:13].copy(), G.state_grid(201))

    step("gate_mean (validation identity) compile", gate_mean)

    from server.engine_bridge import A
    step("FPT node V_G = -2 V dark (photo_nodes cache)", lambda: A.hazard(-2.0, rate=0.4))
    step("dynamic MC, 10 sweeps", lambda: A.sweeps(n=10))
    for mod in ("server.compute.stochastic", "server.compute.circuit", "server.compute.validation"):
        step(f"import {mod}", lambda mod=mod: __import__(mod))

    # The circuit kernels (server/compute/circuit/{element,mna}.py, @njit(cache=True)) compile on first use
    # (~40 s cold): compile them here so the first circuit request of a fresh image is fast.
    def circuit(mode: str):
        from server.compute.circuit import run_circuit
        out = run_circuit({"bench": "load_line", "mode": mode, "device": {"preset": "photo"},
                           "bench_params": {"n_cycles": 1}, "stochastic": {"n_runs": 1}})
        return out["solver_stats"]

    step("circuit load_line, deterministic (numba compile)", lambda: circuit("deterministic"))
    step("circuit load_line, stochastic (numba compile)", lambda: circuit("stochastic"))

    # [circuit-custom] user-drawn circuits (bench "custom"): small RC + STL netlist, deterministic and stochastic
    # (same kernels; also fills the branch-profile / fold caches of the photo condition used by the defaults)
    def circuit_custom(mode: str):
        from server.compute.circuit import run_circuit
        T = 2 * 4.0 / 1200.0
        els = [{"type": "V", "name": "V1", "nodes": ["src", "0"], "wave": {"kind": "pwl", "t": [0, T / 2, T], "v": [0, 4.0, 0]}},
               {"type": "R", "name": "R1", "nodes": ["src", "d"], "value": 1e3},
               {"type": "C", "name": "C1", "nodes": ["d", "0"], "value": 2e-15},
               {"type": "V", "name": "VG", "nodes": ["g", "0"], "wave": {"kind": "dc", "value": -1.8}},
               {"type": "STL", "name": "X1", "nodes": {"d": "d", "g": "g", "s": "0"},
                "device": {"preset": "paper", "vg": -1.8, "light": {"mode": "iph", "iph_pA": 2.63}}, "light_pA": None}]
        out = run_circuit({"bench": "custom", "mode": mode, "netlist": {"elements": els},
                           "tran": {"t_stop_s": T, "dt_max_s": T / 2000}, "stochastic": {"n_runs": 2, "seed": 1}})
        return out["solver_stats"]

    step("circuit custom RC + STL, deterministic", lambda: circuit_custom("deterministic"))
    step("circuit custom RC + STL, stochastic", lambda: circuit_custom("stochastic"))

    if args.validate:
        from server.compute.validation import run_validation
        v = step("validation (fast)", lambda: run_validation({"level": "fast"}))
        if v:
            for c in v["checks"]:
                print(f"  {str(c['pass']):5s} {c['id']:<28s} {c['computed']}")
            if v["summary"]["failed"]:
                return 1
    print(f"[warmup] done in {time.perf_counter() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
