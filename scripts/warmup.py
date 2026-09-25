#!/usr/bin/env python3
"""Warm the numba JIT caches and the engine's run-time caches (used at Docker build time and after install).

    python scripts/warmup.py            # compile + run each deterministic kind once, FPT node, MC sweeps, circuit
    python scripts/warmup.py --quick    # compile only (import + one branch)
    python scripts/warmup.py --validate # additionally run the fast validation level and print the checks

It is also the build gate of the Docker image (and so of the lab-server auto-update): besides the reference folds it
checks the geometry kernels (L = 400 nm folds), a back-gate request, a CSVM transient (current source + C + STL) and
a MOSFET/diode/BJT circuit, and exits with 1 when a result is wrong, so a broken build never replaces a working one.

numba caches land in server/__pycache__/numba/<stamp of the sources> (server/__init__.py; NUMBA_CACHE_DIR if set);
FPT hazard nodes land in engine/photo_extension/photo_nodes/.  Both directories must be writable at run time.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Build-gate expectations (docs/GEOMETRY_MODEL_KO.md): the reference device and the L = 400 nm geometry extension
# (area-only n+ source/drain injection, owner decision D8), V_G = -2 V, dark
REF_V_LU = 3.7037
GEOMETRY_L400 = (3.5821, 2.2423)


def step(name: str, fn):
    t = time.perf_counter()
    try:
        out = fn()
        print(f"[warmup] {name:<44s} {time.perf_counter() - t:7.1f} s", flush=True)
        return out
    except Exception as exc:  # noqa: BLE001 - warm-up must not break the build for optional parts
        print(f"[warmup] {name:<44s} FAILED: {type(exc).__name__}: {exc}", flush=True)
        return None


def _signals(out) -> tuple:
    """(t, {probe: values}) of the first run of a circuit result (lists or numpy arrays), or (None, {})."""
    if not out or not out.get("runs"):
        return None, {}
    run = out["runs"][0]
    return list(run["t"]), {s["key"]: list(s["values"]) for s in run["signals"]}


def _last(out, key: str):
    _, sig = _signals(out)
    v = sig.get(key)
    return float(v[-1]) if v else None


def _at(out, key: str, t: float):
    ts, sig = _signals(out)
    v = sig.get(key)
    if not v or not ts:
        return None
    i = min(range(len(ts)), key=lambda j: abs(ts[j] - t))
    return float(v[i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    step("import engine (numba cache load/compile)", lambda: __import__("server.engine_bridge"))
    from server.compute import deterministic as D

    res = step("branches (paper, V_G = -2 V)", lambda: D.run_branches({"device": {"preset": "paper"}}))
    if res is None or res["folds"]["V_LU"] is None or abs(res["folds"]["V_LU"] - REF_V_LU) > 1e-3:
        print("[warmup] ERROR: fold check failed", res and res["folds"], flush=True)
        return 1
    if args.quick:
        print(f"[warmup] done in {time.perf_counter() - t0:.1f} s")
        return 0

    geo = step("branches, L = 400 nm (geometry kernels)",
               lambda: D.run_branches({"device": {"preset": "paper", "geometry": {"Lg_nm": 400.0}}}))
    f = (geo or {}).get("folds") or {}
    vlu, vld = f.get("V_LU"), f.get("V_LD")
    if (not isinstance(vlu, float) or not isinstance(vld, float) or not math.isfinite(vlu) or not math.isfinite(vld)
            or abs(vlu - GEOMETRY_L400[0]) > 1e-3 or abs(vld - GEOMETRY_L400[1]) > 1e-3 or abs(vlu - REF_V_LU) < 1e-3):
        print(f"[warmup] ERROR: geometry fold check failed (L = 400 nm: expected V_LU/V_LD = "
              f"{GEOMETRY_L400[0]}/{GEOMETRY_L400[1]} V, got {vlu}/{vld})", flush=True)
        return 1
    bg = step("branches, V_BG = 0.5 V at V_G = -0.8 V (back gate)",
              lambda: D.run_branches({"device": {"preset": "paper", "vg": -0.8, "vbg": 0.5}}))
    if bg is None or bg.get("vbg") != 0.5 or len(bg.get("p") or ()) != 33 or not all(
            v is None or math.isfinite(v) for v in (bg["folds"].get("V_LU"), bg["folds"].get("V_LD"))):
        print("[warmup] ERROR: back-gate request failed", bg and bg.get("folds"), flush=True)
        return 1
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

    # CSVM (the Device tab's current-forcing mode, web/src/device/forcing.ts csvmPayload): 1 nA into 1 pF on the drain
    # of the L = 400 nm cell for 2 ms charges the drain linearly to I t / C = 2 V, below V_LU (no latch-up yet)
    def csvm():
        from server.compute.circuit import run_circuit
        T = 2e-3
        els = [{"type": "I", "name": "Iin", "nodes": ["0", "drain"], "wave": {"kind": "dc", "value": 1e-9}},
               {"type": "C", "name": "Cdrain", "nodes": ["drain", "0"], "value": 1e-12},
               {"type": "V", "name": "VG", "nodes": ["gate", "0"], "wave": {"kind": "dc", "value": -2.0}},
               {"type": "STL", "name": "X1", "nodes": {"d": "drain", "g": "gate", "s": "0"},
                "device": {"preset": "paper", "geometry": {"Lg_nm": 400.0}}, "light_pA": None}]
        return run_circuit({"bench": "custom", "mode": "deterministic", "netlist": {"elements": els},
                            "tran": {"t_stop_s": T, "t_start_save_s": 0, "dt_max_s": T / 3000, "dt_min_s": 1e-15,
                                     "method": "BE", "reltol": 1e-3},
                            "detect": {"i_threshold_A": 1e-8, "hysteresis": 10}, "probes": ["V(drain)", "I(X1.d)"]})

    out = step("CSVM 2 ms (I source + C + STL, L = 400 nm)", csvm)
    v_end = _last(out, "V(drain)")
    if out is None or v_end is None or abs(v_end - 2.0) > 0.01 or out["events"]:
        print(f"[warmup] ERROR: CSVM check failed (V(drain) at 2 ms = {v_end}, expected 2.0 V without events)", flush=True)
        return 1

    # MOSFET / diode / BJT (server/compute/circuit/basic.py): CMOS inverter, diode clamp and BJT switch on one input
    def basic_devices():
        from server.compute.circuit import run_circuit
        els = [{"type": "V", "name": "VDD", "nodes": ["vdd", "0"], "wave": {"kind": "dc", "value": 1.8}},
               {"type": "V", "name": "Vin", "nodes": ["in", "0"], "wave": {"kind": "pulse", "v1": 0, "v2": 1.8, "td": 0,
                "tr": 20e-6, "tf": 20e-6, "pw": 0.4e-3, "per": 1e-3, "ncycles": 1}},
               {"type": "MOS", "name": "M1", "nodes": {"d": "out", "g": "in", "s": "0"}, "model": {"polarity": "nmos"}},
               {"type": "MOS", "name": "M2", "nodes": {"d": "out", "g": "in", "s": "vdd"},
                "model": {"polarity": "pmos", "W_um": 20, "k_uA_V2": 50}},
               {"type": "C", "name": "CL", "nodes": ["out", "0"], "value": 10e-12},
               {"type": "R", "name": "R1", "nodes": ["in", "x"], "value": 1e3},
               {"type": "D", "name": "D1", "nodes": {"a": "x", "k": "0"}},
               {"type": "R", "name": "RC", "nodes": ["vdd", "c"], "value": 1e3},
               {"type": "R", "name": "RB", "nodes": ["in", "b"], "value": 10e3},
               {"type": "BJT", "name": "Q1", "nodes": {"c": "c", "b": "b", "e": "0"}, "model": {"polarity": "npn"}}]
        return run_circuit({"bench": "custom", "mode": "deterministic", "netlist": {"elements": els},
                            "tran": {"t_stop_s": 1e-3, "dt_max_s": 2e-6}})

    out = step("MOSFET / diode / BJT circuit (CMOS inverter)", basic_devices)
    hi = {k: _at(out, k, 0.2e-3) for k in ("V(out)", "V(x)", "V(c)")}      # input high
    lo = {k: _at(out, k, 0.9e-3) for k in ("V(out)", "V(c)")}              # input low again
    if (out is None or None in hi.values() or None in lo.values() or not hi["V(out)"] < 0.1 or not lo["V(out)"] > 1.7
            or not 0.5 < hi["V(x)"] < 0.8 or not hi["V(c)"] < 0.3 or not lo["V(c)"] > 1.7):
        print(f"[warmup] ERROR: MOSFET/diode/BJT check failed (input high: {hi}, input low: {lo})", flush=True)
        return 1

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
