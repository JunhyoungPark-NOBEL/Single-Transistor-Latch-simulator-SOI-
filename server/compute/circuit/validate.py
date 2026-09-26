"""Validation of the circuit simulator against the engine (numbers quoted in docs/CIRCUIT_SIMULATOR.md).

    python -m server.compute.circuit.validate [--quick] [--out results.json]

V1  deterministic load line, paper device V_G = -2 V dark, 0.4 V/s, R_s = 100 ohm, C_d = 1 fF:
    V_LU / V_LD at the drain node vs the steady-state folds (MODEL.classify) and the trajectory
    vs the quasi-static HRS/LRS branches; ramp-rate lag at 40 and 1200 V/s.
V2  stochastic load line (carrier noise only), V_G = -1.8 V, I_PH = 2.63 pA (paper calibration),
    1200 V/s and 120 V/s: V_LU distribution vs the compound-jump FPT (A.hazard / photo_fpt.quantiles).
V3  fixed-bias escape at V_D = 3.20 V: mean first-passage time from the event-level circuit
    simulation vs the exact compound backward equation (compound_fpt.make_lattice/backward).
V4  step-size convergence of V2 at 1200 V/s (tau_frac 0.02, 50 events/step).
V5  latch-down with the LRS carrier noise resolved (stochastic.ld_carrier_noise = true) vs the default
    drift-only LRS, 1200 V/s, same device.
V6  engine VALIDATION.md FPT node: paper device V_G = -2 V dark, centre states, 0.4 V/s, carrier noise
    only (expected mean V_LU ~ 3.644 V, SD ~ 8 mV) — event-level circuit simulation of the slow ramp.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from server.engine_bridge import A, S, ct

from . import run_circuit
from .netlist import Netlist
from .sim import SolverConfig, simulate

PHOTO_DEV = {"preset": "paper", "vg": -1.8, "light": {"mode": "iph", "iph_pA": 2.63}}


def v1_deterministic(rates=(0.4, 40.0, 1200.0)) -> dict:
    out = {}
    br = A.branches(-2.0)
    for rate in rates:
        t = time.perf_counter()
        res = run_circuit(dict(bench="load_line", mode="deterministic", device={"preset": "paper"},
                               bench_params={"R_s_ohm": 100.0, "C_d_F": 1e-15, "rate_V_per_s": rate}))
        s = {x["key"]: x["value"] for x in res["summary"]}
        out[f"{rate:g}"] = dict(V_LU=s["V_LU"], V_LD=s["V_LD"], fold_V_LU=br["V_LU"], fold_V_LD=br["V_LD"],
                               lag_LU_mV=1e3 * (s["V_LU"] - br["V_LU"]), lag_LD_mV=1e3 * (s["V_LD"] - br["V_LD"]),
                               hrs_dev_dec=s.get("hrs_branch_dev"), lrs_dev_dec=s.get("lrs_branch_dev"),
                               steps=res["solver_stats"]["steps"], runtime_s=time.perf_counter() - t)
    return out


def fpt_reference(rate: float) -> dict:
    import photo_fpt as F
    rec, q = A.hazard(-1.8, 2.63e-12, rate=rate)
    fold = rec["fold_V"]
    below = q < fold - 1e-9
    return dict(mean=float(q.mean()), sd_mV=float(q.std() * 1e3), fold=fold, atom=float(1 - below.mean()),
                below_mean=float(q[below].mean()) if below.any() else None,
                below_sd_mV=float(q[below].std() * 1e3) if below.any() else None,
                quantiles={f"{p:.2f}": float(np.quantile(q, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}, _q=q)


def v2_stochastic(rate: float, n_runs: int, solver: dict | None = None, seed: int = 2026092920) -> dict:
    t = time.perf_counter()
    res = run_circuit(dict(bench="load_line", mode="stochastic", device=PHOTO_DEV,
                           bench_params={"rate_V_per_s": rate, "v_max_V": 5.0, "R_s_ohm": 100.0, "C_d_F": 1e-15},
                           stochastic={"n_runs": n_runs, "seed": seed}, solver=solver or {}))
    lu = np.asarray([d for d in res["distributions"] if d["key"] == "V_LU"][0]["values"], float)
    lu = lu[np.isfinite(lu)]
    ref = fpt_reference(rate)
    q = ref.pop("_q")
    fold = ref["fold"]
    b = lu < fold
    return dict(rate=rate, n=int(len(lu)), mean=float(lu.mean()), sd_mV=float(lu.std(ddof=1) * 1e3),
                se_mean_mV=float(lu.std(ddof=1) / np.sqrt(len(lu)) * 1e3), frac_beyond_fold=float(1 - b.mean()),
                below_mean=float(lu[b].mean()) if b.any() else None,
                below_sd_mV=float(lu[b].std(ddof=1) * 1e3) if b.sum() > 1 else None,
                quantiles={f"{p:.2f}": float(np.quantile(lu, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)},
                fpt=ref, ks_below_fold=_ks_below(lu, q, fold), steps_per_run=res["solver_stats"]["steps"] / max(n_runs, 1),
                runtime_s=time.perf_counter() - t, seconds_per_run=(time.perf_counter() - t) / n_runs)


def _ks_below(x, q, fold):
    """Max |ECDF difference| over v < fold (the FPT has an atom at the fold)."""
    grid = np.linspace(min(x.min(), q.min()), fold - 1e-6, 400)
    fx = np.searchsorted(np.sort(x), grid, side="right") / len(x)
    fq = np.searchsorted(np.sort(q), grid, side="right") / len(q)
    return float(np.max(np.abs(fx - fq)))


def v3_fixed_bias(vd: float = 3.20, n_runs: int = 100, hold: float = 3e-3) -> dict:
    p = S.params(-1.8, 2.63e-12)
    from .stochastic import classify_checked
    b, i, j, fold = classify_checked(p, 601)[0]
    uf = b[i, 17]
    ug = np.unique(np.round(np.r_[np.linspace(.1, .9, 181), np.linspace(uf - .065, uf + .065, 61)], 12))
    rows = np.array([S.state(u, vd, p) for u in ug])
    xx, ix, r, bt, ii, death = ct.cf.make_lattice(rows, "LU", .06)
    tm, _, _ = ct.cf.backward(r, bt, ii, death, "LU")
    exact = float(tm[ix])
    net = Netlist()
    t_up = 20e-6
    T = t_up + hold
    w = net.add_V("Vsrc", "src", "0", [0, t_up, T], [0, vd, vd], "step")
    net.add_V("VG", "g", "0", [0.0], [-1.8], "DC")
    net.add_R("Rs", "src", "d", 100.0)
    net.add_C("Cd", "d", "0", 1e-15)
    net.add_STL("X1", "d", "g", "0", np.asarray(p, float))
    net.t_end = T
    c = net.compile()
    cfg = SolverConfig(stochastic=True, carrier=True, dt_max=T / 2000, dt_min=1e-15, dt_rec=T / 300, dv_rec=0.2,
                       dlni_rec=3.0)
    t0 = time.perf_counter()
    te = []
    for k in range(n_runs):
        out = simulate(c, cfg, c["P"], T, w, 5000 + k)
        e = out.events
        lu = e[e[:, 0] == 1, 2]
        te.append(lu[0] - t_up if len(lu) else np.inf)
    te = np.array(te)
    fin = np.isfinite(te)
    tc = np.where(fin, np.maximum(te, 0), hold)
    mfpt = float(tc.sum() / max(fin.sum(), 1))
    return dict(vd=vd, n=n_runs, escaped=int(fin.sum()), exact_mfpt_s=exact, circuit_mfpt_s=mfpt,
                ratio=mfpt / exact, ratio_se=1 / np.sqrt(max(fin.sum(), 1)), runtime_s=time.perf_counter() - t0)


def v5_ld_noise(n_runs: int = 20) -> dict:
    out = {}
    for flag in (False, True):
        t = time.perf_counter()
        res = run_circuit(dict(bench="load_line", mode="stochastic", device=PHOTO_DEV,
                               bench_params={"rate_V_per_s": 1200.0, "v_max_V": 5.0, "R_s_ohm": 100.0, "C_d_F": 1e-15},
                               stochastic={"n_runs": n_runs, "ld_carrier_noise": flag}))
        ld = np.asarray([d for d in res["distributions"] if d["key"] == "V_LD"][0]["values"], float)
        ld = ld[np.isfinite(ld)]
        out["ld_noise" if flag else "default"] = dict(n=int(len(ld)), mean=float(ld.mean()),
                                                     sd_mV=float(ld.std(ddof=1) * 1e3) if len(ld) > 1 else None,
                                                     fold_V_LD=res["folds"]["V_LD"],
                                                     seconds_per_run=(time.perf_counter() - t) / n_runs,
                                                     steps_per_run=res["solver_stats"]["steps"] / n_runs)
    return out


def v6_paper_slow(n_runs: int = 40) -> dict:
    t = time.perf_counter()
    res = run_circuit(dict(bench="load_line", mode="stochastic", device={"preset": "paper"},
                           bench_params={"rate_V_per_s": 0.4, "v_max_V": 4.0, "R_s_ohm": 100.0, "C_d_F": 1e-15},
                           stochastic={"n_runs": n_runs}, solver={"max_steps": 2000000}))
    lu = np.asarray([d for d in res["distributions"] if d["key"] == "V_LU"][0]["values"], float)
    lu = lu[np.isfinite(lu)]
    rec, q = A.hazard(-2.0, 0.0, rate=0.4)
    return dict(n=int(len(lu)), mean=float(lu.mean()), sd_mV=float(lu.std(ddof=1) * 1e3),
                se_mean_mV=float(lu.std(ddof=1) / np.sqrt(len(lu)) * 1e3), fpt_mean=float(q.mean()),
                fpt_sd_mV=float(q.std() * 1e3), fold=rec["fold_V"],
                ks=_ks_below(lu, q, rec["fold_V"] + 1.0), steps_per_run=res["solver_stats"]["steps"] / n_runs,
                seconds_per_run=(time.perf_counter() - t) / n_runs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default="v1,v2,v3,v4,v5,v6")
    a = ap.parse_args(argv)
    only = set(a.only.split(","))
    res = {}
    n1200, n120, n3 = (30, 20, 30) if a.quick else (200, 150, 150)
    if "v1" in only:
        res["V1"] = v1_deterministic()
        print(json.dumps(res["V1"], indent=1), flush=True)
    if "v2" in only:
        res["V2_1200"] = v2_stochastic(1200.0, n1200)
        print(json.dumps(res["V2_1200"], indent=1), flush=True)
        res["V2_120"] = v2_stochastic(120.0, n120)
        print(json.dumps(res["V2_120"], indent=1), flush=True)
    if "v3" in only:
        res["V3"] = v3_fixed_bias(n_runs=n3)
        print(json.dumps(res["V3"], indent=1), flush=True)
    if "v4" in only:
        res["V4"] = v2_stochastic(1200.0, n1200 // 2, solver={"tau_frac": 0.02, "max_events_per_step": 50})
        print(json.dumps(res["V4"], indent=1), flush=True)
    if "v5" in only:
        res["V5"] = v5_ld_noise(8 if a.quick else 20)
        print(json.dumps(res["V5"], indent=1), flush=True)
    if "v6" in only:
        res["V6"] = v6_paper_slow(10 if a.quick else 40)
        print(json.dumps(res["V6"], indent=1), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    main()
