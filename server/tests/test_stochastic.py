"""Tests of the stochastic package (server/compute/stochastic.py, stoch_core.py, stoch_mc.py).

Fast tests use the on-disk caches (server/.cache/stochastic) after their first run; the first cold
run of the photo test builds a fold table + 5 hazard nodes (~20 s).  Long checks are marked slow:
    pytest server/tests/test_stochastic.py -m "not slow"
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from server import params as P
from server.compute import stoch_core as C
from server.compute import stoch_mc as MC
from server.compute import stochastic as ST


def _ok_result(r: dict, keys):
    assert isinstance(r["runtime_s"], float) and r["runtime_s"] >= 0
    assert isinstance(r["warnings"], list)
    for k in keys:
        assert k in r, k


def _jsonable(r):
    def conv(o):
        if isinstance(o, np.ndarray):
            return [None if isinstance(x, float) and not math.isfinite(x) else x for x in o.tolist()]
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        raise TypeError(type(o))
    json.dumps(r, default=conv)


# ------------------------------------------------------------------------------------------
# payload validation / caps (no engine work)
# ------------------------------------------------------------------------------------------
def test_payload_caps_and_errors():
    w = []
    dev, sweep, st = ST._resolve({"device": {"preset": "paper", "numerics": {"grid": 5000}},
                                  "sweep": {"vd_max_V": 12, "dv_V": 1e-6},
                                  "stochastic": {"n_cycles": 5000, "fold_nodes": 100, "hazard_nodes": 20,
                                                 "n_traces": 999}}, w)
    assert st["n_cycles"] == 2000 and st["fold_nodes"] == 61 and st["hazard_nodes"] == 9 and st["n_traces"] == 50
    assert sweep["vd_max_V"] == 8.0 and sweep["dv_V"] == 0.0005 and dev["numerics"]["grid"] == 2001
    assert len(w) == 7
    with pytest.raises(ValueError):
        ST._resolve({"sweep": {"rate_V_per_s": 0}}, [])
    with pytest.raises(ValueError):
        ST._resolve({"stochastic": {"local_state": {"mode": "sometimes"}}}, [])
    with pytest.raises(ValueError):
        ST._resolve({"stochastic": {"local_state": {"action": "gate_oxide"}}}, [])
    with pytest.raises(ValueError):
        ST._resolve({"stochastic": {"local_state": {"sigma": -1}}}, [])
    with pytest.raises(ValueError):
        ST._resolve({"stochastic": {"engine": "fast"}}, [])
    with pytest.raises(ValueError):
        ST.run_sweep_mc({"device": {"preset": "photo"}, "stochastic": {"engine": "calibrated_lookup"}})
    with pytest.raises(ValueError):
        ST.run_sweep_mc({"device": {"preset": "paper"}, "sweep": {"vd_max_V": 5},
                         "stochastic": {"engine": "calibrated_lookup"}})
    with pytest.raises(ValueError):
        ST.run_vg_curve_stochastic({"device": {"preset": "paper"}, "vg_min": -1, "vg_max": -2, "n": 3})
    with pytest.raises(ValueError):
        ST.run_hazard({"device": {"preset": "paper"}, "dg": 5.0})


def test_engine_choice():
    dev = P.resolve_device({"preset": "paper"})
    sw = P.resolve_section("paper", "sweep", None)
    st = P.resolve_section("paper", "stochastic", None)
    assert ST._choose_engine(dev, sw, st) == "calibrated_lookup"
    assert ST._choose_engine(dev, dict(sw, vd_max_V=5.0), st) == "general"
    assert ST._choose_engine(P.resolve_device({"preset": "paper", "vg": -1.8}), sw, st) == "general"
    st2 = json.loads(json.dumps(st)); st2["local_state"]["action"] = "junction"
    assert ST._choose_engine(dev, sw, st2) == "general"


# ------------------------------------------------------------------------------------------
# synthetic general-engine checks (no engine, fast)
# ------------------------------------------------------------------------------------------
def _synthetic_table():
    xs = np.linspace(-1, 1, 21)
    vlu = 3.7 - 0.8 * xs          # dV_LU/dX = -0.8 V/V
    vld = np.full_like(xs, 2.6)
    return MC.FoldTable(xs, vlu, vld, [dict(latch=False)] * len(xs))


def test_general_frozen_fold_only_is_exact():
    ft = _synthetic_table()
    r = MC.simulate_general(n=300, seed=7, vd_max=4.5, dv=.002, rate=1.0, mode="frozen", x0=0.0, sigma=0.1,
                            tau_s=5., sigma_e=0.0, tau_e=1., folds=ft, s_lu_e=0., s_ld_e=0.,
                            lu_haz=None, ld_haz=None)
    rng = np.random.default_rng(7)
    Z = rng.standard_normal(300)
    np.testing.assert_allclose(r["V_LU"], 3.7 - 0.8 * 0.1 * Z, atol=1e-9)
    np.testing.assert_allclose(r["V_LD"], 2.6, atol=1e-9)
    assert r["fold_atom"].all()


def test_general_constant_hazard_is_exponential():
    """Constant hazard h over the whole window: V_LU = fold - window + Exp(1)*rate/h (truncated at the fold)."""
    ft = MC.FoldTable(np.array([0.]), np.array([3.0]), np.array([2.0]), [dict(latch=False)])
    d = np.arange(0, 0.5 + 1e-9, 0.001)
    lh = np.where(d <= 0.4, np.log(20.0), C.LOG_ZERO)
    hf = MC.HazardField(np.array([0.]), lh[None, :])
    r = MC.simulate_general(n=4000, seed=3, vd_max=4.0, dv=.001, rate=1.0, mode="none", x0=0., sigma=0.,
                            tau_s=1., sigma_e=0., tau_e=1., folds=ft, s_lu_e=0., s_ld_e=0., lu_haz=hf, ld_haz=None)
    v = r["V_LU"]
    # mean of min(2.6 + Exp/20, 3.0): 2.6 + (1 - e^{-8})/20
    assert abs(v.mean() - (2.6 + (1 - math.exp(-8)) / 20)) < 3e-3
    assert abs(v.std() - 0.05) < 3e-3


def test_general_evolving_ou_correlation():
    ft = _synthetic_table()
    tau = 0.5
    r = MC.simulate_general(n=1500, seed=11, vd_max=4.0, dv=.004, rate=8.0, mode="evolving", x0=0., sigma=0.2,
                            tau_s=tau, sigma_e=0., tau_e=1., folds=ft, s_lu_e=0., s_ld_e=0., lu_haz=None, ld_haz=None)
    x = r["state"]
    T = 2 * 4.0 / 8.0 * (1 + 1 / 1000)   # cycle duration (2*(steps+1) samples of dt)
    lag = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(x.std() - 0.2) < 0.02
    assert abs(lag - math.exp(-T / tau)) < 0.08
    assert abs(C.lag1(r["V_LU"]) - math.exp(-T / tau)) < 0.08


def test_stats_helpers():
    v = np.r_[np.linspace(1, 2, 50), np.nan]
    s = C.stats(v)
    assert s["n"] == 50 and s["censored"] == 1 and abs(s["mean"] - 1.5) < 1e-12
    h = C.histogram(v)
    assert sum(h["counts"]) == 50 and len(h["edges"]) == len(h["counts"]) + 1
    c = C.ecdf(v)
    assert len(c["v"]) == 50 and c["p"][-1] == 1.0


# ------------------------------------------------------------------------------------------
# engine-backed fast checks (cached after the first run)
# ------------------------------------------------------------------------------------------
def test_hazard_node_cross_check():
    """VALIDATION.md / stl_api smoke test: V_G = -2 V dark, 0.4 V/s -> mean 3.6442 V, SD 8.03 mV."""
    r = ST.run_hazard({"device": {"preset": "paper"}, "sweep": {"rate_V_per_s": 0.4}})
    _ok_result(r, ["fold_V", "VLD_fold_V", "voltage", "hazard", "survival", "quantiles", "stats", "rate_V_per_s"])
    assert r["fold_V"] == pytest.approx(3.7037, abs=2e-4)
    assert r["stats"]["mean"] == pytest.approx(3.6442, abs=3e-4)
    assert r["stats"]["sd"] * 1e3 == pytest.approx(8.03, abs=0.1)
    assert len(r["survival"]) == len(r["voltage"])
    assert np.all(np.diff(r["survival"]) <= 1e-15)
    _jsonable(r)


def test_calibrated_engine_validation():
    """VALIDATION.md: 100 sweeps, seed 2026092920, 0.4 V/s -> V_LU 3.634 V / 119 mV, V_LD 2.700 V / 21 mV."""
    r = ST.run_sweep_mc({"device": {"preset": "paper"}})
    _ok_result(r, ["engine", "V_LU", "V_LD", "stats", "hist", "cdf", "traces", "cycle_state", "fold_table",
                   "centre", "measured"])
    assert r["engine"] == "calibrated_lookup"
    lu, ld = r["stats"]["LU"], r["stats"]["LD"]
    assert lu["mean"] == pytest.approx(3.634, abs=0.002) and lu["sd"] * 1e3 == pytest.approx(119.0, abs=1.5)
    assert ld["mean"] == pytest.approx(2.700, abs=0.002) and ld["sd"] * 1e3 == pytest.approx(21.4, abs=1.0)
    assert r["measured"]["stats"]["LU"]["sd"] * 1e3 == pytest.approx(123.1, abs=0.2)
    assert len(r["traces"]) == 12 and len(r["cycle_state"]) == 100
    assert r["centre"]["V_LU"] == pytest.approx(3.7037, abs=2e-4)
    _jsonable(r)


def test_calibrated_state_replay_is_exact():
    g = MC._gdc()
    n, seed, dv, rate = 20, 99, .002, .4
    t, fd, hz, ci, quant = g.setup()
    for mode, amp, tau in [("dynamic", 1., 5.), ("frozen", 1.3, None), ("stationary", .7, 3.)]:
        z, _ = MC.calibrated(n, seed, mode, dv, rate, amp, tau)
        states, draws, _ = MC.calibrated_states(n, seed, mode, dv, rate, amp, tau)
        steps = int(round(4 / dv)); dt = dv / rate
        for d in range(2):
            st = states[d].reshape(-1, 2)
            V = np.linspace(0, 4, steps + 1) if d == 0 else np.linspace(4, 0, steps + 1)
            fold = fd(st)[:, d]; volts = np.tile(V, n)
            dist = fold - volts if d == 0 else volts - fold
            h = hz[d](np.c_[st, np.clip(dist, 0, .32)]).reshape(n, -1); h[dist.reshape(n, -1) > .32] = 0
            integ = np.cumsum(h * dt, axis=1) - h[:, 0, None] * dt
            ev = (integ >= draws[d][:, None]) | (dist.reshape(n, -1) <= 0)
            v = np.where(ev.any(1), V[ev.argmax(1)], np.nan)
            np.testing.assert_array_equal(v, z["transition"][d, :, 0])


def test_photo_dark_calibration_point():
    """Photo preset, V_G = -1.8 V dark, 1200 V/s, 400 cycles, seed 20260922: measured 3.806 V / 173 mV
    (mc_cycles.py reference with the same algorithm: 3.8079 V / 173.8 mV)."""
    r = ST.run_sweep_mc({"device": {"preset": "photo"}})
    assert r["engine"] == "general"
    lu = r["stats"]["LU"]
    assert lu["n"] == 400 and lu["censored"] == 0
    assert lu["mean"] == pytest.approx(3.806, abs=0.008)
    assert lu["sd"] * 1e3 == pytest.approx(173.2, abs=6.0)
    assert r["measured"] is not None and r["measured"]["stats"]["LU"]["n"] == 400
    assert r["fold_table"] is not None and len(r["fold_table"]["delta"]) == 25
    assert len(r["traces"]) == 12
    _jsonable(r)


def test_determinism_fixed_seed():
    pl = {"device": {"preset": "photo"}, "stochastic": {"n_cycles": 50, "seed": 42}}
    a, b = ST.run_sweep_mc(pl), ST.run_sweep_mc(pl)
    np.testing.assert_array_equal(a["V_LU"], b["V_LU"])
    np.testing.assert_array_equal(a["V_LD"], b["V_LD"])
    c = ST.run_sweep_mc({"device": {"preset": "photo"}, "stochastic": {"n_cycles": 50, "seed": 43}})
    assert not np.array_equal(a["V_LU"], c["V_LU"])
    e = {"device": {"preset": "photo"}, "stochastic": {"n_cycles": 30, "seed": 5,
                                                       "local_state": {"mode": "evolving", "tau_s": 0.001}}}
    np.testing.assert_array_equal(ST.run_sweep_mc(e)["V_LU"], ST.run_sweep_mc(e)["V_LU"])


# ------------------------------------------------------------------------------------------
# slow checks
# ------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_general_vs_calibrated_frozen_paper():
    base = {"device": {"preset": "paper"}, "stochastic": {"local_state": {"mode": "frozen"}, "n_cycles": 1000,
                                                          "seed": 1}}
    cal = ST.run_sweep_mc(dict(base, stochastic=dict(base["stochastic"], engine="calibrated_lookup")))
    gen = ST.run_sweep_mc(dict(base, stochastic=dict(base["stochastic"], engine="general", ld_carrier_noise=True)))
    for d, tol_m, tol_s in (("LU", 0.02, 0.012), ("LD", 0.01, 0.004)):
        assert gen["stats"][d]["mean"] == pytest.approx(cal["stats"][d]["mean"], abs=tol_m)
        assert gen["stats"][d]["sd"] == pytest.approx(cal["stats"][d]["sd"], abs=tol_s)


@pytest.mark.slow
def test_vg_curve_paper_points():
    r = ST.run_vg_curve_stochastic({"device": {"preset": "paper"}, "stochastic": {"local_state": {"mode": "frozen"}},
                                    "vg_min": -2.0, "vg_max": -1.1, "n": 4})
    _ok_result(r, ["vg", "mean_VLU", "sd_VLU_mV", "state_sd_mV", "noise_sd_mV", "fold_centre_V", "VLD_fold_V",
                   "no_latch_weight", "measured"])
    assert np.all(np.isfinite(r["mean_VLU"]))
    assert r["mean_VLU"][-1] == pytest.approx(4.354, abs=0.03)
