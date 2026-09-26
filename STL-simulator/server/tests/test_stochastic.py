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


def test_calibrated_dv_guard():
    """Review finding 1: calibrated_lookup only for ΔV = 10 mV / k (else ZeroDivisionError / misaligned traces)."""
    for dv in (0.01, 0.005, 0.01 / 3, 0.0025, 0.002, 0.001, 0.0005):
        assert ST.calibrated_dv_ok(dv), dv
    for dv in (0.1, 0.05, 0.02, 0.015, 0.004, 0.003, 0.0015):
        assert not ST.calibrated_dv_ok(dv), dv
    dev = P.resolve_device({"preset": "paper"})
    st = P.resolve_section("paper", "stochastic", None)
    w = []
    assert ST._choose_engine(dev, dict(P.resolve_section("paper", "sweep", None), dv_V=0.004), st, w) == "general"
    assert any("10 mV / k" in x for x in w)
    with pytest.raises(ValueError, match="10 mV / k"):
        ST.run_sweep_mc({"device": {"preset": "paper"}, "sweep": {"dv_V": 0.02},
                         "stochastic": {"engine": "calibrated_lookup", "n_cycles": 3}})


def test_calibrated_traces_aligned():
    """Review finding 1: every calibrated trace has as many currents as voltages (ΔV = 2.5 mV → k = 4)."""
    r = ST.run_sweep_mc({"device": {"preset": "paper"}, "sweep": {"dv_V": 0.0025}, "stochastic": {"n_cycles": 3}})
    assert r["engine"] == "calibrated_lookup" and r["traces"]
    for t in r["traces"]:
        for d in ("up", "down"):
            assert len(t[d]["vd"]) == len(t[d]["id"]) == 401
        assert t["up"]["vd"][0] == 0 and t["up"]["vd"][-1] == pytest.approx(4.0)


def test_auto_engine_coarse_dv_runs_general():
    """Review finding 1: ΔV = 20 mV on the paper preset used to raise ZeroDivisionError."""
    r = ST.run_sweep_mc({"device": {"preset": "paper"}, "sweep": {"dv_V": 0.02},
                         "stochastic": {"n_cycles": 20, "local_state": {"mode": "none"}}})
    assert r["engine"] == "general" and r["stats"]["LU"]["n"] == 20
    assert any("10 mV / k" in x for x in r["warnings"])
    for t in r["traces"]:
        assert len(t["up"]["vd"]) == len(t["up"]["id"])


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


def _steep_field(lam=0.005, h0=5e3, dmax=0.3):
    d = np.arange(0, dmax + 1e-9, C.DIST_STEP)
    return MC.HazardField(np.array([0.]), (np.log(h0) - d / lam)[None, :])


@pytest.mark.parametrize("mode", ["none", "evolving"])
def test_general_event_independent_of_dv(mode):
    """Review finding 5: a hazard rising e-fold per 5 mV towards the fold; the escape statistics must not depend
    on the sweep step ΔV (0.5 → 50 mV) beyond MC noise.  (Before: mean shifted by ~-30 mV, σ ~ doubled.)"""
    ft = MC.FoldTable(np.array([-1., 1.]), np.array([3.0, 3.0]), np.array([2.0, 2.0]), [dict(latch=False)] * 2)
    hf = _steep_field()
    res = {}
    for dv in (0.0005, 0.002, 0.05, 0.1):
        r = MC.simulate_general(n=3000, seed=5, vd_max=4.0, dv=dv, rate=1.0, mode=mode, x0=0., sigma=0.,
                                tau_s=1., sigma_e=0., tau_e=1., folds=ft, s_lu_e=0., s_ld_e=0., lu_haz=hf, ld_haz=hf)
        res[dv] = r
        assert r["hazard_substeps"] == max(1, int(np.ceil(dv / 0.002 - 1e-9)))
    ref = res[0.0005]
    for dv in (0.002, 0.05, 0.1):
        for key in ("V_LU", "V_LD"):
            assert np.nanmean(res[dv][key]) == pytest.approx(np.nanmean(ref[key]), abs=0.6e-3), (dv, key)
            assert np.nanstd(res[dv][key]) == pytest.approx(np.nanstd(ref[key]), rel=0.05), (dv, key)


def test_general_dv_independent_engine():
    """Review finding 5 on the real hazard node (paper device, centre state, ld_carrier_noise on)."""
    base = {"device": {"preset": "paper"}, "stochastic": {"n_cycles": 400, "engine": "general",
                                                          "local_state": {"mode": "none"}}}
    a = ST.run_sweep_mc(dict(base, sweep={"dv_V": 0.002}))
    b = ST.run_sweep_mc(dict(base, sweep={"dv_V": 0.05}))
    for d in ("LU", "LD"):
        assert b["stats"][d]["mean"] == pytest.approx(a["stats"][d]["mean"], abs=5e-4)
        assert b["stats"][d]["sd"] == pytest.approx(a["stats"][d]["sd"], rel=0.05)


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
    # normalised to every cycle: the censored one keeps the CDF below 1 (review finding 7)
    assert len(c["v"]) == 50 and c["n"] == 50 and c["n_total"] == 51 and c["p"][-1] == pytest.approx(50 / 51)
    assert C.ecdf(v[:-1])["p"][-1] == 1.0


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


def test_truncated_mixture_moments():
    """Review finding 2: exact moments without truncation; conditional moments with it."""
    rng = np.random.default_rng(0)
    fl = np.linspace(3.5, 4.5, 41); wt = np.exp(-((fl - 4.0) / 0.2) ** 2 / 2)
    eps = rng.normal(-0.05, 0.01, 2000); weps = np.ones_like(eps)
    P1, mean, sv, nv = ST.truncated_mixture(fl, wt, eps, weps, 10.0)
    w = wt / wt.sum()
    fmu = np.sum(w * fl)
    assert P1 == 1.0 and mean == pytest.approx(fmu + eps.mean(), abs=1e-12)
    assert sv == pytest.approx(np.sum(w * (fl - fmu) ** 2), rel=1e-10) and nv == pytest.approx(eps.var(), rel=1e-10)
    # brute force over the product grid, truncated at 4.0 V
    V = fl[:, None] + eps[None, :]; W = w[:, None] * np.full(len(eps), 1 / len(eps))[None, :]
    keep = V <= 4.0
    P2, mean2, sv2, nv2 = ST.truncated_mixture(fl, wt, eps, weps, 4.0)
    assert P2 == pytest.approx(W[keep].sum(), rel=1e-12)
    bm = np.sum(W[keep] * V[keep]) / W[keep].sum()
    assert mean2 == pytest.approx(bm, abs=1e-12)
    assert sv2 + nv2 == pytest.approx(np.sum(W[keep] * (V[keep] - bm) ** 2) / W[keep].sum(), rel=1e-9)
    assert ST.truncated_mixture(fl, wt, eps, weps, 3.0)[1] is None


def test_vg_curve_stochastic_censors_beyond_sweep():
    """Review finding 2: the V_G curve honours sweep.vd_max_V like sweep_mc (centre state, carrier noise only:
    V_LU = the first-passage quantiles, censored above vd_max)."""
    pl = {"device": {"preset": "paper"}, "sweep": {"vd_max_V": 3.645, "rate_V_per_s": 0.4},
          "stochastic": {"engine": "general", "local_state": {"mode": "none"}}, "vg_min": -2.0, "vg_max": -1.4, "n": 2}
    r = ST.run_vg_curve_stochastic(pl)
    q = C.quantiles(C.lu_hazard_curve(P.build_p(P.resolve_device({"preset": "paper"}))), 0.4)
    keep = q <= 3.645
    assert r["censored_weight"][0] == pytest.approx(1 - keep.mean(), abs=1e-9)
    assert r["beyond_sweep_weight"][0] == pytest.approx(1 - keep.mean(), abs=1e-9) and r["no_latch_weight"][0] == 0
    assert r["mean_VLU"][0] == pytest.approx(q[keep].mean(), abs=1e-9)
    assert r["sd_VLU_mV"][0] == pytest.approx(1e3 * q[keep].std(), abs=1e-6)
    assert 0.2 < r["censored_weight"][0] < 0.8
    assert r["censored_weight"][1] > 0.99                                     # V_G -1.4: fold 4.19 V
    assert not np.isfinite(r["mean_VLU"][1]) or r["mean_VLU"][1] <= 3.645
    assert any("beyond the sweep maximum" in w for w in r["warnings"])
    # against sweep_mc with the same payload
    m = ST.run_sweep_mc({"device": {"preset": "paper"}, "sweep": pl["sweep"],
                         "stochastic": dict(pl["stochastic"], n_cycles=2000)})
    lu = m["stats"]["LU"]
    assert lu["censored"] / 2000 == pytest.approx(r["censored_weight"][0], abs=0.035)
    assert lu["mean"] == pytest.approx(r["mean_VLU"][0], abs=1e-3)


def test_hazard_high_fold_reports_kernel_range():
    """Review finding 4: a fold above ~5.1 V (l_GIDL = 44 nm) loses every hazard voltage because the lattice
    reverse bias leaves the avalanche kernel (<= 5 V); the warning must say so, not 'window too narrow'."""
    r = ST.run_hazard({"device": {"preset": "paper", "calib": {"l_gidl_nm": 44}}})
    assert r["fold_V"] > 5.3 and len(r["voltage"]) == 0
    assert r["kernel_skipped"] == r["n_voltages"] == r["skipped"] > 0
    assert any("avalanche cluster kernel" in w for w in r["warnings"])
    assert not any("too narrow" in w for w in r["warnings"])
    ok = ST.run_hazard({"device": {"preset": "paper", "calib": {"l_gidl_nm": 40}}})
    assert len(ok["voltage"]) > 50 and ok["kernel_skipped"] == 0 and ok["stats"]["sd"] > 0.005


def test_hazard_rows_use_state_row_fallback():
    """Review finding 4: S.state fails at V_D >~ 5.1 V, u ~ 0.9 V; the hazard rows use the bracket-shrinking
    deterministic.state_row (identical where S.state succeeds)."""
    from server.engine_bridge import S
    p = np.asarray(P.build_p(P.resolve_device({"preset": "paper", "calib": {"l_gidl_nm": 44}})), float)
    with pytest.raises(ValueError):
        S.state(0.9, 5.3, p)
    rows = C._state_rows(np.array([0.5, 0.9]), 5.3, p)
    assert np.isfinite(rows).all()
    np.testing.assert_array_equal(C._state_rows(np.array([0.5]), 3.2, p)[0], S.state(0.5, 3.2, p))


def test_fold_node_matches_deterministic_folds():
    """Review finding 9: the stochastic fold records carry the same refined fold states as the `folds` readout."""
    from server.compute import deterministic as D
    for dev in ({"preset": "paper"}, {"preset": "photo"}):
        f = D.run_folds({"device": dev})["folds"]
        rec = C.fold_node(P.build_p(P.resolve_device(dev)), 601)
        for k in ("V_LU", "V_LD", "I_LU", "I_LD", "u_LU", "u_LD"):
            assert rec[k] == f[k], (dev, k)
    hz = ST.run_hazard({"device": {"preset": "paper"}, "sweep": {"rate_V_per_s": 0.4}})
    assert hz["I_at_fold_A"] == D.run_folds({"device": {"preset": "paper"}})["folds"]["I_LU"]


def test_fold_node_rejects_locus_gap():
    """Review finding 3 (stochastic side): V_G = +0.5 V has no traceable fold."""
    rec = C.fold_node(P.build_p(P.resolve_device({"preset": "paper", "vg": 0.5})), 601)
    assert rec["latch"] is False and rec["locus_gap"] is True
    h = C.lu_hazard_curve(P.build_p(P.resolve_device({"preset": "paper", "vg": 0.5})))
    assert h["fold_V"] is None and h["locus_gap"] is True


def test_mc_cdf_counts_censored_cycles():
    """Review finding 7: with censored cycles the LU CDF plateaus at the latched fraction."""
    r = ST.run_sweep_mc({"device": {"preset": "photo"}, "sweep": {"vd_max_V": 3.9}, "stochastic": {"n_cycles": 200}})
    lu = r["stats"]["LU"]
    assert lu["censored"] > 20
    assert r["cdf"]["LU"]["p"][-1] == pytest.approx(lu["n"] / 200)
    assert r["cdf"]["LU"]["n_total"] == 200 and len(r["cdf"]["LU"]["v"]) == lu["n"]
    assert r["cdf"]["LD"]["p"][-1] == pytest.approx(1.0) and r["cdf"]["LD"]["n_total"] == lu["n"]


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
    """Paper V_G curve (0-6 V sweep as in validation.check_vg_curve_stochastic): mean V_LU 4.354 V at -1.1 V.
    With the preset's 0-4 V sweep the same point is mostly censored (review finding 2)."""
    pl = {"device": {"preset": "paper"}, "sweep": {"vd_max_V": 6.0},
          "stochastic": {"local_state": {"mode": "frozen"}}, "vg_min": -2.0, "vg_max": -1.1, "n": 4}
    r = ST.run_vg_curve_stochastic(pl)
    _ok_result(r, ["vg", "mean_VLU", "sd_VLU_mV", "state_sd_mV", "noise_sd_mV", "fold_centre_V", "VLD_fold_V",
                   "no_latch_weight", "beyond_sweep_weight", "censored_weight", "measured"])
    assert np.all(np.isfinite(r["mean_VLU"]))
    assert r["mean_VLU"][-1] == pytest.approx(4.354, abs=0.03)
    assert np.all(np.asarray(r["beyond_sweep_weight"]) < 1e-6)
    r4 = ST.run_vg_curve_stochastic(dict(pl, sweep={"vd_max_V": 4.0}))
    assert r4["mean_VLU"][0] == pytest.approx(r["mean_VLU"][0], abs=2e-3)       # -2 V: nothing censored
    assert r4["mean_VLU"][-1] < 4.0 and r4["censored_weight"][-1] > 0.5
