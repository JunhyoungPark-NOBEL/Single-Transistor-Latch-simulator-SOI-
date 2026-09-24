"""Tests of the circuit simulator (server/compute/circuit).

Fast tests (~1 min incl. the numba compile on a cold cache):
    pytest server/tests/test_circuit.py -m "not slow"
Slow tests re-run reduced versions of the validation in docs/CIRCUIT_SIMULATOR.md
(stochastic V_LU vs the compound-jump FPT, fixed-bias MFPT vs the exact backward equation).

Reference numbers (full validation, docs/CIRCUIT_SIMULATOR.md §7):
  deterministic load line, paper device V_G = -2 V dark, 0.4 V/s, R_s = 100 ohm, C_d = 1 fF:
      V_LU(drain) = 3.70390 V (fold 3.70369 V, ramp lag +0.21 mV), V_LD = 2.59786 V (fold 2.59787 V)
  stochastic load line, V_G = -1.8 V, I_PH = 2.63 pA, carrier noise only:
      120 V/s (150 runs): 3.2091 V / 34.6 mV vs FPT 3.2077 V / 34.5 mV
      1200 V/s (400 runs): fraction beyond the fold 0.552 +- 0.025 vs FPT atom 0.529; below-fold mean
      3.2521 vs 3.2467 V
  fixed bias V_D = 3.20 V: MFPT(circuit)/MFPT(backward equation) = 0.93 +- 0.08
  paper device, 0.4 V/s (40 runs): 3.6453 V / 7.6 mV vs FPT 3.6442 V / 8.0 mV
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from server.compute.circuit import BENCH_DEFAULTS, run_circuit
from server.progress import JobCancelled

FOLD_LU = 3.7037
FOLD_LD = 2.5979
PHOTO = {"preset": "paper", "vg": -1.8, "light": {"mode": "iph", "iph_pA": 2.63}}


def _summary(res):
    return {s["key"]: s["value"] for s in res["summary"]}


def _check_contract(res: dict, bench: str, mode: str):
    for key in ("bench", "mode", "runs", "events", "summary", "schematic", "solver_stats", "runtime_s", "warnings"):
        assert key in res, key
    assert res["bench"] == bench and res["mode"] == mode
    assert isinstance(res["runtime_s"], float) and isinstance(res["warnings"], list)
    assert 1 <= len(res["runs"]) <= 8
    assert res["runs"][0]["run"] == 0
    for run in res["runs"]:
        n = len(run["t"])
        assert 2 <= n <= 4000
        assert np.all(np.diff(np.asarray(run["t"], float)) >= 0)
        keys = [s["key"] for s in run["signals"]]
        assert len(keys) == len(set(keys))
        for s in run["signals"]:
            assert set(s) >= {"key", "label", "unit", "values"}
            assert set(s["label"]) == {"ko", "en"}
            assert len(s["values"]) == n
    for it in res["summary"]:
        assert set(it) >= {"key", "label", "value"}
    for e in res["events"]:
        assert set(e) >= {"run", "kind", "t"}
    for d in res.get("distributions") or []:
        assert set(d) >= {"key", "label", "unit", "values"}
    for sw in res.get("sweeps") or []:
        assert len(sw["x"]) == len(sw["y"])
    sch = res["schematic"]
    assert "0" in sch["nodes"]
    kinds = {e["kind"] for e in sch["elements"]}
    assert "STL" in kinds and "V" in kinds
    for e in sch["elements"]:
        assert e["kind"] in ("V", "R", "C", "STL", "I", "CMP")
        assert all(nd in sch["nodes"] for nd in e["nodes"])
    st = res["solver_stats"]
    assert set(st) >= {"steps", "rejected", "newton_iters", "runtime_s"}
    assert st["steps"] > 0
    if res.get("trajectory") is not None:
        assert len(res["trajectory"]["vd"]) == len(res["trajectory"]["id"])
    assert "bench_params" in res


# ---- element --------------------------------------------------------------------------------
def test_element_matches_engine_state():
    from server.compute.circuit.element import N_EV, ch_formula, stl_eval
    from server.engine_bridge import MODEL, S, m
    p = S.params(-1.8, 2.63e-12, gamma=0.2794)
    out = np.zeros(N_EV)
    for u, vd in ((0.4, 2.0), (0.55, 3.0), (0.85, 3.5)):
        row = S.state(u, vd, p)
        assert stl_eval(u, row[1], p, MODEL.na, MODEL.vbi, MODEL.rg, MODEL.fg, MODEL.table, out)
        # charge coordinate = compound-FPT lattice coordinate (columns 5+6+7) - C_ox V_GS
        assert out[3] + m.COX_F * p[11] == pytest.approx(row[5] + row[6] + row[7], rel=1e-12)
        assert out[0] == pytest.approx(vd, abs=1e-9)
        assert out[4] == pytest.approx(row[10], rel=1e-12)          # unit-event current
        assert out[5] / m.Q == pytest.approx(row[3], rel=1e-12)     # generation rate
        assert out[6] / m.Q == pytest.approx(row[4], rel=1e-12)     # loss rate
        z = m.components(u, row[1], p, MODEL.na, MODEL.vbi, MODEL.rg, MODEL.fg, MODEL.table)
        assert ch_formula(u, row[1], p) == pytest.approx(z[16], rel=1e-12)


def test_element_extensions_continuous():
    from server.compute.circuit.element import N_EV, stl_eval
    from server.engine_bridge import MODEL, S
    for iph in (0.0, 2.63e-12):
        p = S.params(-2.0, iph)
        a, b = np.zeros(N_EV), np.zeros(N_EV)
        args = (MODEL.na, MODEL.vbi, MODEL.rg, MODEL.fg, MODEL.table)
        assert stl_eval(1e-9, 2.0, p, *args, a) and stl_eval(-1e-9, 2.0, p, *args, b)
        assert np.allclose(a[:4], b[:4], rtol=1e-6, atol=1e-20)
        assert stl_eval(0.5, 1e-9, p, *args, a) and stl_eval(0.5, -1e-9, p, *args, b)
        assert np.allclose(a[:4], b[:4], rtol=1e-6, atol=1e-20)
        # deep forward drain bias and reverse source bias stay finite
        assert stl_eval(0.6, -0.8, p, *args, a) and np.all(np.isfinite(a))
        assert stl_eval(-0.2, 1.0, p, *args, a) and np.all(np.isfinite(a))


# ---- deterministic --------------------------------------------------------------------------
def test_load_line_deterministic_reproduces_folds(progress):
    res = run_circuit({"bench": "load_line", "mode": "deterministic", "device": {"preset": "paper"},
                       "bench_params": {"R_s_ohm": 100.0, "C_d_F": 1e-15, "rate_V_per_s": 0.4}}, progress)
    _check_contract(res, "load_line", "deterministic")
    s = _summary(res)
    assert abs(s["V_LU"] - FOLD_LU) < 1e-3          # obtained +0.21 mV (slow-passage lag at 0.4 V/s)
    assert abs(s["V_LD"] - FOLD_LD) < 1e-3          # obtained -0.01 mV
    assert s["V_LU"] >= s["fold_V_LU"] - 1e-5       # the dynamic latch-up can only lag the fold
    assert s["hrs_branch_dev"] < 1e-2 and s["lrs_branch_dev"] < 1e-2   # decades
    kinds = [e["kind"] for e in res["events"]]
    assert kinds == ["latch_up", "latch_down"]
    assert res["solver_stats"]["steps"] < 20000       # ~2 300 steps (0.2-0.4 s with a warm numba cache)


def test_load_line_trap_matches_be(progress):
    base = {"bench": "load_line", "mode": "deterministic", "device": {"preset": "paper"},
            "bench_params": {"R_s_ohm": 100.0, "C_d_F": 1e-15, "rate_V_per_s": 40.0}}
    be = _summary(run_circuit(base, progress))
    tr = _summary(run_circuit(dict(base, solver={"method": "TRAP"}), progress))
    assert abs(be["V_LU"] - tr["V_LU"]) < 1e-3 and abs(be["V_LD"] - tr["V_LD"]) < 1e-3
    # ramp-rate lag of the slow passage through the fold (~ rate^(2/3)): 4.4 mV at 40 V/s
    assert 1e-3 < be["lag_LU"] < 1e-2


def test_fast_down_ramp_forward_drain(progress):
    """1 ns falls from a latched 5 V state drive the drain junction into forward bias (r < 0)."""
    res = run_circuit({"bench": "pulse", "mode": "deterministic", "device": {"preset": "paper"},
                       "bench_params": {"v_amp_V": 5.0, "rise_s": 1e-9, "fall_s": 1e-9, "width_s": 50e-6,
                                        "period_s": 200e-6, "n_pulses": 2}}, progress)
    _check_contract(res, "pulse", "deterministic")
    assert _summary(res)["P_sw"] == 1.0
    assert any("forward" in w for w in res["warnings"])
    assert not any("truncated" in w or "underflow" in w for w in res["warnings"])


@pytest.mark.parametrize("bench", ["load_line", "pulse", "pbit", "coupled"])
def test_benches_deterministic_contract(bench, progress):
    bp = {"load_line": {"rate_V_per_s": 1200.0}, "pulse": {"n_pulses": 2},
          "pbit": {"n_clocks": 3}, "coupled": {"rate_V_per_s": 1200.0, "vg2_V": -1.9}}[bench]
    res = run_circuit({"bench": bench, "mode": "deterministic", "device": {"preset": "paper"}, "bench_params": bp}, progress)
    _check_contract(res, bench, "deterministic")
    assert set(BENCH_DEFAULTS[bench]) <= set(res["bench_params"])


# ---- stochastic -----------------------------------------------------------------------------
@pytest.mark.parametrize("bench,bp", [
    ("load_line", {"rate_V_per_s": 1200.0, "v_max_V": 5.0}),
    ("pulse", {"n_pulses": 2, "v_amp_V": 3.25, "amplitudes_V": [3.2]}),
    ("pbit", {"n_clocks": 3, "vg_list_V": [-1.85]}),
    ("coupled", {"source": "pulse", "n_pulses": 2, "v_amp_V": 3.3}),
])
def test_benches_stochastic_contract(bench, bp, progress):
    res = run_circuit({"bench": bench, "mode": "stochastic", "device": PHOTO, "bench_params": bp,
                       "stochastic": {"n_runs": 2, "seed": 7}}, progress)
    _check_contract(res, bench, "stochastic")
    assert len(res["runs"]) == 2
    if bench in ("pulse", "pbit"):
        assert len(res["sweeps"]) == 1 and len(res["sweeps"][0]["x"]) == 1
    if bench == "load_line":
        lu = [d for d in res["distributions"] if d["key"] == "V_LU"][0]["values"]
        assert len(lu) == 2 and all(3.0 < v < 3.5 for v in lu)


def test_stochastic_reproducible_and_seeded(progress):
    pl = {"bench": "load_line", "mode": "stochastic", "device": PHOTO,
          "bench_params": {"rate_V_per_s": 1200.0, "v_max_V": 5.0}, "stochastic": {"n_runs": 2, "seed": 11}}
    a = [e["t"] for e in run_circuit(pl, progress)["events"]]
    b = [e["t"] for e in run_circuit(pl, progress)["events"]]
    assert a == b
    c = [e["t"] for e in run_circuit(dict(pl, stochastic={"n_runs": 2, "seed": 12}), progress)["events"]]
    assert a != c


def test_local_states_without_carrier_noise(progress):
    res = run_circuit({"bench": "load_line", "mode": "stochastic", "device": {"preset": "photo"},
                       "bench_params": {"rate_V_per_s": 1200.0},
                       "stochastic": {"n_runs": 3, "carrier_noise": False,
                                      "local_state": {"mode": "frozen", "action": "gidl", "sigma": 0.2154}}}, progress)
    _check_contract(res, "load_line", "stochastic")
    lu = np.array([d for d in res["distributions"] if d["key"] == "V_LU"][0]["values"], float)
    assert np.all(np.isfinite(lu)) and np.std(lu) > 1e-3          # frozen GIDL states spread V_LU
    assert any(s["key"] == "dphi" for s in res["runs"][0]["signals"])


# ---- input handling ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload,match", [
    ({"bench": "nope"}, "unknown bench"),
    ({"bench": "load_line", "mode": "fuzzy"}, "mode"),
    ({"bench": "load_line", "bench_params": {"R_s_ohm": -1}}, "R_s_ohm"),
    ({"bench": "load_line", "bench_params": {"v_max_V": 20}}, "v_max_V"),
    ({"bench": "pulse", "bench_params": {"width_s": 1e-3, "period_s": 1e-4}}, "period"),
    ({"bench": "load_line", "solver": {"method": "RK4"}}, "method"),
    ({"bench": "load_line", "detect": {"i_threshold_A": 1.0}}, "i_threshold"),
    ({"bench": "load_line", "mode": "stochastic", "stochastic": {"local_state": {"mode": "weird"}}}, "local_state"),
])
def test_invalid_input_raises_value_error(payload, match, progress):
    with pytest.raises(ValueError, match=match):
        run_circuit(payload, progress)


def test_expensive_event_level_run_refused(progress):
    """Event-level carrier noise at the paper ramp rate (0.4 V/s) needs ~1e6-1e7 steps per cycle."""
    with pytest.raises(ValueError, match="max_steps"):
        run_circuit({"bench": "load_line", "mode": "stochastic", "device": {"preset": "paper"},
                     "bench_params": {"rate_V_per_s": 0.4}, "solver": {"max_steps": 100000},
                     "stochastic": {"n_runs": 1}}, progress)


def test_auto_rate_fallback_for_stochastic(progress):
    """Under light the HRS relaxes in ~6 µs over the whole noise band: 0.4 V/s (the paper preset rate)
    would need ~3e6 event-level steps, so the unspecified rate falls back to 1200 V/s."""
    res = run_circuit({"bench": "load_line", "mode": "stochastic", "device": PHOTO,
                       "stochastic": {"n_runs": 1}}, progress)
    assert res["bench_params"]["rate_V_per_s"] == 1200.0
    assert any("1200 V/s" in w for w in res["warnings"])


def test_caps_and_cancellation():
    res = run_circuit({"bench": "pulse", "mode": "deterministic", "bench_params": {"n_pulses": 1},
                       "solver": {"max_steps": 5e6}})
    assert any("max_steps capped" in w for w in res["warnings"])

    def cancel(fraction, message=""):
        if fraction > 0.05:
            raise JobCancelled()
    with pytest.raises(JobCancelled):
        run_circuit({"bench": "load_line", "mode": "stochastic", "device": PHOTO,
                     "bench_params": {"rate_V_per_s": 1200.0}, "stochastic": {"n_runs": 3}}, cancel)


# ---- slow: statistical validation ------------------------------------------------------------
@pytest.mark.slow
def test_stochastic_vlu_vs_compound_fpt_120():
    """Atom-free ramp (120 V/s): V_LU distribution vs the exact compound-jump FPT quantiles."""
    from server.compute.circuit.validate import v2_stochastic
    r = v2_stochastic(120.0, 60)
    se = r["sd_mV"] / math.sqrt(r["n"])
    assert abs(r["mean"] - r["fpt"]["mean"]) * 1e3 < 4 * se + 3.0      # full run: +2.9 mV with SE 2.8 mV
    assert 0.7 < r["sd_mV"] / r["fpt"]["sd_mV"] < 1.35
    assert r["ks_below_fold"] < 0.25


@pytest.mark.slow
def test_stochastic_vlu_vs_compound_fpt_1200():
    """1200 V/s: the FPT puts ~53 % at the fold (no post-fold delay); the circuit shows the same
    fraction escaping beyond the fold and the same distribution below it."""
    from server.compute.circuit.validate import v2_stochastic
    r = v2_stochastic(1200.0, 80)
    assert abs(r["frac_beyond_fold"] - r["fpt"]["atom"]) < 0.2
    assert abs(r["below_mean"] - r["fpt"]["below_mean"]) * 1e3 < 15
    assert r["seconds_per_run"] < 5.0


@pytest.mark.slow
def test_fixed_bias_mfpt_vs_backward_equation():
    from server.compute.circuit.validate import v3_fixed_bias
    r = v3_fixed_bias(n_runs=60)
    assert r["escaped"] >= 55
    assert abs(r["ratio"] - 1.0) < 4 * r["ratio_se"]


@pytest.mark.slow
def test_paper_slow_ramp_event_level_vs_fpt_node():
    """engine VALIDATION.md: FPT node V_G = -2 V dark, 0.4 V/s -> mean V_LU ~ 3.644 V, SD ~ 8 mV.
    Full run (40 cycles): 3.6453 V / 7.6 mV (~4.3 s per cycle thanks to the noise bands)."""
    from server.compute.circuit.validate import v6_paper_slow
    r = v6_paper_slow(10)
    assert abs(r["mean"] - r["fpt_mean"]) * 1e3 < 4 * r["se_mean_mV"] + 2.0
    assert 3.0 < r["sd_mV"] < 16.0
    assert r["seconds_per_run"] < 15.0
