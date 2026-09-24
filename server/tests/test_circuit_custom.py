"""Tests of user-drawn circuits: kind "circuit", bench "custom" (docs/WEB_CONTRACT.md §6,
server/compute/circuit/custom.py).

Fast (~30 s with a warm numba cache):  pytest server/tests/test_circuit_custom.py -m "not slow"
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from server.compute.circuit import run_circuit

PHOTO = {"preset": "paper", "vg": -1.8, "light": {"mode": "iph", "iph_pA": 2.63}}


def _custom(elements, tran, mode="deterministic", **kw):
    p = {"bench": "custom", "mode": mode, "netlist": {"elements": elements}, "tran": tran}
    p.update(kw)
    return p


def _sig(res, run=0):
    r = res["runs"][run]
    return np.asarray(r["t"], float), {s["key"]: np.asarray(s["values"], float) for s in r["signals"]}


def _summary(res):
    return {s["key"]: s["value"] for s in res["summary"]}


def _load_line_elements(vmax, T, device, vg, name="X1", d="d", src="src", g="g", rs=1e3):
    return [
        {"type": "V", "name": "Vsrc", "nodes": [src, "0"], "wave": {"kind": "pwl", "t": [0, T / 2, T], "v": [0, vmax, 0]}},
        {"type": "R", "name": "Rs", "nodes": [src, d], "value": rs},
        {"type": "C", "name": "Cd", "nodes": [d, "0"], "value": 2e-15},
        {"type": "V", "name": "VG", "nodes": [g, "gnd"], "wave": {"kind": "dc", "value": vg}},
        {"type": "STL", "name": name, "nodes": {"d": d, "g": g, "s": "0"}, "device": device, "light_pA": None},
    ]


def _check_result(res, mode):
    for key in ("bench", "mode", "runs", "events", "summary", "schematic", "solver_stats", "runtime_s", "warnings",
                "nodes", "elements", "op", "probes", "tran", "feasibility"):
        assert key in res, key
    assert res["bench"] == "custom" and res["mode"] == mode
    assert res["nodes"][0] == "0"
    for run in res["runs"]:
        t = np.asarray(run["t"], float)
        assert 2 <= len(t) <= 4000 and np.all(np.diff(t) >= 0)
        for s in run["signals"]:
            assert set(s) >= {"key", "label", "unit", "values", "axis"}
            assert set(s["label"]) == {"ko", "en"} and s["label"]["ko"] and s["label"]["en"]
            assert len(s["values"]) == len(t)
    assert set(res["op"]) >= {s["key"] for s in res["runs"][0]["signals"] if s["key"] != "V(0)"}
    for e in res["events"]:
        assert e["kind"] in ("latch_up", "latch_down") and isinstance(e["cell"], str) and "v_d" in e


# ---- linear circuits ---------------------------------------------------------------------------
@pytest.mark.parametrize("method,reltol", [("BE", 1e-4), ("TRAP", 1e-3)])
def test_rc_charging_matches_analytic(method, reltol):
    R, C, td = 1e3, 1e-6, 1e-4
    tau = R * C
    res = run_circuit(_custom([
        {"type": "V", "name": "V1", "nodes": ["in", "0"],
         "wave": {"kind": "pulse", "v1": 0, "v2": 1, "td": td, "tr": 1e-9, "tf": 1e-9, "pw": 1, "per": 0}},
        {"type": "R", "name": "R1", "nodes": ["in", "out"], "value": R},
        {"type": "C", "name": "C1", "nodes": ["out", "0"], "value": C},
    ], {"t_stop_s": 6 * tau, "dt_max_s": 1e-5, "method": method, "reltol": reltol}))
    _check_result(res, "deterministic")
    t, s = _sig(res)
    t0 = td + 1e-9
    exact = np.where(t > t0, 1 - np.exp(-(t - t0) / tau), 0.0)
    m = t > td + 2e-9
    err = np.max(np.abs(s["V(out)"] - exact)[m])
    assert err < 5e-3, err                  # < 0.5 % of the 1 V step (BE ~0.07 %, TRAP ~1e-4 %)
    # KCL / branch currents: C current = R current, V source delivers it (SPICE sign: I(V1) = -I(R1))
    assert np.max(np.abs(s["I(C1)"] - s["I(R1)"])) < 1e-9
    assert np.max(np.abs(s["I(V1)"] + s["I(R1)"])) < 1e-12
    assert abs(s["I(R1)"][np.searchsorted(t, t0 + tau)] - math.exp(-1) / R) < 1e-5
    assert res["op"]["V(out)"] == 0.0 and res["op"]["I(C1)"] == 0.0


def test_current_source_sign_convention():
    base = [{"type": "R", "name": "R1", "nodes": ["n", "0"], "value": 1e3}]
    # I1 from 0 to n (through the source): pushed out of its '-' node n into the circuit -> V(n) = +1 V
    res = run_circuit(_custom(base + [{"type": "I", "name": "I1", "nodes": ["0", "n"], "wave": {"kind": "dc", "value": 1e-3}}],
                              {"t_stop_s": 1e-3}))
    _, s = _sig(res)
    assert np.allclose(s["V(n)"], 1.0, atol=1e-9) and np.allclose(s["I(I1)"], 1e-3) and np.allclose(s["I(R1)"], 1e-3)
    res = run_circuit(_custom(base + [{"type": "I", "name": "I1", "nodes": ["n", "0"], "wave": {"kind": "dc", "value": 1e-3}}],
                              {"t_stop_s": 1e-3}))
    _, s = _sig(res)
    assert np.allclose(s["V(n)"], -1.0, atol=1e-9) and np.allclose(s["I(R1)"], -1e-3)
    # a voltage source delivering power has a negative current (SPICE)
    res = run_circuit(_custom([{"type": "V", "name": "V1", "nodes": ["a", "0"], "wave": {"kind": "dc", "value": 2.0}},
                               {"type": "R", "name": "R1", "nodes": ["a", "0"], "value": 1e3}], {"t_stop_s": 1e-3}))
    _, s = _sig(res)
    assert np.allclose(s["I(V1)"], -2e-3) and np.allclose(s["I(R1)"], 2e-3)
    assert res["op"]["I(V1)"] == pytest.approx(-2e-3)


def _pulse_ref(t, v1, v2, td, tr, tf, pw, per, n):
    out = np.full_like(t, v1)
    for k in range(n):
        tl = t - td - k * per
        out = np.where((tl >= 0) & (tl < tr), v1 + (v2 - v1) * tl / tr, out)
        out = np.where((tl >= tr) & (tl < tr + pw), v2, out)
        out = np.where((tl >= tr + pw) & (tl < tr + pw + tf), v2 + (v1 - v2) * (tl - tr - pw) / tf, out)
    return out


def test_pulse_and_sine_waveforms():
    pw = dict(kind="pulse", v1=0.2, v2=1.5, td=1e-4, tr=1e-5, tf=2e-5, pw=1e-4, per=5e-4, ncycles=3)
    sw = dict(kind="sine", vo=0.5, va=1.0, freq=1e3, td=2e-4, theta=100.0)
    res = run_circuit(_custom([
        {"type": "V", "name": "VP", "nodes": ["p", "0"], "wave": pw},
        {"type": "R", "name": "RP", "nodes": ["p", "0"], "value": 1e3},
        {"type": "V", "name": "VS", "nodes": ["s", "0"], "wave": sw},
        {"type": "R", "name": "RS", "nodes": ["s", "0"], "value": 1e3},
    ], {"t_stop_s": 3e-3, "dt_max_s": 1e-5}, probes=["V(p)", "V(s)"]))
    t, s = _sig(res)
    assert [x["key"] for x in res["runs"][0]["signals"]] == ["V(p)", "V(s)"]
    ref = _pulse_ref(t, 0.2, 1.5, 1e-4, 1e-5, 2e-5, 1e-4, 5e-4, 3)
    assert np.max(np.abs(s["V(p)"] - ref)) < 1e-6
    # every pulse corner is a time point of the output (corners are kept by the decimation)
    for k in range(3):
        for c in (0, 1e-5, 1.1e-4, 1.3e-4):
            tc = 1e-4 + k * 5e-4 + c
            assert np.min(np.abs(t - tc)) < 1e-12 * 1e3, tc
    assert np.allclose(s["V(p)"][t > 1.25e-3], 0.2)                     # ncycles = 3: v1 afterwards
    x = t - 2e-4
    sref = np.where(x > 0, 0.5 + np.exp(-100 * x) * np.sin(2 * np.pi * 1e3 * x), 0.5)
    assert np.max(np.abs(s["V(s)"] - sref)) < 1e-3                       # PWL sampling, 128 points per period
    el = {e["name"]: e for e in res["elements"]}
    assert el["VP"]["wave"]["kind"] == "pulse" and el["VS"]["wave"]["points_per_period"] >= 64


def test_zero_rise_time_and_vertical_pwl_step_are_finite_edges():
    res = run_circuit(_custom([
        {"type": "V", "name": "V1", "nodes": ["a", "0"], "wave": {"kind": "pwl", "t": [0, 1e-3, 1e-3, 2e-3], "v": [0, 0, 1, 1]}},
        {"type": "R", "name": "R1", "nodes": ["a", "b"], "value": 1e3},
        {"type": "C", "name": "C1", "nodes": ["b", "0"], "value": 1e-9},
        {"type": "V", "name": "V2", "nodes": ["c", "0"], "wave": {"kind": "pulse", "v1": 0, "v2": 1, "td": 5e-4, "tr": 0, "tf": 0,
                                                                 "pw": 5e-4, "per": 0}},
        {"type": "R", "name": "R2", "nodes": ["c", "0"], "value": 1e3},
    ], {"t_stop_s": 2e-3, "dt_max_s": 1e-5}))
    assert any("vertical pwl step" in w for w in res["warnings"])
    assert any("replaced by a finite edge" in w for w in res["warnings"])
    t, s = _sig(res)
    assert s["V(a)"][-1] == pytest.approx(1.0) and s["V(b)"][-1] == pytest.approx(1.0, abs=1e-3)


# ---- STL circuits ------------------------------------------------------------------------------
def _kcl_residual(res, run=0):
    """max over time and nodes of |sum of currents leaving the node| / current scale."""
    t, s = _sig(res, run)
    nodes = res["nodes"][1:]
    worst = 0.0
    for n in nodes:
        tot = np.zeros_like(t)
        scale = np.zeros_like(t)
        for e in res["elements"]:
            if e["type"] == "STL":
                for term in ("d", "g", "s"):
                    if e["nodes"][term] == n:
                        i = s[f"I({e['name']}.{term})"]
                        tot += i
                        scale = np.maximum(scale, np.abs(i))
            else:
                a, b = e["nodes"]
                i = s[f"I({e['name']})"]
                if a == n:
                    tot += i
                if b == n:
                    tot -= i
                if n in (a, b):
                    scale = np.maximum(scale, np.abs(i))
        # the numerical GMIN (1e-18 S from every node to ground, ~1e-17 A) is not an element: absolute floor 1e-16 A
        worst = max(worst, float(np.max(np.maximum(np.abs(tot) - 1e-16, 0.0) / (scale + 1e-15))))
    return worst


def test_kcl_at_every_node_with_stl():
    T = 2 * 4.2 / 1200.0
    res = run_circuit(_custom([
        {"type": "V", "name": "V1", "nodes": ["src", "0"], "wave": {"kind": "pwl", "t": [0, T / 2, T], "v": [0, 4.2, 0]}},
        {"type": "R", "name": "R1", "nodes": ["src", "d"], "value": 1e3},
        {"type": "C", "name": "C1", "nodes": ["d", "0"], "value": 2e-15},
        {"type": "C", "name": "C2", "nodes": ["d", "s"], "value": 1e-15},
        {"type": "STL", "name": "X1", "nodes": {"d": "d", "g": "g", "s": "s"}, "device": {"preset": "paper"}, "light_pA": None},
        {"type": "R", "name": "R2", "nodes": ["s", "0"], "value": 100.0},
        {"type": "V", "name": "VG", "nodes": ["g", "0"], "wave": {"kind": "dc", "value": -2.0}},
        {"type": "I", "name": "I1", "nodes": ["0", "d"], "wave": {"kind": "dc", "value": 1e-12}},
    ], {"t_stop_s": T, "dt_max_s": T / 2000}))
    _check_result(res, "deterministic")
    assert _kcl_residual(res) < 1e-5
    t, s = _sig(res)
    assert np.all(s["I(X1.g)"] == 0) and np.allclose(s["I(X1.s)"], -s["I(X1.d)"])
    kinds = [e["kind"] for e in res["events"]]
    assert kinds == ["latch_up", "latch_down"]
    sm = _summary(res)
    assert sm["X1.n_latch_up"] == 1 and sm["X1.final_state"] == "HRS" and sm["X1.latched_end"] == 0
    assert 3.7 < sm["X1.vd_first_lu"] < 3.8
    assert res["trajectory"] is not None and len(res["trajectory"]["vd"]) == len(t)


def test_load_line_template_reproduces_bench():
    rate, vmax = 1200.0, 4.0
    T = 2 * vmax / rate
    bench = run_circuit({"bench": "load_line", "mode": "deterministic", "device": {"preset": "paper"},
                         "bench_params": {"rate_V_per_s": rate, "v_max_V": vmax}})
    cust = run_circuit(_custom(_load_line_elements(vmax, T, {"preset": "paper"}, -2.0),
                               {"t_stop_s": T, "dt_max_s": T / 2000, "method": "BE", "reltol": 1e-3}))
    b = _summary(bench)
    ev = {e["kind"]: e["v_d"] for e in cust["events"]}
    assert abs(ev["latch_up"] - b["V_LU"]) < 5e-4          # obtained: identical (same kernel, same steps)
    assert abs(ev["latch_down"] - b["V_LD"]) < 5e-4
    assert _summary(cust)["X1.vd_first_lu"] == pytest.approx(ev["latch_up"])


def test_two_stl_cells_with_different_devices():
    rate, vmax = 1200.0, 4.2
    T = 2 * vmax / rate
    els = _load_line_elements(vmax, T, PHOTO, -1.8)
    els += [{"type": "R", "name": "Rs2", "nodes": ["src", "d2"], "value": 1e3},
            {"type": "V", "name": "VG2", "nodes": ["g2", "0"], "wave": {"kind": "dc", "value": -2.0}},
            {"type": "STL", "name": "X2", "nodes": {"d": "d2", "g": "g2", "s": "0"}, "device": {"preset": "paper"},
             "light_pA": None}]
    res = run_circuit(_custom(els, {"t_stop_s": T, "dt_max_s": T / 2000}))
    sm = _summary(res)
    lu = {e["cell"]: e["v_d"] for e in res["events"] if e["kind"] == "latch_up"}
    assert set(lu) == {"X1", "X2"}
    assert sm["X1.fold_V_LU"] == pytest.approx(3.291, abs=5e-3) and sm["X2.fold_V_LU"] == pytest.approx(3.704, abs=5e-3)
    assert 0 < lu["X1"] - sm["X1.fold_V_LU"] < 0.1 and 0 < lu["X2"] - sm["X2.fold_V_LU"] < 0.1   # ramp lag
    assert lu["X2"] - lu["X1"] > 0.3
    el = {e["name"]: e for e in res["elements"]}
    assert el["X1"]["vgs_V"] == pytest.approx(-1.8) and el["X1"]["device"]["iph_pA"] == pytest.approx(2.63)


def test_gate_driven_by_the_circuit_not_by_the_device_block():
    rate, vmax = 1200.0, 4.2
    T = 2 * vmax / rate
    # device block says V_G = -1.8 V, the circuit drives the gate at -2 V: the circuit wins (warned)
    res = run_circuit(_custom(_load_line_elements(vmax, T, {"preset": "paper", "vg": -1.8}, -2.0), {"t_stop_s": T}))
    assert any("is not used" in w for w in res["warnings"])
    assert _summary(res)["X1.fold_V_LU"] == pytest.approx(3.7037, abs=2e-3)


def test_stochastic_custom_statistics_and_seed():
    rate, vmax = 1200.0, 4.0
    T = 2 * vmax / rate
    pl = _custom(_load_line_elements(vmax, T, PHOTO, -1.8), {"t_stop_s": T, "dt_max_s": T / 2000}, mode="stochastic",
                 stochastic={"n_runs": 5, "seed": 21}, probes=["V(d)", "I(X1.d)", "X1.u"])
    a = run_circuit(pl)
    _check_result(a, "stochastic")
    assert len(a["runs"]) == 5
    assert [e["key"] for e in a["envelopes"]] == ["V(d)", "I(X1.d)", "X1.u"]
    for e in a["envelopes"]:
        n = len(e["t"])
        assert 100 <= n <= 1000 and all(len(e[k]) == n for k in ("mean", "sd", "p05", "p95"))
        p05, p95 = np.asarray(e["p05"], float), np.asarray(e["p95"], float)
        assert np.all(p05 <= p95 + 1e-12)
    dk = {d["key"]: d for d in a["distributions"]}
    assert {"X1.t_first_lu", "X1.vd_first_lu", "end:V(d)", "end:I(X1.d)", "end:X1.u"} <= set(dk)
    lu = np.asarray(dk["X1.vd_first_lu"]["values"], float)
    assert len(lu) == 5 and np.all((lu > 3.1) & (lu < 3.6)) and np.std(lu) > 1e-3
    sm = {s["key"]: s for s in a["summary"]}
    assert sm["X1.p_any_lu"]["value"] == 1.0 and sm["X1.p_latched_end"]["value"] == 0.0
    assert sm["X1.vd_first_lu"]["spread"] > 0
    b = run_circuit(pl)
    assert [e["t"] for e in a["events"]] == [e["t"] for e in b["events"]]
    c = run_circuit(dict(pl, stochastic={"n_runs": 5, "seed": 22}))
    assert [e["t"] for e in a["events"]] != [e["t"] for e in c["events"]]


def test_stochastic_custom_equals_bench_event_level():
    """Same kernel, same seeds, same noise bands/look-ahead: the custom load line reproduces the bench."""
    rate, vmax = 1200.0, 4.0
    T = 2 * vmax / rate
    bench = run_circuit({"bench": "load_line", "mode": "stochastic", "device": PHOTO,
                         "bench_params": {"rate_V_per_s": rate, "v_max_V": vmax}, "stochastic": {"n_runs": 3, "seed": 5}})
    cust = run_circuit(_custom(_load_line_elements(vmax, T, PHOTO, -1.8), {"t_stop_s": T, "dt_max_s": T / 2000},
                               mode="stochastic", stochastic={"n_runs": 3, "seed": 5}, probes=["V(d)"]))
    vb = [round(e["v_d"], 9) for e in bench["events"]]
    vc = [round(e["v_d"], 9) for e in cust["events"]]
    assert vb == vc


def test_per_cell_local_states():
    rate, vmax = 1200.0, 4.0
    T = 2 * vmax / rate
    els = _load_line_elements(vmax, T, PHOTO, -1.8)
    els[-1]["local_state"] = {"mode": "frozen", "action": "gidl", "sigma": 0.2}
    res = run_circuit(_custom(els, {"t_stop_s": T}, mode="stochastic",
                              stochastic={"n_runs": 4, "seed": 3, "carrier_noise": False}, probes=["V(d)", "X1.dphi"]))
    lu = np.asarray([d for d in res["distributions"] if d["key"] == "X1.vd_first_lu"][0]["values"], float)
    assert len(res["runs"]) == 4 and np.std(lu) > 5e-3            # frozen GIDL states spread V_LU
    dphi = [np.asarray(r["signals"][1]["values"], float)[0] for r in res["runs"]]
    assert len(set(np.round(dphi, 9))) == 4
    el = {e["name"]: e for e in res["elements"]}
    assert el["X1"]["local_state"]["mode"] == "frozen"


def test_t_start_save_probes_and_decimation():
    res = run_circuit(_custom([
        {"type": "V", "name": "V1", "nodes": ["a", "0"],
         "wave": {"kind": "pulse", "v1": 0, "v2": 1, "td": 0, "tr": 1e-6, "tf": 1e-6, "pw": 2e-5, "per": 5e-5}},
        {"type": "R", "name": "R1", "nodes": ["a", "b"], "value": 1e3},
        {"type": "C", "name": "C1", "nodes": ["b", "0"], "value": 1e-9},
    ], {"t_stop_s": 1e-2, "t_start_save_s": 2e-3, "dt_max_s": 1e-5, "reltol": 1e-3}, probes=["I(C1)", "V(b)"]))
    t, s = _sig(res)
    assert t[0] == pytest.approx(2e-3) and t[-1] == pytest.approx(1e-2)
    assert len(t) <= 4000 and res["solver_stats"]["steps"] > len(t)
    assert list(s) == ["I(C1)", "V(b)"]
    assert res["probes"] == ["I(C1)", "V(b)"]


def test_no_stl_stochastic_runs_once():
    res = run_circuit(_custom([{"type": "V", "name": "V1", "nodes": ["a", "0"], "wave": {"kind": "dc", "value": 1}},
                               {"type": "R", "name": "R1", "nodes": ["a", "0"], "value": 1e3}], {"t_stop_s": 1e-3},
                              mode="stochastic", stochastic={"n_runs": 10}))
    assert len(res["runs"]) == 1 and any("no random input" in w for w in res["warnings"])


# ---- ERC, limits, feasibility ------------------------------------------------------------------
V1 = {"type": "V", "name": "V1", "nodes": ["a", "0"], "wave": {"kind": "dc", "value": 1.0}}
R1 = {"type": "R", "name": "R1", "nodes": ["a", "b"], "value": 1e3}
RB = {"type": "R", "name": "RB", "nodes": ["b", "0"], "value": 1e3}


def _r(name, a, b, v=1e3):
    return {"type": "R", "name": name, "nodes": [a, b], "value": v}


@pytest.mark.parametrize("elements,match", [
    ([{**V1, "nodes": ["a", "b"]}, R1], "no ground reference"),
    ([V1, R1, {"type": "C", "name": "C1", "nodes": ["b", "c"], "value": 1e-12},
      {"type": "C", "name": "C2", "nodes": ["c", "0"], "value": 1e-12}], "node 'c' has no DC path to ground.*C1, C2"),
    ([V1, {**V1, "name": "V2"}, R1, RB], "V2 is in parallel with V1"),
    ([V1, {**V1, "name": "V2", "nodes": ["a", "b"]}, {**V1, "name": "V3", "nodes": ["b", "0"]}], "in a loop with"),
    ([V1, {**V1, "name": "V2", "nodes": ["b", "b"]}, R1], "short-circuited"),
    ([V1, _r("R1", "a", "0"), {"type": "I", "name": "I1", "nodes": ["0", "x"], "wave": {"kind": "dc", "value": 1e-9}}],
     "current source I1 drives node 'x'.*open circuit"),
    ([V1, R1, RB, {"type": "STL", "name": "X1", "nodes": {"d": "b", "g": "g"}, "device": {}}], "X1: STL terminal.*s not connected"),
    ([V1, R1, RB, {"type": "STL", "name": "X1", "nodes": {"d": "b", "g": "g", "s": "0"}, "device": {}}],
     "node 'g' has no DC path.*X1.g"),
    ([V1, R1, {**RB, "name": "r1"}], "duplicate element name"),
    ([V1, {"type": "L", "name": "L1", "nodes": ["a", "0"], "value": 1e-6}], "unknown element type"),
    ([V1, {**R1, "value": -5}], "R1.*must be >= 0.001"),
    ([V1, {**R1, "name": "R.1"}], "invalid name"),
    ([{**V1, "wave": {"kind": "square"}}, _r("R1", "a", "0")], "unknown wave kind"),
    ([{**V1, "wave": {"kind": "pwl", "t": list(range(2001)), "v": [0.0] * 2001}}, _r("R1", "a", "0")], "limit 2000"),
    ([{**V1, "wave": {"kind": "pulse", "v1": 0, "v2": 1, "tr": 2e-8, "tf": 2e-8, "pw": 2e-8, "per": 1e-7}},
      _r("R1", "a", "0")], "corner points"),
    ([{**V1, "wave": {"kind": "pulse", "v1": 0, "v2": 1, "tr": 1e-6, "tf": 1e-6, "pw": 1e-5, "per": 5e-6}},
      _r("R1", "a", "0")], "exceeds the period"),
    ([{**V1, "wave": {"kind": "sine", "vo": 0, "va": 1, "freq": 1e7}}, _r("R1", "a", "0")], "sine periods"),
    ([V1] + [_r(f"R{i}", "a", "0") for i in range(40)], "limit 40"),
    ([V1] + [_r(f"R{i}", f"n{i}", "0") for i in range(31)], "limit 30"),
    ([V1] + [{"type": "STL", "name": f"X{i}", "nodes": {"d": "a", "g": "0", "s": "0"}, "device": {}} for i in range(9)],
     "limit 8"),
])
def test_erc_and_limits(elements, match):
    with pytest.raises(ValueError, match=match):
        run_circuit(_custom(elements, {"t_stop_s": 1e-3}))


@pytest.mark.parametrize("extra,match", [
    ({"probes": ["V(zz)"]}, "unknown probe 'V\\(zz\\)'.*no node 'zz'"),
    ({"probes": ["I(R9)"]}, "no element 'R9'"),
    ({"tran": None}, "tran is required"),
    ({"tran": {"t_stop_s": 1e-3, "method": "RK4"}}, "tran.method"),
    ({"tran": {"t_stop_s": 1e-3, "t_start_save_s": 2e-3}}, "t_start_save_s"),
    ({"tran": {"t_stop_s": 1e-3, "dt_min_s": 1e-4, "dt_max_s": 1e-5}}, "dt_min_s"),
    ({"mode": "fuzzy"}, "mode"),
    ({"netlist": {"elements": []}}, "non-empty"),
])
def test_invalid_request(extra, match):
    p = _custom([V1, R1, RB], {"t_stop_s": 1e-3})
    p.update(extra)
    with pytest.raises(ValueError, match=match):
        run_circuit(p)


def test_erc_warnings_and_ground_aliases():
    res = run_circuit(_custom([{**V1, "nodes": ["a", "GND"]}, R1, _r("RB", "b", "gnd"), _r("R3", "b", "dangling"),
                               _r("R4", "b", "b")], {"t_stop_s": 1e-3}))
    assert res["nodes"] == ["0", "a", "b", "dangling"]
    assert any("only one connection" in w for w in res["warnings"])
    assert any("R4: both terminals" in w for w in res["warnings"])
    _, s = _sig(res)
    assert np.allclose(s["V(dangling)"], 0.5) and np.allclose(s["I(R3)"], 0.0, atol=1e-15)


def test_feasibility_refusal():
    with pytest.raises(ValueError, match="max_steps.*dt_max"):
        run_circuit(_custom([V1, _r("R1", "a", "0")], {"t_stop_s": 1.0, "dt_max_s": 1e-7}))
    # event-level carrier noise on a slow ramp (0.4 V/s, dark reference device: ~2e5 steps) with a small budget
    T = 2 * 4.0 / 0.4
    with pytest.raises(ValueError, match="event-level carrier noise"):
        run_circuit(_custom(_load_line_elements(4.0, T, {"preset": "paper"}, -2.0), {"t_stop_s": T}, mode="stochastic",
                            stochastic={"n_runs": 1}, solver={"max_steps": 20000}))


# ---- API --------------------------------------------------------------------------------------
def test_api_custom_circuit(client, wait):
    payload = _custom([
        {"type": "V", "name": "V1", "nodes": ["in", "0"],
         "wave": {"kind": "pulse", "v1": 0, "v2": 1, "td": 1e-5, "tr": 1e-7, "tf": 1e-7, "pw": 1e-4, "per": 2e-4}},
        {"type": "R", "name": "R1", "nodes": ["in", "out"], "value": 1e3},
        {"type": "C", "name": "C1", "nodes": ["out", "0"], "value": 1e-8},
    ], {"t_stop_s": 1e-3, "t_start_save_s": 0, "dt_max_s": 1e-6, "dt_min_s": 1e-15, "method": "TRAP", "reltol": 1e-4},
        detect={"i_threshold_A": 1e-8, "hysteresis": 10}, probes=None)
    r = client.post("/api/compute/circuit", params={"wait": 20}, json=payload)
    assert r.status_code == 200, r.text
    st = r.json()
    if st["status"] not in ("done", "error", "cancelled"):
        st = wait(client, st["job_id"], 120)
    assert st["status"] == "done", st
    res = st["result"]
    assert res["bench"] == "custom" and res["nodes"] == ["0", "in", "out"]
    keys = [s["key"] for s in res["runs"][0]["signals"]]
    assert keys == ["V(in)", "V(out)", "I(V1)", "I(R1)", "I(C1)"]
    # ERC errors surface as a job error with the message
    bad = _custom([V1, R1, RB, {"type": "C", "name": "C1", "nodes": ["c", "0"], "value": 1e-12}], {"t_stop_s": 1e-3})
    st = client.post("/api/compute/circuit", params={"wait": 20}, json=bad).json()
    if st["status"] not in ("done", "error", "cancelled"):
        st = wait(client, st["job_id"], 60)
    assert st["status"] == "error" and "no DC path" in st["error"]


def test_cancellation_between_chunks():
    from server.progress import JobCancelled

    def cancel(fraction, message=""):
        if fraction > 0.1:
            raise JobCancelled()
    T = 2 * 4.0 / 1200.0
    with pytest.raises(JobCancelled):
        run_circuit(_custom(_load_line_elements(4.0, T, PHOTO, -1.8), {"t_stop_s": T}, mode="stochastic",
                            stochastic={"n_runs": 3}), cancel)


def test_many_long_generated_waves_estimate_is_bounded():
    """Several 1000-period sines (16 000 PWL points each): the union grid of the estimate is subsampled and the
    total-variation bound keeps the step estimate honest (refused: ~1e6+ steps at dv_max = 2 mV)."""
    els = [{"type": "V", "name": f"V{k}", "nodes": [f"a{k}", "0"],
            "wave": {"kind": "sine", "vo": 0, "va": 1.0, "freq": 1e6 * (1 + 0.04 * k), "td": 0, "theta": 0}} for k in range(6)]
    els += [_r(f"R{k}", f"a{k}", "0") for k in range(6)]
    with pytest.raises(ValueError, match="exceed 2 x solver.max_steps"):
        run_circuit(_custom(els, {"t_stop_s": 1e-3, "reltol": 1e-4}))
