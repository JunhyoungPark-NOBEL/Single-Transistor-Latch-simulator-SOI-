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
            elif e["type"] == "CMP":                       # ideal inputs; output = voltage source out -> ground
                if e["nodes"]["out"] == n:
                    i = s[f"I({e['name']})"]
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
    ([V1, R1, RB, {"type": "MOS", "name": "M1", "nodes": {"d": "b", "g": "gf", "s": "0"}}],
     r"node 'gf' has no DC path to ground: it is connected only through transistor gates / comparator inputs M1\.g\. "
     r"Every node needs a DC path to ground \(resistor, voltage source, diode, STL or MOSFET drain-source, BJT junction "
     r"or comparator output\)"),
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


# ---- current-driven relaxation oscillator (integrate-and-fire, docs/CIRCUIT_SIMULATOR.md §13) ---------
def _osc_elements(i_in, c_par, device=None, vg=-2.0, light=None):
    """The owner's circuit: DC current source into 'out', C_par to ground, STL drain on 'out', source grounded,
    gate at a DC source."""
    return [
        {"type": "I", "name": "Iin", "nodes": ["0", "out"], "wave": {"kind": "dc", "value": i_in}},
        {"type": "C", "name": "Cpar", "nodes": ["out", "0"], "value": c_par},
        {"type": "STL", "name": "X1", "nodes": {"d": "out", "g": "g", "s": "0"}, "device": device or {"preset": "paper"},
         "light_pA": light},
        {"type": "V", "name": "VG", "nodes": ["g", "0"], "wave": {"kind": "dc", "value": vg}},
    ]


# 1 nA into 1 pF, reference device at V_G = -2 V (the schematic template): converged period (BE reltol 1e-5:
# 1.16363 ms, TRAP reltol 1e-5: 1.16381 ms), quasi-static period C (V_LU - V_LD)/I-type integral 1.1093 ms
OSC_T_REF = 1.1637e-3
OSC_T_QS = 1.1093e-3


def test_current_driven_oscillator_inside_the_window():
    res = run_circuit(_custom(_osc_elements(1e-9, 1e-12), {"t_stop_s": 15e-3, "dt_max_s": 5e-6}))
    _check_result(res, "deterministic")
    sm = _summary(res)
    # the DC operating point would be the unstable equilibrium on the NDR branch: 'auto' starts discharged
    assert res["tran"]["initial"] == "auto" and res["tran"]["initial_used"] == "zero"
    assert res["op"]["V(out)"] == pytest.approx(0.0, abs=1e-6) and res["op"]["I(Cpar)"] == pytest.approx(1e-9, rel=1e-6)
    assert any("relaxation oscillator" in w for w in res["warnings"])
    osc = {e["name"]: e for e in res["elements"]}["X1"]["oscillator"]
    assert osc["predicted"] and osc["period_qs_s"] == pytest.approx(OSC_T_QS, rel=0.01) and osc["c_eff_F"] == 1e-12
    # every latch-up is followed by a latch-down: ten sawtooth teeth in 15 ms
    kinds = [e["kind"] for e in res["events"]]
    assert kinds == ["latch_up", "latch_down"] * 10
    t_up = np.array([e["t"] for e in res["events"] if e["kind"] == "latch_up"])
    assert t_up[0] == pytest.approx(1e-12 * sm["X1.fold_V_LU"] / 1e-9, rel=0.02)     # C V_LU / I_in from 0 V
    assert sm["X1.period"] == pytest.approx(OSC_T_REF, rel=0.01)                     # converged within 1 %
    assert 1.0 < sm["X1.period"] / OSC_T_QS < 1.1                                   # + fold lags (4.8 %)
    assert sm["X1.f_osc"] == pytest.approx(1.0 / sm["X1.period"])
    assert sm["X1.isi_cv"] < 1e-3                                                   # deterministic: periodic
    # sawtooth between the folds (+ slow-passage lags): peak just above V_LU, valley just below V_LD
    t, s = _sig(res)
    v = s["V(out)"]
    after = t > t_up[0]
    assert sm["X1.fold_V_LU"] < v[after].max() < sm["X1.fold_V_LU"] + 0.06
    assert sm["X1.fold_V_LD"] - 0.03 < v[after].min() < sm["X1.fold_V_LD"]
    assert sm["X1.vd_lu_mean"] == pytest.approx(v[after].max(), abs=1e-3)           # event timing at the peak
    # the latch-up current spike discharges C_par: I(X1.d) reaches ~µA while I_in = 1 nA
    assert s["I(X1.d)"].max() > 1e-6
    assert _kcl_residual(res) < 1e-5
    fe = res["feasibility"]["estimated_steps_per_run"] / res["solver_stats"]["steps"]
    assert 0.6 < fe < 2.0, fe                                                       # was 0.2 before the oscillator walk


@pytest.mark.parametrize("i_in,t_stop,final", [(5e-12, 2.0, "HRS"), (30e-9, 4e-4, "LRS")])
def test_current_driven_cell_outside_the_window_does_not_oscillate(i_in, t_stop, final):
    res = run_circuit(_custom(_osc_elements(i_in, 1e-12), {"t_stop_s": t_stop}))
    sm = _summary(res)
    osc = {e["name"]: e for e in res["elements"]}["X1"]["oscillator"]
    assert not osc["predicted"]
    assert sm["X1.final_state"] == final and sm["X1.n_latch_down"] == 0 and "X1.period" not in sm
    t, s = _sig(res)
    v = s["V(out)"]
    tail = t > 0.8 * t_stop
    assert np.ptp(v[tail]) < 2e-3                                                    # settled, no sawtooth
    if final == "HRS":      # I_in < I_LU: the load line crosses the HRS below the fold
        assert sm["X1.n_latch_up"] == 0 and 3.5 < v[-1] < sm["X1.fold_V_LU"]
    else:                   # I_in > I_LD: latches once and stays on the LRS (I_LRS(V) = I_in just above V_LD)
        assert sm["X1.n_latch_up"] == 1 and sm["X1.fold_V_LD"] < v[-1] < sm["X1.fold_V_LD"] + 0.05
        assert s["I(X1.d)"][-1] == pytest.approx(i_in, rel=1e-3)


def test_initial_state_modes():
    els = _osc_elements(100e-12, 1e-12)
    # 'op': the DC operating point is the equilibrium on the negative-resistance branch (I_D = I_in, u_i < u < u_j);
    # it is unstable but the deterministic run stays there for many ms (no kick)
    op = run_circuit(_custom(els, {"t_stop_s": 10e-3, "initial": "op"}))
    assert op["tran"]["initial_used"] == "op"
    assert 3.2 < op["op"]["V(out)"] < 3.5 and op["op"]["I(X1.d)"] == pytest.approx(100e-12, rel=1e-4)
    assert _summary(op)["X1.n_latch_up"] == 0
    zero = run_circuit(_custom(els, {"t_stop_s": 10e-3, "initial": "zero"}))
    assert zero["tran"]["initial_used"] == "zero" and zero["op"]["V(out)"] == pytest.approx(0.0, abs=1e-6)
    assert zero["op"]["I(Cpar)"] == pytest.approx(100e-12, rel=1e-6)                 # KCL at t = 0: I_in charges C_par
    # a circuit without a current-biased cell keeps the operating point under 'auto'
    rc = run_circuit(_custom([V1, R1, RB, {"type": "C", "name": "C1", "nodes": ["b", "0"], "value": 1e-9}], {"t_stop_s": 1e-3}))
    assert rc["tran"]["initial_used"] == "op" and rc["op"]["V(b)"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="tran.initial"):
        run_circuit(_custom(els, {"t_stop_s": 1e-3, "initial": "uic"}))


def test_oscillator_stochastic_jitter_and_seed():
    pl = _custom(_osc_elements(1e-9, 1e-12), {"t_stop_s": 9e-3, "dt_max_s": 5e-6}, mode="stochastic",
                 stochastic={"n_runs": 2, "seed": 404}, probes=["V(out)"])
    a = run_circuit(pl)
    sm = {x["key"]: x for x in a["summary"]}
    assert sm["X1.n_latch_up"]["value"] >= 4
    # carrier noise: the latch-up happens at a random V_DS near V_LU -> spike-timing jitter (measured ~2 % CV)
    assert 2e-3 < sm["X1.isi_cv"]["value"] < 0.1
    assert sm["X1.vd_lu_mean"]["spread"] > 2e-3
    assert sm["X1.period"]["value"] == pytest.approx(OSC_T_REF, rel=0.05)
    isi = {d["key"]: d for d in a["distributions"]}["X1.isi"]["values"]
    assert len(isi) >= 6 and np.all(np.asarray(isi) > 0)
    b = run_circuit(pl)
    assert [e["t"] for e in a["events"]] == [e["t"] for e in b["events"]]
    c = run_circuit(dict(pl, stochastic={"n_runs": 2, "seed": 405}))
    assert [e["t"] for e in a["events"]] != [e["t"] for e in c["events"]]


def test_light_shifts_the_oscillation_frequency():
    """Illumination calibration device (V_G = -1.8 V): the photocurrent lowers V_LU, the window shrinks and the
    oscillator fires faster (light-to-frequency conversion)."""
    f = {}
    for iph in (0.0, 2.63):
        dev = {"preset": "photo", "vg": -1.8, "light": {"mode": "iph", "iph_pA": iph}}
        res = run_circuit(_custom(_osc_elements(1e-9, 1e-12, dev, -1.8), {"t_stop_s": 12e-3, "dt_max_s": 5e-6}))
        f[iph] = _summary(res)["X1.f_osc"]
    assert f[2.63] > 1.4 * f[0.0]


def test_oscillator_feasibility_refusal_names_the_cycles():
    with pytest.raises(ValueError, match="relaxation oscillation: ~\\d+ predicted"):
        run_circuit(_custom(_osc_elements(1e-9, 1e-12), {"t_stop_s": 10.0}))


@pytest.mark.slow
def test_oscillator_period_converges_with_the_time_step():
    """Default BE step control vs much finer steps (BE reltol 1e-5, TRAP): period within 0.3 %, peak within 5 mV,
    no numerical overshoot (the default peak lies below the converged one)."""
    out = {}
    for key, tran in (("default", {}), ("be_fine", {"reltol": 1e-5}), ("trap_fine", {"method": "TRAP", "reltol": 1e-5})):
        res = run_circuit(_custom(_osc_elements(1e-9, 1e-12), dict({"t_stop_s": 8e-3}, **tran)))
        sm = _summary(res)
        t, s = _sig(res)
        out[key] = (sm["X1.period"], s["V(out)"][t > 4e-3].max(), s["V(out)"][t > 4e-3].min())
    for ref in ("be_fine", "trap_fine"):
        assert out["default"][0] == pytest.approx(out[ref][0], rel=3e-3)
        assert -5e-3 < out["default"][1] - out[ref][1] <= 1e-4
        assert abs(out["default"][2] - out[ref][2]) < 2e-3
    assert out["be_fine"][0] == pytest.approx(OSC_T_REF, rel=1e-3)


# ---- comparator (CMP) and the p-bit (drain pulses, source resistor, comparator) -----------------------
def _cmp(name="CMP1", inp="a", out="q", v_ref=0.5, **kw):
    return dict({"type": "CMP", "name": name, "nodes": {"in": inp, "out": out}, "v_ref": v_ref}, **kw)


def test_comparator_switching_hysteresis_and_sign():
    tri = {"kind": "pwl", "t": [0, 1e-3, 2e-3], "v": [0, 1, 0]}
    els = [{"type": "V", "name": "V1", "nodes": ["a", "0"], "wave": tri}, _r("RA", "a", "0"),
           _cmp(v_ref=0.5, hysteresis=0.2), _r("RQ", "q", "0", 1e3)]
    res = run_circuit(_custom(els, {"t_stop_s": 2e-3, "dt_max_s": 2e-6}))
    t, s = _sig(res)
    assert {"V(q)", "I(CMP1)", "CMP1.bit"} <= set(s)
    ev = [(e["kind"], e["t"]) for e in res["events"] if e["cell"] == "CMP1"]
    assert [k for k, _ in ev] == ["cmp_rise", "cmp_fall"]
    assert ev[0][1] == pytest.approx(0.6e-3, abs=4e-6) and ev[1][1] == pytest.approx(2e-3 - 0.4e-3, abs=4e-6)  # 0.6 V up, 0.4 V down
    hi = (t > 0.62e-3) & (t < 1.58e-3)
    assert np.allclose(s["V(q)"][hi], 1.0, atol=1e-6) and np.all(s["CMP1.bit"][hi] == 1)
    assert np.allclose(s["V(q)"][t < 0.58e-3], 0.0, atol=1e-6)
    # the output is a voltage source delivering power into RQ: SPICE sign (negative), and KCL at q
    assert np.allclose(s["I(CMP1)"][hi], -1e-3, rtol=1e-6) and np.allclose(s["I(RQ)"][hi], 1e-3, rtol=1e-6)
    sm = _summary(res)
    assert sm["CMP1.n_rise"] == 1 and sm["CMP1.duty"] == pytest.approx(0.5, abs=5e-3)      # high 0.6 … 1.6 ms of 2 ms
    el = {e["name"]: e for e in res["elements"]}["CMP1"]
    assert el["nodes"] == {"in": "a", "inm": "0", "out": "q"} and el["v_ref"] == 0.5 and el["hysteresis"] == 0.2
    assert res["comparators"][0]["window_source"] is None                       # no pulse source: no windows


def test_comparator_differential_input_and_inverted_levels():
    els = [{"type": "V", "name": "V1", "nodes": ["a", "0"], "wave": {"kind": "pwl", "t": [0, 1e-3], "v": [0, 2]}},
           {"type": "V", "name": "V2", "nodes": ["b", "0"], "wave": {"kind": "dc", "value": 1.0}},
           _r("RA", "a", "0"), _r("RB", "b", "0"),
           {"type": "CMP", "name": "U1", "nodes": ["a", "b", "q"], "v_ref": 0.2, "v_high": -1.0, "v_low": 2.5}]
    res = run_circuit(_custom(els, {"t_stop_s": 1e-3, "dt_max_s": 2e-6}))
    t, s = _sig(res)
    # V(a) - V(b) > 0.2 V from V(a) = 1.2 V (t = 0.6 ms): output -1 V ("high" state), 2.5 V before
    assert np.allclose(s["V(q)"][t < 0.58e-3], 2.5, atol=1e-6) and np.allclose(s["V(q)"][t > 0.62e-3], -1.0, atol=1e-6)
    assert np.all(s["U1.bit"][t > 0.62e-3] == 1) and np.all(s["U1.bit"][t < 0.58e-3] == 0)


@pytest.mark.parametrize("elements,match", [
    ([V1, _r("R1", "a", "0"), _cmp(out="a")], "comparator output .*must not be driven|output of comparator CMP1"),
    ([V1, _r("R1", "a", "0"), _cmp(), {"type": "V", "name": "V2", "nodes": ["q", "0"], "wave": {"kind": "dc", "value": 1}}],
     "must not be driven by another source"),
    ([V1, _r("R1", "a", "0"), _cmp(out="0")], "output cannot be ground"),
    ([V1, _r("R1", "a", "0"), _cmp(inp="x")], "node 'x' has no DC path.*CMP1.in"),
    ([V1, _r("R1", "a", "0"), _cmp(v_high=1.0, v_low=1.0)], "v_high and v_low must differ"),
    ([V1, _r("R1", "a", "0"), {"type": "CMP", "name": "CMP1", "nodes": {"in": "a"}, "v_ref": 0.1}], "out not connected"),
])
def test_comparator_erc(elements, match):
    with pytest.raises(ValueError, match=match):
        run_circuit(_custom(elements, {"t_stop_s": 1e-3}))


def _pbit_elements(amp, n=5, rs=100e3, v_ref=0.1):
    return [
        {"type": "V", "name": "Vp", "nodes": ["d", "0"],
         "wave": {"kind": "pulse", "v1": 0, "v2": amp, "td": 0, "tr": 20e-6, "tf": 20e-6, "pw": 200e-6, "per": 1e-3, "ncycles": n}},
        {"type": "STL", "name": "X1", "nodes": {"d": "d", "g": "g", "s": "s"}, "device": {"preset": "paper"}, "light_pA": None},
        {"type": "V", "name": "VG", "nodes": ["g", "0"], "wave": {"kind": "dc", "value": -2.0}},
        _r("RS", "s", "0", rs),
        _cmp(inp="s", out="q", v_ref=v_ref),
    ]


def test_source_degenerated_stl_converges():
    """A 100 kΩ source resistor moves V_S by ~0.4 V when the cell latches: V_GS = v_g - v_s enters the element
    Jacobian (d/dV_GS columns); without them Newton failed at the latch-up (time-step underflow)."""
    res = run_circuit(_custom(_pbit_elements(3.75, n=2), {"t_stop_s": 2e-3}))
    assert not any("underflow" in w for w in res["warnings"]) and "truncated_runs" not in _summary(res)
    t, s = _sig(res)
    assert 0.40 < s["V(s)"].max() < 0.48                                             # R_S I_LRS
    assert np.allclose(s["V(s)"], 1e5 * s["I(X1.d)"], rtol=1e-4, atol=1e-9)
    assert _kcl_residual(res) < 1e-5
    assert _summary(res)["X1.n_latch_up"] == 2


@pytest.mark.parametrize("amp,p", [(3.69, 0.0), (3.72, 1.0)])
def test_pbit_deterministic_fires_never_or_always(amp, p):
    res = run_circuit(_custom(_pbit_elements(amp), {"t_stop_s": 5e-3}))
    c = res["comparators"][0]
    assert c["window_source"] == "Vp" and len(c["t_windows"]) == 5 and c["n_bits"] == 5
    assert c["p_fire"] == p and c["bits"] == [[int(p)] * 5]
    sm = _summary(res)
    assert sm["CMP1.p_fire"] == p and sm["X1.n_latch_up"] == 5 * p


def test_pbit_stochastic_random_firing_seeded():
    pl = _custom(_pbit_elements(3.69, n=10), {"t_stop_s": 10e-3}, mode="stochastic", stochastic={"n_runs": 4, "seed": 17},
                 probes=["V(s)", "CMP1.bit"])
    a = run_circuit(pl)
    c = a["comparators"][0]
    B = np.array(c["bits"], float)
    assert B.shape == (4, 10) and c["n_bits"] == 40
    assert 0.15 < c["p_fire"] < 0.85                                                  # measured 0.50 (200 bits)
    assert len(c["p_fire_window"]) == 10 and len(c["p_fire_run"]) == 4
    # every fired pulse is a latch-up of the cell (the comparator reads the LRS current through R_S)
    n_lu = sum(1 for e in a["events"] if e["kind"] == "latch_up")
    assert n_lu == int(B.sum())
    assert any(d["key"] == "CMP1.p_fire_run" for d in a["distributions"])
    b = run_circuit(pl)
    assert b["comparators"][0]["bits"] == c["bits"]
    other = run_circuit(dict(pl, stochastic={"n_runs": 4, "seed": 18}))
    assert other["comparators"][0]["bits"] != c["bits"]


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


# ---- mixed circuits (STL + MOS/D/BJT): the pre-run V_GS estimate ignores the transistor network (P1-13 stop-gap) ----
def _stl_echo(res, name="X1"):
    return {e["name"]: e for e in res["elements"]}[name]


def test_mixed_circuit_skips_the_estimate_based_oscillator_prediction():
    els = _osc_elements(1e-9, 1e-12) + [
        {"type": "V", "name": "VGM", "nodes": ["gm", "0"], "wave": {"kind": "dc", "value": 0.0}},
        {"type": "MOS", "name": "M1", "nodes": {"d": "out", "g": "gm", "s": "0"}}]
    res = run_circuit(_custom(els, {"t_stop_s": 3e-3, "dt_max_s": 5e-6}))
    x1 = _stl_echo(res)
    assert x1["oscillator"] is None and not any("relaxation oscillator" in w for w in res["warnings"])
    assert x1["vgs_estimate"]["transistors_ignored"] == ["M1"] and x1["vgs_estimate"]["reliable"] is False
    assert x1["vgs_V"] == pytest.approx(-2.0)
    # the same circuit without the MOSFET keeps the prediction
    ref = run_circuit(_custom(_osc_elements(1e-9, 1e-12), {"t_stop_s": 3e-3, "dt_max_s": 5e-6}))
    assert _stl_echo(ref)["oscillator"] is not None and _stl_echo(ref)["vgs_estimate"]["reliable"] is True
    assert any("X1: high-impedance drive" in w or "relaxation oscillator" in w for w in ref["warnings"])


def test_mixed_circuit_gate_clamped_by_a_diode_has_no_premature_no_latch_warning():
    # the linear estimate (diode off) puts the gate at +0.5 V (channel on, no latch window); the diode to the
    # -2.6 V rail actually holds it near -2.04 V, where the window exists
    els = [{"type": "V", "name": "VS", "nodes": ["s1", "0"], "wave": {"kind": "dc", "value": 0.5}},
           _r("RG", "s1", "g", 1e5),
           {"type": "D", "name": "D1", "nodes": {"a": "g", "k": "m"}},
           {"type": "V", "name": "VM", "nodes": ["m", "0"], "wave": {"kind": "dc", "value": -2.6}},
           {"type": "V", "name": "VD", "nodes": ["vd", "0"], "wave": {"kind": "dc", "value": 1.0}},
           _r("RD", "vd", "d"),
           {"type": "STL", "name": "X1", "nodes": {"d": "d", "g": "g", "s": "0"}, "device": {"preset": "paper"}}]
    res = run_circuit(_custom(els, {"t_stop_s": 1e-6}))
    x1 = _stl_echo(res)
    assert res["op"]["V(g)"] == pytest.approx(-2.04, abs=0.03)
    assert x1["vgs_V"] == pytest.approx(0.5, abs=1e-3)                  # the pre-run estimate, flagged as such
    assert x1["vgs_estimate"] == {"method": "linear DC network before the run (C open, STL drain-source off)",
                                  "transistors_ignored": ["D1"], "reliable": False}
    assert x1["latch_window"] is None
    assert not any("no latch window" in w for w in res["warnings"])
    assert any("V_GS = 0.5 V comes from the circuit (linear estimate without the transistor network (D1))" in w
               for w in res["warnings"])


def test_mixed_circuit_stochastic_warns_about_the_noise_band():
    T = 2 * 4.0 / 1200.0
    els = _load_line_elements(4.0, T, {"preset": "paper"}, -2.0, g="g") + [
        {"type": "D", "name": "DG", "nodes": {"a": "g", "k": "0"}}]
    els[3]["nodes"] = ["g", "0"]
    res = run_circuit(_custom(els, {"t_stop_s": T, "dt_max_s": T / 2000}, mode="stochastic",
                              stochastic={"n_runs": 1, "seed": 4}))
    assert any("X1: the carrier-noise band near the folds was set from the linear V_GS estimate without the "
               "transistor network (DG)" in w for w in res["warnings"])
    assert _stl_echo(res)["latch_window"] is True                      # V_GS = -2 V: the window is known
