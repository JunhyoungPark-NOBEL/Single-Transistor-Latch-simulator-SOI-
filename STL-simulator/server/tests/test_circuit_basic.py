"""Educational semiconductor stamps: device trends and observable circuit KCL."""
import numpy as np
import pytest

from server.compute.circuit import run_circuit
from server.compute.circuit.basic import pack, evaluate


def mos_current(vg=1.0, **changes):
    model = dict(polarity="nmos", L_um=1, W_um=10, Vth_V=0.5, SS_mV_dec=80,
                 k_uA_V2=100, lambda_per_V=0.02)
    model.update(changes)
    return evaluate(pack("MOS", [1, 2, 0], model), np.array([1.0, vg, 0.0]))[0][0]


def test_mos_geometry_threshold_and_subthreshold_swing():
    baseline = mos_current()
    assert mos_current(W_um=20) == pytest.approx(2 * baseline)
    assert mos_current(L_um=2) == pytest.approx(baseline / 2)
    assert mos_current(Vth_V=0.7) < baseline
    # In weak inversion a gate change equal to SS increases current by one decade.
    assert mos_current(vg=-0.32) / mos_current(vg=-0.4) == pytest.approx(10, rel=1e-4)
    assert mos_current(vg=-0.24, SS_mV_dec=160) / mos_current(vg=-0.4, SS_mV_dec=160) == pytest.approx(10, rel=0.005)
    assert mos_current(vg=-0.4, SS_mV_dec=160) > mos_current(vg=-0.4)


def _source(name, node, value, ramp=False):
    return dict(type="V", name=name, nodes=[node, "0"],
                wave=dict(kind="pwl", t=[0, 1e-5, 2e-5], v=[0, value, 0]) if ramp else dict(kind="dc", value=value))


def test_ordinary_300k_mos_refuses_subthermal_swing():
    elements = [_source("VDD", "d", 1), _source("VG", "g", 0.5),
                dict(type="MOS", name="M1", nodes=dict(d="d", g="g", s="0"),
                     model=dict(SS_mV_dec=10))]
    with pytest.raises(ValueError, match="SS_mV_dec"):
        run_circuit(dict(bench="custom", mode="deterministic", netlist=dict(elements=elements),
                         tran=dict(t_stop_s=1e-6)))


@pytest.mark.parametrize("kind,polarity", [("MOS", "nmos"), ("MOS", "pmos"), ("D", ""), ("BJT", "npn"), ("BJT", "pnp")])
def test_nonlinear_transient_and_kcl(kind, polarity):
    sign = -1 if polarity in ("pmos", "pnp") else 1
    elements = [_source("VDD", "supply", sign * (1 if kind == "D" else 3)),
                dict(type="R", name="R1", nodes=["supply", "out"], value=1000)]
    if kind == "MOS":
        elements += [_source("VG", "gate", sign * 2, True),
                     dict(type="MOS", name="M1", nodes=dict(d="out", g="gate", s="0"), model=dict(polarity=polarity))]
        current, terminals = "I(M1.d)", ["I(M1.d)", "I(M1.g)", "I(M1.s)"]
    elif kind == "D":
        elements[0] = _source("VDD", "supply", 1, True)
        elements += [dict(type="D", name="D1", nodes=dict(a="out", k="0"))]
        current, terminals = "I(D1)", []
    else:
        elements += [_source("VB", "base", sign * 0.7, True),
                     dict(type="BJT", name="Q1", nodes=dict(c="out", b="base", e="0"), model=dict(polarity=polarity))]
        current, terminals = "I(Q1.c)", ["I(Q1.c)", "I(Q1.b)", "I(Q1.e)"]
    result = run_circuit(dict(bench="custom", mode="deterministic", netlist=dict(elements=elements),
                              tran=dict(t_stop_s=2e-5, dt_max_s=1e-7)))
    run = result["runs"][0]
    sig = {s["key"]: np.array(s["values"]) for s in run["signals"]}
    assert run["t"][-1] == pytest.approx(2e-5)
    assert all(np.all(np.isfinite(v)) for v in sig.values())
    assert np.max(np.abs(sig[current])) > 1e-4
    assert np.max(np.abs(sig[current] - sig["I(R1)"])) < 2e-9
    assert np.max(np.abs(sig["I(VDD)"] + sig["I(R1)"])) < 2e-9
    if terminals:
        assert np.max(np.abs(sum(sig[key] for key in terminals))) < 2e-9
    assert not any("truncat" in warning or "converge" in warning for warning in result["warnings"])


def test_bjt_forward_gain_and_reverse_operation():
    model = dict(polarity="npn", Is_A=1e-15, beta_F=120, beta_R=3)
    row = pack("BJT", [1, 2, 0], model)
    forward, _ = evaluate(row, np.array([2.0, 0.65, 0.0]))
    reverse, _ = evaluate(row, np.array([0.0, 0.65, 2.0]))
    assert forward[0] / forward[1] == pytest.approx(120, rel=1e-6)
    assert reverse[2] / reverse[1] == pytest.approx(3, rel=1e-6)
