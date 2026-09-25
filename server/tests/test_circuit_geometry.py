"""Per-cell geometry must survive compilation and the transient MNA path."""
from __future__ import annotations

import numpy as np
import pytest

from server import params
from server.compute.circuit import run_circuit
from server.compute.circuit.element import N_EV, stl_eval
from server.compute.circuit.netlist import Netlist
from server.compute.circuit.mna import GMIN, ST_DONE
from server.compute.circuit.sim import SolverConfig, simulate
from server.engine_bridge import MODEL
from server.geometry_model import constants_from_p, pack_p


def _p(**geometry):
    return np.asarray(params.build_p({"preset": "paper", "geometry": geometry}), float)


def _evaluate(p, u=0.4, r=2.0):
    out = np.empty(N_EV)
    assert stl_eval(u, r, p, MODEL.na, MODEL.vbi, MODEL.rg, MODEL.fg, MODEL.table, out)
    assert np.all(np.isfinite(out))
    return out


def test_compilation_preserves_reference_and_independent_geometry_fields():
    reference = _p()
    net = Netlist()
    net.add_STL("X1", "d1", "g", "0", reference)
    assert net.compile()["P"].shape == (1, 26)
    wider = _p(W_nm=400)
    doped = _p(Nbody_cm3=2 * params.NA_CM3)
    net.add_STL("X2", "d2", "g", "0", wider)
    net.add_STL("X3", "d3", "g", "0", doped)
    compiled = net.compile()["P"]
    assert compiled.shape[0] == 3 and compiled.shape[1] > 32
    np.testing.assert_array_equal(compiled[0, :26], reference)
    np.testing.assert_allclose(compiled[1], pack_p(wider), rtol=0, atol=0)
    np.testing.assert_allclose(compiled[2], pack_p(doped), rtol=0, atol=0)
    assert constants_from_p(compiled[2])[-2] == 2 * params.NA_CM3
    assert not np.array_equal(compiled[0, 35:], compiled[2, 35:])
    # A second device's lookup table must not change an already-evaluated cell.
    original = _evaluate(compiled[0])
    _evaluate(compiled[2])
    np.testing.assert_array_equal(_evaluate(compiled[0]), original)
    np.testing.assert_allclose(original[:7], _evaluate(reference)[:7], rtol=2e-12, atol=1e-25)


@pytest.mark.parametrize("u,r", [(0.4, 2.0), (-0.1, 1.0), (0.5, -0.1)])
def test_width_scales_currents_charge_and_reverse_bias_extensions(u, r):
    reference = _evaluate(pack_p(_p(), force=True), u, r)
    wider = _evaluate(pack_p(_p(W_nm=400)), u, r)
    assert wider[0] == pytest.approx(reference[0], rel=1e-12, abs=1e-14)
    np.testing.assert_allclose(wider[1:7], 2 * reference[1:7], rtol=2e-12, atol=1e-25)


def test_mixed_width_transient_preserves_terminal_voltage_and_current_ratio():
    net = Netlist()
    net.t_end = 1e-4
    net.add_V("VG", "g", "0", [0.0], [-2.0])
    drive = net.add_V("VD", "d", "0", [0.0, net.t_end / 2, net.t_end], [0.0, 2.0, 2.0])
    net.add_STL("X1", "d", "g", "0", _p())
    net.add_STL("X2", "d", "g", "0", _p(W_nm=400))
    compiled = net.compile()
    cfg = SolverConfig(dt_max=1e-5, dt_rec=1e-6, dv_rec=0.02)
    result = simulate(compiled, cfg, compiled["P"], net.t_end, drive, 0)
    assert result.status == ST_DONE and result.t_reached == pytest.approx(net.t_end)
    assert result.steps > 10
    first = 1 + compiled["n_nodes"] - 1 + compiled["nV"]
    x1 = result.rec[:, first:first + 7]
    x2 = result.rec[:, first + 7:first + 14]
    # Both cells see the same drive and proportional generation/loss/storage.
    np.testing.assert_allclose(x2[:, :2], x1[:, :2], rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(x2[:, 2:5], 2 * x1[:, 2:5], rtol=2e-4, atol=1e-21)
    drain_source_current = result.rec[:, 1 + compiled["n_nodes"] - 1 + 1]
    drain_voltage = result.rec[:, net.nodes.index("d")]
    np.testing.assert_allclose(drain_source_current + x1[:, 3] + x2[:, 3] + GMIN * drain_voltage,
                               0, atol=1e-21)


def test_raw_geometry_vector_is_packed_by_bench_driver():
    net = Netlist()
    net.t_end = 1e-6
    net.add_V("VG", "g", "0", [0.0], [-2.0])
    drive = net.add_V("VD", "d", "0", [0.0], [1.0])
    raw = _p(W_nm=400)
    net.add_STL("X1", "d", "g", "0", raw)
    compiled = net.compile()
    cfg = SolverConfig(dt_max=1e-6, dt_rec=1e-7)
    result = simulate(compiled, cfg, raw[None, :], net.t_end, drive, 0)
    assert result.status == ST_DONE and result.t_reached == pytest.approx(net.t_end)
    assert np.all(np.isfinite(result.rec))


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("device", [
    {"preset": "paper", "geometry": {"W_nm": 400}},
    {"preset": "paper", "vbg": 1.0},
])
def test_geometry_noise_is_refused_before_running_reference_noise_kernel(custom, device):
    payload = {"mode": "stochastic", "device": device}
    if custom:
        payload.update(bench="custom", tran={"t_stop_s": 1e-6}, netlist={"elements": [
            {"type": "V", "name": "VG", "nodes": ["g", "0"], "wave": {"kind": "dc", "value": -2}},
            {"type": "V", "name": "VD", "nodes": ["d", "0"], "wave": {"kind": "dc", "value": 1}},
            {"type": "STL", "name": "X1", "nodes": {"d": "d", "g": "g", "s": "0"}, "device": device},
        ]})
    with pytest.raises(ValueError, match="geometry-stochastic-unavailable"):
        run_circuit(payload)


def test_reference_cell_charge_is_bit_identical_to_the_legacy_expression():
    """A 26-entry (reference) cell must keep the pre-geometry body-charge expression
    C_ox (psi - V_GS) + (z13 - C_ox u) + q N_A A L_n bit for bit: the reassociated form
    C_ox psi - C_ox V_GS differs in the last bit at ~30 % of states, which shifts accepted time
    steps and changes every seeded stochastic realisation relative to earlier releases."""
    from numba import njit
    from server.compute.circuit.element import AREA, COX, QE, VT, components
    from server.compute.circuit.stochastic import _state_charge

    @njit
    def legacy_q(u, r, p, na, vbi, rg, fg, table):
        z = components(u, r, p, na, vbi, rg, fg, table)
        psi = u - VT * np.log1p(z[10])
        return COX * (psi - p[11]) + (z[13] - COX * u) + QE * na * AREA * z[11]

    p = _p()
    args = (MODEL.na, MODEL.vbi, MODEL.rg, MODEL.fg, MODEL.table)
    out = np.empty(N_EV)
    checked = 0
    for u in np.linspace(0.0, 1.0, 21):
        for r in np.linspace(0.0, 10.0, 21):
            if not stl_eval(u, r, p, *args, out):
                continue
            checked += 1
            assert out[3] == legacy_q(u, r, p, *args), (u, r)
            z = components(u, r, p, *args)
            assert _state_charge(z, u, p) == legacy_q(u, r, p, *args), (u, r)
    assert checked > 200
