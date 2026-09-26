"""Simple Model circuit dispatch, contact dynamics and explicit capability boundaries."""
from __future__ import annotations

import numpy as np
import pytest

from server import params
from server.compute.circuit import run_circuit
from server.compute.circuit.element import stl_eval, N_EV
from server.compute.circuit.netlist import Netlist
from server.compute.circuit.sim import SolverConfig, simulate
from server.compute.circuit.mna import ST_DONE, GMIN
from server.compute.circuit.stochastic import _state_charge, branch_profile
from server.engine_bridge import MODEL
from server.simple_model import components, effective_parameters, evaluate, is_simple


def _p(**simple):
    return np.asarray(params.build_p(dict(model="simple", vg=-3, simple=simple)), dtype=float)


def test_simple_element_uses_complete_lumped_charge_and_reservoir_potential():
    p = _p()
    a = np.empty(N_EV)
    b = np.empty(N_EV)
    assert stl_eval(.24, 1.1, p, MODEL.na, MODEL.vbi, MODEL.rg, MODEL.fg, MODEL.table, a)
    assert evaluate(.24, 1.1, p, b)
    np.testing.assert_array_equal(a, b)
    z = components(.24, 1.1, p)
    assert _state_charge(z, .24, p) == a[3]
    beta, tau, cb, resistance, _, _, _, cg, cbg, cs, bias = effective_parameters(p)
    assert a[8] == pytest.approx(.24 + a[1] * resistance)
    assert a[3] == pytest.approx(cb * (a[8] - bias))
    assert a[9] == pytest.approx(cg * (p[11] - a[8]))
    assert a[10] == pytest.approx(cbg * (p[32] - a[8]))
    assert cb == pytest.approx(cg + cbg + cs)
    assert z[5] == pytest.approx(a[3] / tau)
    assert z[6] == pytest.approx(z[3] / beta)


def _net(cap=1e-12, mixed=False):
    n = Netlist(); n.t_end = 1e-3
    n.add_V('VG', 'g', '0', [0], [-3])
    n.add_V('VD', 'd', '0', [0], [1])
    n.add_V('VBG', 'bg', '0', [0], [0])
    n.add_V('VBdrive', 'bdrive', '0', [0, 1e-6, n.t_end], [0, .15, .15])
    n.add_R('Rb', 'bdrive', 'b', 1e9)
    n.add_C('Cb', 'b', '0', cap)
    p = _p(tau_body_s=1e-3, r_lrs_ref_ohm=0., is_ref_A=1e-25,
           btbt_scale=0., gamma_fg=0., gamma_bg=0.)
    n.add_STL('X1', 'd', 'g', '0', p, bg='bg', b='b')
    if mixed:
        n.add_STL('X2', 'd', 'g', '0', params.build_p({'preset': 'paper', 'vg': -3}))
    return n


def _run(n):
    c = n.compile()
    cfg = SolverConfig(dt_max=n.t_end / 200, dt_rec=0, dv_rec=100,
                       lte_v=7e-3, lte_v_abs=7e-6)
    o = simulate(c, cfg, c['P'], n.t_end, -1, 0)
    assert o.status == ST_DONE, o.warnings
    return c, o.rec


def test_simple_body_rc_changes_internal_dynamics_and_conserves_charge():
    n = _net(); c, r = _run(n)
    fast_n = _net(0); _, fast = _run(fast_n)
    base = c['n_nodes'] + c['nV']
    bcol = n.nodes.index('b')
    np.testing.assert_allclose(r[:, base], r[:, bcol], atol=2e-8)
    t = .3e-3
    slow = np.interp(t, r[:, 0], r[:, bcol])
    quick = np.interp(t, fast[:, 0], fast[:, fast_n.nodes.index('b')])
    assert quick - slow > .09
    assert slow == pytest.approx(.15 * (1 - np.exp(-(t - .5e-6) / 1e-3)), abs=2e-3)
    port = base + 7 + c['nC']
    dqdt = np.diff(r[:, base + 2]) / np.diff(r[:, 0])
    assert np.max(np.abs(dqdt - r[1:, base + 4] - r[1:, port + 2])) < 2e-15
    ir = (r[:, bcol] - r[:, n.nodes.index('bdrive')]) / 1e9
    assert np.max(np.abs(ir + r[:, base + 7] + r[:, port + 2] + GMIN * r[:, bcol])) < 2e-15


def test_mixed_detailed_simple_circuit_has_independent_packed_models():
    n = _net(mixed=True)
    c = n.compile()
    assert c['P'].shape == (2, 3038)
    assert is_simple(c['P'][0]) and not is_simple(c['P'][1])
    _, rows = _run(n)
    assert np.isfinite(rows).all()


def _payload():
    return dict(bench='custom', mode='deterministic',
                tran=dict(t_stop_s=2e-5, dt_max_s=1e-6, method='BE', initial='op'),
                netlist=dict(elements=[
                    dict(type='V', name='VD', nodes=['d', '0'], wave=dict(kind='dc', value=1)),
                    dict(type='V', name='VG', nodes=['g', '0'], wave=dict(kind='pwl', t=[0, 1e-5, 2e-5], v=[-3, -2.9, -3])),
                    dict(type='V', name='VBG', nodes=['bg', '0'], wave=dict(kind='pwl', t=[0, 1e-5, 2e-5], v=[0, .2, 0])),
                    dict(type='V', name='VB', nodes=['b', '0'], wave=dict(kind='dc', value=.1)),
                    dict(type='STL', name='X1', nodes=dict(d='d', g='g', s='0', bg='bg', b='b'),
                         device=dict(model='simple', vg=-3)),
                ]), probes=['X1.vb', 'X1.vbody', 'I(X1.d)', 'I(X1.g)', 'I(X1.bg)', 'I(X1.b)', 'I(X1.s)'])


def test_simple_custom_body_probe_echo_and_modulated_gate_charges():
    payload = _payload()
    out = run_circuit(payload)
    signals = {s['key']: np.asarray(s['values']) for s in out['runs'][0]['signals']}
    resistance = effective_parameters(_p())[3]
    np.testing.assert_allclose(signals['X1.vb'], signals['X1.vbody'] + signals['I(X1.d)'] * resistance,
                               atol=2e-7, rtol=2e-6)
    assert np.max(np.abs(signals['I(X1.g)'])) > 0
    assert np.max(np.abs(signals['I(X1.bg)'])) > 0
    assert np.max(np.abs(sum(signals[f'I(X1.{pin})'] for pin in ('d', 's', 'g', 'bg', 'b')))) < 2e-16
    echo = out['elements'][-1]
    assert echo['device']['model'] == 'simple'
    assert echo['device']['simple_effective']['beta'] == pytest.approx(effective_parameters(_p())[0])
    assert echo['device']['geometry_model']['validated'] is False
    assert 'reservoir' in echo['body_potential_model']
    assert 'reference calibration' not in echo['device']['label']


def test_simple_profile_does_not_claim_detailed_carrier_noise():
    profile = branch_profile(_p(), 201)
    assert profile['noise_supported'] is False
    for name in ('HRS', 'LRS'):
        assert np.all(profile[name]['rate'] == 0)
        assert np.all(np.isinf(profile[name]['z']))


def test_body_clamp_auto_initialization_asks_loaded_solver_before_using_free_body_folds(monkeypatch):
    from server.compute.circuit import sim
    calls = []
    original = sim.simulate

    def observe(*args, **kwargs):
        calls.append(kwargs.get('initial'))
        return original(*args, **kwargs)

    monkeypatch.setattr(sim, 'simulate', observe)
    p = _payload()
    els = p['netlist']['elements']
    els[0] = dict(type='I', name='IDrive', nodes=['0', 'd'], wave=dict(kind='dc', value=1e-9))
    els[1]['wave'] = dict(kind='dc', value=-3)
    els[2]['wave'] = dict(kind='dc', value=0)
    els[3]['wave'] = dict(kind='dc', value=.4)
    els.append(dict(type='C', name='CD', nodes=['d', '0'], value=1e-12))
    p['tran']['initial'] = 'auto'
    p['probes'] = ['V(d)', 'X1.u']
    out = run_circuit(p)
    sig = {s['key']: np.asarray(s['values']) for s in out['runs'][0]['signals']}
    # The loaded solver must decide whether DC converges. A real DC failure may
    # still fall back to discharged capacitors; floating-body folds cannot force it.
    assert calls == ['auto']
    np.testing.assert_allclose(sig['X1.u'], .4, atol=2e-7)
    assert not any('current-biased above the HRS' in w for w in out['warnings'])
    echo = next(e for e in out['elements'] if e['name'] == 'X1')
    assert echo['oscillator']['predicted'] is None
    assert echo['oscillator']['period_qs_s'] is None


@pytest.mark.parametrize('bench', ['custom', 'load_line'])
@pytest.mark.parametrize('mode,method,error', [('stochastic', 'BE', 'simple-stochastic-unavailable'),
                                             ('deterministic', 'TRAP', 'simple-method-unavailable')])
def test_simple_unsupported_modes_fail_before_evaluation(bench, mode, method, error):
    if bench == 'custom':
        p = _payload()
        p['netlist']['elements'] = [p['netlist']['elements'][0], p['netlist']['elements'][1], p['netlist']['elements'][-1]]
        p['netlist']['elements'][-1]['nodes'] = dict(d='d', g='g', s='0')
        p['tran']['method'] = method
    else:
        p = dict(bench=bench, device=dict(model='simple'), solver=dict(method=method))
    p['mode'] = mode
    with pytest.raises(ValueError, match=error):
        run_circuit(p)


@pytest.mark.parametrize('stochastic,method,error', [(True, 0, 'simple-stochastic-unavailable'),
                                                   (False, 1, 'simple-method-unavailable')])
def test_low_level_circuit_cannot_bypass_simple_mode_guards(stochastic, method, error):
    n = _net(); c = n.compile()
    with pytest.raises(ValueError, match=error):
        simulate(c, SolverConfig(stochastic=stochastic, method=method), c['P'], n.t_end, -1, 0)
