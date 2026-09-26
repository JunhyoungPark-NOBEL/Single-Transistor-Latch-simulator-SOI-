"""Physical terminal coupling, external body dynamics and deterministic MNA conservation."""
from __future__ import annotations

import numpy as np
import pytest

from server import params
from server.compute.circuit import run_circuit
from server.compute.circuit.netlist import Netlist
from server.compute.circuit.sim import SolverConfig, simulate
from server.compute.circuit.mna import ST_DONE, GMIN


def _run(net, *, dt=None, method=0, stochastic=False):
    c = net.compile()
    cfg = SolverConfig(dt_max=dt or net.t_end / 200, dt_rec=0, dv_rec=100,
                       method=method, stochastic=stochastic, lte_v=7e-3, lte_v_abs=7e-6)
    out = simulate(c, cfg, c['P'], net.t_end, -1, 0)
    assert out.status == ST_DONE, out.warnings
    assert out.t_reached == pytest.approx(net.t_end)
    return c, out.rec


def _rc_net(cap=1e-12, common=0.0):
    n = Netlist(); n.t_end = 3e-3
    s = 's' if common else '0'
    if common:
        n.add_V('VS', s, '0', [0], [common])
    n.add_V('VG', 'g', s, [0], [-2])
    n.add_V('VD', 'd', s, [0], [1])
    n.add_V('VBG', 'bg', s, [0], [0])
    n.add_V('VBdrive', 'bdrive', s, [0, 1e-6, n.t_end], [0, .15, .15])
    n.add_R('Rb', 'bdrive', 'b', 1e9)
    n.add_C('Cb', 'b', s, cap)
    n.add_STL('X1', 'd', 'g', s, params.build_p({'preset': 'paper'}), bg='bg', b='b')
    return n


def test_body_capacitor_changes_internal_charge_and_contact_dynamics():
    n = _rc_net()
    c, r = _run(n)
    fast_n = _rc_net(0)
    _, fast = _run(fast_n)
    bcol = n.nodes.index('b')
    t = r[:, 0]
    vb = r[:, bcol]
    # A 1 Gohm / 1 pF physical external RC is visible in the actual internal state,
    # not only in an output voltage. Intrinsic charge is <1 fF in this low-bias case.
    base = c['n_nodes'] + c['nV']
    np.testing.assert_allclose(r[:, base], vb, atol=2e-8)
    sample = .3e-3
    slow = np.interp(sample, t, vb)
    quick = np.interp(sample, fast[:, 0], fast[:, fast_n.nodes.index('b')])
    assert quick - slow > .09
    reference = .15 * (1 - np.exp(-(sample - .5e-6) / 1e-3))
    assert slow == pytest.approx(reference, abs=2e-3)
    # BE body-charge balance, with external contact current into the semiconductor.
    port = base + 7 + c['nC']
    dqdt = np.diff(r[:, base + 2]) / np.diff(t)
    residual = dqdt - r[1:, base + 4] - r[1:, port + 2]
    assert np.max(np.abs(residual)) < 2e-15
    # Actual external body-node KCL includes the capacitor companion current.
    ir = (r[:, bcol] - r[:, n.nodes.index('bdrive')]) / 1e9
    ic = r[:, base + 7]
    assert np.max(np.abs(ir + ic + r[:, port + 2] + GMIN * vb)) < 2e-15


def test_source_reference_invariance_for_five_terminal_rc():
    n0 = _rc_net(); c0, r0 = _run(n0)
    n1 = _rc_net(common=.4); c1, r1 = _run(n1)
    b0 = c0['n_nodes'] + c0['nV']; b1 = c1['n_nodes'] + c1['nV']
    common_times = np.linspace(1e-6, n0.t_end, 80)
    for offset in (0, 1, 3):  # u, r and drain current are source-relative physics
        a = np.interp(common_times, r0[:, 0], r0[:, b0 + offset])
        b = np.interp(common_times, r1[:, 0], r1[:, b1 + offset])
        np.testing.assert_allclose(a, b, rtol=3e-3, atol=3e-5 if offset != 3 else 2e-17)
    np.testing.assert_allclose(r1[:, n1.nodes.index('b')] - .4, r1[:, b1], atol=2e-8)


def _mod_net(pin):
    n = Netlist(); n.t_end = 2e-5
    n.add_V('VD', 'd', '0', [0], [1.5])
    n.add_V('VB', 'b', '0', [0], [.4])
    for p, bias, excursion in (('g', -2., .8), ('bg', 0., .5)):
        n.add_V('V' + p, p, '0', [0, 1e-5, 2e-5], [bias, bias + excursion if pin == p else bias, bias])
    n.add_STL('X1', 'd', 'g', '0', params.build_p({'preset':'paper'}), bg='bg', b='b')
    return n


@pytest.mark.parametrize('pin', ['g', 'bg'])
def test_driven_gate_and_backgate_modulate_transport_with_fixed_contact(pin):
    n = _mod_net(pin); c, r = _run(n)
    base = c['n_nodes'] + c['nV']; port = base + 7 + c['nC']
    np.testing.assert_allclose(r[:, base], .4, atol=2e-7)
    assert np.ptp(r[:, base + 3]) > .005 * np.max(np.abs(r[:, base + 3]))
    # Five-terminal sum and source circuit KCL use all displacement/contact terms.
    idrain = r[:, base + 3]
    ig, ibg, ib = r[:, port:port+3].T
    source = -(idrain + ig + ibg + ib)
    np.testing.assert_allclose(idrain + ig + ibg + ib + source, 0, atol=1e-24)
    for j, pinname in enumerate(('g', 'bg')):
        branch = c['n_nodes'] + 2 + j
        terminal = ig if j == 0 else ibg
        resid = r[:, branch] + terminal + GMIN * r[:, n.nodes.index(pinname)]
        assert np.max(np.abs(resid)) < 2e-15


def _payload():
    return {'bench':'custom', 'mode':'deterministic',
            'tran':{'t_stop_s':2e-5, 'dt_max_s':1e-6, 'method':'BE', 'initial':'op'},
            'netlist':{'elements':[
                {'type':'V','name':'VG','nodes':['g','0'],'wave':{'kind':'dc','value':-2}},
                {'type':'V','name':'VD','nodes':['d','0'],'wave':{'kind':'dc','value':1}},
                {'type':'V','name':'VBG','nodes':['bg','0'],'wave':{'kind':'pwl','t':[0,1e-5,2e-5],'v':[0,.5,0]}},
                {'type':'R','name':'Rb','nodes':['b','0'],'value':1e10},
                {'type':'C','name':'Cb','nodes':['b','0'],'value':1e-14},
                {'type':'STL','name':'X1','nodes':{'d':'d','g':'g','s':'0','bg':'bg','b':'b'},'device':{'preset':'paper'}},
            ]}}


def test_custom_port_probes_echo_and_body_electrostatic_potential():
    p = _payload()
    p['probes'] = ['X1.vb', 'X1.vbody', 'X1.u', 'V(b)', 'I(Rb)', 'I(Cb)'] + [f'I(X1.{x})' for x in ('d','s','g','bg','b')]
    out = run_circuit(p)
    sig = {s['key']:np.asarray(s['values']) for s in out['runs'][0]['signals']}
    for key in ('X1.vbody', 'X1.u'):
        np.testing.assert_allclose(sig[key], sig['V(b)'], atol=2e-8)
    assert np.max(np.abs(sig['X1.vb'] - sig['X1.vbody'])) > .03
    assert np.max(np.abs(sig['I(X1.b)'] + sig['I(Rb)'] + sig['I(Cb)'])) < 2e-16
    total = sum(sig[f'I(X1.{x})'] for x in ('d','s','g','bg','b'))
    assert np.max(np.abs(total)) < 2e-18
    assert out['elements'][-1]['nodes'] == p['netlist']['elements'][-1]['nodes']
    assert out['elements'][-1]['device']['geometry_model']['validated'] is False
    assert 'not thresholds' in out['elements'][-1]['folds_scope']


@pytest.mark.parametrize('mode,method,error', [('stochastic','BE','five-terminal-stochastic-unavailable'),
                                             ('deterministic','TRAP','five-terminal-method-unavailable')])
def test_unvalidated_port_integrators_fail_explicitly(mode, method, error):
    p = _payload(); p['mode'] = mode; p['tran']['method'] = method
    with pytest.raises(ValueError, match=error):
        run_circuit(p)


def test_unknown_terminal_cannot_be_silently_ignored():
    p = _payload(); p['netlist']['elements'][-1]['nodes']['body'] = 'b'
    with pytest.raises(ValueError, match='unknown STL terminal'):
        run_circuit(p)


def test_wired_zero_bg_preserves_legacy_floating_body_trajectory():
    def net(wired):
        n = Netlist(); n.t_end = 2e-5
        n.add_V('VG', 'g', '0', [0], [-2])
        n.add_V('VD', 'd', '0', [0, n.t_end / 2, n.t_end], [0, 2, 1])
        if wired:
            n.add_V('VBG', 'bg', '0', [0], [0])
        n.add_STL('X', 'd', 'g', '0', params.build_p({'preset':'paper'}), bg='bg' if wired else None)
        return n
    n0, n1 = net(False), net(True)
    c0, r0 = _run(n0); c1, r1 = _run(n1)
    b0 = c0['n_nodes'] + c0['nV']; b1 = c1['n_nodes'] + c1['nV']
    times = np.linspace(0, n0.t_end, 100)
    for offset in (0, 1, 3):
        a = np.interp(times, r0[:, 0], r0[:, b0 + offset])
        b = np.interp(times, r1[:, 0], r1[:, b1 + offset])
        np.testing.assert_allclose(a, b, rtol=5e-4, atol=2e-7 if offset != 3 else 1e-19)
