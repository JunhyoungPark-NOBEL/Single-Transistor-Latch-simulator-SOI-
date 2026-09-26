"""Physical identities and numerical domain of the independent Simple Model."""
from __future__ import annotations

import inspect
import numpy as np
import pytest

from server import params
from server import simple_model as sm


def packed(**changes):
    p = np.zeros(3038)
    p[:26] = params.build_p(None)[:26]
    p[26:32] = list(params.GEOMETRY.values())
    p[34] = -1
    p[11] = -3.0
    # Geometry-scaled paper starting values with unfit diffusion beta.
    p[36:48] = [2.3, 200e-9, .86e-15*200/650, 44000*650/200,
                2e-16*200/650, 2.35, 4, .2, .0525, -3.35, 1, .5]
    for key, value in changes.items():
        p[int(key.removeprefix('p'))] = value
    return p


def test_diffusion_beta_is_bias_constant():
    p = packed()
    beta = sm.effective_parameters(p)[0]
    for u, r, vg, vbg in [(.05, .1, -2., 0.), (.45, 1.9, -3., .5), (-.03, 2., -2.5, -.5)]:
        p[11], p[32] = vg, vbg
        z = sm.components(u, r, p)
        assert z[3] / z[6] == pytest.approx(beta, rel=2e-15)


def test_diffusion_geometry_area_cancellation():
    base = sm.effective_parameters(packed())
    for length, doping in [(250., params.NA_CM3), (500., 2*params.NA_CM3), (750., .5*params.NA_CM3)]:
        p = packed(p26=length, p31=doping)
        beta, _, _, _, saturation, *_ = sm.effective_parameters(p)
        assert beta/base[0] == pytest.approx(500/length*params.NA_CM3/doping)
        assert saturation/beta == pytest.approx(base[4]/base[0], rel=2e-15)
    for width, tsi in [(400., 50.), (200., 25.)]:
        p = packed(p27=width, p28=tsi)
        e = sm.effective_parameters(p)
        assert e[0] == base[0]
        assert e[4]/e[0] == pytest.approx(base[4]/base[0]*(width/200)*(tsi/50))


def test_first_order_loss_uses_charge_reservoir_not_contact_voltage():
    p = packed()
    beta, tau, cb, resistance, saturation, *_ = sm.effective_parameters(p)
    bias = sm.effective_parameters(p)[-1]
    u, r = .53, 1.7
    z = sm.components(u, r, p)
    assert z[13] == pytest.approx(cb*(u + z[1]*resistance - bias), rel=2e-15)
    assert z[5] == pytest.approx(z[13]/tau, rel=2e-15)
    assert z[7] == 0
    assert abs(z[13] - cb*u) > .01*abs(z[13])
    assert z[2] == pytest.approx(z[1] - z[3] - z[5] - z[6], rel=1e-13)


def test_isolated_body_charge_decays_with_effective_lifetime():
    p = packed(p40=1e-100, p46=0, p11=-3.35, p32=-3.35)
    out = np.empty(11)
    assert sm.evaluate(.1, 0., p, out)
    tau = sm.effective_parameters(p)[1]
    assert out[2] == pytest.approx(-out[3]/tau, rel=2e-15)
    h = tau/10
    q_next = out[3]/(1+h/tau)
    assert (q_next-out[3])/h == pytest.approx(-q_next/tau, rel=3e-15)


def test_gate_charges_share_reservoir_and_conserve_incremental_charge():
    p = packed()
    first, second = np.empty(11), np.empty(11)
    assert sm.evaluate(.3, 1., p, first)
    initial = sm.effective_parameters(p)
    p[11] += .17
    p[32] -= .08
    assert sm.evaluate(.42, 1.3, p, second)
    cg, cbg, cs = initial[7:10]
    assert second[9] == pytest.approx(cg*(p[11]-second[8]), rel=2e-15)
    assert second[10] == pytest.approx(cbg*(p[32]-second[8]), rel=2e-15)
    difference = (second[3]-first[3])+(second[9]-first[9])+(second[10]-first[10])-cs*(second[8]-first[8])
    assert abs(difference) < 1e-30
    assert second[8] == pytest.approx(.42+second[1]*initial[3])


def test_surface_lifetime_and_passive_capacitance_geometry():
    base = sm.effective_parameters(packed())
    thin = sm.effective_parameters(packed(p28=25.))
    assert thin[1] == pytest.approx(base[1]/1.5)
    thick_box = sm.effective_parameters(packed(p30=280.))
    assert thick_box[8] < base[8]
    assert thick_box[7] == base[7]
    assert all(v > 0 for v in base[7:10])
    assert base[2] == sum(base[7:10])


@pytest.mark.parametrize('reverse,doping', [(.1, 1e16), (1.5, params.NA_CM3), (2.2, 1e18)])
def test_lateral_quadrature_matches_abrupt_junction_integral(reverse, doping):
    p = packed(p31=doping)
    lateral, _ = sm.btbt_currents(reverse, p)
    vbi = sm.VT*np.log(1e20*doping/sm.NI_CM3**2)
    width = np.sqrt(2*sm.EPS_SI*(vbi+reverse)/(sm.QE*doping))
    z = np.linspace(0, 1, 100001)
    field = 2*(vbi+reverse)/width*z
    generation = sm.BB_A*field**2.5*np.exp(-sm.BB_B/np.maximum(field, 1.))
    reference = sm.QE*p[27]*p[28]*1e-14*width*np.trapezoid(generation, z)*(-np.expm1(-reverse/sm.VT))
    assert lateral == pytest.approx(reference, rel=3e-4, abs=1e-100)


def test_breakdown_domain_is_rejected_without_multiplication_cap():
    p = packed()
    out = np.empty(11)
    assert sm.evaluate(.1, p[41]*(1-1e-6), p, out)
    assert not sm.evaluate(.1, p[41], p, out)
    assert np.isnan(out).all()
    assert not sm.evaluate(.1, -.051, p, out)
    assert sm.evaluate(-.1, -.01, p, out)
    assert out[4] == 0
    assert out[7] == 2


def test_width_scales_all_currents_and_charge_without_moving_voltage():
    a = sm.components(.5, 1.7, packed())
    b = sm.components(.5, 1.7, packed(p27=400.))
    assert b[0] == pytest.approx(a[0], rel=2e-15)
    np.testing.assert_allclose(b[[1,2,3,5,6,8,9,13]], 2*a[[1,2,3,5,6,8,9,13]], rtol=3e-15, atol=0)


def test_equilibrium_branches_have_two_folds_and_correct_length_trend():
    folds = []
    for length in [200., 500.]:
        p = packed(p26=length)
        branch = sm.branch(p, np.linspace(0, 1.12, 601))
        assert len(branch) > 400
        assert np.all(np.isfinite(branch))
        assert np.max(np.abs(branch[:,2])/np.maximum(branch[:,1], 1e-20)) < 1e-7
        dv = np.diff(branch[:,0])
        maxima = np.flatnonzero((dv[:-1] > 0)&(dv[1:] < 0))+1
        minima = np.flatnonzero((dv[:-1] < 0)&(dv[1:] > 0))+1
        assert len(maxima) == len(minima) == 1
        assert maxima[0] < minima[0]
        folds.append(branch[[maxima[0], minima[0]],0])
    assert np.all(folds[0] < folds[1])


def test_no_distributed_srh_or_field_table_dependency():
    source = inspect.getsource(sm)
    assert 'import photo_mean' not in source
    assert 'srh_solve(' not in source
    assert 'Field(' not in source
    assert sm.NREF == params.NA_CM3
    out = np.empty(11)
    assert sm.evaluate(.2, 1.0, packed(), out)
