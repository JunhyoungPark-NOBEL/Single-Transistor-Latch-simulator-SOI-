"""Independent invariants of the reference-normalized geometry model."""
import numpy as np
import pytest

from server import params
from server.payloads import normalize_device
from server.compute.deterministic import run_branches, _pvec, state_row
from server.engine_bridge import A, MODEL, m
from server.geometry_model import constants_from_p, pack_p


def branches(**geometry):
    return run_branches({'device': {'geometry': geometry, 'numerics': {'grid': 201}}})


def test_reference_geometry_is_exact_frozen_engine():
    d=params.resolve_device({'geometry':dict(params.GEOMETRY)})
    p=_pvec(d)
    assert len(p)==26
    ref=A.MODEL.branch(p,m.state_grid(201))
    actual=MODEL.branch(p,m.state_grid(201))
    np.testing.assert_array_equal(actual,ref)
    assert params.is_paper_reference(d)


def test_length_sweep_thresholds_and_no_default_model_mutation():
    result=[branches(Lg_nm=L) for L in (200,300,400,500)]
    for key in ('V_LU','V_LD'):
        vals=[r['folds'][key] for r in result]
        assert all(v is not None for v in vals)
        assert np.all(np.diff(vals)>0)
    before=branches()
    branches(Lg_nm=300,Nbody_cm3=3e17)
    after=branches()
    assert before['folds']==after['folds']


def test_thickness_surface_closure_trend_and_thin_slope():
    ts=np.array([5,10,20,30,50])
    result=[branches(Tsi_nm=t) for t in ts]
    for key in ('V_LU','V_LD'):
        v=np.array([r['folds'][key] for r in result])
        assert np.all(np.diff(v)<0)
        assert abs((v[1]-v[0])/5)>abs((v[-1]-v[-2])/20)


def test_width_scales_current_and_charge_but_dark_thresholds_invariant():
    base=branches()
    wide=branches(W_nm=400)
    for key in ('V_LU','V_LD'):
        assert wide['folds'][key]==pytest.approx(base['folds'][key],abs=1e-10)
    for branch in ('HRS','unstable','LRS'):
        np.testing.assert_allclose(wide[branch]['id'],2*base[branch]['id'],rtol=1e-10,atol=1e-28)
    p0=_pvec(params.resolve_device({}))
    p1=_pvec(params.resolve_device({'geometry':{'W_nm':400}}))
    q0=state_row(.4,3.,p0)[5:8].sum()
    q1=state_row(.4,3.,p1)[5:8].sum()
    assert q1==pytest.approx(2*q0,rel=1e-8)


def test_absolute_photocurrent_is_not_changed_by_width():
    d=params.resolve_device({'geometry':{'W_nm':400},'light':{'iph_pA':10}})
    p=_pvec(d)
    z=m.components(.3,1.,p,MODEL.na,MODEL.vbi,MODEL.rg,MODEL.fg,MODEL.table)
    assert z[18]==pytest.approx(10e-12)


def test_eot_doping_and_box_act_on_intended_quantities():
    base=branches()
    assert branches(EOT_nm=10)['folds']['V_LU']<base['folds']['V_LU']
    assert branches(Nbody_cm3=3e17)['folds']['V_LD']!=base['folds']['V_LD']
    thinbox=branches(Tbox_nm=70)
    assert thinbox['folds']==base['folds']
    p0=_pvec(params.resolve_device({}))
    p1=_pvec(params.resolve_device({'geometry':{'Tbox_nm':70}}))
    assert constants_from_p(p1)[4]>constants_from_p(p0)[4]
    p2=_pvec(params.resolve_device({'geometry':{'EOT_nm':100,'Tbox_nm':1000}}))
    assert constants_from_p(p2)[4]>0


def test_public_geometry_provenance_and_reference_matching():
    result=branches(Lg_nm=300)
    assert len(result['p'])==33
    assert result['geometry']['Lg_nm']==300
    assert result['geometry_model']['scope']=='geometry-extrapolation'
    assert not result['geometry_model']['validated']
    assert not params.is_paper_reference({'geometry':{'Lg_nm':300}})
    assert params.match_photo_condition({'preset':'photo','geometry':{'Lg_nm':300}}) is None


@pytest.mark.parametrize('changes', [
    {'vg': -1.0},
    {'light': {'iph_pA': 1.0}},
    {'calib': {'tau_junction_s': 1e-8}},
    {'ext': {'aloc': 1.0}},
])
def test_reference_dimensions_do_not_claim_validation_for_other_parameters(changes):
    metadata = params.geometry_model_metadata(params.resolve_device(changes))
    assert metadata['reference_geometry']
    assert not metadata['validated']
    assert metadata['scope'] == 'reference-geometry-unvalidated'


def test_reference_calibration_metadata_is_not_independent_validation():
    metadata = params.geometry_model_metadata(params.resolve_device({}))
    assert metadata['validated']
    assert 'not independent predictive validation' in metadata['validation_basis']


def test_unsupported_active_edge_only_avalanche_is_rejected_without_changing_engine():
    with pytest.raises(ValueError, match='local-avalanche-mode-unavailable'):
        params.build_p({'ext': {'aloc': 1.0, 'loc_carriers': 2}})
    # Inactive old saves remain readable; the baseline has no local path.
    assert params.build_p({'ext': {'aloc': 0.0, 'loc_carriers': 2}})[21] == 0.0
    for supported in (0, 1):
        assert params.build_p({'ext': {'aloc': 1.0, 'loc_carriers': supported}})[24] == supported


@pytest.mark.parametrize('geometry',[
    {'Lg_nm':0},{'W_nm':float('nan')},{'Tsi_nm':-1},{'EOT_nm':0},
    {'Tbox_nm':float('inf')},{'Nbody_cm3':True},{'Lg_nm':99},
])
def test_geometry_rejects_invalid_values(geometry):
    with pytest.raises(ValueError):
        normalize_device({'geometry':geometry},[])


def test_backgate_coupling_controls_bjt_body_injection_and_split_charge():
    from server.geometry_model import channel_current, backgate_charge, gate_charge_offset, body_potential
    seeds=[]
    potentials=[]
    channels=[]
    for tbox in (70,140,280):
        device=params.resolve_device({'vg':-.8,'vbg':1.,'geometry':{'Tbox_nm':tbox}})
        p=_pvec(device)
        z=m.components(.4,2.,p,MODEL.na,MODEL.vbi,MODEL.rg,MODEL.fg,MODEL.table)
        seeds.append(z[3])
        potentials.append(body_potential(.4,p))
        channels.append(channel_current(.4,2.,p))
        cf=3.9*m.EPS0*(200e-9)*(500e-9)/14.1e-9
        cb=3.9*m.EPS0*(200e-9)*(500e-9)/((tbox+50/3)*1e-9)
        assert backgate_charge(p)==pytest.approx(-cb)
        assert gate_charge_offset(p)==pytest.approx(-cf*device['vg']-cb)
        assert len(params.build_p(device))==33
    assert np.all(np.diff(seeds)<0)
    assert np.all(np.diff(potentials)<0)
    np.testing.assert_array_equal(channels,[channels[0]]*3)
    d=params.resolve_device({'vbg':1.})
    assert params.uses_geometry_model(d)
    assert not params.is_paper_reference(d)
    assert not params.geometry_model_metadata(d)['validated']


def test_backgate_bias_preserves_equilibrium_and_front_gate_field():
    from server.geometry_model import body_potential, channel_current
    biased=[]
    for vbg in (-1.,0.,1.):
        p=pack_p(params.build_p(params.resolve_device({'vg':0.,'vbg':vbg})),force=True)
        z0=m.components(0.,0.,p,MODEL.na,MODEL.vbi,MODEL.rg,MODEL.fg,MODEL.table)
        np.testing.assert_array_equal(z0[:10],np.zeros(10))
        assert np.isfinite(body_potential(-.02,p))
        assert np.isfinite(body_potential(.7,p))
        # A fixed front/drain internal field must not acquire a second VBG term.
        p[11]=-2.
        z=m.components(.4,2.,p,MODEL.na,MODEL.vbi,MODEL.rg,MODEL.fg,MODEL.table)
        biased.append((z[3],z[6],z[7]/z[14],z[8],z[9],channel_current(.4,2.,p)))
    assert np.all(np.diff(np.array(biased)[:,0])>0)  # BJT seed responds.
    np.testing.assert_allclose(np.array(biased)[:,1:],np.tile(biased[1][1:],(3,1)),rtol=1e-14,atol=0.)


def test_body_bias_potential_and_charge_derivatives_match_finite_difference():
    from server.geometry_model import (body_potential,body_potential_derivative,
        body_potential_backgate_derivative,terminal_capacitances,gate_charge_offset)
    p=_pvec(params.resolve_device({'vbg':.6,'geometry':{'Tbox_nm':70}}))
    cf,cb=terminal_capacitances(p)
    cbody=constants_from_p(p)[4]
    for u in (.05,.5,.8):
        h=1e-6
        analytic_u=body_potential_derivative(u,p)
        assert analytic_u==pytest.approx((body_potential(u+h,p)-body_potential(u-h,p))/(2*h),rel=1e-8)
        plus=p.copy();minus=p.copy();plus[32]+=h;minus[32]-=h
        analytic_bg=body_potential_backgate_derivative(u,p)
        assert analytic_bg==pytest.approx((body_potential(u,plus)-body_potential(u,minus))/(2*h),rel=1e-8)
        def qcap(pp):
            return cbody*body_potential(u,pp)+gate_charge_offset(pp)
        # Explicit gate charge appears once; psi's response is a separate chain rule.
        assert (qcap(plus)-qcap(minus))/(2*h)==pytest.approx(cbody*analytic_bg-cb,rel=2e-8,abs=1e-25)
        plus=p.copy();minus=p.copy();plus[11]+=h;minus[11]-=h
        assert (qcap(plus)-qcap(minus))/(2*h)==pytest.approx(-cf,rel=1e-8)


def test_backgate_outside_source_barrier_domain_is_rejected_not_clipped():
    with pytest.raises(ValueError,match='backgate-domain-unavailable'):
        _pvec(params.resolve_device({'vbg':10.,'geometry':{'Tbox_nm':10}}))
    p=_pvec(params.resolve_device({'vbg':.1}))
    p[32]=100.  # Dynamic terminal bias can change an already-packed vector.
    z=m.components(0.,0.,p,MODEL.na,MODEL.vbi,MODEL.rg,MODEL.fg,MODEL.table)
    assert np.isnan(z).all()


def test_zero_backgate_forced_geometry_matches_reference_kernel():
    p=np.asarray(params.build_p(params.resolve_device({})),dtype=float)
    packed=pack_p(p,force=True)
    for u,r in ((0.,0.),(.1,1.),(.4,2.),(.7,2.)):
        original=m.components(u,r,p,MODEL.na,MODEL.vbi,MODEL.rg,MODEL.fg,MODEL.table)
        extended=m.components(u,r,packed,MODEL.na,MODEL.vbi,MODEL.rg,MODEL.fg,MODEL.table)
        np.testing.assert_allclose(extended,original,rtol=2e-13,atol=1e-29)


def test_reference_anchored_body_capacitance_is_positive_at_domain_corners():
    import itertools
    for eot,tbox,tsi in itertools.product((1,100),(10,1000),(5,200)):
        p=_pvec(params.resolve_device({'geometry':{'EOT_nm':eot,'Tbox_nm':tbox,'Tsi_nm':tsi}}))
        assert constants_from_p(p)[4]>0


def test_vbg_validation_and_public_echo():
    with pytest.raises(ValueError):normalize_device({'vbg':11},[])
    with pytest.raises(ValueError):normalize_device({'vbg':True},[])
    result=run_branches({'device':{'vbg':.5,'numerics':{'grid':201}}})
    assert result['vbg']==.5
    assert result['p'][32]==.5


@pytest.mark.parametrize('geometry',[{'Lg_nm':100},{'Nbody_cm3':1e15},{'Nbody_cm3':1e19}])
def test_outside_reduced_model_domain_is_not_reported_as_physical_no_latch(geometry):
    with pytest.raises(ValueError,match='geometry-domain-unavailable'):
        branches(**geometry)
