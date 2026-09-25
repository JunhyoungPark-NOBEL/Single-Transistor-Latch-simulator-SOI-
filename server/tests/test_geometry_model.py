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


@pytest.mark.parametrize('geometry',[
    {'Lg_nm':0},{'W_nm':float('nan')},{'Tsi_nm':-1},{'EOT_nm':0},
    {'Tbox_nm':float('inf')},{'Nbody_cm3':True},{'Lg_nm':99},
])
def test_geometry_rejects_invalid_values(geometry):
    with pytest.raises(ValueError):
        normalize_device({'geometry':geometry},[])


def test_backgate_coupling_is_capacitive_channel_bias_and_split_charge():
    from server.geometry_model import channel_current, backgate_charge, gate_charge_offset
    currents=[]
    for tbox in (70,140,280):
        device=params.resolve_device({'vg':-.8,'vbg':1.,'geometry':{'Tbox_nm':tbox}})
        p=_pvec(device)
        currents.append(channel_current(.1,1.,p))
        cf=3.9*m.EPS0*(200e-9)*(500e-9)/14.1e-9
        cb=3.9*m.EPS0*(200e-9)*(500e-9)/((tbox+50/3)*1e-9)
        assert backgate_charge(p)==pytest.approx(-cb)
        assert gate_charge_offset(p)==pytest.approx(-cf*device['vg']-cb)
        assert len(params.build_p(device))==33
    assert np.all(np.diff(currents)<0)
    d=params.resolve_device({'vbg':1.})
    assert params.uses_geometry_model(d)
    assert not params.is_paper_reference(d)
    assert not params.geometry_model_metadata(d)['validated']


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
