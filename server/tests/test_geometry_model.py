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


# ---- P1-2: n+ source/drain hole injection scales with the junction area only (owner decision D8) ----
def test_emitter_injection_scales_with_area_only():
    d0=params.resolve_device({})
    z0=m.components(.3,1.,_pvec(d0),MODEL.na,MODEL.vbi,MODEL.rg,MODEL.fg,MODEL.table)
    for geometry,ratio in (({'Lg_nm':300},1.),({'Nbody_cm3':1e17},1.),({'Nbody_cm3':5e17},1.),
                           ({'W_nm':400},2.),({'Tsi_nm':25},.5),({'Lg_nm':700,'W_nm':100},.5)):
        p=_pvec(params.resolve_device({'geometry':geometry}))
        z=m.components(.3,1.,p,MODEL.na,MODEL.vbi,MODEL.rg,MODEL.fg,MODEL.table)
        assert z[6]==pytest.approx(ratio*z0[6],rel=1e-12), geometry    # z[6]: source (emitter) diffusion loss
    meta=params.geometry_model_metadata({'geometry':{'Lg_nm':400}})
    assert meta['emitter_injection_scaling']=='area only (fixed n+ source/drain)'


def test_forward_drain_diode_uses_the_fixed_emitter():
    from server.compute.circuit.element import N_EV, stl_eval
    def ifwd(geometry):
        p=_pvec(params.resolve_device({'geometry':geometry})) if geometry else np.array(params.build_p(params.resolve_device({})))
        out=np.empty(N_EV);ref=np.empty(N_EV)
        assert stl_eval(.2,-.3,p,MODEL.na,MODEL.vbi,MODEL.rg,MODEL.fg,MODEL.table,out)
        assert stl_eval(.2,0.,p,MODEL.na,MODEL.vbi,MODEL.rg,MODEL.fg,MODEL.table,ref)
        return out[6]-ref[6]                                           # hole loss added by the forward drain diode
    base=ifwd(None)
    assert base>0
    # depletion SRH part depends on Nbody, the diffusion part does not: at r = -0.3 V diffusion dominates
    assert ifwd({'Lg_nm':300})==pytest.approx(base,rel=1e-9)
    assert ifwd({'W_nm':400})==pytest.approx(2*base,rel=1e-9)


def test_area_only_fold_voltages():
    def folds(**geometry):
        f=run_branches({'device':{'geometry':geometry,'numerics':{'grid':601}},'sweep':{'vd_max_V':8}})['folds']
        return f['V_LU'],f['V_LD']
    np.testing.assert_allclose(folds(Lg_nm=400),(3.5821,2.2423),atol=1e-3)
    np.testing.assert_allclose(folds(Lg_nm=1000),(4.1646,3.8643),atol=1e-3)
    np.testing.assert_allclose(folds(Nbody_cm3=5e17),(3.6025,2.6349),atol=1e-3)
    np.testing.assert_allclose(folds(),(3.7037,2.5979),atol=1e-4)


# ---- P1-3: model domain is refused, never reported as physics ----
def test_input_limits_and_min_length_match_the_worker_check():
    from server.engine_bridge import m as eng
    assert params._Q_C==eng.Q and params._VT_V==eng.VT and params._EPS0_F_M==eng.EPS0 and params._NI_CM3==eng.NI_CM3
    assert params.GEOMETRY_LIMITS['Nbody_cm3']==(3e16,1.1e18)
    for na in (3e16,1e17,params.NA_CM3,1.1e18):
        lmin=params.min_length_nm(na)
        ok=params.resolve_device({'geometry':{'Nbody_cm3':na,'Lg_nm':lmin+.5}})
        pack_p(params.build_p(ok))                                      # just above: accepted by the worker too
        with pytest.raises(ValueError,match='geometry-domain-unavailable'):
            pack_p(params.build_p(params.resolve_device({'geometry':{'Nbody_cm3':na,'Lg_nm':lmin-.5}})))
    # the whole Nbody input range has a field table up to FIELD_MIN_REVERSE_V
    for na in np.geomspace(3e16,1.1e18,7):
        pack_p(params.build_p(params.resolve_device({'geometry':{'Nbody_cm3':float(na),'Lg_nm':2000}})))


@pytest.mark.parametrize('nbody',[2e16,2e18,3e18])
def test_nbody_outside_the_domain(nbody):
    with pytest.raises(ValueError,match=r'Nbody_cm3 must be within'):
        normalize_device({'geometry':{'Nbody_cm3':nbody}},[])
    with pytest.raises(ValueError,match='geometry-domain-unavailable'):
        pack_p(params.build_p(params.resolve_device({'geometry':{'Nbody_cm3':nbody}})))


def test_short_length_is_refused_at_normalisation():
    with pytest.raises(ValueError,match=r'geometry-domain-unavailable: L = 100 nm .*must exceed 153.6 nm'):
        normalize_device({'geometry':{'Lg_nm':100}},[])
    with pytest.raises(ValueError,match=r'must exceed 412.2 nm'):
        normalize_device({'geometry':{'Lg_nm':400,'Nbody_cm3':3e16}},[])
    normalize_device({'geometry':{'Lg_nm':160}},[])


@pytest.mark.parametrize('vbg',[4.,8.,-4.])
def test_backgate_beyond_linear_coupling_is_refused(vbg):
    thin={'EOT_nm':100,'Tbox_nm':10,'Tsi_nm':5}
    with pytest.raises(ValueError,match=r'geometry-domain-unavailable: back-gate coupling beyond the linear'):
        normalize_device({'geometry':thin,'vbg':vbg},[])
    with pytest.raises(ValueError,match='geometry-domain-unavailable'):
        pack_p(params.build_p(params.resolve_device({'geometry':thin,'vbg':vbg})))
    normalize_device({'geometry':thin,'vbg':.2},[])                    # 100/11.67*0.2 = 1.71 V: inside
    normalize_device({'vbg':10.},[])                                    # reference stack: 14.1/156.7*10 = 0.90 V
    meta=params.geometry_model_metadata({'vbg':1.})
    assert '2 V' in meta['backgate_coupling'] and 'front-channel' in meta['backgate_scope']


def test_channel_softplus_does_not_overflow():
    from server.geometry_model import channel_current
    p=_pvec(params.resolve_device({'geometry':{'Lg_nm':400}}))
    for vg in (-2.,-.8,.5):
        q=p.copy();q[11]=vg
        n=1.7786684648788609;ov=vg+0.49032524444873615;pp=ov/n;vt=m.VT
        sf=np.log1p(np.exp(pp/(2*vt)));sr=np.log1p(np.exp((pp-.3-1.)/(2*vt)))
        ref=2*n*7.52135238967614e-5*vt*vt*(sf-sr)*(sf+sr)/(1+0.6335606399651017*n*vt*np.log1p(np.exp(ov/(n*vt))))
        assert channel_current(.3,1.,q)==pytest.approx(ref*500/400,rel=1e-13)   # same closed form below the guard
    q=p.copy();q[11]=150.
    assert np.isfinite(channel_current(.3,1.,q)) and channel_current(.3,1.,q)>0


@pytest.mark.parametrize('device',[
    {'geometry':{'EOT_nm':100,'Tbox_nm':10,'Tsi_nm':5},'vbg':4},
    {'geometry':{'EOT_nm':100,'Tbox_nm':10,'Tsi_nm':5},'vbg':8},
    {'geometry':{'Lg_nm':100}},
    {'geometry':{'Nbody_cm3':2e18}},
])
def test_domain_errors_are_http_422_not_job_errors(client,device):
    r=client.post('/api/compute/branches',params={'wait':0},json={'device':device})
    assert r.status_code==422, r.text
    detail=r.json()['detail']
    assert 'geometry-domain-unavailable' in detail or 'must be within' in detail


def test_custom_circuit_domain_error_names_the_cell():
    from server.compute.circuit import run_circuit
    els=[{'type':'V','name':'VG','nodes':['g','0'],'wave':{'kind':'dc','value':-2}},
         {'type':'V','name':'VD','nodes':['d','0'],'wave':{'kind':'dc','value':1}},
         {'type':'STL','name':'X7','nodes':{'d':'d','g':'g','s':'0'},'device':{'geometry':{'Lg_nm':120}}}]
    with pytest.raises(ValueError,match=r'^X7: geometry-domain-unavailable: L = 120 nm'):
        run_circuit({'bench':'custom','mode':'deterministic','netlist':{'elements':els},'tran':{'t_stop_s':1e-6}})
