"""Recover known HRS parameters and reject physically ambiguous calibration data."""
import copy
import numpy as np
import pytest
from server import params
from server.compute.simple_calibration import _predict_hrs, run_simple_calibrate
from server.payloads import normalize

@pytest.fixture(scope='module')
def reference():
    d = params.resolve_device({'model':'simple','vg':-3.})
    voltages=np.array([1.,1.7,2.1,2.25])
    currents=_predict_hrs(np.array(params.build_p(d)),voltages)
    return d,[dict(vd_V=float(v),id_A=float(i)) for v,i in zip(voltages,currents)]

@pytest.mark.parametrize('fit',['is','tau','is_tau'])
def test_recover_known_parameters_from_hrs(reference,fit):
    d,points=copy.deepcopy(reference)
    truth=dict(d['simple'])
    if fit!='tau':d['simple']['is_ref_A']*=1.4
    if fit!='is':d['simple']['tau_body_s']*=.8
    if fit!='is_tau':points=points[-1:]
    result=run_simple_calibrate(dict(device=d,points=points,fit=fit))
    assert result['identifiable']
    assert result['rmse_log10']<1e-7
    for key in ['is_ref_A','tau_body_s']:
        assert result['device']['simple'][key]==pytest.approx(truth[key],rel=1e-7)
    assert any('not independent' in x for x in result['warnings'])

def test_geometry_reference_parameters_recovered():
    d=params.resolve_device({'model':'simple','vg':-3.,'geometry':{'Lg_nm':400.,'W_nm':350.,'Tsi_nm':40.}})
    volts=np.array([1.,1.5,2.,2.1])
    currents=_predict_hrs(np.array(params.build_p(d)),volts)
    truth=dict(d['simple'])
    d['simple']['is_ref_A']*=1.2;d['simple']['tau_body_s']*=.7
    result=run_simple_calibrate(dict(device=d,fit='is_tau',points=[dict(vd_V=v,id_A=i) for v,i in zip(volts,currents)]))
    assert result['device']['simple']['is_ref_A']==pytest.approx(truth['is_ref_A'],rel=1e-7)
    assert result['device']['simple']['tau_body_s']==pytest.approx(truth['tau_body_s'],rel=1e-7)

def test_one_point_cannot_fit_two_parameters(reference):
    d,p=reference
    with pytest.raises(ValueError,match='identifiability'):
        run_simple_calibrate(dict(device=d,points=p[:1],fit='is_tau'))

def test_repeated_points_do_not_identify_two_parameters(reference):
    d,p=reference
    with pytest.raises(ValueError,match='identifiability'):
        run_simple_calibrate(dict(device=d,points=p[:1]*3,fit='is_tau'))

@pytest.mark.parametrize('point',[dict(vd_V=3.,id_A=1e-12),dict(vd_V=1.,id_A=1e-3),dict(vd_V=1.,id_A=-1e-12)])
def test_wrong_domain_is_not_fitted(reference,point):
    with pytest.raises(ValueError):
        run_simple_calibrate(dict(device=reference[0],points=[point]))

@pytest.mark.parametrize('kind',['hazard','sweep_mc','vg_curve_stochastic','charge_balance'])
def test_simple_unvalidated_modes_rejected(kind):
    with pytest.raises(ValueError,match='simple-mode-unavailable'):
        normalize(kind,{'device':{'model':'simple'}})

def test_default_remains_detailed():
    d,_=normalize('branches',{})
    assert d['device']['model']=='detailed'
    assert len(params.build_p(d['device']))==26


def test_legacy_gidl_volume_scale_is_translated_with_a_warning():
    from server.payloads import normalize_device
    warnings = []
    d = normalize_device({'model': 'simple', 'simple': {'gidl_volume_scale': 100.}}, warnings)
    assert 'gidl_volume_scale' not in d['simple']
    assert d['simple']['gidl_volume_ref_cm3'] == pytest.approx(100 * 4.55e-18)
    assert any('gidl_volume_scale' in w and 'converted' in w for w in warnings)
    warnings = []
    d = normalize_device({'model': 'simple', 'simple': {'gidl_volume_scale': 100., 'gidl_volume_ref_cm3': 2e-16}}, warnings)
    assert d['simple']['gidl_volume_ref_cm3'] == 2e-16
    assert any('ignored' in w for w in warnings)
    assert normalize_device({'model': 'simple'}, [])['simple']['gidl_volume_ref_cm3'] == 4.55e-16
