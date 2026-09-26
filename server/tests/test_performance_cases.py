"""Runtime descriptors must follow actual UI controls and model domains."""
import copy

from server.performance_cases import (benchmark_cases, family_for_payload, model_for_payload,
                                      workload_features, stochastic_engine_for_payload)


def _case(case_id):
    return next(c for c in benchmark_cases() if c['id']==case_id)


def test_csvm_duration_cost_changes_when_ui_rescales_dtmax():
    p=_case('csvm_1fF.detailed')['payload']
    before=workload_features('circuit',p)
    p['tran']['t_stop_s']*=2
    p['tran']['dt_max_s']*=2
    after=workload_features('circuit',p)
    assert after['minimum_steps']==before['minimum_steps']
    assert after['work_units']>before['work_units']


def test_body_cap_not_mistaken_for_drain_charge_cap():
    p=_case('five_terminal_csvm.detailed')['payload']
    before=workload_features('circuit',p)
    next(e for e in p['netlist']['elements'] if e['name']=='CB')['value']=1e-20
    after=workload_features('circuit',p)
    assert after['drain_cap_F']==before['drain_cap_F']==1e-15
    assert after['charge_progress_V']==before['charge_progress_V']
    assert after['min_cap_F']<before['min_cap_F']


def test_auto_stochastic_engine_selection_changes_with_calibration():
    p=_case('vscm_mc_100.detailed')['payload']
    assert stochastic_engine_for_payload(p)=='calibrated_lookup'
    assert family_for_payload('sweep_mc',p)=='vscm_stochastic_lookup'
    p['device']['vg']=-2.1
    assert stochastic_engine_for_payload(p)=='general'
    assert family_for_payload('sweep_mc',p)=='vscm_stochastic_general'


def test_monte_carlo_work_is_affine_in_cycles():
    p=_case('vscm_mc_10.detailed')['payload']
    before=workload_features('sweep_mc',p)['work_units']
    p['stochastic']['n_cycles']=100
    after=workload_features('sweep_mc',p)['work_units']
    assert before<after<10*before


def test_mixed_and_basic_models_are_distinguished():
    assert model_for_payload('circuit',_case('mixed_two_stl')['payload'])=='mixed'
    assert model_for_payload('circuit',_case('basic_mos')['payload'])=='common'
    assert all(not c['supported'] for c in benchmark_cases() if c['model']=='simple' and c['kind'] in ('hazard','sweep_mc','vg_curve_stochastic','charge_balance'))
