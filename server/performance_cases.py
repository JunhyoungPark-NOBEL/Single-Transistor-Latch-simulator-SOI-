"""Reproducible timing workloads and cheap request descriptors (no numerical imports).

These descriptors estimate cost, not convergence or physical accuracy. In particular,
``minimum_steps`` is only t_stop/dt_max; adaptive stiffness can require far more work.
"""
from __future__ import annotations

import copy
import math


def _number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except (ValueError, TypeError):
        return float(default)


def _elements(payload):
    return (payload.get("netlist") or {}).get("elements") or []


def model_for_payload(kind, payload):
    if kind == "circuit" and payload.get("bench") == "custom":
        models = {str((el.get("device") or {}).get("model", "detailed"))
                  for el in _elements(payload) if el.get("type") == "STL"}
        return next(iter(models)) if len(models) == 1 else ("mixed" if models else "common")
    return str((payload.get("device") or {}).get("model", "detailed"))


def stochastic_engine_for_payload(payload):
    """Mirror the cheap auto-engine eligibility conditions without importing Numba."""
    from server import params
    device=params.resolve_device(payload.get("device"))
    preset=device.get("preset", "paper")
    st=params.resolve_section(preset,"stochastic",payload.get("stochastic"))
    sweep=params.resolve_section(preset,"sweep",payload.get("sweep"))
    engine=st.get("engine","auto")
    if engine != "auto":
        return engine
    dv=_number(sweep.get("dv_V"),.002)
    k=int(round(.01/max(1e-12,dv)))
    dv_ok=k>=1 and int(round(4/max(1e-12,dv)))==400*k
    ok=params.is_paper_reference(device) and (st.get("local_state") or {}).get("action")=="gidl" and abs(_number(sweep.get("vd_max_V"),4)-4)<1e-9 and st.get("carrier_noise",True) and dv_ok
    return "calibrated_lookup" if ok else "general"


def family_for_payload(kind, payload):
    if kind == "sweep_mc":
        return "vscm_stochastic_lookup" if stochastic_engine_for_payload(payload)=="calibrated_lookup" else "vscm_stochastic_general"
    if kind != "circuit":
        return {"branches": "idvd", "folds": "idvd", "vg_curve": "gate_sweep",
                "sweep_mc": "vscm_stochastic", "vg_curve_stochastic": "gate_stochastic",
                "hazard": "hazard", "charge_balance": "charge_balance",
                "simple_calibrate": "hrs_calibration"}.get(kind, kind)
    elements = _elements(payload)
    stl = [el for el in elements if el.get("type") == "STL"]
    if payload.get("bench") != "custom":
        family = "circuit_legacy"
    elif not stl:
        family = "circuit_basic"
    elif any(isinstance(el.get("nodes"), dict) and "b" in el["nodes"] for el in stl):
        family = "circuit_body"
    elif len(stl) > 1:
        family = "circuit_multi"
    elif any(el.get("type") == "I" for el in elements) and any(el.get("type") == "C" for el in elements):
        family = "circuit_csvm"
    else:
        family = "circuit_voltage"
    return family + ("_stochastic" if payload.get("mode") == "stochastic" else "")


def workload_features(kind, payload):
    """Finite numerical descriptors; safe to call before loading the compute engine."""
    from server import params
    device = params.resolve_device(payload.get("device"))
    preset=device.get("preset","paper")
    sweep = params.resolve_section(preset,"sweep",payload.get("sweep"))
    stochastic = params.resolve_section(preset,"stochastic",payload.get("stochastic")) if kind != "circuit" else (payload.get("stochastic") or {})
    grid = _number((device.get("numerics") or {}).get("grid"), 601)
    result = dict(grid=grid)
    if kind in ("branches", "folds", "sweep_mc", "vg_curve_stochastic", "hazard"):
        result["sweep_points"] = min(2001, max(2, round(_number(sweep.get("vd_max_V"), 4) /
                                                        max(1e-8, _number(sweep.get("dv_V"), .002))) + 1))
    if kind in ("vg_curve", "vg_curve_stochastic"):
        result["vg_points"] = _number(payload.get("n"), 37 if kind == "vg_curve" else 19)
    if kind in ("sweep_mc", "vg_curve_stochastic"):
        result.update(n_cycles=_number(stochastic.get("n_cycles"), 400),
                      fold_nodes=_number(stochastic.get("fold_nodes"), 21),
                      hazard_nodes=_number(stochastic.get("hazard_nodes"), 3),
                      n_traces=_number(stochastic.get("n_traces"),12),
                      carrier_noise=int(bool(stochastic.get("carrier_noise",True))),
                      ld_carrier_noise=int(bool(stochastic.get("ld_carrier_noise",False))),
                      local_state_active=int((stochastic.get("local_state") or {}).get("mode","none")!="none"),
                      local_state_sigma=_number((stochastic.get("local_state") or {}).get("sigma")))
    if kind == "simple_calibrate":
        result["fit_points"] = len(payload.get("points") or [])
        result["fit_parameters"] = 2 if payload.get("fit") == "is_tau" else 1
    if kind == "charge_balance":
        result["state_points"] = _number(payload.get("n_u"), 401)
    if kind == "circuit":
        els = _elements(payload)
        stl = [el for el in els if el.get("type") == "STL"]
        tran = payload.get("tran") or {}
        duration = max(1e-18, _number(tran.get("t_stop_s"), .01))
        dt_max = max(1e-18, _number(tran.get("dt_max_s"), duration/2000))
        nodes = set()
        for el in els:
            ns = el.get("nodes") or []
            nodes.update(ns.values() if isinstance(ns, dict) else ns)
        waves = [el.get("wave") or {} for el in els if el.get("type") in ("V", "I")]
        caps = [_number(el.get("value")) for el in els if el.get("type") == "C"]
        currents = [abs(_number((el.get("wave") or {}).get("value"))) for el in els if el.get("type") == "I"]
        devices = [el.get("device") or {} for el in stl]
        # Approximate oscillatory work from injected charge / drain capacitance.
        # Only directly connected D-S capacitances belong here: an unrelated tiny
        # Body capacitor must not inflate the charging-frequency proxy.
        charge_progress = []
        drain_caps = []
        for el in stl:
            ns = el.get("nodes") or {}
            if not isinstance(ns,dict):
                continue
            ds = {ns.get("d"),ns.get("s")}
            drain_cap = sum(max(0,_number(e.get("value"))) for e in els
                            if e.get("type") == "C" and set(e.get("nodes") or []) == ds)
            input_current = sum(abs(_number((e.get("wave") or {}).get("value"))) for e in els
                                if e.get("type") == "I" and set(e.get("nodes") or []) == ds)
            if drain_cap > 0:
                drain_caps.append(drain_cap)
                charge_progress.append(duration*input_current/drain_cap)
        source_cycles = sum(duration*_number(w.get("freq")) if w.get("kind") == "sine" else
                            duration/max(1e-18,_number(w.get("per"),duration)) if w.get("kind") == "pulse" else
                            max(0,len(w.get("t") or [])-1)/2 if w.get("kind") == "pwl" else 0 for w in waves)
        result.update(t_stop_s=duration, dt_max_s=dt_max, minimum_steps=duration/dt_max,
                      n_runs=max(1, _number(stochastic.get("n_runs"), 20)) if payload.get("mode") == "stochastic" else 1,
                      stl_count=len(stl), simple_count=sum(d.get("model") == "simple" for d in devices),
                      detailed_count=sum(d.get("model", "detailed") == "detailed" for d in devices),
                      element_count=len(els), node_count=len(nodes-{"0", "gnd"}),
                      body_contacts=sum("b" in (el.get("nodes") or {}) for el in stl),
                      dynamic_sources=sum(w.get("kind", "dc") != "dc" for w in waves),
                      source_frequency_hz=max([_number(w.get("freq")) for w in waves] + [0]),
                      min_cap_F=min(caps) if caps else 0, forcing_current_A=max(currents) if currents else 0,
                      drain_cap_F=min(drain_caps) if drain_caps else 0,
                      charge_progress_V=sum(charge_progress)/max(1,len(stl)),source_cycles=source_cycles,
                      basic_device_count=sum(el.get("type") in ("MOS","D","BJT") for el in els),
                      reltol=_number(tran.get("reltol"), .001))
        # Explicit conservative workload heuristic, not a solver step prediction.
        # Charge/source terms allow increased duration to cost more even when the
        # UI also increases dt_max=T/3000. Actual runtime observations override it.
        tolerance_factor = (.001/max(1e-6,result["reltol"]))**.25
        result["work_units"] = (result["minimum_steps"]+100*result["charge_progress_V"]+200*source_cycles)*result["n_runs"]*max(1,len(stl))*(1+.02*result["node_count"]+.05*result["basic_device_count"])*tolerance_factor
        if devices:
            device = devices[0]
    else:
        result["work_units"] = grid * result.get("vg_points", 1)
        if kind == "sweep_mc":
            # Affine setup/loop cost observed at 10 and 100 cycles for the shipped
            # calibrated lookup: ~0.01758 s fixed + ~0.001634 s/cycle. Ratios only;
            # local calibration still supplies the seconds on a different host.
            lookup = stochastic_engine_for_payload(payload)=="calibrated_lookup"
            result["stochastic_lookup"] = int(lookup)
            # General-engine 3-fold/1-hazard-node data give a similar fixed/loop
            # ratio (10.722 cycles); table-node scaling is deliberately heuristic.
            setup_units = 10.757 if lookup else 10.722*(.5*result.get("fold_nodes",3)/3 + .5*result.get("hazard_nodes",1))
            result["work_units"] = setup_units + result.get("n_cycles",400)*result.get("sweep_points",2001)/2001
        elif kind == "vg_curve_stochastic":
            result["work_units"] *= max(1,result.get("fold_nodes",21))*max(1,result.get("hazard_nodes",3))
        elif kind == "charge_balance":
            result["work_units"] = result["state_points"]
        elif kind == "simple_calibrate":
            result["work_units"] = 2001 + 30*result["fit_points"]*result["fit_parameters"]
    geometry = device.get("geometry") or {}
    for key, default in (("Lg_nm", 500), ("W_nm", 200), ("Tsi_nm", 50), ("EOT_nm", 14.1),
                         ("Tbox_nm", 140), ("Nbody_cm3", 2.295773162796593e17)):
        result[key] = _number(geometry.get(key), default)
    result["vg_V"] = _number(device.get("vg"), -2)
    result["vbg_V"] = _number(device.get("vbg"), 0)
    return result


def _csvm(cap=1e-12, duration=.015, count=1):
    elements = [dict(type="V", name="VG", nodes=["gate", "0"], wave=dict(kind="dc", value=-3))]
    probes = []
    for i in range(1, count+1):
        drain = "drain" if count == 1 else f"d{i}"
        elements.extend([dict(type="I", name=f"I{i}", nodes=["0", drain], wave=dict(kind="dc", value=1e-9)),
                         dict(type="C", name=f"C{i}", nodes=[drain, "0"], value=cap),
                         dict(type="STL", name=f"X{i}", nodes=dict(d=drain, g="gate", s="0"), device=dict(vg=-3))])
        probes.extend([f"V({drain})", f"I(X{i}.d)", f"X{i}.vb"])
    return dict(bench="custom", mode="deterministic", netlist=dict(elements=elements),
                tran=dict(t_stop_s=duration, t_start_save_s=0, dt_max_s=duration/3000,
                          dt_min_s=1e-15, method="BE", reltol=.001),
                detect=dict(i_threshold_A=1e-8, hysteresis=10), probes=probes)


def _body(csvm=True):
    p = _csvm(1e-15, 4e-5)
    els = p["netlist"]["elements"]
    els[-1]["nodes"].update(bg="bg", b="body")
    els.extend([dict(type="V", name="VBG", nodes=["bg", "0"], wave=dict(kind="sine", vo=0, va=.02 if csvm else .1, freq=5e4)),
                dict(type="R", name="RB", nodes=["body", "0"], value=1e12 if csvm else 2e10),
                dict(type="C", name="CB", nodes=["body", "0"], value=1e-16 if csvm else 1e-14)])
    p["probes"].extend(["V(body)", "V(bg)", "I(X1.b)"])
    if not csvm:
        p["netlist"]["elements"] = [el for el in els if el["name"] not in ("I1", "C1")]
        p["netlist"]["elements"].append(dict(type="V", name="VD", nodes=["drain", "0"], wave=dict(kind="dc", value=1)))
        p["netlist"]["elements"][0]["wave"] = dict(kind="pulse", v1=-3, v2=-2.95, td=5e-6, tr=1e-6, tf=1e-6, pw=8e-6, per=2e-5)
        p["tran"]["dt_max_s"] = 2e-7
    return p


def _voltage():
    p = _csvm(1e-15, .004)
    p["netlist"]["elements"] = [el for el in p["netlist"]["elements"] if el["type"] not in ("I", "C")]
    p["netlist"]["elements"].extend([
        dict(type="V", name="VD", nodes=["supply", "0"], wave=dict(kind="pwl", t=[0,.002,.004], v=[0,4,0])),
        dict(type="R", name="RD", nodes=["supply", "drain"], value=1000)])
    return p


def _basic(kind):
    els = [dict(type="V", name="VDD", nodes=["supply", "0"], wave=dict(kind="dc", value=1 if kind == "D" else 3)),
           dict(type="R", name="R1", nodes=["supply", "out"], value=1000)]
    if kind == "D":
        els[0]["wave"] = dict(kind="pwl", t=[0,1e-5,2e-5], v=[0,1,0])
        els.append(dict(type="D", name="D1", nodes=dict(a="out", k="0")))
    else:
        name, value, node = ("M1", 2, "gate") if kind == "MOS" else ("Q1", .7, "base")
        nodes = dict(d="out", g=node, s="0") if kind == "MOS" else dict(c="out", b=node, e="0")
        els.extend([dict(type="V", name="VIN", nodes=[node,"0"], wave=dict(kind="pwl", t=[0,1e-5,2e-5], v=[0,value,0])),
                    dict(type=kind, name=name, nodes=nodes)])
    return dict(bench="custom", mode="deterministic", netlist=dict(elements=els),
                tran=dict(t_stop_s=2e-5, dt_max_s=1e-7, method="BE", reltol=.001))


def benchmark_cases():
    """Complete small catalogue. Payloads resolve against the shipped calibration files."""
    cases = []
    def add_pair(case_id, kind, label, payload, *, calibration=False):
        for model in ("detailed", "simple"):
            p = copy.deepcopy(payload)
            if kind == "circuit":
                for el in _elements(p):
                    if el.get("type") == "STL":
                        el.setdefault("device", {})["model"] = model
            else:
                p.setdefault("device", {})["model"] = model
            cases.append(dict(id=f"{case_id}.{model}", comparison_id=case_id, model=model, kind=kind,
                              family=family_for_payload(kind,p), label=label, payload=p,
                              features=workload_features(kind,p), supported=True, calibration=calibration))
    add_pair("idvd_601", "branches", {"ko":"ID–VD · 601점", "en":"ID–VD · 601 grid"},
             dict(device=dict(vg=-3,numerics=dict(grid=601)), sweep=dict(vd_max_V=4,dv_V=.002)), calibration=True)
    add_pair("gate_sweep_9", "vg_curve", {"ko":"게이트 스윕 · 9점", "en":"Gate sweep · 9 points"},
             dict(device=dict(vg=-3,numerics=dict(grid=601)), vg_min=-3.6, vg_max=-2.8, n=9))
    add_pair("csvm_1pF", "circuit", {"ko":"CSVM · 1 pF · 15 ms", "en":"CSVM · 1 pF · 15 ms"}, _csvm())
    add_pair("csvm_1fF", "circuit", {"ko":"CSVM · 1 fF · 40 µs", "en":"CSVM · 1 fF · 40 µs"}, _csvm(1e-15,4e-5), calibration=True)
    add_pair("voltage_one_stl", "circuit", {"ko":"전압 구동 회로 · STL 1개", "en":"Voltage-driven circuit · 1 STL"}, _voltage())
    add_pair("four_stl", "circuit", {"ko":"CSVM 회로 · STL 4개", "en":"CSVM circuit · 4 STL"}, _csvm(count=4))
    add_pair("five_terminal_csvm", "circuit", {"ko":"5단자 CSVM · Body RC", "en":"5-terminal CSVM · Body RC"}, _body())
    add_pair("body_rc", "circuit", {"ko":"Body RC · G/BG 변조", "en":"Body RC · G/BG modulation"}, _body(False))
    for kind in ("MOS", "D", "BJT"):
        p = _basic(kind)
        cases.append(dict(id=f"basic_{kind.lower()}",comparison_id=f"basic_{kind.lower()}",model="common",kind="circuit",
                          family=family_for_payload("circuit",p),label={"ko":f"기본 소자 · {kind}","en":f"Basic device · {kind}"},
                          payload=p,features=workload_features("circuit",p),supported=True,calibration=False))
    mc = dict(device=dict(preset="paper",numerics=dict(grid=601)),
              stochastic=dict(n_cycles=100,seed=404,n_traces=4,engine="auto"))
    add_pair("vscm_mc_100", "sweep_mc", {"ko":"확률적 VSCM · 100회", "en":"Stochastic VSCM · 100 cycles"}, mc)
    mc10 = copy.deepcopy(mc)
    mc10["stochastic"]["n_cycles"] = 10
    add_pair("vscm_mc_10", "sweep_mc", {"ko":"확률적 VSCM · 10회", "en":"Stochastic VSCM · 10 cycles"}, mc10)
    for count in (10,100):
        general=copy.deepcopy(mc)
        general["stochastic"].update(engine="general",n_cycles=count,fold_nodes=3,hazard_nodes=1)
        add_pair(f"vscm_general_{count}","sweep_mc",{"ko":f"확률적 VSCM · 일반 모델 · {count}회","en":f"Stochastic VSCM · general engine · {count} cycles"},general)
    general_default=copy.deepcopy(mc)
    general_default["stochastic"]["engine"]="general"
    add_pair("vscm_general_default_100","sweep_mc",{"ko":"확률적 VSCM · 일반 모델 · 기본 격자","en":"Stochastic VSCM · general engine · default tables"},general_default)
    sto = _csvm(1e-15,4e-5)
    sto.update(mode="stochastic",stochastic=dict(n_runs=2,seed=404,carrier_noise=True,local_state=dict(mode="none")))
    add_pair("csvm_stochastic_2", "circuit", {"ko":"확률적 CSVM · 2회", "en":"Stochastic CSVM · 2 runs"},sto)
    # Direct stochastic exports and diagnostic API workloads are also represented.
    add_pair("hazard", "hazard", {"ko":"확률적 전이율", "en":"Transition hazard"},dict(device=dict(preset="paper")))
    add_pair("gate_stochastic_3", "vg_curve_stochastic", {"ko":"확률적 게이트 스윕 · 3점", "en":"Stochastic gate sweep · 3 points"},
             dict(device=dict(preset="paper"),vg_min=-2.2,vg_max=-1.8,n=3,stochastic=dict(fold_nodes=3,hazard_nodes=1,n_cycles=100,seed=404)))
    add_pair("charge_balance_401", "charge_balance", {"ko":"바디 전하 평형 · 401점", "en":"Body charge balance · 401 points"},
             dict(device=dict(preset="paper"),vd=3.2,n_u=401))
    # Model-generated points isolate fitting cost; they are not experimental accuracy data.
    fit = dict(device=dict(model="simple",vg=-3,simple=dict(is_ref_A=2e-16*200/650*1.4,tau_body_s=2e-7*.8)),fit="is_tau",
               points=[dict(vd_V=v,id_A=i) for v,i in [(1.,8.510435048276376e-13),(1.7,1.142967707501993e-12),
                                                        (2.1,2.3716085253986285e-12),(2.25,5.9532969107962995e-12)]])
    cases.append(dict(id="hrs_fit_4.simple",comparison_id="hrs_fit_4",model="simple",kind="simple_calibrate",family="hrs_calibration",
                      label={"ko":"HRS 보정 · 4점 / 2변수","en":"HRS fit · 4 points / 2 parameters"},payload=fit,
                      features=workload_features("simple_calibrate",fit),supported=True,calibration=False,
                      notes=["Synthetic HRS points generated by the same Simple Model; timing only, not measured validation."]))
    half = _csvm(1e-15,2e-5)
    half["tran"]["dt_max_s"] = 4e-5/3000
    add_pair("csvm_half_holdout", "circuit", {"ko":"예측 확인 · CSVM 절반 구간", "en":"Prediction check · half-duration CSVM"},half)
    add_pair("csvm_ui_half_holdout", "circuit", {"ko":"예측 확인 · UI CSVM 절반 구간", "en":"Prediction check · UI half-duration CSVM"},_csvm(1e-15,2e-5))
    mixed=_csvm(1e-15,4e-5,count=2)
    cells=[e for e in _elements(mixed) if e.get("type")=="STL"]
    cells[0]["device"]["model"]="detailed";cells[1]["device"]["model"]="simple"
    cases.append(dict(id="mixed_two_stl",comparison_id="mixed_two_stl",model="mixed",kind="circuit",family=family_for_payload("circuit",mixed),
                      label={"ko":"혼합 회로 · Detailed + Simple","en":"Mixed circuit · Detailed + Simple"},payload=mixed,
                      features=workload_features("circuit",mixed),supported=True,calibration=False,repeats=2))
    for case in cases:
        case["group"] = "holdout" if case["comparison_id"].endswith("holdout") else ("advanced" if case["kind"] in ("hazard","charge_balance","vg_curve_stochastic","simple_calibrate") else "standard")
        if case["model"] == "simple" and (case["kind"] in ("sweep_mc","hazard","vg_curve_stochastic","charge_balance") or case["payload"].get("mode") == "stochastic"):
            case["supported"] = False
            case["unsupported_reason"] = "Detailed charge landscape is unavailable in Simple Model." if case["kind"] == "charge_balance" else "Simple Model has no calibrated stochastic carrier model."
    return cases


def calibration_cases():
    return [case for case in benchmark_cases() if case.get("calibration") and case["supported"]]
