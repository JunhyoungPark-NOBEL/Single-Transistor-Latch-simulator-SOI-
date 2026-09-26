"""HRS-only DC calibration with fixed geometry, beta, capacitance and generation law.

At a measured (V,I), r=V-IR and seed=(I-B)/M are known.  Rearranging
F=0 gives y=(tau/CB)*G + VT*log(IS), with y=VT*log(seed)+IR-bias.
This identifies IS or tau with the other fixed, or both from varying G.
The returned error is evaluated on the lowest stable branch, never on a
continuation through an unstable or LRS solution. It is a fit error, not validation.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import brentq
from server import params, simple_model as sm
from server.payloads import normalize_device
from server.progress import null_progress
from server.simple_config import SIMPLE_LIMITS


def _hrs_curve(p):
    b = sm.branch(p, np.linspace(0., 1.12, 2001))
    if len(b) < 3:
        raise ValueError("simple-calibration-domain: no resolvable HRS branch")
    turns = np.flatnonzero(np.diff(b[:, 0]) <= 0)
    if len(turns):
        b = b[:int(turns[0])+1]
    if len(b) < 3:
        raise ValueError("simple-calibration-domain: no stable HRS interval")
    return b


def _predict_hrs(p, voltages):
    b = _hrs_curve(p)
    result = []
    for vd in voltages:
        k = int(np.searchsorted(b[:, 0], vd))
        if k == 0 or k >= len(b):
            raise ValueError("simple-calibration-domain: a point lies outside the stable HRS branch; use pre-latch HRS data")
        lo, hi = b[k-1, 17], b[k, 17]
        def at_u(u):
            row = sm.branch_grid(p, np.array([u]))
            if len(row) != 1:
                raise ValueError("simple-calibration-domain: discontinuous HRS branch")
            return row[0]
        u = brentq(lambda x: at_u(x)[0]-vd, lo, hi, xtol=2e-14)
        result.append(float(at_u(u)[1]))
    return np.array(result)


def run_simple_calibrate(payload, progress=null_progress):
    started = time.perf_counter()
    warnings = []
    device = normalize_device(payload.get("device"), warnings)
    if device["model"] != "simple":
        raise ValueError("simple_calibrate requires device.model='simple'")
    fit = payload.get("fit", "is")
    if fit not in ("is", "tau", "is_tau"):
        raise ValueError("fit must be 'is', 'tau' or 'is_tau'")
    points = payload.get("points")
    if not isinstance(points, list) or not 1 <= len(points) <= 200:
        raise ValueError("provide 1 to 200 HRS points in volts and amperes")
    try:
        a = np.array([[float(v["vd_V"]), float(v["id_A"])] for v in points])
    except (TypeError, KeyError, ValueError, OverflowError):
        raise ValueError("each HRS point needs numeric vd_V and id_A") from None
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("HRS voltage and current must be finite and positive")
    if fit == "is_tau" and (len(a) < 3 or len(np.unique(a[:, 0])) < 3):
        raise ValueError("identifiability: fitting IS and tau together needs at least 3 distinct HRS voltages")
    p = np.array(params.build_p(device))
    beta, tau, cb, resistance, saturation, vbr, eta, cg, cbg, cs, bias = sm.effective_parameters(p)
    generation, y = [], []
    for vd, current in a:
        r = vd-current*resistance
        if not 0 < r < vbr:
            raise ValueError("simple-calibration-domain: V-I*RLRS must lie between 0 and VBR")
        mult = 1/(1-(r/vbr)**eta)
        bbj, gidl = sm.btbt_currents(r, p)
        other = bbj+gidl+p[13]
        seed = (current-other)/mult
        if seed <= 0:
            raise ValueError("simple-calibration-domain: measured current must exceed the fixed BTBT/GIDL/photo contribution")
        generation.append((mult-1)*seed+other-seed/beta)
        y.append(sm.VT*np.log(seed)+current*resistance-bias)
    generation, y = np.array(generation), np.array(y)
    progress(.25, "fitting the first-order HRS balance")
    log_is = np.log(saturation)
    if fit == "is":
        log_is = float(np.mean((y-tau/cb*generation)/sm.VT))
    elif fit == "tau":
        denom = float(generation@generation)
        if denom < 1e-60:
            raise ValueError("identifiability: HRS points do not constrain body lifetime")
        tau = cb*float(generation@(y-sm.VT*log_is))/denom
    else:
        scale = float(np.max(np.abs(generation)))
        spread = float(np.std(generation))
        if scale <= 1e-30 or spread/scale < 1e-4:
            raise ValueError("identifiability: HRS points are too similar to separate IS and tau; widen the pre-latch voltage interval")
        design = np.column_stack((generation/scale, np.ones(len(a))))
        coeff, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 2:
            raise ValueError("identifiability: degenerate HRS calibration")
        tau = cb*float(coeff[0])/scale
        log_is = float(coeff[1])/sm.VT
    # Translate effective values back to the unchanged reference geometry.
    if not np.isfinite(log_is) or not -800 < log_is < 0 or not np.isfinite(tau) or tau <= 0:
        raise ValueError("simple-calibration-fit: no positive physical IS/lifetime fits these points with the other parameters held fixed")
    is_ref = float(np.exp(log_is)*device["simple"]["is_ref_A"]/saturation)
    tau_ref = float(tau*device["simple"]["tau_body_s"]/sm.effective_parameters(p)[1])
    for key, value in (("is_ref_A", is_ref), ("tau_body_s", tau_ref)):
        lo, hi = SIMPLE_LIMITS[key]
        if not lo <= value <= hi:
            raise ValueError(f"simple-calibration-fit: fitted {key} is outside the supported range; check units and fixed parameters")
        if (key == "is_ref_A" and fit != "tau") or (key == "tau_body_s" and fit != "is"):
            device["simple"][key] = value
    progress(.6, "checking the stable HRS branch")
    pfit = np.array(params.build_p(device))
    predicted = _predict_hrs(pfit, a[:, 0])
    log_error = np.log10(predicted/a[:, 1])
    rmse = float(np.sqrt(np.mean(log_error**2)))
    # A wrong branch must not be labelled a successful HRS fit.
    if np.any(np.abs(log_error) > 1.0):
        raise ValueError("simple-calibration-fit: points do not match the stable HRS branch within one decade; check branch selection and fixed parameters")
    if rmse > .1:
        warnings.append("HRS fit residual exceeds 0.1 decade; review fixed beta, coupling, RLRS and generation parameters")
    warnings.append("HRS fit error is not independent validation; fold and transient accuracy require held-out data")
    if fit == "is_tau":
        warnings.append("CB is held fixed: DC HRS data constrain tau/CB, not tau and CB independently")
    progress(1., "done")
    return dict(device=device, model="simple", fit=fit, identifiable=True,
                points=[dict(vd_V=float(v), id_A=float(i), predicted_id_A=float(pr), error_log10=float(er))
                        for (v,i),pr,er in zip(a,predicted,log_error)],
                rmse_log10=rmse, warnings=warnings, runtime_s=time.perf_counter()-started)
