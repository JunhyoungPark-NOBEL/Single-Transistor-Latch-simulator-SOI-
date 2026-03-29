
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root_scalar


# ============================================================================
# Parameter model
# ============================================================================

@dataclass(frozen=True)
class STLParams:
    # Device
    nb_cm: float = 3e17
    lg: float = 0.5e-6
    w: float = 0.65e-6
    tsi: float = 50e-9
    tox: float = 13e-9
    tbox: float = 140e-9

    # Bias
    vg: float = -3.5
    vbg: float = 0.0
    vd_max: float = 4.0
    vstep: float = 0.005

    # Floating-body / electrostatics
    vfbeff: float = -3.35
    vfb0: float = -1.2
    rho0: float = 5e-3
    krl: float = 0.5

    # II
    vbr: float = 2.35
    k_mult: float = 4.0

    # BTBT
    bb_a: float = 4e14
    bb_b: float = 19e6
    bb_gamma: float = 2.5

    # Recombination / out-diffusion / beta
    tau: float = 3e-7
    beta0: float = 50.0
    beta_scale: float = 1.5
    gamma1_accum: float = 0.01
    gamma2_subthreshold: float = 0.0182

    # GIDL helper
    lov: float = 5e-9
    neff_cm: float = 7e19

    # Oscillation
    iin: float = 10e-9
    cd: float = 170e-12
    vd0: float = 0.0
    t_window: float = 5e-3
    auto_time_window: bool = True
    auto_cycles: float = 5.0

    # Independent strength scales
    btbt_scale: float = 1.0
    ii_scale: float = 1.0
    outdiff_scale: float = 1.0
    recomb_scale: float = 1.0

    # Numerics
    solver_tol: float = 1e-14
    root_grid_points: int = 96
    current_floor: float = 1e-18
    current_ceiling_factor: float = 200.0
    current_ceiling_abs: float = 1e-2
    snap_decades_min: float = 1.0


def consts() -> Dict[str, float]:
    eps0 = 8.854e-12
    return {
        "q": 1.602e-19,
        "eps_si": 11.7 * eps0,
        "eps_ox": 3.9 * eps0,
        "vt": 0.02585,
        "ni_m3": 1.5e10 * 1e6,
        "kappa": 1.0,
        "mcap": 1e12,
    }


def safe_exp(x):
    return np.exp(np.minimum(x, 600.0))


def to_hashable_dict(p: STLParams) -> Dict[str, float]:
    out = {}
    for k, v in asdict(p).items():
        if isinstance(v, (float, int, np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def trim_pair(x, y):
    n = min(len(x), len(y))
    return np.asarray(x[:n], float), np.asarray(y[:n], float)


# ============================================================================
# Physics model builder
# ============================================================================

def build_model(params: STLParams):
    c = consts()
    q = c["q"]
    eps_si = c["eps_si"]
    eps_ox = c["eps_ox"]
    vt = c["vt"]
    ni_m3 = c["ni_m3"]
    kappa = c["kappa"]
    mcap = c["mcap"]

    na = params.nb_cm * 1e6
    tsi = params.tsi  # unified thickness; exposed to UI as TSi only

    wb = params.lg
    a_eb = params.w * tsi
    mu_n_base = 0.05
    db = mu_n_base * vt
    i_s = 3.0 * q * a_eb * (db / (wb * na)) * (ni_m3 ** 2)

    rlrs = params.rho0 * (1e17 / params.nb_cm) ** params.krl * params.lg / (tsi * params.w)

    tau_eff = params.tau / max(params.recomb_scale, 1e-18)

    vgeff = params.vg - params.vfbeff
    vbgeff = params.vbg - params.vfbeff
    kvbi_exp_vg = 0.01 if params.vg < params.vfbeff else 0.2
    kvbi_exp_vbg = 0.1224
    kvbi_btbt = kvbi_exp_vg

    dvbi_btbt = -kvbi_btbt * vgeff
    dvbi_exp = kvbi_exp_vg * vgeff + kvbi_exp_vbg * vbgeff
    vbi0 = vt * np.log(1e20 * params.nb_cm / (1.5e10 ** 2))

    lamb = np.sqrt((eps_si / eps_ox) * tsi * params.tox)

    aj_cm2 = params.w * tsi * 1e4
    neff_m3 = params.neff_cm * 1e6
    wt_m = np.sqrt(min(2 * eps_si * 1.12 / (q * neff_m3), params.tsi))
    wt_cm = 100.0 * wt_m
    vgidl_cm3 = (params.w * params.lov * 1e6) * max(wt_cm, 1e-7)

    def wdep(v):
        v = np.asarray(v, float)
        return np.sqrt(np.maximum(2 * eps_si * np.maximum(v, 0.0) / (q * na), 0.0))

    def lbvd_vn(i_d, v_d):
        i_d = np.asarray(i_d, float)
        return np.maximum(params.lg - wdep(v_d - i_d * rlrs + 0.8) - wdep(0.2), 1e-9)

    def cb_fun(i_d, v_d):
        lb = lbvd_vn(i_d, v_d)
        cox = eps_ox * lb * params.w / params.tox
        csi = eps_si * lb * params.w / max(params.tsi, 1e-12)
        cbox = eps_ox * lb * params.w / max(params.tbox, 1e-12)
        cpar = (csi * cbox) / (csi + cbox + 1e-30)
        return cox + cpar

    def rrec_fun(i_d, v_d):
        return tau_eff / np.maximum(cb_fun(i_d, v_d), 1e-30)

    def delta_vdibl(v_d):
        return kappa * (np.asarray(v_d, float) + 1.0) * lamb

    def vbeff_bias(v_d):
        arg = (
            (params.vg - params.vfbeff) * params.gamma1_accum
            + (params.vbg - params.vfbeff) * params.gamma2_subthreshold
            + delta_vdibl(v_d)
        ) / (2 * vt)
        return np.exp(arg)

    def beta_fun(v_d):
        v_d = np.asarray(v_d, float)
        return (
            params.beta0 ** -1
            + (params.beta_scale * (0.2e-6 / params.lg) * vbeff_bias(v_d)) ** -1
        ) ** -1

    def vn_fun(i_d, v_d):
        return np.maximum(np.asarray(v_d, float) - np.asarray(i_d, float) * rlrs, 0.0)

    def mfun(vn):
        vn = np.asarray(vn, float)
        den = np.maximum(1.0 - (vn / params.vbr) ** params.k_mult, 1.0 / mcap)
        return np.minimum(1.0 / den, mcap)

    def wdepj_cm(i_d, v_d):
        vn = vn_fun(i_d, v_d)
        return 100.0 * np.sqrt(
            np.maximum(2 * eps_si * (vbi0 + dvbi_btbt + np.maximum(vn, 0.0)) / (q * na), 0.0)
        )

    def e_cm(i_d, v_d):
        return 2.0 * (vbi0 + dvbi_btbt + vn_fun(i_d, v_d)) / np.maximum(wdepj_cm(i_d, v_d), 1e-7)

    def ibtbt_junc_raw(i_d, v_d):
        e = np.maximum(e_cm(i_d, v_d), 1e3)
        g = params.bb_a * (e ** params.bb_gamma) * safe_exp(-params.bb_b / e)
        return q * g * (aj_cm2 * np.maximum(wdepj_cm(i_d, v_d), 1e-7))

    def igidl_raw(v_d):
        v_d = np.asarray(v_d, float)
        e = np.maximum(
            0.01 * ((v_d - params.vg + params.vfb0 + 1.12) / max(3 * params.tox, 1e-12)),
            1e3,
        )
        g = params.bb_a * (e ** params.bb_gamma) * safe_exp(-params.bb_b / e)
        return q * g * vgidl_cm3

    def ibtbt(i_d, v_d):
        return params.btbt_scale * (ibtbt_junc_raw(i_d, v_d) + igidl_raw(v_d))

    def dynamic_terms(i_d, v_d):
        i_d = np.asarray(i_d, float)
        v_d = np.asarray(v_d, float)
        vn = vn_fun(i_d, v_d)
        ib = ibtbt(i_d, v_d)
        rr = rrec_fun(i_d, v_d)
        mult = (vn / params.vbr) ** params.k_mult

        ii_v = params.ii_scale * mult * (i_d - ib) * rr
        outdiff_v = params.outdiff_scale * (1.0 - mult) * (i_d - ib) * rr / np.maximum(beta_fun(v_d), 1e-30)
        btbt_v = ib * rr
        resistive_v = i_d * rlrs

        body_core_v = btbt_v + ii_v - outdiff_v - resistive_v
        body_total_v = body_core_v + dvbi_exp

        return {
            "vn": vn,
            "ibtbt_current": ib,
            "recomb_resistance_ohm": rr,
            "ii_drive_v": ii_v,
            "outdiff_drive_v": outdiff_v,
            "btbt_drive_v": btbt_v,
            "series_drop_v": resistive_v,
            "body_potential_v": body_core_v,      # dynamic body term
            "body_total_exponent_v": body_total_v # body term + bias offset used in the exponential
        }

    def current_components(i_d, v_d):
        terms = dynamic_terms(i_d, v_d)
        bjt_current = i_s * np.exp(delta_vdibl(v_d) / vt) * safe_exp(terms["body_total_exponent_v"] / vt)
        ii_current = bjt_current * (mfun(terms["vn"]) - 1.0)
        total_model = terms["ibtbt_current"] + bjt_current + ii_current
        return {
            **terms,
            "bjt_current": bjt_current,
            "ii_current": ii_current,
            "total_model_current": total_model,
            "beta_eff": beta_fun(v_d),
            "m_factor": mfun(terms["vn"]),
            "dvbi_exp_v": np.asarray(v_d, float) * 0 + dvbi_exp,
            "dvbi_btbt_v": np.asarray(v_d, float) * 0 + dvbi_btbt,
            "i_s": np.asarray(v_d, float) * 0 + i_s,
        }

    def f_id(i_d, v_d):
        comps = current_components(i_d, v_d)
        return np.asarray(i_d, float) - comps["total_model_current"]

    return {
        "q": q,
        "vt": vt,
        "i_s_scalar": i_s,
        "rlrs_scalar": rlrs,
        "dvbi_exp_scalar": dvbi_exp,
        "dvbi_btbt_scalar": dvbi_btbt,
        "f_id": f_id,
        "current_components": current_components,
    }


# ============================================================================
# Root search / branch following
# ============================================================================

def unique_roots(roots: Sequence[float], tol_log: float = 1e-2) -> List[float]:
    roots = [float(r) for r in roots if np.isfinite(r) and r >= 0]
    if not roots:
        return []
    roots = sorted(roots)
    out = [roots[0]]
    for r in roots[1:]:
        if abs(np.log(max(r, 1e-300)) - np.log(max(out[-1], 1e-300))) > tol_log:
            out.append(r)
    return out


def choose_branch_root(roots: Sequence[float], target_i: float) -> Optional[float]:
    if not roots:
        return None
    arr = np.asarray(roots, float)
    target_i = max(float(target_i), 1e-300)
    idx = np.argmin(np.abs(np.log(np.maximum(arr, 1e-300)) - np.log(target_i)))
    return float(arr[idx])


def find_roots_and_minres(fun_vec, target_i: float, i_low: float, i_high: float, ngrid: int, tol: float):
    i_low = max(i_low, 1e-18)
    i_high = max(i_high, i_low * 10.0)

    grid = np.geomspace(i_low, i_high, int(ngrid))
    vals = np.asarray(fun_vec(grid), float)

    roots: List[float] = []
    prod = vals[:-1] * vals[1:]
    sign_mask = np.isfinite(prod) & (prod <= 0)
    idxs = np.where(sign_mask)[0]

    def f_scalar(x: float) -> float:
        return float(fun_vec(np.asarray([x], float))[0])

    for j in idxs:
        a = float(grid[j])
        b = float(grid[j + 1])
        if a == b:
            continue
        try:
            sol = root_scalar(f_scalar, bracket=[a, b], method="brentq", xtol=tol, maxiter=120)
            if sol.converged and np.isfinite(sol.root) and sol.root >= 0:
                roots.append(float(sol.root))
        except Exception:
            pass

    roots = unique_roots(roots, tol_log=1e-2)

    finite = np.isfinite(vals)
    minres = None
    minres_residual = np.nan
    if finite.any():
        target_i = max(float(target_i), 1e-300)
        score = np.log10(np.abs(vals) + 1e-300) + 0.03 * np.abs(np.log(np.maximum(grid, 1e-300) / target_i))
        score = np.where(finite, score, np.inf)
        idx_best = int(np.argmin(score))
        minres = float(grid[idx_best])
        minres_residual = float(vals[idx_best])

    return roots, minres, minres_residual


def extract_vlu_vld(vfwd, idfwd, vrev, idrev):
    vfwd, idfwd = trim_pair(vfwd, idfwd)
    vrev, idrev = trim_pair(vrev, idrev)

    valid_fwd = np.isfinite(idfwd) & (idfwd > 0)
    valid_rev = np.isfinite(idrev) & (idrev > 0)
    if valid_fwd.sum() < 10 or valid_rev.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan

    vf = vfwd[valid_fwd]
    ifw = idfwd[valid_fwd]
    vr = vrev[valid_rev]
    irv = idrev[valid_rev]

    idx_l_end = max(3, int(np.floor(0.2 * len(ifw))))
    idx_h_start = max(idx_l_end + 1, int(np.floor(0.7 * len(ifw))))
    ioff = np.mean(np.maximum(ifw[:idx_l_end], np.finfo(float).tiny))
    ion = np.mean(np.maximum(ifw[idx_h_start:], np.finfo(float).tiny))
    if not (ion > ioff):
        return np.nan, np.nan, np.nan, np.nan

    ith = np.sqrt(ioff * ion)

    idx_lu = np.where((ifw[1:] >= ith) & (ifw[:-1] < ith))[0]
    idx_ld = np.where((irv[1:] <= ith) & (irv[:-1] > ith))[0]

    vlu = vf[idx_lu[0] + 1] if len(idx_lu) else np.nan
    ilu = ifw[idx_lu[0] + 1] if len(idx_lu) else np.nan
    vld = vr[idx_ld[0] + 1] if len(idx_ld) else np.nan
    ild = irv[idx_ld[0] + 1] if len(idx_ld) else np.nan
    return vlu, ilu, vld, ild


def detect_snap_indices(idfwd: np.ndarray, idrev: np.ndarray, snap_decades_min: float) -> Tuple[Optional[int], Optional[int], float, float]:
    log_ifwd = np.log10(np.maximum(np.asarray(idfwd, float), 1e-300))
    log_irev = np.log10(np.maximum(np.asarray(idrev, float), 1e-300))

    d_fwd = np.diff(log_ifwd)
    d_rev = np.diff(log_irev)

    if d_fwd.size == 0 or d_rev.size == 0:
        return None, None, np.nan, np.nan

    idx_up = int(np.argmax(d_fwd))
    idx_down = int(np.argmin(d_rev))
    jump_up = float(d_fwd[idx_up])
    jump_down = float(-d_rev[idx_down])

    if jump_up < snap_decades_min:
        idx_up = None
    if jump_down < snap_decades_min:
        idx_down = None

    return idx_up, idx_down, jump_up, jump_down


# ============================================================================
# DC simulation
# ============================================================================

def evaluate_solution_branches(model, vd_arr: np.ndarray, id_arr: np.ndarray) -> Dict[str, np.ndarray]:
    comps = model["current_components"](id_arr, vd_arr)
    return {k: np.asarray(v, float) for k, v in comps.items()}


def simulate_double_sweep(params: STLParams) -> Dict:
    model = build_model(params)

    vfwd = np.arange(0.0, params.vd_max + params.vstep / 2, params.vstep)
    vrev_full = np.arange(params.vd_max, -params.vstep / 2, -params.vstep)
    vvec = np.concatenate([vfwd, vrev_full[1:]])
    n_fwd = len(vfwd)

    ids = np.zeros_like(vvec, dtype=float)
    residuals = np.zeros_like(vvec, dtype=float)
    root_count = np.zeros_like(vvec, dtype=int)

    prev_i = max(model["i_s_scalar"], params.current_floor)
    prev_prev_i = None

    nominal_ceiling = max(1e-6, params.current_ceiling_factor * params.vd_max / max(model["rlrs_scalar"], 1e-30))
    nominal_ceiling = min(nominal_ceiling, params.current_ceiling_abs)

    for k, vd in enumerate(vvec):
        if prev_prev_i is not None and prev_i > 0 and prev_prev_i > 0:
            dlog = np.log(prev_i) - np.log(prev_prev_i)
            dlog = float(np.clip(dlog, -1.5, 1.5))
            target_i = float(np.exp(np.log(prev_i) + dlog))
        else:
            target_i = prev_i

        i_high = max(nominal_ceiling, target_i * params.current_ceiling_factor)
        i_high = min(i_high, params.current_ceiling_abs)

        roots, minres, _ = find_roots_and_minres(
            lambda x, vd_=vd: model["f_id"](x, vd_),
            target_i=target_i,
            i_low=params.current_floor,
            i_high=i_high,
            ngrid=params.root_grid_points,
            tol=params.solver_tol,
        )

        chosen = choose_branch_root(roots, target_i)
        if chosen is None:
            chosen = minres if minres is not None else prev_i

        ids[k] = float(chosen)
        residuals[k] = float(model["f_id"](np.asarray([chosen]), vd)[0])
        root_count[k] = len(roots)
        prev_prev_i, prev_i = prev_i, max(chosen, params.current_floor)

    idfwd = ids[:n_fwd]
    vrev = vvec[n_fwd:]
    idrev = ids[n_fwd:]

    vlu, ilu, vld, ild = extract_vlu_vld(vfwd, idfwd, vrev, idrev)
    up_idx, down_idx, jump_up_dec, jump_down_dec = detect_snap_indices(idfwd, idrev, params.snap_decades_min)

    fwd_q = evaluate_solution_branches(model, vfwd, idfwd)
    rev_q = evaluate_solution_branches(model, vrev, idrev)

    snap_up_v = float(vfwd[up_idx]) if up_idx is not None else np.nan
    snap_up_next_v = float(vfwd[up_idx + 1]) if up_idx is not None and up_idx + 1 < len(vfwd) else np.nan
    snap_down_v = float(vrev[down_idx]) if down_idx is not None else np.nan
    snap_down_next_v = float(vrev[down_idx + 1]) if down_idx is not None and down_idx + 1 < len(vrev) else np.nan

    return {
        "vfwd": vfwd,
        "idfwd": idfwd,
        "vrev": vrev,
        "idrev": idrev,
        "vlu": vlu,
        "ilu": ilu,
        "vld": vld,
        "ild": ild,
        "params": to_hashable_dict(params),
        "residuals": residuals,
        "root_count": root_count,
        "model_rlrs": model["rlrs_scalar"],
        "model_is": model["i_s_scalar"],
        "fwd": fwd_q,
        "rev": rev_q,
        "snap_up_idx": up_idx,
        "snap_down_idx": down_idx,
        "snap_up_decades": jump_up_dec,
        "snap_down_decades": jump_down_dec,
        "snap_up_v": snap_up_v,
        "snap_up_next_v": snap_up_next_v,
        "snap_down_v": snap_down_v,
        "snap_down_next_v": snap_down_next_v,
    }


@st.cache_data(show_spinner=False)
def simulate_double_sweep_cached(param_dict: Dict) -> Dict:
    return simulate_double_sweep(STLParams(**param_dict))


# ============================================================================
# Oscillation
# ============================================================================

def build_oscillation_branches(dc: Dict, snap_decades_min: float) -> Optional[Dict]:
    vfwd = np.asarray(dc["vfwd"], float)
    idfwd = np.asarray(dc["idfwd"], float)
    vrev = np.asarray(dc["vrev"], float)
    idrev = np.asarray(dc["idrev"], float)

    up_idx, down_idx, jump_up_dec, jump_down_dec = detect_snap_indices(idfwd, idrev, snap_decades_min)
    if up_idx is None or down_idx is None:
        return None

    low_v = vfwd[: up_idx + 1].copy()
    low_i = idfwd[: up_idx + 1].copy()

    high_v_desc = vrev[: down_idx + 1].copy()
    high_i_desc = idrev[: down_idx + 1].copy()

    order = np.argsort(high_v_desc)
    high_v = high_v_desc[order]
    high_i = high_i_desc[order]

    if len(low_v) < 4 or len(high_v) < 4:
        return None

    overlap_low = max(float(high_v.min()), float(low_v.min()))
    overlap_high = min(float(high_v.max()), float(low_v.max()))
    if not (overlap_high > overlap_low):
        return None

    return {
        "low_v": low_v,
        "low_i": low_i,
        "high_v": high_v,
        "high_i": high_i,
        "overlap_low": overlap_low,
        "overlap_high": overlap_high,
        "jump_up_dec": jump_up_dec,
        "jump_down_dec": jump_down_dec,
        "up_idx": up_idx,
        "down_idx": down_idx,
    }


def estimate_oscillation_period(branches: Dict, iin: float, cd: float) -> Tuple[float, Dict]:
    low_v = np.asarray(branches["low_v"], float)
    low_i = np.asarray(branches["low_i"], float)
    high_v = np.asarray(branches["high_v"], float)
    high_i = np.asarray(branches["high_i"], float)

    lo = float(branches["overlap_low"])
    hi = float(branches["overlap_high"])
    if not (hi > lo):
        return np.nan, {"reason": "No overlap between charge and discharge ranges in the DC sweep."}

    low_fun = PchipInterpolator(low_v, low_i, extrapolate=False)
    high_fun = PchipInterpolator(high_v, high_i, extrapolate=False)

    vg = np.linspace(lo, hi, 1200)
    ilow = np.asarray(low_fun(vg), float)
    ihigh = np.asarray(high_fun(vg), float)

    if not np.all(np.isfinite(ilow)) or not np.all(np.isfinite(ihigh)):
        return np.nan, {"reason": "Transient time-window estimate failed during interpolation."}

    net_charge = iin - ilow
    net_discharge = ihigh - iin

    if np.min(net_charge) <= 0:
        return np.nan, {
            "reason": "IIN is too small to charge through the lower branch in the overlap range.",
            "min_charge_margin_a": float(np.min(net_charge)),
        }

    if np.min(net_discharge) <= 0:
        return np.nan, {
            "reason": "IIN is too large to discharge through the upper branch in the overlap range.",
            "min_discharge_margin_a": float(np.min(net_discharge)),
        }

    t_charge = float(cd * np.trapezoid(1.0 / net_charge, vg))
    t_discharge = float(cd * np.trapezoid(1.0 / net_discharge, vg))
    return t_charge + t_discharge, {
        "t_charge_s": t_charge,
        "t_discharge_s": t_discharge,
        "overlap_low_v": lo,
        "overlap_high_v": hi,
    }


def estimate_time_window(params: STLParams, dc_result: Dict) -> Tuple[float, float, Dict]:
    if not params.auto_time_window:
        return max(params.t_window, 1e-9), np.nan, {"source": "manual"}

    branches = build_oscillation_branches(dc_result, params.snap_decades_min)
    if branches is None:
        return max(params.t_window, 1e-9), np.nan, {
            "source": "manual",
            "reason": "No clear DC snap pair was found, so the manual window was used.",
        }

    period_est, diag = estimate_oscillation_period(branches, params.iin, params.cd)
    low_fun = PchipInterpolator(branches["low_v"], branches["low_i"], extrapolate=False)
    v_start = max(float(params.vd0), float(branches["low_v"].min()))
    v_up = float(branches["low_v"].max())
    startup_time = np.nan
    if v_up > v_start:
        vg = np.linspace(v_start, v_up, 1600)
        ilow = np.asarray(low_fun(vg), float)
        net = params.iin - ilow
        if np.all(np.isfinite(net)) and np.min(net) > 0:
            startup_time = float(params.cd * np.trapezoid(1.0 / net, vg))

    if not np.isfinite(period_est) or period_est <= 0:
        out = {"source": "manual", "startup_time_s": float(startup_time) if np.isfinite(startup_time) else np.nan}
        out.update(diag)
        return max(params.t_window, 1e-9), np.nan, out

    if not np.isfinite(startup_time) or startup_time < 0:
        startup_time = 0.0

    t_stop = max(startup_time + params.auto_cycles * period_est, startup_time + 1.5 * period_est, 1e-9)
    out = {"source": "dc_estimate", "startup_time_s": float(startup_time)}
    out.update(diag)
    return t_stop, period_est, out


def solve_static_current_transient(model: Dict, params: STLParams, vd: float, seed: float) -> float:
    vd = float(max(vd, 0.0))
    seed = max(float(seed), params.current_floor)

    def f_scalar(x: float) -> float:
        return float(model["f_id"](np.asarray([x], float), vd)[0])

    x1 = max(seed * 1.02, seed + 1e-18)
    try:
        sol = root_scalar(f_scalar, x0=seed, x1=x1, method="secant", xtol=params.solver_tol, maxiter=80)
        if sol.converged and np.isfinite(sol.root) and sol.root >= 0:
            return float(sol.root)
    except Exception:
        pass

    nominal_ceiling = max(1e-6, params.current_ceiling_factor * max(vd, 1e-3) / max(model["rlrs_scalar"], 1e-30))
    i_high = min(max(nominal_ceiling, seed * params.current_ceiling_factor), params.current_ceiling_abs)
    roots, minres, _ = find_roots_and_minres(
        lambda x, vd_=vd: model["f_id"](x, vd_),
        target_i=seed,
        i_low=params.current_floor,
        i_high=i_high,
        ngrid=params.root_grid_points,
        tol=params.solver_tol,
    )
    chosen = choose_branch_root(roots, seed)
    if chosen is not None:
        return float(chosen)
    if minres is not None and np.isfinite(minres) and minres >= 0:
        return float(minres)
    return float(seed)


def estimate_frequency_from_trace(t: np.ndarray, vd_t: np.ndarray) -> Tuple[float, float, np.ndarray]:
    t = np.asarray(t, float)
    vd_t = np.asarray(vd_t, float)
    if t.size < 6:
        return np.nan, np.nan, np.array([], dtype=int)

    dv = np.diff(vd_t)
    s = np.sign(dv)
    peak_idx = np.where((s[:-1] > 0) & (s[1:] <= 0))[0] + 1
    if peak_idx.size == 0:
        return np.nan, np.nan, peak_idx

    late_mask = t[peak_idx] >= (t[0] + 0.2 * (t[-1] - t[0]))
    late_peaks = peak_idx[late_mask]
    if late_peaks.size >= 2:
        peak_idx = late_peaks

    if peak_idx.size < 2:
        return np.nan, np.nan, peak_idx

    periods = np.diff(t[peak_idx])
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if periods.size == 0:
        return np.nan, np.nan, peak_idx

    period = float(np.mean(periods))
    return 1.0 / period, period, peak_idx


def simulate_oscillation(params: STLParams, dc_result: Optional[Dict] = None) -> Dict:
    if dc_result is None:
        dc_result = simulate_double_sweep(params)

    model = build_model(params)
    t_stop, period_guess, window_diag = estimate_time_window(params, dc_result)
    startup_time = float(window_diag.get("startup_time_s", np.nan))

    if np.isfinite(period_guess) and period_guess > 0:
        cycles_total = params.auto_cycles + (startup_time / period_guess if np.isfinite(startup_time) and startup_time > 0 else 1.0)
        n_eval = int(np.clip(np.ceil(180.0 * cycles_total), 1200, 2600))
    else:
        n_eval = 1400
    t_arr = np.linspace(0.0, t_stop, n_eval)

    vd_arr = np.zeros(n_eval, dtype=float)
    id_arr = np.zeros(n_eval, dtype=float)
    vd_arr[0] = max(params.vd0, 0.0)

    seed = max(model["i_s_scalar"], params.current_floor)
    for k in range(n_eval - 1):
        dt = t_arr[k + 1] - t_arr[k]
        id_now = solve_static_current_transient(model, params, float(vd_arr[k]), seed)
        id_arr[k] = id_now
        dvdt_1 = (params.iin - id_now) / max(params.cd, 1e-30)

        vd_pred = max(vd_arr[k] + dt * dvdt_1, 0.0)
        id_pred = solve_static_current_transient(model, params, float(vd_pred), max(id_now, params.current_floor))
        dvdt_2 = (params.iin - id_pred) / max(params.cd, 1e-30)

        vd_arr[k + 1] = max(vd_arr[k] + 0.5 * dt * (dvdt_1 + dvdt_2), 0.0)
        seed = max(id_pred, params.current_floor)

    id_arr[-1] = solve_static_current_transient(model, params, float(vd_arr[-1]), seed)

    iin_arr = np.full_like(t_arr, params.iin, dtype=float)
    inet_arr = iin_arr - id_arr
    qcap_arr = params.cd * vd_arr
    comps = model["current_components"](id_arr, vd_arr)

    freq, measured_period, peak_idx = estimate_frequency_from_trace(t_arr, vd_arr)
    period_out = measured_period if np.isfinite(measured_period) else period_guess

    diagnostics = {
        "solver_method": "direct_time_step_heun",
        "solver_success": True,
        "solver_message": "Direct time stepping finished.",
        "solver_exception": "",
        "nfev": int(2 * (n_eval - 1)),
        "time_window_s": float(t_stop),
        "dt_s": float(t_arr[1] - t_arr[0]) if n_eval > 1 else np.nan,
        "period_guess_from_dc_s": float(period_guess) if np.isfinite(period_guess) else np.nan,
        "measured_period_s": float(measured_period) if np.isfinite(measured_period) else np.nan,
        "peak_count": int(len(peak_idx)),
        "startup_from_zero": True,
        **window_diag,
    }

    return {
        "t": t_arr,
        "vd_t": vd_arr,
        "id_t": id_arr,
        "iin_t": iin_arr,
        "inet_t": inet_arr,
        "qcap_t": qcap_arr,
        "freq": float(freq) if np.isfinite(freq) else np.nan,
        "estimated_period_s": float(period_out) if np.isfinite(period_out) else np.nan,
        "diagnostics": diagnostics,
        "body_potential_t": np.asarray(comps["body_potential_v"], float),
        "peak_idx": np.asarray(peak_idx, int),
    }


@st.cache_data(show_spinner=False)
def simulate_oscillation_cached(param_dict: Dict, dc_result: Dict) -> Dict:
    return simulate_oscillation(STLParams(**param_dict), dc_result)


# ============================================================================
# Export helpers
# ============================================================================

def build_excel_bytes(dc_rows: List[Tuple[str, Dict]], osc_result: Optional[Dict] = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_rows = []
        for label, res in dc_rows:
            p = res["params"]
            summary_rows.append({
                "Case": label,
                "VG_V": p["vg"],
                "VBG_V": p["vbg"],
                "NB_cm-3": p["nb_cm"],
                "LG_um": p["lg"] * 1e6,
                "W_um": p["w"] * 1e6,
                "TSi_nm": p["tsi"] * 1e9,
                "Tox_nm": p["tox"] * 1e9,
                "TBOX_nm": p["tbox"] * 1e9,
                "BTBT_scale": p["btbt_scale"],
                "II_scale": p["ii_scale"],
                "OutDiff_scale": p["outdiff_scale"],
                "Recomb_scale": p["recomb_scale"],
                "VLU_V": res["vlu"],
                "VLD_V": res["vld"],
                "SnapUp_pre_V": res["snap_up_v"],
                "SnapUp_post_V": res["snap_up_next_v"],
                "SnapDown_pre_V": res["snap_down_v"],
                "SnapDown_post_V": res["snap_down_next_v"],
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        for label, res in dc_rows:
            n = max(len(res["vfwd"]), len(res["vrev"]))
            df = pd.DataFrame({
                "Vfwd_V": list(res["vfwd"]) + [np.nan] * (n - len(res["vfwd"])),
                "IDfwd_A": list(res["idfwd"]) + [np.nan] * (n - len(res["idfwd"])),
                "BTBT_fwd_A": list(res["fwd"]["ibtbt_current"]) + [np.nan] * (n - len(res["fwd"]["ibtbt_current"])),
                "BJT_fwd_A": list(res["fwd"]["bjt_current"]) + [np.nan] * (n - len(res["fwd"]["bjt_current"])),
                "II_fwd_A": list(res["fwd"]["ii_current"]) + [np.nan] * (n - len(res["fwd"]["ii_current"])),
                "Body_fwd_V": list(res["fwd"]["body_potential_v"]) + [np.nan] * (n - len(res["fwd"]["body_potential_v"])),
                "Vrev_V": list(res["vrev"]) + [np.nan] * (n - len(res["vrev"])),
                "IDrev_A": list(res["idrev"]) + [np.nan] * (n - len(res["idrev"])),
                "BTBT_rev_A": list(res["rev"]["ibtbt_current"]) + [np.nan] * (n - len(res["rev"]["ibtbt_current"])),
                "BJT_rev_A": list(res["rev"]["bjt_current"]) + [np.nan] * (n - len(res["rev"]["bjt_current"])),
                "II_rev_A": list(res["rev"]["ii_current"]) + [np.nan] * (n - len(res["rev"]["ii_current"])),
                "Body_rev_V": list(res["rev"]["body_potential_v"]) + [np.nan] * (n - len(res["rev"]["body_potential_v"])),
            })
            sheet = label[:31] if label else "Case"
            df.to_excel(writer, sheet_name=sheet, index=False)

        if osc_result is not None:
            pd.DataFrame({
                "t_s": osc_result["t"],
                "Vd_V": osc_result["vd_t"],
                "Id_A": osc_result["id_t"],
                "Iin_A": osc_result["iin_t"],
                "Iin_minus_Id_A": osc_result["inet_t"],
                "Qcap_C": osc_result["qcap_t"],
                "Body_V": osc_result["body_potential_t"],
            }).to_excel(writer, sheet_name="Oscillation", index=False)

    buf.seek(0)
    return buf.getvalue()


# ============================================================================
# Plotting
# ============================================================================

def fig_dc(results: List[Tuple[str, Dict]]) -> go.Figure:
    fig = go.Figure()
    for label, res in results:
        fig.add_trace(
            go.Scatter(
                x=res["vfwd"],
                y=np.maximum(res["idfwd"], np.finfo(float).tiny),
                mode="lines",
                name=f"{label} forward",
                hovertemplate="VD=%{x:.4f} V<br>ID=%{y:.4e} A<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=res["vrev"],
                y=np.maximum(res["idrev"], np.finfo(float).tiny),
                mode="lines",
                name=f"{label} reverse",
                line=dict(dash="dash"),
                hovertemplate="VD=%{x:.4f} V<br>ID=%{y:.4e} A<extra></extra>",
            )
        )

        if np.isfinite(res["vlu"]) and np.isfinite(res["ilu"]):
            fig.add_trace(
                go.Scatter(
                    x=[res["vlu"]],
                    y=[max(res["ilu"], np.finfo(float).tiny)],
                    mode="markers",
                    name=f"{label} VLU",
                    marker=dict(symbol="circle", size=10),
                    hovertemplate="VLU=%{x:.4f} V<br>ID=%{y:.4e} A<extra></extra>",
                )
            )
        if np.isfinite(res["vld"]) and np.isfinite(res["ild"]):
            fig.add_trace(
                go.Scatter(
                    x=[res["vld"]],
                    y=[max(res["ild"], np.finfo(float).tiny)],
                    mode="markers",
                    name=f"{label} VLD",
                    marker=dict(symbol="square", size=10),
                    hovertemplate="VLD=%{x:.4f} V<br>ID=%{y:.4e} A<extra></extra>",
                )
            )

    fig.update_layout(
        height=620,
        legend=dict(orientation="h", y=-0.20),
        margin=dict(l=10, r=10, t=40, b=80),
        title="ID–VD double sweep",
    )
    fig.update_xaxes(title="V_D [V]")
    fig.update_yaxes(title="I_D [A]", type="log")
    return fig


def fig_internal_quantities(res: Dict) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    current_floor = np.finfo(float).tiny

    quantities = [
        ("ibtbt_current", "BTBT current", "forward", "solid"),
        ("bjt_current", "BJT current", "forward", "solid"),
        ("ii_current", "II current", "forward", "solid"),
        ("body_potential_v", "Body potential", "forward", "solid"),
        ("ibtbt_current", "BTBT current", "reverse", "dash"),
        ("bjt_current", "BJT current", "reverse", "dash"),
        ("ii_current", "II current", "reverse", "dash"),
        ("body_potential_v", "Body potential", "reverse", "dash"),
    ]

    for key, label, direction, dash in quantities:
        if direction == "forward":
            x = res["vfwd"]
            y = np.asarray(res["fwd"][key], float)
        else:
            x = res["vrev"]
            y = np.asarray(res["rev"][key], float)

        is_body = key == "body_potential_v"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y if is_body else np.maximum(y, current_floor),
                mode="lines",
                name=f"{label} ({direction})",
                line=dict(dash=dash),
                legendgroup=label,
                hovertemplate="VD=%{x:.4f} V<br>" + ("Body=%{y:.4f} V" if is_body else "I=%{y:.4e} A") + "<extra></extra>",
            ),
            secondary_y=is_body,
        )

    fig.update_layout(
        height=560,
        legend=dict(orientation="h", y=-0.23),
        margin=dict(l=10, r=10, t=40, b=90),
        title="Internal quantities — click legend items to hide/show",
    )
    fig.update_xaxes(title="V_D [V]")
    fig.update_yaxes(title_text="Current [A]", type="log", secondary_y=False)
    fig.update_yaxes(title_text="Body potential [V]", secondary_y=True)
    return fig


def fig_oscillation(osc: Dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    fig.add_trace(
        go.Scatter(
            x=osc["t"],
            y=osc["vd_t"],
            mode="lines",
            name="V_D(t)",
            hovertemplate="t=%{x:.4e} s<br>V_D=%{y:.4f} V<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=osc["t"],
            y=np.maximum(osc["id_t"], np.finfo(float).tiny),
            mode="lines",
            name="I_D(t)",
            hovertemplate="t=%{x:.4e} s<br>I_D=%{y:.4e} A<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=osc["t"],
            y=np.maximum(osc["iin_t"], np.finfo(float).tiny),
            mode="lines",
            name="I_IN",
            line=dict(dash="dash"),
            hovertemplate="t=%{x:.4e} s<br>I_IN=%{y:.4e} A<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=640,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=10, r=10, t=40, b=70),
        title="Transient response from t = 0",
    )
    fig.update_xaxes(title="t [s]", row=2, col=1)
    fig.update_yaxes(title="V_D [V]", row=1, col=1)
    fig.update_yaxes(title="Current [A]", type="log", row=2, col=1)
    return fig


def fig_oscillation_idvd(dc: Dict, osc: Dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dc["vfwd"],
            y=np.maximum(dc["idfwd"], np.finfo(float).tiny),
            mode="lines",
            name="DC forward",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dc["vrev"],
            y=np.maximum(dc["idrev"], np.finfo(float).tiny),
            mode="lines",
            name="DC reverse",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=osc["vd_t"],
            y=np.maximum(osc["id_t"], np.finfo(float).tiny),
            mode="lines",
            name="Transient trajectory",
            line=dict(width=2.5),
        )
    )
    if len(osc["vd_t"]) > 0:
        fig.add_trace(
            go.Scatter(
                x=[osc["vd_t"][0]],
                y=[max(osc["id_t"][0], np.finfo(float).tiny)],
                mode="markers",
                name="Start",
                marker=dict(size=10, symbol="circle"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[osc["vd_t"][-1]],
                y=[max(osc["id_t"][-1], np.finfo(float).tiny)],
                mode="markers",
                name="End",
                marker=dict(size=10, symbol="x"),
            )
        )
    fig.update_layout(
        height=520,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=10, r=10, t=40, b=70),
        title="Transient trajectory on the ID–VD plane",
    )
    fig.update_xaxes(title="V_D [V]")
    fig.update_yaxes(title="I_D [A]", type="log")
    return fig


# ============================================================================
# Device schematic
# ============================================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def soi_svg(params: STLParams) -> str:
    lg_um = params.lg * 1e6
    tsi_nm = params.tsi * 1e9
    tox_nm = params.tox * 1e9
    tbox_nm = params.tbox * 1e9

    gate_w = clamp(120 + 85 * (lg_um / 0.5 - 1.0), 86, 186)
    tsi_h = clamp(32 + 0.55 * tsi_nm, 30, 86)
    tox_h = clamp(12 + 0.45 * tox_nm, 12, 28)
    box_h = clamp(22 + 0.16 * tbox_nm, 24, 56)
    sub_h = 92

    left_margin = 64
    right_margin = 86
    total_w = 470
    device_w = total_w - left_margin - right_margin
    source_w = clamp((device_w - gate_w) * 0.50, 78, 118)
    drain_w = device_w - gate_w - source_w

    x0 = left_margin
    y_gate = 28
    gate_h = 26
    y_tox = y_gate + gate_h
    y_tsi = y_tox + tox_h
    y_box = y_tsi + tsi_h
    y_sub = y_box + box_h

    x_source = x0
    x_gate = x0 + source_w
    x_drain = x_gate + gate_w
    gate_center = x_gate + gate_w / 2.0

    svg = f"""
    <svg width="100%" viewBox="0 0 520 360" xmlns="http://www.w3.org/2000/svg">
      <style>
        .lbl {{ font: 14px sans-serif; fill: #2b2b2b; font-weight: 600; }}
        .mid {{ font: 12px sans-serif; fill: #253858; }}
        .small {{ font: 11px sans-serif; fill: #54637a; }}
        .edge {{ stroke: #333; stroke-width: 1.2; }}
        .tox {{ fill: #7fb3ff; stroke: #333; stroke-width: 1.0; }}
        .boxf {{ fill: #a9c8f2; stroke: #333; stroke-width: 1.0; }}
        .si {{ fill: #e7b0b0; stroke: #333; stroke-width: 1.0; }}
        .sd {{ fill: #de8282; stroke: #333; stroke-width: 1.0; }}
        .gatef {{ fill: #5a2335; stroke: #333; stroke-width: 1.0; }}
        .subf {{ fill: #d9d9d9; stroke: #333; stroke-width: 1.0; }}
        .dim {{ stroke: #4b4b4b; stroke-width: 1.0; marker-start: url(#a); marker-end: url(#a); }}
      </style>
      <defs>
        <marker id="a" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
          <path d="M0,3 L6,0 L6,6 z" fill="#4b4b4b"/>
        </marker>
      </defs>

      <rect x="{x0}" y="{y_tox}" width="{device_w}" height="{tox_h}" class="tox"/>
      <rect x="{x0}" y="{y_tsi}" width="{device_w}" height="{tsi_h}" class="si"/>
      <rect x="{x0}" y="{y_box}" width="{device_w}" height="{box_h}" class="boxf"/>
      <rect x="{x0}" y="{y_sub}" width="{device_w}" height="{sub_h}" class="subf"/>

      <rect x="{x_source}" y="{y_tsi}" width="{source_w}" height="{tsi_h}" class="sd"/>
      <rect x="{x_drain}" y="{y_tsi}" width="{drain_w}" height="{tsi_h}" class="sd"/>

      <rect x="{x_gate}" y="{y_gate}" width="{gate_w}" height="{gate_h}" class="gatef"/>

      <circle cx="{x_source+18}" cy="{y_gate-16}" r="6" fill="white" stroke="#333" stroke-width="1.2"/>
      <circle cx="{x_drain+drain_w-18}" cy="{y_gate-16}" r="6" fill="white" stroke="#333" stroke-width="1.2"/>

      <text x="{x_source+10}" y="{y_gate-28}" class="lbl">Source</text>
      <text x="{gate_center-17}" y="{y_gate-28}" class="lbl">Gate</text>
      <text x="{x_drain+drain_w-46}" y="{y_gate-28}" class="lbl">Drain</text>

      <text x="{gate_center-18}" y="{y_gate + gate_h/2 + 4:.1f}" class="mid" fill="white">Gate</text>
      <text x="{gate_center-30}" y="{y_tox + tox_h/2 + 4:.1f}" class="mid">Oxide</text>
      <text x="{gate_center-36}" y="{y_tsi + tsi_h/2 + 4:.1f}" class="mid">TSi / p-body</text>
      <text x="{gate_center-11}" y="{y_box + box_h/2 + 4:.1f}" class="mid">BOX</text>
      <text x="{gate_center-33}" y="{y_sub + sub_h/2 + 4:.1f}" class="mid">p-substrate</text>

      <text x="{x_source + source_w/2 - 10}" y="{y_tsi + tsi_h/2 + 4:.1f}" class="mid">n+</text>
      <text x="{x_drain + drain_w/2 - 10}" y="{y_tsi + tsi_h/2 + 4:.1f}" class="mid">n+</text>

      <line x1="{x0 + device_w + 24}" y1="{y_tsi}" x2="{x0 + device_w + 24}" y2="{y_box}" class="dim"/>
      <text x="{x0 + device_w + 34}" y="{(y_tsi + y_box)/2 + 4:.1f}" class="small">TSi={tsi_nm:.1f} nm</text>

      <line x1="{x0 + device_w + 54}" y1="{y_tox}" x2="{x0 + device_w + 54}" y2="{y_tsi}" class="dim"/>
      <text x="{x0 + device_w + 64}" y="{(y_tox + y_tsi)/2 + 4:.1f}" class="small">Tox={tox_nm:.1f} nm</text>

      <line x1="{x0 + device_w + 84}" y1="{y_box}" x2="{x0 + device_w + 84}" y2="{y_sub}" class="dim"/>
      <text x="{x0 + device_w + 94}" y="{(y_box + y_sub)/2 + 4:.1f}" class="small">TBOX={tbox_nm:.1f} nm</text>

      <line x1="{x_gate}" y1="{y_sub + sub_h + 18}" x2="{x_gate + gate_w}" y2="{y_sub + sub_h + 18}" class="dim"/>
      <text x="{gate_center - 16}" y="{y_sub + sub_h + 36}" class="small">L_G={lg_um:.3f} μm</text>

      <text x="18" y="340" class="small">W={params.w * 1e6:.3f} μm, source/drain junction depth = TSi</text>
    </svg>
    """
    return svg


# ============================================================================
# UI helpers
# ============================================================================

PRESETS = {
    "Nominal": STLParams(),
    "Stronger BTBT": STLParams(btbt_scale=1.3, bb_a=5.0e14),
    "Weaker recombination": STLParams(recomb_scale=0.7),
    "Thicker oxide": STLParams(tox=18e-9),
    "Shorter gate": STLParams(lg=0.35e-6),
}


def apply_preset():
    if "ui_params" not in st.session_state:
        st.session_state["ui_params"] = asdict(STLParams())
    preset_name = st.session_state.get("preset_name", "Nominal")
    st.session_state["ui_params"] = asdict(PRESETS[preset_name])


def get_ui_value(key: str, default):
    return st.session_state.get("ui_params", {}).get(key, default)


def sidebar_params() -> Tuple[str, STLParams, Optional[Dict], Dict]:
    st.sidebar.header("Simulator controls")

    if "ui_params" not in st.session_state:
        st.session_state["ui_params"] = asdict(STLParams())

    st.sidebar.selectbox(
        "Preset",
        list(PRESETS.keys()),
        key="preset_name",
        on_change=apply_preset,
    )

    mode = st.sidebar.selectbox("Mode", ["Single run", "Overlay sweep"])

    with st.sidebar.expander("Bias / Sweep", expanded=True):
        vg = st.number_input("V_G [V]", value=float(get_ui_value("vg", -3.5)), step=0.1, format="%.3f")
        vbg = st.number_input("V_BG [V]", value=float(get_ui_value("vbg", 0.0)), step=0.1, format="%.3f")
        vd_max = st.number_input("V_D max [V]", value=float(get_ui_value("vd_max", 4.0)), step=0.1, format="%.3f")
        vstep = st.number_input("V_D step [V]", value=float(get_ui_value("vstep", 0.005)), step=0.001, format="%.4f")

    with st.sidebar.expander("Device", expanded=True):
        nb_cm = st.number_input("N_B [cm^-3]", value=float(get_ui_value("nb_cm", 3.0e17)), format="%.4e")
        lg_um = st.number_input("L_G [μm]", value=float(get_ui_value("lg", 0.5e-6) * 1e6), step=0.05, format="%.3f")
        w_um = st.number_input("W [μm]", value=float(get_ui_value("w", 0.65e-6) * 1e6), step=0.05, format="%.3f")
        tsi_nm = st.number_input("T_Si [nm]", value=float(get_ui_value("tsi", 50e-9) * 1e9), step=5.0, format="%.2f")
        tox_nm = st.number_input("T_ox [nm]", value=float(get_ui_value("tox", 13e-9) * 1e9), step=1.0, format="%.2f")
        tbox_nm = st.number_input("T_BOX [nm]", value=float(get_ui_value("tbox", 140e-9) * 1e9), step=10.0, format="%.2f")

    with st.sidebar.expander("BTBT", expanded=True):
        bb_a = st.number_input("A", value=float(get_ui_value("bb_a", 4.0e14)), format="%.4e")
        bb_b = st.number_input("B [V/cm]", value=float(get_ui_value("bb_b", 19.0e6)), format="%.4e")
        bb_gamma = st.number_input("γ_BTBT", value=float(get_ui_value("bb_gamma", 2.5)), step=0.1, format="%.2f")
        btbt_scale = st.number_input("BTBT strength scale", value=float(get_ui_value("btbt_scale", 1.0)), step=0.1, format="%.2f")

    with st.sidebar.expander("II", expanded=False):
        vbr = st.number_input("V_BR [V]", value=float(get_ui_value("vbr", 2.35)), step=0.05, format="%.3f")
        k_mult = st.number_input("k (multiplication exponent)", value=float(get_ui_value("k_mult", 4.0)), step=0.5, format="%.2f")
        ii_scale = st.number_input("II strength scale", value=float(get_ui_value("ii_scale", 1.0)), step=0.1, format="%.2f")

    with st.sidebar.expander("Out-diffusion", expanded=False):
        beta0 = st.number_input("β0", value=float(get_ui_value("beta0", 50.0)), step=1.0, format="%.2f")
        beta_scale = st.number_input("β scale", value=float(get_ui_value("beta_scale", 1.5)), step=0.1, format="%.2f")
        outdiff_scale = st.number_input("Out-diffusion strength scale", value=float(get_ui_value("outdiff_scale", 1.0)), step=0.1, format="%.2f")

    with st.sidebar.expander("Recombination", expanded=False):
        tau = st.number_input("τ_B [s]", value=float(get_ui_value("tau", 3.0e-7)), format="%.4e")
        recomb_scale = st.number_input("Recombination strength scale", value=float(get_ui_value("recomb_scale", 1.0)), step=0.1, format="%.2f")

    with st.sidebar.expander("Oscillation", expanded=False):
        iin = st.number_input("I_IN [A]", value=float(get_ui_value("iin", 10e-9)), format="%.4e")
        cd = st.number_input("C_D [F]", value=float(get_ui_value("cd", 170e-12)), format="%.4e")
        auto_time_window = st.checkbox("Auto time window", value=bool(get_ui_value("auto_time_window", True)))
        auto_cycles = st.number_input("Cycles to show (auto)", value=float(get_ui_value("auto_cycles", 5.0)), min_value=2.0, step=1.0, format="%.0f")
        t_window = st.number_input("Manual time window [s]", value=float(get_ui_value("t_window", 5e-3)), format="%.4e")

    with st.sidebar.expander("Numerics", expanded=False):
        root_grid_points = int(st.slider("Root grid points", min_value=48, max_value=200, value=int(get_ui_value("root_grid_points", 96)), step=8))
        solver_tol = float(st.select_slider("Root tolerance", options=[1e-10, 1e-12, 1e-14], value=float(get_ui_value("solver_tol", 1e-14))))
        current_ceiling_abs = st.number_input("Absolute current ceiling [A]", value=float(get_ui_value("current_ceiling_abs", 1e-2)), format="%.3e")

    base = STLParams(
        nb_cm=nb_cm,
        lg=lg_um * 1e-6,
        w=w_um * 1e-6,
        tsi=tsi_nm * 1e-9,
        tox=tox_nm * 1e-9,
        tbox=tbox_nm * 1e-9,
        vg=vg,
        vbg=vbg,
        vd_max=vd_max,
        vstep=vstep,
        bb_a=bb_a,
        bb_b=bb_b,
        bb_gamma=bb_gamma,
        btbt_scale=btbt_scale,
        vbr=vbr,
        k_mult=k_mult,
        ii_scale=ii_scale,
        beta0=beta0,
        beta_scale=beta_scale,
        outdiff_scale=outdiff_scale,
        tau=tau,
        recomb_scale=recomb_scale,
        iin=iin,
        cd=cd,
        auto_time_window=auto_time_window,
        auto_cycles=auto_cycles,
        t_window=t_window,
        root_grid_points=root_grid_points,
        snap_decades_min=float(get_ui_value("snap_decades_min", 1.0)),
        solver_tol=solver_tol,
        current_ceiling_abs=current_ceiling_abs,
    )

    # persist latest UI values
    st.session_state["ui_params"] = asdict(base)

    sweep_cfg = None
    if mode == "Overlay sweep":
        with st.sidebar.expander("Overlay sweep setup", expanded=True):
            sweep_param = st.selectbox(
                "Parameter to sweep",
                [
                    "bb_a",
                    "bb_b",
                    "tox_nm",
                    "tsi_nm",
                    "lg_um",
                    "vg",
                    "btbt_scale",
                    "ii_scale",
                    "outdiff_scale",
                    "recomb_scale",
                ],
            )
            defaults = {
                "bb_a": "2.8e14,4.0e14,5.2e14",
                "bb_b": "18.05e6,19.0e6,19.95e6",
                "tox_nm": "8,13,18",
                "tsi_nm": "30,50,70",
                "lg_um": "0.35,0.50,0.70",
                "vg": "-3.2,-3.5,-3.8",
                "btbt_scale": "0.5,1.0,1.5",
                "ii_scale": "0.5,1.0,1.5",
                "outdiff_scale": "0.5,1.0,1.5",
                "recomb_scale": "0.5,1.0,1.5",
            }
            values_text = st.text_input("Values", defaults[sweep_param])
            sweep_cfg = {"param": sweep_param, "values_text": values_text}

    meta = {
        "mode": mode,
        "device_note": "TSi is the single unified silicon thickness parameter shared by the DC sweep and the transient simulation.",
    }
    return mode, base, sweep_cfg, meta


def make_overlay_cases(base: STLParams, sweep_cfg: Dict) -> List[Tuple[str, STLParams]]:
    vals = [float(x.strip()) for x in sweep_cfg["values_text"].split(",") if x.strip()]
    cases = []
    for v in vals:
        d = asdict(base)
        if sweep_cfg["param"] == "bb_a":
            d["bb_a"] = v
            label = f"A={v:.3e}"
        elif sweep_cfg["param"] == "bb_b":
            d["bb_b"] = v
            label = f"B={v:.3e}"
        elif sweep_cfg["param"] == "tox_nm":
            d["tox"] = v * 1e-9
            label = f"Tox={v:.1f} nm"
        elif sweep_cfg["param"] == "tsi_nm":
            d["tsi"] = v * 1e-9
            label = f"TSi={v:.1f} nm"
        elif sweep_cfg["param"] == "lg_um":
            d["lg"] = v * 1e-6
            label = f"LG={v:.2f} μm"
        elif sweep_cfg["param"] == "vg":
            d["vg"] = v
            label = f"VG={v:.2f} V"
        elif sweep_cfg["param"] == "btbt_scale":
            d["btbt_scale"] = v
            label = f"BTBT={v:.2f}"
        elif sweep_cfg["param"] == "ii_scale":
            d["ii_scale"] = v
            label = f"II={v:.2f}"
        elif sweep_cfg["param"] == "outdiff_scale":
            d["outdiff_scale"] = v
            label = f"OutDiff={v:.2f}"
        elif sweep_cfg["param"] == "recomb_scale":
            d["recomb_scale"] = v
            label = f"Recomb={v:.2f}"
        else:
            label = f"{sweep_cfg['param']}={v}"
        cases.append((label, STLParams(**d)))
    return cases


# ============================================================================
# Main app
# ============================================================================

def main():
    st.set_page_config(page_title="SOI STL Simulator", layout="wide")
    st.title("SOI MOSFET STL simulator")
    st.caption(
        "Branch-preserving ID–VD sweep, direct transient oscillation with the same device model, and live SOI cross-section view."
    )

    mode, base_params, sweep_cfg, meta = sidebar_params()

    top_left, top_right = st.columns([1.05, 1.35])
    with top_left:
        st.subheader("Live device view")
        st.markdown(soi_svg(base_params), unsafe_allow_html=True)
    with top_right:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("L_G", f"{base_params.lg * 1e6:.3f} μm")
        c2.metric("TSi", f"{base_params.tsi * 1e9:.1f} nm")
        c3.metric("Tox", f"{base_params.tox * 1e9:.1f} nm")
        c4.metric("TBOX", f"{base_params.tbox * 1e9:.1f} nm")
        st.info(
            "편의 기능: 프리셋, 오버레이 sweep, JSON 파라미터 다운로드, Excel export가 포함돼 있다."
        )
        st.caption(meta["device_note"])

    tab_dc, tab_osc, tab_export = st.tabs(["DC / Hysteresis", "Oscillation", "Export / Share"])

    with tab_dc:
        left, right = st.columns([1.0, 2.0])
        with left:
            run_dc = st.button("Run DC sweep", type="primary")
            st.write("Current base parameters")
            st.json(
                {
                    "VG [V]": base_params.vg,
                    "VBG [V]": base_params.vbg,
                    "LG [μm]": base_params.lg * 1e6,
                    "TSi [nm]": base_params.tsi * 1e9,
                    "Tox [nm]": base_params.tox * 1e9,
                    "BTBT scale": base_params.btbt_scale,
                    "II scale": base_params.ii_scale,
                    "OutDiff scale": base_params.outdiff_scale,
                    "Recomb scale": base_params.recomb_scale,
                    "Vstep [V]": base_params.vstep,
                }
            )
            st.download_button(
                "Download current params (JSON)",
                data=json.dumps(to_hashable_dict(base_params), indent=2).encode("utf-8"),
                file_name="stl_params.json",
                mime="application/json",
            )
        with right:
            if run_dc:
                with st.spinner("Running branch-preserving DC sweep..."):
                    try:
                        if mode == "Single run":
                            results = [("Nominal", simulate_double_sweep_cached(to_hashable_dict(base_params)))]
                        else:
                            cases = make_overlay_cases(base_params, sweep_cfg)
                            results = [(label, simulate_double_sweep_cached(to_hashable_dict(p))) for label, p in cases]
                        st.session_state["dc_results"] = results
                    except Exception as exc:
                        st.error(f"DC simulation failed: {exc}")

            results = st.session_state.get("dc_results")
            if results:
                st.plotly_chart(fig_dc(results), use_container_width=True)

                summary_rows = []
                for label, res in results:
                    summary_rows.append({
                        "Case": label,
                        "VLU [V]": res["vlu"],
                        "VLD [V]": res["vld"],
                        "Snap-up pre [V]": res["snap_up_v"],
                        "Snap-up post [V]": res["snap_up_next_v"],
                        "Snap-down pre [V]": res["snap_down_v"],
                        "Snap-down post [V]": res["snap_down_next_v"],
                        "Jump-up [dec]": res["snap_up_decades"],
                        "Jump-down [dec]": res["snap_down_decades"],
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                selected_case = st.selectbox("Internal quantity view", [label for label, _ in results], key="internal_case")
                selected_res = dict(results)[selected_case]
                st.plotly_chart(fig_internal_quantities(selected_res), use_container_width=True)
                st.caption("Plotly legend를 클릭하면 body potential, BTBT current, BJT current, II current를 각각 숨기거나 다시 표시할 수 있다.")

                with st.expander("Numerical diagnostics"):
                    fwd_resid = float(np.nanmax(np.abs(selected_res["residuals"][: len(selected_res["vfwd"])])))
                    rev_resid = float(np.nanmax(np.abs(selected_res["residuals"][len(selected_res["vfwd"]):])))
                    st.write({
                        "Max |residual| forward": fwd_resid,
                        "Max |residual| reverse": rev_resid,
                        "RLRS [Ω]": selected_res["model_rlrs"],
                        "IS [A]": selected_res["model_is"],
                    })

                st.download_button(
                    "Download Excel",
                    data=build_excel_bytes(results),
                    file_name="stl_dc_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    with tab_osc:
        left, right = st.columns([1.0, 2.0])
        with left:
            run_osc = st.button("Run oscillation", type="primary")
            st.write({
                "IIN [A]": base_params.iin,
                "CD [F]": base_params.cd,
                "Initial VD [V]": base_params.vd0,
                "Auto time window": base_params.auto_time_window,
                "Cycles": base_params.auto_cycles,
                "Manual window [s]": base_params.t_window,
            })
            st.info(
                "오실레이션은 DC sweep과 동일한 소자 모델 fID(ID,VD)=0을 사용하고, t=0에서 V_D=0 V로 시작하는 직접 시간영역 적분으로 계산한다."
            )
        with right:
            if run_osc:
                with st.spinner("Running DC baseline + direct transient... "):
                    try:
                        dc = simulate_double_sweep_cached(to_hashable_dict(base_params))
                        osc = simulate_oscillation_cached(to_hashable_dict(base_params), dc)
                        st.session_state["osc_result"] = osc
                        st.session_state["osc_dc"] = dc
                    except Exception as exc:
                        st.error(f"Oscillation simulation failed: {exc}")

            osc = st.session_state.get("osc_result")
            dc = st.session_state.get("osc_dc")
            if osc is not None and dc is not None:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Simulated window", f"{osc['diagnostics']['time_window_s']:.3e} s")
                m2.metric("Period", "N/A" if not np.isfinite(osc["estimated_period_s"]) else f"{osc['estimated_period_s']:.3e} s")
                m3.metric("Measured frequency", "N/A" if not np.isfinite(osc["freq"]) else f"{osc['freq']:.3e} Hz")
                m4.metric("Peak count", str(int(osc["diagnostics"].get("peak_count", 0))))

                st.plotly_chart(fig_oscillation(osc), use_container_width=True)
                st.plotly_chart(fig_oscillation_idvd(dc, osc), use_container_width=True)

                if not np.isfinite(osc["freq"]):
                    st.warning("No sustained oscillation was confirmed in the current time window. The startup transient from t=0 is still shown above.")

                with st.expander("Oscillation diagnostics", expanded=True):
                    st.json(osc["diagnostics"])

                st.download_button(
                    "Download Excel (DC + oscillation)",
                    data=build_excel_bytes([("Nominal", dc)], osc),
                    file_name="stl_dc_osc_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    with tab_export:
        st.subheader("How to run and share this app")
        st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
        st.markdown(
            """
            1. Put `app.py` and `requirements.txt` in a GitHub repository.
            2. Deploy it on Streamlit Community Cloud.
            3. Choose a public `*.streamlit.app` URL so outside users can open it directly.
            """
        )
        st.caption("The package below is ready to upload into a GitHub repo and then deploy from Streamlit Community Cloud.")


if __name__ == "__main__":
    main()
