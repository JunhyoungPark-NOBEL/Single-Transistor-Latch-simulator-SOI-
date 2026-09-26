"""Quasi-static drive of a high-impedance (current-biased) STL cell: the relaxation-oscillator estimate.

A cell whose drain is fed through a large source impedance (a current source, or a resistor of
>= 100 MOhm) does not follow the open-circuit drive voltage: its own current moves V_DS.  With a
capacitance C across the cell, V_DS obeys (quasi-static body, hysteretic HRS/LRS state)

    C dV/dt = I_N(t) - G V - I_b(V),   b = HRS until V reaches V_LU, then LRS until V falls to V_LD,

(I_N, G: Norton equivalent of the linear network seen by the cell).  When the load line
I_N - G V crosses only the unstable (negative-resistance) branch — for a current source
I_LU < I_N < I_LD — the walk is a relaxation oscillation (integrate-and-fire sawtooth): V ramps up on
the HRS at (I_N - I_HRS(V))/C to V_LU, latches, the LRS current discharges C to V_LD, unlatches.
Quasi-static period (no fold lags):

    T = integral_{V_LD}^{V_LU} C dV / (I_N - G V - I_HRS(V)) + integral_{V_LD}^{V_LU} C dV / (I_LRS(V) - I_N + G V)
      ~ C (V_LU - V_LD) / I_N   when I_LU << I_N << I_LD.

``qs_drive`` integrates this walk (semi-implicit Euler, <= dv_step per step) and returns the V_DS(t)
samples, which replace the open-circuit drive in the feasibility estimate (``estimate_steps``), plus the
latch-up times (predicted spikes).  The kernel resolves what this walk leaves out (the slow passage
through the folds, the body relaxation); the measured periods exceed the quasi-static value by the fold
lags (docs/CIRCUIT_SIMULATOR.md §13).
"""
from __future__ import annotations

import numpy as np
from numba import njit

R_HIGH = 1e8           # Ohm: cells whose d-s port resistance exceeds this use the quasi-static walk
QS_DV = 5e-3           # V per walk step
QS_MAX_POINTS = 300_000
FOLD_MARK = 2e-3       # V: the walk marks each switch 2 mV beyond the fold (the step estimator's walk sees it)


@njit(cache=True)
def _interp_lnI(v, V, lnI):
    """ln I and d ln I / dV of a branch table at v (linear in ln I; constant slope extrapolation)."""
    n = V.shape[0]
    if n == 1:
        return lnI[0], 0.0
    if v <= V[0]:
        s = (lnI[1] - lnI[0]) / (V[1] - V[0])
        return lnI[0] + s * (v - V[0]), s
    if v >= V[n - 1]:
        s = (lnI[n - 1] - lnI[n - 2]) / (V[n - 1] - V[n - 2])
        if s < 0.0:
            s = 0.0
        return lnI[n - 1] + s * (v - V[n - 1]), s
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if V[mid] <= v:
            lo = mid
        else:
            hi = mid
    s = (lnI[hi] - lnI[lo]) / (V[hi] - V[lo])
    return lnI[lo] + s * (v - V[lo]), s


@njit(cache=True)
def _walk(tn, In, G, C, VH, lnH, VL, lnL, V_LU, V_LD, latch, v0, t_end, dt_cap, dv, mark, max_pts,
          out_t, out_v, lu_t):
    n = 0
    t = 0.0
    v = v0
    lrs = False
    nlu = 0
    nld = 0
    j = 0
    out_t[0] = 0.0
    out_v[0] = v
    n = 1
    nb = tn.shape[0]
    while t < t_end and n < max_pts - 2:
        while j < nb - 1 and tn[j + 1] <= t:
            j += 1
        # Norton current at t (PWL on tn)
        if j >= nb - 1:
            i_n = In[nb - 1]
            t_next = t_end
        else:
            a = (t - tn[j]) / (tn[j + 1] - tn[j]) if tn[j + 1] > tn[j] else 1.0
            i_n = In[j] + a * (In[j + 1] - In[j])
            t_next = tn[j + 1]
        if lrs:
            li, s = _interp_lnI(v, VL, lnL)
        else:
            li, s = _interp_lnI(v, VH, lnH)
        ib = np.exp(min(li, 50.0))
        f = (i_n - G * v - ib) / C
        fp = -(G + ib * s) / C
        h = dt_cap
        if f != 0.0:
            h = min(h, dv / abs(f))
        if t_next > t:
            h = min(h, t_next - t)
        h = min(h, t_end - t)
        if h <= 0.0:
            h = 1e-3 * dt_cap
        den = 1.0 - h * fp
        if den < 1.0:
            den = 1.0
        vn = v + h * f / den
        t += h
        if latch and (not lrs) and vn >= V_LU:
            vn = V_LU + mark
            lrs = True
            if nlu < lu_t.shape[0]:
                lu_t[nlu] = t
            nlu += 1
        elif latch and lrs and vn <= V_LD:
            vn = V_LD - mark
            lrs = False
            nld += 1
        v = vn
        out_t[n] = t
        out_v[n] = v
        n += 1
    return n, t, nlu, nld


def qs_drive(t_grid: np.ndarray, i_norton: np.ndarray, g_ext: float, c_eff: float, profile: dict, t_end: float,
             dt_max: float, v0: float = 0.0, dv: float = QS_DV, max_points: int = QS_MAX_POINTS) -> dict:
    """Quasi-static V_DS walk of a high-impedance cell (module docstring).  Returns dict(t, v (samples),
    t_reached, n_lu, n_ld, lu_t (latch-up times), period (mean interval of the last latch-ups or None),
    oscillating (>= 2 latch-ups), scale = t_end / t_reached (>= 1 when the point budget ran out))."""
    iv = profile.get("iv") or {}
    H = iv.get("HRS")
    L = iv.get("LRS", H)
    if H is None or len(H[0]) < 2:
        return dict(t=np.array([0.0, t_end]), v=np.array([v0, v0]), t_reached=t_end, n_lu=0, n_ld=0,
                    lu_t=np.zeros(0), period=None, oscillating=False, scale=1.0)
    if L is None or len(L[0]) < 2:
        L = H
    V_LU, V_LD = profile["folds"]
    latch = bool(profile["latch"]) and np.isfinite(V_LU) and np.isfinite(V_LD)
    tn = np.ascontiguousarray(t_grid, float)
    In = np.ascontiguousarray(i_norton, float)
    out_t = np.zeros(max_points)
    out_v = np.zeros(max_points)
    lu_t = np.zeros(4096)
    n, t_r, nlu, nld = _walk(tn, In, float(max(g_ext, 0.0)), float(c_eff),
                             np.ascontiguousarray(H[0], float), np.log(np.ascontiguousarray(H[1], float)),
                             np.ascontiguousarray(L[0], float), np.log(np.ascontiguousarray(L[1], float)),
                             float(V_LU) if latch else np.inf, float(V_LD) if latch else -np.inf, latch,
                             float(v0), float(t_end), float(dt_max), float(dv), FOLD_MARK, int(max_points),
                             out_t, out_v, lu_t)
    lt = lu_t[:min(nlu, len(lu_t))]
    period = float(np.mean(np.diff(lt[-min(len(lt), 11):]))) if len(lt) >= 2 else None
    return dict(t=out_t[:n].copy(), v=out_v[:n].copy(), t_reached=float(t_r), n_lu=int(nlu), n_ld=int(nld),
                lu_t=lt.copy(), period=period, oscillating=nlu >= 2, scale=float(t_end / t_r) if t_r > 0 else 1.0)


def quasi_static_period(i_n: float, g_ext: float, c_eff: float, profile: dict, n: int = 4001) -> tuple[float, float]:
    """(charge time on the HRS, discharge time on the LRS) of one quasi-static cycle for a constant Norton
    current (inf when the load line crosses that branch: no oscillation)."""
    iv = profile.get("iv") or {}
    V_LU, V_LD = profile["folds"]
    if not profile.get("latch") or "LRS" not in iv or not np.isfinite(V_LU):
        return np.inf, np.inf
    V = np.linspace(V_LD, V_LU, n)
    IH = np.exp(np.interp(V, iv["HRS"][0], np.log(iv["HRS"][1])))
    IL = np.exp(np.interp(V, iv["LRS"][0], np.log(iv["LRS"][1])))
    load = i_n - g_ext * V
    up, dn = load - IH, IL - load
    tc = float(np.trapezoid(c_eff / up, V)) if np.all(up > 0) else np.inf
    td = float(np.trapezoid(c_eff / dn, V)) if np.all(dn > 0) else np.inf
    return tc, td
