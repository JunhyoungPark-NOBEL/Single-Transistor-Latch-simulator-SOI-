"""STL circuit element: evaluation of the engine's ``components`` with the circuit-level
extensions, the body-charge coordinate and the stochastic event rates (numba).

Everything here is ``@njit(cache=True)`` and uses the geometry-aware component dispatcher.

Element outputs (``out`` array of length ``N_EV``)::

    out[0] V_D   internal drain-source voltage  V_D(u, r)          (V)    = z[0]
    out[1] I_D   drain current                                       (A)    = z[1]
    out[2] F     net hole current into the body  G - L               (A)    = z[2]
    out[3] Q     body-charge coordinate                              (C)
                 Reference: Q = C_ox (psi - V_GS) + Q_exc + q N_A A L_n
                 Geometry: Q = C_body psi - C_front V_GS - C_back V_BG + Q_exc + q N_A A L_n
                 psi = u - V_T ln(1 + delta/N_A),  Q_exc = z[13] - C_ox u,  L_n = z[11]
    out[4] unit  unit-event current  z[8] + z[9] + z[18]            (A)   (junction BTBT + GIDL + photo)
    out[5] G     total hole generation current  z[1] - z[3] - z[16] (A)
    out[6] L     hole loss current  z[5] + z[6] + z[7]              (A)
    out[7] flag  1 = u < 0 extension used, 2 = r < 0 extension used, 3 = both

Extensions outside the engine's steady-state domain (the engine never evaluates them):

* ``u < 0`` (source junction reverse biased, e.g. after a fast drain down-ramp in the dark):
  ``components`` returns NaN for any u < 0.  The core is evaluated at u = 0 and every current
  X in {I_D - I_ch, G, L, unit} is continued as X(0) + X'(0+) V_T expm1(u/V_T) (the low-injection
  diode law, C1 at u = 0, saturating at the reverse generation current); the channel current is
  re-evaluated with the true u (same closed form as ``photo_mean``); V_D = V_D(0,r) + u +
  (R_c + R_acc) dI_D; Q uses the exact psi(u) and source depletion width w_s(u).
* ``r < 0`` (drain junction forward biased, e.g. fast down-ramps or photovoltaic charging at
  V_D ~ 0): the core is evaluated at r = 0 and a *symmetric forward drain diode* is added — the
  same n+ emitter diffusion saturation current as the source (q A D_n n_i^2 / (N_A L_ref beta),
  without the source-edge state phi_E) plus depletion SRH q A w_d n_i / (2 tau_j) expm1(-r/2V_T).
  Its current is a hole loss (L += I_fwd, F -= I_fwd) and flows out of the drain terminal
  (I_D -= I_fwd); the channel current is re-evaluated with the true u + r; V_D = V_D(u,0) + r +
  (R_c + R_acc) dI_D; the neutral length grows by w_d(0) - w_d(r).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from server.engine_bridge import m
from server.geometry_model import channel_current, constants_from_p, gate_charge_offset

components = m.components

QE = float(m.Q)
VT = float(m.VT)
AREA = float(m.AREA_CM2)
COX = float(m.COX_F)
NI = float(m.NI_CM3)
DN = float(m.DN)
LCH = float(m.LENGTH_M) * 100.0          # channel length (cm)
WREF = float(m.WIDTH_M) * 100.0          # calibrated width (cm)
TSIREF = float(m.TSI_M) * 100.0          # calibrated silicon thickness (cm)
EPS_SI = 11.7 * float(m.EPS0) / 100.0    # F/cm (as in photo_mean)

N_EV = 8


@njit(cache=True)
def wdep(v, na):
    """Abrupt-junction depletion width (cm) for a total junction voltage v (V)."""
    return np.sqrt(2.0 * EPS_SI * v / (QE * na))


@njit(cache=True)
def ch_formula(u, r, p):
    """Per-device channel current, without the optional high-V_D seed p[17]."""
    return channel_current(u, r, p)


@njit(cache=True)
def stl_eval(u, r, p, na, vbi, rg, fg, table, out):
    """Evaluate the element at internal state (u, r); p[11] must hold V_GS.  Returns False
    (and NaNs in ``out``) outside the valid domain."""
    # Geometry and fixed backgate bias travel with each cell's parameter vector.
    # The gate offset keeps front- and back-gate capacitance coefficients separate.
    lch, width, tsi, area, cox, na, vbi = constants_from_p(p)
    uc = u if u > 0.0 else 0.0
    rc = r if r > 0.0 else 0.0
    z = components(uc, rc, p, na, vbi, rg, fg, table)
    if not (np.isfinite(z[0]) and np.isfinite(z[1]) and np.isfinite(z[2]) and np.isfinite(z[13])):
        for i in range(N_EV):
            out[i] = np.nan
        return False
    vd = z[0]
    idr = z[1]
    unit = z[8] + z[9] + z[18]
    gq = z[1] - z[3] - z[16]
    lq = z[5] + z[6] + z[7]
    width_scale = width / WREF
    rser = p[3] / width_scale + z[12]
    flag = 0.0
    if u >= 0.0:
        psi = u - VT * np.log1p(z[10])
        q = cox * psi + gate_charge_offset(p) + (z[13] - cox * u) + QE * na * area * z[11]
    else:
        flag += 1.0
        du = 1e-4
        zd = components(du, rc, p, na, vbi, rg, fg, table)
        if not (np.isfinite(zd[0]) and np.isfinite(zd[2])):
            for i in range(N_EV):
                out[i] = np.nan
            return False
        e = VT * np.expm1(u / VT)
        d_id = ((zd[1] - zd[16]) - (z[1] - z[16])) / du
        d_g = ((zd[1] - zd[3] - zd[16]) - gq) / du
        d_l = ((zd[5] + zd[6] + zd[7]) - lq) / du
        d_u = ((zd[8] + zd[9] + zd[18]) - unit) / du
        id_new = idr + d_id * e + (ch_formula(u, rc, p) - ch_formula(0.0, rc, p))
        gq += d_g * e
        lq += d_l * e
        unit += d_u * e
        vd = vd + u + rser * (id_new - idr)
        idr = id_new
        prod = NI * NI * np.expm1(u / VT)
        delta = 2 * prod / (na + np.sqrt(na * na + 4 * prod))
        sb = vbi - u + VT * np.log1p(delta / na)
        ws = wdep(sb, na)
        wd = wdep(vbi + rc, na)
        psi = u - VT * np.log1p(delta / na)
        q = cox * psi + gate_charge_offset(p) + z[13] + QE * na * area * (lch - wd - ws)
    if r < 0.0:
        flag += 2.0
        if vbi + r < 0.02:
            for i in range(N_EV):
                out[i] = np.nan
            return False
        wd0 = wdep(vbi, na)
        wdr = wdep(vbi + r, na)
        lref = lch - 2.0 * wd0
        isd = QE * area * DN * NI * NI / (na * lref * p[0])
        tj = p[2] * (tsi / TSIREF)
        ifwd = isd * np.expm1(-r / VT) + QE * area * wdr * NI / (2.0 * tj) * np.expm1(-r / (2.0 * VT))
        dch = ch_formula(u, r, p) - ch_formula(u, 0.0, p)
        id_new = idr - ifwd + dch
        vd = vd + r + rser * (id_new - idr)
        idr = id_new
        lq += ifwd
        q += QE * na * area * (wd0 - wdr)
    out[0] = vd
    out[1] = idr
    out[2] = gq - lq
    out[3] = q
    out[4] = unit
    out[5] = gq
    out[6] = lq
    out[7] = flag
    return True


@njit(cache=True)
def pmf_at(r, rv, pmf, pk):
    """Linear interpolation of the avalanche cluster pmf (sizes k >= 1) at reverse bias r
    (clamped to the tabulated range, as np.interp in compound_fpt.backward).  Fills pk[k]
    (k = 0..K) and returns (P1, M1, M2) = (sum_k>=1 p_k, sum k p_k, sum k^2 p_k)."""
    nrv = rv.shape[0]
    K = pmf.shape[1] - 1
    if r <= rv[0]:
        i = 0
        a = 0.0
    elif r >= rv[nrv - 1]:
        i = nrv - 2
        a = 1.0
    else:
        i = int((r - rv[0]) / (rv[1] - rv[0]))
        if i > nrv - 2:
            i = nrv - 2
        while i > 0 and rv[i] > r:
            i -= 1
        while i < nrv - 2 and rv[i + 1] < r:
            i += 1
        a = (r - rv[i]) / (rv[i + 1] - rv[i])
    p1 = 0.0
    m1 = 0.0
    m2 = 0.0
    pk[0] = 0.0
    for k in range(1, K + 1):
        v = (1.0 - a) * pmf[i, k] + a * pmf[i + 1, k]
        pk[k] = v
        p1 += v
        m1 += k * v
        m2 += k * k * v
    return p1, m1, m2
