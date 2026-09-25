"""Generic small-netlist MNA transient kernel with STL elements (numba).

Unknown vector x = [node voltages v_1..v_{N-1} (node 0 = ground), V-source branch currents,
(u_k, r_k) for every STL element k].

Residuals:
  KCL at every non-ground node (A):  sum of currents leaving the node = 0
      R: (v_a - v_b)/R; C: companion (BE: C/h dv, TRAP: 2C/h dv - i_n); V: branch current;
      I: source value; STL: I_D(u,r) leaves the drain node and enters the source node.
  V source:  v_a - v_b - V(t) = 0
  comparator output (a V source whose wave index is -1 - j, j = comparator row of ``cmp``):
             v_out - y_j(v_in - v_inm) = 0,  y = v_low + (v_high - v_low) (1 + tanh((d - thr)/w))/2,
             thr = v_ref + hyst/2 (output low) or v_ref - hyst/2 (output high), w = smoothing width
  STL, E1:   V_D(u,r) - (v_d - v_s) = 0                               (V)
  STL, E2:   [Q(u,r;V_GS) - Qc - th*h*F(u,r)] / C_ox = 0               (V)
      deterministic BE: Qc = Q_n, th = 1;  TRAP: Qc = Q_n + h/2 F_n, th = 1/2
      stochastic, per cell and step one of the tiers (docs/CIRCUIT_SIMULATOR.md §4):
        1 event-level explicit tau-leap (Eq. 2): Qc = Q_n + q (N_unit + sum k_i - N_loss), th = 0
        2 Gaussian, drift-implicit, variance-corrected: Qc = Q_n + eta, th = 1
        3 drift only, relaxation too fast to resolve (tau_rel < gauss_tau_min): Qc = Q_n, th = 1
        4 drift only, latched cell with ld_carrier_noise off
        5 drift only, outside the cell's noise band (no escape possible there)
  (DC initialisation: E2 replaced by u - u_fix = 0, then pseudo-transient BE.)

Geometry model v1 with split front-/back-gate charge: each row P[k] retains its own geometry and electrostatic field
table through every element evaluation and finite difference.  Q and F therefore
use that cell's dimensions and doping.  The C_ox denominator above remains the
reference capacitance solely as a common numerical scale; residual, Jacobian and
charge sensitivities all use the same scale.  Updating this kernel's source also
invalidates Numba's pre-geometry parent-function caches on existing installs.

Latch state (independent of the current thresholds).  Along the quasi-static branch (parameterised
by u) the HRS is u < u_i (u at the latch-up fold), the unstable branch u_i < u < u_j and the LRS
u > u_j (u at the latch-down fold).  A cell latches when u reaches u_j and unlatches when u falls to
u_i (hysteresis = the unstable-branch range; no latch window -> u_i = u_j = +inf, never latched).
This state (SS_LAT = SS_PHYS) selects the noise tier / band and is what the events and samples
report.  An event is timed at the I_D threshold crossing (latch-up: I_D >= i_threshold with u >= u_i;
latch-down: I_D < i_threshold/hysteresis with u <= u_j) when that crossing lies inside the switching
transient, otherwise at the moment the body reaches the new branch.  I_D crossings of i_threshold
with the body still on the HRS (channel / HRS conduction) are counted (SI_HRSX), never reported.

Newton: the STL block of the Jacobian is a forward finite difference (d/du, d/dr) of
``stl_eval``; |du| <= 50 mV and |dr| <= 1 V per iteration; backtracking (halving) when the trial
point leaves the domain where ``components`` is finite.

The integration loop (``run_chunk``) advances until t_stop, a step budget or a full record buffer
and keeps its whole state in the arrays passed in, so the Python driver can call it repeatedly
(progress reporting / cancellation between chunks).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .element import COX, N_EV, QE, stl_eval, pmf_at
from .basic import stamp as stamp_basic, step_limit as basic_step_limit

GMIN = 1e-18          # S, node-to-ground conductance (keeps floating nodes regular)
DU_LIM = 0.05         # V, max |du| per Newton iteration
DR_LIM = 1.0          # V, max |dr| per Newton iteration
FD_U = 1e-6           # V, finite-difference step in u
FD_R = 1e-6           # V, finite-difference step in r

# ---- integer config (ci) -----------------------------------------------------------------
CI_NN, CI_NV, CI_NR, CI_NC, CI_NI, CI_NS, CI_METHOD, CI_STOCH, CI_CARRIER, CI_MAXSTEPS, \
    CI_MAINW, CI_LSMODE, CI_LSIDX, CI_CHUNK, CI_MAIN_STL, CI_LDNOISE = range(16)
N_CI = 16
# ---- float config (cf) -------------------------------------------------------------------
CF_TEND, CF_DTMIN, CF_DTMAX, CF_DUMAX, CF_DLNIMAX, CF_DVMAX, CF_LTEU, CF_TAUFRAC, CF_NEVMAX, \
    CF_HNOISEMIN, CF_GAUSS, CF_ITH, CF_IFLOOR, CF_DTREC, CF_DVREC, CF_DLNIREC, CF_LSSIG, CF_LSTAU, \
    CF_LSESIG, CF_LSETAU, CF_HINIT, CF_NEWTOL, CF_ITHDN, CF_GTAUMIN, CF_GTAUFRAC, CF_NLOOK, \
    CF_LTEV, CF_LTEVABS = range(28)
N_CF = 28
# CF_LTEV / CF_LTEVABS: optional node-voltage LTE control (custom circuits; 0 = off, the benches):
#   LTE_i = |v_i - v_i,pred| h / (h + h_prev) (linear-extrapolation predictor, BE-order estimate)
#   err  >= LTE_i / (CF_LTEV max(|v_i|, |v_i,prev|) + CF_LTEVABS), deterministic integration only
# ---- float state (sf) --------------------------------------------------------------------
SF_T, SF_HNEXT, SF_HPREV, SF_TREC, SF_TUNRES, SF_MINU, SF_MINR, SF_TNEGU, SF_TNEGR, SF_TSTOP, SF_TGAUSS, \
    SF_TLRS, SF_TBAND = range(13)
N_SF = 13
# ---- int state (si) ----------------------------------------------------------------------
SI_STEPS, SI_REJ, SI_NEWT, SI_BP, SI_NEV, SI_NREC, SI_NSAMP, SI_STATUS, SI_UNRES, SI_TRAPBE, \
    SI_REFRESH, SI_HAVEPREV, SI_FAILNEWTON, SI_CHUNKSTEPS, SI_GAUSS = range(15)
SI_DIAG = 15          # 15 + 3*regime + (0 steps, 1 newton iterations, 2 rejections), regime 0..5
SI_HRSX = 33          # I_D up-crossings of i_threshold with the body still on the HRS (not a latch-up)
N_SI = 34
# per-step regime of the charge update (stochastic mode): 1 event-level explicit tau-leap,
# 2 Gaussian drift-implicit (variance-corrected), 3 drift only (noise-active but relaxation too fast),
# 4 drift only (latched cell, ld_carrier_noise off), 5 drift only (outside the noise band: barrier to
# the saddle > noise_z_max stationary SDs, no escape possible); 0 = deterministic
# status codes
ST_DONE, ST_CHUNK, ST_FAIL, ST_MAXSTEPS, ST_BUFFER = 0, 1, 2, 3, 4
# per-STL state columns (ss): SS_LAT = SS_PHYS latch state (body branch, see the module docstring),
# SS_PEND timing candidate of the running transition with its time / v_DS / v_src / I_D in SS_PT..SS_PI
SS_QN, SS_FN, SS_IN, SS_TAU, SS_UNIT, SS_G, SS_L, SS_R, SS_LAT, SS_DQ, SS_LNI_REC, SS_VDS_REC, \
    SS_PHYS, SS_PEND, SS_PT, SS_PV, SS_PS, SS_PI = range(18)
N_SS = 18
# per-STL configuration columns (win): noise bands (unlatched lo/hi, latched lo/hi, V_DS), the u values of
# the latch-up fold (u_i) and latch-down fold (u_j) of the quasi-static branch, the noise look-ahead drive
# (waveform index or -1, gain dV_DS/dw, mode 0: V_DS ~ w (benches) | 1: V_DS(t') ~ V_DS(t) + g (w(t') - w(t)))
# and the cell's local-state configuration (mode 0/1/2, p index of the action point, sigma, tau, sigma_E, tau_E)
W_LULO, W_LUHI, W_LDLO, W_LDHI, W_UI, W_UJ, W_LAW, W_LAG, W_LAMODE, \
    W_LSMODE, W_LSIDX, W_LSSIG, W_LSTAU, W_LSESIG, W_LSETAU = range(15)
N_WIN = 15
# sample buffer: t, then per STL k: I_D (1 + 3k), v_DS (2 + 3k), reported latch state (3 + 3k)
N_SAMPC = 3
# partial derivative columns (part): d/du, d/dr and, for cells whose V_GS can move (P_VGSJ = 1: source not grounded
# or gate not held by a constant source), d/dV_GS; the V_GS columns enter the Jacobian at the gate and source nodes
P_VU, P_VR, P_IU, P_IR, P_FU, P_FR, P_QU, P_QR, P_VG, P_IG, P_FG, P_QG, P_VGSJ = range(13)
N_PART = 13
FD_VGS = 1e-6         # V, finite-difference step in V_GS
# event columns
EV_KIND, EV_STL, EV_T, EV_VDS, EV_VSRC, EV_I = range(6)
N_EVC = 6
# comparator rows (cmp): input node, inverting input node (0 = ground), v_ref, v_high, v_low, hysteresis,
# smoothing width, output state (0 low / 1 high; selects the hysteresis threshold)
K_IN, K_INM, K_REF, K_HI, K_LO, K_HYST, K_W, K_STATE = range(8)
N_CMPC = 8


@njit(cache=True)
def cmp_value(x, cmp, j):
    """Comparator j output y and dy/d(v_in - v_inm) at the node voltages x."""
    ni = int(cmp[j, K_IN])
    nm = int(cmp[j, K_INM])
    d = (0.0 if ni == 0 else x[ni - 1]) - (0.0 if nm == 0 else x[nm - 1])
    thr = cmp[j, K_REF] + (-0.5 if cmp[j, K_STATE] > 0.5 else 0.5) * cmp[j, K_HYST]
    w = cmp[j, K_W]
    z = (d - thr) / w
    if z > 40.0:
        z = 40.0
    elif z < -40.0:
        z = -40.0
    th = np.tanh(z)
    span = cmp[j, K_HI] - cmp[j, K_LO]
    return cmp[j, K_LO] + span * 0.5 * (1.0 + th), span * 0.5 * (1.0 - th * th) / w


@njit(cache=True)
def cmp_update_state(x, cmp):
    """After an accepted time point: output state = output above the mid level (hysteresis memory)."""
    for j in range(cmp.shape[0]):
        y, _ = cmp_value(x, cmp, j)
        cmp[j, K_STATE] = 1.0 if y > 0.5 * (cmp[j, K_HI] + cmp[j, K_LO]) else 0.0


@njit(cache=True)
def wave_value(w, t, wt, wv, woff):
    """Piecewise-linear waveform w at time t (constant outside its breakpoints)."""
    a = woff[w]
    b = woff[w + 1]
    if t <= wt[a]:
        return wv[a]
    if t >= wt[b - 1]:
        return wv[b - 1]
    lo = a
    hi = b - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if wt[mid] <= t:
            lo = mid
        else:
            hi = mid
    dt = wt[hi] - wt[lo]
    if dt <= 0.0:
        return wv[hi]
    return wv[lo] + (wv[hi] - wv[lo]) * (t - wt[lo]) / dt


@njit(cache=True)
def wave_reaches(w, t0, t1, level, above, wt, wv, woff):
    """True when the PWL waveform w reaches ``level`` (from below if ``above``, else from above)
    anywhere in [t0, t1] (checked at the endpoints and the corners; early exit)."""
    v0 = wave_value(w, t0, wt, wv, woff)
    v1 = wave_value(w, t1, wt, wv, woff)
    if above:
        if v0 >= level or v1 >= level:
            return True
    elif v0 <= level or v1 <= level:
        return True
    a = woff[w]
    b = woff[w + 1]
    lo = a
    hi = b
    while hi - lo > 1:                      # last corner <= t0
        mid = (lo + hi) // 2
        if wt[mid] <= t0:
            lo = mid
        else:
            hi = mid
    i = lo if wt[lo] > t0 else lo + 1
    while i < b and wt[i] < t1:
        if (above and wv[i] >= level) or ((not above) and wv[i] <= level):
            return True
        i += 1
    return False


@njit(cache=True)
def _nv(x, node):
    return 0.0 if node == 0 else x[node - 1]


@njit(cache=True)
def eval_all(x, ci, sD, sG, sS, P, na, vbi, rg, fg, table, ev):
    nn = ci[CI_NN]
    nv = ci[CI_NV]
    ns = ci[CI_NS]
    ok = True
    for k in range(ns):
        ku = nn - 1 + nv + 2 * k
        P[k, 11] = _nv(x, sG[k]) - _nv(x, sS[k])
        if not stl_eval(x[ku], x[ku + 1], P[k], na, vbi, rg, fg, table, ev[k]):
            ok = False
    return ok


@njit(cache=True)
def fd_partials(x, ci, P, na, vbi, rg, fg, table, ev, part, tmp):
    nn = ci[CI_NN]
    nv = ci[CI_NV]
    ns = ci[CI_NS]
    for k in range(ns):
        ku = nn - 1 + nv + 2 * k
        u = x[ku]
        r = x[ku + 1]
        okU = stl_eval(u + FD_U, r, P[k], na, vbi, rg, fg, table, tmp)
        if okU:
            part[k, P_VU] = (tmp[0] - ev[k, 0]) / FD_U
            part[k, P_IU] = (tmp[1] - ev[k, 1]) / FD_U
            part[k, P_FU] = (tmp[2] - ev[k, 2]) / FD_U
            part[k, P_QU] = (tmp[3] - ev[k, 3]) / FD_U
        else:
            okU = stl_eval(u - FD_U, r, P[k], na, vbi, rg, fg, table, tmp)
            if okU:
                part[k, P_VU] = (ev[k, 0] - tmp[0]) / FD_U
                part[k, P_IU] = (ev[k, 1] - tmp[1]) / FD_U
                part[k, P_FU] = (ev[k, 2] - tmp[2]) / FD_U
                part[k, P_QU] = (ev[k, 3] - tmp[3]) / FD_U
        okR = stl_eval(u, r + FD_R, P[k], na, vbi, rg, fg, table, tmp)
        if okR:
            part[k, P_VR] = (tmp[0] - ev[k, 0]) / FD_R
            part[k, P_IR] = (tmp[1] - ev[k, 1]) / FD_R
            part[k, P_FR] = (tmp[2] - ev[k, 2]) / FD_R
            part[k, P_QR] = (tmp[3] - ev[k, 3]) / FD_R
        else:
            okR = stl_eval(u, r - FD_R, P[k], na, vbi, rg, fg, table, tmp)
            if okR:
                part[k, P_VR] = (ev[k, 0] - tmp[0]) / FD_R
                part[k, P_IR] = (ev[k, 1] - tmp[1]) / FD_R
                part[k, P_FR] = (ev[k, 2] - tmp[2]) / FD_R
                part[k, P_QR] = (ev[k, 3] - tmp[3]) / FD_R
        if not (okU and okR):
            return False
        if part[k, P_VGSJ] > 0.5:
            vgs = P[k, 11]
            P[k, 11] = vgs + FD_VGS
            okG = stl_eval(u, r, P[k], na, vbi, rg, fg, table, tmp)
            P[k, 11] = vgs
            if okG:
                part[k, P_VG] = (tmp[0] - ev[k, 0]) / FD_VGS
                part[k, P_IG] = (tmp[1] - ev[k, 1]) / FD_VGS
                part[k, P_FG] = (tmp[2] - ev[k, 2]) / FD_VGS
                part[k, P_QG] = (tmp[3] - ev[k, 3]) / FD_VGS
            else:
                part[k, P_VG] = 0.0
                part[k, P_IG] = 0.0
                part[k, P_FG] = 0.0
                part[k, P_QG] = 0.0
    return True


@njit(cache=True)
def assemble(x, ci, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
             sD, sG, sS, ev, part, emode, qc, tha, h, J, f, vW, cmp, basic):
    """Residual f(x) and Jacobian J.  emode 0: charge equation, 1: u fixed at qc[k]."""
    nn = ci[CI_NN]
    nv = ci[CI_NV]
    ns = ci[CI_NS]
    n = f.shape[0]
    for i in range(n):
        f[i] = 0.0
        for j in range(n):
            J[i, j] = 0.0
    for i in range(nn - 1):
        f[i] += GMIN * x[i]
        J[i, i] += GMIN
    for e in range(rA.shape[0]):
        a = rA[e]
        b = rB[e]
        g = rG[e]
        cur = g * (_nv(x, a) - _nv(x, b))
        if a > 0:
            f[a - 1] += cur
            J[a - 1, a - 1] += g
            if b > 0:
                J[a - 1, b - 1] -= g
        if b > 0:
            f[b - 1] -= cur
            J[b - 1, b - 1] += g
            if a > 0:
                J[b - 1, a - 1] -= g
    for e in range(cA.shape[0]):
        a = cA[e]
        b = cB[e]
        g = cGeq[e]
        cur = g * (_nv(x, a) - _nv(x, b)) + cIeq[e]
        if a > 0:
            f[a - 1] += cur
            J[a - 1, a - 1] += g
            if b > 0:
                J[a - 1, b - 1] -= g
        if b > 0:
            f[b - 1] -= cur
            J[b - 1, b - 1] += g
            if a > 0:
                J[b - 1, a - 1] -= g
    for e in range(vA.shape[0]):
        a = vA[e]
        b = vB[e]
        iv = nn - 1 + e
        cur = x[iv]
        if a > 0:
            f[a - 1] += cur
            J[a - 1, iv] += 1.0
            J[iv, a - 1] += 1.0
        if b > 0:
            f[b - 1] -= cur
            J[b - 1, iv] -= 1.0
            J[iv, b - 1] -= 1.0
        if vW[e] >= 0:
            f[iv] = _nv(x, a) - _nv(x, b) - vval[e]
        else:
            # comparator output: behavioural (input-controlled) voltage source
            j = -1 - vW[e]
            y, dy = cmp_value(x, cmp, j)
            f[iv] = _nv(x, a) - _nv(x, b) - y
            ni = int(cmp[j, K_IN])
            nm = int(cmp[j, K_INM])
            if ni > 0:
                J[iv, ni - 1] -= dy
            if nm > 0:
                J[iv, nm - 1] += dy
    for e in range(iA.shape[0]):
        a = iA[e]
        b = iB[e]
        if a > 0:
            f[a - 1] += ival[e]
        if b > 0:
            f[b - 1] -= ival[e]
    stamp_basic(x, basic, f, J)
    for k in range(ns):
        ku = nn - 1 + nv + 2 * k
        kr = ku + 1
        d = sD[k]
        s = sS[k]
        idr = ev[k, 1]
        if d > 0:
            f[d - 1] += idr
            J[d - 1, ku] += part[k, P_IU]
            J[d - 1, kr] += part[k, P_IR]
        if s > 0:
            f[s - 1] -= idr
            J[s - 1, ku] -= part[k, P_IU]
            J[s - 1, kr] -= part[k, P_IR]
        f[ku] = ev[k, 0] - (_nv(x, d) - _nv(x, s))
        J[ku, ku] = part[k, P_VU]
        J[ku, kr] = part[k, P_VR]
        if d > 0:
            J[ku, d - 1] -= 1.0
        if s > 0:
            J[ku, s - 1] += 1.0
        if emode == 0:
            th = tha[k]
            f[kr] = (ev[k, 3] - qc[k] - th * h * ev[k, 2]) / COX
            J[kr, ku] = (part[k, P_QU] - th * h * part[k, P_FU]) / COX
            J[kr, kr] = (part[k, P_QR] - th * h * part[k, P_FR]) / COX
        else:
            f[kr] = x[ku] - qc[k]
            J[kr, ku] = 1.0
        if part[k, P_VGSJ] > 0.5:
            # V_GS = v_g - v_s enters I_D, V_D, F and Q (the element re-evaluates it at every iteration)
            gq = (part[k, P_QG] - tha[k] * h * part[k, P_FG]) / COX if emode == 0 else 0.0
            for node, sgn in ((sG[k], 1.0), (s, -1.0)):
                if node > 0:
                    c = node - 1
                    if d > 0:
                        J[d - 1, c] += sgn * part[k, P_IG]
                    if s > 0:
                        J[s - 1, c] -= sgn * part[k, P_IG]
                    J[ku, c] += sgn * part[k, P_VG]
                    J[kr, c] += sgn * gq


@njit(cache=True)
def newton(x, ci, cf, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
           sD, sG, sS, P, na, vbi, rg, fg, table, ev, part, tmp, emode, qc, tha, h,
           refresh, maxit, J, f, xt, evt, vW, cmp, basic, tolmul=1.0):
    """Solve the nonlinear system at one time point in place (x, ev, part).
    Returns (converged, iterations)."""
    nn = ci[CI_NN]
    nv = ci[CI_NV]
    ns = ci[CI_NS]
    n = x.shape[0]
    tol = cf[CF_NEWTOL] * tolmul
    if not eval_all(x, ci, sD, sG, sS, P, na, vbi, rg, fg, table, ev):
        return False, 0
    last_norm = 1e300
    for it in range(maxit):
        if refresh or it >= 2:
            if not fd_partials(x, ci, P, na, vbi, rg, fg, table, ev, part, tmp):
                return False, it + 1
        assemble(x, ci, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                 sD, sG, sS, ev, part, emode, qc, tha, h, J, f, vW, cmp, basic)
        for i in range(n):
            if not np.isfinite(f[i]):
                return False, it + 1
        dx = np.linalg.solve(J, -f)
        ok = True
        for i in range(n):
            if not np.isfinite(dx[i]):
                ok = False
        if not ok:
            return False, it + 1
        alpha = basic_step_limit(x, dx, basic)
        small = True
        for k in range(ns):
            ku = nn - 1 + nv + 2 * k
            au = abs(dx[ku])
            ar = abs(dx[ku + 1])
            if au > DU_LIM:
                alpha = min(alpha, DU_LIM / au)
            if ar > DR_LIM:
                alpha = min(alpha, DR_LIM / ar)
            if au > tol or ar > 10 * tol:
                small = False
        for i in range(nn - 1):
            if abs(dx[i]) > 10 * tol * (1.0 + abs(x[i])):
                small = False
        # residual-based part of the convergence test (E1, E2 rows in volts)
        for k in range(ns):
            ku = nn - 1 + nv + 2 * k
            if abs(f[ku]) > 100 * tol or abs(f[ku + 1]) > 100 * tol:
                small = False
        if small:
            for i in range(n):
                x[i] += dx[i]
            return True, it + 1
        nrm = 0.0
        for k in range(ns):
            ku = nn - 1 + nv + 2 * k
            nrm = max(nrm, abs(dx[ku]), 0.1 * abs(dx[ku + 1]))
        found = False
        for bt in range(14):
            for i in range(n):
                xt[i] = x[i] + alpha * dx[i]
            if eval_all(xt, ci, sD, sG, sS, P, na, vbi, rg, fg, table, evt):
                found = True
                break
            alpha *= 0.5
        if not found:
            return False, it + 1
        for i in range(n):
            x[i] = xt[i]
        for k in range(ns):
            for j in range(N_EV):
                ev[k, j] = evt[k, j]
        # slow convergence -> refresh the Jacobian next iteration
        refresh = nrm > 0.3 * last_norm or alpha < 1.0
        last_norm = nrm
    return False, maxit


@njit(cache=True)
def cap_companion(ci, method_cap, h, cA, cB, cC, cv, cI, cGeq, cIeq):
    for e in range(cA.shape[0]):
        if method_cap == 1:
            g = 2.0 * cC[e] / h
            cGeq[e] = g
            cIeq[e] = -g * cv[e] - cI[e]
        else:
            g = cC[e] / h
            cGeq[e] = g
            cIeq[e] = -g * cv[e]


@njit(cache=True)
def sources_at(t, vW, iW, sW, wt, wv, woff, vval, ival, P, Pbase):
    for e in range(vW.shape[0]):
        vval[e] = wave_value(vW[e], t, wt, wv, woff) if vW[e] >= 0 else 0.0
    for e in range(iW.shape[0]):
        ival[e] = wave_value(iW[e], t, wt, wv, woff)
    for k in range(sW.shape[0]):
        if sW[k] >= 0:
            P[k, 13] = wave_value(sW[k], t, wt, wv, woff)
        else:
            P[k, 13] = Pbase[k, 13]


@njit(cache=True)
def sensitivities(x, ci, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                  sD, sG, sS, ev, part, J, f, qc0, sens, ss, vW, cmp, basic):
    """dx/dQ_k (charge fixed, theta = 0) and the local relaxation time
    tau_k = 1/|dF_k/dQ_k| along the circuit constraints."""
    nn = ci[CI_NN]
    nv = ci[CI_NV]
    ns = ci[CI_NS]
    n = x.shape[0]
    if ns == 0:
        return
    assemble(x, ci, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
             sD, sG, sS, ev, part, 0, qc0, np.zeros(ns), 0.0, J, f, vW, cmp, basic)
    B = np.zeros((n, ns))
    for k in range(ns):
        B[nn - 1 + nv + 2 * k + 1, k] = 1.0 / COX
    S = np.linalg.solve(J, B)
    for k in range(ns):
        ku = nn - 1 + nv + 2 * k
        for i in range(n):
            sens[k, i] = S[i, k]
        dfdq = part[k, P_FU] * S[ku, k] + part[k, P_FR] * S[ku + 1, k]
        if dfdq != 0.0 and np.isfinite(dfdq):
            ss[k, SS_TAU] = 1.0 / abs(dfdq)
            ss[k, SS_DQ] = dfdq
        else:
            ss[k, SS_TAU] = 1e30
            ss[k, SS_DQ] = 0.0


@njit(cache=True)
def poisson_or_gauss(lam, gth):
    if lam <= 0.0:
        return 0.0
    if lam > gth:
        v = lam + np.sqrt(lam) * np.random.normal()
        return v if v > 0.0 else 0.0
    return float(np.random.poisson(lam))


@njit(cache=True)
def draw_dq(h, unit, gq, lq, r, rv, pmf, pk, gth):
    """Charge increment over h from the event rates of the current state (Eq. 2):
    dQ = q (N_unit + sum of cluster sizes - N_loss).  Signs of the extension currents are
    folded consistently so that E[dQ] = h (G - L) = h F exactly."""
    ii = gq - unit
    up1 = (max(unit, 0.0) + max(-lq, 0.0)) / QE
    dn1 = (max(lq, 0.0) + max(-unit, 0.0) + max(-ii, 0.0)) / QE
    iic = max(ii, 0.0) / QE
    n_up = poisson_or_gauss(up1 * h, gth)
    n_dn = poisson_or_gauss(dn1 * h, gth)
    s = 0.0
    if iic > 0.0:
        p1, m1, m2 = pmf_at(r, rv, pmf, pk)
        if m1 <= 0.0 or p1 <= 0.0:
            s = poisson_or_gauss(iic * h, gth)
        else:
            lam = iic * h * p1 / m1
            if lam > gth:
                mean = iic * h
                v = mean + np.sqrt(iic * h * m2 / m1) * np.random.normal()
                s = v if v > 0.0 else 0.0
            else:
                nc = np.random.poisson(lam)
                K = pk.shape[0] - 1
                for c in range(nc):
                    uu = np.random.random() * p1
                    acc = 0.0
                    kk = K
                    for k in range(1, K + 1):
                        acc += pk[k]
                        if uu <= acc:
                            kk = k
                            break
                    s += kk
    return QE * (n_up + s - n_dn)


@njit(cache=True)
def noise_var_rate(unit, gq, lq, r, rv, pmf, pk):
    """Diffusion coefficient of the body charge, q^2 (N_up + N_down + II rate * M2/M1)  (C^2/s)."""
    ii = gq - unit
    up1 = (max(unit, 0.0) + max(-lq, 0.0)) / QE
    dn1 = (max(lq, 0.0) + max(-unit, 0.0) + max(-ii, 0.0)) / QE
    iic = max(ii, 0.0) / QE
    m2m1 = 1.0
    if iic > 0.0:
        p1, m1, m2 = pmf_at(r, rv, pmf, pk)
        if m1 > 0.0 and p1 > 0.0:
            m2m1 = m2 / m1
    return QE * QE * (up1 + dn1 + iic * m2m1)


@njit(cache=True)
def total_event_rate(unit, gq, lq, r, rv, pmf, pk):
    ii = gq - unit
    up1 = (max(unit, 0.0) + max(-lq, 0.0)) / QE
    dn1 = (max(lq, 0.0) + max(-unit, 0.0) + max(-ii, 0.0)) / QE
    iic = max(ii, 0.0) / QE
    lam = 0.0
    if iic > 0.0:
        p1, m1, m2 = pmf_at(r, rv, pmf, pk)
        if m1 > 0.0 and p1 > 0.0:
            lam = iic * p1 / m1
        else:
            lam = iic
    return up1 + dn1 + lam


@njit(cache=True)
def dc_op(x, ci, cf, rA, rB, rG, cA, cB, cC, vA, vB, vW, iA, iB, iW, sD, sG, sS, sW,
          wt, wv, woff, P, Pbase, na, vbi, rg, fg, table, ev, part, t0, hold, cmp, basic):
    """DC operating point at t0: u fixed at 0 (empty body) -> pseudo-transient BE continuation
    of the charge equation (finds the low-current state reachable from an empty body).
    ``hold`` (one conductance per capacitor, S): 0 = capacitor open (the DC operating point); > 0 =
    the capacitor is held at 0 V by that conductance (initial state 'zero', SPICE UIC with IC = 0:
    discharged capacitors, the bodies relax to their steady state at the resulting terminal voltages)."""
    nn = ci[CI_NN]
    nv = ci[CI_NV]
    ns = ci[CI_NS]
    n = x.shape[0]
    J = np.zeros((n, n))
    f = np.zeros(n)
    xt = np.zeros(n)
    evt = np.zeros((ns, N_EV))
    tmp = np.zeros(N_EV)
    vval = np.zeros(vA.shape[0])
    ival = np.zeros(iA.shape[0])
    nc = cA.shape[0]
    cGeq = np.zeros(nc)
    cIeq = np.zeros(nc)
    cv = np.zeros(nc)
    cI = np.zeros(nc)
    qc = np.zeros(ns)
    sources_at(t0, vW, iW, sW, wt, wv, woff, vval, ival, P, Pbase)
    held = False
    for e in range(nc):
        cGeq[e] = hold[e]
        if hold[e] > 0.0:
            held = True
    # phase 1: u = 0, capacitors open (h = inf) or held at 0 V
    for k in range(ns):
        qc[k] = 0.0
    ok, it = newton(x, ci, cf, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                    sD, sG, sS, P, na, vbi, rg, fg, table, ev, part, tmp, 1, qc, np.ones(ns), 0.0,
                    True, 60, J, f, xt, evt, vW, cmp, basic)
    if not ok:
        return False
    # phase 2: pseudo-transient
    xold = x.copy()
    h = 1e-12
    for k in range(ns):
        qc[k] = ev[k, 3]
    for e in range(nc):
        cv[e] = _nv(x, cA[e]) - _nv(x, cB[e])
    for itr in range(600):
        if held:
            for e in range(nc):
                cGeq[e] = hold[e]
                cIeq[e] = 0.0
        else:
            cap_companion(ci, 0, h, cA, cB, cC, cv, cI, cGeq, cIeq)
        ok, it = newton(x, ci, cf, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                        sD, sG, sS, P, na, vbi, rg, fg, table, ev, part, tmp, 0, qc, np.ones(ns), h,
                        True, 40, J, f, xt, evt, vW, cmp, basic)
        if not ok:
            for i in range(n):
                x[i] = xold[i]
            h *= 0.2
            if h < 1e-18:
                return False
            continue
        dmax = 0.0
        for k in range(ns):
            ku = nn - 1 + nv + 2 * k
            dmax = max(dmax, abs(x[ku] - xold[ku]))
        for k in range(ns):
            qc[k] = ev[k, 3]
        for e in range(nc):
            cv[e] = _nv(x, cA[e]) - _nv(x, cB[e])
        for i in range(n):
            xold[i] = x[i]
        if h > 1e4 and dmax < 1e-10:
            eval_all(x, ci, sD, sG, sS, P, na, vbi, rg, fg, table, ev)
            fd_partials(x, ci, P, na, vbi, rg, fg, table, ev, part, tmp)
            return True
        h = min(h * 3.0, 1e8)
    return False


@njit(cache=True)
def _xfrac(i0, i1, ith, up):
    """Fraction of the step at which I_D crosses ith (log-linear interpolation)."""
    if i0 > 0 and i1 > 0 and ith > 0:
        a = (np.log(ith) - np.log(i0)) / (np.log(i1) - np.log(i0)) if i1 != i0 else 1.0
    else:
        a = (ith - i0) / (i1 - i0) if i1 != i0 else 1.0
    return min(max(a, 0.0), 1.0)


@njit(cache=True)
def _set_pending(ss, k, a, t_old, h, xp, x, sD, sS, i0, i1, mainw, wt, wv, woff):
    """Store a candidate latch transition at the fraction a of the step just accepted."""
    a = min(max(a, 0.0), 1.0)
    te = t_old + a * h
    vds0 = _nv(xp, sD[k]) - _nv(xp, sS[k])
    vds1 = _nv(x, sD[k]) - _nv(x, sS[k])
    ss[k, SS_PEND] = 1.0
    ss[k, SS_PT] = te
    ss[k, SS_PV] = vds0 + a * (vds1 - vds0)
    ss[k, SS_PS] = wave_value(mainw, te, wt, wv, woff) if mainw >= 0 else np.nan
    if i0 > 0 and i1 > 0:
        ss[k, SS_PI] = np.exp(np.log(i0) + a * (np.log(i1) - np.log(i0)))
    else:
        ss[k, SS_PI] = i0 + a * (i1 - i0)


@njit(cache=True)
def run_chunk(x, xp, ci, cf, rA, rB, rG, cA, cB, cC, vA, vB, vW, iA, iB, iW, sD, sG, sS, sW,
              wt, wv, woff, bp, samp, P, Pbase, na, vbi, rg, fg, table, rv, pmf,
              ss, part, sens, ls, cv, cI, sf, si, rec, evb, sbuf, win, cmp, basic):
    """Advance the transient from sf[SF_T] to sf[SF_TSTOP] (or until a budget is hit).
    All state is kept in the arrays; returns the status code (also stored in si[SI_STATUS])."""
    nn = ci[CI_NN]
    nv = ci[CI_NV]
    ns = ci[CI_NS]
    nc = cA.shape[0]
    n = x.shape[0]
    stoch = ci[CI_STOCH] == 1
    carrier = stoch and ci[CI_CARRIER] == 1
    method = ci[CI_METHOD]
    lsmode = ci[CI_LSMODE] if stoch else 0      # > 0: some cell has local states (per-cell config in win)
    t_end = cf[CF_TEND]
    t_stop = min(sf[SF_TSTOP], t_end)
    dt_min = cf[CF_DTMIN]
    dt_max = cf[CF_DTMAX]
    du_max = cf[CF_DUMAX]
    dlni_max = cf[CF_DLNIMAX]
    dv_max = cf[CF_DVMAX]
    lte_u = cf[CF_LTEU]
    tau_frac = cf[CF_TAUFRAC]
    nev_max = cf[CF_NEVMAX]
    h_noise_min = cf[CF_HNOISEMIN]
    gth = cf[CF_GAUSS]
    i_th = cf[CF_ITH]
    i_dn = cf[CF_ITHDN]
    i_floor = cf[CF_IFLOOR]
    rec_w = rec.shape[1]
    nb = bp.shape[0]
    nsamp = samp.shape[0]
    mainw = ci[CI_MAINW]
    J = np.zeros((n, n))
    f = np.zeros(n)
    xt = np.zeros(n)
    x0 = np.zeros(n)
    evt = np.zeros((ns, N_EV))
    ev = np.zeros((ns, N_EV))
    ev0 = np.zeros((ns, N_EV))
    tmp = np.zeros(N_EV)
    vval = np.zeros(vA.shape[0])
    ival = np.zeros(iA.shape[0])
    cGeq = np.zeros(nc)
    cIeq = np.zeros(nc)
    qc = np.zeros(ns)
    dq = np.zeros(ns)
    ls_new = np.zeros((ns, 2))
    pk = np.zeros(pmf.shape[1])
    reg = np.zeros(ns, np.int64)
    tha = np.ones(ns)
    lat_prev = np.zeros(ns)
    te_step = np.zeros(ns)
    ldnoise = ci[CI_LDNOISE] == 1
    # comparator output nodes are algebraic (ideal controlled source): not part of the node-voltage error control
    algn = np.zeros(max(nn - 1, 1), np.bool_)
    for e in range(vA.shape[0]):
        if vW[e] < 0 and vA[e] > 0:
            algn[vA[e] - 1] = True
    si[SI_CHUNKSTEPS] = 0
    t = sf[SF_T]
    # current element outputs at the accepted state
    for k in range(ns):
        P[k, 11] = _nv(x, sG[k]) - _nv(x, sS[k])
    sources_at(t, vW, iW, sW, wt, wv, woff, vval, ival, P, Pbase)
    for k in range(ns):
        if lsmode > 0 and win[k, W_LSMODE] > 0.5:
            ix = int(win[k, W_LSIDX])
            P[k, ix] = Pbase[k, ix] + ls[k, 0]
            P[k, 10] = Pbase[k, 10] + ls[k, 1]
    eval_all(x, ci, sD, sG, sS, P, na, vbi, rg, fg, table, ev0)
    status = ST_DONE
    while True:
        if t >= t_stop - 1e-15 * max(1.0, abs(t_stop)) or t >= t_end:
            status = ST_DONE
            break
        if si[SI_STEPS] >= ci[CI_MAXSTEPS]:
            status = ST_MAXSTEPS
            break
        if si[SI_CHUNKSTEPS] >= ci[CI_CHUNK]:
            status = ST_CHUNK
            break
        if si[SI_NREC] >= rec.shape[0] - 2 or si[SI_NEV] >= evb.shape[0] - 2 * ns - 2:
            status = ST_BUFFER
            break
        # ---- step proposal ------------------------------------------------------------
        while si[SI_BP] < nb and bp[si[SI_BP]] <= t + 1e-12 * max(1e-9, abs(t)):
            si[SI_BP] += 1
        h = min(sf[SF_HNEXT], dt_max)
        sreg = 0
        if carrier:
            sreg = 5
            for k in range(ns):
                # tier choice from the physical (committed) latch state, not from the reporting threshold
                latk = ss[k, SS_PHYS] > 0.5
                reg[k] = 4
                if latk and not ldnoise:
                    sreg = min(sreg, 4)
                    continue
                tk = ss[k, SS_TAU]
                vdsk = _nv(x, sD[k]) - _nv(x, sS[k])
                if latk:
                    blo = win[k, W_LDLO]
                    bhi = win[k, W_LDHI]
                else:
                    blo = win[k, W_LULO]
                    bhi = win[k, W_LUHI]
                outside = vdsk < blo or vdsk > bhi
                lw = int(win[k, W_LAW])
                if outside and lw >= 0 and cf[CF_NLOOK] > 0 and tk < 1e29:
                    # look-ahead: resolve the noise already n_look relaxation times before the drive
                    # enters the band, so the stationary fluctuation is built up on fast ramps / edges
                    tl = t + cf[CF_NLOOK] * tk
                    if win[k, W_LAMODE] < 0.5:
                        # benches: the drive waveform is the cell's V_DS (series R, grounded source)
                        if latk:
                            outside = not wave_reaches(lw, t, tl, bhi, False, wt, wv, woff)
                        else:
                            outside = not wave_reaches(lw, t, tl, blo, True, wt, wv, woff)
                    else:
                        # general circuit: V_DS(t') ~ V_DS(t) + g (w(t') - w(t)), g = dV_DS/dw (linear network)
                        gk = win[k, W_LAG]
                        if gk != 0.0 and np.isfinite(gk):
                            lev = wave_value(lw, t, wt, wv, woff) + ((bhi if latk else blo) - vdsk) / gk
                            outside = not wave_reaches(lw, t, tl, lev, (gk > 0.0) != latk, wt, wv, woff)
                if outside:
                    # outside the noise band (barrier > noise_z_max SDs or monostable): drift only
                    reg[k] = 5
                elif tk < cf[CF_GTAUMIN] and tau_frac * tk < h_noise_min:
                    reg[k] = 3
                elif tau_frac * tk >= h_noise_min:
                    reg[k] = 1
                    h = min(h, tau_frac * tk)
                    rate = total_event_rate(ss[k, SS_UNIT], ss[k, SS_G], ss[k, SS_L], ss[k, SS_R], rv, pmf, pk)
                    if rate > 0.0:
                        h = min(h, nev_max / rate)
                elif tk >= cf[CF_GTAUMIN]:
                    reg[k] = 2
                    h = min(h, cf[CF_GTAUFRAC] * tk)
                else:
                    reg[k] = 3
                if reg[k] <= 2:
                    fk = abs(ss[k, SS_FN])
                    if fk > 0.0:
                        h = min(h, du_max * abs(part[k, P_QU]) / fk)
                sreg = min(sreg, reg[k])
        hit = False
        if si[SI_BP] < nb:
            tb = bp[si[SI_BP]]
            if t + h >= tb - 1e-3 * h:
                h = tb - t
                hit = True
            elif t + 2.0 * h > tb:
                h = 0.5 * (tb - t)
        if t + h > t_end:
            h = t_end - t
            hit = True
        if h < dt_min:
            h = dt_min
        if t + h == t:
            status = ST_FAIL
            break
        # ---- attempts ------------------------------------------------------------------
        accepted = False
        err = 0.0
        use_trap = False
        iters = 0
        for att in range(40):
            t1 = t + h
            sources_at(t1, vW, iW, sW, wt, wv, woff, vval, ival, P, Pbase)
            for k in range(ns):
                if lsmode > 0 and win[k, W_LSMODE] > 1.5:
                    # evolving local states: Ornstein-Uhlenbeck step (per-cell sigma / tau)
                    a = np.exp(-h / win[k, W_LSTAU]) if win[k, W_LSTAU] > 0 else 0.0
                    ls_new[k, 0] = ls[k, 0] * a + win[k, W_LSSIG] * np.sqrt(max(1.0 - a * a, 0.0)) * np.random.normal()
                    a2 = np.exp(-h / win[k, W_LSETAU]) if win[k, W_LSETAU] > 0 else 0.0
                    ls_new[k, 1] = ls[k, 1] * a2 + win[k, W_LSESIG] * np.sqrt(max(1.0 - a2 * a2, 0.0)) * np.random.normal()
                else:
                    ls_new[k, 0] = ls[k, 0]
                    ls_new[k, 1] = ls[k, 1]
            if lsmode > 0:
                for k in range(ns):
                    if win[k, W_LSMODE] > 0.5:
                        ix = int(win[k, W_LSIDX])
                        P[k, ix] = Pbase[k, ix] + ls_new[k, 0]
                        P[k, 10] = Pbase[k, 10] + ls_new[k, 1]
            # charge-equation mode for this step (per element)
            use_trap = False
            if carrier:
                for k in range(ns):
                    if reg[k] == 1:
                        # event-level explicit tau-leap (Eq. 2): Q_{n+1} = Q_n + q (N_unit + sum k_i - N_loss)
                        tha[k] = 0.0
                        dq[k] = draw_dq(h, ss[k, SS_UNIT], ss[k, SS_G], ss[k, SS_L], ss[k, SS_R], rv, pmf, pk, gth)
                        qc[k] = ss[k, SS_QN] + dq[k]
                    elif reg[k] == 2:
                        # Gaussian limit, drift-implicit, variance-corrected (exact stationary variance
                        # of the linearised OU process): eta ~ N(0, D h (1 + h/(2 tau)))
                        tha[k] = 1.0
                        dvar = noise_var_rate(ss[k, SS_UNIT], ss[k, SS_G], ss[k, SS_L], ss[k, SS_R], rv, pmf, pk)
                        dq[k] = np.sqrt(dvar * h * (1.0 + 0.5 * h / ss[k, SS_TAU])) * np.random.normal()
                        qc[k] = ss[k, SS_QN] + dq[k]
                    else:
                        # drift only (implicit BE)
                        tha[k] = 1.0
                        qc[k] = ss[k, SS_QN]
                cap_companion(ci, 0, h, cA, cB, cC, cv, cI, cGeq, cIeq)
            else:
                taumin = 1e300
                for k in range(ns):
                    taumin = min(taumin, ss[k, SS_TAU])
                if method == 1 and h < 2.0 * taumin:
                    use_trap = True
                    for k in range(ns):
                        tha[k] = 0.5
                        qc[k] = ss[k, SS_QN] + 0.5 * h * ss[k, SS_FN]
                    cap_companion(ci, 1, h, cA, cB, cC, cv, cI, cGeq, cIeq)
                else:
                    for k in range(ns):
                        tha[k] = 1.0
                        qc[k] = ss[k, SS_QN]
                    cap_companion(ci, 0, h, cA, cB, cC, cv, cI, cGeq, cIeq)
            # predictor
            for i in range(n):
                x0[i] = x[i]
            if carrier:
                # linear response of the charge-fixed solution: dQ = (Qc - Q_n + th h F_n)/(1 - th h dF/dQ)
                for k in range(ns):
                    den = 1.0 - tha[k] * h * ss[k, SS_DQ]
                    if den < 0.2:
                        den = 0.2
                    dqp = (qc[k] - ss[k, SS_QN] + tha[k] * h * ss[k, SS_FN]) / den
                    for i in range(n):
                        x0[i] += sens[k, i] * dqp
            elif si[SI_HAVEPREV] == 1 and sf[SF_HPREV] > 0:
                ratio = h / sf[SF_HPREV]
                for i in range(n):
                    x0[i] += (x[i] - xp[i]) * ratio
            for k in range(ns):
                ku = nn - 1 + nv + 2 * k
                du = x0[ku] - x[ku]
                if du > 0.02:
                    x0[ku] = x[ku] + 0.02
                elif du < -0.02:
                    x0[ku] = x[ku] - 0.02
            refresh = si[SI_REFRESH] == 1 or att > 0
            tolmul = 20.0 if (carrier and sreg <= 2) else 1.0
            ok, iters = newton(x0, ci, cf, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                               sD, sG, sS, P, na, vbi, rg, fg, table, ev, part, tmp, 0, qc, tha, h,
                               refresh, 14, J, f, xt, evt, vW, cmp, basic, tolmul)
            si[SI_NEWT] += iters
            si[SI_DIAG + 3 * sreg + 1] += iters
            if not ok:
                si[SI_DIAG + 3 * sreg + 2] += 1
                si[SI_REJ] += 1
                si[SI_FAILNEWTON] += 1
                si[SI_REFRESH] = 1
                h *= 0.25
                hit = False
                if h < dt_min or t + h == t:
                    break
                continue
            # ---- error control ---------------------------------------------------------
            dvm = 0.0
            for i in range(nn - 1):
                if not algn[i]:
                    dvm = max(dvm, abs(x0[i] - x[i]))
            err = dvm / dv_max
            for k in range(ns):
                ku = nn - 1 + nv + 2 * k
                duk = abs(x0[ku] - x[ku])
                if carrier and reg[k] <= 2:
                    # noise-driven moves are not step-controlled (a priori step rules instead)
                    if duk > 20.0 * du_max:
                        err = max(err, 10.0)
                    continue
                dlk = abs(np.log(abs(ev[k, 1]) + i_floor) - np.log(abs(ev0[k, 1]) + i_floor))
                qu = abs(part[k, P_QU])
                ltek = 0.5 * h * abs(ev[k, 2] - ev0[k, 2]) / qu if qu > 0 else 0.0
                err = max(err, duk / du_max, dlk / dlni_max, ltek / lte_u)
            if (cf[CF_LTEV] > 0.0 and not carrier and lsmode != 2 and si[SI_HAVEPREV] == 1
                    and sf[SF_HPREV] > 0.0):
                # node-voltage LTE (custom circuits): BE-order estimate from the linear-extrapolation predictor
                hp = sf[SF_HPREV]
                wl = h / (h + hp)
                for i in range(nn - 1):
                    if algn[i]:
                        continue
                    pred = x[i] + (x[i] - xp[i]) * (h / hp)
                    tolv = cf[CF_LTEV] * max(abs(x0[i]), abs(x[i])) + cf[CF_LTEVABS]
                    ev_ = abs(x0[i] - pred) * wl / tolv
                    if ev_ > err:
                        err = ev_
            if err > 1.5 and h > dt_min * 1.01:
                si[SI_DIAG + 3 * sreg + 2] += 1
                si[SI_REJ] += 1
                h = max(h * max(0.1, 0.7 / err), dt_min)
                hit = False
                continue
            accepted = True
            break
        if not accepted:
            status = ST_FAIL
            break
        # ---- accept ---------------------------------------------------------------------
        t_old = t
        t = t + h
        sf[SF_T] = t
        si[SI_STEPS] += 1
        si[SI_DIAG + 3 * sreg] += 1
        si[SI_CHUNKSTEPS] += 1
        if method == 1 and not carrier and not use_trap:
            si[SI_TRAPBE] += 1
        if carrier:
            # regime times are cell-averaged (h / n_cells per cell), so they add up to at most t
            f3 = False
            f2 = False
            hc = h / ns if ns > 0 else 0.0
            for k in range(ns):
                if reg[k] == 3:
                    f3 = True
                    sf[SF_TUNRES] += hc
                elif reg[k] == 2:
                    f2 = True
                    sf[SF_TGAUSS] += hc
                elif reg[k] == 4:
                    sf[SF_TLRS] += hc
                elif reg[k] == 5:
                    sf[SF_TBAND] += hc
            if f3:
                si[SI_UNRES] += 1
            if f2:
                si[SI_GAUSS] += 1
        # capacitor state
        for e in range(nc):
            vnew = _nv(x0, cA[e]) - _nv(x0, cB[e])
            cI[e] = cGeq[e] * vnew + cIeq[e]
            cv[e] = vnew
        for i in range(n):
            xp[i] = x[i]
            x[i] = x0[i]
        si[SI_HAVEPREV] = 0 if hit else 1
        sf[SF_HPREV] = h
        for k in range(ns):
            ls[k, 0] = ls_new[k, 0]
            ls[k, 1] = ls_new[k, 1]
        # refresh partials for the next step if the state moved a lot
        si[SI_REFRESH] = 1 if (err > 0.3 or iters > 2) else 0
        if carrier and sreg <= 2:
            si[SI_REFRESH] = 1 if iters > 2 else 0
        # per-element state
        for k in range(ns):
            ku = nn - 1 + nv + 2 * k
            # charge state = value of the integration formula (exact event bookkeeping in the
            # explicit tier; Newton residual <= ~1e-4 q is not accumulated)
            ss[k, SS_QN] = qc[k] + tha[k] * h * ev[k, 2]
            ss[k, SS_FN] = ev[k, 2]
            ss[k, SS_UNIT] = ev[k, 4]
            ss[k, SS_G] = ev[k, 5]
            ss[k, SS_L] = ev[k, 6]
            ss[k, SS_R] = x[ku + 1]
            if x[ku] < sf[SF_MINU]:
                sf[SF_MINU] = x[ku]
            if x[ku + 1] < sf[SF_MINR]:
                sf[SF_MINR] = x[ku + 1]
            if x[ku] < 0:
                sf[SF_TNEGU] += h
            if x[ku + 1] < 0:
                sf[SF_TNEGR] += h
            qc[k] = ev[k, 3]
        sensitivities(x, ci, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                      sD, sG, sS, ev, part, J, f, qc, sens, ss, vW, cmp, basic)
        cmp_update_state(x, cmp)
        # ---- latch state and event detection ---------------------------------------------
        # The latch state is the body's branch (hysteresis on u between the fold values u_i and u_j);
        # an event is the switch of that state, timed at the I_D threshold crossing when it lies inside
        # the switching transient, else when the body reaches the new branch (module docstring).
        evflag = False
        for k in range(ns):
            ku = nn - 1 + nv + 2 * k
            i0 = ev0[k, 1]
            i1 = ev[k, 1]
            u0 = xp[ku]
            u1 = x[ku]
            u_i = win[k, W_UI]
            u_j = win[k, W_UJ]
            lat_prev[k] = ss[k, SS_LAT]
            te_step[k] = -1e300
            kind = 0
            if ss[k, SS_LAT] < 0.5:
                if i0 < i_th and i1 >= i_th and u1 < u_i:
                    si[SI_HRSX] += 1                       # HRS / channel conduction above i_th: no event
                if ss[k, SS_PEND] > 0.5 and u1 < u_i:
                    ss[k, SS_PEND] = 0.0                   # body back on the HRS: timing candidate withdrawn
                if ss[k, SS_PEND] < 0.5 and i1 >= i_th and u1 >= u_i:
                    # timing candidate: the later of the I_D crossing and the body leaving the HRS (u_i)
                    aI = _xfrac(i0, i1, i_th, True) if i0 < i_th else 0.0
                    aU = (u_i - u0) / (u1 - u0) if (u0 < u_i and u1 != u0) else 0.0
                    _set_pending(ss, k, max(aI, aU), t_old, h, xp, x, sD, sS, i0, i1, mainw, wt, wv, woff)
                if u1 >= u_j:                              # body reached the LRS: latch-up
                    if ss[k, SS_PEND] < 0.5:
                        aU = (u_j - u0) / (u1 - u0) if (u0 < u_j and u1 != u0) else 0.0
                        _set_pending(ss, k, aU, t_old, h, xp, x, sD, sS, i0, i1, mainw, wt, wv, woff)
                    kind = 1
            else:
                if ss[k, SS_PEND] > 0.5 and u1 > u_j:
                    ss[k, SS_PEND] = 0.0                   # body back on the LRS: timing candidate withdrawn
                if ss[k, SS_PEND] < 0.5 and i1 < i_dn and u1 <= u_j:
                    aI = _xfrac(i0, i1, i_dn, False) if i0 >= i_dn else 0.0
                    aU = (u0 - u_j) / (u0 - u1) if (u0 > u_j and u1 != u0) else 0.0
                    _set_pending(ss, k, max(aI, aU), t_old, h, xp, x, sD, sS, i0, i1, mainw, wt, wv, woff)
                if u1 <= u_i:                              # body reached the HRS: latch-down
                    if ss[k, SS_PEND] < 0.5:
                        aU = (u0 - u_i) / (u0 - u1) if (u0 > u_i and u1 != u0) else 0.0
                        _set_pending(ss, k, aU, t_old, h, xp, x, sD, sS, i0, i1, mainw, wt, wv, woff)
                    kind = 2
            ss[k, SS_PHYS] = ss[k, SS_LAT] if kind == 0 else (1.0 if kind == 1 else 0.0)
            if kind > 0:
                ne = si[SI_NEV]
                evb[ne, EV_KIND] = kind
                evb[ne, EV_STL] = k
                evb[ne, EV_T] = ss[k, SS_PT]
                evb[ne, EV_VDS] = ss[k, SS_PV]
                evb[ne, EV_VSRC] = ss[k, SS_PS]
                evb[ne, EV_I] = ss[k, SS_PI]
                si[SI_NEV] = ne + 1
                ss[k, SS_LAT] = 1.0 if kind == 1 else 0.0
                ss[k, SS_PEND] = 0.0
                te_step[k] = evb[ne, EV_T]
                evflag = True
        # ---- samples ---------------------------------------------------------------------
        sampflag = False
        while si[SI_NSAMP] < nsamp and samp[si[SI_NSAMP]] <= t + 1e-12 * max(1e-9, abs(t)):
            j = si[SI_NSAMP]
            ts = samp[j]
            a = (ts - t_old) / h if h > 0 else 1.0
            a = min(max(a, 0.0), 1.0)
            sbuf[j, 0] = ts
            for k in range(ns):
                sbuf[j, 1 + N_SAMPC * k] = ev0[k, 1] + a * (ev[k, 1] - ev0[k, 1])
                vds0 = _nv(xp, sD[k]) - _nv(xp, sS[k])
                vds1 = _nv(x, sD[k]) - _nv(x, sS[k])
                sbuf[j, 2 + N_SAMPC * k] = vds0 + a * (vds1 - vds0)
                # reported latch state at ts (the pre-step state if this step's event lies after ts)
                sbuf[j, 3 + N_SAMPC * k] = lat_prev[k] if te_step[k] > ts else ss[k, SS_LAT]
            si[SI_NSAMP] = j + 1
            sampflag = True
        # ---- recording ---------------------------------------------------------------------
        dorec = evflag or sampflag or (t - sf[SF_TREC] >= cf[CF_DTREC]) or t >= t_end
        if not dorec and t - sf[SF_TREC] >= 0.1 * cf[CF_DTREC]:
            for k in range(ns):
                vds = _nv(x, sD[k]) - _nv(x, sS[k])
                lni = np.log(abs(ev[k, 1]) + i_floor)
                if abs(vds - ss[k, SS_VDS_REC]) >= cf[CF_DVREC] or abs(lni - ss[k, SS_LNI_REC]) >= cf[CF_DLNIREC]:
                    dorec = True
        if dorec:
            j = si[SI_NREC]
            rec[j, 0] = t
            c = 1
            for i in range(nn - 1 + nv):
                rec[j, c] = x[i]
                c += 1
            for k in range(ns):
                ku = nn - 1 + nv + 2 * k
                rec[j, c] = x[ku]
                rec[j, c + 1] = x[ku + 1]
                rec[j, c + 2] = ev[k, 3]
                rec[j, c + 3] = ev[k, 1]
                rec[j, c + 4] = ev[k, 2]
                rec[j, c + 5] = ls[k, 0]
                rec[j, c + 6] = ls[k, 1]
                c += 7
                ss[k, SS_VDS_REC] = _nv(x, sD[k]) - _nv(x, sS[k])
                ss[k, SS_LNI_REC] = np.log(abs(ev[k, 1]) + i_floor)
            for e in range(nc):
                # capacitor current a -> b of the accepted step (companion current of the integration formula)
                rec[j, c] = cI[e]
                c += 1
            si[SI_NREC] = j + 1
            sf[SF_TREC] = t
        for k in range(ns):
            for jj in range(N_EV):
                ev0[k, jj] = ev[k, jj]
        # ---- next step size ------------------------------------------------------------------
        if err < 1e-6:
            fac = 2.5
        else:
            fac = min(2.5, max(0.3, 0.8 / err))
        hn = h * fac
        if hit:
            hn = max(hn, min(sf[SF_HNEXT], dt_max))
        sf[SF_HNEXT] = max(hn, dt_min)
    si[SI_STATUS] = status
    return status


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def init_state(x, ci, cf, rA, rB, rG, cA, cB, cC, vA, vB, vW, iA, iB, iW, sD, sG, sS, sW,
               wt, wv, woff, P, Pbase, na, vbi, rg, fg, table, ss, part, sens, ls, cv, cI, t0, win, cmp, basic):
    """Element state, partials, sensitivities and latch flags at the (DC) initial point
    (latched = physically on the LRS, u >= u_j, independent of the current threshold)."""
    nn = ci[CI_NN]
    ns = ci[CI_NS]
    nv = ci[CI_NV]
    n = x.shape[0]
    nc = cA.shape[0]
    ev = np.zeros((ns, N_EV))
    tmp = np.zeros(N_EV)
    J = np.zeros((n, n))
    f = np.zeros(n)
    vval = np.zeros(vA.shape[0])
    ival = np.zeros(iA.shape[0])
    cGeq = np.zeros(nc)
    cIeq = np.zeros(nc)
    qc = np.zeros(ns)
    sources_at(t0, vW, iW, sW, wt, wv, woff, vval, ival, P, Pbase)
    if ci[CI_STOCH] == 1 and ci[CI_LSMODE] > 0:
        for k in range(ns):
            if win[k, W_LSMODE] > 0.5:
                ix = int(win[k, W_LSIDX])
                P[k, ix] = Pbase[k, ix] + ls[k, 0]
                P[k, 10] = Pbase[k, 10] + ls[k, 1]
    if not eval_all(x, ci, sD, sG, sS, P, na, vbi, rg, fg, table, ev):
        return False
    if not fd_partials(x, ci, P, na, vbi, rg, fg, table, ev, part, tmp):
        return False
    for e in range(nc):
        cv[e] = _nv(x, cA[e]) - _nv(x, cB[e])
        cI[e] = 0.0
    for k in range(ns):
        ku = nn - 1 + nv + 2 * k
        ss[k, SS_QN] = ev[k, 3]
        ss[k, SS_FN] = ev[k, 2]
        ss[k, SS_IN] = ev[k, 1]
        ss[k, SS_UNIT] = ev[k, 4]
        ss[k, SS_G] = ev[k, 5]
        ss[k, SS_L] = ev[k, 6]
        ss[k, SS_R] = x[ku + 1]
        ss[k, SS_PHYS] = 1.0 if x[ku] >= win[k, W_UJ] else 0.0
        ss[k, SS_LAT] = ss[k, SS_PHYS]
        ss[k, SS_PEND] = 0.0
        ss[k, SS_VDS_REC] = _nv(x, sD[k]) - _nv(x, sS[k])
        ss[k, SS_LNI_REC] = np.log(abs(ev[k, 1]) + cf[CF_IFLOOR])
        qc[k] = ev[k, 3]
    sensitivities(x, ci, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                  sD, sG, sS, ev, part, J, f, qc, sens, ss, vW, cmp, basic)
    cmp_update_state(x, cmp)
    return True
