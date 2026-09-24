"""Generic small-netlist MNA transient kernel with STL elements (numba).

Unknown vector x = [node voltages v_1..v_{N-1} (node 0 = ground), V-source branch currents,
(u_k, r_k) for every STL element k].

Residuals:
  KCL at every non-ground node (A):  sum of currents leaving the node = 0
      R: (v_a - v_b)/R; C: companion (BE: C/h dv, TRAP: 2C/h dv - i_n); V: branch current;
      I: source value; STL: I_D(u,r) leaves the drain node and enters the source node.
  V source:  v_a - v_b - V(t) = 0
  STL, E1:   V_D(u,r) - (v_d - v_s) = 0                               (V)
  STL, E2:   [Q(u,r;V_GS) - Qc - th*h*F(u,r)] / C_ox = 0               (V)
      deterministic BE: Qc = Q_n, th = 1;  TRAP: Qc = Q_n + h/2 F_n, th = 1/2
      stochastic (explicit tau-leap, noise resolved): Qc = Q_n + dQ_events, th = 0
      stochastic (drift-implicit, fast relaxation):   Qc = Q_n + dQ_events - h F_n, th = 1
  (DC initialisation: E2 replaced by u - u_fix = 0, then pseudo-transient BE.)

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

GMIN = 1e-18          # S, node-to-ground conductance (keeps floating nodes regular)
DU_LIM = 0.05         # V, max |du| per Newton iteration
DR_LIM = 1.0          # V, max |dr| per Newton iteration
FD_U = 1e-6           # V, finite-difference step in u
FD_R = 1e-6           # V, finite-difference step in r

# ---- integer config (ci) -----------------------------------------------------------------
CI_NN, CI_NV, CI_NR, CI_NC, CI_NI, CI_NS, CI_METHOD, CI_STOCH, CI_CARRIER, CI_MAXSTEPS, \
    CI_MAINW, CI_LSMODE, CI_LSIDX, CI_CHUNK, CI_MAIN_STL = range(15)
N_CI = 15
# ---- float config (cf) -------------------------------------------------------------------
CF_TEND, CF_DTMIN, CF_DTMAX, CF_DUMAX, CF_DLNIMAX, CF_DVMAX, CF_LTEU, CF_TAUFRAC, CF_NEVMAX, \
    CF_HNOISEMIN, CF_GAUSS, CF_ITH, CF_IFLOOR, CF_DTREC, CF_DVREC, CF_DLNIREC, CF_LSSIG, CF_LSTAU, \
    CF_LSESIG, CF_LSETAU, CF_HINIT, CF_NEWTOL = range(22)
N_CF = 22
# ---- float state (sf) --------------------------------------------------------------------
SF_T, SF_HNEXT, SF_HPREV, SF_TREC, SF_TUNRES, SF_MINU, SF_MINR, SF_TNEGU, SF_TNEGR, SF_TSTOP = range(10)
N_SF = 10
# ---- int state (si) ----------------------------------------------------------------------
SI_STEPS, SI_REJ, SI_NEWT, SI_BP, SI_NEV, SI_NREC, SI_NSAMP, SI_STATUS, SI_UNRES, SI_TRAPBE, \
    SI_REFRESH, SI_HAVEPREV, SI_FAILNEWTON, SI_CHUNKSTEPS = range(14)
N_SI = 14
# status codes
ST_DONE, ST_CHUNK, ST_FAIL, ST_MAXSTEPS, ST_BUFFER = 0, 1, 2, 3, 4
# per-STL state columns (ss)
SS_QN, SS_FN, SS_IN, SS_TAU, SS_UNIT, SS_G, SS_L, SS_R, SS_LAT, SS_DQ, SS_LNI_REC, SS_VDS_REC = range(12)
N_SS = 12
# partial derivative columns (part)
P_VU, P_VR, P_IU, P_IR, P_FU, P_FR, P_QU, P_QR = range(8)
# event columns
EV_KIND, EV_STL, EV_T, EV_VDS, EV_VSRC, EV_I = range(6)
N_EVC = 6


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
    return True


@njit(cache=True)
def assemble(x, ci, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
             sD, sG, sS, ev, part, emode, qc, th, h, J, f):
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
        f[iv] = _nv(x, a) - _nv(x, b) - vval[e]
    for e in range(iA.shape[0]):
        a = iA[e]
        b = iB[e]
        if a > 0:
            f[a - 1] += ival[e]
        if b > 0:
            f[b - 1] -= ival[e]
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
            f[kr] = (ev[k, 3] - qc[k] - th * h * ev[k, 2]) / COX
            J[kr, ku] = (part[k, P_QU] - th * h * part[k, P_FU]) / COX
            J[kr, kr] = (part[k, P_QR] - th * h * part[k, P_FR]) / COX
        else:
            f[kr] = x[ku] - qc[k]
            J[kr, ku] = 1.0


@njit(cache=True)
def newton(x, ci, cf, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
           sD, sG, sS, P, na, vbi, rg, fg, table, ev, part, tmp, emode, qc, th, h,
           refresh, maxit, J, f, xt, evt):
    """Solve the nonlinear system at one time point in place (x, ev, part).
    Returns (converged, iterations)."""
    nn = ci[CI_NN]
    nv = ci[CI_NV]
    ns = ci[CI_NS]
    n = x.shape[0]
    tol = cf[CF_NEWTOL]
    if not eval_all(x, ci, sD, sG, sS, P, na, vbi, rg, fg, table, ev):
        return False, 0
    last_norm = 1e300
    for it in range(maxit):
        if refresh or it >= 2:
            if not fd_partials(x, ci, P, na, vbi, rg, fg, table, ev, part, tmp):
                return False, it + 1
        assemble(x, ci, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                 sD, sG, sS, ev, part, emode, qc, th, h, J, f)
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
        alpha = 1.0
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
        vval[e] = wave_value(vW[e], t, wt, wv, woff)
    for e in range(iW.shape[0]):
        ival[e] = wave_value(iW[e], t, wt, wv, woff)
    for k in range(sW.shape[0]):
        if sW[k] >= 0:
            P[k, 13] = wave_value(sW[k], t, wt, wv, woff)
        else:
            P[k, 13] = Pbase[k, 13]


@njit(cache=True)
def sensitivities(x, ci, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                  sD, sG, sS, ev, part, J, f, qc0, sens, ss):
    """dx/dQ_k (charge fixed, theta = 0) and the local relaxation time
    tau_k = 1/|dF_k/dQ_k| along the circuit constraints."""
    nn = ci[CI_NN]
    nv = ci[CI_NV]
    ns = ci[CI_NS]
    n = x.shape[0]
    assemble(x, ci, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
             sD, sG, sS, ev, part, 0, qc0, 0.0, 0.0, J, f)
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
        else:
            ss[k, SS_TAU] = 1e30


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
          wt, wv, woff, P, Pbase, na, vbi, rg, fg, table, ev, part, t0):
    """DC operating point at t0: u fixed at 0 (empty body) -> pseudo-transient BE continuation
    of the charge equation (finds the low-current state reachable from an empty body)."""
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
    # phase 1: u = 0, capacitors open (h = inf)
    for k in range(ns):
        qc[k] = 0.0
    ok, it = newton(x, ci, cf, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                    sD, sG, sS, P, na, vbi, rg, fg, table, ev, part, tmp, 1, qc, 0.0, 0.0,
                    True, 60, J, f, xt, evt)
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
        cap_companion(ci, 0, h, cA, cB, cC, cv, cI, cGeq, cIeq)
        ok, it = newton(x, ci, cf, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                        sD, sG, sS, P, na, vbi, rg, fg, table, ev, part, tmp, 0, qc, 1.0, h,
                        True, 40, J, f, xt, evt)
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
def run_chunk(x, xp, ci, cf, rA, rB, rG, cA, cB, cC, vA, vB, vW, iA, iB, iW, sD, sG, sS, sW,
              wt, wv, woff, bp, samp, P, Pbase, na, vbi, rg, fg, table, rv, pmf,
              ss, part, sens, ls, cv, cI, sf, si, rec, evb, sbuf):
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
    lsmode = ci[CI_LSMODE] if stoch else 0
    lsidx = ci[CI_LSIDX]
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
    si[SI_CHUNKSTEPS] = 0
    t = sf[SF_T]
    # current element outputs at the accepted state
    for k in range(ns):
        P[k, 11] = _nv(x, sG[k]) - _nv(x, sS[k])
    sources_at(t, vW, iW, sW, wt, wv, woff, vval, ival, P, Pbase)
    for k in range(ns):
        if lsmode > 0:
            P[k, lsidx] = Pbase[k, lsidx] + ls[k, 0]
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
        resolved = False
        if carrier:
            taumin = 1e300
            rate_max = 0.0
            hdrift = 1e300
            for k in range(ns):
                taumin = min(taumin, ss[k, SS_TAU])
                rate_max = max(rate_max, total_event_rate(ss[k, SS_UNIT], ss[k, SS_G], ss[k, SS_L],
                                                         ss[k, SS_R], rv, pmf, pk))
                fk = abs(ss[k, SS_FN])
                if fk > 0.0:
                    hdrift = min(hdrift, du_max * abs(part[k, P_QU]) / fk)
            if tau_frac * taumin >= h_noise_min:
                resolved = True
                h = min(h, tau_frac * taumin, hdrift)
                if rate_max > 0.0:
                    h = min(h, nev_max / rate_max)
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
        th = 1.0
        use_trap = False
        for att in range(40):
            t1 = t + h
            sources_at(t1, vW, iW, sW, wt, wv, woff, vval, ival, P, Pbase)
            if lsmode == 2:
                for k in range(ns):
                    a = np.exp(-h / cf[CF_LSTAU]) if cf[CF_LSTAU] > 0 else 0.0
                    ls_new[k, 0] = ls[k, 0] * a + cf[CF_LSSIG] * np.sqrt(max(1.0 - a * a, 0.0)) * np.random.normal()
                    a2 = np.exp(-h / cf[CF_LSETAU]) if cf[CF_LSETAU] > 0 else 0.0
                    ls_new[k, 1] = ls[k, 1] * a2 + cf[CF_LSESIG] * np.sqrt(max(1.0 - a2 * a2, 0.0)) * np.random.normal()
            else:
                for k in range(ns):
                    ls_new[k, 0] = ls[k, 0]
                    ls_new[k, 1] = ls[k, 1]
            if lsmode > 0:
                for k in range(ns):
                    P[k, lsidx] = Pbase[k, lsidx] + ls_new[k, 0]
                    P[k, 10] = Pbase[k, 10] + ls_new[k, 1]
            # charge-equation mode for this step
            use_trap = False
            if carrier:
                for k in range(ns):
                    dq[k] = draw_dq(h, ss[k, SS_UNIT], ss[k, SS_G], ss[k, SS_L], ss[k, SS_R], rv, pmf, pk, gth)
                if resolved:
                    th = 0.0
                    for k in range(ns):
                        qc[k] = ss[k, SS_QN] + dq[k]
                else:
                    th = 1.0
                    for k in range(ns):
                        qc[k] = ss[k, SS_QN] + dq[k] - h * ss[k, SS_FN]
                cap_companion(ci, 0, h, cA, cB, cC, cv, cI, cGeq, cIeq)
            else:
                taumin = 1e300
                for k in range(ns):
                    taumin = min(taumin, ss[k, SS_TAU])
                if method == 1 and h < 2.0 * taumin:
                    use_trap = True
                    th = 0.5
                    for k in range(ns):
                        qc[k] = ss[k, SS_QN] + 0.5 * h * ss[k, SS_FN]
                    cap_companion(ci, 1, h, cA, cB, cC, cv, cI, cGeq, cIeq)
                else:
                    th = 1.0
                    for k in range(ns):
                        qc[k] = ss[k, SS_QN]
                    cap_companion(ci, 0, h, cA, cB, cC, cv, cI, cGeq, cIeq)
            # predictor
            for i in range(n):
                x0[i] = x[i]
            if carrier and resolved:
                for k in range(ns):
                    for i in range(n):
                        x0[i] += sens[k, i] * (qc[k] - ss[k, SS_QN] + th * h * ss[k, SS_FN])
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
            ok, iters = newton(x0, ci, cf, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                               sD, sG, sS, P, na, vbi, rg, fg, table, ev, part, tmp, 0, qc, th, h,
                               refresh, 14, J, f, xt, evt)
            si[SI_NEWT] += iters
            if not ok:
                si[SI_REJ] += 1
                si[SI_FAILNEWTON] += 1
                si[SI_REFRESH] = 1
                h *= 0.25
                hit = False
                if h < dt_min or t + h == t:
                    break
                continue
            # ---- error control ---------------------------------------------------------
            du = 0.0
            dl = 0.0
            lte = 0.0
            for k in range(ns):
                ku = nn - 1 + nv + 2 * k
                du = max(du, abs(x0[ku] - x[ku]))
                dl = max(dl, abs(np.log(abs(ev[k, 1]) + i_floor) - np.log(abs(ev0[k, 1]) + i_floor)))
                qu = abs(part[k, P_QU])
                if qu > 0:
                    lte = max(lte, 0.5 * h * abs(ev[k, 2] - ev0[k, 2]) / qu)
            dvm = 0.0
            for i in range(nn - 1):
                dvm = max(dvm, abs(x0[i] - x[i]))
            if carrier and resolved:
                err = dvm / dv_max
                hard = du > 20.0 * du_max
                if hard:
                    err = 10.0
            else:
                err = max(du / du_max, dl / dlni_max, dvm / dv_max)
                if not carrier or not resolved:
                    err = max(err, lte / lte_u)
            if err > 1.5 and h > dt_min * 1.01:
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
        si[SI_CHUNKSTEPS] += 1
        if method == 1 and not carrier and not use_trap:
            si[SI_TRAPBE] += 1
        if carrier and not resolved:
            si[SI_UNRES] += 1
            sf[SF_TUNRES] += h
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
        if carrier and resolved:
            si[SI_REFRESH] = 1 if iters > 2 else 0
        # per-element state
        for k in range(ns):
            ku = nn - 1 + nv + 2 * k
            ss[k, SS_QN] = ev[k, 3]
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
                      sD, sG, sS, ev, part, J, f, qc, sens, ss)
        # ---- event detection (threshold crossings of I_D) ----------------------------
        evflag = False
        for k in range(ns):
            i0 = ev0[k, 1]
            i1 = ev[k, 1]
            lat = ss[k, SS_LAT]
            kind = 0
            if lat < 0.5 and i0 < i_th and i1 >= i_th:
                kind = 1
            elif lat > 0.5 and i0 >= i_th and i1 < i_th:
                kind = 2
            if kind > 0:
                if i0 > 0 and i1 > 0:
                    a = (np.log(i_th) - np.log(i0)) / (np.log(i1) - np.log(i0))
                else:
                    a = (i_th - i0) / (i1 - i0)
                a = min(max(a, 0.0), 1.0)
                te = t_old + a * h
                ne = si[SI_NEV]
                vds0 = _nv(xp, sD[k]) - _nv(xp, sS[k])
                vds1 = _nv(x, sD[k]) - _nv(x, sS[k])
                evb[ne, EV_KIND] = kind
                evb[ne, EV_STL] = k
                evb[ne, EV_T] = te
                evb[ne, EV_VDS] = vds0 + a * (vds1 - vds0)
                evb[ne, EV_VSRC] = wave_value(mainw, te, wt, wv, woff) if mainw >= 0 else np.nan
                evb[ne, EV_I] = i_th
                si[SI_NEV] = ne + 1
                ss[k, SS_LAT] = 1.0 if kind == 1 else 0.0
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
                sbuf[j, 1 + 2 * k] = ev0[k, 1] + a * (ev[k, 1] - ev0[k, 1])
                vds0 = _nv(xp, sD[k]) - _nv(xp, sS[k])
                vds1 = _nv(x, sD[k]) - _nv(x, sS[k])
                sbuf[j, 2 + 2 * k] = vds0 + a * (vds1 - vds0)
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
               wt, wv, woff, P, Pbase, na, vbi, rg, fg, table, ss, part, sens, ls, cv, cI, t0):
    """Element state, partials, sensitivities and latch flags at the (DC) initial point."""
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
            P[k, ci[CI_LSIDX]] = Pbase[k, ci[CI_LSIDX]] + ls[k, 0]
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
        ss[k, SS_LAT] = 1.0 if ev[k, 1] >= cf[CF_ITH] else 0.0
        ss[k, SS_VDS_REC] = _nv(x, sD[k]) - _nv(x, sS[k])
        ss[k, SS_LNI_REC] = np.log(abs(ev[k, 1]) + cf[CF_IFLOOR])
        qc[k] = ev[k, 3]
    sensitivities(x, ci, rA, rB, rG, cA, cB, cGeq, cIeq, vA, vB, vval, iA, iB, ival,
                  sD, sG, sS, ev, part, J, f, qc, sens, ss)
    return True
