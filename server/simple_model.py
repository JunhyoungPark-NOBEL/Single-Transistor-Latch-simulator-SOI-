"""Fast, paper-inspired first-order body-charge STL model.

This evaluator deliberately never imports/calls the distributed SRH solver.  It
implements the accepted manuscript's forward STL closure with bias-constant,
geometry-dependent diffusion beta.  The default numbers are starting values, not
an independently calibrated replacement for the distributed model.

Coordinates: u is the effective source-side BJT/contact voltage, r the internal
collector voltage, w=u+ID*RLRS the lumped charge reservoir potential.  The decaying
charge is Q=CB*(w-bias), not CB*u.  Attaching an external B terminal at u is a
separate lumped extension; the paper does not specify external-contact transport.

The Miller expression has a strict r<VBR domain.  For -50 mV <= r <= 0 we use
M=1 and zero BTBT only as a startup/small-ringing numerical continuation of the
forward model.  IS*exp(u/VT) is retained exactly: this continuation is NOT a
zero-bias-equilibrium-valid or reverse-operation transistor model.

BTBT uses the existing abrupt drain depletion field integral (16-point Gauss
quadrature) and gate-edge volume, with effective collector voltage r in GIDL.
This last approximation avoids an implicit ID-dependent tunnelling solve.  The
first-order lifetime/surface split and passive gate-capacitance partition are
explicit, uncalibrated geometry closures.  No channel, distributed SRH, or extra
stored ambipolar charge is silently inherited from the full model.
"""
from __future__ import annotations

import numpy as np
from numba import njit

QE = 1.602176634e-19
VT = 1.380649e-23 * 300.0 / QE
EPS_SI = 11.7 * 8.8541878128e-12 / 100.0  # F/cm
NI_CM3 = 1e10
BB_A = 4e14
BB_B = 19e6
# Existing reference calibration doping (server.params.NA_CM3). Kept local so
# this numerical module does not form an import cycle with parameter packing.
NREF = 2.295773162796593e17
LREF, WREF, TREF, EOTREF, BOXREF = 500.0, 200.0, 50.0, 14.1, 140.0
_x, _w = np.polynomial.legendre.leggauss(16)
GAUSS_X = np.ascontiguousarray(0.5 * (_x + 1.0))
GAUSS_W = np.ascontiguousarray(0.5 * _w)


@njit(cache=True)
def is_simple(p):
    return len(p) >= 48 and p[34] == -1.0


@njit(cache=True)
def effective_parameters(p):
    """Return beta,tau,CB,R,IS,VBR,eta,Cg,Cbg,Cs,bias in SI units.

    Diffusion beta~1/(Nbody*L); IS~W*Tsi/(Nbody*L), so IS/beta
    has the expected emitter-area scaling with no artificial L or Nbody factor.
    """
    lr, wr, tr = p[26] / LREF, p[27] / WREF, p[28] / TREF
    nr = NREF / p[31]
    beta = p[36] * nr / lr
    tau = p[37] / ((1.0 - p[47]) + p[47] / tr)
    area = lr * wr
    cg = p[38] * p[43] * area * EOTREF / p[29]
    cbg = p[38] * p[44] * area * (BOXREF + TREF / 3.0) / (p[30] + p[28] / 3.0)
    cs = p[38] * (1.0 - p[43] - p[44]) * area
    cb = cg + cbg + cs
    resistance = p[39] * lr / (wr * tr)
    saturation = p[40] * wr * tr * nr / lr
    bias = (cg * (p[11] - p[45]) + cbg * (p[32] - p[45])) / cb
    return beta, tau, cb, resistance, saturation, p[41], p[42], cg, cbg, cs, bias


@njit(cache=True)
def btbt_currents(r, p):
    """(lateral BTBT, GIDL) in A; no field-table or SRH construction.

    The lateral integrand and abrupt field match the full model's Field.btbt.
    Gate field length is 3*EOT and overlap is 5 nm; the same n+ drain tunnelling
    depth as the full model is used, capped by physical Tsi.
    """
    if r <= 0.0 or p[46] == 0.0:
        return 0.0, 0.0
    na = p[31]
    vbi = VT * np.log(1e20 * na / NI_CM3**2)
    wd = np.sqrt(2.0 * EPS_SI * (vbi + r) / (QE * na))
    peak = 2.0 * (vbi + r) / wd
    integral = 0.0
    for i in range(len(GAUSS_X)):
        field = peak * GAUSS_X[i]
        integral += GAUSS_W[i] * BB_A * field**2.5 * np.exp(-BB_B / field)
    balance = -np.expm1(-r / VT)
    lateral = QE * (p[27] * p[28] * 1e-14) * wd * integral * balance
    field = max((r - p[11] - 0.3 - 1.12 + p[9]) / (3.0 * p[29] * 1e-7), 0.0)
    depth = min(np.sqrt(2.0 * EPS_SI * 1.12 / (QE * 7e19)), p[28] * 1e-7)
    volume = p[27] * 1e-7 * 5e-7 * depth
    gidl = QE * volume * BB_A * field**2.5 * np.exp(-BB_B / max(field, 1.0)) * balance
    return p[46] * lateral, p[46] * gidl


@njit(cache=True)
def components(u, r, p, na=0.0, vbi=0.0, rg=None, fg=None, table=None):
    """Return the shared 19-column component contract without SRH transport.

    The shared bulk-loss slot z[5] contains the *entire* first-order Q/tau
    term; z[7] is zero (no second junction-SRH term).  z[13] is the complete
    decaying charge, and z[17]=ID*RLRS is the paper's ohmic potential drop.
    """
    z = np.full(19, np.nan)
    if not is_simple(p) or not np.isfinite(u) or not np.isfinite(r):
        return z
    beta, tau, cb, resistance, saturation, vbr, eta, cg, cbg, cs, bias = effective_parameters(p)
    if (r < -0.05 or r >= vbr or beta <= 0 or tau <= 0 or cb <= 0
            or saturation <= 0 or resistance < 0 or cs < 0 or u / VT > 700):
        return z
    mult = 1.0 if r <= 0 else 1.0 / (1.0 - (r / vbr)**eta)
    seed = saturation * np.exp(u / VT)
    bbj, gidl = btbt_currents(r, p)
    photo = p[13]
    drain = mult * seed + bbj + gidl + photo
    drop = drain * resistance
    reservoir = u + drop
    charge = cb * (reservoir - bias)
    rec = charge / tau
    diffusion = seed / beta
    generation = (mult - 1.0) * seed + bbj + gidl + photo
    z[:] = 0.0
    z[0] = r + drop
    z[1] = drain
    z[2] = generation - diffusion - rec
    z[3] = seed
    z[4] = seed + diffusion
    z[5] = rec
    z[6] = diffusion
    z[8] = bbj
    z[9] = gidl
    z[11] = p[26] * 1e-7
    z[12] = resistance
    z[13] = charge
    # Geometric depletion diagnostics are not part of this lumped charge law.
    z[17] = drop
    z[18] = photo
    return z


@njit(cache=True)
def evaluate(u, r, p, out):
    """Fill the circuit's 11 outputs; out[8] is reservoir w, not contact u."""
    z = components(u, r, p)
    if not (np.isfinite(z[0]) and np.isfinite(z[1]) and np.isfinite(z[2]) and np.isfinite(z[13])):
        out[:] = np.nan
        return False
    beta, tau, cb, resistance, saturation, vbr, eta, cg, cbg, cs, bias = effective_parameters(p)
    w = u + z[17]
    out[0] = z[0]
    out[1] = z[1]
    out[2] = z[2]
    out[3] = z[13]
    out[4] = z[8] + z[9] + z[18]
    out[5] = z[1] - z[3]
    out[6] = z[5] + z[6]
    out[7] = 2.0 if r < 0.0 else 0.0
    out[8] = w
    out[9] = cg * (p[11] - w)
    out[10] = cbg * (p[32] - w)
    for i in range(len(out)):
        if not np.isfinite(out[i]):
            return False
    return True


@njit(cache=True)
def branch_grid(p, ug):
    """Find F(u,r)=0 at each u by monotone bracketing below Miller breakdown.

    The branch grid is an equilibrium sampling operation only. Transient circuit
    integration evaluates the continuous evaluator directly, not this table.
    """
    out = np.empty((len(ug), 21))
    count = 0
    vbr = p[41]
    for u in ug:
        lo = 0.0
        hi = vbr * (1.0 - 1e-10)
        a, b = components(u, lo, p), components(u, hi, p)
        if not np.isfinite(a[2]) or not np.isfinite(b[2]) or a[2] > 0 or b[2] < 0:
            continue
        # Linear bracket avoids the former 1e-100..VBR logarithmic search and
        # resolves near-breakdown HRS roots to a voltage-based tolerance.
        for _ in range(52):
            mid = 0.5 * (lo + hi)
            z = components(u, mid, p)
            if not np.isfinite(z[2]) or z[2] > 0:
                hi = mid
            else:
                lo = mid
        r = 0.5 * (lo + hi)
        z = components(u, r, p)
        if not np.isfinite(z[2]) or abs(z[2]) > 1e-7 * max(abs(z[1]), 1e-20):
            continue
        out[count, :17] = z[:17]
        out[count, 17] = u
        out[count, 18] = r
        out[count, 19] = np.sqrt(2.0 * QE * p[31] * (VT * np.log(1e20 * p[31] / NI_CM3**2) + r) / EPS_SI)
        out[count, 20] = z[17]
        count += 1
    return out[:count]


def branch(p, ug):
    """Equilibrium grid including negative body voltages when bias requires it.

    Existing callers pass a nonnegative full-model grid.  Preserve its number of
    broad samples but extend toward the actual bias-dependent HRS origin.  A
    one/few-point call (used to refine folds) always evaluates the requested u.
    """
    p = np.asarray(p, dtype=float)
    ug = np.asarray(ug, dtype=float)
    if len(ug) <= 3:
        return branch_grid(p, ug)
    beta, tau, cb, resistance, saturation, vbr, eta, cg, cbg, cs, bias = effective_parameters(p)
    # At r=0 F is strictly decreasing in u; isolate its zero, rather than
    # manufacturing a zero-current point incompatible with the forward law.
    lo, hi = min(-1.0, bias - 1.0), max(1.2, bias + 0.1)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if components(mid, 0.0, p)[2] > 0:
            lo = mid
        else:
            hi = mid
    u0 = 0.5 * (lo + hi)
    broad = np.linspace(u0 + 1e-10, max(1.12, u0 + 0.2), max(201, len(ug)))
    fine = u0 + np.geomspace(1e-12, 0.02, 101)
    states = np.unique(np.r_[ug[ug >= u0], broad, fine])
    return branch_grid(p, states)
