"""Fast, first-order body-charge STL model of the published analytical framework.

Reference (the "Simple Model" implements its equations (1)-(4), (7) and
Table I): J.-H. Park, H.-B. Noh, S.-W. Lee, S.-Y. Yun, and Y.-K. Choi,
"Analytical Model for Single Transistor Latch in MOSFETs," IEEE Electron
Device Lett., 2026, doi: 10.1109/LED.2026.3737574.  The Detailed Model is the
calibrated "updated accuracy" model that extends this framework.

This evaluator deliberately never imports/calls the distributed SRH solver.
It implements the paper's forward STL closure with a bias-constant,
geometry-dependent diffusion beta (the paper's bias-dependent beta, eq. (8),
is not used - owner decision).  The default numbers are starting values, not
an independently calibrated replacement for the distributed model.

Coordinates: u is the effective source-side BJT/contact voltage, r the internal
collector voltage, w=u+ID*RLRS the lumped charge reservoir potential.  The decaying
charge is Q=CB*(w-bias), not CB*u.  Attaching an external B terminal at u is a
separate lumped extension; the paper does not specify external-contact transport.

Accumulation (paper, eq. (2)): a gate below the flat-band voltage V_FB no longer
couples to the body because the accumulated hole layer screens it, so each
gate's coupling term uses max(V - V_FB, 0).  Making V_G more negative than V_FB
therefore stops lowering V_bias while it keeps raising the GIDL field; this is
what turns V_LU(V_G) over into the bell shape of the paper's Fig. 5(a).

BTBT follows the paper's reference script: junction (lateral) BTBT evaluates
Kane's generation rate at the peak field of the abrupt drain junction over the
depletion volume W*Tsi*Wd, with the junction built-in voltage lowered by
gamma_G*(V_G-V_FB) above flat band; GIDL uses the vertical field
(r - V_G + 1.2 - Eg)/(3*EOT) over the volume gidl_volume_scale*W*5nm*Wt.  The
effective collector voltage r replaces the external V_D in GIDL, which avoids an
implicit ID-dependent tunnelling solve.

The Miller expression has a strict r<VBR domain.  For -50 mV <= r <= 0 we use
M=1 and zero BTBT only as a startup/small-ringing numerical continuation of the
forward model.  IS*exp(u/VT) is retained exactly: this continuation is NOT a
zero-bias-equilibrium-valid or reverse-operation transistor model.

The first-order lifetime/surface split and passive gate-capacitance partition are
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
BB_A = 4e14          # Kane prefactor, cm^-0.5 V^-2.5 s^-1 (paper Table I)
BB_B = 19e6          # Kane exponent, V/cm (19 MV/cm, paper Table I)
EG = 1.12            # Si band gap, eV
VFB_GIDL = -1.2      # gate/n+ drain flat-band voltage of the GIDL field (reference script)
LOV_CM = 5e-7        # gate-drain overlap, 5 nm
ND_CM3 = 7e19        # n+ drain doping bounding the GIDL tunnelling depth
# Existing reference calibration doping (server.params.NA_CM3). Kept local so
# this numerical module does not form an import cycle with parameter packing.
NREF = 2.295773162796593e17
LREF, WREF, TREF, EOTREF, BOXREF = 500.0, 200.0, 50.0, 14.1, 140.0


@njit(cache=True)
def is_simple(p):
    return len(p) >= 49 and p[34] == -1.0


@njit(cache=True)
def gate_voltages(p):
    """(V_G,eff, V_BG,eff): each gate clamped at V_FB once it is in accumulation.

    Below flat band the accumulated hole layer screens the gate-to-body
    coupling (paper, eq. (2) discussion), so the body sees max(V, V_FB).
    """
    return max(p[11], p[45]), max(p[32], p[45])


@njit(cache=True)
def effective_parameters(p):
    """Return beta,tau,CB,R,IS,VBR,eta,Cg,Cbg,Cs,bias in SI units.

    Diffusion beta~1/(Nbody*L); IS~W*Tsi/(Nbody*L), so IS/beta
    has the expected emitter-area scaling with no artificial L or Nbody factor.
    bias is the paper's V_BS,bias = gamma_G*(V_G-V_FB) + gamma_BG*(V_BG-V_FB) with
    the accumulation clamp of gate_voltages.
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
    vg, vbg = gate_voltages(p)
    bias = (cg * (vg - p[45]) + cbg * (vbg - p[45])) / cb
    return beta, tau, cb, resistance, saturation, p[41], p[42], cg, cbg, cs, bias


@njit(cache=True)
def junction_vbi(p):
    """Built-in voltage of the abrupt n+ drain junction seen by lateral BTBT.

    The paper's script lowers it by gamma_G*(V_G-V_FB) while the gate is above
    flat band (depletion/inversion) and leaves it unchanged in accumulation.
    """
    vg, _ = gate_voltages(p)
    return VT * np.log(1e20 * p[31] / NI_CM3**2) - p[43] * (vg - p[45])


@njit(cache=True)
def btbt_currents(r, p):
    """(lateral junction BTBT, GIDL) in A; no field-table or SRH construction.

    Both use Kane's rate G(E)=A*E^2.5*exp(-B/E) evaluated at one field, times a
    generation volume, as in the paper's reference script (Table I integral
    collapsed to peak field x volume):
      lateral: E = 2(Vbi+r)/Wd of the abrupt junction, volume W*Tsi*Wd;
      GIDL:    E = (r - V_G + 1.2 - Eg)/(3*EOT), volume gidl_volume_scale*W*5nm*Wt,
               Wt = min(n+ tunnelling depth, Tsi).
    The factor 1-exp(-r/VT) keeps both continuous at r -> 0.
    """
    if r <= 0.0 or p[46] == 0.0:
        return 0.0, 0.0
    balance = -np.expm1(-r / VT)
    lateral = 0.0
    vj = junction_vbi(p) + r
    if vj > 0.0:
        wd = np.sqrt(2.0 * EPS_SI * vj / (QE * p[31]))
        peak = 2.0 * vj / wd
        lateral = QE * BB_A * peak**2.5 * np.exp(-BB_B / peak) * (p[27] * p[28] * 1e-14) * wd
    gidl = 0.0
    field = (r - p[11] - VFB_GIDL - EG) / (3.0 * p[29] * 1e-7)
    if field > 0.0 and p[48] > 0.0:
        depth = min(np.sqrt(2.0 * EPS_SI * EG / (QE * ND_CM3)), p[28] * 1e-7)
        volume = p[48] * p[27] * 1e-7 * LOV_CM * depth
        gidl = QE * volume * BB_A * field**2.5 * np.exp(-BB_B / field)
    return p[46] * lateral * balance, p[46] * gidl * balance


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
    vg, vbg = gate_voltages(p)
    out[0] = z[0]
    out[1] = z[1]
    out[2] = z[2]
    out[3] = z[13]
    out[4] = z[8] + z[9] + z[18]
    out[5] = z[1] - z[3]
    out[6] = z[5] + z[6]
    out[7] = 2.0 if r < 0.0 else 0.0
    out[8] = w
    # Gate charges use the same clamped voltages as bias, so the reservoir
    # charge, Q_G, Q_BG and C_S*w stay conserved; in accumulation the gate's
    # extra displacement charge sits on the screening hole layer, outside the
    # lumped reservoir.
    out[9] = cg * (vg - w)
    out[10] = cbg * (vbg - w)
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
