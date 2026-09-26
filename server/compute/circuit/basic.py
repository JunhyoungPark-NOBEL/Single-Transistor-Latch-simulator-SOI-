"""Small educational semiconductor models; memoryless, 300 K, no parasitic charge.

MOS: symmetric, smooth square-law model with independently configurable subthreshold
swing. D: Shockley. BJT: reciprocal Ebers–Moll with transport saturation current Is.
These elements share the nonlinear MNA solve with STL cells, R/C and sources.
"""
from __future__ import annotations

import numpy as np
from numba import njit

WIDTH = 12
PINS = {"MOS": ("d", "g", "s"), "D": ("a", "k"), "BJT": ("c", "b", "e")}
VT = 0.025852


def pack(kind: str, nodes, model: dict) -> np.ndarray:
    """kind, n0,n1,n2, polarity, model parameters (fixed-width numba row)."""
    row = np.zeros(WIDTH)
    row[0] = {"MOS": 1, "D": 2, "BJT": 3}[kind]
    row[1:1 + len(nodes)] = nodes
    if kind == "MOS":
        row[4:10] = [1 if model["polarity"] == "nmos" else -1,
                     model["k_uA_V2"] * 1e-6 * model["W_um"] / model["L_um"],
                     abs(model["Vth_V"]), model["SS_mV_dec"] * 1e-3 / np.log(10),
                     model["lambda_per_V"], 0]
    elif kind == "D":
        row[4:7] = [1, model["Is_A"], model["n"] * VT]
    else:
        row[4:8] = [1 if model["polarity"] == "npn" else -1,
                    model["Is_A"], model["beta_F"], model["beta_R"]]
    return row


@njit(cache=True)
def _expm1(x):
    # Linear continuation is C1 at 40 and prevents overflow during Newton trials.
    if x > 40.0:
        ex = np.exp(40.0)
        return ex * (1.0 + x - 40.0) - 1.0, ex
    ex = np.exp(x)
    return np.expm1(x), ex


@njit(cache=True)
def _soft_voltage(v, scale):
    x = v / scale
    if x > 40.0:
        return v, 1.0
    if x < -40.0:
        e = np.exp(x)
        return scale * e, e
    return scale * np.log1p(np.exp(x)), 1.0 / (1.0 + np.exp(-x))


@njit(cache=True)
def evaluate(row, voltages):
    """Terminal currents INTO device and their analytic node-voltage Jacobian."""
    current = np.zeros(3)
    jac = np.zeros((3, 3))
    kind, pol = int(row[0]), row[4]
    if kind == 1:
        vd, vg, vs = pol * voltages[0], pol * voltages[1], pol * voltages[2]
        fs, ds = _soft_voltage(vg - vs - row[6], 2.0 * row[7])
        fd, dd = _soft_voltage(vg - vd - row[6], 2.0 * row[7])
        base = 0.5 * row[5] * (fs * fs - fd * fd)
        vds = vd - vs
        clm = 1.0 + row[8] * abs(vds)
        sgn = 1.0 if vds > 0.0 else (-1.0 if vds < 0.0 else 0.0)
        current[0] = pol * base * clm
        current[2] = -current[0]
        jac[0, 0] = row[5] * fd * dd * clm + base * row[8] * sgn
        jac[0, 1] = row[5] * (fs * ds - fd * dd) * clm
        jac[0, 2] = -row[5] * fs * ds * clm - base * row[8] * sgn
        for j in range(3):
            jac[2, j] = -jac[0, j]
    elif kind == 2:
        val, slope = _expm1((voltages[0] - voltages[1]) / row[6])
        current[0] = row[5] * val
        current[1] = -current[0]
        conductance = row[5] * slope / row[6]
        jac[0, 0], jac[0, 1] = conductance, -conductance
        jac[1, 0], jac[1, 1] = -conductance, conductance
    else:
        ef, gf = _expm1(pol * (voltages[1] - voltages[2]) / VT)
        er, gr = _expm1(pol * (voltages[1] - voltages[0]) / VT)
        inv_af, inv_ar = 1.0 + 1.0 / row[6], 1.0 + 1.0 / row[7]
        current[0] = pol * row[5] * (ef - inv_ar * er)
        current[2] = pol * row[5] * (er - inv_af * ef)
        current[1] = -current[0] - current[2]
        gf *= row[5] / VT
        gr *= row[5] / VT
        jac[0, 0], jac[0, 1], jac[0, 2] = inv_ar * gr, gf - inv_ar * gr, -gf
        jac[2, 0], jac[2, 1], jac[2, 2] = -gr, gr - inv_af * gf, inv_af * gf
        for j in range(3):
            jac[1, j] = -jac[0, j] - jac[2, j]
    return current, jac


@njit(cache=True)
def stamp(x, basic, residual, jacobian):
    for row in basic:
        count = 2 if int(row[0]) == 2 else 3
        voltages = np.zeros(3)
        for j in range(count):
            node = int(row[1 + j])
            voltages[j] = x[node - 1] if node > 0 else 0.0
        currents, jac = evaluate(row, voltages)
        for i in range(count):
            node = int(row[1 + i])
            if node > 0:
                residual[node - 1] += currents[i]
                for j in range(count):
                    other = int(row[1 + j])
                    if other > 0:
                        jacobian[node - 1, other - 1] += jac[i, j]


@njit(cache=True)
def step_limit(dx, basic):
    """Damp large junction-voltage changes without altering ideal-source constraints."""
    alpha = 1.0
    for row in basic:
        kind = int(row[0])
        for i, j in ((0, 1), (1, 2)):
            if kind == 2 and j == 2:
                continue
            a, b = int(row[1 + i]), int(row[1 + j])
            change = abs((dx[a - 1] if a else 0.0) - (dx[b - 1] if b else 0.0))
            limit = 1.0 if kind == 1 else 0.2
            if change > limit:
                alpha = min(alpha, limit / change)
    return alpha


@njit(cache=True)
def current_series(rows, cols, row, terminal):
    out = np.empty(len(rows))
    for i in range(len(rows)):
        volts = np.zeros(3)
        for j in range(len(cols)):
            volts[j] = rows[i, cols[j]] if cols[j] > 0 else 0.0
        cur, _ = evaluate(row, volts)
        out[i] = cur[terminal]
    return out
