"""Deterministic compute kinds: branches, charge_balance, vg_curve (+ the internal `folds` alias kind).

Runs inside worker processes only (imports the numba engine through server.engine_bridge).
Result shapes follow docs/WEB_CONTRACT.md §2.

Branch array columns (photo_mean.curve_grid): 0 V_D, 1 I_D, 2 F (net hole current into the body),
3 BJT seed, 4 emitter, 5 bulk SRH loss, 6 out-diffusion loss, 7 junction SRH loss, 8 junction BTBT,
9 GIDL, 10 δ/N_A, 11 neutral length (cm), 12 R_acc (Ω), 13 charge term, 14 w_s, 15 w_d, 16 channel,
17 u, 18 r, 19 peak junction field, 20 hole drop (V).  I_PH (p[13]) is a constant supply.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import brentq

from server import params
from server.engine_bridge import MODEL, S, m
from server.progress import null_progress

Q = float(m.Q)
GRID_MIN, GRID_MAX = 201, 2001
SWEEP_MAX_POINTS = 2001
# A u step this large between the three rows of a fold means MODEL.classify fitted its parabola across an
# untraced gap of the locus (e.g. V_G >~ 0 V: no steady state for 1e-40 < u < 0.84 V).  Legitimate folds have
# rows one linear grid step apart (<= 5.5 mV at grid 201); the gaps seen are >= 0.83 V.
LOCUS_GAP_U = 0.05
GAP_WARNING = ("steady-state locus not traceable at the fold (gap in u between the fold rows, or a spurious fold "
               "pair with V_LD >= V_LU or above 8 V): reported as no latch")
V_FOLD_CAP = 8.0   # fold pairs above the sweep cap are extrapolation artefacts (e.g. beta <= 0.3x: V_LD > V_LU at 9-13 V)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _device(payload: dict) -> dict:
    return params.resolve_device(payload.get("device"))


def _grid(device: dict, warnings: list[str]) -> int:
    g = int(round(float(device.get("numerics", {}).get("grid", 601))))
    if not GRID_MIN <= g <= GRID_MAX:
        warnings.append(f"numerics.grid clamped from {g} to [{GRID_MIN}, {GRID_MAX}]")
        g = min(max(g, GRID_MIN), GRID_MAX)
    return g


def _pvec(device: dict, **kw: float) -> np.ndarray:
    return np.asarray(params.build_p(device, **kw), dtype=float)


def _fnum(x: Any) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def curve_dict(b: np.ndarray, iph: float) -> dict:
    """Contract `Curve` from a curve_grid block."""
    b = np.asarray(b, float).reshape(-1, 21)
    vd, cur = b[:, 0], b[:, 1]
    comp = dict(
        channel=b[:, 16], seed=b[:, 3],
        ii_total=cur - b[:, 3] - b[:, 8] - b[:, 9] - b[:, 16] - iph,
        btbt_junction=b[:, 8], gidl=b[:, 9], photo=np.full(len(b), float(iph)),
        loss_bulk_srh=b[:, 5], loss_diffusion=b[:, 6], loss_junction_srh=b[:, 7], net_F=b[:, 2],
        hole_drop_V=b[:, 20], injection=b[:, 10], r_access_ohm=b[:, 12],
    )
    return dict(vd=vd, id=cur, u=b[:, 17], r=b[:, 18], comp=comp)


def fold_point(b: np.ndarray, ind: int) -> tuple[float, float, float]:
    """(V, I, u) at the fold near row `ind`: quadratic V_D(u) through 3 rows (as FastModel.classify),
    current by log-interpolation along u."""
    ind = int(min(max(ind, 1), len(b) - 2))
    uu = b[ind - 1:ind + 2, 17]
    xx = uu - b[ind, 17]
    co = np.polyfit(xx, b[ind - 1:ind + 2, 0], 2)
    du = -co[1] / (2 * co[0]) if co[0] != 0 else 0.0
    du = float(np.clip(du, xx[0], xx[-1]))
    v = float(np.polyval(co, du))
    uf = float(b[ind, 17] + du)
    cur = float(np.exp(np.interp(uf, uu, np.log(np.maximum(b[ind - 1:ind + 2, 1], 1e-300)))))
    return v, cur, uf


def _monotone(part: np.ndarray) -> np.ndarray:
    """Rows with strictly increasing V_D (keeps the first occurrence), for np.interp."""
    if len(part) < 2:
        return part
    keep = [0]
    last = part[0, 0]
    for k in range(1, len(part)):
        if part[k, 0] > last:
            keep.append(k)
            last = part[k, 0]
    return part[keep]


def _interp_log(part: np.ndarray, v: np.ndarray) -> np.ndarray:
    """log-interpolated current of a monotone branch block at voltages v (NaN outside its range)."""
    out = np.full(len(v), np.nan)
    if len(part) == 0:
        return out
    if len(part) == 1:
        out[np.isclose(v, part[0, 0])] = part[0, 1]
        return out
    x, y = part[:, 0], np.log(np.maximum(part[:, 1], 1e-300))
    ok = (v >= x[0] - 1e-12) & (v <= x[-1] + 1e-12)
    out[ok] = np.exp(np.interp(v[ok], x, y))
    return out


def _sweep_grid(vd_max: float, dv: float, warnings: list[str] | None = None) -> np.ndarray:
    """Voltage grid of the quasi-static double sweep.  A finer dv is coarsened to SWEEP_MAX_POINTS points per
    direction without a warning: the curve is a resampling of the traced branches (≈ 600 rows) and the fold
    jumps are inserted exactly, so the step only sets the display resolution (reported as `sweep_dv_V`)."""
    n = min(int(round(vd_max / dv)) + 1, SWEEP_MAX_POINTS)
    return np.linspace(0.0, vd_max, max(n, 2))


def _fold_row(v: float, cur: float) -> np.ndarray:
    row = np.full((1, 21), np.nan)
    row[0, 0], row[0, 1] = v, cur
    return row


def double_sweep(b: np.ndarray | None, i: int | None, j: int | None, folds: dict, vd_max: float, dv: float,
                 warnings: list[str], full: np.ndarray | None = None) -> dict:
    """Quasi-static triangular sweep 0 -> vd_max -> 0: up follows HRS until V_LU then LRS; down follows LRS
    until V_LD then HRS.  Currents are log-interpolated along each monotone branch (as FastModel.double_curve).
    The fold jumps are inserted explicitly (two points at the same V_D)."""
    v = _sweep_grid(vd_max, dv, warnings)
    vdown = v[::-1]
    if b is None:  # no two-fold branch: follow the monotone low-u part of the traced locus
        part = _monotone(full) if full is not None and len(full) else np.empty((0, 21))
        # cut at the first local maximum, if any
        if full is not None and len(full) > 2:
            dvv = np.diff(full[:, 0])
            mx = np.flatnonzero((dvv[:-1] > 0) & (dvv[1:] < 0))
            if len(mx):
                part = _monotone(full[: mx[0] + 2])
        up_i, dn_i = _interp_log(part, v), _interp_log(part, vdown)
        up_i[v == 0] = 0.0
        dn_i[vdown == 0] = 0.0
        return dict(up=dict(vd=v, id=up_i), down=dict(vd=vdown, id=dn_i))

    V_LU, V_LD = folds["V_LU"], folds["V_LD"]
    I_LU, I_LD = folds["I_LU"], folds["I_LD"]
    # the refined fold points close the branches (the grid rows b[i], b[j] stop just short of V_LU / V_LD)
    hrs, lrs = b[: i + 1], b[j:]
    if V_LU > hrs[-1, 0]:
        hrs = np.vstack([hrs, _fold_row(V_LU, I_LU)])
    if V_LD < lrs[0, 0]:
        lrs = np.vstack([_fold_row(V_LD, I_LD), lrs])
    hrs, lrs = _monotone(hrs), _monotone(lrs)
    latches = V_LU <= vd_max
    # up
    if latches:
        lo = v[v < V_LU]
        hi = v[v > V_LU]
        up_v = np.r_[lo, V_LU, V_LU, hi]
        up_i = np.r_[_interp_log(hrs, lo), I_LU, _interp_log(lrs, np.array([V_LU]))[0], _interp_log(lrs, hi)]
    else:
        up_v, up_i = v, _interp_log(hrs, v)
    # down
    if latches and V_LD > 0:
        hi = vdown[vdown > V_LD]
        lo = vdown[vdown < V_LD]
        dn_v = np.r_[hi, V_LD, V_LD, lo]
        dn_i = np.r_[_interp_log(lrs, hi), I_LD, _interp_log(hrs, np.array([V_LD]))[0], _interp_log(hrs, lo)]
    elif latches:
        dn_v, dn_i = vdown, _interp_log(lrs, vdown)
    else:
        dn_v, dn_i = vdown, _interp_log(hrs, vdown)
    up_i = np.where(up_v == 0, 0.0, up_i)
    dn_i = np.where(dn_v == 0, 0.0, dn_i)
    return dict(up=dict(vd=up_v, id=up_i), down=dict(vd=dn_v, id=dn_i))


def fold_gap(z) -> bool:
    """True when the three rows of either fold are not contiguous in u, i.e. MODEL.classify took the first row
    after an untraced gap of the locus as the maximum and fitted its fold parabola across the gap."""
    b, i, j, _ = z
    return any(float(np.max(np.diff(b[ind - 1:ind + 2, 17]))) > LOCUS_GAP_U for ind in (i, j))


def classify_checked(p: np.ndarray, grid: int) -> tuple[tuple | None, bool]:
    """(MODEL.classify result or None, gap) — a fold fitted across a gap in the traced locus, or a spurious fold
    pair (V_LD >= V_LU, or V_LU above 8 V), is rejected (returned as None with gap = True).  engine/ is verbatim,
    so the check lives here."""
    z = MODEL.classify(p, m.state_grid(grid))
    if z is not None and (fold_gap(z) or not (float(z[3][0]) > float(z[3][1]) and float(z[3][0]) <= V_FOLD_CAP)):
        return None, True
    return z, False


def classify(p: np.ndarray, grid: int, warnings: list[str] | None = None) -> tuple[np.ndarray, int, int, np.ndarray] | None:
    z, gap = classify_checked(p, grid)
    if gap and warnings is not None:
        warnings.append(GAP_WARNING)
    return z


def folds_of(z) -> dict:
    """Contract `folds` block from a classify() tuple (None → all null)."""
    if z is None:
        return dict(V_LU=None, V_LD=None, I_LU=None, I_LD=None, u_LU=None, u_LD=None, window_V=None)
    b, i, j, fold = z
    _, I_LU, u_LU = fold_point(b, i)
    _, I_LD, u_LD = fold_point(b, j)
    V_LU, V_LD = float(fold[0]), float(fold[1])
    return dict(V_LU=V_LU, V_LD=V_LD, I_LU=I_LU, I_LD=I_LD, u_LU=u_LU, u_LD=u_LD, window_V=V_LU - V_LD)


# ---------------------------------------------------------------------------------------------
# kind "branches"
# ---------------------------------------------------------------------------------------------
def run_branches(payload: dict, progress=null_progress) -> dict:
    t0 = time.perf_counter()
    warnings: list[str] = []
    device = _device(payload)
    sweep = params.resolve_section(device.get("preset"), "sweep", payload.get("sweep"))
    vd_max = float(sweep["vd_max_V"])
    dv = float(sweep["dv_V"])
    if not (vd_max > 0 and dv > 0):
        raise ValueError("sweep.vd_max_V and sweep.dv_V must be positive")
    grid = _grid(device, warnings)
    p = _pvec(device)
    iph = float(p[13])
    progress(0.05, "tracing the steady-state locus")
    z, gap = classify_checked(p, grid)
    progress(0.6, "splitting branches")
    empty = curve_dict(np.empty((0, 21)), iph)
    if z is None:
        full = MODEL.branch(p, m.state_grid(grid))
        traced = full
        if gap:
            warnings.append(GAP_WARNING)
            # the quasi-static sweep follows only the part of the locus before the first gap
            cut = np.flatnonzero(np.diff(full[:, 17]) > LOCUS_GAP_U)
            traced = full[: int(cut[0]) + 1] if len(cut) else full
        else:
            warnings.append("no two-fold branch at these parameters: the device does not latch (folds are null)")
        fl = folds_of(None)
        ds = double_sweep(None, None, None, fl, vd_max, dv, warnings, full=traced)
        res = dict(latch=False, HRS=empty, unstable=empty, LRS=empty, full=curve_dict(full, iph), folds=fl,
                   double_sweep=ds)
    else:
        b, i, j, _ = z
        fl = folds_of(z)
        lrs = b[j:]
        beyond = np.flatnonzero(lrs[:, 0] > vd_max + 1.0)   # LRS V_D increases along u: cut at the first row beyond
        if len(beyond):
            lrs = lrs[: max(int(beyond[0]), 2)]
        if fl["V_LU"] > vd_max:
            warnings.append(f"V_LU = {fl['V_LU']:.4f} V exceeds the sweep maximum {vd_max:g} V: "
                            "the quasi-static sweep never latches")
        ds = double_sweep(b, i, j, fl, vd_max, dv, warnings)
        res = dict(latch=True, HRS=curve_dict(b[: i + 1], iph), unstable=curve_dict(b[i: j + 1], iph),
                   LRS=curve_dict(lrs, iph), full=curve_dict(b, iph), folds=fl, double_sweep=ds)
    progress(1.0, "done")
    sg = _sweep_grid(vd_max, dv)
    res.update(iph_A=iph, p=p.tolist(), grid=grid, vd_max_V=vd_max, sweep_dv_V=float(sg[1] - sg[0]),
               runtime_s=time.perf_counter() - t0, warnings=warnings)
    return res


def run_folds(payload: dict, progress=null_progress) -> dict:
    """Internal kind behind the /api/folds alias: folds only (no branch arrays)."""
    t0 = time.perf_counter()
    warnings: list[str] = []
    device = _device(payload)
    grid = _grid(device, warnings)
    p = _pvec(device)
    progress(0.1, "classifying")
    z, gap = classify_checked(p, grid)
    if gap:
        warnings.append(GAP_WARNING)
    elif z is None:
        warnings.append("no two-fold branch: the device does not latch")
    progress(1.0, "done")
    return dict(latch=z is not None, folds=folds_of(z), iph_A=float(p[13]), p=p.tolist(), grid=grid,
                runtime_s=time.perf_counter() - t0, warnings=warnings)


# ---------------------------------------------------------------------------------------------
# kind "charge_balance"
# ---------------------------------------------------------------------------------------------
STATE_COLS = ["u", "r", "I_D", "gen_per_s", "loss_per_s", "Cox_psi", "Q_exc", "qNA_A_Ln", "F_A", "seed_A", "unit_A"]


def _comp(u: float, r: float, p: np.ndarray) -> np.ndarray:
    return m.components(u, r, p, MODEL.na, MODEL.vbi, MODEL.rg, MODEL.fg, MODEL.table)


def _finite_r_max(u: float, vd: float, p: np.ndarray) -> float:
    """Largest r ≤ vd - u where components() is finite (the depletion edge reaches the neutral-length limit
    at large r, where the engine returns NaN)."""
    hi = vd - u
    if np.isfinite(_comp(u, hi, p)[0]):
        return hi
    lo = 0.0
    if not np.isfinite(_comp(u, lo, p)[0]):
        return np.nan
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if np.isfinite(_comp(u, mid, p)[0]):
            lo = mid
        else:
            hi = mid
    return lo


def state_row(u: float, vd: float, p: np.ndarray) -> np.ndarray:
    """setup_photo.state(u, vd, p) with a wider fallback: when the r bracket [0, vd-u] ends in the region where
    components() is NaN, the bracket is shrunk to the finite part.  Same row formulas as S.state:
    [u, r, I_D, gen(1/s), loss(1/s), C_ox ψ, Q_exc, q N_A A L_n, F(A), seed(A), unit(A)]."""
    try:
        return S.state(float(u), float(vd), p)
    except ValueError:
        pass
    u, vd = float(u), float(vd)
    rmax = _finite_r_max(u, vd, p)
    if not np.isfinite(rmax) or rmax <= 0 or _comp(u, rmax, p)[0] < vd:
        raise ValueError("no drain-junction solution")
    r = brentq(lambda rr: _comp(u, rr, p)[0] - vd, 0.0, rmax, xtol=1e-11)
    z = _comp(u, r, p)
    psi = u - m.VT * np.log1p(z[10])
    ratio = (p[5] * 1e-7 / (m.TSI_M * 100)) * (p[7] * 1e-7 / z[11])
    qb = (z[13] - m.COX_F * u) / (1 + ratio)
    qa = qb * ratio
    return np.array([u, r, z[1], (z[1] - z[3] - z[16]) / m.Q, (z[5] + z[6] + z[7]) / m.Q, m.COX_F * psi, qb + qa,
                     m.Q * MODEL.na * m.AREA_CM2 * z[11], z[2], z[3], z[8] + z[9] + z[18]])


def _state_row(u: float, vd: float, p: np.ndarray) -> np.ndarray | None:
    try:
        row = state_row(float(u), float(vd), p)
    except (ValueError, ZeroDivisionError, FloatingPointError, RuntimeError):
        return None
    return row if np.all(np.isfinite(row[:11])) else None


def body_rows(vd: float, p: np.ndarray, ug: np.ndarray, progress=null_progress, lo: float = 0.0, hi: float = 1.0,
              label: str = "body-charge rows") -> np.ndarray:
    """S.state rows over the u grid (rows where the r-solve fails are skipped)."""
    rows = []
    n = len(ug)
    for k, u in enumerate(ug):
        if k % 25 == 0:
            progress(lo + (hi - lo) * k / max(n, 1), f"{label} {k}/{n}")
        row = _state_row(u, vd, p)
        if row is not None:
            rows.append(row)
    return np.array(rows).reshape(-1, 11)


def charge_coordinate(rows: np.ndarray) -> np.ndarray:
    """Q_C = C_ox ψ + Q_exc + q N_A A L_n (C): the compound-FPT lattice coordinate × q."""
    return rows[:, 5] + rows[:, 6] + rows[:, 7]


def quasi_potential(x: np.ndarray, gen: np.ndarray, loss: np.ndarray) -> np.ndarray:
    """U(x) = -∫ ln(G/L) dx along the hole-count coordinate x (trapezoid), U(x[0]) = 0 (shifted later)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(gen / loss)
    lg = np.where(np.isfinite(lg), lg, np.nan)
    # fill isolated gaps so the integral stays defined
    if np.isnan(lg).any() and np.isfinite(lg).sum() >= 2:
        ok = np.isfinite(lg)
        lg = np.interp(np.arange(len(lg)), np.flatnonzero(ok), lg[ok])
    seg = 0.5 * (lg[1:] + lg[:-1]) * np.diff(x)
    return -np.r_[0.0, np.cumsum(seg)]


def u_grid(u_min: float, u_max: float, n_u: int) -> np.ndarray:
    """Linear grid on [max(u_min, 0.02), u_max] plus 41 log-spaced points on [1e-9, 0.02] when u_min < 0.02
    (the deep-equilibrium HRS root at low V_D sits at u ~ 1e-7 V)."""
    lin = np.linspace(max(u_min, 0.02), u_max, n_u)
    if u_min < 0.02:
        lin = np.r_[np.geomspace(max(u_min, 1e-9), 0.02, 41)[:-1], lin]
    return np.unique(lin)


def run_charge_balance(payload: dict, progress=null_progress) -> dict:
    """Body-charge landscape at fixed V_D.
    Optional payload keys: u_min (0 → log-spaced from 1e-9 V), u_max (1.10 V), n_u (401 linear points, ≤ 2001)."""
    t0 = time.perf_counter()
    warnings: list[str] = []
    device = _device(payload)
    vd = float(payload.get("vd", 3.2))
    if not 0.05 <= vd <= 8.0:
        raise ValueError("vd must be within [0.05, 8] V")
    u_min = float(payload.get("u_min", 0.0))
    u_max = float(payload.get("u_max", 1.10))
    n_u = int(payload.get("n_u", 401))
    if not (0 <= u_min < u_max <= 1.12):
        raise ValueError("need 0 <= u_min < u_max <= 1.12 V")
    n_u = min(max(n_u, 21), 2001)
    p = _pvec(device)
    ug = u_grid(u_min, min(u_max, vd - 1e-3), n_u)
    rows = body_rows(vd, p, ug, progress, 0.0, 0.8)
    skipped = len(ug) - len(rows)
    if skipped:
        warnings.append(f"{skipped} of {len(ug)} u points skipped (no drain-junction solution)")
    if len(rows) < 3:
        raise ValueError(f"no body-charge solutions at V_D = {vd} V")
    u = rows[:, 0]
    QC = charge_coordinate(rows)
    gen_A, loss_A, unit_A, F_A = rows[:, 3] * Q, rows[:, 4] * Q, rows[:, 10], rows[:, 8]

    progress(0.85, "refining roots")
    roots = []
    sgn = np.sign(F_A)
    for k in np.flatnonzero(sgn[:-1] * sgn[1:] < 0):
        ua, ub = u[k], u[k + 1]
        try:
            ur = brentq(lambda uu: state_row(uu, vd, p)[8], ua, ub, xtol=1e-12)
            rr = state_row(ur, vd, p)
        except (ValueError, RuntimeError):
            ur = ua - F_A[k] * (ub - ua) / (F_A[k + 1] - F_A[k])
            rr = None
        qc = float(charge_coordinate(rr[None])[0]) if rr is not None else float(np.interp(ur, u, QC))
        cur = float(rr[2]) if rr is not None else float(np.exp(np.interp(ur, u, np.log(np.maximum(rows[:, 2], 1e-300)))))
        roots.append(dict(u=float(ur), Q_C=qc, kind="stable" if F_A[k] > 0 else "unstable", id=cur,
                          r=float(rr[1]) if rr is not None else None))

    x = QC / Q
    U = quasi_potential(x, rows[:, 3], rows[:, 4])
    stable = [r for r in roots if r["kind"] == "stable"]
    if stable:
        U = U - float(np.interp(stable[0]["u"], u, U))
    else:
        U = U - float(np.nanmin(U))
        warnings.append("no stable root at this V_D: potential zeroed at its minimum")
    progress(1.0, "done")
    return dict(vd=vd, u=u, r=rows[:, 1], id=rows[:, 2], Q_C=QC, generation_A=gen_A, loss_A=loss_A,
                unit_A=unit_A, ii_A=gen_A - unit_A, F_A=F_A, potential=U, roots=roots,
                holes=x, iph_A=float(p[13]), runtime_s=time.perf_counter() - t0, warnings=warnings)


# ---------------------------------------------------------------------------------------------
# kind "vg_curve"
# ---------------------------------------------------------------------------------------------
def _latch_at(device: dict, vg: float, grid: int):
    return classify(_pvec(dict(device, vg=float(vg))), grid)


def _bisect_edge(device: dict, grid: int, v_no: float, v_yes: float, tol: float = 1e-3, progress=None) -> float:
    """Bisection between a no-latch V_G and a latch V_G; returns the latch-side end (within tol)."""
    while abs(v_yes - v_no) > tol:
        mid = 0.5 * (v_no + v_yes)
        if _latch_at(device, mid, grid) is None:
            v_no = mid
        else:
            v_yes = mid
        if progress is not None:
            progress()
    return 0.5 * (v_no + v_yes)


def latch_window(device: dict, grid: int, vg: np.ndarray, latch: np.ndarray, tol: float = 1e-3,
                 warnings: list[str] | None = None, progress=None) -> dict:
    """Refine the latch-existence edges of a scanned V_G grid by bisection."""
    warnings = warnings if warnings is not None else []
    idx = np.flatnonzero(latch)
    if not len(idx):
        warnings.append("no latch anywhere in the scanned V_G range")
        return dict(vg_low=None, vg_high=None)
    k0, k1 = int(idx[0]), int(idx[-1])
    if k0 == 0:
        low = None
        warnings.append(f"the latch window extends below vg_min = {vg[0]:g} V")
    else:
        low = _bisect_edge(device, grid, float(vg[k0 - 1]), float(vg[k0]), tol, progress)
    if k1 == len(vg) - 1:
        high = None
        warnings.append(f"the latch window extends above vg_max = {vg[-1]:g} V")
    else:
        high = _bisect_edge(device, grid, float(vg[k1 + 1]), float(vg[k1]), tol, progress)
    if len(idx) != k1 - k0 + 1:
        warnings.append("the latch region in V_G is not contiguous; edges refer to its outermost ends")
    return dict(vg_low=low, vg_high=high)


def run_vg_curve(payload: dict, progress=null_progress) -> dict:
    """Folds vs V_G.  Payload: device, vg_min (-4.2), vg_max (-0.6), n (37, ≤ 61), refine (true)."""
    t0 = time.perf_counter()
    warnings: list[str] = []
    device = _device(payload)
    vg_min = float(payload.get("vg_min", -4.2))
    vg_max = float(payload.get("vg_max", -0.6))
    n = int(payload.get("n", 37))
    if not vg_min < vg_max:
        raise ValueError("vg_min must be < vg_max")
    if not 2 <= n <= 61:
        warnings.append(f"n clamped from {n} to [2, 61]")
        n = min(max(n, 2), 61)
    grid = _grid(device, warnings)
    vg = np.linspace(vg_min, vg_max, n)
    V_LU = np.full(n, np.nan)
    V_LD = np.full(n, np.nan)
    I_LU = np.full(n, np.nan)
    I_LD = np.full(n, np.nan)
    latch = np.zeros(n, bool)
    refine = bool(payload.get("refine", True))
    total = n + (16 if refine else 0)
    gaps = []
    for k, v in enumerate(vg):
        progress(k / total, f"V_G = {v:.3f} V ({k + 1}/{n})")
        z, gap = classify_checked(_pvec(dict(device, vg=float(v))), grid)
        if gap:
            gaps.append(float(v))
        if z is not None:
            f = folds_of(z)
            latch[k] = True
            V_LU[k], V_LD[k], I_LU[k], I_LD[k] = f["V_LU"], f["V_LD"], f["I_LU"], f["I_LD"]
    window = dict(vg_low=None, vg_high=None)
    if refine:
        count = [n]

        def tick() -> None:
            count[0] += 1
            progress(min(count[0] / total, 0.99), "refining the latch-window edges")

        window = latch_window(device, grid, vg, latch, 1e-3, warnings, tick)
    if gaps:
        warnings.append(f"steady-state locus not traceable at the fold (gap in u) at {len(gaps)} V_G value(s) "
                        f"({gaps[0]:+.3f} … {gaps[-1]:+.3f} V): reported as no latch")
    progress(1.0, "done")
    return dict(vg=vg, V_LU=V_LU, V_LD=V_LD, I_LU=I_LU, I_LD=I_LD, latch=latch.tolist(), window=window,
                grid=grid, runtime_s=time.perf_counter() - t0, warnings=warnings)
