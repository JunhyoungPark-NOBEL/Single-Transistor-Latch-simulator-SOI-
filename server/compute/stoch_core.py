"""Shared numerics of the stochastic package (fold nodes, first-passage hazard curves, statistics).

Everything here works on the full 26-element parameter vector ``p`` from ``params.build_p`` so the
photo/extension levers (p[13..25]) and the local-state centre are honoured.  The algorithms are the
engine's, generalised from a handful of keyword arguments to an arbitrary ``p``:

* ``lu_hazard_curve``  = ``photo_fpt.hazard_curve`` (compound-jump first passage, II clusters + unit
  events incl. photogeneration, ``ct.cf.make_lattice(rows, 'LU', .06)`` + ``ct.cf.backward``).
* ``ld_hazard_curve``  = the LD half of ``conditional_table.calculate`` with the photo ``S.state`` rows.
* ``fold_node``        = ``MODEL.classify`` folds + the HRS/LRS branches (for traces).
* ``quantiles``        = ``photo_fpt.quantiles`` (identical fold-atom handling).

Results are cached on disk under ``server/.cache/stochastic/`` keyed by sha256 of the rounded ``p``
(+ algorithm settings), so fold tables, hazard nodes and V_G curves share work across requests.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid

# fold/hazard node cache (override with STL_STOCH_CACHE_DIR, e.g. for cold-start timing)
CACHE_DIR = Path(os.environ.get("STL_STOCH_CACHE_DIR") or Path(__file__).resolve().parents[1] / ".cache" / "stochastic")
CACHE_VERSION = "stoch-1"
PROB = (np.arange(10001) + 0.5) / 10001          # photo_fpt.PROB
HAZARD_FLOOR = 1e-4                               # photo_fpt / compound_fpt resolved-hazard cut (1/s)
LOG_ZERO = -745.0                                 # log of a zero hazard (exp -> 0)
DIST_STEP = 0.001                                 # common distance-to-fold grid (V)


# --------------------------------------------------------------------------------------------
# engine access (lazy: importing numba only inside worker processes)
# --------------------------------------------------------------------------------------------
_ENGINE: dict[str, Any] = {}


def engine():
    if not _ENGINE:
        from server import engine_bridge as eb
        _ENGINE.update(S=eb.S, m=eb.m, MODEL=eb.MODEL, ct=eb.ct, eb=eb)
    return _ENGINE


# --------------------------------------------------------------------------------------------
# disk cache
# --------------------------------------------------------------------------------------------
def p_key(p, *extra) -> str:
    txt = json.dumps([CACHE_VERSION, [format(float(x), ".12g") for x in np.asarray(p, float)],
                      [e if isinstance(e, str) else format(float(e), ".12g") for e in extra]])
    return hashlib.sha256(txt.encode()).hexdigest()[:40]


def cache_get(kind: str, key: str) -> dict | None:
    path = CACHE_DIR / kind / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def cache_put(kind: str, key: str, rec: dict) -> None:
    folder = CACHE_DIR / kind
    try:
        folder.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(rec, f, allow_nan=True)
        os.replace(tmp, folder / f"{key}.json")      # atomic: several worker processes share the cache
    except OSError:
        pass


def _floats(a) -> list:
    return [float(x) for x in np.asarray(a, float).ravel()]


# --------------------------------------------------------------------------------------------
# folds and branches
# --------------------------------------------------------------------------------------------
BRANCH_VMAX = 9.0     # store branches up to vd_max cap (8 V) + 1 V


def _branch_part(part: np.ndarray) -> dict:
    keep = part[:, 0] <= BRANCH_VMAX
    part = part[keep] if keep.sum() >= 2 else part[:2]
    r7 = lambda a: [float(f"{x:.7g}") for x in np.asarray(a, float)]     # 7 digits: traces only
    return dict(vd=r7(part[:, 0]), id=r7(part[:, 1]))


def fold_node(p, grid: int = 601, with_branches: bool = True) -> dict:
    """Folds (V_LU, V_LD, I_LU, u_LU, u_LD) and HRS/LRS branches at one parameter vector (cached)."""
    p = np.asarray(p, float)
    key = p_key(p, "fold", grid)
    rec = cache_get("fold", key)
    if rec is not None:
        return rec
    E = engine()
    tic = time.perf_counter()
    z = E["MODEL"].classify(p, E["m"].state_grid(int(grid)))
    if z is None:
        rec = dict(latch=False, V_LU=None, V_LD=None)
    else:
        b, i, j, fold = z
        rec = dict(latch=True, V_LU=float(fold[0]), V_LD=float(fold[1]), I_LU=float(b[i, 1]),
                   u_LU=float(b[i, 17]), u_LD=float(b[j, 17]), I_LD=float(b[j, 1]),
                   HRS=_branch_part(b[:i + 1]), LRS=_branch_part(b[j:]))
    rec["seconds"] = time.perf_counter() - tic
    cache_put("fold", key, rec)
    return rec


# --------------------------------------------------------------------------------------------
# first-passage hazard curves
# --------------------------------------------------------------------------------------------
def _lattice_hazard(rows, direction):
    ct = engine()["ct"]
    xx, ix, r, bt, ii, death = ct.cf.make_lattice(rows, direction, .06)
    tm, _A, _check = ct.cf.backward(r, bt, ii, death, direction)
    return 1 / tm[ix] if np.isfinite(tm[ix]) and tm[ix] > 0 else np.nan


def lu_hazard_curve(p, window: float = .28, step: float = .004, grid: int = 601,
                    progress: Callable[[float], None] | None = None) -> dict:
    """photo_fpt.hazard_curve for an arbitrary 26-element p (same voltages, u grid, lattice)."""
    p = np.asarray(p, float)
    key = p_key(p, "LU", window, step, grid)
    rec = cache_get("hazard", key)
    if rec is not None:
        return rec
    E = engine()
    S, m, M = E["S"], E["m"], E["MODEL"]
    tic = time.perf_counter()
    clas = M.classify(p, m.state_grid(int(grid)))
    if clas is None:
        rec = dict(direction="LU", fold_V=None, VLD_fold_V=None, voltage=[], hazard=[], skipped=0,
                   window_V=window, step_V=step, seconds=time.perf_counter() - tic)
        cache_put("hazard", key, rec)
        return rec
    b, i, j, fold = clas
    uf = b[i, 17]
    volts = np.arange(max(fold[0] - window, fold[1] + .001), fold[0] - .001, step)
    ug = np.unique(np.round(np.r_[np.linspace(.1, .9, 181), np.linspace(uf - .065, uf + .065, 61)], 12))
    V, Hz, skipped = [], [], 0
    for k, vd in enumerate(volts):
        try:
            rows = np.array([S.state(u, vd, p) for u in ug])
            h = _lattice_hazard(rows, "LU")
        except (ValueError, IndexError, AssertionError, ZeroDivisionError, FloatingPointError):
            skipped += 1
            continue
        V.append(float(vd))
        Hz.append(float(h))
        if progress is not None and k % 8 == 0:
            progress((k + 1) / max(len(volts), 1))
    rec = dict(direction="LU", fold_V=float(fold[0]), VLD_fold_V=float(fold[1]), I_at_fold_A=float(b[i, 1]),
               channel_at_fold_A=float(b[i, 16]), voltage=V, hazard=Hz, skipped=skipped, window_V=window,
               step_V=step, seconds=time.perf_counter() - tic)
    cache_put("hazard", key, rec)
    return rec


def ld_hazard_curve(p, window: float = .30, step: float = .004, grid: int = 601,
                    progress: Callable[[float], None] | None = None) -> dict:
    """LD first passage (conditional_table.calculate, direction 'LD') with the photo S.state rows."""
    p = np.asarray(p, float)
    key = p_key(p, "LD", window, step, grid)
    rec = cache_get("hazard", key)
    if rec is not None:
        return rec
    E = engine()
    S, m, M = E["S"], E["m"], E["MODEL"]
    tic = time.perf_counter()
    clas = M.classify(p, m.state_grid(int(grid)))
    if clas is None:
        rec = dict(direction="LD", fold_V=None, VLU_fold_V=None, voltage=[], hazard=[], skipped=0,
                   window_V=window, step_V=step, seconds=time.perf_counter() - tic)
        cache_put("hazard", key, rec)
        return rec
    b, i, j, fold = clas
    uf = b[j, 17]
    volts = np.arange(fold[1] + window, fold[1] + .001, -step)
    volts = volts[volts < fold[0] - .001]
    ug = np.unique(np.round(np.r_[np.linspace(.55, 1.04, 181), np.linspace(uf - .065, uf + .065, 61)], 12))
    V, Hz, skipped = [], [], 0
    for k, vd in enumerate(volts):
        try:
            rows = np.array([S.state(u, vd, p) for u in ug])
            h = _lattice_hazard(rows, "LD")
        except (ValueError, IndexError, AssertionError, ZeroDivisionError, FloatingPointError):
            skipped += 1
            continue
        V.append(float(vd))
        Hz.append(float(h))
        if progress is not None and k % 8 == 0:
            progress((k + 1) / max(len(volts), 1))
    rec = dict(direction="LD", fold_V=float(fold[1]), VLU_fold_V=float(fold[0]), voltage=V, hazard=Hz,
               skipped=skipped, window_V=window, step_V=step, seconds=time.perf_counter() - tic)
    cache_put("hazard", key, rec)
    return rec


def resolved_hazard(rec: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """(V, h, begin) with the engine's cut: keep the final connected region with h >= 1e-4 /s."""
    V = np.asarray(rec["voltage"], float)
    h = np.asarray(rec["hazard"], float)
    if len(V) == 0:
        return V, h, 0
    valid = np.isfinite(h) & (h >= HAZARD_FLOOR)
    bad = np.flatnonzero(~valid)
    begin = int(bad[-1] + 1) if len(bad) else 0
    h = h.copy()
    h[:begin] = 0.
    return V, h, begin


def cumulative_hazard(rec: dict, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """(V, cum) with cum = ∫h dV/rate along the ramp direction (LU: increasing V, LD: decreasing V)."""
    V, h, begin = resolved_hazard(rec)
    if len(V) < 2 or begin >= len(h):
        return V, np.zeros_like(V)
    prog = V if rec.get("direction", "LU") == "LU" else -V
    return V, cumulative_trapezoid(h / rate, prog, initial=0)


def quantiles(rec: dict, rate: float) -> np.ndarray | None:
    """photo_fpt.quantiles (LU); above-fold mass sits at the fold (deterministic fold atom)."""
    if rec["fold_V"] is None:
        return None
    if len(rec["voltage"]) < 2:
        return np.full(len(PROB), rec["fold_V"])
    V, h, begin = resolved_hazard(rec)
    if begin >= len(h):
        return np.full(len(PROB), rec["fold_V"])
    prog = V if rec.get("direction", "LU") == "LU" else -V
    cum = cumulative_trapezoid(h / rate, prog, initial=0)
    return np.interp(-np.log1p(-PROB), cum, V, right=rec["fold_V"])


def hazard_vs_distance(rec: dict, dist_grid: np.ndarray) -> np.ndarray | None:
    """log hazard on a distance-to-fold grid (LU: fold - V, LD: V - fold).

    Between the last computed voltage and the fold the hazard is held constant (as in the
    calibrated lookup gate_state_lookup.npz); beyond the computed window it is zero."""
    if rec["fold_V"] is None or len(rec["voltage"]) < 2:
        return None
    V, h, begin = resolved_hazard(rec)
    if begin >= len(h):
        return None
    d = (rec["fold_V"] - V) if rec.get("direction", "LU") == "LU" else (V - rec["fold_V"])
    order = np.argsort(d)
    d, lh = d[order], np.log(np.maximum(h[order], 1e-300))
    lh[h[order] <= 0] = LOG_ZERO
    out = np.interp(dist_grid, d, lh, left=lh[0], right=LOG_ZERO)
    out[dist_grid > d[-1] + 1e-12] = LOG_ZERO
    return out


# --------------------------------------------------------------------------------------------
# statistics helpers (result shapes of WEB_CONTRACT §2)
# --------------------------------------------------------------------------------------------
def _num(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def lag1(v) -> float | None:
    v = np.asarray(v, float)
    a, b = v[:-1], v[1:]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    a, b = a[ok], b[ok]
    if np.ptp(a) < 1e-9 or np.ptp(b) < 1e-9:      # constant series (e.g. V_LD exactly at a fixed fold)
        return None
    return _num(np.corrcoef(a, b)[0, 1])


def stats(v) -> dict:
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    x = v[ok]
    if len(x) == 0:
        return dict(n=0, mean=None, sd=None, median=None, p05=None, p95=None, min=None, max=None,
                    censored=int((~ok).sum()), lag1=None)
    return dict(n=int(len(x)), mean=_num(x.mean()), sd=_num(x.std(ddof=1)) if len(x) > 1 else None,
                median=_num(np.median(x)), p05=_num(np.quantile(x, .05)), p95=_num(np.quantile(x, .95)),
                min=_num(x.min()), max=_num(x.max()), censored=int((~ok).sum()), lag1=lag1(v))


def histogram(v) -> dict:
    x = np.asarray(v, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(edges=[], counts=[])
    if np.ptp(x) < 1e-6:
        edges = np.array([x.mean() - .005, x.mean() + .005])
    else:
        try:
            edges = np.histogram_bin_edges(x, "fd")
        except (ValueError, MemoryError):
            edges = np.histogram_bin_edges(x, 30)
        if not (10 <= len(edges) - 1 <= 80):
            edges = np.histogram_bin_edges(x, int(np.clip(len(edges) - 1, 10, 80)))
    counts, edges = np.histogram(x, edges)
    return dict(edges=_floats(edges), counts=[int(c) for c in counts])


def ecdf(v) -> dict:
    x = np.sort(np.asarray(v, float))
    x = x[np.isfinite(x)]
    n = len(x)
    return dict(v=_floats(x), p=_floats(np.arange(1, n + 1) / n) if n else [])


def gauss_hermite(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Probabilists' nodes t (state = centre + sigma*t) and normalised weights."""
    x, w = np.polynomial.hermite.hermgauss(int(k))
    return np.sqrt(2) * x, w / np.sqrt(np.pi)
