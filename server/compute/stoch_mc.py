"""Sweep Monte Carlo engines.

* ``GeneralModel`` / ``simulate_general``: any V_G, light, extension, ramp and V_D range.  Fold table
  V_LU(X), V_LD(X) of the action-point state X (PCHIP over classify nodes), optional linear emitter
  shift, compound-FPT hazard nodes at Gauss-Hermite states (log-interpolated in X, expressed vs the
  distance to the current fold) and a triangular-sweep time-stepping escape identical in spirit to
  ``gate_dynamic_compare.simulate`` (event when ∫h dt exceeds an Exp(1) draw, or at the fold).
* ``calibrated``: thin wrapper around ``gate_dynamic_compare.simulate`` (paper device only) plus a
  replay of its random-number stream to report the per-cycle state.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import lfilter

from . import stoch_core as C


# --------------------------------------------------------------------------------------------
# fold table in the action-point state
# --------------------------------------------------------------------------------------------
@dataclass
class FoldTable:
    x: np.ndarray                  # node states (absolute lever value)
    vlu: np.ndarray                # NaN where no latch
    vld: np.ndarray
    recs: list                     # fold_node records (branches for traces)
    xf: np.ndarray = field(init=False)
    vluf: np.ndarray = field(init=False)
    vldf: np.ndarray = field(init=False)

    def __post_init__(self):
        ok = np.isfinite(self.vlu) & np.isfinite(self.vld)
        if len(self.x) == 1 or ok.sum() == 0:
            self.xf = np.array([self.x[0], self.x[-1] + (1e-9 if len(self.x) == 1 else 0.)])
            v1 = self.vlu[ok][0] if ok.any() else np.nan
            v2 = self.vld[ok][0] if ok.any() else np.nan
            self.vluf, self.vldf = np.array([v1, v1]), np.array([v2, v2])
            return
        xs = np.linspace(self.x[0], self.x[-1], 4001)
        if ok.sum() == 1:
            lu = np.full_like(xs, self.vlu[ok][0]); ld = np.full_like(xs, self.vld[ok][0])
        else:
            lu = PchipInterpolator(self.x[ok], self.vlu[ok])(xs)
            ld = PchipInterpolator(self.x[ok], self.vld[ok])(xs)
        nearest = np.abs(xs[:, None] - self.x[None, :]).argmin(axis=1)
        bad = ~ok[nearest] | (xs < self.x[ok].min() - 1e-12) | (xs > self.x[ok].max() + 1e-12)
        lu[bad] = np.nan; ld[bad] = np.nan
        self.xf, self.vluf, self.vldf = xs, lu, ld

    def lu(self, X):
        return np.interp(X, self.xf, self.vluf)

    def ld(self, X):
        return np.interp(X, self.xf, self.vldf)

    def nearest_rec(self, X):
        ok = [k for k, r in enumerate(self.recs) if r.get("latch")]
        if not ok:
            return None
        k = ok[int(np.argmin(np.abs(self.x[ok] - X)))]
        return self.recs[k]

    def latch_fraction_gauss(self, x0, sigma) -> float:
        """Probability that a N(x0, sigma) state has a latch (fine-grid quadrature)."""
        if sigma <= 0:
            return float(np.isfinite(self.lu(x0)))
        t = np.linspace(-6, 6, 2401); w = np.exp(-t * t / 2); w /= w.sum()
        return float(np.sum(w * np.isfinite(self.lu(x0 + sigma * t))))


# --------------------------------------------------------------------------------------------
# hazard field h(X, distance)
# --------------------------------------------------------------------------------------------
@dataclass
class HazardField:
    x: np.ndarray            # node states (ascending), len K
    logh: np.ndarray         # (K, L) log hazard on the distance grid
    dstep: float = C.DIST_STEP

    def __call__(self, X, dist):
        X = np.asarray(X, float); dist = np.asarray(dist, float)
        K, L = self.logh.shape
        t = dist / self.dstep
        valid = np.isfinite(t) & (dist > 0) & (t <= L - 1)
        t = np.where(valid, t, 0.)
        j0 = np.clip(np.floor(t).astype(np.int64), 0, max(L - 2, 0))
        wy = np.clip(t - j0, 0., 1.) if L > 1 else np.zeros_like(t)
        j1 = np.minimum(j0 + 1, L - 1)
        if K == 1:
            lh = (1 - wy) * self.logh[0, j0] + wy * self.logh[0, j1]
        else:
            s = np.interp(np.where(np.isfinite(X), X, self.x[0]), self.x, np.arange(K, dtype=float))
            i0 = np.clip(np.floor(s).astype(np.int64), 0, K - 2)
            wx = s - i0
            lh = ((1 - wx) * ((1 - wy) * self.logh[i0, j0] + wy * self.logh[i0, j1])
                  + wx * ((1 - wy) * self.logh[i0 + 1, j0] + wy * self.logh[i0 + 1, j1]))
        return np.where(valid, np.exp(lh), 0.)


def build_hazard_field(xs, recs) -> HazardField | None:
    good = [(x, r) for x, r in zip(xs, recs, strict=True) if r is not None and r.get("fold_V") is not None
            and len(r.get("voltage", [])) >= 2]
    if not good:
        return None
    dmax = max(abs(r["fold_V"] - np.min(r["voltage"])) if r.get("direction", "LU") == "LU"
               else abs(np.max(r["voltage"]) - r["fold_V"]) for _, r in good)
    grid = np.arange(0., dmax + 2 * C.DIST_STEP, C.DIST_STEP)
    rows, xk = [], []
    for x, r in good:
        lh = C.hazard_vs_distance(r, grid)
        if lh is not None:
            rows.append(lh); xk.append(x)
    if not rows:
        return None
    order = np.argsort(xk)
    return HazardField(np.asarray(xk, float)[order], np.asarray(rows)[order])


# --------------------------------------------------------------------------------------------
# general sweep MC
# --------------------------------------------------------------------------------------------
def _ou_chunk(n, rho, rng, zi):
    """n samples of a unit-variance AR(1)/OU series; zi carries rho*x_prev across chunks."""
    a, zf = lfilter([np.sqrt(1 - rho * rho)], [1, -rho], rng.standard_normal(n), zi=zi)
    return a, zf


def _first_event(V, dv, sign, dist, h, dt, thr):
    """First escape per row. V: (s,) sweep voltages, sign +1 up / -1 down, dist (n,s) distance to the
    fold (<= 0 means beyond the fold, NaN = no latch), h (n,s) hazard (1/s), thr (n,) Exp(1) draws.
    Returns the event voltage (NaN = none) and whether it came from the fold (deterministic atom)."""
    n, s = dist.shape
    inc = 0.5 * (h[:, 1:] + h[:, :-1]) * dt
    I = np.concatenate([np.zeros((n, 1)), np.cumsum(inc, axis=1)], axis=1)
    ev_h = I >= thr[:, None]
    with np.errstate(invalid="ignore"):
        ev_f = dist <= 0
    rows = np.arange(n)
    vh = np.full(n, np.nan); vf = np.full(n, np.nan)
    has_h = ev_h.any(1); kh = ev_h.argmax(1)
    k = kh[has_h]; r = rows[has_h]
    km = np.maximum(k - 1, 0)
    den = I[r, k] - I[r, km]
    frac = np.where(den > 0, (thr[r] - I[r, km]) / np.where(den > 0, den, 1.), 1.)
    vh[r] = V[km] + sign * np.clip(frac, 0, 1) * dv * (k > 0)
    has_f = ev_f.any(1); kf = ev_f.argmax(1)
    k = kf[has_f]; r = rows[has_f]
    km = np.maximum(k - 1, 0)
    d0 = dist[r, km]; d1 = dist[r, k]
    ok = (k > 0) & np.isfinite(d0) & (d0 > 0)
    frac = np.where(ok, d0 / np.where(ok, d0 - d1, 1.), 1.)
    vf[r] = np.where(ok, V[km] + sign * np.clip(frac, 0, 1) * dv, V[k])
    out = np.where(np.isnan(vh), vf, vh)
    both = np.isfinite(vh) & np.isfinite(vf)
    out[both] = np.minimum(vh[both], vf[both]) if sign > 0 else np.maximum(vh[both], vf[both])
    at_fold = np.isfinite(vf) & (np.isnan(vh) | ((vf <= vh) if sign > 0 else (vf >= vh)))
    return out, at_fold


def simulate_general(*, n, seed, vd_max, dv, rate, mode, x0, sigma, tau_s, sigma_e, tau_e, folds: FoldTable,
                     s_lu_e, s_ld_e, lu_haz: HazardField | None, ld_haz: HazardField | None, progress=None):
    steps = max(int(round(vd_max / dv)), 2)
    dvv = vd_max / steps
    dt = dvv / rate
    Vup = np.linspace(0, vd_max, steps + 1)
    Vdn = Vup[::-1].copy()
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)            # frozen drain-edge (action-point) state  [mc_cycles order]
    U = rng.random(n)                     # latch-up escape threshold (common random numbers)
    ZE = rng.standard_normal(n)           # frozen emitter state
    U2 = rng.random(n)                    # latch-down escape threshold
    thr_up = -np.log1p(-U); thr_dn = -np.log1p(-U2)
    rx = np.random.default_rng([int(seed), 11]); re_ = np.random.default_rng([int(seed), 12])
    rho_x = np.exp(-dt / tau_s) if mode == "evolving" else 1.
    rho_e = np.exp(-dt / tau_e) if mode == "evolving" else 1.
    zx = np.array([Z[0]]) * rho_x; ze = np.array([ZE[0]]) * rho_e
    per = 2 * (steps + 1)
    chunk = int(max(1, min(n, 600_000 // per)))
    VLU = np.full(n, np.nan); VLD = np.full(n, np.nan); XS = np.full(n, np.nan)
    ES = np.full(n, np.nan); ATOM = np.zeros(n, bool)
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk); nc = c1 - c0
        if mode == "evolving":
            xs, zx = _ou_chunk(nc * per, rho_x, rx, zx)
            es, ze = _ou_chunk(nc * per, rho_e, re_, ze)
            X = x0 + sigma * xs.reshape(nc, 2, steps + 1)
            E = sigma_e * es.reshape(nc, 2, steps + 1)
        elif mode == "frozen":
            X = np.broadcast_to((x0 + sigma * Z[c0:c1])[:, None, None], (nc, 2, 1))
            E = np.broadcast_to((sigma_e * ZE[c0:c1])[:, None, None], (nc, 2, 1))
        else:
            X = np.full((nc, 2, 1), float(x0)); E = np.zeros((nc, 2, 1))
        # --- latch-up (HRS -> LRS) ---
        fu = folds.lu(X[:, 0]) + s_lu_e * E[:, 0]
        dist = np.broadcast_to(fu - Vup[None, :], (nc, steps + 1))
        if lu_haz is not None:
            h = lu_haz(np.broadcast_to(X[:, 0], (nc, steps + 1)), dist)
        else:
            h = np.zeros((nc, steps + 1))
        vlu, atom = _first_event(Vup, dvv, +1, dist, h, dt, thr_up[c0:c1])
        # --- latch-down (LRS -> HRS), only for cycles that latched up ---
        fd = folds.ld(X[:, 1]) + s_ld_e * E[:, 1]
        dist = np.broadcast_to(Vdn[None, :] - fd, (nc, steps + 1))
        if ld_haz is not None:
            h = ld_haz(np.zeros((nc, steps + 1)), dist)
        else:
            h = np.zeros((nc, steps + 1))
        vld, _ = _first_event(Vdn, dvv, -1, dist, h, dt, thr_dn[c0:c1])
        vld[~np.isfinite(vlu)] = np.nan
        VLU[c0:c1] = vlu; VLD[c0:c1] = vld; ATOM[c0:c1] = atom
        XS[c0:c1] = X[:, 0, min(steps // 2, X.shape[2] - 1)]
        ES[c0:c1] = E[:, 0, min(steps // 2, E.shape[2] - 1)]
        if progress is not None:
            progress(c1 / n)
    return dict(V_LU=VLU, V_LD=VLD, state=XS, emitter=ES, fold_atom=ATOM, steps=steps, dv=dvv, dt=dt)


# --------------------------------------------------------------------------------------------
# traces (approximate I-V of individual cycles)
# --------------------------------------------------------------------------------------------
def _branch_current(br: dict, V):
    vd = np.asarray(br["vd"], float); idd = np.asarray(br["id"], float)
    order = np.argsort(vd)
    vd, li = vd[order], np.log(np.maximum(idd[order], 1e-30))
    return np.exp(np.interp(V, vd, li))


def trace(rec: dict, vd_max: float, vlu, vld, npts: int = 401) -> dict:
    """Up/down I-V of one cycle along the HRS/LRS branches of `rec` with that cycle's own jumps
    (branches clamped at their ends when the cycle switches beyond them)."""
    Vu = np.linspace(0, vd_max, npts); Vd = Vu[::-1]
    hrs_u = _branch_current(rec["HRS"], Vu); lrs_u = _branch_current(rec["LRS"], Vu)
    vlu = float(vlu) if vlu is not None and np.isfinite(vlu) else np.inf      # no latch-up: HRS throughout
    vld = float(vld) if vld is not None and np.isfinite(vld) else -np.inf     # no latch-down: stays LRS
    up = np.where(Vu >= vlu, lrs_u, hrs_u)
    dn = np.where(Vd > vld, lrs_u[::-1], hrs_u[::-1]) if np.isfinite(vlu) else hrs_u[::-1]
    up[Vu == 0] = 0.; dn[Vd == 0] = 0.
    return dict(up=dict(vd=Vu, id=up), down=dict(vd=Vd, id=dn))


# --------------------------------------------------------------------------------------------
# calibrated lookup engine (paper device)
# --------------------------------------------------------------------------------------------
def _gdc():
    C.engine()
    import gate_dynamic_compare as g  # noqa: E402  (path set by engine_bridge / stl_api)
    return g


def calibrated(n, seed, mode, dv, rate, amplitude_scale, tau_up_s):
    g = _gdc()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        z, rep = g.simulate(n=n, seed=seed, mode=mode, dv=dv, rate=rate, amplitude_scale=amplitude_scale,
                            tau_up_s=tau_up_s)
    return z, rep


def calibrated_states(n, seed, mode, dv, rate, amplitude_scale, tau_up_s):
    """Replay of gate_dynamic_compare.simulate's random-number stream up to the state arrays.
    Returns (states[d] of shape (n, steps+1, 2) with (phi_G, phi_E) in V, exponential draws[d])."""
    g = _gdc()
    fit = json.loads((g.H / "gate_dynamic_calibration.json").read_text())
    j0, e0, sj, se = fit["parameters"]; se *= .001; e0 *= .001
    kin = fit["kinetic_fit"]; sf = kin["fraction_slow"]; te = kin["tau_fast_s"]
    raw = np.load(g.R / "outputs/measured_idvd_parsed.npz")
    lu = (raw["VLU_low"] + raw["VLU_high"]) / 2
    k = np.linspace(-1, 1, 100); trend = np.polyval(np.polyfit(k, lu, 2), k)
    frac = np.var(trend, ddof=1) / np.var(lu, ddof=1)
    trend = -(trend - trend.mean()) / trend.std(ddof=1)
    sj *= amplitude_scale; se *= amplitude_scale
    dt = dv / rate; steps = int(round(4 / dv)); rng = np.random.default_rng(seed)
    states, draws = [], []
    for d in range(2):
        count = n * (steps + 1)
        x = g.ou(count, dt, 5. if tau_up_s is None else tau_up_s, rng)
        y = np.sqrt(sf) * g.ou(count, dt, 1000., rng) + np.sqrt(1 - sf) * g.ou(count, dt, te, rng)
        if d == 0 and mode != "stationary":
            trendn = np.interp(np.linspace(0, 99, n), np.arange(100), trend)
            x = np.sqrt(frac) * np.repeat(trendn, steps + 1) + np.sqrt(1 - frac) * x
        if mode == "frozen":
            x = np.repeat(rng.normal(size=n), steps + 1); y = np.repeat(rng.normal(size=n), steps + 1)
        if mode == "fast_only":
            x[:] = 0; y[:] = 0
        states.append(np.c_[j0 + sj * x, e0 + se * y].reshape(n, steps + 1, 2))
        draws.append(rng.exponential(size=n))
    return states, draws, dict(j0=j0, e0=e0)


def calibrated_lookup_table():
    g = _gdc()
    t, fd, hz, ci, quant = g.setup()
    return t, fd
