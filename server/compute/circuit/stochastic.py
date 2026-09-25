"""Stochastic-mode support: local-state configuration, per-run draws, the quasi-static branch
profiles (tau_rel, event rates, barrier to the saddle, fold u values), the noise bands and the
up-front feasibility estimate (expected number of time steps) used to refuse or cap expensive runs.

The kernel (mna.run_chunk) advances each cell in one of five tiers (docs/CIRCUIT_SIMULATOR.md §4):

    4 drift only   cell physically latched (u >= u_j) and ld_carrier_noise off
    5 drift only   V_DS outside the cell's noise band, and the drive does not enter the band within
                   noise_lookahead (4) relaxation times
    3 drift only   tau_rel < gauss_tau_min and tau_frac * tau_rel < noise_dt_min (too fast to resolve)
    1 event level  explicit tau-leap of the Eq. 2 increments:
                       h <= tau_frac * tau_rel            tau_rel = 1/|dF/dQ| along the circuit constraint
                       h <= max_events_per_step / R_tot   R_tot = unit + L + II-cluster event rate (1/s)
                       h <= du_max * (dQ/du) / |F|         (deterministic drift per step)
    2 Gaussian     drift-implicit, variance-corrected, h <= gauss_tau_frac * tau_rel
    (all tiers: h <= dv_max / |dv_src/dt|, h <= dt_max, breakpoints)

``estimate_steps`` mirrors these rules along the quasi-static branches.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numba import njit

from server.engine_bridge import MODEL, ct, m

ACTION_INDEX = {"gidl": 9, "local_avalanche": 23, "junction": 19, "multiplication": 20}
ACTION_UNIT = {"gidl": "V", "local_avalanche": "1", "junction": "V", "multiplication": "1"}
SECONDS_PER_STEP = 40e-6          # measured: ~35 us per accepted step (1 STL), incl. Newton + FD


@dataclass
class LocalStateConfig:
    mode: str = "none"             # none | frozen | evolving
    action: str = "gidl"
    sigma: float = 0.0
    tau_s: float = 5.0
    sigma_E_V: float = 0.0
    tau_E_s: float = 1.62

    @property
    def mode_code(self) -> int:
        return {"none": 0, "frozen": 1, "evolving": 2}[self.mode]

    @property
    def index(self) -> int:
        return ACTION_INDEX[self.action]


def parse_local_state(block: dict | None, warnings: list[str]) -> LocalStateConfig:
    b = dict(block or {})
    mode = str(b.get("mode", "none"))
    if mode not in ("none", "frozen", "evolving"):
        raise ValueError(f"stochastic.local_state.mode must be none|frozen|evolving, got {mode!r}")
    action = str(b.get("action", "gidl"))
    if action not in ACTION_INDEX:
        raise ValueError(f"stochastic.local_state.action must be one of {sorted(ACTION_INDEX)}, got {action!r}")
    cfg = LocalStateConfig(mode=mode, action=action,
                           sigma=float(b.get("sigma", 0.1534) or 0.0), tau_s=float(b.get("tau_s", 5.0) or 5.0),
                           sigma_E_V=float(b.get("sigma_E_V", 0.0) or 0.0), tau_E_s=float(b.get("tau_E_s", 1.62) or 1.62))
    if cfg.sigma < 0 or cfg.sigma_E_V < 0:
        raise ValueError("local-state SDs must be >= 0")
    if cfg.tau_s <= 0 or cfg.tau_E_s <= 0:
        raise ValueError("local-state correlation times must be > 0")
    if mode != "none" and b.get("acquisition_trend"):
        warnings.append("local_state.acquisition_trend is specific to the calibrated reference-record lookup engine "
                        "and is not applied in the circuit simulator")
    return cfg


def draw_local_states(cfg: LocalStateConfig, n_stl: int, seed: int, run: int) -> np.ndarray:
    """Initial (frozen: whole-run) local-state deviations per STL: columns (action, emitter)."""
    out = np.zeros((n_stl, 2))
    if cfg.mode == "none":
        return out
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 7349, int(run)])
    out[:, 0] = cfg.sigma * rng.standard_normal(n_stl)
    out[:, 1] = cfg.sigma_E_V * rng.standard_normal(n_stl)
    return out


# ---- feasibility -----------------------------------------------------------------------
def _state_charge(z, u, p):
    psi = u - m.VT * np.log1p(z[10])
    return m.COX_F * (psi - p[11]) + (z[13] - m.COX_F * u) + m.Q * MODEL.na * m.AREA_CM2 * z[11]


def classify_checked(p: np.ndarray, grid: int):
    """MODEL.classify with the deterministic engine's locus-gap check (``deterministic.classify_checked``):
    a fold whose parabola was fitted across an untraced gap of the locus (V_G above about -0.2...0 V,
    the channel-on regime) is rejected.  Returns (classify result or None, gap)."""
    from server.compute.deterministic import classify_checked as _cc   # worker-only import
    return _cc(np.asarray(p, float), int(grid))


@lru_cache(maxsize=256)
def _fold_u_cached(key: bytes, grid: int) -> tuple[float, float]:
    p = np.frombuffer(key, dtype=float)
    cl, _gap = classify_checked(p, grid)
    if cl is None:
        return (np.inf, np.inf)
    b, i, j, _ = cl
    return (float(b[i, 17]), float(b[j, 17]))


def fold_u(p: np.ndarray, grid: int = 301) -> tuple[float, float]:
    """u at the latch-up fold (u_i: end of the HRS) and at the latch-down fold (u_j: start of the LRS)
    of the quasi-static branch, which is parameterised by u: HRS u < u_i, unstable branch
    u_i < u < u_j, LRS u > u_j.  (inf, inf) without a latch window (the cell can never latch)."""
    return _fold_u_cached(np.ascontiguousarray(p, dtype=float).tobytes(), int(grid))


def branch_profile(p: np.ndarray, grid: int = 301) -> dict | None:
    """Quasi-static branches of the device and, along the HRS and LRS: tau_rel (fixed V_D), the
    total event rate, and z = |Q - Q_saddle| / SD(Q) (barrier to the saddle on the unstable branch in
    units of the stationary charge fluctuation, SD^2 = D tau/2, D = q^2 (unit + L + II M2/M1))."""
    p = np.asarray(p, float)
    args = (p, MODEL.na, MODEL.vbi, MODEL.rg, MODEL.fg, MODEL.table)
    cl, gap = classify_checked(p, grid)
    rv, pmf = np.asarray(ct.cf.rv, float), np.asarray(ct.cf.pmf, float)
    ks = np.arange(pmf.shape[1])
    if cl is None:
        b = MODEL.branch(p, m.state_grid(grid))
        folds = (np.nan, np.nan)
        parts = [("HRS", b)]
        Vs = Qs = None
    else:
        b, i, j, fold = cl
        folds = (float(fold[0]), float(fold[1]))
        parts = [("HRS", b[:i + 1]), ("LRS", b[j:])]
        U = b[i:j + 1]
        Vs, Qs = [], []
        for row in U:
            z = m.components(row[17], row[18], *args)
            if np.isfinite(z[0]):
                Vs.append(row[0])
                Qs.append(_state_charge(z, row[17], p))
        o = np.argsort(Vs)
        Vs, Qs = np.asarray(Vs)[o], np.asarray(Qs)[o]
    out = dict(folds=folds, latch=cl is not None, gap=bool(gap),
               u_fold=(float(b[i, 17]), float(b[j, 17])) if cl is not None else (np.inf, np.inf))
    # quasi-static I-V of the stable branches (sorted by V_D; used by the quasi-static drive walk of
    # current-biased / high-impedance cells, ``oscillator.qs_drive``)
    iv = {}
    for name, part in parts:
        ok = np.isfinite(part[:, 0]) & np.isfinite(part[:, 1]) & (part[:, 1] > 0)
        o = np.argsort(part[ok, 0], kind="stable")
        Vv, Iv = part[ok, 0][o], part[ok, 1][o]
        keep = np.r_[True, np.diff(Vv) > 1e-9] if len(Vv) else np.zeros(0, bool)
        iv[name] = (Vv[keep], Iv[keep])
    out["iv"] = iv
    if cl is not None:
        # branch currents at the folds: the reporting thresholds must lie between the branches there
        H, L_ = b[:i + 1], b[j:]

        def _logI(part, v):
            o = np.argsort(part[:, 0])
            return float(np.exp(np.interp(v, part[o, 0], np.log(np.maximum(part[o, 1], 1e-300)))))
        out["fold_I"] = dict(hrs_at_lu=float(b[i, 1]), lrs_at_ld=float(b[j, 1]),
                             lrs_at_lu=_logI(L_, folds[0]), hrs_at_ld=_logI(H, folds[1]))
    for name, part in parts:
        steady = np.abs(part[:, 2]) <= 1e-4 * np.maximum(np.abs(part[:, 1]), 1e-18) + 1e-22
        part = part[steady] if steady.sum() >= 2 else part
        sel = np.unique(np.linspace(0, len(part) - 1, min(len(part), 90)).astype(int))
        V, T, R, Z = [], [], [], []
        for row in part[sel]:
            vd, u, r = row[0], row[17], row[18]
            if not np.isfinite(vd) or vd > 8.5:
                continue
            du = 1e-5
            z0 = m.components(u, r, *args)
            z1 = m.components(u + du, r, *args)
            z2 = m.components(u, r + du, *args)
            if not (np.isfinite(z0[0]) and np.isfinite(z1[0]) and np.isfinite(z2[0])):
                continue
            q0, q1, q2 = _state_charge(z0, u, p), _state_charge(z1, u + du, p), _state_charge(z2, u, p)
            dQu, dQr = (q1 - q0) / du, (q2 - q0) / du
            dFu, dFr = (z1[2] - z0[2]) / du, (z2[2] - z0[2]) / du
            dVu, dVr = (z1[0] - z0[0]) / du, (z2[0] - z0[0]) / du
            drdu = -dVu / dVr if dVr != 0 else 0.0
            dQ = dQu + dQr * drdu
            dF = dFu + dFr * drdu
            tau = abs(dQ / dF) if dF != 0 else 1e30
            G = (z0[1] - z0[3] - z0[16]) / m.Q
            L = (z0[5] + z0[6] + z0[7]) / m.Q
            unit = (z0[8] + z0[9] + z0[18]) / m.Q
            pk = np.array([np.interp(r, rv, pmf[:, k]) for k in range(1, pmf.shape[1])])
            m1, m2 = float(pk @ ks[1:]), float(pk @ ks[1:] ** 2)
            D = m.Q ** 2 * (unit + L + max(G - unit, 0.0) * (m2 / m1 if m1 > 0 else 1.0))
            zb = np.inf
            if Vs is not None and len(Vs) > 1 and Vs[0] <= vd <= Vs[-1] and tau < 1e29:
                zb = abs(q0 - np.interp(vd, Vs, Qs)) / np.sqrt(D * tau / 2.0)
            V.append(vd)
            T.append(tau)
            R.append(G + L)
            Z.append(zb)
        if not V:
            V, T, R, Z = [0.0], [1.0], [0.0], [np.inf]
        order = np.argsort(V)
        out[name] = dict(V=np.asarray(V)[order], tau=np.asarray(T)[order], rate=np.asarray(R)[order],
                         z=np.asarray(Z)[order])
    if "LRS" not in out:
        out["LRS"] = out["HRS"]
    return out


def _interp_log(v, V, Y):
    return np.exp(np.interp(v, V, np.log(np.maximum(Y, 1e-300))))


# empirical escape hazard of the compound carrier noise vs the barrier z (in stationary SDs), from the
# z_max calibration table (paper V_G = -2 V dark: 3.3 /s at z = 7.5, 2.1e-3 /s at 10.5, 1.1e-7 /s at 13.5)
_LAM0, _ZREF, _ZSLOPE = 3.3, 7.5, 1.2
# per-event step costs (calibrated on the benches at reltol 1e-3, LTE_u = 30 uV)
_STEPS_PER_TRANSITION = 250.0
_STEPS_PER_CORNER = 60.0
_STEPS_PER_BREAKPOINT = 6.0


@njit(cache=True)
def _estimate_loop(vm, dtm, slope, tauH, rateH, zH, tauL, rateL, zL, V_LU, V_LD, latch, noise, ld_noise, escape,
                   dt_max, dv_max, tau_frac, n_ev, noise_dt_min, gauss_tau_min, gauss_tau_frac, w0, w1, w2, w3, n_look):
    lrs = False
    H = 0.0
    steps = 0.0
    trans = 0
    for a in range(vm.shape[0]):
        v = vm[a]
        dt = dtm[a]
        s = slope[a]
        if latch:
            if not lrs:
                if v > V_LU or (escape and H >= 0.6931471805599453):
                    lrs = True
                    trans += 1
                    H = 0.0
            elif v < V_LD or (escape and ld_noise and H >= 0.6931471805599453):
                lrs = False
                trans += 1
                H = 0.0
        h = dt_max
        if s > 0:
            h = min(h, dv_max / s)
        if noise and (ld_noise or not lrs):
            tau = tauL[a] if lrs else tauH[a]
            rate = rateL[a] if lrs else rateH[a]
            z = zL[a] if lrs else zH[a]
            ext = n_look * tau * s                  # the kernel's look-ahead into the band
            inb = (w2 - ext <= v <= w3 + ext) if lrs else (w0 - ext <= v <= w1 + ext)
            if inb:
                if tau_frac * tau >= noise_dt_min:
                    h = min(h, tau_frac * tau)
                    if rate > 0:
                        h = min(h, n_ev / rate)
                elif tau >= gauss_tau_min:
                    h = min(h, gauss_tau_frac * tau)
                if escape:
                    H += min(_LAM0 * 10.0 ** (-_ZSLOPE * (z - _ZREF)), 1e12) * dt
        steps += dt / max(h, 1e-18)
    return steps, trans


def estimate_steps(wave_t: np.ndarray, wave_v: np.ndarray, t_end: float, profile: dict, stochastic: bool,
                   carrier: bool, dt_max: float, dv_max: float, tau_frac: float, n_ev: float,
                   noise_dt_min: float, n_breakpoints: int, ld_noise: bool = False,
                   gauss_tau_min: float = 2e-9, gauss_tau_frac: float = 0.5,
                   window: tuple = (-np.inf, np.inf, -np.inf, np.inf), n_look: float = 4.0,
                   corner_weights: np.ndarray | None = None, transition_steps: float | None = None) -> float:
    """Expected number of accepted steps for one run (drain voltage ~ source voltage).

    The drive is sampled on a voltage-resolved grid (<= 2 mV per point on ramps, one point per flat
    segment); along it a hysteretic HRS/LRS state is walked with the step rules of the kernel (dt_max,
    dv_max/|dv/dt|, the noise tiers inside the noise bands entered n_look relaxation times early).
    With carrier noise the HRS -> LRS switch happens at the noise-induced escape (median of the
    empirical compound-noise hazard lambda(z)) or at the fold, whichever comes first (likewise for the
    LRS with ld_carrier_noise).  Fixed costs per latch transition, waveform corner and other breakpoint
    cover the resolved switching transients.  Typically within about +-30 % of the actual count.
    ``corner_weights`` (one per interior waveform point, in [0, 1]; custom circuits): how sharp each
    corner is (1 = a pulse corner, ~0 = a sampled smooth waveform); default: every point is a corner.
    ``transition_steps``: fixed cost per latch transition (default 250, calibrated on the benches; the
    quasi-static oscillator drive of custom circuits uses its own calibration)."""
    V_LU, V_LD = profile["folds"]
    latch = bool(profile["latch"]) and np.isfinite(V_LU) and np.isfinite(V_LD)
    wt = np.asarray(wave_t, float)
    wv = np.asarray(wave_v, float)
    if wt[-1] < t_end:
        wt, wv = np.r_[wt, t_end], np.r_[wv, wv[-1]]
    dT, dV = np.diff(wt), np.diff(wv)
    ok = dT > 0
    dT, dV, t0, v0 = dT[ok], dV[ok], wt[:-1][ok], wv[:-1][ok]
    if len(dT) == 0:
        return float(_STEPS_PER_BREAKPOINT * n_breakpoints)
    dv_res = max(2e-3, float(np.abs(dV).sum()) / 3e5)
    n = np.where(np.abs(dV) > 1e-12, np.clip(np.ceil(np.abs(dV) / dv_res), 8, 4000), 1).astype(np.int64)
    seg = np.repeat(np.arange(len(dT)), n)
    k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    frac = (k + 0.5) / n[seg]
    tm = t0[seg] + dT[seg] * frac
    keep = tm < t_end
    vm = (v0[seg] + dV[seg] * frac)[keep]
    dtm = (dT[seg] / n[seg])[keep]
    slope = (np.abs(dV[seg]) / dT[seg])[keep]
    noise = bool(stochastic and carrier)
    arr = {}
    for name in ("HRS", "LRS"):
        pr = profile[name]
        if noise:
            arr[name] = (_interp_log(vm, pr["V"], pr["tau"]), _interp_log(vm, pr["V"], pr["rate"] + 1e-300),
                         np.interp(vm, pr["V"], np.minimum(pr["z"], 1e3)))
        else:
            z = np.zeros_like(vm)
            arr[name] = (z, z, z)
    w = [float(x) for x in window]
    steps, trans = _estimate_loop(vm, dtm, slope, *arr["HRS"], *arr["LRS"],
                                  float(V_LU) if latch else np.inf, float(V_LD) if latch else -np.inf, latch, noise,
                                  bool(ld_noise), noise and latch, float(dt_max), float(dv_max), float(tau_frac),
                                  float(n_ev), float(noise_dt_min), float(gauss_tau_min), float(gauss_tau_frac),
                                  w[0], w[1], w[2], w[3], float(n_look))
    corners = max(len(wave_t) - 2, 0)
    if corner_weights is not None:
        corners = float(np.sum(corner_weights))
    other = max(int(n_breakpoints) - corners, 0)
    per_tr = _STEPS_PER_TRANSITION if transition_steps is None else float(transition_steps)
    return float(steps + per_tr * trans + _STEPS_PER_CORNER * corners + _STEPS_PER_BREAKPOINT * other)


def noise_bands(profile: dict, ls_cfg: "LocalStateConfig | None", stochastic: bool,
                z_max: float = 12.0) -> tuple[float, float, float, float]:
    """V_DS bands where carrier noise is resolved: (lu_lo, lu_hi, ld_lo, ld_hi).

    Unlatched cell: from the HRS voltage where the barrier to the saddle is z_max stationary SDs
    upward, with no upper limit (lu_hi = +inf: an unlatched cell beyond V_LU is in the post-fold
    passage, where the noise shapes V_LU on fast ramps and the delay of supra-fold pulses).
    Latched cell: from below (ld_lo = -inf: post-fold passage below V_LD) up to the LRS voltage
    with z = z_max.  The z-edges are widened by 4 sigma of the local-state fold shifts (bounds
    |dV_LU/dphi_G| <= 1, |dV_LU/dphi_E| <= 5, |dV_LD/dphi_G| <= 0.05, |dV_LD/dphi_E| <= 50 V/V);
    unknown sensitivities (other actions, sigma_E > 2 mV) -> whole axis.  No latch window: no band
    (drift only, the noise cannot cause a transition) unless local states may create one (whole
    axis).  The kernel additionally resolves the noise noise_lookahead relaxation times before the
    drive enters a band (fast ramps and edges)."""
    V_LU, V_LD = profile["folds"]
    full = (-np.inf, np.inf, -np.inf, np.inf)
    ls_on = stochastic and ls_cfg is not None and ls_cfg.mode != "none"
    if not profile["latch"] or not np.isfinite(V_LU) or not np.isfinite(V_LD):
        return full if ls_on else (np.inf, -np.inf, np.inf, -np.inf)
    ex_lu = ex_ld = 0.0
    if ls_on:
        if ls_cfg.action != "gidl" or ls_cfg.sigma_E_V > 0.002:
            return full
        # fold sensitivities of the paper model (MODEL_SPEC §4 / fold tables): dV_LU/dphi_G ~ -0.8 V/V,
        # dV_LD/dphi_G ~ 0, dV_LD/dphi_E ~ -41 V/V, dV_LU/dphi_E small; bounds used: 1, 0.05, 50, 5
        ex_lu = 4.0 * (ls_cfg.sigma * 1.0 + ls_cfg.sigma_E_V * 5.0)
        ex_ld = 4.0 * (ls_cfg.sigma * 0.05 + ls_cfg.sigma_E_V * 50.0)
    H, L = profile["HRS"], profile["LRS"]
    zh = H["z"] < z_max
    lu_lo = float(H["V"][zh].min()) if zh.any() else V_LU - 0.05
    zl = L["z"] < z_max
    ld_hi = float(L["V"][zl].max()) if zl.any() else V_LD + 0.05
    return (lu_lo - ex_lu, np.inf, -np.inf, ld_hi + ex_ld)
