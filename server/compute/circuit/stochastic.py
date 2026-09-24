"""Stochastic-mode support: local-state configuration, per-run draws, and the up-front
feasibility estimate (expected number of time steps) used to refuse or cap expensive runs.

Step-size rule of the event-level (explicit tau-leap) integration, mirrored here for the estimate:

    h <= tau_frac * tau_rel           tau_rel = (dQ/du) / |dF/du|  along the circuit constraint
    h <= max_events_per_step / R_tot  R_tot = unit + L + II-cluster event rate  (1/s)
    h <= du_max * (dQ/du) / |F|        (deterministic drift per step)
    h <= dv_max / |dv_src/dt|,  h <= dt_max,  breakpoints

Where tau_frac * tau_rel < noise_dt_min (fast-relaxing states: the LRS at high current and the
end of the switching transient) the element is integrated drift-only (implicit BE), which keeps
the LRS affordable; the time spent there is reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
        warnings.append("local_state.acquisition_trend is specific to the calibrated paper-record lookup engine "
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


def branch_profile(p: np.ndarray, grid: int = 301) -> dict | None:
    """Quasi-static branches of the device and, along the HRS and LRS: tau_rel (fixed V_D), the
    total event rate, and z = |Q - Q_saddle| / SD(Q) (barrier to the saddle on the unstable branch in
    units of the stationary charge fluctuation, SD^2 = D tau/2, D = q^2 (unit + L + II M2/M1))."""
    p = np.asarray(p, float)
    args = (p, MODEL.na, MODEL.vbi, MODEL.rg, MODEL.fg, MODEL.table)
    cl = MODEL.classify(p, m.state_grid(grid))
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
    out = dict(folds=folds, latch=cl is not None)
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


def estimate_steps(wave_t: np.ndarray, wave_v: np.ndarray, t_end: float, profile: dict, stochastic: bool,
                   carrier: bool, dt_max: float, dv_max: float, tau_frac: float, n_ev: float,
                   noise_dt_min: float, n_breakpoints: int, ld_noise: bool = False,
                   gauss_tau_min: float = 2e-9, gauss_tau_frac: float = 0.5,
                   window: tuple = (-np.inf, np.inf, -np.inf, np.inf)) -> float:
    """Rough number of accepted steps for one run (drain voltage ~ source voltage)."""
    V_LU, V_LD = profile["folds"]
    latch = profile["latch"]
    # fine time grid: 40 points per PWL segment
    pts = [np.linspace(a, b, 41)[:-1] for a, b in zip(wave_t[:-1], wave_t[1:]) if b > a]
    tg = np.unique(np.r_[np.concatenate(pts) if pts else np.zeros(0), t_end, 0.0])
    tg = tg[tg <= t_end]
    vg = np.interp(tg, wave_t, wave_v)
    state_lrs = False
    steps = 0.0
    transitions = 0
    for a in range(len(tg) - 1):
        dt = tg[a + 1] - tg[a]
        v = 0.5 * (vg[a] + vg[a + 1])
        slope = abs(vg[a + 1] - vg[a]) / dt if dt > 0 else 0.0
        if latch:
            if not state_lrs and v > V_LU:
                state_lrs = True
                transitions += 1
            elif state_lrs and v < V_LD:
                state_lrs = False
                transitions += 1
        h = dt_max
        if slope > 0:
            h = min(h, dv_max / slope)
        if stochastic and carrier and (ld_noise or not state_lrs):
            prof = profile["LRS" if state_lrs else "HRS"]
            tau = _interp_log(v, prof["V"], prof["tau"])
            rate = _interp_log(v, prof["V"], prof["rate"] + 1e-300)
            if not ((window[2] <= v <= window[3]) if state_lrs else (window[0] <= v <= window[1])):
                pass                                   # outside the noise band: drift only
            elif tau_frac * tau >= noise_dt_min:
                h = min(h, tau_frac * tau)
                if rate > 0:
                    h = min(h, n_ev / rate)
            elif tau >= gauss_tau_min:
                h = min(h, gauss_tau_frac * tau)
        steps += dt / max(h, 1e-18)
    steps += 250 * transitions + 6 * n_breakpoints
    return float(steps)


def noise_bands(profile: dict, ls_cfg: "LocalStateConfig | None", stochastic: bool, z_max: float = 12.0,
                margin: float = 0.25) -> tuple[float, float, float, float]:
    """V_DS bands where carrier noise can cause escape: (lu_lo, lu_hi) for an unlatched cell (HRS
    with barrier z < z_max ... V_LU + margin) and (ld_lo, ld_hi) for a latched cell (V_LD - margin ...
    LRS with z < z_max).  Widened by 4 sigma of the local-state fold shifts (bounds |dV_LU/dphi_G| <= 1,
    |dV_LU/dphi_E| <= 5, |dV_LD/dphi_G| <= 0.05, |dV_LD/dphi_E| <= 50 V/V); unknown sensitivities (other
    actions, sigma_E > 2 mV) -> whole axis."""
    V_LU, V_LD = profile["folds"]
    full = (-np.inf, np.inf, -np.inf, np.inf)
    if not profile["latch"] or not np.isfinite(V_LU) or not np.isfinite(V_LD):
        return full
    ex_lu = ex_ld = 0.0
    if stochastic and ls_cfg is not None and ls_cfg.mode != "none":
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
    return (lu_lo - ex_lu, V_LU + margin + ex_lu, V_LD - margin - ex_ld, ld_hi + ex_ld)
