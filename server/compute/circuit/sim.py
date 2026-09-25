"""Python driver around the numba MNA kernel: one transient run = DC operating point + chunked
integration (progress / cancellation between chunks), buffers flushed to Python lists."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from server.engine_bridge import MODEL, ct
from server.geometry_model import PACK_SIZE, pack_p

from . import mna as K

REC_CAP = 20000
EV_CAP = 4096
CHUNK_STEPS = 4000
G_HOLD = 1e6          # S, conductance that holds a capacitor at 0 V in the initial state 'zero'


@dataclass
class SolverConfig:
    method: int = 0                 # 0 BE, 1 TRAP
    stochastic: bool = False
    carrier: bool = False
    max_steps: int = 1_000_000
    dt_min: float = 1e-15
    dt_max: float = 1e-3
    reltol: float = 1e-3
    tau_frac: float = 0.05
    max_events_per_step: float = 200.0
    noise_dt_min: float = 2e-9
    gauss_threshold: float = 100.0
    i_threshold: float = 1e-8
    i_threshold_down: float = 1e-9
    gauss_tau_min: float = 2e-9
    gauss_tau_frac: float = 0.5
    i_floor: float = 1e-15
    dt_rec: float = 1e-3
    dv_rec: float = 0.02
    dlni_rec: float = 0.3
    ls_mode: int = 0                # 0 none, 1 frozen, 2 evolving (OU)
    ls_idx: int = 9
    ls_sigma: float = 0.0
    ls_tau: float = 1.0
    lsE_sigma: float = 0.0
    lsE_tau: float = 1.0
    h_init: float = 1e-9
    newton_tol: float = 1e-7
    ld_noise: bool = False
    noise_lookahead: float = 4.0    # stochastic: noise on n relaxation times before the drive enters a band
    lte_v: float = 0.0              # node-voltage LTE control (custom circuits): relative factor, 0 = off
    lte_v_abs: float = 0.0          # absolute part of the node-voltage LTE tolerance (V)

    def arrays(self, net: dict, t_end: float, main_wave: int, main_stl: int = 0):
        ci = np.zeros(K.N_CI, np.int64)
        ci[K.CI_NN] = net["n_nodes"]
        ci[K.CI_NV] = net["nV"]
        ci[K.CI_NR] = net["nR"]
        ci[K.CI_NC] = net["nC"]
        ci[K.CI_NI] = net["nI"]
        ci[K.CI_NS] = net["nS"]
        ci[K.CI_METHOD] = self.method
        ci[K.CI_STOCH] = int(self.stochastic)
        ci[K.CI_CARRIER] = int(self.carrier)
        ci[K.CI_MAXSTEPS] = int(self.max_steps)
        ci[K.CI_MAINW] = main_wave
        ci[K.CI_LSMODE] = self.ls_mode
        ci[K.CI_LSIDX] = self.ls_idx
        ci[K.CI_CHUNK] = CHUNK_STEPS
        ci[K.CI_MAIN_STL] = main_stl
        ci[K.CI_LDNOISE] = int(self.ld_noise)
        sc = self.reltol / 1e-3
        cf = np.zeros(K.N_CF)
        cf[K.CF_TEND] = t_end
        cf[K.CF_DTMIN] = self.dt_min
        cf[K.CF_DTMAX] = self.dt_max
        cf[K.CF_DUMAX] = float(np.clip(0.01 * sc, 1e-3, 0.05))
        cf[K.CF_DLNIMAX] = float(np.clip(0.2 * sc, 0.02, 1.0))
        cf[K.CF_DVMAX] = float(np.clip(0.02 * sc, 1e-3, 0.2))
        # LTE_u 30 uV at reltol 1e-3: converges the slow-passage lag of BE to ~1 % (1 mV left a 6-13 % error)
        cf[K.CF_LTEU] = float(np.clip(3e-5 * sc, 1e-6, 0.02))
        cf[K.CF_TAUFRAC] = self.tau_frac
        cf[K.CF_NEVMAX] = self.max_events_per_step
        cf[K.CF_HNOISEMIN] = self.noise_dt_min
        cf[K.CF_GAUSS] = self.gauss_threshold
        cf[K.CF_ITH] = self.i_threshold
        cf[K.CF_IFLOOR] = self.i_floor
        cf[K.CF_DTREC] = self.dt_rec
        cf[K.CF_DVREC] = self.dv_rec
        cf[K.CF_DLNIREC] = self.dlni_rec
        cf[K.CF_LSSIG] = self.ls_sigma
        cf[K.CF_LSTAU] = self.ls_tau
        cf[K.CF_LSESIG] = self.lsE_sigma
        cf[K.CF_LSETAU] = self.lsE_tau
        cf[K.CF_HINIT] = self.h_init
        cf[K.CF_NEWTOL] = self.newton_tol
        cf[K.CF_ITHDN] = self.i_threshold_down
        cf[K.CF_GTAUMIN] = self.gauss_tau_min
        cf[K.CF_GTAUFRAC] = self.gauss_tau_frac
        cf[K.CF_NLOOK] = self.noise_lookahead
        cf[K.CF_LTEV] = self.lte_v
        cf[K.CF_LTEVABS] = self.lte_v_abs
        return ci, cf


@dataclass
class RunOutput:
    rec: np.ndarray                 # (n, 1 + (N-1) + nV + 7 nS + nC) decimated recording (empty with rec_sink)
    events: np.ndarray              # (n, 6) kind, stl, t, v_ds, v_src, I
    samples: np.ndarray             # (n_samp, 1 + 3 nS) t, (I_D, v_ds, reported latch flag) per STL; NaN = not reached
    steps: int
    rejected: int
    newton_iters: int
    status: int
    t_reached: float
    unresolved_steps: int
    t_unresolved: float
    gauss_steps: int
    t_gauss: float
    t_lrs_drift: float
    t_band_drift: float
    min_u: float
    min_r: float
    t_neg_u: float
    t_neg_r: float
    trap_be: int
    runtime_s: float
    hrs_crossings: int = 0          # I_D up-crossings of i_threshold with the body on the HRS (not latch-up)
    warnings: list[str] = field(default_factory=list)
    diag: list[int] = field(default_factory=list)
    initial: str = "op"             # initial state used: "op" (DC operating point) or "zero" (discharged capacitors)
    ndr_op: list = field(default_factory=list)   # 'auto': cells whose DC operating point was on the NDR branch
                                                 # [(cell, V_DS, I_D)] (the run then started from 'zero')


def _vgs_moves(net: dict, k: int) -> bool:
    if int(net["sS"][k]) != 0:
        return True
    g = int(net["sG"][k])
    if g == 0:
        return False
    for e in range(int(net["nV"])):
        a, b, w = int(net["vA"][e]), int(net["vB"][e]), int(net["vW"][e])
        if w >= 0 and ((a == g and b == 0) or (b == g and a == 0)):
            vals = net["wv"][net["woff"][w]:net["woff"][w + 1]]
            return bool(np.ptp(vals) > 0)
    return True


def _tables():
    return (float(MODEL.na), float(MODEL.vbi), np.ascontiguousarray(MODEL.rg, dtype=np.float64),
            np.ascontiguousarray(MODEL.fg, dtype=np.float64), np.ascontiguousarray(MODEL.table, dtype=np.float64),
            np.ascontiguousarray(ct.cf.rv, dtype=np.float64), np.ascontiguousarray(ct.cf.pmf, dtype=np.float64))


def simulate(net: dict, cfg: SolverConfig, P: np.ndarray, t_end: float, main_wave: int, seed: int,
             ls_init: np.ndarray | None = None, progress=None, window: np.ndarray | None = None,
             rec_sink=None, initial: str = "op", cap_hold: np.ndarray | None = None) -> RunOutput:
    """One transient run.  ``progress(fraction_of_run)`` is called between chunks (it may raise
    JobCancelled).  ``window`` (n_STL x 4, 6, 9 or 15 columns, see ``mna.W_*``): noise bands (lu_lo,
    lu_hi, ld_lo, ld_hi), optionally the fold u values (u_i, u_j) of the physical latch state (missing:
    computed from each element's quasi-static branch, ``stochastic.fold_u``), the noise look-ahead drive
    (wave, gain, mode; default: ``main_wave``, 1, 0 = the benches) and the per-cell local-state
    configuration (default: the global ``cfg`` values for every cell).
    ``rec_sink(rows, events)``: when given, every flushed block of recorded rows (and the latch events
    of that block) is passed to it instead of being accumulated (``RunOutput.rec`` is then empty).
    ``initial``: "op" = DC operating point (the benches); "zero" = discharged capacitors (every capacitor
    with ``cap_hold`` > 0 — default all — held at 0 V by G_HOLD while the bodies relax; SPICE UIC with IC = 0);
    "auto" = "op" unless the operating point puts a cell on the unstable (negative-resistance) branch of its
    quasi-static curve (u_i < u < u_j: an equilibrium that exists only because the cell is current-biased),
    then "zero" (``RunOutput.ndr_op`` names the cells)."""
    tic = time.perf_counter()
    na, vbi, rg, fg, table, rv, pmf = _tables()
    ci, cf = cfg.arrays(net, t_end, main_wave)
    nn, nv, ns, nc = net["n_nodes"], net["nV"], net["nS"], net["nC"]
    n = nn - 1 + nv + 2 * ns
    P = np.ascontiguousarray(P, dtype=np.float64).copy()
    # Bench callers can supply raw geometry vectors, whereas custom
    # netlists already carry packed per-device electrostatic lookup tables.
    if ns and 26 < P.shape[1] < PACK_SIZE:
        P = np.vstack([pack_p(p, force=True) for p in P])
    Pbase = P.copy()
    x = np.zeros(n)
    for k in range(ns):
        ku = nn - 1 + nv + 2 * k
        x[ku] = 0.0
        x[ku + 1] = 0.0
    xp = x.copy()
    ls = np.zeros((ns, 2)) if ls_init is None else np.ascontiguousarray(ls_init, dtype=np.float64).reshape(ns, 2).copy()
    ss = np.zeros((ns, K.N_SS))
    part = np.zeros((ns, K.N_PART))
    # cells whose V_GS can move (source not grounded, or gate not held by a constant voltage source to ground) get
    # the d/dV_GS columns in their Jacobian; the benches (grounded source, DC gate source) do not
    for k in range(ns):
        part[k, K.P_VGSJ] = 1.0 if _vgs_moves(net, k) else 0.0
    sens = np.zeros((ns, n))
    cv = np.zeros(nc)
    cI = np.zeros(nc)
    sf = np.zeros(K.N_SF)
    si = np.zeros(K.N_SI, np.int64)
    W = 1 + (nn - 1) + nv + 7 * ns + nc
    rec = np.zeros((REC_CAP, W))
    evb = np.zeros((EV_CAP, K.N_EVC))
    nsamp = len(net["samp"])
    sbuf = np.full((max(nsamp, 1), 1 + K.N_SAMPC * ns), np.nan)
    a = net
    win = np.empty((ns, K.N_WIN))
    win[:, 0:4:2] = -np.inf
    win[:, 1:4:2] = np.inf
    wk = None if window is None else np.asarray(window, float).reshape(ns, -1)
    if wk is not None:
        win[:, :4] = wk[:, :4]
    if wk is not None and wk.shape[1] >= 6:
        win[:, 4:6] = wk[:, 4:6]
    else:
        from .stochastic import fold_u
        for k in range(ns):
            win[k, K.W_UI], win[k, K.W_UJ] = fold_u(P[k])
    win[:, K.W_LAW] = main_wave
    win[:, K.W_LAG] = 1.0
    win[:, K.W_LAMODE] = 0.0
    if wk is not None and wk.shape[1] >= 9:
        win[:, 6:9] = wk[:, 6:9]
    win[:, K.W_LSMODE] = cfg.ls_mode
    win[:, K.W_LSIDX] = cfg.ls_idx
    win[:, K.W_LSSIG] = cfg.ls_sigma
    win[:, K.W_LSTAU] = cfg.ls_tau
    win[:, K.W_LSESIG] = cfg.lsE_sigma
    win[:, K.W_LSETAU] = cfg.lsE_tau
    if wk is not None and wk.shape[1] >= K.N_WIN:
        win[:, 9:K.N_WIN] = wk[:, 9:K.N_WIN]
    K.seed_rng(int(seed) % (2 ** 32 - 1))
    # local states enter p before the DC point
    Pdc = P.copy()
    if cfg.stochastic and cfg.ls_mode > 0:
        for k in range(ns):
            if win[k, K.W_LSMODE] > 0.5:
                ix = int(win[k, K.W_LSIDX])
                Pdc[k, ix] = Pbase[k, ix] + ls[k, 0]
                Pdc[k, 10] = Pbase[k, 10] + ls[k, 1]
    if initial not in ("op", "zero", "auto"):
        raise ValueError(f"initial state must be 'op', 'zero' or 'auto', got {initial!r}")
    hold_z = np.full(nc, G_HOLD) if cap_hold is None else np.ascontiguousarray(cap_hold, dtype=np.float64).reshape(nc)
    cmp = np.ascontiguousarray(net.get("cmp", np.zeros((0, K.N_CMPC))), dtype=np.float64).copy()
    basic = np.ascontiguousarray(net.get("basic", np.zeros((0, 12))), dtype=np.float64)
    x_start = x.copy()

    def _dc(hold):
        return K.dc_op(x, ci, cf, a["rA"], a["rB"], a["rG"], a["cA"], a["cB"], a["cC"], a["vA"], a["vB"], a["vW"],
                       a["iA"], a["iB"], a["iW"], a["sD"], a["sG"], a["sS"], a["sW"], a["wt"], a["wv"], a["woff"],
                       Pdc, Pbase if not (cfg.stochastic and cfg.ls_mode > 0) else Pdc, na, vbi, rg, fg, table,
                       np.zeros((ns, K.N_EV)), part, 0.0, hold, cmp, basic)

    used = "zero" if initial == "zero" else "op"
    ok = _dc(hold_z if used == "zero" else np.zeros(nc))
    ndr_op = []
    if not ok and initial == "auto" and nc:
        # no operating point from the empty body (e.g. a current source forcing more current than the HRS
        # can carry): start from discharged capacitors
        x[:] = x_start
        used = "zero"
        ndr_op.append((-1, float("nan"), float("nan")))
        ok = _dc(hold_z)
    elif ok and initial == "auto" and ns:
        ev_ = np.zeros(K.N_EV)
        from .element import stl_eval
        for k in range(ns):
            ku = nn - 1 + nv + 2 * k
            if win[k, K.W_UI] < x[ku] < win[k, K.W_UJ]:
                pk = Pdc[k].copy()
                pk[11] = (x[a["sG"][k] - 1] if a["sG"][k] > 0 else 0.0) - (x[a["sS"][k] - 1] if a["sS"][k] > 0 else 0.0)
                stl_eval(x[ku], x[ku + 1], pk, na, vbi, rg, fg, table, ev_)
                vds = (x[a["sD"][k] - 1] if a["sD"][k] > 0 else 0.0) - (x[a["sS"][k] - 1] if a["sS"][k] > 0 else 0.0)
                ndr_op.append((k, float(vds), float(ev_[1])))
        if ndr_op:
            x[:] = x_start
            used = "zero"
            ok = _dc(hold_z)
    if not ok:
        hint = " or start from discharged capacitors (tran.initial = 'zero')" if (used == "op" and nc) else ""
        if ns == 0:
            raise ValueError(f"DC operating point at t = 0 did not converge (check transistor polarity, bias and connections{hint})")
        raise ValueError("DC operating point at t = 0 did not converge (check the bias: the STL model is "
                         f"valid only where the source barrier and the neutral base exist{hint})")
    ok = K.init_state(x, ci, cf, a["rA"], a["rB"], a["rG"], a["cA"], a["cB"], a["cC"], a["vA"], a["vB"], a["vW"],
                      a["iA"], a["iB"], a["iW"], a["sD"], a["sG"], a["sS"], a["sW"], a["wt"], a["wv"], a["woff"],
                      P, Pbase, na, vbi, rg, fg, table, ss, part, sens, ls, cv, cI, 0.0, win, cmp, basic)
    if not ok:
        raise ValueError("element evaluation failed at the initial operating point")
    if used == "zero":
        # a held capacitor absorbs the current the rest of the circuit pushes into it at t = 0 (KCL-consistent
        # initial capacitor current; also the TRAP companion's i_n)
        cI[:] = hold_z * cv
    xp[:] = x
    sf[K.SF_T] = 0.0
    sf[K.SF_HNEXT] = cfg.h_init
    sf[K.SF_TREC] = -1.0
    sf[K.SF_MINU] = 1e9
    sf[K.SF_MINR] = 1e9
    si[K.SI_REFRESH] = 1
    si[K.SI_HAVEPREV] = 0
    # initial record
    recs: list[np.ndarray] = []
    evs: list[np.ndarray] = []
    row = np.zeros(W)
    row[0] = 0.0
    row[1:1 + nn - 1 + nv] = x[:nn - 1 + nv]
    c = 1 + nn - 1 + nv
    ev0 = np.zeros(K.N_EV)
    from .element import stl_eval
    for k in range(ns):
        ku = nn - 1 + nv + 2 * k
        stl_eval(x[ku], x[ku + 1], P[k], na, vbi, rg, fg, table, ev0)
        row[c:c + 7] = [x[ku], x[ku + 1], ev0[3], ev0[1], ev0[2], ls[k, 0], ls[k, 1]]
        c += 7
    row[c:c + nc] = cI                           # capacitor currents at t = 0 (0 at the DC operating point)
    if rec_sink is not None:
        rec_sink(row[None, :].copy(), np.zeros((0, K.N_EVC)))
    else:
        recs.append(row[None, :].copy())
    warnings: list[str] = []
    while True:
        sf[K.SF_TSTOP] = t_end
        status = K.run_chunk(x, xp, ci, cf, a["rA"], a["rB"], a["rG"], a["cA"], a["cB"], a["cC"], a["vA"], a["vB"],
                             a["vW"], a["iA"], a["iB"], a["iW"], a["sD"], a["sG"], a["sS"], a["sW"], a["wt"], a["wv"],
                             a["woff"], a["bp"], a["samp"] if nsamp else np.zeros(0), P, Pbase, na, vbi, rg, fg, table,
                             rv, pmf, ss, part, sens, ls, cv, cI, sf, si, rec, evb, sbuf, win, cmp, basic)
        nr = int(si[K.SI_NREC])
        ne = int(si[K.SI_NEV])
        if rec_sink is not None:
            if nr or ne:
                rec_sink(rec[:nr].copy(), evb[:ne].copy())
        elif nr:
            recs.append(rec[:nr].copy())
        si[K.SI_NREC] = 0
        if ne:
            evs.append(evb[:ne].copy())
            si[K.SI_NEV] = 0
        if progress is not None:
            progress(min(1.0, sf[K.SF_T] / t_end))
        if status in (K.ST_CHUNK, K.ST_BUFFER):
            continue
        if status == K.ST_MAXSTEPS:
            warnings.append(f"run stopped at the step budget solver.max_steps = {cfg.max_steps:.0f} "
                            f"(t = {sf[K.SF_T]:.4g} s of {t_end:.4g} s)")
        elif status == K.ST_FAIL:
            warnings.append(f"time step underflow / no convergence at t = {sf[K.SF_T]:.6g} s; run truncated "
                            f"(state may have left the model's valid domain)")
        break
    rec_all = np.concatenate(recs) if recs else np.zeros((0, W))
    ev_all = np.concatenate(evs) if evs else np.zeros((0, K.N_EVC))
    if len(ev_all) > 1:
        # latch events are written when confirmed (after the crossing): restore time order
        ev_all = ev_all[np.argsort(ev_all[:, K.EV_T], kind="stable")]
    return RunOutput(rec=rec_all, events=ev_all, samples=sbuf[:nsamp].copy(), steps=int(si[K.SI_STEPS]),
                     rejected=int(si[K.SI_REJ]), newton_iters=int(si[K.SI_NEWT]), status=int(status),
                     t_reached=float(sf[K.SF_T]), unresolved_steps=int(si[K.SI_UNRES]),
                     t_unresolved=float(sf[K.SF_TUNRES]), gauss_steps=int(si[K.SI_GAUSS]),
                     t_gauss=float(sf[K.SF_TGAUSS]), t_lrs_drift=float(sf[K.SF_TLRS]), t_band_drift=float(sf[K.SF_TBAND]), min_u=float(sf[K.SF_MINU]), min_r=float(sf[K.SF_MINR]),
                     t_neg_u=float(sf[K.SF_TNEGU]), t_neg_r=float(sf[K.SF_TNEGR]), trap_be=int(si[K.SI_TRAPBE]),
                     runtime_s=time.perf_counter() - tic, hrs_crossings=int(si[K.SI_HRSX]), warnings=warnings,
                     diag=[int(v) for v in si[K.SI_DIAG:K.SI_DIAG + 18]], initial=used, ndr_op=ndr_op)
