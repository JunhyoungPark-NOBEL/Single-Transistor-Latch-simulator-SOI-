"""Stochastic compute kinds: ``hazard``, ``sweep_mc``, ``vg_curve_stochastic`` (WEB_CONTRACT §1/§2).

Physics (engine/docs/MODEL_SPEC.md §4):
* carrier noise (Eq. 2): compound-jump first passage (II clusters + unit events incl. photogeneration)
  on the body-charge lattice, hazard h(V_D) = 1/MFPT below the fold, integrated along the ramp;
* local states: the action-point state X (drain-edge GIDL potential p[9] by default) is Gaussian,
  frozen per cycle or an OU process in time; the emitter state (p[10]) shifts the folds linearly.

Engines for ``sweep_mc``: ``calibrated_lookup`` wraps ``gate_dynamic_compare.simulate`` (paper device,
V_G = -2 V dark, 0-4 V only); ``general`` builds fold tables and hazard nodes for any device (see
stoch_mc.py).  Tables are cached on disk (server/.cache/stochastic) keyed by the parameter vector.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from server import params as P
from server.progress import null_progress

from . import stoch_core as C
from . import stoch_mc as MC

ENGINE_DIR = Path(__file__).resolve().parents[2] / "engine"
CAPS = dict(grid=(201, 2001), n_cycles=(1, 2000), vg_points=(2, 61), fold_nodes=(3, 61), hazard_nodes=(1, 9),
            vd_max_V=(0.1, 8.0), dv_V=(0.0005, 0.1), n_traces=(0, 50))
ACTIONS = {
    # action: (ext key or None for p[9], unit, label)
    "gidl": (None, "V", "drain-edge GIDL potential δφ_G (p[9])"),
    "local_avalanche": ("dloc", "ln", "local avalanche log-fluctuation (p[23])"),
    "junction": ("dj", "V", "junction potential offset (p[19])"),
    "multiplication": ("dm", "ln", "log (M-1) scale (p[20])"),
}
MODES = ("none", "frozen", "evolving")


# ============================================================================================
# payload helpers
# ============================================================================================
def _prog(progress):
    return progress if progress is not None else null_progress


def _clamp(value, name, warnings, integer=False):
    lo, hi = CAPS[name]
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number (got {value!r})") from None
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    if v < lo or v > hi:
        warnings.append(f"{name} = {v:g} clamped to [{lo:g}, {hi:g}]")
        v = min(max(v, lo), hi)
    return int(round(v)) if integer else v


def _resolve(payload: dict, warnings: list, need_stochastic=True):
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    device = P.resolve_device(payload.get("device"))
    preset = device.get("preset") or "paper"
    sweep = P.resolve_section(preset, "sweep", payload.get("sweep"))
    sweep["vd_max_V"] = _clamp(sweep["vd_max_V"], "vd_max_V", warnings)
    sweep["dv_V"] = _clamp(sweep["dv_V"], "dv_V", warnings)
    rate = float(sweep["rate_V_per_s"])
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sweep.rate_V_per_s must be > 0")
    if rate > 1e8:
        warnings.append(f"sweep.rate_V_per_s = {rate:g} clamped to 1e8")
        rate = 1e8
    sweep["rate_V_per_s"] = rate
    device["numerics"]["grid"] = _clamp(device["numerics"].get("grid", 601), "grid", warnings, integer=True)
    stoch = None
    if need_stochastic:
        stoch = P.resolve_section(preset, "stochastic", payload.get("stochastic"))
        stoch["n_cycles"] = _clamp(stoch["n_cycles"], "n_cycles", warnings, integer=True)
        stoch["fold_nodes"] = _clamp(stoch["fold_nodes"], "fold_nodes", warnings, integer=True)
        stoch["hazard_nodes"] = _clamp(stoch["hazard_nodes"], "hazard_nodes", warnings, integer=True)
        stoch["n_traces"] = _clamp(stoch.get("n_traces", 12), "n_traces", warnings, integer=True)
        try:
            stoch["seed"] = int(stoch["seed"]) % (2 ** 63)
        except (TypeError, ValueError):
            raise ValueError("stochastic.seed must be an integer") from None
        ls = stoch["local_state"]
        if ls.get("mode") not in MODES:
            raise ValueError(f"local_state.mode must be one of {MODES}")
        if ls.get("action") not in ACTIONS:
            raise ValueError(f"local_state.action must be one of {tuple(ACTIONS)}")
        for k in ("sigma", "sigma_E_V"):
            v = float(ls.get(k, 0.) or 0.)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"local_state.{k} must be >= 0")
            ls[k] = v
        for k in ("tau_s", "tau_E_s"):
            v = float(ls.get(k, 1.) or 0.)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"local_state.{k} must be > 0")
            ls[k] = v
        if stoch.get("engine", "auto") not in ("auto", "general", "calibrated_lookup"):
            raise ValueError("stochastic.engine must be auto | general | calibrated_lookup")
        stoch["carrier_noise"] = bool(stoch.get("carrier_noise", True))
        stoch["ld_carrier_noise"] = bool(stoch.get("ld_carrier_noise", False))
    return device, sweep, stoch


class StateMap:
    """Maps the action-point state X to the parameter vector (X absolute lever value)."""

    def __init__(self, device: dict, action: str, warnings: list):
        self.device, self.action = device, action
        key, unit, label = ACTIONS[action]
        self.key, self.unit, self.label = key, unit, label
        self.grid = int(device["numerics"]["grid"])
        self.over = {}
        ext = device["ext"]
        if action == "gidl":
            self.x0 = 0.0          # build_p already adds delta_phi_G0 to p[9]
        else:
            self.x0 = float(ext[key])
        if action == "local_avalanche":
            aloc = float(ext.get("aloc", 0.))
            if aloc <= 0:
                warnings.append("local_avalanche action needs a local avalanche strength ext.aloc (p[21]) > 0; "
                                "using aloc = 1.0 (the verify_hloc.py value)")
                aloc = 1.0
            self.over["aloc"] = aloc
        if action != "gidl":
            warnings.append(f"action point '{action}' is an experimental hypothesis lever (hypotheses.py), "
                            "not the paper's calibrated GIDL action point")

    def p(self, x: float, de: float = 0.0) -> np.ndarray:
        if self.action == "gidl":
            return np.asarray(P.build_p(self.device, dg=float(x), de=de, **self.over), float)
        return np.asarray(P.build_p(self.device, de=de, **{self.key: float(x)}, **self.over), float)


KERNEL_NOTE = ("{d} first-passage hazard not computable where the drain-junction reverse bias of the lattice "
               "exceeds the avalanche cluster kernel range (r ≤ {hi:.1f} V, V_D ≳ 5.1 V): escapes there are "
               "placed at the fold (carrier noise underestimated)")


def _hazard_notes(rec: dict, warnings: list, d: str = "LU") -> None:
    """Warnings for hazard voltages that could not be computed (fold-independent text so that several
    hazard nodes produce one deduplicated message)."""
    if rec.get("fold_V") is None:
        return
    if rec.get("kernel_skipped"):
        warnings.append(KERNEL_NOTE.format(d=d, hi=rec["kernel_range_V"][1]))
    elif rec.get("n_voltages") and rec.get("skipped", 0) >= rec["n_voltages"]:
        warnings.append(f"{d} first-passage lattice failed at every voltage below the fold: escape at the fold "
                        "(no carrier noise)")


def _lu_hazard(p, rate, grid, warnings, progress=None, notes=True):
    """photo_fpt window (0.28 V) extended to 0.5 / 0.8 V when the escape probability in the first
    10 mV of the window is not negligible at this ramp rate (gate_fpt uses 0.5 V near V_G = -1.1 V)."""
    rec = None
    for window in (.28, .5, .8):
        rec = C.lu_hazard_curve(p, window=window, grid=grid, progress=progress)
        V, h, begin = C.resolved_hazard(rec)
        if len(V) < 2 or begin > 0 or not np.isfinite(h[0]):
            break
        if h[0] * .01 / rate < 1e-3 or V[0] <= (rec["VLD_fold_V"] or -np.inf) + .0015:
            break
    else:
        warnings.append(f"LU hazard still non-negligible {rec['window_V']:.2f} V below the fold at "
                        f"{rate:g} V/s; earlier escapes are truncated")
    if notes:
        _hazard_notes(rec, warnings, "LU")
    return rec


def _measured_photo(device, sweep, warnings):
    k = P.match_photo_condition(device)
    if k is None:
        return None
    raw = np.load(ENGINE_DIR / "data" / "raw_VLU.npy")
    vg, pw = P.MEASURED_PHOTO_CONDITIONS[k]
    v = raw[:, k].astype(float)
    if device.get("preset") != "photo":
        warnings.append("measured overlay is the photo device's record (different device from the paper preset)")
    if abs(sweep["rate_V_per_s"] - 1200.) > 1e-6:
        warnings.append("measured overlay was recorded at 1200 V/s")
    return dict(label=f"photo device, V_G = {vg:.1f} V, P = {pw:.2f} mW (400 cycles, 1200 V/s)",
                V_LU=v, V_LD=None, stats=dict(LU=C.stats(v), LD=None))


def _measured_paper():
    raw = np.load(ENGINE_DIR / "model/janus_calibration_20260920/outputs/measured_idvd_parsed.npz")
    lu = (raw["VLU_low"] + raw["VLU_high"]) / 2
    ld = (raw["VLD_low"] + raw["VLD_high"]) / 2
    return dict(label="paper device, V_G = -2 V dark, 100 sweeps (0.4 V/s, 10 mV steps)",
                V_LU=lu, V_LD=ld, stats=dict(LU=C.stats(lu), LD=C.stats(ld)))


def _measured(device, sweep, warnings):
    if P.is_paper_reference(device):
        return _measured_paper()
    return _measured_photo(device, sweep, warnings)


def _branch_xy(br: dict | None, vmax: float) -> dict:
    if not br:
        return dict(vd=[], id=[])
    vd = np.asarray(br["vd"], float); idd = np.asarray(br["id"], float)
    keep = vd <= vmax
    return dict(vd=vd[keep], id=idd[keep])


def _centre(rec: dict, vd_max: float) -> dict:
    if not rec.get("latch"):
        return dict(V_LU=None, V_LD=None, HRS=dict(vd=[], id=[]), LRS=dict(vd=[], id=[]))
    return dict(V_LU=rec["V_LU"], V_LD=rec["V_LD"], HRS=_branch_xy(rec["HRS"], vd_max + 1),
                LRS=_branch_xy(rec["LRS"], vd_max + 1))


def _fnum(x):
    return None if x is None or not np.isfinite(x) else float(x)


# ============================================================================================
# kind "hazard"
# ============================================================================================
def run_hazard(payload: dict, progress=None) -> dict:
    """Compound first-passage hazard h(V_D) below the latch-up fold, survival and quantiles."""
    tic = time.perf_counter()
    progress = _prog(progress)
    warnings: list[str] = []
    device, sweep, _ = _resolve(payload, warnings, need_stochastic=False)
    try:
        dg = float(payload.get("dg", 0.) or 0.); de = float(payload.get("de", 0.) or 0.)
    except (TypeError, ValueError):
        raise ValueError("dg and de must be numbers (V)") from None
    if not (abs(dg) <= 2.0 and abs(de) <= 0.05):
        raise ValueError("dg must be within ±2 V and de within ±0.05 V")
    rate = sweep["rate_V_per_s"]
    grid = int(device["numerics"]["grid"])
    p = np.asarray(P.build_p(device, dg=dg, de=de), float)
    progress(0.02, "first-passage hazard")
    rec = _lu_hazard(p, rate, grid, warnings, progress=lambda f: progress(0.02 + 0.93 * f, "first-passage hazard"),
                     notes=False)
    out = dict(fold_V=rec["fold_V"], VLD_fold_V=rec["VLD_fold_V"], rate_V_per_s=rate, dg=dg, de=de,
               window_V=rec["window_V"], step_V=rec["step_V"], skipped=rec.get("skipped", 0),
               n_voltages=rec.get("n_voltages"), kernel_skipped=rec.get("kernel_skipped", 0))
    if rec["fold_V"] is None:
        warnings.append("no latch: steady-state locus not traceable at the fold (gap in u)" if rec.get("locus_gap")
                        else "no latch (classify found no two-fold branch) for this device")
        out.update(voltage=[], hazard=[], survival=[], quantiles=dict(prob=[], v=[]), stats=C.stats([]),
                   fold_atom=None)
    else:
        V = np.asarray(rec["voltage"], float)
        q = C.quantiles(rec, rate)
        Vc, cum = C.cumulative_hazard(rec, rate)
        surv = np.exp(-cum) if len(Vc) else np.array([])
        st = C.stats(q); st["lag1"] = None; st["censored"] = 0
        sub = np.r_[np.arange(0, len(C.PROB), 50), len(C.PROB) - 1]
        atom = float(surv[-1]) if len(surv) else 1.0
        out.update(voltage=V, hazard=np.asarray(rec["hazard"], float), survival=surv,
                   quantiles=dict(prob=C.PROB[sub], v=q[sub]), stats=st, fold_atom=atom,
                   I_at_fold_A=C.fold_node(p, grid).get("I_LU"))       # refined, as the deterministic folds
        n_v, ks = int(rec.get("n_voltages") or 0), int(rec.get("kernel_skipped") or 0)
        if ks:
            r0, r1 = rec["kernel_r_V"]
            k0, k1 = rec["kernel_range_V"]
            warnings.append(f"first-passage hazard not computable at {ks} of {n_v} voltages below the fold: the "
                            f"lattice reverse bias ({r0:.2f} … {r1:.2f} V) leaves the avalanche cluster kernel range "
                            f"({k0:.1f} … {k1:.1f} V); escapes there are placed at the fold")
        if len(V) < 2:
            if n_v >= 2 and rec.get("skipped", 0) >= n_v:
                if not ks:
                    warnings.append(f"first-passage lattice failed at all {n_v} voltages below the fold: escape "
                                    "assumed at the fold")
            else:
                warnings.append("hazard window too narrow (fold close to V_LD): escape assumed at the fold")
        if atom > 0.01:
            warnings.append(f"{100 * atom:.1f} % of escapes reach the fold (deterministic fold atom)")
    progress(1.0, "done")
    out.update(runtime_s=time.perf_counter() - tic, warnings=warnings)
    return out


# ============================================================================================
# kind "sweep_mc"
# ============================================================================================
def calibrated_dv_ok(dv: float) -> bool:
    """gate_dynamic_compare.simulate steps 0 → 4 V in round(4/ΔV) steps, reads the current every round(0.01/ΔV)
    steps and pairs it with a fixed 401-point axis (0.01 V spacing): only ΔV = 10 mV / k (k = 1, 2, …) makes the
    read-out grid that axis (ΔV > 20 mV divides by zero, other ΔV give traces of the wrong length)."""
    k = int(round(0.01 / dv))
    return k >= 1 and int(round(4.0 / dv)) == 400 * k


def _choose_engine(device, sweep, stoch, warnings=None) -> str:
    eng = stoch.get("engine", "auto")
    ls = stoch["local_state"]
    dv_ok = calibrated_dv_ok(sweep["dv_V"])
    if eng == "auto":
        ok = (P.is_paper_reference(device) and ls["action"] == "gidl" and abs(sweep["vd_max_V"] - 4.0) < 1e-9
              and stoch["carrier_noise"])
        if ok and not dv_ok and warnings is not None:
            warnings.append(f"voltage step ΔV = {1e3 * sweep['dv_V']:.4g} mV is not 10 mV / k (10, 5, 3.333, 2.5, "
                            "2, … mV), which the calibrated_lookup engine needs: using the general engine")
        return "calibrated_lookup" if ok and dv_ok else "general"
    if eng == "calibrated_lookup":
        if not P.is_paper_reference(device):
            raise ValueError("calibrated_lookup covers only the calibrated paper device at V_G = -2 V, dark, no "
                             "extensions and zero state centre; use engine 'general' (or 'auto')")
        if ls["action"] != "gidl":
            raise ValueError("calibrated_lookup supports only the GIDL action point; use engine 'general'")
        if abs(sweep["vd_max_V"] - 4.0) > 1e-9:
            raise ValueError("calibrated_lookup is tabulated for 0 → 4 V sweeps; use engine 'general'")
        if not dv_ok:
            raise ValueError(f"calibrated_lookup needs a voltage step ΔV = 10 mV / k (10, 5, 3.333, 2.5, 2, … mV); "
                             f"got {1e3 * sweep['dv_V']:.4g} mV — use engine 'general' (or 'auto')")
    return eng


def _finish_mc(out, stoch, sweep, device, warnings):
    vlu = np.asarray(out["V_LU"], float); vld = np.asarray(out["V_LD"], float)
    out["stats"] = dict(LU=C.stats(vlu), LD=C.stats(vld))
    out["hist"] = dict(LU=C.histogram(vlu), LD=C.histogram(vld))
    # CDF over every cycle that ran the sweep direction: LU plateaus at the latched fraction; a latch-down never
    # reached by 0 V lies below the sweep, so it enters LD as an offset.  LD population: every cycle for the
    # calibrated engine (unpaired up/down records), the latched-up cycles for the general engine.
    n_ld = len(vld) if out.get("engine") == "calibrated_lookup" else int(np.isfinite(vlu).sum())
    ld_cdf = C.ecdf(vld, n_total=n_ld)
    if ld_cdf["n"] and n_ld > ld_cdf["n"]:
        ld_cdf["p"] = [float(x) for x in (np.asarray(ld_cdf["p"]) + (n_ld - ld_cdf["n"]) / n_ld)]
    out["cdf"] = dict(LU=C.ecdf(vlu), LD=ld_cdf)
    out["measured"] = _measured(device, sweep, warnings)
    n_c = int((~np.isfinite(vlu)).sum())
    if n_c:
        warnings.append(f"{n_c} of {len(vlu)} cycles did not latch up within 0 → {sweep['vd_max_V']:g} V (censored)")
    return out


def _run_calibrated(device, sweep, stoch, progress, warnings):
    ls = stoch["local_state"]
    n, seed = stoch["n_cycles"], stoch["seed"]
    mode = {"none": "fast_only", "frozen": "frozen"}.get(ls["mode"])
    if ls["mode"] == "evolving":
        mode = "dynamic" if ls.get("acquisition_trend", False) else "stationary"
    amp = ls["sigma"] / P.SIGMA_PHI_G_V
    if ls["mode"] != "none":
        amp_e = ls["sigma_E_V"] / P.SIGMA_PHI_E_V
        if abs(amp_e - amp) > 1e-3 * max(1., amp):
            warnings.append(f"calibrated_lookup scales σ_φE with the same factor as σ_φG ({amp:.3g}); the "
                            f"requested σ_E ratio {amp_e:.3g} is not used (effective σ_E = "
                            f"{1e3 * amp * P.SIGMA_PHI_E_V:.3f} mV)")
        if ls["mode"] == "evolving" and abs(ls["tau_E_s"] - P.TAU_E_S) > 1e-6:
            warnings.append(f"calibrated_lookup uses the calibrated emitter τ_E = {P.TAU_E_S:.3g} s "
                            f"(requested {ls['tau_E_s']:g} s ignored)")
    if not stoch["ld_carrier_noise"]:
        warnings.append("calibrated_lookup always includes the LU and LD first-passage (carrier) noise")
    tau_up = ls["tau_s"] if ls["mode"] == "evolving" else None
    dv, rate = sweep["dv_V"], sweep["rate_V_per_s"]
    progress(0.05, "calibrated lookup Monte Carlo")
    z, rep = MC.calibrated(n, seed, mode, dv, rate, amp, tau_up)
    progress(0.6, "state replay")
    states, _draws, c0 = MC.calibrated_states(n, seed, mode, dv, rate, amp, tau_up)
    steps = states[0].shape[1] - 1
    t, fd = MC.calibrated_lookup_table()
    J = np.linspace(t["J"][0], t["J"][-1], 61)
    ff = fd(np.c_[J, np.full_like(J, c0["e0"])])
    centre_rec = C.fold_node(P.build_p(device), int(device["numerics"]["grid"]))
    tr = z["transition"]
    out = dict(engine="calibrated_lookup", V_LU=tr[0, :, 1], V_LD=tr[1, :, 1],
               V_LU_continuous=tr[0, :, 0], V_LD_continuous=tr[1, :, 0], quantisation_V=0.01,
               cycle_state=states[0][:, steps // 2, 0] - c0["j0"],
               cycle_state_emitter=states[0][:, steps // 2, 1] - c0["e0"],
               fold_table=dict(delta=J - c0["j0"], V_LU=ff[:, 0], V_LD=ff[:, 1]),
               centre=_centre(centre_rec, 4.0),
               state_axis=dict(action="gidl", unit="V", label=ACTIONS["gidl"][2], centre=c0["j0"],
                               sigma=ls["sigma"], mode=ls["mode"], calibrated_mode=mode),
               lookup_outside_fraction=rep["outside_fraction"])
    # the engine reads the current every k = round(0.01/ΔV) steps of its own grid (as gate_dynamic_compare)
    k = int(round(0.01 / dv))
    v_up = np.linspace(0, 4, steps + 1)[::k]
    v_dn = np.linspace(4, 0, steps + 1)[::k]
    if z["current"].shape[2] != len(v_up):
        raise RuntimeError("calibrated_lookup current read-out does not match its voltage grid")
    traces = []
    for c in range(min(stoch["n_traces"], n)):
        traces.append(dict(cycle=c, V_LU=_fnum(tr[0, c, 1]), V_LD=_fnum(tr[1, c, 1]),
                           up=dict(vd=v_up, id=z["current"][0, c]), down=dict(vd=v_dn, id=z["current"][1, c])))
    out["traces"] = traces
    warnings.append("calibrated_lookup: V_LU/V_LD are 10 mV read-out midpoints (as the measurement); "
                    "continuous values in V_LU_continuous/V_LD_continuous")
    if max(rep["outside_fraction"]) > 0:
        warnings.append(f"{100 * max(rep['outside_fraction']):.2f} % of state samples lie outside the lookup "
                        "table (linear extrapolation)")
    return out


def _general_tables(device, sweep, stoch, progress, warnings, lo=0.0, hi=1.0):
    """Fold table, emitter slopes, hazard fields for the general engine (all cached on disk)."""
    ls = stoch["local_state"]
    sm = StateMap(device, ls["action"], warnings)
    grid = sm.grid
    rate = sweep["rate_V_per_s"]
    sigma = ls["sigma"] if ls["mode"] != "none" else 0.0
    span = hi - lo
    t0 = time.perf_counter()
    if sigma > 0:
        xs = sm.x0 + sigma * np.linspace(-4.5, 4.5, stoch["fold_nodes"])
    else:
        xs = np.array([sm.x0])
    recs = []
    for k, x in enumerate(xs):
        progress(lo + span * 0.4 * k / len(xs), f"fold table {k + 1}/{len(xs)}")
        recs.append(C.fold_node(sm.p(x), grid))
    centre = C.fold_node(sm.p(sm.x0), grid)
    vlu = np.array([r["V_LU"] if r.get("latch") else np.nan for r in recs], float)
    vld = np.array([r["V_LD"] if r.get("latch") else np.nan for r in recs], float)
    folds = MC.FoldTable(xs, vlu, vld, recs)
    t_fold = time.perf_counter() - t0
    s_lu_e = s_ld_e = 0.0
    sig_e = ls["sigma_E_V"] if ls["mode"] != "none" else 0.0
    if sig_e > 0:
        a, b = C.fold_node(sm.p(sm.x0, de=+sig_e), grid), C.fold_node(sm.p(sm.x0, de=-sig_e), grid)
        if a.get("latch") and b.get("latch"):
            s_lu_e = (a["V_LU"] - b["V_LU"]) / (2 * sig_e); s_ld_e = (a["V_LD"] - b["V_LD"]) / (2 * sig_e)
        else:
            warnings.append("emitter-state sensitivity unavailable (no latch at ±σ_E); emitter state ignored")
    t0 = time.perf_counter()
    lu_field = None; hnodes = []
    if stoch["carrier_noise"]:
        if sigma > 0:
            tt, _w = C.gauss_hermite(stoch["hazard_nodes"])
            hx = sm.x0 + sigma * tt
        else:
            hx = np.array([sm.x0])
        hrecs = []
        for k, x in enumerate(hx):
            base = lo + span * (0.4 + 0.45 * k / len(hx))
            msg = f"first-passage hazard node {k + 1}/{len(hx)}"
            progress(base, msg)
            rec = _lu_hazard(sm.p(x), rate, grid, warnings,
                             progress=lambda f, b=base, m_=msg: progress(b + span * 0.45 * f / len(hx), m_))
            hrecs.append(rec)
            hnodes.append(dict(delta=float(x - sm.x0), fold_V=rec["fold_V"], window_V=rec["window_V"],
                               points=len(rec["voltage"])))
        lu_field = MC.build_hazard_field(hx, hrecs)
        if lu_field is None:
            warnings.append("no resolved LU hazard at the hazard nodes: escape at the fold")
    ld_field = None
    if stoch["ld_carrier_noise"]:
        progress(lo + span * 0.86, "latch-down hazard")
        rec = C.ld_hazard_curve(sm.p(sm.x0), grid=grid)
        _hazard_notes(rec, warnings, "LD")
        ld_field = MC.build_hazard_field([sm.x0], [rec])
        if ld_field is None:
            warnings.append("no resolved LD hazard at the centre state: latch-down at the fold")
    t_haz = time.perf_counter() - t0
    return dict(sm=sm, xs=xs, folds=folds, centre=centre, s_lu_e=s_lu_e, s_ld_e=s_ld_e, lu=lu_field, ld=ld_field,
                hazard_nodes=hnodes, sigma=sigma, sigma_e=sig_e, timing=dict(fold_s=t_fold, hazard_s=t_haz))


def _run_general(device, sweep, stoch, progress, warnings):
    ls = stoch["local_state"]
    tb = _general_tables(device, sweep, stoch, progress, warnings, 0.0, 0.9)
    sm, folds = tb["sm"], tb["folds"]
    if ls["mode"] == "evolving" and ls.get("acquisition_trend"):
        warnings.append("acquisition_trend applies only to the calibrated_lookup engine (paper record); ignored")
    if not np.isfinite(folds.vluf).any():
        warnings.append("no latch anywhere in the state table: every cycle is censored")
    t0 = time.perf_counter()
    progress(0.9, "Monte Carlo sweeps")
    r = MC.simulate_general(n=stoch["n_cycles"], seed=stoch["seed"], vd_max=sweep["vd_max_V"], dv=sweep["dv_V"],
                            rate=sweep["rate_V_per_s"], mode=ls["mode"], x0=sm.x0, sigma=tb["sigma"],
                            tau_s=ls["tau_s"], sigma_e=tb["sigma_e"], tau_e=ls["tau_E_s"], folds=folds,
                            s_lu_e=tb["s_lu_e"], s_ld_e=tb["s_ld_e"], lu_haz=tb["lu"], ld_haz=tb["ld"],
                            progress=lambda f: progress(0.9 + 0.08 * f, "Monte Carlo sweeps"))
    t_mc = time.perf_counter() - t0
    no_latch = 1 - folds.latch_fraction_gauss(sm.x0, tb["sigma"])
    if no_latch > 1e-3:
        warnings.append(f"{100 * no_latch:.1f} % of the state distribution has no latch (fold table); such cycles "
                        "are censored")
    traces = []
    for c in range(min(stoch["n_traces"], stoch["n_cycles"])):
        rec = folds.nearest_rec(r["state"][c]) or tb["centre"]
        if not rec.get("latch"):
            continue
        tr = MC.trace(rec, sweep["vd_max_V"], r["V_LU"][c], r["V_LD"][c])
        traces.append(dict(cycle=c, V_LU=_fnum(r["V_LU"][c]), V_LD=_fnum(r["V_LD"][c]), **tr))
    if traces:
        warnings.append("traces are approximate: quasi-static HRS/LRS branches of the nearest fold-table state "
                        "with each cycle's own switching voltages")
    fold_table = None
    if ls["mode"] != "none" and tb["sigma"] > 0:
        fold_table = dict(delta=tb["xs"] - sm.x0, V_LU=folds.vlu, V_LD=folds.vld)
    x_eval = np.array([-1e-3, 1e-3]) * max(tb["sigma"], 1e-3) + sm.x0
    out = dict(engine="general", V_LU=r["V_LU"], V_LD=r["V_LD"], cycle_state=r["state"] - sm.x0,
               cycle_state_emitter=r["emitter"] if tb["sigma_e"] > 0 else None,
               fold_table=fold_table, centre=_centre(tb["centre"], sweep["vd_max_V"]), traces=traces,
               quantisation_V=None,
               state_axis=dict(action=ls["action"], unit=sm.unit, label=sm.label, centre=sm.x0, sigma=tb["sigma"],
                               mode=ls["mode"]),
               sensitivity=dict(dVLU_dX=_fnum(np.diff(folds.lu(x_eval))[0] / np.diff(x_eval)[0]) if tb["sigma"] > 0 else None,
                                dVLD_dX=_fnum(np.diff(folds.ld(x_eval))[0] / np.diff(x_eval)[0]) if tb["sigma"] > 0 else None,
                                dVLU_dE=tb["s_lu_e"], dVLD_dE=tb["s_ld_e"]),
               hazard_nodes=tb["hazard_nodes"], no_latch_weight=no_latch,
               fold_atom_fraction=float(np.mean(r["fold_atom"][np.isfinite(r["V_LU"])])) if np.isfinite(r["V_LU"]).any() else None,
               timing=dict(tb["timing"], mc_s=t_mc), dt_s=r["dt"])
    if not stoch["carrier_noise"]:
        warnings.append("carrier (first-passage) noise off: latch-up exactly at the (state-dependent) fold")
    if not stoch["ld_carrier_noise"]:
        warnings.append("latch-down first-passage noise off: V_LD at the (state-dependent) fold; at slow ramps the "
                        "LD escape can sit well above the fold (paper device, 0.4 V/s: +0.10 V) — enable "
                        "ld_carrier_noise")
    return out


def run_sweep_mc(payload: dict, progress=None) -> dict:
    """Monte Carlo triangular sweeps 0 → vd_max → 0: V_LU/V_LD per cycle, statistics, traces."""
    tic = time.perf_counter()
    progress = _prog(progress)
    warnings: list[str] = []
    device, sweep, stoch = _resolve(payload, warnings)
    engine = _choose_engine(device, sweep, stoch, warnings)
    if engine == "calibrated_lookup":
        out = _run_calibrated(device, sweep, stoch, progress, warnings)
    else:
        out = _run_general(device, sweep, stoch, progress, warnings)
    out = _finish_mc(out, stoch, sweep, device, warnings)
    out.update(n_cycles=stoch["n_cycles"], seed=stoch["seed"], rate_V_per_s=sweep["rate_V_per_s"],
               vd_max_V=sweep["vd_max_V"])
    progress(1.0, "done")
    out.update(runtime_s=time.perf_counter() - tic, warnings=_dedup(warnings))
    return out


# ============================================================================================
# kind "vg_curve_stochastic"
# ============================================================================================
def truncated_mixture(fl, wt, eps, weps, vd_max):
    """Moments of V_LU = F + ε conditional on latching within the sweep (V_LU ≤ vd_max), as sweep_mc reports them.

    F: fold states fl (finite) with weights wt; ε: carrier-noise offsets eps with weights weps, independent of F.
    Returns (P_latched, mean, state_var, noise_var) with the law-of-total-variance split
    state_var = Var_F(E[V | F, latched]) and noise_var = E_F[Var(V | F, latched)]; without truncation these are
    exactly Σ wt·F + E[ε], Var(F) and Var(ε).  mean/vars are None when no weight latches within the sweep."""
    fl = np.asarray(fl, float)
    wt = np.asarray(wt, float) / np.sum(wt)
    order = np.argsort(eps, kind="stable")
    e = np.asarray(eps, float)[order]
    w = np.asarray(weps, float)[order] / np.sum(weps)
    e0 = float(np.sum(w * e))
    ec = e - e0                                               # centred offsets (well-conditioned moments)
    cw, c1, c2 = (np.r_[0., np.cumsum(x)] for x in (w, w * ec, w * ec * ec))
    k = np.searchsorted(e, vd_max - fl, side="right")          # count of ε with F + ε <= vd_max
    full = k == len(e)
    pk = np.where(full, 1.0, cw[k])
    safe = np.where(pk > 0, pk, 1.0)
    m1 = np.where(full, 0.0, c1[k] / safe)                     # E[ε - e0 | F, latched]
    m2 = np.where(full, float(np.sum(w * ec * ec)), c2[k] / safe)
    W = wt * pk
    P = float(W.sum())
    if P <= 1e-12:
        return P, None, None, None
    mk = fl + e0 + m1                                          # E[V | F, latched]
    vk = np.maximum(m2 - m1 * m1, 0.0)                         # Var(V | F, latched)
    mean = float(np.sum(W * mk) / P)
    return P, mean, float(np.sum(W * (mk - mean) ** 2) / P), float(np.sum(W * vk) / P)


def _vg_point(device, sweep, stoch, progress, warnings, lo, hi):
    ls = stoch["local_state"]
    sm = StateMap(device, ls["action"], [])
    grid = sm.grid
    rate = sweep["rate_V_per_s"]
    sigma = ls["sigma"] if ls["mode"] != "none" else 0.0
    span = hi - lo
    centre = C.fold_node(sm.p(sm.x0), grid)
    if sigma > 0:
        xs = sm.x0 + sigma * np.linspace(-4.5, 4.5, stoch["fold_nodes"])
        recs = []
        for k, x in enumerate(xs):
            progress(lo + span * 0.5 * k / len(xs), "")
            recs.append(C.fold_node(sm.p(x), grid))
        vlu = np.array([r["V_LU"] if r.get("latch") else np.nan for r in recs], float)
        vld = np.array([r["V_LD"] if r.get("latch") else np.nan for r in recs], float)
        folds = MC.FoldTable(xs, vlu, vld, recs)
        t = np.linspace(-4, 4, 81); wt = np.exp(-t * t / 2); wt /= wt.sum()      # vg_sweep.py grid
        fl = folds.lu(sm.x0 + sigma * t)
    else:
        t = np.array([0.]); wt = np.array([1.])
        fl = np.array([centre["V_LU"] if centre.get("latch") else np.nan])
    ok = np.isfinite(fl)
    no_latch = float(1 - wt[ok].sum())
    row = dict(mean=None, sd=None, state_sd=None, noise_sd=None, fold_centre=_fnum(centre.get("V_LU") or np.nan),
               vld=_fnum(centre.get("V_LD") or np.nan), no_latch=no_latch, beyond=0.0, censored=no_latch,
               latch=bool(centre.get("latch")))
    if not ok.any():
        return row
    wk = wt[ok] / wt[ok].sum()
    eps, weps = np.array([0.]), np.array([1.])
    if stoch["carrier_noise"]:
        if sigma > 0:
            tt, ww = C.gauss_hermite(stoch["hazard_nodes"])
        else:
            tt, ww = np.array([0.]), np.array([1.])
        noise, shift, ws = [], [], []
        for k, (ti, wi) in enumerate(zip(tt, ww, strict=True)):
            b = lo + span * (0.5 + 0.5 * k / len(tt))
            progress(b, "")
            rec = _lu_hazard(sm.p(sm.x0 + sigma * ti), rate, grid, warnings,
                             progress=lambda f, b=b: progress(b + span * 0.5 * f / len(tt), ""))
            q = C.quantiles(rec, rate)
            if q is None:
                continue
            noise.append(q - q.mean()); shift.append(q.mean() - rec["fold_V"]); ws.append(wi)
        if ws:
            ws = np.asarray(ws) / np.sum(ws)
            msh = float(np.sum(ws * np.asarray(shift)))
            # pooled noise offsets, each node centred on the common mean shift: mean msh, variance Σ w_i SD_i²
            eps = np.concatenate(noise) + msh
            weps = np.concatenate([np.full(len(q_), wi / len(q_)) for q_, wi in zip(noise, ws, strict=True)])
    vd_max = float(sweep["vd_max_V"])
    P, mean, svar, nvar = truncated_mixture(fl[ok], wk, eps, weps, vd_max)
    beyond = float((1 - no_latch) * (1 - P))
    row.update(beyond=beyond, censored=no_latch + beyond)
    if mean is not None:
        row.update(mean=mean, sd=float(np.sqrt(svar + nvar)), state_sd=float(np.sqrt(svar)),
                   noise_sd=float(np.sqrt(nvar)))
    return row


def _latch_edge(device, sm_action, grid, a, b, tol=0.0025):
    """Bisection of the V_G where the centre-state latch appears/disappears between a (latch) and b."""
    def has(vg):
        d = copy.deepcopy(device); d["vg"] = float(vg)
        sm = StateMap(d, sm_action, [])
        return bool(C.fold_node(sm.p(sm.x0), grid).get("latch"))
    la = has(a)
    for _ in range(40):
        if abs(b - a) <= tol:
            break
        mid = 0.5 * (a + b)
        if has(mid) == la:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def run_vg_curve_stochastic(payload: dict, progress=None) -> dict:
    """Mean V_LU and σ_LU versus V_G: Gaussian fold mixture (frozen states) + first-passage noise."""
    tic = time.perf_counter()
    progress = _prog(progress)
    warnings: list[str] = []
    device, sweep, stoch = _resolve(payload, warnings)
    try:
        vg_min = float(payload.get("vg_min", -3.6)); vg_max = float(payload.get("vg_max", -0.9))
    except (TypeError, ValueError):
        raise ValueError("vg_min / vg_max must be numbers (V)") from None
    n = _clamp(payload.get("n", 19), "vg_points", warnings, integer=True)
    if not (np.isfinite(vg_min) and np.isfinite(vg_max)) or vg_min >= vg_max:
        raise ValueError("need vg_min < vg_max")
    if vg_min < -6 or vg_max > 1:
        raise ValueError("V_G range must lie within [-6, 1] V")
    ls = stoch["local_state"]
    if ls["mode"] == "evolving":
        warnings.append("evolving local states: V_G curve uses their stationary Gaussian distribution "
                        "(frozen-state mixture per cycle)")
    if ls["action"] != "gidl":
        StateMap(device, ls["action"], warnings)
    if ls["mode"] != "none" and ls["sigma_E_V"] > 0:
        warnings.append("emitter state σ_E is not included in the V_LU curve (it mainly moves V_LD)")
    vgs = np.linspace(vg_min, vg_max, n)
    keys = ("mean", "sd", "state_sd", "noise_sd", "fold_centre", "vld", "no_latch", "beyond", "censored", "latch")
    rows = {k: [] for k in keys}
    for k, vg in enumerate(vgs):
        d = copy.deepcopy(device); d["vg"] = float(vg)
        lo, hi = 0.97 * k / n, 0.97 * (k + 1) / n
        msg = f"V_G = {vg:+.3f} V ({k + 1}/{n})"
        row = _vg_point(d, sweep, stoch, lambda f, m_="", msg=msg: progress(f, msg), warnings, lo, hi)
        for key in keys:
            rows[key].append(row[key])
        progress(hi, msg)
    latch = np.array(rows["latch"], bool)
    beyond = np.array(rows["beyond"], float)
    if (beyond > 1e-3).any():
        kb = np.flatnonzero(beyond > 1e-3)
        warnings.append(f"V_LU beyond the sweep maximum {sweep['vd_max_V']:g} V for part of the cycles at {len(kb)} "
                        f"V_G value(s) (up to {100 * beyond.max():.1f} %, {vgs[kb[0]]:+.3f} … {vgs[kb[-1]]:+.3f} V): "
                        "mean and σ are over the cycles that latch within the sweep, as in sweep_mc "
                        "(beyond_sweep_weight / censored_weight)")
    window = dict(vg_low=None, vg_high=None)
    grid = int(device["numerics"]["grid"])
    if latch.any():
        idx = np.flatnonzero(latch)
        i0, i1 = idx[0], idx[-1]
        # null when the edge is not bracketed by the scanned range
        window["vg_low"] = None if i0 == 0 else _latch_edge(device, ls["action"], grid, vgs[i0], vgs[i0 - 1])
        window["vg_high"] = None if i1 == n - 1 else _latch_edge(device, ls["action"], grid, vgs[i1], vgs[i1 + 1])
    arr = lambda key, s=1.0: np.array([np.nan if v is None else s * v for v in rows[key]], float)
    measured = []
    if device.get("preset") == "photo":
        light = device["light"]
        pw = float(light["power_mW"]) if light.get("mode") == "power" else P.iph_A(device) * 1e12 / P.RESPONSIVITY_PA_PER_MW
        for m in json.loads((ENGINE_DIR / "data" / "measured_stats.json").read_text()):
            if abs(float(m["P_mW"]) - pw) < 0.005:
                measured.append(dict(vg=float(m["VG"]), power_mW=float(m["P_mW"]), mean_V=float(m["mean_V"]),
                                     sd_mV=float(m["sd_mV"])))
        if measured and abs(sweep["rate_V_per_s"] - 1200.) > 1e-6:
            warnings.append("measured points were recorded at 1200 V/s")
    progress(1.0, "done")
    return dict(vg=vgs, mean_VLU=arr("mean"), sd_VLU_mV=arr("sd", 1e3), state_sd_mV=arr("state_sd", 1e3),
                noise_sd_mV=arr("noise_sd", 1e3), fold_centre_V=arr("fold_centre"), VLD_fold_V=arr("vld"),
                no_latch_weight=arr("no_latch"), beyond_sweep_weight=arr("beyond"), censored_weight=arr("censored"),
                latch=latch, window=window, measured=measured, vd_max_V=sweep["vd_max_V"],
                rate_V_per_s=sweep["rate_V_per_s"], runtime_s=time.perf_counter() - tic, warnings=_dedup(warnings))


def _dedup(w):
    seen, out = set(), []
    for x in w:
        if x not in seen:
            seen.add(x); out.append(x)
    return out
