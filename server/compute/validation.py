"""Kind "validation": reproduces engine/docs/VALIDATION.md.  payload {level: "fast" | "full"}.

fast: fold checks (3), extension-terms-zero identity vs gate_mean, light conversion, latch-window edges,
      measured-record statistics, FPT node (compound first passage), dynamic MC (100 sweeps).
full: + paper values over 10 records, carrier-noise-only breakdown (channel-subset first passage),
      stochastic package checks (sweep_mc photo dark, vg_curve_stochastic paper frozen) and the circuit
      load-line check.  Modules of other packages are imported lazily; when missing or failing, the check
      is reported with pass = null and the reason.
Each check: {id, label{ko,en}, expected, computed, pass, tolerance, note?, seconds}.
"""
from __future__ import annotations

import importlib
import time
import traceback
import warnings as _warnings
from typing import Any, Callable

import numpy as np

from server import params
from server.compute import data as data_mod
from server.compute import deterministic as D
from server.engine_bridge import A, MODEL, S, ct, m
from server.progress import JobCancelled, null_progress

TOL_FOLD = 1e-3


def _label(ko: str, en: str) -> dict:
    return {"ko": ko, "en": en}


def _check(cid: str, ko: str, en: str, expected: str, computed: str, ok: bool | None, tolerance: str,
           note: str | None = None) -> dict:
    out = dict(id=cid, label=_label(ko, en), expected=expected, computed=computed,
               **{"pass": None if ok is None else bool(ok)}, tolerance=tolerance)
    if note:
        out["note"] = note
    return out


def _sub(progress: Callable, lo: float, hi: float) -> Callable:
    def f(fraction: float = 0.0, message: str = "") -> None:
        progress(lo + (hi - lo) * min(max(float(fraction), 0.0), 1.0), message)
    return f


# ---------------------------------------------------------------------------------------------
# fast checks
# ---------------------------------------------------------------------------------------------
def _folds(device: dict) -> tuple[float, float]:
    z = D.classify(np.asarray(params.build_p(device)), 601)
    return (np.nan, np.nan) if z is None else (float(z[3][0]), float(z[3][1]))


def _fold_check(cid: str, ko: str, en: str, device: dict, exp: tuple[float, float]) -> dict:
    lu, ld = _folds(device)
    ok = abs(lu - exp[0]) <= TOL_FOLD + 1e-9 and abs(ld - exp[1]) <= TOL_FOLD + 1e-9
    return _check(cid, ko, en, f"V_LU {exp[0]:.4f} V, V_LD {exp[1]:.4f} V", f"V_LU {lu:.4f} V, V_LD {ld:.4f} V",
                  ok, "±1 mV")


def check_fold_paper_m2(progress) -> dict:
    return _fold_check("fold_paper_vg-2", "논문 모델 V_G = −2 V 암조건 fold", "Paper model, V_G = −2 V dark, folds",
                       {"preset": "paper", "vg": -2.0}, (3.7037, 2.5979))


def check_fold_paper_m18(progress) -> dict:
    return _fold_check("fold_paper_vg-1.8", "논문 모델 V_G = −1.8 V 암조건 fold", "Paper model, V_G = −1.8 V dark, folds",
                       {"preset": "paper", "vg": -1.8}, (3.8644, 2.5979))


def check_fold_photo(progress) -> dict:
    return _fold_check("fold_photo_2.63pA", "광 모델 V_G = −1.8 V, I_PH = 2.63 pA fold",
                       "Photo model, V_G = −1.8 V, I_PH = 2.63 pA, folds",
                       {"preset": "paper", "vg": -1.8, "light": {"mode": "iph", "iph_pA": 2.63}}, (3.2913, 2.596))


def check_extension_identity(progress) -> dict:
    """photo_mean with all extension terms at their neutral values vs the original gate_mean."""
    import gate_mean as G   # on sys.path via setup_photo

    gm = G.FastModel(MODEL.na, G.load_transport_table()[0])
    rng = np.random.default_rng(20260924)
    npts = 400
    us, rs = rng.uniform(0.0, 1.1, npts), rng.uniform(0.0, 5.0, npts)
    vgs = rng.choice([-3.0, -2.0, -1.8, -1.5, -1.1], npts)
    max_dv, max_rel, nan_mismatch, finite = 0.0, 0.0, 0, 0
    for k in range(npts):
        if k % 50 == 0:
            progress(0.5 * k / npts, "components identity")
        p = S.params(float(vgs[k]), 0.0)
        a = m.components(us[k], rs[k], p, MODEL.na, MODEL.vbi, MODEL.rg, MODEL.fg, MODEL.table)[:18]
        b = G.components(us[k], rs[k], p[:13].copy(), gm.na, gm.vbi, gm.rg, gm.fg, gm.table)
        fa, fb = np.isfinite(a), np.isfinite(b)
        if not np.array_equal(fa, fb):
            nan_mismatch += 1
            continue
        if not fa.all():
            continue
        finite += 1
        max_dv = max(max_dv, abs(a[0] - b[0]))
        den = np.maximum(np.abs(b), 1e-300)
        max_rel = max(max_rel, float(np.max(np.abs(a - b) / den)))
    fold_d = 0.0
    for vg in (-2.0, -1.8, -1.1):
        p = S.params(vg, 0.0)
        za, zb = MODEL.classify(p, m.state_grid(601)), gm.classify(p[:13].copy(), G.state_grid(601))
        fold_d = max(fold_d, float(np.max(np.abs(za[3] - zb[3]))))
        progress(0.5 + 0.15 * (vg + 2.0 + 1.0), "fold identity")
    ok = max_dv <= 1e-12 and fold_d <= 1e-12 and nan_mismatch == 0
    return _check("extension_identity", "확장 항 = 0 → gate_mean과 동일", "Extension terms = 0 reproduce gate_mean",
                  "identical to gate_mean (≤ 1e-12 V)",
                  f"max |ΔV_D| = {max_dv:.2e} V over {finite} finite (u, r) points, max rel. Δ(currents) = {max_rel:.1e}, "
                  f"max |Δfold| = {fold_d:.2e} V (V_G −2, −1.8, −1.1)", ok, "≤ 1e-12 V",
                  note=f"{npts} random (u, r) points, u ∈ [0, 1.1] V, r ∈ [0, 5] V; NaN-pattern mismatches: {nan_mismatch}")


def check_light_conversion(progress) -> dict:
    R = params.RESPONSIVITY_PA_PER_MW
    vals = [R * pw for pw in (1.15, 2.55, 3.51)]
    exp = [0.86, 1.91, 2.63]
    ok = all(abs(v - e) <= 0.005 for v, e in zip(vals, exp))
    return _check("light_conversion", "광 변환 I_PH = R·P", "Light conversion I_PH = R·P",
                  "0.86 / 1.91 / 2.63 pA at 1.15 / 2.55 / 3.51 mW (R = 0.75 pA/mW)",
                  " / ".join(f"{v:.4f}" for v in vals) + f" pA (R = {R:.4f} pA/mW)", ok, "±0.005 pA")


def check_latch_window(progress) -> dict:
    dev = params.resolve_device({"preset": "paper"})
    brackets = {"low": (-4.0, -3.8), "high": (-0.7, -0.9)}   # (no latch, latch)
    edges = {}
    for n_, (v_no, v_yes) in brackets.items():
        if D._latch_at(dev, v_no, 601) is not None or D._latch_at(dev, v_yes, 601) is None:
            return _check("latch_window", "래치 창 (논문, 고정 상태)", "Latch window (paper, fixed states)",
                          "V_G = −3.90 … −0.815 V", "bracket assumption failed", False, "±10 mV")
        edges[n_] = D._bisect_edge(dev, 601, v_no, v_yes, 5e-4, lambda: progress(0.5, "bisection"))
    ok = abs(edges["low"] + 3.90) <= 0.010 and abs(edges["high"] + 0.815) <= 0.010
    return _check("latch_window", "래치 창 (논문, 고정 상태)", "Latch window (paper, fixed states)",
                  "V_G = −3.90 … −0.815 V", f"V_G = {edges['low']:.4f} … {edges['high']:.4f} V", ok, "±10 mV",
                  note="edges = latch-existence limits of classify() (grid 601), bisection to 0.5 mV")


def check_measured_records(progress) -> dict:
    md = data_mod.measured()
    lu, ld = md["paper_idvd"]["stats"]["LU"]["sd"] * 1e3, md["paper_idvd"]["stats"]["LD"]["sd"] * 1e3
    ph = md["photo"]["conditions"][0]["stats"]
    ok = abs(lu - 123.1) <= 0.1 and abs(ld - 19.5) <= 0.1 and abs(ph["mean"] - 3.806) <= 0.001 and abs(ph["sd"] * 1e3 - 173.2) <= 0.1
    return _check("measured_records", "측정 기록 통계", "Measured record statistics",
                  "paper σ_LU 123.1 mV, σ_LD 19.5 mV; photo −1.8 V dark mean 3.806 V, SD 173.2 mV",
                  f"paper σ_LU {lu:.1f} mV, σ_LD {ld:.1f} mV; photo mean {ph['mean']:.4f} V, SD {ph['sd'] * 1e3:.1f} mV",
                  ok, "±0.1 mV / ±1 mV")


def check_fpt_node(progress) -> dict:
    progress(0.1, "compound first passage (FPT node, cached under engine/photo_extension/photo_nodes)")
    rec, q = A.hazard(-2.0, rate=0.4)
    mean, sd = float(np.mean(q)), float(np.std(q)) * 1e3
    ok = abs(mean - 3.644) <= 0.003 and abs(sd - 8.0) <= 1.5
    return _check("fpt_node", "FPT 노드 V_G = −2 V, 0.4 V/s (중심 상태)", "FPT node V_G = −2 V dark, 0.4 V/s (centre states)",
                  "mean V_LU ≈ 3.644 V, SD ≈ 8 mV", f"mean V_LU {mean:.4f} V, SD {sd:.2f} mV", ok,
                  "mean ±3 mV, SD ±1.5 mV",
                  note="stl_api.py prints '~6.8 mV' in its comment; VALIDATION.md states ≈ 8 mV")


def check_dynamic_mc(progress) -> dict:
    progress(0.1, "gate_dynamic_compare.simulate(100, seed 2026092920)")
    sw = A.sweeps(n=100, seed=2026092920)
    lu, ld = np.asarray(sw["V_LU"], float), np.asarray(sw["V_LD"], float)
    mlu, slu = float(np.nanmean(lu)), float(np.nanstd(lu, ddof=1)) * 1e3
    mld, sld = float(np.nanmean(ld)), float(np.nanstd(ld, ddof=1)) * 1e3
    ok = abs(mlu - 3.63) <= 0.010 and abs(slu - 120) <= 10 and abs(mld - 2.70) <= 0.010 and abs(sld - 20) <= 3
    return _check("dynamic_mc", "동적 MC 100 스윕 (seed 2026092920)", "Dynamic MC, 100 sweeps, seed 2026092920",
                  "V_LU 3.63 V / 120 mV, V_LD 2.70 V / 20 mV",
                  f"V_LU {mlu:.4f} V / {slu:.1f} mV, V_LD {mld:.4f} V / {sld:.1f} mV", ok,
                  "means ±10 mV, SD_LU ±10 mV, SD_LD ±3 mV")


# ---------------------------------------------------------------------------------------------
# full checks
# ---------------------------------------------------------------------------------------------
def check_paper_records(progress) -> dict:
    # Same ten records as gate_dynamic_compare.main(): seeds 2026093000 … 2026093009, pooled (1000 sweeps).
    lu, ld = [], []
    for k in range(10):
        progress(k / 10, f"record {k + 1}/10")
        sw = A.sweeps(n=100, seed=2026093000 + k)
        lu.append(np.asarray(sw["V_LU"], float))
        ld.append(np.asarray(sw["V_LD"], float))
    s_lu = float(np.nanstd(np.concatenate(lu), ddof=1)) * 1e3
    s_ld = float(np.nanstd(np.concatenate(ld), ddof=1)) * 1e3
    ok = abs(s_lu - 125.8) <= 1.0 and abs(s_ld - 19.6) <= 0.5
    return _check("paper_records", "논문 값 (10 기록 × 100 스윕)", "Paper values (10 records × 100 sweeps)",
                  "σ_LU 125.8 mV, σ_LD 19.6 mV (measured 123.1 / 19.5)",
                  f"σ_LU {s_lu:.1f} mV, σ_LD {s_ld:.1f} mV (pooled over 1000 sweeps)", ok, "σ_LU ±1 mV, σ_LD ±0.5 mV",
                  note="seeds 2026093000 … 2026093009, as in gate_dynamic_compare.main()")


# --- carrier-noise-only breakdown -------------------------------------------------------------
def _noise_rows(vd: float, p: np.ndarray, ug: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.array([S.state(u, vd, p) for u in ug])
    fr = np.empty((len(rows), 2))
    for k, (u, r) in enumerate(rows[:, :2]):
        z = m.components(u, r, p, MODEL.na, MODEL.vbi, MODEL.rg, MODEL.fg, MODEL.table)
        fr[k] = (z[5] + z[7], z[6])                      # REC (bulk + junction SRH), DIFF
    return rows, fr / fr.sum(axis=1, keepdims=True)


def _noise_lattice(rows: np.ndarray, fr: np.ndarray, N: int):
    """compound_fpt.make_lattice('LU') on a lattice of spacing 1/N, with the loss split into REC and DIFF."""
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq

    u = rows[:, 0]
    Fv = rows[:, 3] - rows[:, 4]
    ff = PchipInterpolator(u, Fv)
    s = np.flatnonzero((Fv[:-1] > 0) & (Fv[1:] < 0))
    wi = s[0]
    uw = brentq(ff, u[wi], u[wi + 1])
    i0 = np.flatnonzero(rows[:, 2] >= 1e-8)[0]
    uth = brentq(PchipInterpolator(u, rows[:, 2] - 1e-8), u[i0 - 1], u[i0])
    q = (rows[:, 5] + rows[:, 6] + rows[:, 7]) / m.Q
    qfun = PchipInterpolator(u, q)
    lo, end, well = float(qfun(0.1)), float(qfun(uth)), float(qfun(uw))
    xx = np.arange(np.ceil(lo), np.ceil(end), 1.0 / N)
    start = int(np.clip(np.round((well - xx[0]) * N), 0, len(xx) - 1))
    rates = np.exp(PchipInterpolator(q, np.log(rows[:, [3, 4, 10]]), axis=0)(xx))
    reverse = PchipInterpolator(q, rows[:, 1])(xx)
    fdiff = np.clip(PchipInterpolator(q, fr[:, 1])(xx), 0.0, 1.0)
    bt = rates[:, 2] / m.Q
    ii = np.maximum(rates[:, 0] - bt, 0.0)
    death = rates[:, 1]
    return start, reverse, bt, ii, death * (1 - fdiff), death * fdiff


def _subset_mfpt(start, reverse, bt, ii, rec, dif, N: int, noisy: tuple[str, ...]) -> float:
    """Mean first-passage time with only the `noisy` channels as jumps; the others enter as deterministic drift
    (piecewise-deterministic process), discretised upwind on the refined lattice (error O(1/N))."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    cf = ct.cf
    n = len(bt)
    ar = np.arange(n)
    probs = np.array([np.interp(reverse, cf.rv, cf.pmf[:, k]) for k in range(1, cf.K + 1)]).T
    mk = probs @ np.arange(1, cf.K + 1)
    jumps: list[tuple[int, np.ndarray]] = []
    drift = np.zeros(n)
    if "II" in noisy:
        jumps += [(k * N, ii * probs[:, k - 1] / mk) for k in range(1, cf.K + 1)]
    else:
        drift += ii
    for name, step, rate in (("BTBT", N, bt), ("REC", -N, rec), ("DIFF", -N, dif)):
        if name in noisy:
            jumps.append((step, rate))
        else:
            drift += np.sign(step) * rate
    jumps.append((1, np.where(drift > 0, drift * N, 0.0)))
    jumps.append((-1, np.where(drift < 0, -drift * N, 0.0)))
    total = np.zeros(n)
    dests = []
    for k, rate in jumps:
        tg = np.maximum(ar + k, 0)
        rate = np.where(tg == ar, 0.0, rate)
        dests.append((tg, rate))
        total += rate
    rr, cc, dd = [ar], [ar], [np.ones(n)]
    for tg, rate in dests:
        ok = (tg < n) & (rate > 0)
        rr.append(ar[ok])
        cc.append(tg[ok])
        dd.append(-rate[ok] / total[ok])
    Amat = coo_matrix((np.concatenate(dd), (np.concatenate(rr), np.concatenate(cc))), shape=(n, n)).tocsc()
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        t = spsolve(Amat, 1 / total)
    return float(t[start])


def carrier_noise_breakdown(vg: float = -2.0, rate: float = 0.4, window: float = 0.14, step: float = 0.004,
                            levels: tuple[int, int] = (4, 8), progress=null_progress) -> dict:
    """SD of V_LU (mV) with only a subset of carrier channels stochastic (centre states, no local states).

    Channels: II (compound clusters), BTBT (unit generation: GIDL + junction BTBT + photo), REC (bulk + junction
    SRH loss), DIFF (emitter out-diffusion).  The remaining channels act as deterministic drift; the upwind
    refinement error O(1/N) is removed by Richardson extrapolation of log h between N = levels.
    'all' uses every channel as jumps (identical to the compound FPT node)."""
    import photo_fpt as F

    p = S.params(vg, 0.0)
    b, i, j, fold = MODEL.classify(p, m.state_grid(601))
    uf = b[i, 17]
    ug = np.unique(np.round(np.r_[np.linspace(0.1, 0.9, 181), np.linspace(uf - 0.065, uf + 0.065, 61)], 12))
    volts = np.arange(fold[0] - window, fold[0] - 0.001, step)
    data = []
    for k, vd in enumerate(volts):
        progress(0.25 * k / len(volts), "rows")
        try:
            data.append((vd, _noise_rows(vd, p, ug)))
        except (ValueError, IndexError):
            continue
    subsets = {"all": ("II", "BTBT", "REC", "DIFF"), "II": ("II",), "BTBT": ("BTBT",), "REC": ("REC",), "DIFF": ("DIFF",)}
    out: dict[str, Any] = {}
    plan = [(name, noisy, ((1,) if name == "all" else levels)) for name, noisy in subsets.items()]
    steps = sum(len(ns) for _, _, ns in plan)
    done = 0
    for name, noisy, ns in plan:   # 'all' has no deterministic drift: exact at any N
        logs = []
        for N in ns:
            progress(0.25 + 0.75 * done / steps, f"subset {name}, N = {N}")
            done += 1
            V, H = [], []
            for vd, (rows, fr) in data:
                try:
                    tm = _subset_mfpt(*_noise_lattice(rows, fr, N), N, noisy)
                except (ValueError, IndexError):
                    continue
                V.append(float(vd))
                H.append(1 / tm if np.isfinite(tm) and tm > 0 else np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                logs.append((np.array(V), np.log(np.array(H))))
        if len(logs) == 2 and np.array_equal(logs[0][0], logs[1][0]):
            V, lh = logs[1][0], 2 * logs[1][1] - logs[0][1]
        else:
            V, lh = logs[-1]
        q = F.quantiles(dict(fold_V=float(fold[0]), voltage=V.tolist(), hazard=np.exp(lh).tolist()), rate)
        out[name] = dict(sd_mV=float(np.std(q)) * 1e3, mean_V=float(np.mean(q)))
    return dict(fold_V=float(fold[0]), subsets=out, levels=list(levels))


def check_carrier_noise(progress) -> dict:
    res = carrier_noise_breakdown(progress=progress)
    s = {k: v["sd_mV"] for k, v in res["subsets"].items()}
    exp = {"II": 4.6, "BTBT": 2.7, "REC": 4.3, "DIFF": 1.8, "all": 7.8}
    ok = all(abs(s[k] - exp[k]) <= 1.0 for k in exp)
    return _check("carrier_noise_breakdown", "캐리어 잡음만 (논문, V_G = −2 V)", "Carrier noise only (paper, V_G = −2 V)",
                  "II 4.6, BTBT 2.7, REC 4.3, DIFF 1.8, all four 7.8 mV",
                  f"II {s['II']:.1f}, BTBT {s['BTBT']:.1f}, REC {s['REC']:.1f}, DIFF {s['DIFF']:.1f}, all four {s['all']:.1f} mV",
                  ok, "±1 mV each",
                  note="derived here: compound first passage with one channel class as jumps and the others as "
                       "deterministic drift (upwind lattice refined N = 4, 8, Richardson-extrapolated); "
                       "BTBT = all unit generation events (GIDL + junction BTBT); the paper's own breakdown code is not "
                       "in the handoff package")


def _optional(module: str) -> Any:
    try:
        return importlib.import_module(module), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{module} unavailable: {type(exc).__name__}: {exc}"


def check_sweep_mc_photo(progress) -> dict:
    ids = ("sweep_mc_photo_dark", "이 소자 V_G = −1.8 V 암조건, 1200 V/s (sweep_mc)", "This device, V_G = −1.8 V dark, 1200 V/s (sweep_mc)",
           "measured mean 3.806 V, SD 173 mV (400 cycles)")
    mod, err = _optional("server.compute.stochastic")
    if mod is None:
        return _check(ids[0], ids[1], ids[2], ids[3], "not run", None, "mean ±20 mV, SD ±20 mV", note=err)
    payload = {"device": {"preset": "photo"},
               "sweep": params.PRESETS["photo"]["sweep"], "stochastic": params.PRESETS["photo"]["stochastic"]}
    try:
        res = mod.run_sweep_mc(payload, progress)
        st = res["stats"]["LU"]
        mean, sd = float(st["mean"]), float(st["sd"]) * 1e3
    except JobCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        return _check(ids[0], ids[1], ids[2], ids[3], "error", None, "mean ±20 mV, SD ±20 mV",
                      note=f"run_sweep_mc failed: {type(exc).__name__}: {exc}")
    ok = abs(mean - 3.806) <= 0.020 and abs(sd - 173.2) <= 20
    return _check(ids[0], ids[1], ids[2], ids[3], f"mean {mean:.4f} V, SD {sd:.1f} mV (engine {res.get('engine')})",
                  ok, "mean ±20 mV, SD ±20 mV")


def check_vg_curve_stochastic(progress) -> dict:
    ids = ("vg_curve_stochastic_paper", "V_G 곡선 (논문, 고정 상태): σ_LU·평균 최대", "V_G curve (paper, frozen states): σ_LU and mean peaks",
           "σ_LU peak 129.8 mV at −1.25 V; mean peak 4.354 V at −1.10 V")
    tol = "σ peak ±10 mV, mean peak ±20 mV, positions ±0.1 V"
    mod, err = _optional("server.compute.stochastic")
    if mod is None:
        return _check(ids[0], ids[1], ids[2], ids[3], "not run", None, tol, note=err)
    sto = dict(params.PRESETS["paper"]["stochastic"])
    sto["local_state"] = dict(sto["local_state"], mode="frozen", acquisition_trend=False)
    sto["engine"] = "general"
    payload = {"device": {"preset": "paper"}, "sweep": {"vd_max_V": 6.0, "rate_V_per_s": 0.4, "dv_V": 0.002},
               "stochastic": sto, "vg_min": -1.6, "vg_max": -0.9, "n": 15}
    try:
        res = mod.run_vg_curve_stochastic(payload, progress)
        vg = np.asarray(res["vg"], float)
        sd = np.asarray([np.nan if x is None else x for x in res["sd_VLU_mV"]], float)
        mean = np.asarray([np.nan if x is None else x for x in res["mean_VLU"]], float)
        ks, km = int(np.nanargmax(sd)), int(np.nanargmax(mean))
    except JobCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        return _check(ids[0], ids[1], ids[2], ids[3], "error", None, tol,
                      note=f"run_vg_curve_stochastic failed: {type(exc).__name__}: {exc}")
    ok = (abs(sd[ks] - 129.8) <= 10 and abs(vg[ks] + 1.25) <= 0.1 and abs(mean[km] - 4.354) <= 0.02
          and abs(vg[km] + 1.10) <= 0.1)
    return _check(ids[0], ids[1], ids[2], ids[3],
                  f"σ_LU peak {sd[ks]:.1f} mV at {vg[ks]:.3f} V; mean peak {mean[km]:.3f} V at {vg[km]:.3f} V", ok, tol,
                  note="V_G grid −1.6 … −0.9 V (15 points), 0–6 V at 0.4 V/s, frozen states σ_φG = 0.1534 V")


def check_circuit_load_line(progress) -> dict:
    ids = ("circuit_load_line", "회로: 부하선 램프가 fold 재현", "Circuit: load-line ramp reproduces the folds",
           "latch-up at V_D ≈ 3.7037 V, latch-down at V_D ≈ 2.5979 V")
    tol = "±30 mV"
    mod, err = _optional("server.compute.circuit")
    if mod is None or not hasattr(mod, "run_circuit"):
        return _check(ids[0], ids[1], ids[2], ids[3], "not run", None, tol, note=err or "run_circuit missing")
    try:
        res = mod.run_circuit({"bench": "load_line", "mode": "deterministic", "device": {"preset": "paper"}}, progress)
        ev = res.get("events") or []
        up = [e for e in ev if e.get("kind") == "latch_up" and e.get("v_d") is not None]
        dn = [e for e in ev if e.get("kind") == "latch_down" and e.get("v_d") is not None]
    except JobCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        return _check(ids[0], ids[1], ids[2], ids[3], "error", None, tol,
                      note=f"run_circuit failed: {type(exc).__name__}: {exc}")
    if not up:
        return _check(ids[0], ids[1], ids[2], ids[3], f"no latch_up event with v_d ({len(ev)} events)", None, tol,
                      note="the load_line bench defaults may not sweep through the fold")
    vu = float(up[0]["v_d"])
    vdn = float(dn[0]["v_d"]) if dn else None
    ok = abs(vu - 3.7037) <= 0.03 and (vdn is None or abs(vdn - 2.5979) <= 0.03)
    comp = f"latch-up at V_D {vu:.4f} V" + (f", latch-down at V_D {vdn:.4f} V" if vdn is not None else ", no latch-down event")
    return _check(ids[0], ids[1], ids[2], ids[3], comp, ok, tol, note="bench 'load_line' with its default bench_params")


FAST: list[Callable] = [check_fold_paper_m2, check_fold_paper_m18, check_fold_photo, check_extension_identity,
                        check_light_conversion, check_latch_window, check_measured_records, check_fpt_node,
                        check_dynamic_mc]
FULL: list[Callable] = FAST + [check_paper_records, check_carrier_noise, check_sweep_mc_photo,
                               check_vg_curve_stochastic, check_circuit_load_line]

_IDS = {  # id / label used when a check raises before building its record
    f.__name__: f.__name__.removeprefix("check_") for f in FULL
}


def run_validation(payload: dict, progress=null_progress) -> dict:
    t0 = time.perf_counter()
    level = (payload or {}).get("level", "fast")
    if level not in ("fast", "full"):
        raise ValueError("level must be 'fast' or 'full'")
    fns = FAST if level == "fast" else FULL
    checks = []
    for k, fn in enumerate(fns):
        lo, hi = k / len(fns), (k + 1) / len(fns)
        progress(lo, f"check {k + 1}/{len(fns)}: {_IDS[fn.__name__]}")
        t = time.perf_counter()
        try:
            rec = fn(_sub(progress, lo, hi))
        except JobCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            rec = _check(_IDS[fn.__name__], _IDS[fn.__name__], _IDS[fn.__name__], "-", "error", None, "-",
                         note=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
        rec["seconds"] = time.perf_counter() - t
        checks.append(rec)
    progress(1.0, "done")
    summary = dict(n=len(checks), passed=sum(c["pass"] is True for c in checks),
                   failed=sum(c["pass"] is False for c in checks), skipped=sum(c["pass"] is None for c in checks))
    return dict(level=level, checks=checks, summary=summary, runtime_s=time.perf_counter() - t0, warnings=[])
