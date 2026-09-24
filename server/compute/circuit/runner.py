"""``run_circuit(payload, progress)``: payload parsing/validation, bench construction, feasibility
estimate, the run loop (progress + cancellation), per-bench analysis and the result assembly
(WEB_CONTRACT §4)."""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from server import params as PR

from . import benches as B
from .sim import SolverConfig, simulate
from .stochastic import (ACTION_UNIT, SECONDS_PER_STEP, branch_profile, draw_local_states, estimate_steps,
                         classify_checked, noise_bands, parse_local_state)

MAX_POINTS = 4000
MAX_POINTS_OTHER = 1500        # stored runs 1..7 (keeps the JSON result a few MB at most)
MAX_STORED_RUNS = 8
MAX_TOTAL_STEPS = 4e7          # all runs of one request (~25 min at 40 us/step)
MAX_EVENTS_OUT = 20000


def _L(ko: str, en: str) -> dict:
    return {"ko": ko, "en": en}


# ---- parsing ---------------------------------------------------------------------------------
def _solver(payload: dict, t_end: float, warnings: list[str]) -> dict:
    s = B.merged(B.SOLVER_DEFAULTS, payload.get("solver"))
    if s["method"] not in ("BE", "TRAP"):
        raise ValueError("solver.method must be 'BE' or 'TRAP'")
    try:
        s["reltol"] = float(s["reltol"])
    except (TypeError, ValueError):
        raise ValueError("solver.reltol must be a number")
    if not (1e-5 <= s["reltol"] <= 0.1):
        raise ValueError("solver.reltol must be within [1e-5, 0.1]")
    ms = int(float(s["max_steps"]))
    if ms < 1000:
        raise ValueError("solver.max_steps must be >= 1000")
    if ms > B.CAPS["max_steps"]:
        warnings.append(f"solver.max_steps capped at {B.CAPS['max_steps']:.0f}")
        ms = B.CAPS["max_steps"]
    s["max_steps"] = ms
    for key in ("dt_min_s", "dt_max_s"):
        if s[key] is not None:
            v = float(s[key])
            if not v > 0:
                raise ValueError(f"solver.{key} must be > 0")
            s[key] = v
    if s["dt_min_s"] is None:
        s["dt_min_s"] = max(1e-15, 1e-13 * t_end)
    if s["dt_max_s"] is None:
        s["dt_max_s"] = t_end / 2000.0
    if s["dt_min_s"] >= s["dt_max_s"]:
        raise ValueError("solver.dt_min_s must be smaller than solver.dt_max_s")
    for key, lo, hi in (("tau_frac", 1e-3, 0.5), ("max_events_per_step", 1.0, 1e6), ("noise_dt_min_s", 0.0, 1e-3),
                        ("gauss_threshold", 10.0, 1e6), ("gauss_tau_min_s", 0.0, 1e-3), ("gauss_tau_frac", 0.01, 1.0),
                        ("noise_z_max", 1.0, 1e3)):
        v = float(s[key])
        if not (lo <= v <= hi):
            raise ValueError(f"solver.{key} must be within [{lo}, {hi}]")
        s[key] = v
    return s


def _stochastic(payload: dict, mode: str, device: dict, warnings: list[str]) -> dict:
    st = B.merged(B.STOCHASTIC_DEFAULTS, payload.get("stochastic"))
    st["seed"] = int(float(st.get("seed", B.STOCHASTIC_DEFAULTS["seed"])))
    n = int(float(st.get("n_runs", 20)))
    if n < 1:
        raise ValueError("stochastic.n_runs must be >= 1")
    if n > B.CAPS["n_runs"]:
        warnings.append(f"stochastic.n_runs capped at {B.CAPS['n_runs']}")
        n = B.CAPS["n_runs"]
    st["n_runs"] = n if mode == "stochastic" else 1
    st["carrier_noise"] = bool(st.get("carrier_noise", True))
    st["ld_carrier_noise"] = bool(st.get("ld_carrier_noise", False))
    st["local_state"] = st.get("local_state") or {"mode": "none"}
    return st


def _detect(payload: dict) -> dict:
    d = B.merged({"i_threshold_A": 1e-8, "hysteresis": 10.0}, payload.get("detect"))
    ith = float(d["i_threshold_A"])
    if not (1e-15 <= ith <= 1e-3):
        raise ValueError("detect.i_threshold_A must be within [1e-15, 1e-3] A")
    hy = float(d["hysteresis"])
    if not (1.0 <= hy <= 1e4):
        raise ValueError("detect.hysteresis must be within [1, 1e4]")
    return dict(i_threshold_A=ith, hysteresis=hy, i_threshold_down_A=ith / hy)


def _check_thresholds(det: dict, prof: dict, warnings: list[str], cell: str = "") -> None:
    """The latch state comes from the body (u vs the fold values); the current thresholds only time
    the events.  Warn when a threshold does not separate the branches at its fold: the event is then
    timed where the body leaves / reaches a branch instead of at the threshold crossing."""
    fi = prof.get("fold_I")
    if not fi:
        return
    ith, idn = det["i_threshold_A"], det["i_threshold_down_A"]
    pre = f"{cell}: " if cell else ""
    rng_up = f"(HRS {fi['hrs_at_lu']:.3g} A, LRS {fi['lrs_at_lu']:.3g} A at the latch-up fold)"
    rng_dn = f"(HRS {fi['hrs_at_ld']:.3g} A, LRS {fi['lrs_at_ld']:.3g} A at the latch-down fold)"
    if ith <= fi["hrs_at_lu"]:
        warnings.append(f"{pre}detect.i_threshold_A = {ith:.3g} A is not above the HRS current {rng_up}: latch-up is "
                        "timed where the body leaves the HRS, not at the threshold crossing")
    elif ith >= fi["lrs_at_lu"]:
        warnings.append(f"{pre}detect.i_threshold_A = {ith:.3g} A is not below the LRS current {rng_up}: latch-up is "
                        "timed where the body reaches the LRS, not at the threshold crossing")
    if idn >= fi["lrs_at_ld"]:
        warnings.append(f"{pre}detect.i_threshold_A / hysteresis = {idn:.3g} A is not below the LRS current {rng_dn}: "
                        "latch-down is timed where the body leaves the LRS, not at the threshold crossing")
    elif idn <= fi["hrs_at_ld"]:
        warnings.append(f"{pre}detect.i_threshold_A / hysteresis = {idn:.3g} A is not above the HRS current {rng_dn}: "
                        "latch-down is timed where the body reaches the HRS, not at the threshold crossing")


def _device_p(device: dict, vg: float | None = None, iph_pA: float | None = None) -> np.ndarray:
    d = PR.resolve_device(device)
    if vg is not None:
        d["vg"] = float(vg)
    if iph_pA is not None:
        d["light"] = dict(d["light"], mode="iph", iph_pA=float(iph_pA))
    return np.array(PR.build_p(d), dtype=float)


def _label(p: np.ndarray) -> str:
    return f"V_G = {p[11]:g} V, I_PH = {p[13] * 1e12:.3g} pA"


# ---- decimation / signals ---------------------------------------------------------------------
def _decimate_idx(t: np.ndarray, feats: list[np.ndarray], n_max: int = MAX_POINTS) -> np.ndarray:
    n = len(t)
    if n <= n_max:
        return np.arange(n)
    T = max(t[-1] - t[0], 1e-300)
    comps = [(t - t[0]) / T]
    for f in feats:
        f = np.asarray(f, float)
        rng = np.nanmax(f) - np.nanmin(f)
        comps.append(np.nan_to_num((f - np.nanmin(f)) / rng) if rng > 0 else np.zeros(n))
    X = np.vstack(comps)
    seg = np.sqrt(np.sum(np.diff(X, axis=1) ** 2, axis=0))
    s = np.r_[0.0, np.cumsum(seg)]
    k_arc = int(n_max * 0.6) - 2
    k_time = n_max - k_arc - 2
    ia = np.searchsorted(s, np.linspace(0, s[-1], k_arc))
    it = np.searchsorted(t, np.linspace(t[0], t[-1], k_time))
    idx = np.unique(np.clip(np.r_[0, n - 1, ia, it], 0, n - 1))
    return idx[:n_max] if len(idx) > n_max else idx


def _signals(spec: B.BenchSpec, net_c: dict, out, ls_unit: str, stochastic_ls: bool, has_E: bool, cfg_i_floor: float,
             n_max: int = MAX_POINTS):
    rec = out.rec
    nn = net_c["n_nodes"]
    nv = net_c["nV"]
    ns = net_c["nS"]
    nodes = spec.net.nodes
    t = rec[:, 0]
    base = 1 + (nn - 1) + nv
    cells = spec.meta["cells"]
    feats = []
    for k in range(ns):
        feats.append(np.log10(np.abs(rec[:, base + 7 * k + 3]) + cfg_i_floor))
        feats.append(rec[:, 1 + nodes.index(cells[k]["drain"]) - 1])
    idx = _decimate_idx(t, feats, n_max)
    r = rec[idx]
    sig = []
    src_node = "clk" if spec.bench == "pbit" else "src"
    sig.append(dict(key="v_clk" if spec.bench == "pbit" else "v_src",
                    label=_L("클럭 전원 전압" if spec.bench == "pbit" else "전원 전압", "Clock supply" if spec.bench == "pbit" else "Source voltage"),
                    unit="V", values=r[:, nodes.index(src_node)], axis="voltage"))
    multi = ns > 1
    for k in range(ns):
        sfx = str(k + 1) if multi else ""
        cell_ko = f" (셀 {k + 1})" if multi else ""
        cell_en = f" (cell {k + 1})" if multi else ""
        c0 = base + 7 * k
        vd = r[:, nodes.index(cells[k]["drain"])]
        sig.append(dict(key=f"v_d{sfx}", label=_L("드레인 전압" + cell_ko, "Drain voltage" + cell_en), unit="V", values=vd, axis="voltage"))
        sig.append(dict(key=f"i_d{sfx}", label=_L("드레인 전류" + cell_ko, "Drain current" + cell_en), unit="A", values=r[:, c0 + 3], axis="current"))
        sig.append(dict(key=f"u{sfx}", label=_L("소스-바디 준페르미 분리 u" + cell_ko, "Source–body splitting u" + cell_en), unit="V", values=r[:, c0], axis="state"))
        sig.append(dict(key=f"r{sfx}", label=_L("드레인 접합 역바이어스 r" + cell_ko, "Drain-junction reverse bias r" + cell_en), unit="V", values=r[:, c0 + 1], axis="state"))
        sig.append(dict(key=f"q_b{sfx}", label=_L("바디 전하 변화 ΔQ_B" + cell_ko, "Body-charge change ΔQ_B" + cell_en), unit="C",
                        values=r[:, c0 + 2] - rec[0, c0 + 2], axis="charge"))
        sig.append(dict(key=f"f_body{sfx}", label=_L("바디 순 정공 전류 F" + cell_ko, "Net body hole current F" + cell_en), unit="A",
                        values=r[:, c0 + 4], axis="current"))
        if stochastic_ls:
            sig.append(dict(key=f"dphi{sfx}", label=_L("국소 상태 편차 δ" + cell_ko, "Local-state deviation δ" + cell_en), unit=ls_unit,
                            values=r[:, c0 + 5], axis="state"))
            if has_E:
                sig.append(dict(key=f"dphi_E{sfx}", label=_L("이미터 상태 편차 δφ_E" + cell_ko, "Emitter-state deviation δφ_E" + cell_en),
                                unit="V", values=r[:, c0 + 6], axis="state"))
    reached = np.isfinite(out.samples[:, 0]).any() if len(out.samples) else False
    if spec.bench == "pbit" and reached:
        v_th = spec.meta["v_th"]
        smp = out.samples[np.isfinite(out.samples[:, 0])]          # clocks reached by the run
        ts = smp[:, 0]
        bits = (smp[:, 2] < v_th).astype(float)
        j = np.searchsorted(ts, r[:, 0], side="right") - 1
        held = np.where(j >= 0, bits[np.clip(j, 0, len(bits) - 1)], np.nan)
        sig.append(dict(key="bit", label=_L("비교기 출력 비트", "Comparator bit"), unit="1", values=held, axis="logic"))
    drain0 = r[:, nodes.index(cells[0]["drain"])]
    traj = dict(vd=drain0, id=r[:, base + 3])
    return dict(t=r[:, 0], signals=sig), traj


# ---- analyses -----------------------------------------------------------------------------------
def _events_of(out, cell: int, kind: int):
    e = out.events
    if len(e) == 0:
        return e
    return e[(e[:, 0] == kind) & (e[:, 1] == cell)]


def _analyse_ramp(outs, meta, cells: list[int]):
    """Per-cycle V_LU / V_LD (drain = device V_DS, and supply) for the given cells.  A cycle the run
    did not complete (truncated run) without a latch-up is censored: NaN and excluded from n_obs."""
    Tc = meta["cycle_s"]
    nc = meta["n_cycles"]
    res = {}
    for k in cells:
        LUd, LUs, LDd, LDs, tLU = [], [], [], [], []
        n_obs = n_cens = 0
        for out in outs:
            up = _events_of(out, k, 1)
            dn = _events_of(out, k, 2)
            for c in range(nc):
                a, b = c * Tc, (c + 1) * Tc
                done = out.t_reached >= b * (1.0 - 1e-9)
                u = up[(up[:, 2] >= a) & (up[:, 2] < b)]
                if len(u):
                    n_obs += 1
                    LUd.append(u[0, 3])
                    LUs.append(u[0, 4])
                    tLU.append(u[0, 2])
                    d = dn[(dn[:, 2] > u[0, 2]) & (dn[:, 2] < b)]
                    if len(d):
                        LDd.append(d[0, 3])
                        LDs.append(d[0, 4])
                    else:
                        LDd.append(np.nan)
                        LDs.append(np.nan)
                else:
                    n_obs += int(done)
                    n_cens += int(not done)
                    LUd.append(np.nan)
                    LUs.append(np.nan)
                    LDd.append(np.nan)
                    LDs.append(np.nan)
                    tLU.append(np.nan)
        res[k] = dict(LUd=np.array(LUd), LUs=np.array(LUs), LDd=np.array(LDd), LDs=np.array(LDs), tLU=np.array(tLU),
                      n_obs=n_obs, n_cens=n_cens)
    return res


def _sample_rows(out, times) -> tuple[np.ndarray, np.ndarray]:
    """Row index into out.samples for each requested sample time and whether the run reached it
    (samples of a truncated run stay NaN)."""
    s = out.samples
    times = np.asarray(times, float)
    if len(s) == 0 or len(times) == 0:
        return np.zeros(len(times), int), np.zeros(len(times), bool)
    ts = s[:, 0]
    ok = np.isfinite(ts)
    order = np.flatnonzero(ok)
    if len(order) == 0:
        return np.zeros(len(times), int), np.zeros(len(times), bool)
    tv = ts[order]
    j = np.clip(np.searchsorted(tv, times - 1e-12 * np.maximum(np.abs(times), 1e-9)), 0, len(tv) - 1)
    hit = np.abs(tv[j] - times) <= 1e-9 * np.maximum(np.abs(times), 1e-9)
    return order[j], hit


def _pulse_bits(out, meta, cell: int, key="top_end"):
    """Reported latch state of the cell at the requested sample times (1 = latched); NaN where the
    run did not get there (censored)."""
    rows, hit = _sample_rows(out, meta[key])
    if len(rows) == 0:
        return np.zeros(0)
    v = out.samples[rows, 3 + 3 * cell].astype(float)
    return np.where(hit, v, np.nan)


def _pbit_bits(out, v_th: float):
    """(comparator bit, reported latch state) per clock; NaN for clocks the run did not reach."""
    s = out.samples
    if len(s) == 0:
        return np.zeros(0), np.zeros(0)
    ok = np.isfinite(s[:, 0])
    bit = np.where(ok, (s[:, 2] < v_th).astype(float), np.nan)
    lat = np.where(ok, s[:, 3], np.nan)
    return bit, lat


def _lag1_pairs(bl) -> float | None:
    """Pooled lag-1 autocorrelation over consecutive pairs that are both observed."""
    pa = [b[:-1][np.isfinite(b[:-1]) & np.isfinite(b[1:])] for b in bl if len(b) > 1]
    pb = [b[1:][np.isfinite(b[:-1]) & np.isfinite(b[1:])] for b in bl if len(b) > 1]
    pa = np.concatenate(pa) if pa else np.zeros(0)
    pb = np.concatenate(pb) if pb else np.zeros(0)
    if len(pa) > 2 and np.std(pa) > 0 and np.std(pb) > 0:
        return float(np.corrcoef(pa, pb)[0, 1])
    return None


def _frac_err(bits) -> tuple[float, float, int]:
    """Mean of the observed bits, its binomial error and the number of observed bits."""
    b = np.asarray(bits, float)
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return np.nan, np.nan, 0
    pm = float(b.mean())
    return pm, float(np.sqrt(max(pm * (1 - pm), 0.0) / len(b))), int(len(b))


def _branch_deviation(traj_vd, traj_id, cl):
    """Median |Δlog10 I_D| of a (deterministic) trajectory w.r.t. the quasi-static HRS / LRS branches
    (``cl`` = checked MODEL.classify result of the cell's device)."""
    if cl is None:
        return None, None
    b, i, j, fold = cl
    out = []
    for part, lo, hi in ((b[:i + 1], None, fold[0]), (b[j:], fold[1], None)):
        V, I = part[:, 0], part[:, 1]
        o = np.argsort(V)
        V, I = V[o], I[o]
        sel = np.isfinite(traj_vd) & (traj_id > 1e-14) & (traj_vd >= V[0]) & (traj_vd <= V[-1])
        if lo is None:
            sel &= traj_id < 1e-9
            sel &= traj_vd < hi - 0.01
        else:
            sel &= traj_id > 1e-7
            sel &= traj_vd > lo + 0.01
        if sel.sum() < 5:
            out.append(None)
            continue
        ref = np.interp(traj_vd[sel], V, np.log10(np.maximum(I, 1e-300)))
        out.append(float(np.median(np.abs(np.log10(traj_id[sel]) - ref))))
    return out[0], out[1]


def _item(key, ko, en, value, unit=None, spread=None):
    d = dict(key=key, label=_L(ko, en), value=None if value is None or (isinstance(value, float) and not np.isfinite(value)) else value)
    if unit is not None:
        d["unit"] = unit
    if spread is not None:
        d["spread"] = None if not np.isfinite(spread) else float(spread)
    return d


def _mean_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None, None
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else None)


# ---- main ------------------------------------------------------------------------------------
def run_circuit(payload: dict, progress: Callable[[float, str], None] | None = None) -> dict:
    tic = time.perf_counter()
    progress = progress or (lambda f, msg="": None)
    payload = dict(payload or {})
    warnings: list[str] = []
    bench = payload.get("bench", "load_line")
    if bench == "custom":                      # user-drawn netlist (WEB_CONTRACT §6)
        from .custom import run_custom
        return run_custom(payload, progress)
    if bench not in B.BENCH_DEFAULTS:
        raise ValueError(f"unknown bench {bench!r}; choose one of {sorted(B.BENCH_DEFAULTS) + ['custom']}")
    mode = payload.get("mode", "deterministic")
    if mode not in ("deterministic", "stochastic"):
        raise ValueError("mode must be 'deterministic' or 'stochastic'")
    device = PR.resolve_device(payload.get("device"))
    preset = device.get("preset") or "paper"
    sweep_preset = PR.resolve_section(preset, "sweep", None)
    bp = B.merged(B.BENCH_DEFAULTS[bench], payload.get("bench_params"))
    unknown = set((payload.get("bench_params") or {}).keys()) - set(B.BENCH_DEFAULTS[bench].keys())
    if unknown:
        warnings.append(f"unknown bench_params ignored: {sorted(unknown)}")
    st = _stochastic(payload, mode, device, warnings)
    det = _detect(payload)
    ls_cfg = parse_local_state(st["local_state"], warnings) if mode == "stochastic" else parse_local_state(None, warnings)
    stochastic = mode == "stochastic"
    carrier = stochastic and st["carrier_noise"]
    if stochastic and not carrier and ls_cfg.mode == "none" and st["n_runs"] > 1:
        warnings.append("stochastic mode with carrier_noise = false and no local states has no random input: "
                        f"running 1 run instead of {st['n_runs']} identical ones (use the deterministic mode)")
        st["n_runs"] = 1
    if stochastic and ls_cfg.mode != "none" and ls_cfg.action == "local_avalanche":
        # same rule as the device-level MC (compute/stochastic.StateMap): the local avalanche path must exist
        if float(device["ext"].get("aloc", 0.0)) <= 0:
            warnings.append("local_avalanche action needs a local avalanche strength ext.aloc (p[21]) > 0; "
                            "using aloc = 1.0 (the verify_hloc.py value)")
            device = dict(device, ext=dict(device["ext"], aloc=1.0))
    if stochastic and ls_cfg.mode != "none" and ls_cfg.action != "gidl":
        warnings.append(f"action point '{ls_cfg.action}' is an experimental hypothesis lever (hypotheses.py), "
                        "not the paper's calibrated GIDL action point")
    progress(0.01, "device branches / folds")

    vg1 = float(bp["vg_V"]) if bp.get("vg_V") is not None else float(device["vg"])
    p1 = _device_p(device, vg=vg1)
    grid = int(np.clip(int(device.get("numerics", {}).get("grid", 601)), 201, 2001))
    prof1 = branch_profile(p1, 301)
    cl, gap = classify_checked(p1, grid)
    folds = dict(V_LU=None, V_LD=None)
    if cl is not None:
        folds = dict(V_LU=float(cl[3][0]), V_LD=float(cl[3][1]))
    elif gap:
        from server.compute.deterministic import GAP_WARNING
        warnings.append(GAP_WARNING + " (channel-on regime): no latch-up expected; I_D above detect.i_threshold_A "
                        "is channel conduction")
    else:
        warnings.append("the device has no latch window at this bias (no two-fold branch): no latch-up expected")
    fold_lu = folds["V_LU"] if folds["V_LU"] is not None else 4.0
    _check_thresholds(det, prof1, warnings)

    # ---- bench parameters (validated, auto values resolved) ----
    def ramp_params(d):
        if d.get("v_max_V") is None:
            d["v_max_V"] = float(sweep_preset["vd_max_V"])
        user_rate = d.get("rate_V_per_s") is not None
        if not user_rate:
            d["rate_V_per_s"] = float(sweep_preset["rate_V_per_s"])
        _ = B._num(d, "v_min_V", -B.CAPS["v_abs_max"], B.CAPS["v_abs_max"])
        _ = B._num(d, "v_max_V", -B.CAPS["v_abs_max"], B.CAPS["v_abs_max"])
        if d["v_max_V"] <= d["v_min_V"]:
            raise ValueError("bench_params.v_max_V must be larger than v_min_V")
        B._num(d, "rate_V_per_s", positive=True, hi=1e9)
        B._int(d, "n_cycles", 1, B.CAPS["n_cycles"], warnings)
        return user_rate

    def pulse_params(d, amp_key="v_amp_V", base_key="v_base_V", width="width_s", period="period_s", n="n_pulses",
                     auto_amp=0.10):
        if d.get(amp_key) is None:
            d[amp_key] = round(fold_lu + auto_amp, 4)
        for key in (amp_key, base_key):
            B._num(d, key, -B.CAPS["v_abs_max"], B.CAPS["v_abs_max"])
        for key in (width, period, "rise_s", "fall_s"):
            B._num(d, key, positive=True, hi=1e3)
        if "delay_s" in d:
            B._num(d, "delay_s", 0.0, 1e3)
        B._int(d, n, 1, B.CAPS["n_clocks" if n == "n_clocks" else "n_pulses"], warnings)
        if d["rise_s"] + d[width] + d["fall_s"] > d[period]:
            raise ValueError("rise + width + fall must not exceed the period")

    specs: list[tuple[str, Any, B.BenchSpec, int]] = []   # (config key, x value, spec, n_runs)
    sweep_meta: dict[str, Any] = {}
    cells_p = [p1]
    if bench == "load_line":
        user_rate = ramp_params(bp)
        B._num(bp, "R_s_ohm", positive=True, hi=1e12)
        B._num(bp, "C_d_F", 0.0, 1e-6)
        if carrier and not user_rate:
            est = _estimate_for(B.build_load_line(bp, p1, vg1, _label(p1)), prof1, stochastic, carrier, st, payload, ls_cfg)
            if est > 0.5 * _max_steps(payload):
                warnings.append(f"event-level stochastic simulation at the preset ramp rate {bp['rate_V_per_s']:g} V/s "
                                f"would need ~{est:.2g} steps per run; using 1200 V/s instead (set bench_params."
                                f"rate_V_per_s explicitly to override)")
                bp["rate_V_per_s"] = 1200.0
        specs.append(("nominal", None, B.build_load_line(bp, p1, vg1, _label(p1)), st["n_runs"]))
    elif bench == "pulse":
        pulse_params(bp)
        B._num(bp, "R_s_ohm", positive=True, hi=1e12)
        B._num(bp, "C_d_F", 0.0, 1e-6)
        amps = B._list(bp, "amplitudes_V", warnings, -B.CAPS["v_abs_max"], B.CAPS["v_abs_max"])
        specs.append(("nominal", bp["v_amp_V"], B.build_pulse(bp, p1, vg1, _label(p1)), st["n_runs"]))
        for a in amps:
            specs.append(("amp", a, B.build_pulse(bp, p1, vg1, _label(p1), amp=a), st["n_runs"]))
    elif bench == "pbit":
        if bp.get("v_high_V") is None:
            bp["v_high_V"] = round(fold_lu - 0.02, 4)
        pulse_params(bp, amp_key="v_high_V", base_key="v_low_V", width="clock_width_s", period="clock_period_s",
                     n="n_clocks")
        B._num(bp, "R_L_ohm", positive=True, hi=1e12)
        B._num(bp, "C_d_F", 0.0, 1e-6)
        if bp.get("cmp_threshold_V") is None:
            bp["cmp_threshold_V"] = round(bp["v_high_V"] - bp["R_L_ohm"] * 1e-7, 6)
        B._num(bp, "cmp_threshold_V", -B.CAPS["v_abs_max"], B.CAPS["v_abs_max"])
        vgl = B._list(bp, "vg_list_V", warnings, -6.0, 3.0)
        ll = B._list(bp, "light_list_pA", warnings, 0.0, 1e6)
        specs.append(("nominal", None, B.build_pbit(bp, p1, vg1, _label(p1)), st["n_runs"]))
        if vgl:
            for v in vgl:
                pv = _device_p(device, vg=v)
                specs.append(("vg", v, B.build_pbit(bp, pv, v, _label(pv)), st["n_runs"]))
        elif ll:
            for lpa in ll:
                pv = _device_p(device, vg=vg1, iph_pA=lpa)
                specs.append(("iph", lpa, B.build_pbit(bp, pv, vg1, _label(pv)), st["n_runs"]))
    elif bench == "coupled":
        if bp["source"] not in ("ramp", "pulse"):
            raise ValueError("bench_params.source must be 'ramp' or 'pulse'")
        vg2 = float(bp["vg2_V"]) if bp.get("vg2_V") is not None else vg1
        iph2 = float(bp["iph2_pA"]) if bp.get("iph2_pA") is not None else None
        p2 = _device_p(device, vg=vg2, iph_pA=iph2)
        cells_p = [p1, p2]
        for key in ("R_s1_ohm", "R_s2_ohm", "R_c_ohm"):
            B._num(bp, key, positive=True, hi=1e12)
        B._num(bp, "C_d_F", 0.0, 1e-6)
        if bp["source"] == "ramp":
            user_rate = ramp_params(bp)
            if carrier and not user_rate:
                est = _estimate_for(B.build_coupled(bp, p1, p2, vg1, vg2, "", ""), prof1, stochastic, carrier, st, payload, ls_cfg)
                if est > 0.5 * _max_steps(payload):
                    warnings.append(f"event-level stochastic simulation at {bp['rate_V_per_s']:g} V/s would need ~{est:.2g} "
                                    "steps per run; using 1200 V/s instead")
                    bp["rate_V_per_s"] = 1200.0
        else:
            pulse_params(bp)
        specs.append(("nominal", None, B.build_coupled(bp, p1, p2, vg1, vg2, _label(p1), _label(p2)), st["n_runs"]))

    ramp = bench == "load_line" or (bench == "coupled" and bp["source"] == "ramp")
    if ramp and folds["V_LU"] is not None and folds["V_LU"] > float(bp["v_max_V"]):
        warnings.append(f"the latch-up fold V_LU = {folds['V_LU']:.3f} V lies above the ramp maximum "
                        f"v_max_V = {float(bp['v_max_V']):g} V: no latch-up expected (a large I_D at this V_G is "
                        "channel/HRS conduction, not a latch)")

    # ---- solver config + feasibility ----
    nominal = specs[0][2]
    t_end = nominal.net.t_end
    sol = _solver(payload, t_end, warnings)
    total_est = 0.0
    est_nominal = None
    profiles: dict[bytes, dict] = {p1.tobytes(): prof1}
    for key, xval, spec, nr in specs:
        wins = []
        for stl in spec.net.STL:
            pk = stl["p"]
            prof = profiles.get(pk.tobytes())
            if prof is None:
                prof = branch_profile(pk, 301)
                profiles[pk.tobytes()] = prof
                if key == "nominal" and len(spec.net.STL) > 1:
                    _check_thresholds(det, prof, warnings, f"cell {len(wins) + 1}")
                if not prof["latch"] and (key != "nominal" or len(wins) > 0):
                    warnings.append(f"{key if key != 'nominal' else 'cell %d' % (len(wins) + 1)}"
                                    f"{' = %g' % xval if key != 'nominal' else ''}: no latch window at this bias"
                                    f"{' (locus gap at the fold, channel-on regime)' if prof.get('gap') else ''}: "
                                    "no latch expected")
            # noise bands (lu_lo, lu_hi, ld_lo, ld_hi) + fold u values (u_i, u_j) of the physical latch state
            wins.append(tuple(noise_bands(prof, ls_cfg, stochastic, sol["noise_z_max"])) + tuple(prof["u_fold"]))
        spec.meta["window"] = np.array(wins, float)
        prof = profiles[spec.net.STL[0]["p"].tobytes()]
        e = _estimate(spec, prof, stochastic, carrier, st, sol, window=wins[0][:4])
        spec.meta["estimated_steps"] = e
        if est_nominal is None:
            est_nominal = e
        total_est += e * nr
        if e > 2.0 * sol["max_steps"]:
            if carrier:
                why = ("event-level carrier noise needs h <= tau_frac*tau_rel (µs) while the drain is inside the noise "
                       "band). Use a faster ramp/shorter waveform, the deterministic mode, carrier_noise = false (local "
                       "states only), or the device-level stochastic MC (hazard method).")
            else:
                why = ("the waveform is long compared with the step limits dt_max / |dv| <= 20 mV x reltol/1e-3 and "
                       "the resolved switching transients). Use a shorter waveform (fewer pulses/clocks/cycles), a "
                       "larger solver.max_steps or a larger solver.reltol.")
            raise ValueError(f"estimated ~{e:.3g} time steps per run exceed 2 x solver.max_steps = "
                             f"{2 * sol['max_steps']:.0f} ({why}")
        if e > 0.5 * sol["max_steps"]:
            warnings.append(f"estimated ~{e:.3g} steps per run is close to solver.max_steps = {sol['max_steps']:.0f}; "
                            "runs may be truncated")
    if total_est > MAX_TOTAL_STEPS:
        raise ValueError(f"estimated total work ~{total_est:.3g} time steps (~{total_est * SECONDS_PER_STEP / 60:.0f} min) "
                         f"exceeds the per-request limit {MAX_TOTAL_STEPS:.0g}; reduce n_runs, sweep points or the waveform length")

    cfg = SolverConfig(
        method=0 if sol["method"] == "BE" else 1, stochastic=stochastic, carrier=carrier,
        max_steps=sol["max_steps"], dt_min=sol["dt_min_s"], dt_max=sol["dt_max_s"], reltol=sol["reltol"],
        tau_frac=sol["tau_frac"], max_events_per_step=sol["max_events_per_step"], noise_dt_min=sol["noise_dt_min_s"],
        gauss_threshold=sol["gauss_threshold"], gauss_tau_min=sol["gauss_tau_min_s"], gauss_tau_frac=sol["gauss_tau_frac"],
        i_threshold=det["i_threshold_A"],
        i_threshold_down=det["i_threshold_down_A"], ls_mode=ls_cfg.mode_code if stochastic else 0,
        ls_idx=ls_cfg.index, ls_sigma=ls_cfg.sigma, ls_tau=ls_cfg.tau_s, lsE_sigma=ls_cfg.sigma_E_V,
        lsE_tau=ls_cfg.tau_E_s, ld_noise=st["ld_carrier_noise"] and carrier,
        dt_rec=t_end / 3000.0, dlni_rec=1.0 if carrier else 0.3,
        h_init=min(sol["dt_max_s"], max(10 * sol["dt_min_s"], 1e-7 * t_end)),
    )
    vr = np.ptp(nominal.net.waves[nominal.main_wave][1]) or 1.0
    cfg.dv_rec = max(vr / 300.0, 1e-4)
    if stochastic and ls_cfg.mode != "none" and ls_cfg.action == "local_avalanche" and float(p1[21]) <= 0:
        warnings.append("local_state.action = local_avalanche acts on p[23] only when the local avalanche path is "
                        "enabled (ext.aloc = p[21] > 0); it has no effect here")

    # ---- run loop ----
    total_runs = sum(nr for *_, nr in specs)
    done_units = 0.0
    all_outs: dict[int, list] = {}
    stored_runs = []
    traj = None
    solver_stats = dict(steps=0, rejected=0, newton_iters=0, runtime_s=0.0)
    regime_time = dict(t_total=0.0, t_drift_fast=0.0, t_gauss=0.0, t_lrs_drift=0.0, t_band_drift=0.0, t_neg_u=0.0,
                       t_neg_r=0.0, steps_by_tier=[0] * 6, newton_by_tier=[0] * 6, rejected_by_tier=[0] * 6)
    min_u, min_r = np.inf, np.inf
    hrs_x = {}                     # config index -> I_D threshold crossings with the body on the HRS
    run_warn: list[str] = []
    for ci_, (key, xval, spec, nr) in enumerate(specs):
        net_c = spec.net.compile()
        outs = []
        for run in range(nr):
            frac0 = done_units / total_runs

            def prog(f, _f0=frac0, _run=run, _ci=ci_):
                progress(min(0.99, 0.02 + 0.96 * (_f0 + f / total_runs)),
                         f"{'config %d/%d, ' % (_ci + 1, len(specs)) if len(specs) > 1 else ''}run {_run + 1}/{nr}")

            ls0 = draw_local_states(ls_cfg, net_c["nS"], st["seed"], run) if stochastic else None
            seed = (st["seed"] + 1_000_003 * run) % (2 ** 31 - 1)
            out = simulate(net_c, cfg, net_c["P"], spec.net.t_end, spec.main_wave, seed, ls0, prog,
                           window=spec.meta.get("window"))
            done_units += 1
            outs.append(out)
            solver_stats["steps"] += out.steps
            solver_stats["rejected"] += out.rejected
            solver_stats["newton_iters"] += out.newton_iters
            solver_stats["runtime_s"] += out.runtime_s
            regime_time["t_total"] += out.t_reached
            regime_time["t_drift_fast"] += out.t_unresolved
            regime_time["t_gauss"] += out.t_gauss
            regime_time["t_lrs_drift"] += out.t_lrs_drift
            regime_time["t_band_drift"] += out.t_band_drift
            regime_time["t_neg_u"] += out.t_neg_u
            regime_time["t_neg_r"] += out.t_neg_r
            for tier in range(6):
                regime_time["steps_by_tier"][tier] += out.diag[3 * tier]
                regime_time["newton_by_tier"][tier] += out.diag[3 * tier + 1]
                regime_time["rejected_by_tier"][tier] += out.diag[3 * tier + 2]
            min_u, min_r = min(min_u, out.min_u), min(min_r, out.min_r)
            hrs_x[ci_] = hrs_x.get(ci_, 0) + out.hrs_crossings
            for w in out.warnings:
                run_warn.append(f"run {run}" + (f" ({key} = {xval:g})" if key != "nominal" else "") + f": {w}")
            if ci_ == 0 and run < (MAX_STORED_RUNS if stochastic else 1):
                sigs, tr = _signals(spec, net_c, out, ACTION_UNIT[ls_cfg.action], stochastic and ls_cfg.mode != "none",
                                    ls_cfg.sigma_E_V > 0, cfg.i_floor, MAX_POINTS if run == 0 else MAX_POINTS_OTHER)
                stored_runs.append(dict(run=run, **sigs))
                if run == 0:
                    traj = tr
        all_outs[ci_] = outs
    warnings.extend(run_warn[:20])
    if len(run_warn) > 20:
        warnings.append(f"... {len(run_warn) - 20} more run warnings")
    if min_r < 0:
        warnings.append(f"the drain junction became forward biased (min r = {min_r:.3f} V, {regime_time['t_neg_r']:.3g} s "
                        "summed over runs): symmetric forward drain-diode extension used (outside the calibrated model)")
    if min_u < 0:
        warnings.append(f"the source junction became reverse biased (min u = {min_u * 1e3:.2f} mV, "
                        f"{regime_time['t_neg_u']:.3g} s summed over runs): low-injection diode extension used for u < 0")
    n_hrs_x = sum(hrs_x.values())
    if n_hrs_x:
        where = [f"{specs[c][0]} = {specs[c][1]:g}" if specs[c][0] != "nominal" else "nominal" for c in sorted(hrs_x) if hrs_x[c]]
        warnings.append(f"I_D crossed detect.i_threshold_A = {det['i_threshold_A']:.3g} A {n_hrs_x} time(s) while the body "
                        f"stayed on the HRS ({', '.join(where[:6])}): channel/HRS conduction, not a latch-up; such "
                        "crossings are not counted (latch events require the body to switch branch)")
    if carrier and regime_time["t_drift_fast"] > 0:
        warnings.append(f"carrier noise not resolved (drift only) during {regime_time['t_drift_fast']:.3g} s of "
                        f"{regime_time['t_total']:.3g} s simulated (states relaxing faster than gauss_tau_min / noise_dt_min)")

    # ---- analysis ----
    main_outs = all_outs[0]
    spec0 = specs[0][2]
    events: list[dict] = []
    summary: list[dict] = []
    distributions: list[dict] = []
    sweeps: list[dict] = []

    def add_events(outs, extra: Callable[[int, np.ndarray], dict] | None = None):
        for run, out in enumerate(outs):
            for e in out.events:
                if len(events) >= MAX_EVENTS_OUT:
                    return
                d = dict(run=run, kind="latch_up" if e[0] == 1 else "latch_down", t=float(e[2]), value=float(e[3]),
                         v_src=float(e[4]), v_d=float(e[3]), cell=int(e[1]) + 1)
                if extra:
                    d.update(extra(run, e))
                events.append(d)

    if bench in ("load_line",) or (bench == "coupled" and bp["source"] == "ramp"):
        Tc = spec0.meta["cycle_s"]
        add_events(main_outs, lambda run, e: dict(cycle=int(e[2] // Tc)))
        cells = list(range(len(spec0.meta["cells"])))
        res = _analyse_ramp(main_outs, spec0.meta, cells)
        n_cens = sum(res[k]["n_cens"] for k in cells)
        if n_cens:
            warnings.append(f"{n_cens} cycle(s) were not completed (truncated runs) and are excluded from V_LU / V_LD "
                            "statistics and from n_latch_up")
        for k in cells:
            sfx = f"_{k + 1}" if len(cells) > 1 else ""
            cko = f" (셀 {k + 1})" if len(cells) > 1 else ""
            cen = f" (cell {k + 1})" if len(cells) > 1 else ""
            r = res[k]
            m_lu, s_lu = _mean_sd(r["LUd"])
            m_ld, s_ld = _mean_sd(r["LDd"])
            m_lus, s_lus = _mean_sd(r["LUs"])
            m_lds, s_lds = _mean_sd(r["LDs"])
            summary += [
                _item("V_LU" + sfx, "래치업 전압 V_LU (드레인)" + cko, "Latch-up voltage V_LU (drain)" + cen, m_lu, "V", s_lu),
                _item("V_LD" + sfx, "래치다운 전압 V_LD (드레인)" + cko, "Latch-down voltage V_LD (drain)" + cen, m_ld, "V", s_ld),
                _item("window" + sfx, "히스테리시스 창 V_LU − V_LD" + cko, "Hysteresis window V_LU − V_LD" + cen,
                      (m_lu - m_ld) if (m_lu is not None and m_ld is not None) else None, "V"),
                _item("V_LU_src" + sfx, "래치업 시 전원 전압" + cko, "Supply voltage at latch-up" + cen, m_lus, "V", s_lus),
                _item("V_LD_src" + sfx, "래치다운 시 전원 전압" + cko, "Supply voltage at latch-down" + cen, m_lds, "V", s_lds),
                _item("n_latch_up" + sfx, "래치업 횟수 / 관측 사이클" + cko, "Latch-ups / observed cycles" + cen,
                      f"{int(np.isfinite(r['LUd']).sum())}/{r['n_obs']}"),
            ]
            distributions += [
                dict(key="V_LU" + sfx, label=_L("V_LU (드레인 노드)" + cko, "V_LU (drain node)" + cen), unit="V", values=r["LUd"]),
                dict(key="V_LD" + sfx, label=_L("V_LD (드레인 노드)" + cko, "V_LD (drain node)" + cen), unit="V", values=r["LDd"]),
                dict(key="V_LU_src" + sfx, label=_L("V_LU (전원)" + cko, "V_LU (supply)" + cen), unit="V", values=r["LUs"]),
                dict(key="V_LD_src" + sfx, label=_L("V_LD (전원)" + cko, "V_LD (supply)" + cen), unit="V", values=r["LDs"]),
            ]
        if len(cells) == 2:
            a, b = res[0]["LUd"], res[1]["LUd"]
            ok = np.isfinite(a) & np.isfinite(b)
            corr = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else None
            summary.append(_item("corr_LU", "두 셀 V_LU 상관계수", "Correlation of V_LU between cells", corr))
            dt = res[1]["tLU"] - res[0]["tLU"]
            summary.append(_item("dt_LU", "래치업 시간차 (셀2 − 셀1)", "Latch-up time difference (cell 2 − cell 1)",
                                 *(_mean_sd(dt)[0], "s", _mean_sd(dt)[1])))
        summary += [
            _item("fold_V_LU", "정상상태 폴드 V_LU (준정적)", "Steady-state fold V_LU (quasi-static)", folds["V_LU"], "V"),
            _item("fold_V_LD", "정상상태 폴드 V_LD (준정적)", "Steady-state fold V_LD (quasi-static)", folds["V_LD"], "V"),
        ]
        m_lu = _mean_sd(res[0]["LUd"])[0]
        m_ld = _mean_sd(res[0]["LDd"])[0]
        if folds["V_LU"] is not None and m_lu is not None:
            summary.append(_item("lag_LU", "V_LU − 폴드 (램프 지연/잡음)", "V_LU − fold (ramp lag / noise)", m_lu - folds["V_LU"], "V"))
        if folds["V_LD"] is not None and m_ld is not None:
            summary.append(_item("lag_LD", "V_LD − 폴드", "V_LD − fold", m_ld - folds["V_LD"], "V"))
        if not stochastic and traj is not None and bench == "load_line":
            dh, dl = _branch_deviation(np.asarray(traj["vd"]), np.asarray(traj["id"]), cl)
            summary.append(_item("hrs_branch_dev", "HRS 가지 대비 |Δlog10 I_D| 중앙값", "HRS branch deviation, median |Δlog10 I_D|", dh, "dec"))
            summary.append(_item("lrs_branch_dev", "LRS 가지 대비 |Δlog10 I_D| 중앙값", "LRS branch deviation, median |Δlog10 I_D|", dl, "dec"))
    elif bench == "pulse" or (bench == "coupled" and bp["source"] == "pulse"):
        starts = np.asarray(spec0.meta["starts"])
        per = bp["period_s"]
        add_events(main_outs, lambda run, e: dict(pulse=int(np.clip(np.searchsorted(starts, e[2], side="right") - 1, 0, len(starts) - 1))))
        cells = list(range(len(spec0.meta["cells"])))
        bits_cells = []
        n_cens = 0
        for k in cells:
            sfx = f"_{k + 1}" if len(cells) > 1 else ""
            cko = f" (셀 {k + 1})" if len(cells) > 1 else ""
            cen = f" (cell {k + 1})" if len(cells) > 1 else ""
            # latched = reported latch state at the end of the flat top (NaN = pulse not reached: censored)
            sw = [_pulse_bits(o, spec0.meta, k) for o in main_outs]
            n_cens += int(sum(np.sum(~np.isfinite(b)) for b in sw))
            bits_cells.append(np.concatenate(sw) if sw else np.zeros(0))
            per_run = np.array([np.nanmean(b) if np.isfinite(b).any() else np.nan for b in sw])
            summary.append(_item("P_sw" + sfx, "스위칭 확률 P_sw (펄스 끝)" + cko, "Switching probability P_sw (end of pulse)" + cen,
                                 float(np.nanmean(per_run)) if np.isfinite(per_run).any() else None, "1",
                                 float(np.nanstd(per_run, ddof=1)) if np.isfinite(per_run).sum() > 1 else None))
            if bench == "pulse":
                ret = [_pulse_bits(o, spec0.meta, k, key="per_sample") for o in main_outs]
                rr = np.array([np.nanmean(b) if np.isfinite(b).any() else np.nan for b in ret])
                summary.append(_item("P_retained", "주기 끝까지 래치 유지 확률", "Still latched at the end of the period",
                                     float(np.nanmean(rr)) if np.isfinite(rr).any() else None, "1"))
            delays = []
            for o in main_outs:
                up = _events_of(o, k, 1)
                for i0, t0 in enumerate(starts):
                    u = up[(up[:, 2] >= t0) & (up[:, 2] < t0 + per)]
                    delays.append(u[0, 2] - t0 if len(u) else np.nan)
            md, sd = _mean_sd(delays)
            summary.append(_item("delay" + sfx, "스위칭 지연 (펄스 시작 기준)" + cko, "Switching delay (from pulse start)" + cen, md, "s", sd))
            distributions.append(dict(key="delay" + sfx, label=_L("스위칭 지연" + cko, "Switching delay" + cen), unit="s",
                                      values=np.array(delays)))
            for run, s in enumerate(sw[:MAX_STORED_RUNS]):
                for i0, bit in enumerate(s):
                    if np.isfinite(bit):
                        events.append(dict(run=run, kind="pulse", t=float(spec0.meta["top_end"][i0]), value=float(bit),
                                           cell=k + 1, pulse=i0))
        if n_cens:
            warnings.append(f"{n_cens} pulse(s) were not reached (truncated runs) and are excluded from P_sw / P_retained")
        if len(cells) == 2 and len(bits_cells[0]) == len(bits_cells[1]) and len(bits_cells[0]):
            ok = np.isfinite(bits_cells[0]) & np.isfinite(bits_cells[1])
            a, b = bits_cells[0][ok], bits_cells[1][ok]
            summary.append(_item("P_both", "두 셀 동시 스위칭 확률", "Both cells switched",
                                 float(np.mean(a * b)) if len(a) else None, "1"))
            corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else None
            summary.append(_item("corr_sw", "두 셀 스위칭 상관계수", "Switching correlation between cells", corr))
        summary.append(_item("amplitude", "펄스 진폭", "Pulse amplitude", float(bp["v_amp_V"]), "V"))
        summary.append(_item("fold_V_LU", "정상상태 폴드 V_LU", "Steady-state fold V_LU", folds["V_LU"], "V"))
        if bench == "pulse" and len(specs) > 1:
            xs, ys, es = [], [], []
            for ci_, (key, xval, spec, nr) in enumerate(specs[1:], start=1):
                pm, err, _n = _frac_err(np.concatenate([_pulse_bits(o, spec.meta, 0) for o in all_outs[ci_]]))
                xs.append(xval)
                ys.append(pm)
                es.append(err)
            sweeps.append(dict(key="P_sw_vs_amplitude", label=_L("펄스 진폭에 따른 스위칭 확률", "Switching probability vs pulse amplitude"),
                               x=xs, x_label="V_amp", x_unit="V", y=ys, y_label="P_sw", y_unit="1", y_err=es))
    elif bench == "pbit":
        v_th = bp["cmp_threshold_V"]
        add_events(main_outs)

        pairs = [_pbit_bits(o, v_th) for o in main_outs]
        bl = [b for b, _ in pairs]
        ll = [lt for _, lt in pairs]
        allb = np.concatenate(bl) if bl else np.zeros(0)
        alll = np.concatenate(ll) if ll else np.zeros(0)
        obs = np.isfinite(allb)
        n_cens = int(np.sum(~obs))
        if n_cens:
            warnings.append(f"{n_cens} clock(s) were not reached (truncated runs) and are excluded from P(1) / lag1")
        per_run = np.array([np.nanmean(b) if np.isfinite(b).any() else np.nan for b in bl])
        lag1 = _lag1_pairs(bl)
        p1, _e, n_bits = _frac_err(allb)
        pl, _e, _n = _frac_err(alll)
        summary += [
            _item("P1", "P(1) (비교기 출력)", "P(1) (comparator output)", p1 if n_bits else None, "1",
                  float(np.nanstd(per_run, ddof=1)) if np.isfinite(per_run).sum() > 1 else None),
            _item("P_latched", "래치 상태 비율 (클럭 high 끝)", "Latched fraction (end of clock high)", pl if n_bits else None, "1"),
            _item("lag1", "비트열 lag-1 자기상관", "Bit lag-1 autocorrelation", lag1),
            _item("n_bits", "비트 수", "Number of bits", int(n_bits)),
            _item("v_th", "비교기 문턱 전압", "Comparator threshold", float(v_th), "V"),
            _item("v_high", "클럭 high 전압", "Clock high level", float(bp["v_high_V"]), "V"),
            _item("fold_V_LU", "정상상태 폴드 V_LU", "Steady-state fold V_LU", folds["V_LU"], "V"),
        ]

        def _mismatch(bits, lat, label):
            mask = np.isfinite(bits) & np.isfinite(lat) & (bits > 0.5) & (lat < 0.5)
            if mask.any():
                warnings.append(f"{label}: the comparator read 1 in {int(mask.sum())} of {int(np.isfinite(bits).sum())} clocks "
                                "while the cell was not latched (channel/HRS conduction pulls v_D below v_th): P(1) is "
                                "then not a latch probability")

        _mismatch(allb, alll, "nominal")
        for run, b in enumerate(bl[:MAX_STORED_RUNS]):
            ts = main_outs[run].samples[:, 0]
            for i0, bit in enumerate(b):
                if np.isfinite(bit):
                    events.append(dict(run=run, kind="bit", t=float(ts[i0]), value=float(bit), cell=1, clock=i0))
        if len(specs) > 1:
            xs, ys, es = [], [], []
            for ci_, (key, xval, spec, nr) in enumerate(specs[1:], start=1):
                pr = [_pbit_bits(o, v_th) for o in all_outs[ci_]]
                bits = np.concatenate([b for b, _ in pr])
                _mismatch(bits, np.concatenate([lt for _, lt in pr]), f"{key} = {xval:g}")
                pm, err, _n = _frac_err(bits)
                xs.append(xval)
                ys.append(pm)
                es.append(err)
            key = specs[1][0]
            sweeps.append(dict(key="P1_vs_vg" if key == "vg" else "P1_vs_iph",
                               label=_L("V_G에 따른 P(1)" if key == "vg" else "광전류에 따른 P(1)",
                                        "P(1) vs V_G" if key == "vg" else "P(1) vs photocurrent"),
                               x=xs, x_label="V_G" if key == "vg" else "I_PH", x_unit="V" if key == "vg" else "pA",
                               y=ys, y_label="P(1)", y_unit="1", y_err=es))

    if carrier:
        tt = max(regime_time["t_total"], 1e-300)
        summary.append(_item("t_noise_resolved_frac", "잡음 분해 시간 비율 (사건 수준+가우스)",
                             "Fraction of time with resolved carrier noise",
                             float(np.clip(1.0 - (regime_time["t_drift_fast"] + regime_time["t_lrs_drift"]
                                                  + regime_time["t_band_drift"]) / tt, 0.0, 1.0)),
                             "1"))
    summary.append(_item("runs", "실행 수", "Runs", int(sum(nr for *_, nr in specs))))
    summary.append(_item("steps_per_run", "실행당 시간 스텝", "Time steps per run",
                         float(solver_stats["steps"] / max(total_runs, 1))))

    runtime = time.perf_counter() - tic
    solver_stats["runtime_s"] = float(solver_stats["runtime_s"])
    progress(1.0, "done")
    return dict(
        bench=bench, mode=mode, runs=stored_runs, events=events, summary=summary, distributions=distributions,
        sweeps=sweeps, trajectory=traj, schematic=spec0.net.schematic(), solver_stats=solver_stats,
        bench_params=_jsonable(bp), solver=_jsonable(sol), detect=det,
        stochastic=_jsonable(dict(st, local_state=vars(ls_cfg))) if stochastic else None,
        folds=folds, feasibility=dict(estimated_steps_per_run=est_nominal, estimated_total_steps=total_est,
                                      estimated_runtime_s=total_est * SECONDS_PER_STEP, total_runs=total_runs),
        regimes=regime_time, runtime_s=runtime, warnings=warnings,
    )


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d


def _max_steps(payload: dict) -> float:
    try:
        return min(float((payload.get("solver") or {}).get("max_steps", B.SOLVER_DEFAULTS["max_steps"])), B.CAPS["max_steps"])
    except (TypeError, ValueError):
        return B.SOLVER_DEFAULTS["max_steps"]


def _estimate(spec: B.BenchSpec, prof: dict, stochastic: bool, carrier: bool, st: dict, sol: dict,
              window=(-np.inf, np.inf, -np.inf, np.inf)) -> float:
    tw, vw = spec.net.waves[spec.main_wave]
    sc = sol["reltol"] / 1e-3
    return estimate_steps(tw, vw, spec.net.t_end, prof, stochastic, carrier, sol["dt_max_s"],
                          float(np.clip(0.02 * sc, 1e-3, 0.2)), sol["tau_frac"], sol["max_events_per_step"],
                          sol["noise_dt_min_s"], len(spec.net.breakpoints()),
                          ld_noise=bool(st.get("ld_carrier_noise")) and carrier,
                          gauss_tau_min=float(sol.get("gauss_tau_min_s", 2e-9)), gauss_tau_frac=float(sol.get("gauss_tau_frac", 0.5)),
                          window=window)


def _estimate_for(spec: B.BenchSpec, prof: dict, stochastic: bool, carrier: bool, st: dict, payload: dict,
                  ls_cfg=None) -> float:
    sol = B.merged(B.SOLVER_DEFAULTS, payload.get("solver"))
    t_end = spec.net.t_end
    sol["dt_max_s"] = float(sol["dt_max_s"]) if sol.get("dt_max_s") else t_end / 2000.0
    sol["reltol"] = float(sol["reltol"])
    for k in ("tau_frac", "max_events_per_step", "noise_dt_min_s", "gauss_tau_min_s", "gauss_tau_frac", "noise_z_max"):
        sol[k] = float(sol[k])
    return _estimate(spec, prof, stochastic, carrier, st, sol,
                     window=noise_bands(prof, ls_cfg, stochastic, sol["noise_z_max"]))
