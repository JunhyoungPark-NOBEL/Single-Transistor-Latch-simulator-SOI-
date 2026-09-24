"""Circuit test benches: documented defaults, netlist builders and per-bench analysis.

This module is pure Python/numpy (no numba import), so the API process may import
``BENCH_DEFAULTS`` / ``BENCH_INFO`` / ``SOLVER_DEFAULTS`` for documentation.

All voltages in V, times in s, resistances in ohm, capacitances in F, currents in A
(light: pA, as in the device block).  ``None`` means "automatic" (resolved value returned in
``result["bench_params"]``).

Benches
-------
load_line  V_src(t) triangular v_min -> v_max -> v_min at rate_V_per_s, n_cycles; series R_s;
           drain capacitance C_d; DC gate V_G.  Latch-up / latch-down detected per cycle as I_D
           crossing detect.i_threshold_A upward / downward; V_LU, V_LD at the drain node (device
           V_DS) and at the source (supply).  The body charge Q_B is integrated continuously across
           cycles, so any residual body memory between cycles is included automatically.
pulse      supply pulse train (v_base, v_amp, width = flat top, period, rise, fall, n_pulses)
           through R_s; P_sw = fraction of pulses latched (I_D >= i_threshold) at the end of the flat
           top; also the fraction still latched at the end of the period (retention) and the
           switching delay.  Optional amplitude list -> sweep of P_sw vs amplitude.
pbit       STL + load resistor R_L from a clocked supply; comparator on v_D with threshold
           (default v_high - R_L * 100 nA): bit = 1 when v_D < v_th at the end of the clock high
           phase (cell latched).  Statistics P(1) and lag-1 autocorrelation; optional list of V_G or
           light values -> sweep of P(1).
coupled    two STL cells, each with its own series R to a common ramp or pulse source, coupled by R_c
           between the drains; the second cell may override V_G / light.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .netlist import Netlist, pulse_train, triangle

BENCH_INFO = {
    "load_line": {"ko": "부하선 스윕 (직렬 저항 + 삼각파)", "en": "Load-line sweep (series R, triangular ramp)"},
    "pulse": {"ko": "펄스 열 (스위칭 확률 P_sw)", "en": "Pulse train (switching probability P_sw)"},
    "pbit": {"ko": "p-비트 (부하 저항 + 클럭 + 비교기)", "en": "p-bit (load resistor, clock, comparator)"},
    "coupled": {"ko": "저항 결합 STL 두 개", "en": "Two resistively coupled STLs"},
}

BENCH_DEFAULTS: dict[str, dict[str, Any]] = {
    "load_line": dict(
        v_min_V=0.0,            # ramp start/end voltage
        v_max_V=None,           # auto: device preset sweep vd_max (paper 4 V, photo 5 V)
        rate_V_per_s=None,      # auto: preset sweep rate (paper 0.4 V/s, photo 1200 V/s); stochastic
                                #       event-level runs fall back to 1200 V/s when the preset rate is too slow
        n_cycles=1,             # triangular cycles (<= 50)
        R_s_ohm=1e3,            # series resistor
        C_d_F=2e-15,            # drain-node capacitance to ground
        vg_V=None,              # gate voltage (default: device.vg)
    ),
    "pulse": dict(
        v_base_V=0.0,
        v_amp_V=None,           # auto: deterministic latch-up fold V_LU of the device + 0.10 V
        width_s=200e-6,         # flat top
        period_s=1e-3,
        rise_s=1e-6,
        fall_s=1e-6,
        n_pulses=10,            # <= 2000
        delay_s=0.0,            # start of the first pulse
        R_s_ohm=1e3,
        C_d_F=2e-15,
        vg_V=None,
        amplitudes_V=[],        # optional sweep of v_amp (P_sw vs amplitude)
    ),
    "pbit": dict(
        v_low_V=0.0,
        v_high_V=None,          # auto: fold V_LU - 0.02 V (stochastic switching regime)
        clock_period_s=1e-3,
        clock_width_s=200e-6,   # flat top of the clock
        rise_s=1e-6,
        fall_s=1e-6,
        n_clocks=50,            # <= 5000
        R_L_ohm=100e3,
        C_d_F=2e-15,
        cmp_threshold_V=None,   # auto: v_high - R_L * 100 nA
        vg_V=None,
        vg_list_V=[],           # optional sweep of V_G -> P(1)
        light_list_pA=[],       # optional sweep of I_PH -> P(1) (used when vg_list_V is empty)
    ),
    "coupled": dict(
        source="ramp",          # "ramp" | "pulse"
        v_min_V=0.0, v_max_V=None, rate_V_per_s=None, n_cycles=1,               # ramp source
        v_base_V=0.0, v_amp_V=None, width_s=200e-6, period_s=1e-3, rise_s=1e-6,  # pulse source
        fall_s=1e-6, n_pulses=10, delay_s=0.0,
        R_s1_ohm=100e3, R_s2_ohm=100e3,     # series resistors of cell 1 / cell 2
        R_c_ohm=1e6,                        # coupling resistor between the drains
        C_d_F=2e-15,                        # each drain node
        vg_V=None,                          # cell 1 gate (default device.vg)
        vg2_V=None,                         # cell 2 gate (default = cell 1)
        iph2_pA=None,                       # cell 2 photocurrent (default = device light)
    ),
}

SOLVER_DEFAULTS = dict(
    method="BE",                # "BE" | "TRAP" (TRAP falls back to BE on stiff steps h > 2 tau_rel)
    dt_min_s=None,              # auto: max(1e-15, 1e-13 * t_end)
    dt_max_s=None,              # auto: t_end / 2000
    reltol=1e-3,                # scales the per-step limits: |du| <= 10 mV, |dln I| <= 0.2, |dv| <= 20 mV,
                                # LTE(u) <= 1 mV at reltol = 1e-3 (linear in reltol, clipped)
    max_steps=1_000_000,        # per run (cap 2e6)
    tau_frac=0.05,              # stochastic: h <= tau_frac * tau_rel
    max_events_per_step=200,    # stochastic: expected events per step
    noise_dt_min_s=2e-9,        # stochastic: carrier noise resolved where tau_frac*tau_rel >= this; drift-only elsewhere
    gauss_threshold=100,        # stochastic: Poisson counts with mean > this use the Gaussian limit
)

STOCHASTIC_DEFAULTS = dict(seed=2026092920, n_runs=20, carrier_noise=True, local_state=dict(mode="none"))

CAPS = dict(max_steps=2_000_000, n_runs=200, n_cycles=50, n_pulses=2000, n_clocks=5000, sweep_points=25,
            v_abs_max=8.0)


def merged(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merged(out[k], v)
        else:
            out[k] = v
    return out


def _num(d: dict, key: str, lo: float | None = None, hi: float | None = None, positive=False) -> float:
    try:
        v = float(d[key])
    except (TypeError, ValueError):
        raise ValueError(f"bench_params.{key} must be a number")
    if not np.isfinite(v):
        raise ValueError(f"bench_params.{key} must be finite")
    if positive and v <= 0:
        raise ValueError(f"bench_params.{key} must be > 0")
    if lo is not None and v < lo:
        raise ValueError(f"bench_params.{key} must be >= {lo}")
    if hi is not None and v > hi:
        raise ValueError(f"bench_params.{key} must be <= {hi}")
    return v


def _int(d: dict, key: str, lo: int, hi: int, warnings: list[str]) -> int:
    try:
        v = int(round(float(d[key])))
    except (TypeError, ValueError):
        raise ValueError(f"bench_params.{key} must be an integer")
    if v < lo:
        raise ValueError(f"bench_params.{key} must be >= {lo}")
    if v > hi:
        warnings.append(f"bench_params.{key} = {v} capped at {hi}")
        v = hi
    d[key] = v
    return v


def _list(d: dict, key: str, warnings: list[str], lo=None, hi=None) -> list[float]:
    raw = d.get(key) or []
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"bench_params.{key} must be a list of numbers")
    vals = []
    for x in raw:
        try:
            v = float(x)
        except (TypeError, ValueError):
            raise ValueError(f"bench_params.{key} must contain numbers")
        if not np.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
            raise ValueError(f"bench_params.{key}: value {x} out of range [{lo}, {hi}]")
        vals.append(v)
    if len(vals) > CAPS["sweep_points"]:
        warnings.append(f"bench_params.{key}: only the first {CAPS['sweep_points']} values are used")
        vals = vals[:CAPS["sweep_points"]]
    d[key] = vals
    return vals


# ---- builders -------------------------------------------------------------------------------
class BenchSpec:
    """A built bench: netlist + everything the analysis needs."""

    def __init__(self, bench: str, net: Netlist, main_wave: int, meta: dict):
        self.bench = bench
        self.net = net
        self.main_wave = main_wave
        self.meta = meta


def _add_cell(net: Netlist, k: int, drain: str, gate: str, p, vg: float, label: str):
    net.add_V(f"VG{k}", gate, "0", [0.0], [vg], f"{vg:g} V (DC)")
    net.add_STL(f"X{k}", drain, gate, "0", p, label=label)


def build_load_line(bp: dict, p, vg: float, label: str) -> BenchSpec:
    net = Netlist()
    t, v, Tc = triangle(bp["v_min_V"], bp["v_max_V"], bp["rate_V_per_s"], bp["n_cycles"])
    w = net.add_V("Vsrc", "src", "0", t, v, f"triangle {bp['v_min_V']:g}→{bp['v_max_V']:g} V, {bp['rate_V_per_s']:g} V/s")
    net.add_R("Rs", "src", "d", bp["R_s_ohm"])
    net.add_C("Cd", "d", "0", bp["C_d_F"])
    _add_cell(net, 1, "d", "g", p, vg, label)
    net.t_end = float(t[-1])
    net.main_wave = w
    return BenchSpec("load_line", net, w, dict(cycle_s=Tc, n_cycles=bp["n_cycles"], cells=[dict(drain="d", name="X1")]))


def _pulse_wave(bp: dict, amp: float):
    return pulse_train(bp["v_base_V"], amp, bp["width_s"], bp["period_s"], bp["rise_s"], bp["fall_s"],
                       bp["n_pulses"], bp["delay_s"])


def build_pulse(bp: dict, p, vg: float, label: str, amp: float | None = None) -> BenchSpec:
    net = Netlist()
    a = bp["v_amp_V"] if amp is None else amp
    t, v, top_end, per_end = _pulse_wave(bp, a)
    w = net.add_V("Vsrc", "src", "0", t, v, f"pulses {bp['v_base_V']:g}→{a:g} V, {bp['width_s']:.3g} s / {bp['period_s']:.3g} s")
    net.add_R("Rs", "src", "d", bp["R_s_ohm"])
    net.add_C("Cd", "d", "0", bp["C_d_F"])
    _add_cell(net, 1, "d", "g", p, vg, label)
    # sample just before the fall (end of the flat top) and at the end of each period
    per_sample = [te - 1e-6 * bp["period_s"] for te in per_end]
    net.samples = sorted(set(top_end) | set(per_sample))
    net.t_end = float(t[-1])
    net.main_wave = w
    starts = [bp["delay_s"] + i * bp["period_s"] for i in range(bp["n_pulses"])]
    return BenchSpec("pulse", net, w, dict(top_end=top_end, per_sample=per_sample, starts=starts, amp=a,
                                           cells=[dict(drain="d", name="X1")]))


def build_pbit(bp: dict, p, vg: float, label: str) -> BenchSpec:
    net = Netlist()
    t, v, top_end, per_end = pulse_train(bp["v_low_V"], bp["v_high_V"], bp["clock_width_s"], bp["clock_period_s"],
                                         bp["rise_s"], bp["fall_s"], bp["n_clocks"], 0.0)
    w = net.add_V("Vclk", "clk", "0", t, v, f"clock {bp['v_low_V']:g}/{bp['v_high_V']:g} V, {bp['clock_period_s']:.3g} s")
    net.add_R("RL", "clk", "d", bp["R_L_ohm"])
    net.add_C("Cd", "d", "0", bp["C_d_F"])
    _add_cell(net, 1, "d", "g", p, vg, label)
    net.add_CMP("CMP", "d", bp["cmp_threshold_V"], f"bit = [v_D < {bp['cmp_threshold_V']:.4g} V] at clock-high end")
    net.samples = list(top_end)
    net.t_end = float(t[-1])
    net.main_wave = w
    return BenchSpec("pbit", net, w, dict(top_end=top_end, v_th=bp["cmp_threshold_V"], cells=[dict(drain="d", name="X1")]))


def build_coupled(bp: dict, p1, p2, vg1: float, vg2: float, label1: str, label2: str) -> BenchSpec:
    net = Netlist()
    meta: dict[str, Any] = dict(source=bp["source"])
    if bp["source"] == "ramp":
        t, v, Tc = triangle(bp["v_min_V"], bp["v_max_V"], bp["rate_V_per_s"], bp["n_cycles"])
        lab = f"triangle {bp['v_min_V']:g}→{bp['v_max_V']:g} V, {bp['rate_V_per_s']:g} V/s"
        meta.update(cycle_s=Tc, n_cycles=bp["n_cycles"])
    else:
        t, v, top_end, per_end = _pulse_wave(bp, bp["v_amp_V"])
        lab = f"pulses {bp['v_base_V']:g}→{bp['v_amp_V']:g} V"
        net.samples = list(top_end)
        meta.update(top_end=top_end, starts=[bp["delay_s"] + i * bp["period_s"] for i in range(bp["n_pulses"])])
    w = net.add_V("Vsrc", "src", "0", t, v, lab)
    net.add_R("Rs1", "src", "d1", bp["R_s1_ohm"])
    net.add_R("Rs2", "src", "d2", bp["R_s2_ohm"])
    net.add_R("Rc", "d1", "d2", bp["R_c_ohm"])
    net.add_C("Cd1", "d1", "0", bp["C_d_F"])
    net.add_C("Cd2", "d2", "0", bp["C_d_F"])
    _add_cell(net, 1, "d1", "g1", p1, vg1, label1)
    _add_cell(net, 2, "d2", "g2", p2, vg2, label2)
    net.t_end = float(t[-1])
    net.main_wave = w
    meta["cells"] = [dict(drain="d1", name="X1"), dict(drain="d2", name="X2")]
    return BenchSpec("coupled", net, w, meta)


# ---- statistics helpers ---------------------------------------------------------------------
def stats(values) -> dict:
    x = np.asarray([np.nan if v is None else v for v in values], float)
    fin = x[np.isfinite(x)]
    out = dict(n=int(len(fin)), mean=None, sd=None, median=None, p05=None, p95=None, min=None, max=None,
               censored=int(np.sum(~np.isfinite(x))), lag1=None)
    if len(fin):
        out.update(mean=float(fin.mean()), median=float(np.median(fin)), p05=float(np.quantile(fin, 0.05)),
                   p95=float(np.quantile(fin, 0.95)), min=float(fin.min()), max=float(fin.max()))
    if len(fin) > 1:
        out["sd"] = float(fin.std(ddof=1))
    if len(x) > 2:
        a, b = x[:-1], x[1:]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
            out["lag1"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return out


def lag1_bits(bits: np.ndarray) -> float | None:
    b = np.asarray(bits, float)
    if len(b) < 3 or np.std(b[:-1]) == 0 or np.std(b[1:]) == 0:
        return None
    return float(np.corrcoef(b[:-1], b[1:])[0, 1])
