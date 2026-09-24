"""Small netlist builder (pure Python / numpy; no numba import) and its compilation to the
integer/float arrays consumed by the MNA kernel (``mna.py``).

Element kinds: R (ohm), C (F), V (piecewise-linear source, V), I (piecewise-linear source, A),
STL (terminals d, g, s; device parameter vector; optional light waveform I_PH(t) in A) and CMP
(comparator: schematic/post-processing only, no electrical stamp).
Node "0" is ground.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _fmt(value: float, unit: str) -> str:
    v = float(value)
    if v == 0:
        return f"0 {unit}"
    prefixes = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"),
                (1e-9, "n"), (1e-12, "p"), (1e-15, "f"), (1e-18, "a")]
    for scale, pre in prefixes:
        if abs(v) >= scale * 0.999:
            return f"{v / scale:.4g} {pre}{unit}"
    return f"{v:.3g} {unit}"


@dataclass
class Netlist:
    nodes: list[str] = field(default_factory=lambda: ["0"])
    R: list[tuple[str, int, int, float]] = field(default_factory=list)
    C: list[tuple[str, int, int, float]] = field(default_factory=list)
    V: list[tuple[str, int, int, int, str]] = field(default_factory=list)      # name, a, b, wave, label
    I: list[tuple[str, int, int, int, str]] = field(default_factory=list)
    STL: list[dict[str, Any]] = field(default_factory=list)                   # name, d, g, s, p, light_wave, label
    CMP: list[dict[str, Any]] = field(default_factory=list)
    waves: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    samples: list[float] = field(default_factory=list)
    extra_breakpoints: list[float] = field(default_factory=list)
    t_end: float = 0.0
    main_wave: int = -1

    # ---- construction -----------------------------------------------------------------
    def node(self, name: str) -> int:
        if name in ("0", "gnd", "GND"):
            return 0
        if name not in self.nodes:
            self.nodes.append(name)
        return self.nodes.index(name)

    def wave(self, t, v) -> int:
        t = np.asarray(t, float)
        v = np.asarray(v, float)
        if t.ndim != 1 or t.shape != v.shape or len(t) < 1:
            raise ValueError("waveform needs matching 1-D time/value arrays")
        if np.any(np.diff(t) < 0):
            raise ValueError("waveform times must be non-decreasing")
        self.waves.append((t, v))
        return len(self.waves) - 1

    def add_R(self, name, a, b, ohm):
        if not (float(ohm) > 0):
            raise ValueError(f"resistor {name} must be > 0 ohm")
        self.R.append((name, self.node(a), self.node(b), float(ohm)))

    def add_C(self, name, a, b, farad):
        if float(farad) < 0:
            raise ValueError(f"capacitor {name} must be >= 0 F")
        if float(farad) > 0:
            self.C.append((name, self.node(a), self.node(b), float(farad)))

    def add_V(self, name, a, b, t, v, label="") -> int:
        w = self.wave(t, v)
        self.V.append((name, self.node(a), self.node(b), w, label))
        return w

    def add_I(self, name, a, b, t, v, label="") -> int:
        w = self.wave(t, v)
        self.I.append((name, self.node(a), self.node(b), w, label))
        return w

    def add_STL(self, name, d, g, s, p, light=None, label=""):
        lw = -1
        if light is not None:
            lw = self.wave(*light)
        self.STL.append(dict(name=name, d=self.node(d), g=self.node(g), s=self.node(s),
                             p=np.asarray(p, float).copy(), light_wave=lw, label=label))

    def add_CMP(self, name, node, threshold, label=""):
        self.CMP.append(dict(name=name, node=self.node(node), threshold=float(threshold), label=label))

    # ---- views ------------------------------------------------------------------------
    def schematic(self) -> dict:
        els = []
        for name, a, b, w, label in self.V:
            els.append(dict(kind="V", name=name, nodes=[self.nodes[a], self.nodes[b]], value=label))
        for name, a, b, w, label in self.I:
            els.append(dict(kind="I", name=name, nodes=[self.nodes[a], self.nodes[b]], value=label))
        for name, a, b, val in self.R:
            els.append(dict(kind="R", name=name, nodes=[self.nodes[a], self.nodes[b]], value=_fmt(val, "Ω")))
        for name, a, b, val in self.C:
            els.append(dict(kind="C", name=name, nodes=[self.nodes[a], self.nodes[b]], value=_fmt(val, "F")))
        for s in self.STL:
            els.append(dict(kind="STL", name=s["name"], nodes=[self.nodes[s["d"]], self.nodes[s["g"]], self.nodes[s["s"]]],
                            value=s["label"]))
        for c in self.CMP:
            els.append(dict(kind="CMP", name=c["name"], nodes=[self.nodes[c["node"]]],
                            value=c["label"] or f"v_th = {c['threshold']:.4g} V"))
        return dict(nodes=list(self.nodes), elements=els)

    def breakpoints(self) -> np.ndarray:
        pts = [np.asarray(t) for t, _ in self.waves] + [np.asarray(self.samples, float),
                                                         np.asarray(self.extra_breakpoints, float),
                                                         np.array([self.t_end])]
        b = np.unique(np.concatenate(pts)) if pts else np.array([self.t_end])
        b = b[(b > 0) & (b <= self.t_end)]
        # merge breakpoints closer than 1e-12 relative
        if len(b) > 1:
            keep = np.r_[True, np.diff(b) > 1e-12 * np.maximum(b[1:], 1e-9)]
            b = b[keep]
        return b

    def compile(self) -> dict:
        n_nodes = len(self.nodes)
        ii = lambda xs, j: np.array([x[j] for x in xs], dtype=np.int64)
        ff = lambda xs, j: np.array([x[j] for x in xs], dtype=np.float64)
        wt = np.concatenate([w[0] for w in self.waves]) if self.waves else np.zeros(1)
        wv = np.concatenate([w[1] for w in self.waves]) if self.waves else np.zeros(1)
        woff = np.r_[0, np.cumsum([len(w[0]) for w in self.waves])].astype(np.int64)
        P = np.array([s["p"] for s in self.STL], dtype=np.float64).reshape(len(self.STL), 26)
        samples = np.sort(np.asarray(self.samples, float))
        return dict(
            n_nodes=n_nodes, nV=len(self.V), nR=len(self.R), nC=len(self.C), nI=len(self.I), nS=len(self.STL),
            rA=ii(self.R, 1), rB=ii(self.R, 2), rG=1.0 / ff(self.R, 3) if self.R else np.zeros(0),
            cA=ii(self.C, 1), cB=ii(self.C, 2), cC=ff(self.C, 3),
            vA=ii(self.V, 1), vB=ii(self.V, 2), vW=ii(self.V, 3),
            iA=ii(self.I, 1), iB=ii(self.I, 2), iW=ii(self.I, 3),
            sD=np.array([s["d"] for s in self.STL], np.int64), sG=np.array([s["g"] for s in self.STL], np.int64),
            sS=np.array([s["s"] for s in self.STL], np.int64),
            sW=np.array([s["light_wave"] for s in self.STL], np.int64),
            wt=wt.astype(np.float64), wv=wv.astype(np.float64), woff=woff,
            bp=self.breakpoints().astype(np.float64), samp=samples.astype(np.float64), P=P,
        )

    def wave_eval(self, w: int, t) -> np.ndarray:
        tw, vw = self.waves[w]
        return np.interp(t, tw, vw)


# ---- waveform helpers ---------------------------------------------------------------------
def triangle(v0: float, v1: float, rate: float, n_cycles: int):
    """0 -> v1 -> v0 triangular cycles at |dv/dt| = rate (V/s)."""
    if rate <= 0:
        raise ValueError("ramp rate must be > 0")
    half = abs(v1 - v0) / rate
    if half <= 0:
        raise ValueError("ramp amplitude must be non-zero")
    t = np.arange(2 * n_cycles + 1) * half
    v = np.where(np.arange(2 * n_cycles + 1) % 2 == 0, v0, v1).astype(float)
    return t, v, 2 * half


def pulse_train(v_base, v_amp, width, period, rise, fall, n, delay=0.0, amps=None):
    """Trapezoidal pulses; width = flat top (between the end of the rise and the start of the fall).
    Returns (t, v, t_top_end list, t_period_end list)."""
    if rise <= 0 or fall <= 0 or width <= 0 or period <= 0:
        raise ValueError("pulse rise, fall, width and period must be > 0")
    if rise + width + fall > period * (1 + 1e-12):
        raise ValueError("pulse rise + width + fall must not exceed the period")
    ts = [0.0]
    vs = [v_base]
    top_end = []
    per_end = []
    for i in range(int(n)):
        a = v_amp if amps is None else amps[i]
        t0 = delay + i * period
        pts = [(t0, v_base), (t0 + rise, a), (t0 + rise + width, a), (t0 + rise + width + fall, v_base)]
        for tt, vv in pts:
            if tt <= ts[-1] + 1e-18:
                if tt < ts[-1] - 1e-18:
                    raise ValueError("overlapping pulses")
                vs[-1] = vv if tt > 0 else vs[-1]
                continue
            ts.append(tt)
            vs.append(vv)
        top_end.append(t0 + rise + width)
        per_end.append(t0 + period)
    t_end = delay + int(n) * period
    if t_end > ts[-1]:
        ts.append(t_end)
        vs.append(v_base)
    return np.array(ts), np.array(vs), top_end, per_end
