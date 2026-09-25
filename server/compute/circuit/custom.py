"""User-drawn circuits: ``kind: "circuit"``, ``bench: "custom"`` (docs/WEB_CONTRACT.md §6).

Pipeline of ``run_custom(payload, progress)``:

1. parse and validate the netlist (R, C, V, I, STL; ground aliases 0/gnd/GND; waves dc / pulse / pwl /
   sine -> piecewise-linear with SPICE PULSE semantics, bounded point counts, every corner a breakpoint),
   the ``tran`` block, ``stochastic``, ``detect`` and ``probes`` (limits: 40 elements, 8 STL, 30 nodes,
   2000 points per wave);
2. electrical rule check (no ground, nodes without a DC path to ground, voltage-source loops, current
   sources into open nodes, shorted terminals) with messages naming the element / node;
3. per STL: device block (``payloads.normalize_device`` + ``params.build_p``), the gate-source voltage and
   light range the cell will see (linear DC estimate of the network with the STLs open), the quasi-static
   branch profiles at those extremes -> latch-state fold values u_i/u_j, noise bands, latch window;
4. feasibility estimate (the bench estimator walked along each cell's open-circuit drive + the Δv / dt_max
   limits of every node) -> refuse or warn;
5. transient runs with the shared MNA kernel (``sim.simulate``), every accepted step recorded and reduced on
   the fly (``_Sink``), signals / events / summary / statistics assembled in the §4 + §6.2 result shape.

Sign conventions (§6.1): ``V(n)`` node voltage to ground; ``I(R1)``, ``I(C1)`` current through the element from
its first to its second node; ``I(V1)``, ``I(I1)`` current through the source from its + (first) to its -
(second) node (a source delivering power has I(V1) < 0; for I sources it equals the wave value);
``I(X1.d)``, ``I(X1.s)``, ``I(X1.g)`` currents INTO the STL terminals (I(X1.s) = -I(X1.d), I(X1.g) = 0: ideal
gate, the body-gate displacement current is not stamped).
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from server import params as PR

from .netlist import Netlist, _fmt
from .basic import PINS as BASIC_PINS, pack as pack_basic, current_series
from .oscillator import R_HIGH, qs_drive, quasi_static_period

LIMITS = dict(elements=40, stl=8, cmp=8, nodes=30, wave_points=2000, generated_points=20000)
# wave_points: user PWL lists; generated_points: PULSE corners / SINE samples generated server-side
GROUND_ALIASES = ("0", "gnd", "GND", "Gnd")
ELEMENT_TYPES = ("R", "C", "V", "I", "STL", "CMP", "MOS", "D", "BJT")
_NAME_RE = re.compile(r"^[^\s().,;\"'\[\]{}]{1,32}$")
_NODE_RE = re.compile(r"^[^\s()\"'\[\]{},;]{1,32}$")
V_ABS_MAX = 1000.0             # V, source values
I_ABS_MAX = 1.0                # A
LIGHT_MAX_PA = 1e7             # pA
T_STOP_MAX = 1e3               # s
TRTOL = 7.0                    # SPICE: allowed node-voltage LTE = TRTOL (reltol |v| + VNTOL)
VNTOL = 1e-6                   # V
G_OFF = 1e-12                  # S, STL drain-source in the linear DC estimate (open cell)
MAX_TOTAL_STEPS = 4e7
OSC_TRANSITION_STEPS = 400.0   # steps per latch transition of a quasi-static oscillator walk (calibrated, §13)
MAX_STORED_RUNS = 8
MAX_EVENTS_OUT = 20000
BUDGET_RUN0 = 400_000          # output values (points x signals) of run 0
BUDGET_OTHER = 100_000         # ... of each stored run 1..7
BUDGET_ENV = 300_000           # ... of all envelopes (5 arrays per signal)
ENV_MEMORY = 5_000_000         # float32 values kept for the envelopes (runs x signals x grid)
DIGITS_T = 10                  # significant digits of the plotted time axes
DIGITS_SIG = 7                 # ... of the plotted signal values (events / summary / op keep full precision)
DIGITS_ENV = 6                 # ... of the envelopes


def _L(ko: str, en: str) -> dict:
    return {"ko": ko, "en": en}


# ================================================================== waves
@dataclass
class Wave:
    t: np.ndarray                      # PWL times (strictly increasing, t[0] >= 0)
    v: np.ndarray                      # values (V, A, or A for light)
    resolved: dict                     # echo (parameters after defaults / replacements)
    varying: bool                      # not constant over [0, t_stop]


def _f(spec: dict, key: str, where: str, default=None, lo=None, hi=None, positive=False) -> float:
    v = spec.get(key, default)
    if v is None:
        if default is None:
            raise ValueError(f"{where}: missing '{key}'")
        v = default
    if isinstance(v, bool):
        raise ValueError(f"{where}: '{key}' must be a number")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: '{key}' must be a number (got {str(v)[:30]!r})") from None
    if not math.isfinite(x):
        raise ValueError(f"{where}: '{key}' must be finite")
    if positive and x <= 0:
        raise ValueError(f"{where}: '{key}' must be > 0")
    if lo is not None and x < lo:
        raise ValueError(f"{where}: '{key}' = {x:g} must be >= {lo:g}")
    if hi is not None and x > hi:
        raise ValueError(f"{where}: '{key}' = {x:g} must be <= {hi:g}")
    return x


def parse_wave(spec: Any, where: str, t_stop: float, edge0: float, vmax: float, unit: str,
               warnings: list[str], nonneg: bool = False) -> Wave:
    """Wave block -> PWL.  ``edge0``: finite edge used for zero rise/fall times and vertical PWL steps."""
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: wave must be an object {{kind: dc|pulse|pwl|sine, ...}}")
    kind = spec.get("kind")
    npmax = LIMITS["generated_points"]
    lo = 0.0 if nonneg else -vmax
    if kind == "dc":
        val = _f(spec, "value", where, lo=lo, hi=vmax)
        return Wave(np.array([0.0]), np.array([val]), dict(kind="dc", value=val), False)
    if kind == "pulse":
        v1 = _f(spec, "v1", where, 0.0, lo, vmax)
        v2 = _f(spec, "v2", where, lo=lo, hi=vmax)
        td = _f(spec, "td", where, 0.0, 0.0, T_STOP_MAX)
        tr = _f(spec, "tr", where, 0.0, 0.0, T_STOP_MAX)
        tf = _f(spec, "tf", where, 0.0, 0.0, T_STOP_MAX)
        pw = _f(spec, "pw", where, t_stop, 0.0, T_STOP_MAX)
        per = _f(spec, "per", where, 0.0, 0.0, T_STOP_MAX)
        ncyc = spec.get("ncycles", 0)
        ncyc = int(round(_f({"n": 0 if ncyc is None else ncyc}, "n", where + " ncycles", 0.0, 0.0, 1e9)))
        e = edge0
        if pw > 0:
            e = min(e, 0.1 * pw)
        if per > 0:
            e = min(e, 0.1 * per)
        e = max(e, 1e-18)
        repl = []
        if tr == 0:
            tr = e
            repl.append("tr")
        if tf == 0:
            tf = e
            repl.append("tf")
        if repl:
            warnings.append(f"{where}: pulse {' and '.join(repl)} = 0 replaced by a finite edge of {e:.3g} s")
        single = per <= 0
        if not single and tr + pw + tf > per * (1 + 1e-12):
            raise ValueError(f"{where}: pulse tr + pw + tf = {tr + pw + tf:.4g} s exceeds the period per = {per:.4g} s")
        ts, vs = [0.0], [v1]
        k = 0
        while True:
            if ncyc > 0 and k >= ncyc:
                break
            if single and k >= 1:
                break
            t0 = td + (0.0 if single else k * per)
            if t0 >= t_stop and k > 0:
                break
            for tt, vv in ((t0, v1), (t0 + tr, v2), (t0 + tr + pw, v2), (t0 + tr + pw + tf, v1)):
                if tt <= ts[-1] * (1 + 1e-15) + 1e-300:
                    if abs(tt - ts[-1]) <= 1e-15 * max(tt, 1e-300) and vv != vs[-1]:
                        vs[-1] = vv if tt > 0 else vs[-1]
                    continue
                ts.append(tt)
                vs.append(vv)
            if len(ts) > npmax:
                n_per = (t_stop - td) / per if per > 0 else 1
                raise ValueError(f"{where}: the pulse needs more than {npmax} corner points within t_stop "
                                 f"(~{n_per:.0f} periods, 4 corners each; at most {npmax // 4} periods): shorten t_stop, "
                                 "lengthen per or set ncycles")
            if t0 >= t_stop:
                break
            k += 1
        res = dict(kind="pulse", v1=v1, v2=v2, td=td, tr=tr, tf=tf, pw=pw, per=per, ncycles=ncyc, n_points=len(ts))
        return Wave(np.array(ts), np.array(vs), res, v1 != v2)
    if kind == "pwl":
        tl, vl = spec.get("t"), spec.get("v")
        if not isinstance(tl, (list, tuple)) or not isinstance(vl, (list, tuple)):
            raise ValueError(f"{where}: pwl needs lists 't' and 'v'")
        if len(tl) != len(vl) or len(tl) < 1:
            raise ValueError(f"{where}: pwl 't' and 'v' must have the same non-zero length")
        if len(tl) > LIMITS["wave_points"]:
            raise ValueError(f"{where}: pwl has {len(tl)} points (limit {LIMITS['wave_points']})")
        t = np.array([_f({"t": x}, "t", where + " pwl t") for x in tl])
        v = np.array([_f({"v": x}, "v", where + " pwl v", lo=lo, hi=vmax) for x in vl])
        if t[0] < 0:
            raise ValueError(f"{where}: pwl times must be >= 0")
        if np.any(np.diff(t) < 0):
            raise ValueError(f"{where}: pwl times must be non-decreasing")
        steps = 0
        t = t.copy()
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                nxt = t[i + 1:][t[i + 1:] > t[i - 1]]
                eps = edge0 if len(nxt) == 0 else min(edge0, 0.5 * (nxt[0] - t[i - 1]))
                t[i] = t[i - 1] + max(eps, 1e-18)
                steps += 1
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"{where}: pwl has too many coincident time points")
        if steps:
            warnings.append(f"{where}: {steps} vertical pwl step(s) given a finite edge of <= {edge0:.3g} s")
        res = dict(kind="pwl", t=t.tolist(), v=v.tolist(), n_points=len(t))
        vr = v[(t <= t_stop)] if np.any(t <= t_stop) else v[:1]
        vend = np.interp(t_stop, t, v)
        return Wave(t, v, res, bool(np.ptp(np.r_[vr, vend]) > 0))
    if kind == "sine":
        vo = _f(spec, "vo", where, 0.0, lo, vmax)
        va = _f(spec, "va", where, lo=-vmax, hi=vmax)
        freq = _f(spec, "freq", where, lo=0.0, hi=1e12)
        td = _f(spec, "td", where, 0.0, 0.0, T_STOP_MAX)
        theta = _f(spec, "theta", where, 0.0, -1e12, 1e12)
        phase = _f(spec, "phase", where, 0.0, -1e4, 1e4)        # degrees (extension, SPICE PHASE)
        ph = math.radians(phase)
        if freq == 0 or va == 0 or td >= t_stop:
            val = vo + va * math.sin(ph) if td < t_stop else vo
            res = dict(kind="sine", vo=vo, va=va, freq=freq, td=td, theta=theta, phase=phase, n_points=1)
            return Wave(np.array([0.0, max(td, 0.0), t_stop]) if td > 0 else np.array([0.0]),
                        np.array([vo, vo, val]) if td > 0 else np.array([val]), res, td > 0 and val != vo)
        n_per = (t_stop - td) * freq
        per_pts = int(min(128, math.floor((npmax - 3) / max(n_per, 1e-12))))
        if per_pts < 16:
            raise ValueError(f"{where}: {n_per:.0f} sine periods within t_stop need more than {npmax} samples "
                             f"(>= 16 per period, at most {npmax // 16} periods): lower freq or shorten t_stop")
        if per_pts < 48:
            err = 100 * (1 - math.cos(math.pi / per_pts))
            warnings.append(f"{where}: sine sampled with {per_pts} points per period (PWL error <= {err:.2g} % of va)")
        dt = 1.0 / (freq * per_pts)
        kk = np.arange(int(math.ceil(n_per * per_pts)) + 1)
        ts = td + kk * dt
        s = ts - td
        vs = vo + va * np.exp(-theta * s) * np.sin(2 * np.pi * freq * s + ph)
        if not np.all(np.isfinite(vs)) or np.max(np.abs(vs)) > vmax * (1 + 1e-9):
            raise ValueError(f"{where}: sine values exceed the allowed range (theta too negative?)")
        if td > 0:
            ts = np.r_[0.0, ts]
            vs = np.r_[vo, vs] if phase == 0 else np.r_[vo + va * math.sin(ph), vs]
        res = dict(kind="sine", vo=vo, va=va, freq=freq, td=td, theta=theta, phase=phase, n_points=len(ts),
                   points_per_period=per_pts)
        return Wave(ts, vs, res, True)
    raise ValueError(f"{where}: unknown wave kind {kind!r} (dc | pulse | pwl | sine)")


def _wave_text(w: dict, unit: str) -> str:
    k = w["kind"]
    if k == "dc":
        return f"DC {_fmt(w['value'], unit)}"
    if k == "pulse":
        return (f"PULSE({w['v1']:g} {w['v2']:g} {w['td']:g} {w['tr']:.3g} {w['tf']:.3g} {w['pw']:.4g} {w['per']:.4g}"
                f"{' %d' % w['ncycles'] if w['ncycles'] else ''})")
    if k == "pwl":
        return f"PWL({w['n_points']} points)"
    return f"SIN({w['vo']:g} {w['va']:g} {w['freq']:.4g} {w['td']:g} {w['theta']:g})"


# ================================================================== elements
@dataclass
class El:
    type: str
    name: str
    nodes: list[str]                       # canonical node names (ground = "0"); STL: [d, g, s]
    value: float | None = None
    wave: Wave | None = None
    # STL
    device: dict | None = None
    light: Wave | None = None
    ls_block: dict | None = None
    p: np.ndarray | None = None
    cmp: dict | None = None                # CMP: v_ref, v_high, v_low, hysteresis, width (nodes = [in, inm, out])
    model: dict | None = None              # educational semiconductor model
    idx: int = -1                          # index within its kind in the compiled netlist (-1: not stamped)
    wave_idx: int = -1


def _canon_node(x: Any, where: str) -> str:
    if isinstance(x, bool) or not isinstance(x, (str, int)):
        raise ValueError(f"{where}: node names must be strings (got {str(x)[:30]!r})")
    s = str(x).strip()
    if s in GROUND_ALIASES or s.lower() == "gnd":
        return "0"
    if not _NODE_RE.match(s):
        raise ValueError(f"{where}: invalid node name {s[:40]!r} (1-32 characters, no spaces or parentheses)")
    return s


def parse_elements(netlist: Any, t_stop: float, edge0: float, warnings: list[str]) -> list[El]:
    if not isinstance(netlist, dict):
        raise ValueError("netlist must be an object {elements: [...]}")
    raw = netlist.get("elements")
    if not isinstance(raw, list) or not raw:
        raise ValueError("netlist.elements must be a non-empty list")
    if len(raw) > LIMITS["elements"]:
        raise ValueError(f"the circuit has {len(raw)} elements (limit {LIMITS['elements']})")
    els: list[El] = []
    seen: dict[str, str] = {}
    for i, e in enumerate(raw):
        if not isinstance(e, dict):
            raise ValueError(f"netlist.elements[{i}] must be an object")
        typ = e.get("type")
        if typ not in ELEMENT_TYPES:
            raise ValueError(f"netlist.elements[{i}]: unknown element type {typ!r} ({' | '.join(ELEMENT_TYPES)})")
        name = e.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name.strip()):
            raise ValueError(f"netlist.elements[{i}] ({typ}): invalid name {str(name)[:40]!r} "
                             "(1-32 characters; no spaces, dots, commas or brackets)")
        name = name.strip()
        low = name.lower()
        if low in seen:
            raise ValueError(f"duplicate element name {name!r} (also used as {seen[low]!r}); names are case-insensitive")
        seen[low] = name
        where = f"{name}"
        nd = e.get("nodes")
        if typ in BASIC_PINS:
            pins = BASIC_PINS[typ]
            if isinstance(nd, (list, tuple)) and len(nd) == len(pins):
                nd = dict(zip(pins, nd))
            if not isinstance(nd, dict) or any(nd.get(k) in (None, "") for k in pins):
                raise ValueError(f"{where}: {typ} needs connected terminals {', '.join(pins)}")
            nodes = [_canon_node(nd[k], f"{where}.{k}") for k in pins]
        elif typ == "STL":
            if isinstance(nd, (list, tuple)) and len(nd) == 3:
                nd = dict(zip(("d", "g", "s"), nd))
            if not isinstance(nd, dict):
                raise ValueError(f"{where}: STL nodes must be an object {{d, g, s}}")
            missing = [k for k in ("d", "g", "s") if nd.get(k) in (None, "")]
            if missing:
                raise ValueError(f"{where}: STL terminal(s) {', '.join(missing)} not connected (nodes needs d, g and s)")
            nodes = [_canon_node(nd[k], f"{where}.{k}") for k in ("d", "g", "s")]
        elif typ == "CMP":
            if isinstance(nd, (list, tuple)) and len(nd) in (2, 3):
                nd = dict(zip(("in", "out"), nd)) if len(nd) == 2 else dict(zip(("in", "inm", "out"), nd))
            if not isinstance(nd, dict):
                raise ValueError(f"{where}: comparator nodes must be an object {{in, out}} (optionally inm)")
            missing = [k for k in ("in", "out") if nd.get(k) in (None, "")]
            if missing:
                raise ValueError(f"{where}: comparator terminal(s) {', '.join(missing)} not connected (nodes needs in and out)")
            nodes = [_canon_node(nd["in"], f"{where}.in"), _canon_node(nd.get("inm") or "0", f"{where}.inm"),
                     _canon_node(nd["out"], f"{where}.out")]
            if nodes[2] == "0":
                raise ValueError(f"{where}: the comparator output cannot be ground (it is a voltage source to ground)")
        else:
            if not isinstance(nd, (list, tuple)) or len(nd) != 2:
                raise ValueError(f"{where}: {typ} needs exactly two nodes [n1, n2]")
            nodes = [_canon_node(x, where) for x in nd]
        el = El(type=typ, name=name, nodes=nodes)
        if typ in BASIC_PINS:
            m = e.get("model", {})
            if m is None:
                m = {}
            if not isinstance(m, dict):
                raise ValueError(f"{where}: model must be an object")
            if typ == "MOS":
                polarity = m.get("polarity", "nmos")
                if polarity not in ("nmos", "pmos"):
                    raise ValueError(f"{where}: MOS polarity must be nmos or pmos")
                fields = {"L_um": (1.0, 0.001, 10000.0), "W_um": (10.0, 0.001, 100000.0),
                          "Vth_V": (0.5, -100.0, 100.0), "SS_mV_dec": (80.0, 10.0, 1000.0),
                          "k_uA_V2": (100.0, 0.001, 1e6), "lambda_per_V": (0.02, 0.0, 10.0)}
                el.model = {key: _f(m, key, where + " model", *limits) for key, limits in fields.items()}
                el.model["Vth_V"] = abs(el.model["Vth_V"])
                el.model["polarity"] = polarity
            elif typ == "D":
                el.model = dict(Is_A=_f(m, "Is_A", where, 1e-14, 1e-30, 1.0),
                                n=_f(m, "n", where, 1.0, 0.1, 10.0))
            else:
                polarity = m.get("polarity", "npn")
                if polarity not in ("npn", "pnp"):
                    raise ValueError(f"{where}: BJT polarity must be npn or pnp")
                el.model = dict(polarity=polarity, Is_A=_f(m, "Is_A", where, 1e-15, 1e-30, 1.0),
                                beta_F=_f(m, "beta_F", where, 100.0, 0.01, 1e6),
                                beta_R=_f(m, "beta_R", where, 1.0, 0.01, 1e6))
        elif typ == "R":
            el.value = _f(e, "value", where + " (ohm)", lo=1e-3, hi=1e15)
        elif typ == "C":
            el.value = _f(e, "value", where + " (F)", lo=0.0, hi=1.0)
        elif typ == "CMP":
            vref = _f(e, "v_ref", where + " v_ref (V)", lo=-V_ABS_MAX, hi=V_ABS_MAX)
            vhi = _f(e, "v_high", where + " v_high (V)", 1.0, -V_ABS_MAX, V_ABS_MAX)
            vlo = _f(e, "v_low", where + " v_low (V)", 0.0, -V_ABS_MAX, V_ABS_MAX)
            hyst = _f(e, "hysteresis", where + " hysteresis (V)", 0.0, 0.0, 10.0)
            wid = _f(e, "width", where + " width (V)", 1e-3, 1e-6, 0.1)
            if vhi == vlo:
                raise ValueError(f"{where}: v_high and v_low must differ")
            el.cmp = dict(v_ref=vref, v_high=vhi, v_low=vlo, hysteresis=hyst, width=wid)
        elif typ in ("V", "I"):
            w = e.get("wave")
            if w is None and "value" in e:
                w = {"kind": "dc", "value": e["value"]}
            if w is None:
                raise ValueError(f"{where}: missing 'wave'")
            vmax = V_ABS_MAX if typ == "V" else I_ABS_MAX
            el.wave = parse_wave(w, where, t_stop, edge0, vmax, "V" if typ == "V" else "A", warnings)
        else:
            dev = e.get("device")
            if dev is not None and not isinstance(dev, dict):
                raise ValueError(f"{where}: device must be an object (device block)")
            el.device = dev or {}
            lw = e.get("light_pA")
            if lw is not None:
                wv = parse_wave(lw, where + " light_pA", t_stop, edge0, LIGHT_MAX_PA, "pA", warnings, nonneg=True)
                el.light = Wave(wv.t, wv.v * 1e-12, wv.resolved, wv.varying)      # pA -> A for the kernel
            ls = e.get("local_state")
            if ls is not None and not isinstance(ls, dict):
                raise ValueError(f"{where}: local_state must be an object")
            el.ls_block = ls
        els.append(el)
    if sum(1 for e in els if e.type == "STL") > LIMITS["stl"]:
        raise ValueError(f"the circuit has {sum(1 for e in els if e.type == 'STL')} STL cells (limit {LIMITS['stl']})")
    if sum(1 for e in els if e.type == "CMP") > LIMITS["cmp"]:
        raise ValueError(f"the circuit has {sum(1 for e in els if e.type == 'CMP')} comparators (limit {LIMITS['cmp']})")
    return els


# ================================================================== ERC
class _DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def _q(n: str) -> str:
    return "0 (ground)" if n == "0" else repr(n)


def erc(els: list[El], warnings: list[str]) -> list[str]:
    """Electrical rule check.  Returns the node list (ground first); raises ValueError on errors."""
    nodes = ["0"]
    for e in els:
        for n in e.nodes:
            if n not in nodes:
                nodes.append(n)
    if len(nodes) - 1 > LIMITS["nodes"]:
        raise ValueError(f"the circuit has {len(nodes) - 1} nodes besides ground (limit {LIMITS['nodes']})")
    conns: dict[str, list[str]] = {n: [] for n in nodes}
    for e in els:
        labels = BASIC_PINS[e.type] if e.type in BASIC_PINS else ("d", "g", "s") if e.type == "STL" else ("in", "inm", "out") if e.type == "CMP" else \
            ("+", "-") if e.type in ("V", "I") else ("1", "2")
        for lab, n in zip(labels, e.nodes):
            if e.type == "CMP" and lab == "inm" and n == "0":
                continue                          # single-ended comparator: inm is ground implicitly
            conns[n].append(f"{e.name}.{lab}" if e.type in ("STL", "CMP") else e.name)
    if not conns["0"]:
        raise ValueError("no ground reference: no element is connected to node 0 (gnd). Connect the circuit to "
                         "ground (node '0', 'gnd' or 'GND')")
    # voltage-source loops (incl. parallel sources and sources shorted by a wire)
    dsu = _DSU(nodes)
    vadj: dict[str, list[tuple[str, str]]] = {n: [] for n in nodes}
    for e in els:
        if e.type not in ("V", "CMP"):
            continue
        # a comparator output is an ideal voltage source from 'out' to ground
        a, b = (e.nodes[2], "0") if e.type == "CMP" else e.nodes
        if a == b:
            raise ValueError(f"voltage source {e.name} is short-circuited: both terminals on node {_q(a)}")
        if not dsu.union(a, b):
            # path a -> b through the voltage sources added so far
            prev: dict[str, tuple[str, str] | None] = {a: None}
            queue = [a]
            while queue:
                n = queue.pop(0)
                if n == b:
                    break
                for m, nm in vadj[n]:
                    if m not in prev:
                        prev[m] = (n, nm)
                        queue.append(m)
            loop, n = [], b
            while prev.get(n):
                n, nm = prev[n]
                loop.append(nm)
            names = ", ".join(sorted(set(loop))) or "?"
            kind = "in parallel with" if len(loop) == 1 else "in a loop with"
            what = f"the output of comparator {e.name}" if e.type == "CMP" else f"voltage source {e.name}"
            if e.type == "CMP" or any(x.type == "CMP" and x.name in loop for x in els):
                raise ValueError(f"{what} is {kind} {names} (between nodes {_q(a)} and {_q(b)}): a comparator output "
                                 "is an ideal voltage source and must not be driven by another source — connect it "
                                 "through a resistor or use another node")
            raise ValueError(f"{what} is {kind} {names} (between nodes {_q(a)} and {_q(b)}): "
                             "a loop of ideal voltage sources has no unique solution — add a series resistor or "
                             "remove one source")
        vadj[a].append((b, e.name))
        vadj[b].append((a, e.name))
    # DC paths to ground: R, V, diodes, the STL/MOSFET drain-source path, BJT junctions and comparator outputs
    # conduct; C, I sources, STL/MOSFET gates and comparator inputs do not
    dc = _DSU(nodes)
    for e in els:
        if e.type in ("R", "V", "D"):
            dc.union(e.nodes[0], e.nodes[1])
        elif e.type in ("STL", "MOS"):
            dc.union(e.nodes[0], e.nodes[2])
        elif e.type == "BJT":
            dc.union(e.nodes[0], e.nodes[1])
            dc.union(e.nodes[1], e.nodes[2])
        elif e.type == "CMP":
            dc.union(e.nodes[2], "0")             # the output is a voltage source to ground; the inputs are ideal
    floating = [n for n in nodes[1:] if dc.find(n) != dc.find("0")]
    if floating:
        n = floating[0]
        others = [m for m in floating[1:6]]
        srcs = [e.name for e in els if e.type == "I" and n in e.nodes]
        caps = [e.name for e in els if e.type == "C" and n in e.nodes]
        gates = [f"{e.name}.g" for e in els if e.type in ("STL", "MOS") and e.nodes[1] == n] + \
            [f"{e.name}.{lab}" for e in els if e.type == "CMP" for lab, m in (("in", e.nodes[0]), ("inm", e.nodes[1])) if m == n]
        via = []
        if caps:
            via.append("capacitors " + ", ".join(caps))
        if srcs:
            via.append("current sources " + ", ".join(srcs))
        if gates:
            via.append("transistor gates / comparator inputs " + ", ".join(gates))
        extra = f" (also: {', '.join(repr(m) for m in others)}{' ...' if len(floating) > 6 else ''})" if others else ""
        if srcs and not caps and not gates and all(c in srcs for c in conns[n]):
            raise ValueError(f"current source {srcs[0]} drives node {n!r}, which has no other connection (open circuit): "
                             f"add a resistor or another DC path from {n!r} to ground{extra}")
        how = " and ".join(via) if via else "nothing that conducts DC"
        raise ValueError(f"node {n!r} has no DC path to ground: it is connected only through {how}. Every node needs "
                         "a DC path to ground (resistor, voltage source, diode, STL or MOSFET drain-source, BJT "
                         f"junction or comparator output){extra}")
    # warnings: single connections, shorted terminals
    for n in nodes[1:]:
        if len(conns[n]) == 1 and not conns[n][0].endswith(".out"):      # an unloaded comparator output is fine
            warnings.append(f"node {n!r} has only one connection ({conns[n][0]}): no current flows into it")
    for e in els:
        if e.type in ("R", "C", "I") and e.nodes[0] == e.nodes[1]:
            warnings.append(f"{e.name}: both terminals on node {_q(e.nodes[0])} (shorted, no effect)")
        if e.type == "STL":
            d, g, s = e.nodes
            if d == s:
                warnings.append(f"{e.name}: drain and source on the same node {_q(d)} (V_DS = 0: the cell cannot latch)")
            if d == g:
                warnings.append(f"{e.name}: drain and gate on the same node {_q(d)} (V_GS = V_DS)")
            if g == s:
                warnings.append(f"{e.name}: gate and source on the same node {_q(g)} (V_GS = 0 V: channel-on regime, "
                                "usually no latch window)")
        if e.type == "CMP":
            i_, m_, o_ = e.nodes
            if o_ in (i_, m_):
                warnings.append(f"{e.name}: the output drives its own input node {_q(o_)} (feedback through an ideal "
                                "comparator: the solution may not be unique)")
            if i_ == m_:
                warnings.append(f"{e.name}: both inputs on node {_q(i_)} (the output is constant)")
    return nodes


# ================================================================== linear DC estimate
def linear_dc(nodes: list[str], els: list[El]):
    """Node-voltage sensitivities of the linear DC network (C open, STL drain-source G_OFF, gate open) to
    every independent source (value 1) and to a unit current injected into each STL drain (out of its
    source).  Returns (S_src (N x n_src), src_list, S_stl (N x n_stl)) with rows = nodes[1:]."""
    N = len(nodes) - 1
    idx = {n: i - 1 for i, n in enumerate(nodes)}          # ground -> -1
    Vs = [e for e in els if e.type in ("V", "CMP")]        # comparator outputs: fixed node (value 0 here)
    srcs = [e for e in els if e.type in ("V", "I")]
    stls = [e for e in els if e.type == "STL"]
    M = N + len(Vs)
    G = np.zeros((M, M))

    def stamp(a, b, g):
        ia, ib = idx[a], idx[b]
        if ia >= 0:
            G[ia, ia] += g
        if ib >= 0:
            G[ib, ib] += g
        if ia >= 0 and ib >= 0:
            G[ia, ib] -= g
            G[ib, ia] -= g

    for i in range(N):
        G[i, i] += 1e-18
    for e in els:
        if e.type == "R":
            stamp(e.nodes[0], e.nodes[1], 1.0 / e.value)
        elif e.type in ("STL", "MOS"):
            stamp(e.nodes[0], e.nodes[2], G_OFF)
        elif e.type == "D":
            stamp(e.nodes[0], e.nodes[1], G_OFF)
        elif e.type == "BJT":
            stamp(e.nodes[0], e.nodes[1], G_OFF)
            stamp(e.nodes[1], e.nodes[2], G_OFF)
    for j, e in enumerate(Vs):
        ia, ib = (idx[e.nodes[2]], -1) if e.type == "CMP" else (idx[e.nodes[0]], idx[e.nodes[1]])
        r = N + j
        if ia >= 0:
            G[ia, r] += 1
            G[r, ia] += 1
        if ib >= 0:
            G[ib, r] -= 1
            G[r, ib] -= 1
    Bm = np.zeros((M, len(srcs) + len(stls)))
    vj = {e.name: j for j, e in enumerate(Vs)}
    for c, e in enumerate(srcs):
        if e.type == "V":
            Bm[N + vj[e.name], c] = 1.0
        else:                                   # current leaves node + into the source and enters node -
            ia, ib = idx[e.nodes[0]], idx[e.nodes[1]]
            if ia >= 0:
                Bm[ia, c] -= 1.0
            if ib >= 0:
                Bm[ib, c] += 1.0
    for k, e in enumerate(stls):
        c = len(srcs) + k
        ia, ib = idx[e.nodes[0]], idx[e.nodes[2]]
        if ia >= 0:
            Bm[ia, c] += 1.0
        if ib >= 0:
            Bm[ib, c] -= 1.0
    try:
        X = np.linalg.solve(G, Bm)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(G, Bm, rcond=None)[0]
    X = X[:N]
    return X[:, :len(srcs)], srcs, X[:, len(srcs):]


GRID_MAX = 60000               # points of the union time grid used by the estimates


def _union_grid(waves: list[Wave], t_stop: float) -> np.ndarray:
    pts = [np.array([0.0, t_stop])] + [w.t[(w.t >= 0) & (w.t <= t_stop)] for w in waves]
    g = np.unique(np.concatenate(pts))
    if len(g) > 1:
        g = g[np.r_[True, np.diff(g) > 1e-12 * np.maximum(g[1:], 1e-12)]]
    if len(g) > GRID_MAX:                 # many long generated waves: subsample (the TV bound below covers aliasing)
        g = g[np.unique(np.round(np.linspace(0, len(g) - 1, GRID_MAX)).astype(np.int64))]
    return g


def _total_variation(w: Wave, t_stop: float) -> float:
    t = np.r_[w.t[w.t < t_stop], t_stop]
    return float(np.sum(np.abs(np.diff(np.interp(t, w.t, w.v))))) if len(t) > 1 else 0.0


def _corner_weights(tg: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Sharpness of every interior grid point of the PWL signals Y (rows): |Δslope| / (|s-| + |s+|),
    max over the rows (1 = a pulse corner, ~0 = a smoothly sampled waveform)."""
    if len(tg) < 3 or Y.size == 0:
        return np.zeros(max(len(tg) - 2, 0))
    Y = np.atleast_2d(Y)
    dt = np.diff(tg)
    s = np.diff(Y, axis=1) / dt
    rng = np.ptp(Y, axis=1)[:, None] / max(tg[-1] - tg[0], 1e-300)
    eps = 1e-3 * rng + 1e-300
    w = np.abs(s[:, 1:] - s[:, :-1]) / (np.abs(s[:, 1:]) + np.abs(s[:, :-1]) + eps)
    w = np.where(rng > 0, w, 0.0)
    return np.clip(w.max(axis=0), 0.0, 1.0)


# ================================================================== signals
@dataclass
class SigSpec:
    key: str
    label: dict
    unit: str
    axis: str
    recipe: tuple
    state: bool = False                     # STL internal state (noisy in stochastic runs)


def _signal_specs(nodes: list[str], els: list[El], layout: dict, cell_ls: list, sigE: list) -> list[SigSpec]:
    nn, nv, ns = layout["nn"], layout["nv"], layout["ns"]
    base = nn + nv
    cbase = base + 7 * ns
    col = {n: i for i, n in enumerate(nodes)}             # node -> rec column (ground 0 -> constant 0)
    out: list[SigSpec] = []
    for n in nodes[1:]:
        out.append(SigSpec(f"V({n})", _L(f"노드 {n} 전압", f"Voltage at node {n}"), "V", "voltage", ("col", col[n])))
    k = 0
    for e in els:
        a, b = e.nodes[0], e.nodes[-1]
        if e.type == "R":
            out.append(SigSpec(f"I({e.name})", _L(f"{e.name} 전류 ({a} → {b})", f"Current through {e.name} ({a} → {b})"),
                               "A", "current", ("rcur", col[e.nodes[0]], col[e.nodes[1]], 1.0 / e.value)))
        elif e.type == "C":
            rc = ("col", cbase + e.idx) if e.idx >= 0 else ("zero",)
            out.append(SigSpec(f"I({e.name})", _L(f"{e.name} 전류 ({a} → {b})", f"Current through {e.name} ({a} → {b})"),
                               "A", "current", rc))
        elif e.type == "V":
            out.append(SigSpec(f"I({e.name})", _L(f"전압원 {e.name} 전류 (+ → −, 소스 내부)", f"Current through {e.name} (+ → −)"),
                               "A", "current", ("col", nn + e.idx)))
        elif e.type == "I":
            out.append(SigSpec(f"I({e.name})", _L(f"전류원 {e.name} 전류 (+ → −)", f"Current of {e.name} (+ → −)"),
                               "A", "current", ("wave", e.wave.t, e.wave.v)))
        elif e.type in BASIC_PINS:
            model_row = pack_basic(e.type, [col[n] for n in e.nodes], e.model)
            for terminal, pin in enumerate(BASIC_PINS[e.type]):
                key = f"I({e.name})" if e.type == "D" else f"I({e.name}.{pin})"
                if e.type == "D" and terminal > 0:
                    continue
                out.append(SigSpec(key, _L(f"{e.name}.{pin} 전류", f"{e.name}.{pin} current"),
                                   "A", "current", ("basic", np.array([col[n] for n in e.nodes]), model_row, terminal)))
        elif e.type == "CMP":
            out.append(SigSpec(f"I({e.name})", _L(f"비교기 {e.name} 출력 전류 (출력 → 접지, 소스 내부)",
                                                  f"Output current of comparator {e.name} (out → ground, inside the source)"),
                               "A", "current", ("col", nn + e.idx)))
            mid = 0.5 * (e.cmp["v_high"] + e.cmp["v_low"])
            out.append(SigSpec(f"{e.name}.bit", _L(f"비교기 {e.name} 디지털 출력 (1 = high)", f"Comparator {e.name} digital output (1 = high)"),
                               "1", "logic", ("bit", col[e.nodes[2]], mid, e.cmp["v_high"] > e.cmp["v_low"])))
        else:
            c0 = base + 7 * e.idx
            out.append(SigSpec(f"I({e.name}.d)", _L(f"{e.name} 드레인 전류 (단자로 유입)", f"{e.name} drain current (into the terminal)"),
                               "A", "current", ("col", c0 + 3)))
            out.append(SigSpec(f"I({e.name}.s)", _L(f"{e.name} 소스 전류 (단자로 유입)", f"{e.name} source current (into the terminal)"),
                               "A", "current", ("neg", c0 + 3)))
            out.append(SigSpec(f"I({e.name}.g)", _L(f"{e.name} 게이트 전류 (이상적 게이트: 0)", f"{e.name} gate current (ideal gate: 0)"),
                               "A", "current", ("zero",)))
            k += 1
    for e in els:
        if e.type != "STL":
            continue
        c0 = base + 7 * e.idx
        out.append(SigSpec(f"{e.name}.u", _L(f"{e.name} 소스–바디 준페르미 분리 u", f"{e.name} source–body quasi-Fermi splitting u"),
                           "V", "state", ("col", c0), True))
        out.append(SigSpec(f"{e.name}.r", _L(f"{e.name} 드레인 접합 역바이어스 r", f"{e.name} drain-junction reverse bias r"),
                           "V", "state", ("col", c0 + 1), True))
        out.append(SigSpec(f"{e.name}.q_b", _L(f"{e.name} 바디 전하 변화 ΔQ_B", f"{e.name} body-charge change ΔQ_B"),
                           "C", "charge", ("qb", c0 + 2), True))
        if cell_ls[e.idx] is not None:
            u = cell_ls[e.idx]
            out.append(SigSpec(f"{e.name}.dphi", _L(f"{e.name} 국소 상태 편차 δ", f"{e.name} local-state deviation δ"),
                               u, "state", ("col", c0 + 5), True))
            if sigE[e.idx]:
                out.append(SigSpec(f"{e.name}.dphi_E", _L(f"{e.name} 이미터 상태 편차 δφ_E", f"{e.name} emitter-state deviation δφ_E"),
                                   "V", "state", ("col", c0 + 6), True))
    return out


def _convert(rows: np.ndarray, specs: list[SigSpec], q0: dict) -> np.ndarray:
    t = rows[:, 0]
    out = np.empty((len(rows), len(specs)))
    for j, sp in enumerate(specs):
        r = sp.recipe
        k = r[0]
        if k == "col":
            out[:, j] = rows[:, r[1]] if r[1] > 0 else 0.0
        elif k == "neg":
            out[:, j] = -rows[:, r[1]]
        elif k == "rcur":
            va = rows[:, r[1]] if r[1] > 0 else 0.0
            vb = rows[:, r[2]] if r[2] > 0 else 0.0
            out[:, j] = (va - vb) * r[3]
        elif k == "wave":
            out[:, j] = np.interp(t, r[1], r[2])
        elif k == "basic":
            out[:, j] = current_series(rows, r[1], r[2], r[3])
        elif k == "qb":
            out[:, j] = rows[:, r[1]] - q0.get(r[1], 0.0)
        elif k == "vds":
            va = rows[:, r[1]] if r[1] > 0 else 0.0
            vb = rows[:, r[2]] if r[2] > 0 else 0.0
            out[:, j] = va - vb
        elif k == "bit":
            v = rows[:, r[1]] if r[1] > 0 else np.zeros(len(rows))
            out[:, j] = ((v > r[2]) if r[3] else (v < r[2])).astype(float)
        else:
            out[:, j] = 0.0
    return out


def _select(t: np.ndarray, F: np.ndarray | None, n_max: int, forced: np.ndarray | None = None) -> np.ndarray:
    """Row selection for plotting: forced rows (corners, events), 60 % arc length over the normalised
    features, 40 % uniform in time.  Never more than n_max rows."""
    n = len(t)
    if n <= n_max:
        return np.arange(n)
    fz = np.zeros(0, np.int64) if forced is None or len(forced) == 0 else np.unique(np.clip(forced, 0, n - 1))
    cap = max(n_max // 3, 1)
    if len(fz) > cap:
        fz = fz[np.round(np.linspace(0, len(fz) - 1, cap)).astype(np.int64)]
    k = max(n_max - 2 - len(fz), 2)
    k_arc = int(0.6 * k)
    k_time = k - k_arc
    T = max(t[-1] - t[0], 1e-300)
    comps = [((t - t[0]) / T)[:, None]]
    if F is not None and F.size:
        lo = np.nanmin(F, axis=0)
        rg = np.nanmax(F, axis=0) - lo
        rg = np.where(rg > 0, rg, 1.0)
        comps.append(np.nan_to_num((F - lo) / rg))
    X = np.hstack(comps)
    seg = np.sqrt(np.sum(np.diff(X, axis=0) ** 2, axis=1))
    s = np.r_[0.0, np.cumsum(seg)]
    ia = np.searchsorted(s, np.linspace(0, s[-1], k_arc)) if s[-1] > 0 else np.zeros(0, np.int64)
    it = np.searchsorted(t, np.linspace(t[0], t[-1], k_time))
    return np.unique(np.clip(np.r_[0, n - 1, fz, ia, it], 0, n - 1)).astype(np.int64)


class _Sink:
    """Receives every recorded kernel row block, converts it to the output signals, applies t_start_save
    and reduces the buffer on the fly (bounded memory for up to 2e6 steps)."""

    def __init__(self, specs: list[SigSpec], feat: list[int], logfeat: list[int], t_save: float,
                 bp: np.ndarray, qcols: list[int]):
        self.specs = specs
        self.feat = feat
        self.logfeat = logfeat
        self.t_save = t_save
        self.bp = bp
        self.qcols = qcols
        self.q0: dict[int, float] = {}
        self.first_raw: np.ndarray | None = None
        self.tp: list[np.ndarray] = []
        self.sp: list[np.ndarray] = []
        self.n = 0
        self.prev: tuple[float, np.ndarray] | None = None
        self.started = False
        self.ev_t: list[float] = []
        self.cmp: _CmpTrack | None = None
        ncol = max(len(specs), 1)
        self.max_rows = int(np.clip(4e6 / ncol, 8000, 60000))
        self.keep_rows = self.max_rows // 2

    def features(self, S: np.ndarray) -> np.ndarray:
        parts = [S[:, self.feat]] if self.feat else []
        if self.logfeat:
            parts.append(0.3 * np.log10(np.abs(S[:, self.logfeat]) + 1e-15))
        return np.hstack(parts) if parts else np.zeros((len(S), 0))

    def forced(self, t: np.ndarray) -> np.ndarray:
        out = []
        if len(self.bp):
            j = np.searchsorted(t, self.bp)
            out += [j - 1, j]
        if self.ev_t:
            j = np.searchsorted(t, np.asarray(self.ev_t))
            out += [j - 1, j]
        if not out:
            return np.zeros(0, np.int64)
        f = np.concatenate(out)
        return f[(f >= 0) & (f < len(t))]

    def __call__(self, rows: np.ndarray, events: np.ndarray) -> None:
        if len(events):
            self.ev_t.extend(events[:, 2].tolist())
        if not len(rows):
            return
        if self.cmp is not None:
            self.cmp.feed(rows)
        if self.first_raw is None:
            self.first_raw = rows[0].copy()
            self.q0 = {c: float(rows[0, c]) for c in self.qcols}
        S = _convert(rows, self.specs, self.q0)
        t = rows[:, 0].copy()
        if not self.started:
            keep = t >= self.t_save * (1 - 1e-12)
            if not keep.any():
                self.prev = (float(t[-1]), S[-1].copy())
                return
            i0 = int(np.argmax(keep))
            if t[i0] > self.t_save and (i0 > 0 or self.prev is not None):
                ta, Sa = (float(t[i0 - 1]), S[i0 - 1]) if i0 > 0 else self.prev
                a = (self.t_save - ta) / (t[i0] - ta) if t[i0] > ta else 1.0
                Si = Sa + a * (S[i0] - Sa)
                t = np.r_[self.t_save, t[i0:]]
                S = np.vstack([Si[None, :], S[i0:]])
            else:
                t, S = t[i0:], S[i0:]
            self.started = True
        self.tp.append(t)
        self.sp.append(S)
        self.n += len(t)
        if self.n > self.max_rows:
            self._reduce(self.keep_rows)

    def _reduce(self, n_target: int) -> None:
        t = np.concatenate(self.tp)
        S = np.vstack(self.sp)
        idx = _select(t, self.features(S), n_target, self.forced(t))
        self.tp, self.sp = [t[idx]], [S[idx]]
        self.n = len(idx)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.tp:
            if self.prev is not None:
                return np.array([self.prev[0]]), self.prev[1][None, :]
            return np.zeros(0), np.zeros((0, len(self.specs)))
        return np.concatenate(self.tp), np.vstack(self.sp)


class _CmpTrack:
    """Full-resolution comparator bookkeeping of one run (fed with every raw row block by the sink): output
    edges (mid-level crossings, linearly interpolated), time spent high, and whether the output was high at
    any time within each window (one window per period of the pulse source)."""

    def __init__(self, cmps: list[dict], windows: tuple | None):
        self.cmps = cmps
        self.win = windows
        nw = len(windows[1]) if windows else 0
        self.fired = np.zeros((len(cmps), nw), bool)
        self.edges: list[list[tuple[float, int]]] = [[] for _ in cmps]
        self.t_high = np.zeros(len(cmps))
        self.prev: list[tuple[float, float, bool] | None] = [None] * len(cmps)
        self.t_last = 0.0

    def feed(self, rows: np.ndarray) -> None:
        if not len(rows):
            return
        t = rows[:, 0]
        self.t_last = float(t[-1])
        for j, c in enumerate(self.cmps):
            v = rows[:, c["col"]] if c["col"] > 0 else np.zeros(len(rows))
            hi = (v > c["mid"]) if c["up"] else (v < c["mid"])
            pv = self.prev[j]
            tt, vv, hh = (np.r_[pv[0], t], np.r_[pv[1], v], np.r_[pv[2], hi]) if pv is not None else (t, v, hi)
            if len(tt) > 1:
                ch = np.flatnonzero(hh[1:] != hh[:-1]) + 1
                for i in ch:
                    dv = vv[i] - vv[i - 1]
                    a = (c["mid"] - vv[i - 1]) / dv if dv != 0 else 1.0
                    self.edges[j].append((float(tt[i - 1] + min(max(a, 0.0), 1.0) * (tt[i] - tt[i - 1])), 1 if hh[i] else 0))
                self.t_high[j] += float(np.sum(np.diff(tt)[hh[:-1]]))
            if self.win is not None and hi.any():
                starts, ends = self.win[1], self.win[2]
                th = t[hi]
                k = np.searchsorted(starts, th, side="right") - 1
                ok = (k >= 0) & (th < ends[np.clip(k, 0, len(ends) - 1)])
                self.fired[j, k[ok]] = True
            self.prev[j] = (float(t[-1]), float(v[-1]), bool(hi[-1]))


def _pulse_windows(els: list[El], t_stop: float) -> tuple | None:
    """Firing windows of the comparators: one per period of the periodic pulse source with the most complete
    periods (a window counts when the pulse's flat top ends before t_stop).  (source name, starts, ends)."""
    best = None
    for e in els:
        if e.type not in ("V", "I") or e.wave is None or e.wave.resolved.get("kind") != "pulse":
            continue
        r = e.wave.resolved
        if not (r["per"] > 0) or r["v1"] == r["v2"]:
            continue
        n = int(np.floor((t_stop - r["td"] - r["tr"] - r["pw"]) / r["per"])) + 1
        if r["ncycles"] > 0:
            n = min(n, r["ncycles"])
        if n >= 2 and (best is None or n > len(best[1])):
            starts = r["td"] + r["per"] * np.arange(n)
            best = (e.name, starts, np.minimum(starts + r["per"], t_stop))
    return best


def _drain_capacitance(e: El, els: list[El]) -> float:
    """Capacitance across a cell for the quasi-static walk: every capacitor from the drain node — or from a node
    tied to it by a resistor below 1 MΩ — to any other node (approximation: the far end is taken as AC ground)."""
    d = e.nodes[0]
    grp = {d}
    grow = True
    while grow:
        grow = False
        for x in els:
            if x.type == "R" and x.value is not None and x.value < 1e6:
                a, b = x.nodes
                if (a in grp) != (b in grp) and "0" not in (a, b):
                    grp |= {a, b}
                    grow = True
    return float(sum(x.value for x in els if x.type == "C" and x.value and ((x.nodes[0] in grp) != (x.nodes[1] in grp))))


def _osc_warning(e: El, osc: dict, prof: dict, t_stop: float, warnings: list[str]) -> None:
    """Explain what a high-impedance cell is expected to do (quasi-static load-line analysis, cell's nominal V_GS)."""
    w = osc["walks"][0]
    fi = prof.get("fold_I") or {}
    lo, hi = osc["i_n"]
    feed = _fmt(lo, "A") if abs(hi - lo) <= 1e-3 * max(abs(hi), 1e-30) else f"{_fmt(lo, 'A')} … {_fmt(hi, 'A')}"
    rx = f"R_ext ≈ {_fmt(osc['r_ext'], 'Ω')}" if osc["r_ext"] and osc["r_ext"] < 1e13 else "current source"
    if not prof.get("latch"):
        return
    ilu, ild = fi.get("hrs_at_lu"), fi.get("lrs_at_ld")
    win = (f" (fold currents I_LU = {_fmt(ilu, 'A')}, I_LD = {_fmt(ild, 'A')})" if ilu and ild else "")
    if w["oscillating"]:
        tc, td = quasi_static_period(0.5 * (lo + hi), osc["g_ext"], osc["c_eff"], prof)
        per = w["period"] if w["period"] else tc + td
        warnings.append(f"{e.name}: relaxation oscillator — fed by {feed} ({rx}) with {_fmt(osc['c_eff'], 'F')} across "
                        f"the cell, the load line crosses only the negative-resistance branch{win}: V_DS saws between "
                        f"≈ V_LD and V_LU with a quasi-static period ≈ {_fmt(per, 's')} (~{w['n_lu'] * w['scale']:.0f} "
                        "latch-ups in t_stop; the fold lags lengthen it by a few %)")
    elif w["n_lu"] == 0 and hi > 0:
        warnings.append(f"{e.name}: high-impedance drive ({feed}, {rx}): the load line crosses the HRS{win}, the cell "
                        f"settles near V_DS ≈ {w['v'][-1]:.3g} V without latching (no oscillation)")
    elif w["n_lu"] == 1 and w["n_ld"] == 0:
        warnings.append(f"{e.name}: high-impedance drive ({feed}, {rx}): the load line crosses the LRS{win}, the cell "
                        "latches and stays latched (no oscillation)")


# ================================================================== profiles (cached)
_PROFILE_CACHE: dict[bytes, dict] = {}


def _profile(p: np.ndarray) -> dict:
    from .stochastic import branch_profile
    key = np.ascontiguousarray(p, dtype=float).tobytes()
    pr = _PROFILE_CACHE.get(key)
    if pr is None:
        pr = branch_profile(np.asarray(p, float), 301)
        if len(_PROFILE_CACHE) > 128:
            _PROFILE_CACHE.clear()
        _PROFILE_CACHE[key] = pr
    return pr


def _stats(x) -> tuple[float | None, float | None]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None, None
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else None)


def _item(key, ko, en, value, unit=None, spread=None):
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    d = dict(key=key, label=_L(ko, en), value=value)
    if unit is not None:
        d["unit"] = unit
    if spread is not None:
        d["spread"] = float(spread) if math.isfinite(spread) else None
    return d


# ================================================================== main
def run_custom(payload: dict, progress: Callable[[float, str], None] | None = None) -> dict:
    from server.payloads import normalize_device

    from . import runner as RN
    from .stochastic import (ACTION_UNIT, estimate_steps, noise_bands, parse_local_state, GEOMETRY_NOISE_ERROR)

    tic = time.perf_counter()
    progress = progress or (lambda f, msg="": None)
    warnings: list[str] = []
    mode = payload.get("mode", "deterministic")
    if mode not in ("deterministic", "stochastic"):
        raise ValueError("mode must be 'deterministic' or 'stochastic'")
    stochastic = mode == "stochastic"
    for k in ("bench_params",):
        if payload.get(k):
            warnings.append(f"{k} is ignored for bench 'custom' (the circuit is given by netlist)")

    # ---- tran ----
    tran_in = payload.get("tran")
    if tran_in is None:
        raise ValueError("tran is required for bench 'custom' ({t_stop_s, dt_max_s, ...})")
    if not isinstance(tran_in, dict):
        raise ValueError("tran must be an object")
    t_stop = _f(tran_in, "t_stop_s", "tran", positive=True, hi=T_STOP_MAX)
    t_save = _f(tran_in, "t_start_save_s", "tran", 0.0, 0.0)
    if t_save >= t_stop:
        raise ValueError(f"tran.t_start_save_s = {t_save:g} s must be smaller than t_stop_s = {t_stop:g} s")
    dt_max = tran_in.get("dt_max_s")
    dt_max = t_stop / 2000.0 if dt_max in (None, 0) else _f(tran_in, "dt_max_s", "tran", positive=True)
    if dt_max > t_stop:
        warnings.append(f"tran.dt_max_s = {dt_max:g} s is larger than t_stop_s; using {t_stop:g} s")
        dt_max = t_stop
    dt_min = tran_in.get("dt_min_s")
    dt_min = max(1e-15, 1e-13 * t_stop) if dt_min in (None, 0) else _f(tran_in, "dt_min_s", "tran", positive=True)
    if dt_min < 1e-18:
        raise ValueError("tran.dt_min_s must be >= 1e-18 s")
    if dt_min >= dt_max:
        raise ValueError(f"tran.dt_min_s = {dt_min:g} s must be smaller than dt_max_s = {dt_max:g} s")
    method = tran_in.get("method", "BE") or "BE"
    if method not in ("BE", "TRAP"):
        raise ValueError("tran.method must be 'BE' or 'TRAP'")
    reltol = _f(tran_in, "reltol", "tran", 1e-3, 1e-5, 0.1)
    initial = tran_in.get("initial", "auto")
    initial = "auto" if initial in (None, "") else initial
    if initial not in ("auto", "op", "zero"):
        raise ValueError("tran.initial must be 'auto' (default), 'op' (DC operating point) or 'zero' (discharged "
                         "capacitors, SPICE UIC)")
    sol_in = dict(payload.get("solver") or {})
    for k in ("method", "dt_min_s", "dt_max_s", "reltol"):
        if k in sol_in:
            warnings.append(f"solver.{k} is ignored for bench 'custom' (set it in tran)")
    sol_in.update(method=method, dt_min_s=dt_min, dt_max_s=dt_max, reltol=reltol)
    sol = RN._solver({"solver": sol_in}, t_stop, warnings)
    edge0 = max(10 * dt_min, min(dt_max, 1e-3 * t_stop))

    # ---- netlist + ERC ----
    progress(0.01, "netlist / ERC")
    els = parse_elements(payload.get("netlist"), t_stop, edge0, warnings)
    nodes = erc(els, warnings)
    stls = [e for e in els if e.type == "STL"]
    ns = len(stls)

    # ---- stochastic block, detection ----
    st = RN._stochastic(payload, mode, None, warnings)
    det = RN._detect(payload)
    ls_global = parse_local_state(st["local_state"], warnings) if stochastic else parse_local_state(None, warnings)
    override = bool((payload.get("stochastic") or {}).get("local_state_override", False))
    carrier = stochastic and st["carrier_noise"] and ns > 0

    # ---- STL devices ----
    cell_ls = []                          # LocalStateConfig per cell (stochastic) or None
    for e in stls:
        try:
            dev = normalize_device(e.device, warnings)
        except ValueError as exc:
            raise ValueError(f"{e.name}: {exc}") from None
        if stochastic and PR.uses_geometry_model(dev):
            raise ValueError(f"{GEOMETRY_NOISE_ERROR} ({e.name})")
        cfg_k = ls_global
        if stochastic and e.ls_block is not None and not override:
            try:
                cfg_k = parse_local_state(e.ls_block, warnings)
            except ValueError as exc:
                raise ValueError(f"{e.name}: {exc}") from None
        if stochastic and cfg_k.mode != "none" and cfg_k.action == "local_avalanche" and float(dev["ext"].get("aloc", 0.0)) <= 0:
            warnings.append(f"{e.name}: local_avalanche action needs ext.aloc (p[21]) > 0; using aloc = 1.0")
            dev = dict(dev, ext=dict(dev["ext"], aloc=1.0))
        if stochastic and cfg_k.mode != "none" and cfg_k.action != "gidl":
            warnings.append(f"{e.name}: action point '{cfg_k.action}' is an experimental hypothesis lever, not the "
                            "calibrated GIDL action point")
        e.device = dev
        e.p = np.array(PR.build_p(dev), dtype=float)
        cell_ls.append(cfg_k if (stochastic and cfg_k.mode != "none") else None)
    if stochastic:
        any_ls = any(c is not None for c in cell_ls)
        if ns == 0:
            warnings.append("stochastic mode without STL cells has no random input: running 1 run")
            st["n_runs"] = 1
        elif not st["carrier_noise"] and not any_ls and st["n_runs"] > 1:
            warnings.append("stochastic mode with carrier_noise = false and no local states has no random input: "
                            f"running 1 run instead of {st['n_runs']} identical ones (use the deterministic mode)")
            st["n_runs"] = 1
    n_runs = st["n_runs"] if stochastic else 1

    # ---- netlist for the kernel ----
    net = Netlist()
    for n in nodes[1:]:
        net.node(n)
    for e in els:
        if e.type == "R":
            e.idx = len(net.R)
            net.add_R(e.name, e.nodes[0], e.nodes[1], e.value)
        elif e.type == "C":
            if e.value > 0:
                e.idx = len(net.C)
                net.add_C(e.name, e.nodes[0], e.nodes[1], e.value)
        elif e.type == "V":
            e.idx = len(net.V)
            e.wave_idx = net.add_V(e.name, e.nodes[0], e.nodes[1], e.wave.t, e.wave.v, _wave_text(e.wave.resolved, "V"))
        elif e.type == "I":
            e.idx = len(net.I)
            e.wave_idx = net.add_I(e.name, e.nodes[0], e.nodes[1], e.wave.t, e.wave.v, _wave_text(e.wave.resolved, "A"))
        elif e.type in BASIC_PINS:
            e.idx = len(net.basic)
            net.add_basic(e.type, e.name, e.nodes, e.model)
    for e in els:
        if e.type == "CMP":                       # after every V source: branch index = len(V sources) + j
            c = e.cmp
            e.idx = net.add_CMP(e.name, e.nodes[0], e.nodes[2], c["v_ref"], c["v_high"], c["v_low"], c["hysteresis"],
                                c["width"], inm=e.nodes[1], label=_cmp_label(e))
    for e in stls:
        e.idx = len(net.STL)
        light = (e.light.t, e.light.v) if e.light is not None else None
        net.add_STL(e.name, e.nodes[0], e.nodes[1], e.nodes[2], e.p, light=light, label=_stl_label(e))
    net.t_end = t_stop
    net.main_wave = -1
    if [n for n in net.nodes] != nodes:
        raise RuntimeError("node order mismatch")

    # ---- linear DC estimate: V_GS range, drives, look-ahead gains ----
    progress(0.02, "device branches / noise bands")
    S_src, srcs, S_stl = linear_dc(nodes, els)
    src_waves = [e.wave for e in srcs]
    tg = _union_grid(src_waves, t_stop)
    Wg = np.array([np.interp(tg, w.t, w.v) for w in src_waves]) if srcs else np.zeros((0, len(tg)))
    rowi = {n: i - 1 for i, n in enumerate(nodes)}

    def node_resp(n: str) -> np.ndarray:
        return S_src[rowi[n]] if rowi[n] >= 0 else np.zeros(len(srcs))

    cells = []
    for k, e in enumerate(stls):
        d, g, s = e.nodes
        sg = node_resp(g) - node_resp(s)
        vgs = sg @ Wg if len(srcs) else np.zeros(len(tg))
        vgs_lo, vgs_hi = float(np.min(vgs)), float(np.max(vgs))
        sd = node_resp(d) - node_resp(s)
        drive = sd @ Wg if len(srcs) else np.zeros(len(tg))
        rth = float((S_stl[rowi[d], k] if rowi[d] >= 0 else 0.0) - (S_stl[rowi[s], k] if rowi[s] >= 0 else 0.0))
        cells.append(dict(vgs=vgs, vgs_lo=vgs_lo, vgs_hi=vgs_hi, sd=sd, drive=drive, rth=rth))

    # [P1-13 stop-gap] the linear DC estimate leaves MOS/D/BJT out (G_OFF): a cell whose drain, gate or source net
    # touches one of their pins gets estimated V_GS / drive values that may be far off, so the estimate-based
    # oscillation and no-latch predictions are not reported for it and its V_GS is marked as an estimate (the
    # estimate still sizes the run and picks the initial state)
    tx_nets: dict[str, list[str]] = {}
    for x in els:
        if x.type in BASIC_PINS:
            for n in x.nodes:
                if n != "0" and x.name not in tx_nets.setdefault(n, []):
                    tx_nets[n].append(x.name)
    wins = np.zeros((ns, 15))
    cell_info = []
    total_est_cells = 0.0
    base_cells = 0.0
    est_parts = []
    dv_max = float(np.clip(0.02 * reltol / 1e-3, 1e-3, 0.2))
    n_bp = len(net.breakpoints())
    for k, e in enumerate(stls):
        c = cells[k]
        tx = sorted({nm for n in set(e.nodes) for nm in tx_nets.get(n, [])})
        tx_note = f" without the transistor network ({', '.join(tx)})" if tx else ""
        vg_dev = float(e.p[11])
        vgs_nom = 0.5 * (c["vgs_lo"] + c["vgs_hi"])
        vg_vals = [vgs_nom] if c["vgs_hi"] - c["vgs_lo"] <= 1e-3 else [c["vgs_lo"], c["vgs_hi"]]
        if len(vg_vals) > 1:
            warnings.append(f"{e.name}: the gate-source voltage varies ({c['vgs_lo']:.4g} … {c['vgs_hi']:.4g} V, linear "
                            "estimate): latch-state fold values and noise bands are taken over this range")
        if abs(vgs_nom - vg_dev) > 1e-3 and len(vg_vals) == 1:
            warnings.append(f"{e.name}: V_GS = {vgs_nom:.4g} V comes from the circuit"
                            f"{' (linear estimate' + tx_note + ')' if tx else ''}; the device block's V_G = "
                            f"{vg_dev:g} V is not used")
        base_light = float(e.p[13])
        if e.light is not None:
            lv = e.light.v[e.light.t <= t_stop] if np.any(e.light.t <= t_stop) else e.light.v[:1]
            lv = np.r_[lv, np.interp(t_stop, e.light.t, e.light.v)]
            li_vals = [float(lv.min()), float(lv.max())] if np.ptp(lv) > 1e-3 * max(lv.max(), 1e-18) else [float(lv.mean())]
        else:
            li_vals = [base_light]
        variants = []
        for vg_ in vg_vals:
            for li in li_vals:
                pv = e.p.copy()
                pv[11] = vg_
                pv[13] = li
                variants.append(pv)
        profs = [_profile(pv) for pv in variants]
        pnom = profs[0]
        lsk = cell_ls[k]
        bands = [noise_bands(pr, lsk, stochastic, sol["noise_z_max"]) for pr in profs]
        b = (min(x[0] for x in bands), max(x[1] for x in bands), min(x[2] for x in bands), max(x[3] for x in bands))
        lat = [pr for pr in profs if pr["latch"]]
        u_i = min(pr["u_fold"][0] for pr in lat) if lat else np.inf
        u_j = max(pr["u_fold"][1] for pr in lat) if lat else np.inf
        if tx:
            pass                                  # V_GS unknown before the run: no latch-window prediction
        elif not lat:
            warnings.append(f"{e.name}: no latch window at V_GS = {vgs_nom:.4g} V"
                            f"{' (locus gap at the fold, channel-on regime)' if pnom.get('gap') else ''}: no latch-up expected")
        elif len(lat) < len(profs):
            warnings.append(f"{e.name}: the latch window disappears over part of the V_GS / light range")
        if tx and stochastic:
            warnings.append(f"{e.name}: the carrier-noise band near the folds was set from the linear V_GS estimate"
                            f"{tx_note}; with a different actual V_GS the noise near the folds is approximate")
        RN._check_thresholds(det, pnom, warnings, e.name)
        # noise look-ahead drive: the source that moves V_DS most
        law, lag = -1, 0.0
        best = 0.0
        for j, se in enumerate(srcs):
            if not se.wave.varying:
                continue
            amp = abs(c["sd"][j]) * float(np.ptp(Wg[j]))
            if amp > best and amp > 1e-6:
                best, law, lag = amp, se.wave_idx, float(c["sd"][j])
        cfgk = lsk
        wins[k, :] = [b[0], b[1], b[2], b[3], u_i, u_j, law, lag if law >= 0 else 0.0, 1.0,
                      cfgk.mode_code if cfgk else 0, cfgk.index if cfgk else 9, cfgk.sigma if cfgk else 0.0,
                      cfgk.tau_s if cfgk else 1.0, cfgk.sigma_E_V if cfgk else 0.0, cfgk.tau_E_s if cfgk else 1.0]
        # feasibility along the drive: the open-circuit (Thevenin) voltage for low-impedance drives; for a
        # high-impedance cell (current source, R >= 100 MΩ) the quasi-static walk of its own V_DS with the
        # capacitance on its drain (relaxation oscillator when the load line crosses only the NDR branch)
        drive = np.asarray(c["drive"], float)
        rth = abs(c["rth"])
        g_ext = max(1.0 / rth - G_OFF, 0.0) if rth > 0 else np.inf
        c_eff = _drain_capacitance(e, els)
        osc = None
        if rth > R_HIGH and c_eff > 0:
            i_n = drive / c["rth"]
            walks = [qs_drive(tg, i_n, g_ext, c_eff, pr, t_stop, dt_max) for pr in profs]
            osc = dict(walks=walks, i_n=(float(np.min(i_n)), float(np.max(i_n))), g_ext=g_ext, c_eff=c_eff,
                       r_ext=(1.0 / g_ext if g_ext > 1e-15 else None))
            drives = [(w["t"], w["v"], np.zeros(max(len(w["t"]) - 2, 0)), w["scale"], OSC_TRANSITION_STEPS) for w in walks]
            if not tx:                            # the walk still sizes the run and picks the initial state
                _osc_warning(e, osc, pnom, t_stop, warnings)
        else:
            if rth > 1e11:
                drive = np.clip(drive, -8.5, 8.5)
            cw = _corner_weights(tg, drive[None, :])
            drives = [(tg, drive, cw, 1.0, None)] * len(profs)
        e_k = 0.0
        for pr, bb, (dt_, dv_, cw_, sc_, tr_) in zip(profs, bands, drives):
            e_k = max(e_k, sc_ * estimate_steps(dt_, dv_, t_stop / sc_, pr, stochastic, carrier, dt_max, dv_max,
                                                sol["tau_frac"], sol["max_events_per_step"], sol["noise_dt_min_s"],
                                                n_bp if sc_ == 1.0 else 0, ld_noise=bool(st["ld_carrier_noise"]) and carrier,
                                                gauss_tau_min=sol["gauss_tau_min_s"], gauss_tau_frac=sol["gauss_tau_frac"],
                                                window=bb, corner_weights=cw_, transition_steps=tr_))
        # the part shared with the other cells / the linear network (dt_max, Δv, corners) vs this cell's own
        # extra steps (latch transitions, resolved noise): cells add their extras, the shared part counts once
        flat = dict(pnom, folds=(np.nan, np.nan), latch=False)
        dt_, dv_, cw_, sc_, _ = drives[0]
        base_k = sc_ * estimate_steps(dt_, dv_, t_stop / sc_, flat, False, False, dt_max, dv_max, sol["tau_frac"],
                                      sol["max_events_per_step"], sol["noise_dt_min_s"], n_bp if sc_ == 1.0 else 0,
                                      corner_weights=cw_)
        est_parts.append(e_k)
        base_cells = max(base_cells, base_k)
        total_est_cells += max(e_k - base_k, 0.0)
        cell_info.append(dict(vgs_nom=vgs_nom, vgs_range=(c["vgs_lo"], c["vgs_hi"]), light_range_A=(min(li_vals), max(li_vals)),
                              folds=pnom["folds"], latch=bool(lat), u_fold=(u_i, u_j), band=b, rth=c["rth"],
                              lookahead=(law, lag), est=e_k, osc=osc, fold_I=pnom.get("fold_I"), tx=tx))

    # ---- feasibility: linear part (every node) + cells ----
    if len(srcs) and len(tg) > 1:
        # nodes held only by the open-cell conductance G_OFF (current-driven STLs) would swing to kV in the
        # linear estimate: clip to the plausible range of the circuit
        v_clip = max(10.0, 1.5 * max((float(np.max(np.abs(e.wave.v))) for e in els if e.type == "V"), default=0.0))
        Vn = np.clip(S_src @ Wg, -v_clip, v_clip)
        # the drain (and source) nodes of high-impedance cells follow the cell, not the linear network: their steps
        # are counted by the cell's quasi-static walk (current-source nodes would otherwise swing by kV here)
        walk_rows = sorted({rowi[n] for k, e in enumerate(stls) if cell_info[k]["osc"] is not None
                            for n in (e.nodes[0], e.nodes[2]) if rowi[n] >= 0})
        keep_rows = np.ones(len(Vn), bool)
        keep_rows[walk_rows] = False
        Vk = Vn[keep_rows]
        dT = np.diff(tg)
        dV = np.abs(np.diff(Vk, axis=1)).max(axis=0) if len(Vk) else np.zeros(len(dT))
        lin = float(np.sum(np.maximum(dT / dt_max, dV / dv_max)))
        # upper bound of the node-voltage total variation from each source's own total variation (exact when
        # one source dominates; protects against aliasing of a subsampled grid); a node cannot swing by more than
        # 2 v_clip per source corner
        tv = np.array([_total_variation(w, t_stop) for w in src_waves])
        if len(Vk):
            tvb = np.minimum(np.abs(S_src[keep_rows]) @ tv, 2.0 * v_clip * (n_bp + 1))
            lin = max(lin, float(np.max(tvb)) / dv_max)
        cw_all = _corner_weights(tg, Vk) if len(Vk) else np.zeros(0)
        lin += 60.0 * float(cw_all.sum()) + 6.0 * max(n_bp - float(cw_all.sum()), 0.0)
    else:
        lin = t_stop / dt_max + 6.0 * n_bp
    est = max(lin, base_cells) + total_est_cells
    # measured (4-CPU container, warm cache): ~3 µs per step without STL, +60-90 µs per STL cell
    sec_step = 3e-6 + 70e-6 * ns + 1e-9 * (len(nodes) + len(net.V) + 2 * ns) ** 3
    total_est = est * n_runs
    # initial state: 'auto' starts from discharged capacitors right away when a high-impedance cell's load line at
    # t = 0 misses the HRS (current bias between the folds -> NDR equilibrium, or above I_LD -> LRS): the DC operating
    # point from the empty body is then an unstable equilibrium or does not exist (the kernel also checks, sim.simulate)
    initial_run = initial
    pre_zero = []
    if initial == "auto":
        for k, ci_ in enumerate(cell_info):
            osc = ci_["osc"]
            fi = ci_["fold_I"] or {}
            if osc is None or not ci_["latch"] or not fi:
                continue
            i0 = float(cells[k]["drive"][0] / cells[k]["rth"])
            if i0 - osc["g_ext"] * ci_["folds"][0] > fi["hrs_at_lu"]:
                pre_zero.append(stls[k].name)
        if pre_zero:
            initial_run = "zero"
    n_osc = sum(ci_["osc"]["walks"][0]["n_lu"] * ci_["osc"]["walks"][0]["scale"] for ci_ in cell_info
                if ci_["osc"] is not None and ci_["osc"]["walks"][0]["oscillating"])
    if est > 2.0 * sol["max_steps"]:
        if n_osc >= 2:
            why = (f"relaxation oscillation: ~{n_osc:.0f} predicted latch-up/latch-down cycles in t_stop, each resolved "
                   "with several hundred steps, more with event-level carrier noise near the fold). Shorten t_stop, "
                   "increase the capacitance or reduce the drive current (longer period), or raise solver.max_steps.")
        elif carrier and total_est_cells > max(lin, base_cells):
            why = ("event-level carrier noise needs h <= tau_frac·tau_rel (µs) while a cell's V_DS is inside its noise "
                   "band). Use a shorter t_stop or faster edges/ramps, the deterministic mode, carrier_noise = false "
                   "(local states only) or the device-level stochastic MC.")
        else:
            why = ("the waveforms are long compared with the step limits dt_max and |Δv| <= "
                   f"{dv_max * 1e3:.3g} mV per step (reltol) and the resolved switching transients). Shorten t_stop, "
                   "increase tran.dt_max_s / tran.reltol or reduce the number of pulses/periods.")
        # "circuit-step-budget:" is a stable prefix the web app translates (web/src/api/geometryPolicy.ts)
        raise ValueError(f"circuit-step-budget: estimated ~{est:.3g} time steps per run exceed 2 x solver.max_steps = "
                         f"{2 * sol['max_steps']:.0f} (~{est * sec_step:.0f} s per run; {why}")
    if est > 0.5 * sol["max_steps"]:
        warnings.append(f"estimated ~{est:.3g} steps per run is close to solver.max_steps = {sol['max_steps']:.0f}; "
                        "runs may be truncated")
    if total_est > MAX_TOTAL_STEPS:
        raise ValueError(f"circuit-step-budget: estimated total work ~{total_est:.3g} time steps "
                         f"(~{total_est * sec_step / 60:.0f} min) exceeds the per-request limit {MAX_TOTAL_STEPS:.0g}; "
                         "reduce n_runs or t_stop")
    if total_est * sec_step > 120:
        warnings.append(f"estimated run time ~{total_est * sec_step:.0f} s")

    # ---- probes / signals ----
    layout = dict(nn=len(nodes), nv=len(net.V), ns=ns)
    units = [ACTION_UNIT[c.action] if c is not None else None for c in cell_ls]
    sigE = [bool(c is not None and c.sigma_E_V > 0) for c in cell_ls]
    specs_all = _signal_specs(nodes, els, layout, units, sigE)
    by_key = {sp.key: sp for sp in specs_all}
    probes_in = payload.get("probes")
    if probes_in is None:
        specs = list(specs_all)
    else:
        if not isinstance(probes_in, list):
            raise ValueError("probes must be null or a list of signal names such as \"V(d)\", \"I(R1)\", \"I(X1.d)\"")
        if len(probes_in) == 0:
            raise ValueError("probes is an empty list: pass null to record every signal")
        specs = []
        for pk in probes_in:
            if not isinstance(pk, str):
                raise ValueError(f"probe {str(pk)[:40]!r} must be a string")
            key = pk.strip()
            m = re.match(r"^V\((.+)\)$", key)
            if m and (m.group(1) in GROUND_ALIASES or m.group(1).lower() == "gnd"):
                sp = SigSpec("V(0)", _L("접지 전압 (0 V)", "Ground (0 V)"), "V", "voltage", ("zero",))
            elif key in by_key:
                sp = by_key[key]
            else:
                hint = ""
                if m:
                    hint = f" (no node {m.group(1)!r}; nodes: {', '.join(nodes[1:12])}{' ...' if len(nodes) > 12 else ''})"
                else:
                    mi = re.match(r"^I\(([^.()]+)(?:\.(\w+))?\)$", key)
                    names = {e.name: e for e in els}
                    if mi and mi.group(1) not in names:
                        hint = f" (no element {mi.group(1)!r})"
                    elif mi and names[mi.group(1)].type in ("STL", "MOS", "BJT") and not mi.group(2):
                        typ = names[mi.group(1)].type
                        pins = BASIC_PINS.get(typ, ("d", "g", "s"))
                        hint = " (use " + ", ".join(f"I({mi.group(1)}.{pin})" for pin in pins) + ")"
                    elif mi and mi.group(2):
                        hint = " (terminal currents: STL/MOS .d, .g, .s; BJT .c, .b, .e; diode I(name))"
                    else:
                        hint = " (use V(node), I(element), I(X.d|s|g), X.u, X.r or X.q_b)"
                raise ValueError(f"unknown probe {key!r}{hint}")
            if sp.key not in [x.key for x in specs]:
                specs.append(sp)
    n_sig = len(specs)
    # hidden trajectory columns of the first STL (V_DS, I_D) for the I–V overlay
    hidden = []
    if ns:
        e0 = stls[0]
        col = {n: i for i, n in enumerate(nodes)}
        c0 = len(nodes) + len(net.V)
        hidden = [SigSpec("__vds", _L("", ""), "V", "voltage", ("vds", col[e0.nodes[0]], col[e0.nodes[2]])),
                  SigSpec("__id", _L("", ""), "A", "current", ("col", c0 + 3 + 7 * e0.idx))]
    specs_run = specs + hidden
    feat = [j for j, sp in enumerate(specs) if not (sp.state and carrier)]
    if not feat:
        feat = list(range(n_sig))
    logfeat = [j for j, sp in enumerate(specs) if sp.key.startswith("I(") and sp.key.endswith((".d)", ".s)"))] if not carrier else []
    qcols = [sp.recipe[1] for sp in specs_run if sp.recipe[0] == "qb"]
    bp = net.breakpoints()

    pts0 = int(np.clip(BUDGET_RUN0 / max(n_sig, 1), 500, 4000))
    pts_other = int(np.clip(BUDGET_OTHER / max(n_sig, 1), 300, 1500))
    grid_n = int(np.clip(min(BUDGET_ENV / (5 * max(n_sig, 1)), ENV_MEMORY / (max(n_sig, 1) * max(n_runs, 1))), 100, 1000))

    # ---- solver config ----
    from .sim import SolverConfig, simulate
    ls_any = max([c.mode_code for c in cell_ls if c is not None], default=0) if stochastic else 0
    first_ls = next((c for c in cell_ls if c is not None), ls_global)
    cfg = SolverConfig(
        method=0 if method == "BE" else 1, stochastic=stochastic, carrier=carrier, max_steps=sol["max_steps"],
        dt_min=dt_min, dt_max=dt_max, reltol=reltol, tau_frac=sol["tau_frac"],
        max_events_per_step=sol["max_events_per_step"], noise_dt_min=sol["noise_dt_min_s"],
        gauss_threshold=sol["gauss_threshold"], gauss_tau_min=sol["gauss_tau_min_s"], gauss_tau_frac=sol["gauss_tau_frac"],
        i_threshold=det["i_threshold_A"], i_threshold_down=det["i_threshold_down_A"], ls_mode=ls_any,
        ls_idx=first_ls.index, ls_sigma=first_ls.sigma, ls_tau=first_ls.tau_s, lsE_sigma=first_ls.sigma_E_V,
        lsE_tau=first_ls.tau_E_s, ld_noise=st["ld_carrier_noise"] and carrier,
        dt_rec=0.0, dv_rec=1e9, dlni_rec=1e9, h_init=min(dt_max, max(10 * dt_min, 1e-7 * t_stop)),
        lte_v=TRTOL * reltol, lte_v_abs=TRTOL * VNTOL,
    )
    net_c = net.compile()
    # initial state 'zero': every capacitor is held at 0 V at t = 0 except those whose voltage is fixed by
    # voltage sources alone (both terminals in one voltage-source component: they start at that voltage)
    from .sim import G_HOLD
    vcomp = _DSU(nodes)
    for e in els:
        if e.type == "V":
            vcomp.union(e.nodes[0], e.nodes[1])
        elif e.type == "CMP":
            vcomp.union(e.nodes[2], "0")
    cap_hold = np.zeros(len(net.C))
    for e in els:
        if e.type == "C" and e.idx >= 0:
            cap_hold[e.idx] = 0.0 if vcomp.find(e.nodes[0]) == vcomp.find(e.nodes[1]) else G_HOLD

    # ---- comparators: output tracking at full resolution, firing windows (periods of the pulse source) ----
    cmps = [e for e in els if e.type == "CMP"]
    colmap = {n: i for i, n in enumerate(nodes)}
    cmp_specs = [dict(name=e.name, col=colmap[e.nodes[2]], mid=0.5 * (e.cmp["v_high"] + e.cmp["v_low"]),
                      up=e.cmp["v_high"] > e.cmp["v_low"]) for e in cmps]
    windows = _pulse_windows(els, t_stop) if cmps else None
    nwin = len(windows[1]) if windows else 0
    cmp_bits = np.full((len(cmps), n_runs, nwin), np.nan)
    cmp_rise = np.full((len(cmps), n_runs), np.nan)
    cmp_duty = np.full((len(cmps), n_runs), np.nan)

    # ---- runs ----
    stored_runs = []
    events: list[dict] = []
    n_up = np.zeros((n_runs, ns))
    n_dn = np.zeros((n_runs, ns))
    t_lu = np.full((n_runs, ns), np.nan)
    v_lu = np.full((n_runs, ns), np.nan)
    lat_end = np.full((n_runs, ns), np.nan)
    lu_all: list[list[np.ndarray]] = [[] for _ in range(ns)]       # per cell, per run: latch-up times
    vlu_all: list[list[np.ndarray]] = [[] for _ in range(ns)]      # ... V_DS at the latch-ups
    vld_all: list[list[np.ndarray]] = [[] for _ in range(ns)]      # ... V_DS at the latch-downs
    v_end = np.full((n_runs, n_sig), np.nan)
    completed = np.zeros(n_runs, bool)
    env = None
    grid = None
    op: dict[str, float] = {}
    traj = None
    solver_stats = dict(steps=0, rejected=0, newton_iters=0, runtime_s=0.0)
    reg = dict(t_total=0.0, t_drift_fast=0.0, t_gauss=0.0, t_lrs_drift=0.0, t_band_drift=0.0, t_neg_u=0.0, t_neg_r=0.0,
               steps_by_tier=[0] * 6, newton_by_tier=[0] * 6, rejected_by_tier=[0] * 6)
    min_u = min_r = np.inf
    hrs_x = 0
    initial_used, ndr_op = initial, []
    run_warn: list[str] = []
    for run in range(n_runs):
        def prog(f, _run=run):
            progress(min(0.97, 0.04 + 0.93 * (_run + f) / n_runs), f"run {_run + 1}/{n_runs}")

        ls0 = _draw_ls(cell_ls, st["seed"], run) if stochastic else None
        seed = (st["seed"] + 1_000_003 * run) % (2 ** 31 - 1)
        sink = _Sink(specs_run, feat, logfeat, t_save, bp, qcols)
        if cmps:
            sink.cmp = _CmpTrack(cmp_specs, windows)
        try:
            out = simulate(net_c, cfg, net_c["P"], t_stop, -1, seed, ls0, prog, window=wins if ns else None,
                           rec_sink=sink, initial=initial_run, cap_hold=cap_hold)
        except ValueError as exc:
            raise ValueError(f"run {run}: {exc}") from None
        solver_stats["steps"] += out.steps
        solver_stats["rejected"] += out.rejected
        solver_stats["newton_iters"] += out.newton_iters
        solver_stats["runtime_s"] += out.runtime_s
        reg["t_total"] += out.t_reached
        reg["t_drift_fast"] += out.t_unresolved
        reg["t_gauss"] += out.t_gauss
        reg["t_lrs_drift"] += out.t_lrs_drift
        reg["t_band_drift"] += out.t_band_drift
        reg["t_neg_u"] += out.t_neg_u
        reg["t_neg_r"] += out.t_neg_r
        for tier in range(6):
            reg["steps_by_tier"][tier] += out.diag[3 * tier]
            reg["newton_by_tier"][tier] += out.diag[3 * tier + 1]
            reg["rejected_by_tier"][tier] += out.diag[3 * tier + 2]
        min_u, min_r = min(min_u, out.min_u), min(min_r, out.min_r)
        hrs_x += out.hrs_crossings
        if run == 0:
            initial_used, ndr_op = out.initial, out.ndr_op
        for w in out.warnings:
            run_warn.append(f"run {run}: {w}")
        done = out.t_reached >= t_stop * (1 - 1e-9)
        completed[run] = done
        t_all, S_all = sink.finish()
        Sv = S_all[:, :n_sig]
        # op and per-cell initial latch state from the first (t = 0) row
        raw0 = sink.first_raw
        if run == 0 and raw0 is not None:
            vals = _convert(raw0[None, :], specs_all, {sp.recipe[1]: raw0[sp.recipe[1]] for sp in specs_all
                                                       if sp.recipe[0] == "qb"})[0]
            op = {sp.key: float(v) for sp, v in zip(specs_all, vals)}
        # events
        ev = out.events
        for k, e in enumerate(stls):
            ek = ev[ev[:, 1] == k] if len(ev) else np.zeros((0, 6))
            ups = ek[ek[:, 0] == 1]
            n_up[run, k] = len(ups)
            n_dn[run, k] = int(np.sum(ek[:, 0] == 2))
            if len(ups):
                t_lu[run, k] = ups[0, 2]
                v_lu[run, k] = ups[0, 3]
            lu_all[k].append(ups[:, 2].copy())
            vlu_all[k].append(ups[:, 3].copy())
            vld_all[k].append(ek[ek[:, 0] == 2, 3].copy())
            u0 = raw0[len(nodes) + len(net.V) + 7 * k] if raw0 is not None else 0.0
            state = 1.0 if u0 >= wins[k, 5] else 0.0
            if len(ek):
                state = 1.0 if ek[-1, 0] == 1 else 0.0
            lat_end[run, k] = state if done else np.nan
        if cmps:
            tr_ = sink.cmp
            obs = (windows[2] <= out.t_reached * (1 + 1e-12)) if nwin else np.zeros(0, bool)
            for j, e in enumerate(cmps):
                if nwin:
                    cmp_bits[j, run] = np.where(obs, tr_.fired[j].astype(float), np.nan)
                cmp_rise[j, run] = sum(1 for _, up_ in tr_.edges[j] if up_)
                cmp_duty[j, run] = tr_.t_high[j] / out.t_reached if out.t_reached > 0 else np.nan
                for te, up_ in tr_.edges[j]:
                    if len(events) >= MAX_EVENTS_OUT:
                        break
                    events.append(dict(run=run, kind="cmp_rise" if up_ else "cmp_fall", t=te, cell=e.name,
                                       value=float(e.cmp["v_high"] if up_ else e.cmp["v_low"])))
        for row in ev:
            if len(events) >= MAX_EVENTS_OUT:
                break
            events.append(dict(run=run, kind="latch_up" if row[0] == 1 else "latch_down", t=float(row[2]),
                               cell=stls[int(row[1])].name, v_d=float(row[3]), value=float(row[3]), i_d=float(row[5])))
        if done and len(t_all):
            v_end[run] = Sv[-1]
        # stored waveforms
        if run < (MAX_STORED_RUNS if stochastic else 1) and len(t_all):
            npts = pts0 if run == 0 else pts_other
            idx = _select(t_all, sink.features(S_all), npts, sink.forced(t_all))
            stored_runs.append(dict(run=run, t=_round_sig(t_all[idx], DIGITS_T), signals=[
                dict(key=sp.key, label=sp.label, unit=sp.unit, values=_round_sig(Sv[idx, j], DIGITS_SIG), axis=sp.axis)
                for j, sp in enumerate(specs)]))
            if run == 0 and hidden:
                traj = dict(vd=_round_sig(S_all[idx, n_sig], DIGITS_SIG), id=_round_sig(S_all[idx, n_sig + 1], DIGITS_SIG),
                            cell=stls[0].name)
        # envelopes
        if stochastic and n_runs > 1 and len(t_all):
            if grid is None:
                half = grid_n // 2
                g1 = np.linspace(t_save, t_stop, half)
                idx0 = _select(t_all, sink.features(S_all), max(grid_n - half, 2), sink.forced(t_all))
                grid = np.unique(np.r_[g1, t_all[idx0]])
                if len(grid) > grid_n:
                    grid = grid[np.round(np.linspace(0, len(grid) - 1, grid_n)).astype(np.int64)]
                env = np.full((n_runs, n_sig, len(grid)), np.nan, dtype=np.float32)
            for j in range(n_sig):
                env[run, j] = np.interp(grid, t_all, Sv[:, j])
            if not done:
                env[run, :, grid > out.t_reached] = np.nan
    progress(0.98, "statistics")
    warnings.extend(run_warn[:20])
    if len(run_warn) > 20:
        warnings.append(f"... {len(run_warn) - 20} more run warnings")
    n_trunc = int(np.sum(~completed))
    if n_trunc:
        warnings.append(f"{n_trunc} run(s) did not reach t_stop (step budget / convergence): their end-state and "
                        "t_stop values are excluded from the statistics")
    if ns and min_r < 0:
        warnings.append(f"a drain junction became forward biased (min r = {min_r:.3f} V): symmetric forward drain-diode "
                        "extension used (outside the calibrated model)")
    if ns and min_u < 0:
        warnings.append(f"a source junction became reverse biased (min u = {min_u * 1e3:.2f} mV): low-injection diode "
                        "extension used for u < 0")
    if pre_zero:
        warnings.append(f"{', '.join(pre_zero)}: current-biased above the HRS (the load line at t = 0 does not cross the HRS, "
                        "so the DC operating point would be an equilibrium on the negative-resistance branch or the LRS): "
                        "the transient starts from discharged capacitors, as if the sources were switched on at t = 0 "
                        "(tran.initial = 'auto'; 'op' uses the DC operating point)")
    elif ndr_op and ndr_op[0][0] < 0:
        warnings.append("no DC operating point was found from the empty body (a current-biased cell forced beyond its "
                        "HRS): the transient starts from discharged capacitors, as if the sources were switched on at "
                        "t = 0 (tran.initial = 'auto')")
    elif ndr_op:
        cells_txt = "; ".join(f"{stls[k].name} at V_DS = {v:.4g} V, I_D = {_fmt(i, 'A')}" for k, v, i in ndr_op)
        warnings.append(f"the DC operating point puts {cells_txt} on the negative-resistance branch (an equilibrium of "
                        "the current bias, unstable with the capacitance of a relaxation oscillator): the transient starts "
                        "from discharged capacitors instead, as if the sources were switched on at t = 0 "
                        "(tran.initial = 'auto'; 'op' keeps the operating point)")
    if hrs_x:
        warnings.append(f"I_D crossed detect.i_threshold_A = {det['i_threshold_A']:.3g} A {hrs_x} time(s) while the body "
                        "stayed on the HRS (channel/HRS conduction, not a latch-up; not counted)")
    if carrier and reg["t_drift_fast"] > 0:
        warnings.append(f"carrier noise not resolved (drift only) during {reg['t_drift_fast']:.3g} s of {reg['t_total']:.3g} s "
                        "simulated (states relaxing faster than gauss_tau_min / noise_dt_min)")

    # ---- summary ----
    summary: list[dict] = []
    distributions: list[dict] = []
    envelopes: list[dict] = []
    for k, e in enumerate(stls):
        nm = e.name
        ci_ = cell_info[k]
        if not stochastic:
            summary += [
                _item(f"{nm}.n_latch_up", f"{nm} 래치업 횟수", f"{nm} latch-ups", int(n_up[0, k]), "1"),
                _item(f"{nm}.n_latch_down", f"{nm} 래치다운 횟수", f"{nm} latch-downs", int(n_dn[0, k]), "1"),
                _item(f"{nm}.t_first_lu", f"{nm} 첫 래치업 시각", f"{nm} first latch-up time",
                      float(t_lu[0, k]) if np.isfinite(t_lu[0, k]) else None, "s"),
                _item(f"{nm}.vd_first_lu", f"{nm} 첫 래치업 시 V_DS", f"{nm} V_DS at the first latch-up",
                      float(v_lu[0, k]) if np.isfinite(v_lu[0, k]) else None, "V"),
                _item(f"{nm}.latched_end", f"{nm} t_stop에서 래치 상태 (1 = 래치)", f"{nm} latched at t_stop (1 = yes)",
                      None if not np.isfinite(lat_end[0, k]) else int(lat_end[0, k]), "1"),
                _item(f"{nm}.final_state", f"{nm} 최종 상태", f"{nm} final state",
                      None if not np.isfinite(lat_end[0, k]) else ("LRS" if lat_end[0, k] > 0.5 else "HRS")),
            ]
        else:
            le = lat_end[:, k]
            le = le[np.isfinite(le)]
            anyu = (n_up[:, k] > 0).astype(float)
            p_end = float(le.mean()) if len(le) else None
            sd_end = float(le.std(ddof=1)) if len(le) > 1 else None
            m_t, s_t = _stats(t_lu[:, k])
            m_v, s_v = _stats(v_lu[:, k])
            m_nu, s_nu = _stats(n_up[:, k])
            m_nd, s_nd = _stats(n_dn[:, k])
            summary += [
                _item(f"{nm}.p_any_lu", f"{nm} 래치업 확률 (1회 이상)", f"P({nm} latches up at least once)",
                      float(anyu.mean()), "1", float(anyu.std(ddof=1)) if n_runs > 1 else None),
                _item(f"{nm}.p_latched_end", f"{nm} t_stop에서 래치 확률", f"P({nm} latched at t_stop)", p_end, "1", sd_end),
                _item(f"{nm}.t_first_lu", f"{nm} 첫 래치업 시각 (평균 ± SD)", f"{nm} first latch-up time (mean ± SD)", m_t, "s", s_t),
                _item(f"{nm}.vd_first_lu", f"{nm} 첫 래치업 시 V_DS (평균 ± SD)", f"{nm} V_DS at the first latch-up (mean ± SD)",
                      m_v, "V", s_v),
                _item(f"{nm}.n_latch_up", f"{nm} 래치업 횟수 (평균 ± SD)", f"{nm} latch-ups per run (mean ± SD)", m_nu, "1", s_nu),
                _item(f"{nm}.n_latch_down", f"{nm} 래치다운 횟수 (평균 ± SD)", f"{nm} latch-downs per run (mean ± SD)", m_nd, "1", s_nd),
            ]
            distributions += [
                dict(key=f"{nm}.t_first_lu", label=_L(f"{nm} 첫 래치업 시각", f"{nm} first latch-up time"), unit="s",
                     values=t_lu[:, k]),
                dict(key=f"{nm}.vd_first_lu", label=_L(f"{nm} 첫 래치업 시 V_DS", f"{nm} V_DS at the first latch-up"),
                     unit="V", values=v_lu[:, k]),
                dict(key=f"{nm}.n_latch_up", label=_L(f"{nm} 실행당 래치업 횟수", f"{nm} latch-ups per run"), unit="1",
                     values=n_up[:, k]),
            ]
        # repeated latch-ups (relaxation oscillator, pulse trains): intervals between consecutive latch-ups of a run
        per_run = [d[np.isfinite(d) & (d > 0)] for d in (np.diff(x) for x in lu_all[k])]
        isi = np.concatenate(per_run) if per_run else np.zeros(0)
        if len(isi):
            m_i = float(isi.mean())
            sd_i = float(isi.std(ddof=1)) if len(isi) > 1 else None
            # jitter of one oscillator: CV of the intervals within a run, averaged over the runs; the run-to-run
            # spread of the mean period (e.g. frozen / slowly evolving local states) is reported separately
            cvs = [float(d.std(ddof=1) / d.mean()) for d in per_run if len(d) > 1]
            cv_in = float(np.mean(cvs)) if cvs else None
            means = np.array([d.mean() for d in per_run if len(d)])
            vlu_ = np.concatenate(vlu_all[k]) if vlu_all[k] else np.zeros(0)
            vld_ = np.concatenate(vld_all[k]) if vld_all[k] else np.zeros(0)
            m_u, s_u = _stats(vlu_)
            m_d, s_d = _stats(vld_)
            summary += [
                _item(f"{nm}.period", f"{nm} 래치업 간격 평균 (발진 주기)", f"{nm} mean interval between latch-ups (period)",
                      m_i, "s", sd_i),
                _item(f"{nm}.f_osc", f"{nm} 래치업 주파수 (1 / 평균 간격)", f"{nm} latch-up rate (1 / mean interval)", 1.0 / m_i, "Hz"),
                _item(f"{nm}.isi_cv", f"{nm} 한 실행 안에서 래치업 간격의 변동계수 (CV, 스파이크 타이밍 지터)",
                      f"{nm} coefficient of variation of the intervals within a run (spike-timing jitter)", cv_in, "1"),
                _item(f"{nm}.vd_lu_mean", f"{nm} 래치업 시 V_DS (모든 사건 평균 ± SD)", f"{nm} V_DS at latch-up (all events, mean ± SD)",
                      m_u, "V", s_u),
                _item(f"{nm}.vd_ld_mean", f"{nm} 래치다운 시 V_DS (모든 사건 평균 ± SD)", f"{nm} V_DS at latch-down (all events, mean ± SD)",
                      m_d, "V", s_d),
            ]
            if stochastic and len(means) > 1:
                summary.append(_item(f"{nm}.period_cv_runs", f"{nm} 실행 간 평균 주기의 변동계수 (국소 상태 등)",
                                     f"{nm} run-to-run CV of the mean period (local states etc.)",
                                     float(means.std(ddof=1) / means.mean()), "1"))
            if stochastic:
                distributions.append(dict(key=f"{nm}.isi", label=_L(f"{nm} 래치업 간격 (모든 실행)", f"{nm} intervals between latch-ups (all runs)"),
                                          unit="s", values=isi[:MAX_EVENTS_OUT]))
        fl = ci_["folds"]
        summary += [
            _item(f"{nm}.fold_V_LU", f"{nm} 준정적 폴드 V_LU (V_GS = {ci_['vgs_nom']:.3g} V)",
                  f"{nm} quasi-static fold V_LU (V_GS = {ci_['vgs_nom']:.3g} V)", float(fl[0]) if np.isfinite(fl[0]) else None, "V"),
            _item(f"{nm}.fold_V_LD", f"{nm} 준정적 폴드 V_LD", f"{nm} quasi-static fold V_LD",
                  float(fl[1]) if np.isfinite(fl[1]) else None, "V"),
        ]
    if stochastic:
        for j, sp in enumerate(specs):
            distributions.append(dict(key=f"end:{sp.key}", label=_L(f"t_stop에서의 {sp.key}", f"{sp.key} at t_stop"),
                                      unit=sp.unit, values=v_end[:, j]))
        if env is not None:
            with np.errstate(all="ignore"):
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore", RuntimeWarning)
                    E = env.astype(np.float64)
                    mean = np.nanmean(E, axis=0)
                    sd = np.nanstd(E, axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean)
                    p05, p95 = _nanquantiles(E, (0.05, 0.95))
            for j, sp in enumerate(specs):
                envelopes.append(dict(key=sp.key, t=_round_sig(grid, DIGITS_T), mean=_round_sig(mean[j], DIGITS_ENV),
                                      sd=_round_sig(sd[j], DIGITS_ENV), p05=_round_sig(p05[j], DIGITS_ENV),
                                      p95=_round_sig(p95[j], DIGITS_ENV)))
    comparators = []
    for j, e in enumerate(cmps):
        nm = e.name
        c = e.cmp
        m_r, s_r = _stats(cmp_rise[j])
        m_d, s_d = _stats(cmp_duty[j])
        block = dict(name=nm, nodes=dict(zip(("in", "inm", "out"), e.nodes)), v_ref=c["v_ref"], v_high=c["v_high"],
                     v_low=c["v_low"], hysteresis=c["hysteresis"], width=c["width"], window_source=None, t_windows=[],
                     bits=[], p_fire_window=[], p_fire_window_err=[], p_fire=None, lag1=None, n_bits=0, p_fire_run=[])
        if nwin:
            B = cmp_bits[j]
            fin = np.isfinite(B)
            n_bits = int(fin.sum())
            P = float(np.nanmean(B)) if n_bits else None
            with np.errstate(all="ignore"):
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore", RuntimeWarning)
                    pw_ = np.nanmean(B, axis=0)
                    nw_ = fin.sum(axis=0)
                    ew_ = np.sqrt(np.maximum(pw_ * (1 - pw_), 0.0) / np.maximum(nw_, 1))
                    pr_ = np.nanmean(B, axis=1)
            lag1 = RN._lag1_pairs([B[r] for r in range(n_runs)])
            cap_w = int(max(1, min(nwin, 100_000 // max(n_runs, 1))))
            block.update(window_source=windows[0], t_windows=[float(x) for x in windows[1]],
                         bits=[[None if not np.isfinite(v) else int(v) for v in B[r, :cap_w]] for r in range(n_runs)],
                         p_fire_window=[_fin(x) for x in pw_], p_fire_window_err=[_fin(x) for x in ew_],
                         p_fire=P, lag1=lag1, n_bits=n_bits, p_fire_run=[_fin(x) for x in pr_])
            summary += [
                _item(f"{nm}.p_fire", f"{nm} 발화 확률 (펄스당 출력 high, {windows[0]} 주기 기준)",
                      f"{nm} firing probability (output high within a period of {windows[0]})", P, "1",
                      float(np.nanstd(pr_, ddof=1)) if stochastic and np.isfinite(pr_).sum() > 1 else None),
                _item(f"{nm}.lag1", f"{nm} 발화 비트열 lag-1 자기상관", f"{nm} lag-1 autocorrelation of the firing bits", lag1),
                _item(f"{nm}.n_bits", f"{nm} 관측한 펄스 수 (실행 × 펄스)", f"{nm} observed pulses (runs × pulses)", n_bits),
            ]
            if stochastic:
                distributions.append(dict(key=f"{nm}.p_fire_run", label=_L(f"{nm} 실행별 발화 비율", f"{nm} firing fraction per run"),
                                          unit="1", values=pr_))
        summary += [
            _item(f"{nm}.n_rise", f"{nm} 출력 상승 에지 수 (실행당)", f"{nm} output rising edges per run", m_r, "1",
                  s_r if stochastic else None),
            _item(f"{nm}.duty", f"{nm} 출력 high 시간 비율", f"{nm} fraction of time with the output high", m_d, "1",
                  s_d if stochastic else None),
        ]
        comparators.append(block)
    if carrier:
        tt = max(reg["t_total"], 1e-300)
        summary.append(_item("t_noise_resolved_frac", "잡음 분해 시간 비율 (사건 수준+가우스)", "Fraction of time with resolved carrier noise",
                             float(np.clip(1.0 - (reg["t_drift_fast"] + reg["t_lrs_drift"] + reg["t_band_drift"]) / tt, 0, 1)), "1"))
    summary.append(_item("runs", "실행 수", "Runs", int(n_runs)))
    summary.append(_item("steps_per_run", "실행당 시간 스텝", "Time steps per run", float(solver_stats["steps"] / max(n_runs, 1))))
    if n_trunc:
        summary.append(_item("truncated_runs", "t_stop에 도달하지 못한 실행", "Runs that did not reach t_stop", n_trunc))
    events.sort(key=lambda d: (d["run"], d["t"]))

    # ---- echo ----
    elements_out = []
    for k_e, e in enumerate(els):
        d: dict[str, Any] = dict(type=e.type, name=e.name)
        if e.type == "STL":
            k = e.idx
            ci_ = cell_info[k]
            dev = e.device
            d["nodes"] = dict(d=e.nodes[0], g=e.nodes[1], s=e.nodes[2])
            d["device"] = dict(preset=dev.get("preset"), vg_device_V=float(dev["vg"]),
                               vbg_device_V=float(dev.get("vbg", 0.0)), geometry=dict(dev["geometry"]),
                               geometry_model=PR.geometry_model_metadata(dev), iph_pA=float(e.p[13]) * 1e12,
                               label=_stl_label(e))
            d["light_pA"] = None if e.light is None else e.light.resolved
            d["vgs_V"] = ci_["vgs_nom"]
            d["vgs_range_V"] = list(ci_["vgs_range"])
            # both come from the linear DC network before the run; with MOS/D/BJT on the cell's nets they are rough
            d["vgs_estimate"] = dict(method="linear DC network before the run (C open, STL drain-source off)",
                                     transistors_ignored=list(ci_["tx"]), reliable=not ci_["tx"])
            d["folds"] = dict(V_LU=_fin(ci_["folds"][0]), V_LD=_fin(ci_["folds"][1]))
            d["latch_window"] = ci_["latch"] if (ci_["latch"] or not ci_["tx"]) else None
            d["u_fold"] = dict(u_i=_fin(ci_["u_fold"][0]), u_j=_fin(ci_["u_fold"][1]))
            if stochastic:
                d["local_state"] = None if cell_ls[k] is None else vars(cell_ls[k])
                d["noise_band_V"] = dict(unlatched_from=_fin(ci_["band"][0]), latched_up_to=_fin(ci_["band"][3]))
            d["estimated_steps"] = ci_["est"]
            osc = ci_["osc"] if not ci_["tx"] else None
            if osc is not None:
                w0 = osc["walks"][0]
                d["oscillator"] = dict(predicted=bool(w0["oscillating"]), period_qs_s=w0["period"],
                                       latch_ups_expected=float(w0["n_lu"] * w0["scale"]), c_eff_F=osc["c_eff"],
                                       i_norton_A=list(osc["i_n"]), r_ext_ohm=osc["r_ext"])
            else:
                d["oscillator"] = None
        elif e.type == "CMP":
            d["nodes"] = dict(zip(("in", "inm", "out"), e.nodes))
            d.update(e.cmp)
            d["value_label"] = _cmp_label(e)
        elif e.type in BASIC_PINS:
            d["nodes"] = dict(zip(BASIC_PINS[e.type], e.nodes))
            d["model"] = e.model
            d["value_label"] = e.model.get("polarity", "Diode")
        else:
            d["nodes"] = list(e.nodes)
            if e.type in ("R", "C"):
                d["value"] = e.value
                d["value_label"] = _fmt(e.value, "Ω" if e.type == "R" else "F")
            else:
                d["wave"] = e.wave.resolved
                d["value_label"] = _wave_text(e.wave.resolved, "V" if e.type == "V" else "A")
        elements_out.append(d)

    runtime = time.perf_counter() - tic
    progress(1.0, "done")
    return dict(
        bench="custom", mode=mode, runs=stored_runs, events=events, summary=summary, distributions=distributions,
        envelopes=envelopes, sweeps=[], trajectory=traj, schematic=net.schematic(), nodes=list(nodes), comparators=comparators,
        elements=elements_out, op=op, probes=[sp.key for sp in specs],
        solver_stats=solver_stats,
        tran=dict(t_stop_s=t_stop, t_start_save_s=t_save, dt_max_s=dt_max, dt_min_s=dt_min, method=method, reltol=reltol,
                  initial=initial, initial_used=initial_used),
        solver=RN._jsonable(sol), detect=det,
        stochastic=RN._jsonable(dict(st, n_runs=n_runs, local_state=vars(ls_global))) if stochastic else None,
        feasibility=dict(estimated_steps_per_run=est, estimated_total_steps=total_est,
                         estimated_runtime_s=total_est * sec_step, total_runs=n_runs, linear_steps=lin,
                         cell_steps=est_parts),
        regimes=reg, runtime_s=runtime, warnings=warnings,
    )


def _nanquantiles(E: np.ndarray, qs) -> list[np.ndarray]:
    """Quantiles along axis 0 ignoring NaN (linear interpolation, like np.nanquantile), vectorised."""
    Es = np.sort(E, axis=0)                              # NaN sorted last
    n = np.sum(np.isfinite(E), axis=0)
    out = []
    for q in qs:
        pos = q * np.maximum(n - 1, 0)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
        a = np.take_along_axis(Es, lo[None], axis=0)[0]
        b = np.take_along_axis(Es, hi[None], axis=0)[0]
        r = a + (pos - lo) * (b - a)
        out.append(np.where(n > 0, r, np.nan))
    return out


def _round_sig(a, digits: int) -> np.ndarray:
    """Round to ``digits`` significant digits (page weight: plotted arrays only; NaN kept)."""
    a = np.asarray(a, float)
    with np.errstate(all="ignore"):
        e = np.floor(np.log10(np.abs(a)))
        f = 10.0 ** (digits - 1 - e)
        return np.where(np.isfinite(e) & np.isfinite(f) & (f > 0), np.round(a * f) / f, a)


def _fin(x):
    return float(x) if x is not None and np.isfinite(x) else None


def _cmp_label(e: El) -> str:
    c = e.cmp or {}
    inp = e.nodes[0] if e.nodes[1] == "0" else f"{e.nodes[0]} − {e.nodes[1]}"
    return (f"comparator: {c.get('v_high', 1):g} V if V({inp}) > {c.get('v_ref', 0):.4g} V else {c.get('v_low', 0):g} V"
            + (f", hysteresis {c['hysteresis']:.3g} V" if c.get("hysteresis") else ""))


def _stl_label(e: El) -> str:
    dev = e.device or {}
    pre = {"paper": "FDSOI reference calibration", "photo": "FDSOI illumination calibration"}.get(
        dev.get("preset", "paper"), "FDSOI custom")
    iph = float(e.p[13]) * 1e12 if e.p is not None else 0.0
    lt = "light wave" if e.light is not None else (f"I_PH {iph:.3g} pA" if iph else "dark")
    return f"STL ({pre}, {lt})"


def _draw_ls(cell_ls: list, seed: int, run: int) -> np.ndarray:
    """Per-run local-state draws per cell: columns (action point, emitter); the same random stream as
    ``stochastic.draw_local_states`` (which uses one configuration for all cells)."""
    n = len(cell_ls)
    out = np.zeros((n, 2))
    if all(c is None for c in cell_ls):
        return out
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 7349, int(run)])
    z0 = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    for k, c in enumerate(cell_ls):
        if c is not None:
            out[k, 0] = c.sigma * z0[k]
            out[k, 1] = c.sigma_E_V * z1[k]
    return out
