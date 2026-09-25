#!/usr/bin/env python3
"""Parameter sensitivity of the latch-up / latch-down voltages (V_LU, V_LD) with the real engine.

For every sidebar parameter (FieldDef keys of web/src/params/schema.ts) this script evaluates the deterministic
folds (MODEL.classify via server.compute.deterministic.classify_checked / folds_of) at the baseline and at an
"increase" and a "decrease" step, for both presets:

    ref   = reference calibration   (preset "paper": V_G = -2 V, dark, 0.4 V/s)
    photo = illumination calibration (preset "photo": V_G = -1.8 V, P = 0 mW, 1200 V/s, gamma = 0.2794,
            delta_phi_G0 = +74.4 mV); the light intensity is additionally evaluated around P = 2.55 mW.

Mechanism: the engine's current components (photo_mean.components) are evaluated at the *baseline* fold state
(u, r of the HRS fold row for V_LU, of the LRS fold row for V_LD) with the baseline and the increased parameter.
The change of every term of the net body-hole balance

    F = II(BJT seed) + II(channel) + II(photo) + local path + junction BTBT + GIDL + I_PH
        - emitter back-diffusion - bulk SRH - junction SRH

is expressed in % of the baseline hole generation at that state (positive = more net holes = the body charges
more easily), together with the change of the terminal voltage needed for the same internal state
(series contact/access resistance and hole drop, V_D = u + r + hole drop + (R_c + R_acc) I_D).

Sweep / stochastic parameters use run_hazard (carrier-noise first passage at the centre state) and run_sweep_mc
(small n, fixed seed).  Results -> web/src/content/params/sensitivity.json.

    python scripts/param_sensitivity.py            # everything (~5-10 min on one core, uses the shared disk cache)
    python scripts/param_sensitivity.py --det-only # deterministic folds only (~1 min)
    python scripts/param_sensitivity.py --only ls_sigma   # recompute one stochastic block, merge into the JSON
    python scripts/param_sensitivity.py --geometry [--geometry-out FILE]   # Geometry (L, W, Tsi, EOT, Tbox, Nbody) and
                                                   # V_BG steps only: table on stdout, blocks as JSON in FILE (~1 min);
                                                   # sensitivity.json is not touched
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import params as P  # noqa: E402
from server.compute import deterministic as D  # noqa: E402
from server.engine_bridge import MODEL, m  # noqa: E402
from server.geometry_model import pack_p, packed_field  # noqa: E402

OUT = ROOT / "web" / "src" / "content" / "params" / "sensitivity.json"
GRID = 601                       # UI default (device.numerics.grid)
NOEFF_MV = 0.05                  # |ΔV| below this is reported as "no effect"
V_CAP = 8.0                      # server sweep cap (V): folds beyond it are not a usable latch window
PRESET_OF = {"ref": "paper", "photo": "photo"}
VD_MAX = {"ref": 4.0, "photo": 5.0}
RESP = P.RESPONSIVITY_PA_PER_MW   # 0.75 pA/mW
IPH_2P55 = 2.55 * RESP            # 1.9125 pA

# ------------------------------------------------------------------------------------------------------------
# parameter table: key -> how to step it
#   path   : location in the device object
#   kind   : "mul" (x2 / x0.5, slope per +10 %) | "add" (+/- step, slope per `per`) | "abs" (absolute values)
#   unit / scale : UI display unit and display = stored * scale (as schema.ts)
#   ctx    : device overrides needed for the parameter to act (e.g. aloc = 1 for the local-path levers)
#   lo / hi: UI bounds (stored units); a step outside them is skipped (null)
# ------------------------------------------------------------------------------------------------------------
PARAMS: list[dict] = [
    dict(key="vg", path=("vg",), kind="add", step=0.1, per=0.1, unit="V", scale=1, lo=-6, hi=1, slope_unit="mV per +0.1 V",
         extra=[("abs", v, f"{v:+.1f} V") for v in (-3.0, -2.5, -2.2, -1.8, -1.5, -1.3, -1.1, -0.9)],
         edge_probe=[-5.0, 0.5]),
    dict(key="dphiG0", path=("state", "delta_phi_G0_V"), kind="add", step=0.010, per=0.010, unit="mV", scale=1e3,
         lo=-0.5, hi=0.5, slope_unit="mV per +10 mV"),
    dict(key="dphiE0", path=("state", "delta_phi_E0_V"), kind="add", step=1e-4, per=1e-4, unit="mV", scale=1e3,
         lo=-0.01, hi=0.01, slope_unit="mV per +0.1 mV"),
    dict(key="beta", path=("calib", "beta"), kind="mul", unit="", scale=1, lo=0.01, hi=100,
         extra=[("rel", f, f"×{f:g}") for f in (0.3, 0.4, 0.7, 1.5, 4.0)], edge_probe=[0.05, 20.0]),
    dict(key="tau_bulk", path=("calib", "tau_bulk_s"), kind="mul", unit="µs", scale=1e6, lo=1e-10, hi=1e-2),
    dict(key="tau_junction", path=("calib", "tau_junction_s"), kind="mul", unit="ns", scale=1e9, lo=1e-12, hi=1e-4,
         extra=[("rel", f, f"×{f:g}") for f in (0.25, 4.0, 10.0)], edge_probe=[0.02, 50.0]),
    dict(key="r_contact", path=("calib", "r_contact_ohm"), kind="mul", unit="Ω", scale=1, lo=0, hi=1e6,
         extra=[("abs", 1e4, "10 kΩ"), ("abs", 1e5, "100 kΩ"), ("abs", 1e6, "1 MΩ")]),
    dict(key="l_gidl", path=("calib", "l_gidl_nm"), kind="mul", unit="nm", scale=1, lo=0.1, hi=500,
         extra=[("rel", f, f"×{f:g}") for f in (0.7, 0.8, 0.9, 1.1, 1.2, 1.5)]),
    dict(key="t_access", path=("calib", "t_access_nm"), kind="mul", unit="nm", scale=1, lo=0.1, hi=100),
    dict(key="na_access", path=("calib", "na_access_cm3"), kind="mul", unit="10¹⁷ cm⁻³", scale=1e-17, lo=1e14, hi=1e21),
    dict(key="l_access", path=("calib", "l_access_nm"), kind="mul", unit="nm", scale=1, lo=1, hi=1000),
    dict(key="tau_ratio", path=("calib", "tau_ratio"), kind="mul", unit="", scale=1, lo=1e-3, hi=1e5),
    dict(key="phi_gidl0", path=("calib", "phi_gidl0_V"), kind="add", step=0.010, per=0.010, unit="mV", scale=1e3,
         lo=-0.5, hi=0.5, slope_unit="mV per +10 mV"),
    dict(key="phi_emitter0", path=("calib", "phi_emitter0_V"), kind="add", step=1e-4, per=1e-4, unit="mV", scale=1e3,
         lo=-0.05, hi=0.05, slope_unit="mV per +0.1 mV"),
    dict(key="ch_ii", path=("calib", "channel_ii_scale"), kind="mul", unit="", scale=1, lo=0, hi=100, alt_vg=-1.1,
         extra=[("abs", 0.0, "0 (off)"), ("abs", 10.0, "10"), ("abs", 100.0, "100")]),
    dict(key="dibl", path=("ext", "dibl"), kind="add", step=0.01, per=0.01, unit="V/V", scale=1, lo=0, hi=1, alt_vg=-1.1,
         slope_unit="mV per +0.01", extra=[("abs", 0.03, "+0.03"), ("abs", 0.05, "+0.05")]),
    dict(key="gamma", path=("ext", "gamma"), kind="add", step=0.05, per=0.01, unit="", scale=1, lo=0, hi=2, alt_vg=-1.1,
         slope_unit="mV per +0.01"),
    dict(key="kappa", path=("ext", "kappa"), kind="add", step=0.05, per=0.01, unit="1/V", scale=1, lo=-10, hi=10, alt_vg=-1.1,
         slope_unit="mV per +0.01 /V"),
    dict(key="seed_ip", path=("ext", "seed_ip_pA"), kind="abs", up=1.33, down=None, per=0.1, unit="pA", scale=1, alt_vg=-1.1,
         lo=0, hi=1e4, slope_unit="mV per +0.1 pA", extra=[("abs", 0.133, "0.133 pA"), ("abs", 2.66, "2.66 pA"),
                                                         ("abs", 13.3, "13.3 pA")]),
    dict(key="seed_S", path=("ext", "seed_S"), kind="add", step=0.1, per=0.1, unit="V/dec", scale=1, lo=0.01, hi=10, alt_vg=-1.1,
         slope_unit="mV per +0.1 V/dec", ctx={("ext", "seed_ip_pA"): 1.33, ("ext", "seed_S"): 0.8},
         ctx_note="evaluated with the 'high-V_D seed' option (I_p = 1.33 pA, S = 0.8 V/dec); with the preset I_p = 0 "
                  "the slope S has no effect", default_ctx_check=True),
    dict(key="dj", path=("ext", "dj"), kind="add", step=0.05, per=0.01, unit="V", scale=1, lo=-1, hi=1,
         slope_unit="mV per +10 mV"),
    dict(key="dm", path=("ext", "dm"), kind="add", step=0.3, per=0.1, unit="ln", scale=1, lo=-5, hi=5, edge_probe=[-3.0, 3.0],
         slope_unit="mV per +0.1"),
    dict(key="aloc", path=("ext", "aloc"), kind="abs", up=1.0, down=None, per=0.1, unit="", scale=1, lo=0, hi=1e3,
         slope_unit="mV per +0.1", extra=[("abs", 0.5, "0.5"), ("abs", 2.0, "2"), ("abs", 10.0, "10")],
         ctx_note="preset value 0 = local avalanche path off; 'up' switches it on with a_loc = 1 (verify_hloc.py)"),
    dict(key="isat", path=("ext", "isat_pA"), kind="mul", unit="pA", scale=1, lo=0, hi=1e6,
         ctx={("ext", "aloc"): 1.0}, ctx_note="evaluated with a_loc = 1 (the saturation acts only when the local "
                                               "path is on; with the preset a_loc = 0 it has no effect)",
         default_ctx_check=True),
    dict(key="dloc", path=("ext", "dloc"), kind="add", step=0.5, per=0.1, unit="ln", scale=1, lo=-10, hi=10,
         slope_unit="mV per +0.1", ctx={("ext", "aloc"): 1.0},
         ctx_note="evaluated with a_loc = 1 (no effect with the preset a_loc = 0)", default_ctx_check=True),
    dict(key="loc_carriers", path=("ext", "loc_carriers"), kind="abs", up=1, down=None, per=None, unit="", scale=1,
         lo=0, hi=2, ctx={("ext", "aloc"): 1.0}, slope_unit=None,
         ctx_note="0 (edge incl. channel) vs 1 (bulk), evaluated with a_loc = 1 (no effect with a_loc = 0)",
         default_ctx_check=True),
    dict(key="kappaF", path=("ext", "kappaF"), kind="add", step=0.5, per=0.1, unit="1/V", scale=1, lo=-10, hi=10,
         slope_unit="mV per +0.1 /V", ctx={("ext", "aloc"): 1.0},
         ctx_note="evaluated with a_loc = 1 (no effect with the preset a_loc = 0)", default_ctx_check=True),
    dict(key="light", path=("light", "iph_pA"), kind="abs", up=IPH_2P55, down=None, per=0.1, unit="pA", scale=1,
         lo=0, hi=1e6, slope_unit="mV per +0.1 pA",
         extra=[("abs", 1.15 * RESP, "1.15 mW"), ("abs", 3.51 * RESP, "3.51 mW"), ("abs", 5.0, "5 pA"),
                ("abs", 10.0, "10 pA"), ("abs", 20.0, "20 pA")], edge_probe=[200.0]),
]
GRID_KEYS = ("grid",)

# Geometry extension (server/geometry_model.py) and back-gate bias: deterministic steps only (the stochastic model is
# calibrated at the reference geometry).  Steps stay inside params.GEOMETRY_LIMITS and the model domain.
_GL = P.GEOMETRY_LIMITS
GEOMETRY_PARAMS: list[dict] = [
    dict(key="Lg_nm", path=("geometry", "Lg_nm"), kind="add", step=100.0, per=10.0, unit="nm", scale=1,
         lo=_GL["Lg_nm"][0], hi=_GL["Lg_nm"][1], slope_unit="mV per +10 nm",
         extra=[("abs", v, f"{v:g} nm") for v in (200.0, 300.0, 700.0, 1000.0)]),
    dict(key="W_nm", path=("geometry", "W_nm"), kind="mul", unit="nm", scale=1, lo=_GL["W_nm"][0], hi=_GL["W_nm"][1]),
    dict(key="Tsi_nm", path=("geometry", "Tsi_nm"), kind="add", step=10.0, per=1.0, unit="nm", scale=1,
         lo=_GL["Tsi_nm"][0], hi=_GL["Tsi_nm"][1], slope_unit="mV per +1 nm",
         extra=[("abs", v, f"{v:g} nm") for v in (5.0, 10.0, 20.0, 70.0)]),
    dict(key="EOT_nm", path=("geometry", "EOT_nm"), kind="add", step=1.0, per=0.1, unit="nm", scale=1,
         lo=_GL["EOT_nm"][0], hi=_GL["EOT_nm"][1], slope_unit="mV per +0.1 nm"),
    dict(key="Tbox_nm", path=("geometry", "Tbox_nm"), kind="mul", unit="nm", scale=1,
         lo=_GL["Tbox_nm"][0], hi=_GL["Tbox_nm"][1]),
    dict(key="Nbody_cm3", path=("geometry", "Nbody_cm3"), kind="mul", unit="10¹⁷ cm⁻³", scale=1e-17,
         lo=_GL["Nbody_cm3"][0], hi=_GL["Nbody_cm3"][1],
         extra=[("abs", v, f"{v:.3g} cm⁻³") for v in (5e16, 1e17, 3e17, 5e17, 1e18)]),
    dict(key="vbg", path=("vbg",), kind="add", step=0.5, per=0.1, unit="V", scale=1, lo=-10.0, hi=10.0, alt_vg=-1.1,
         slope_unit="mV per +0.1 V", extra=[("abs", v, f"{v:+g} V") for v in (-2.0, 2.0, 5.0)],
         ctx_note="V_BG only shifts the front-channel overdrive (EOT/(Tbox + Tsi/3) x V_BG); with the channel off "
                  "(V_G = -2 V) the folds hardly move, see the V_G = -1.1 V block"),
]

TERM_LABEL = {
    "ii_seed": "impact ionisation of the BJT (seed) electrons",
    "ii_ch": "channel-electron impact ionisation",
    "ii_ph": "multiplied photo-electrons",
    "iloc": "local avalanche path",
    "bbj": "junction BTBT",
    "gidl": "GIDL",
    "iph": "photogeneration I_PH",
    "diff": "emitter back-diffusion loss",
    "bulk": "bulk SRH loss",
    "junction": "junction (depletion) SRH loss",
}
LOSS_TERMS = ("diff", "bulk", "junction")


# ------------------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------------------
def rnd(x, nd=5):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, nd)


def sig(x, n=4):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x) or x == 0:
        return 0.0 if x == 0 else None
    return float(f"{x:.{n}g}")


def base_device(which: str) -> dict:
    d = P.resolve_device({"preset": PRESET_OF[which]})
    # the light block is always addressed as I_PH (pA); photo preset P = 0 mW -> I_PH = 0
    d["light"] = dict(d["light"], mode="iph", iph_pA=0.0)
    return d


def get_path(d: dict, path) -> float:
    x = d
    for k in path:
        x = x[k]
    return float(x)


def set_path(d: dict, path, value) -> dict:
    d = copy.deepcopy(d)
    x = d
    for k in path[:-1]:
        x = x[k]
    x[path[-1]] = value
    return d


_FOLD_CACHE: dict = {}


def evaluate(dev: dict, grid: int = GRID, vd_max: float = 4.0) -> dict:
    p = np.asarray(P.build_p(dev), float)
    key = (tuple(np.round(p, 15)), grid)
    if key not in _FOLD_CACHE:
        z, gap = D.classify_checked(p, grid)
        _FOLD_CACHE[key] = (z, gap, p)
    z, gap, p = _FOLD_CACHE[key]
    p = pack_p(p)                 # geometry vectors carry their own field tables (the reference vector is unchanged)
    f = D.folds_of(z)
    # a physical hysteresis window: two folds with V_LU > V_LD, V_LU within the server's 8 V sweep cap.  classify
    # can return spurious fold pairs far above 8 V (e.g. beta <= 0.28x: V_LD > V_LU at 9-13 V); those are rejected.
    spurious = z is not None and not (f["V_LU"] - f["V_LD"] > 1e-3 and f["V_LU"] <= V_CAP)
    win = z is not None and not spurious
    if spurious:
        z = None
        f = D.folds_of(None)
    return dict(V_LU=f["V_LU"], V_LD=f["V_LD"], window_V=f["window_V"], window_exists=win, locus_gap=bool(gap),
                spurious=spurious, latch_up_exists=bool(win and f["V_LU"] <= vd_max),
                latch_down_exists=bool(win and f["V_LD"] is not None and f["V_LD"] > 0),
                _z=z, _p=p)


def pub(ev: dict | None, value_disp=None) -> dict | None:
    if ev is None:
        return None
    out = dict(V_LU=rnd(ev["V_LU"]), V_LD=rnd(ev["V_LD"]), window_V=rnd(ev["window_V"]),
               window_exists=ev["window_exists"], latch_up_exists=ev["latch_up_exists"],
               latch_down_exists=ev["latch_down_exists"])
    if ev.get("locus_gap"):
        out["locus_gap"] = True
    if ev.get("spurious"):
        out["spurious_fold_rejected"] = True
    if value_disp is not None:
        out["value"] = sig(value_disp, 5)
    return out


def dmv(a: dict | None, b: dict | None, k: str):
    if a is None or b is None or a.get(k) is None or b.get(k) is None:
        return None
    return round(1e3 * (b[k] - a[k]), 3)


# ------------------------------------------------------------------------------------------------------------
# mechanism: current-term changes at the baseline fold state
# ------------------------------------------------------------------------------------------------------------
def terms(u: float, r: float, p: np.ndarray) -> dict | None:
    c = m.components(float(u), float(r), p, MODEL.na, MODEL.vbi, MODEL.rg, MODEL.fg, MODEL.table)
    if not np.all(np.isfinite(c[:19])):
        return None
    vd, drain, _net, seed, _em, bulk, diff, junction, bbj, gidl = c[:10]
    racc, ch, hole_drop, iph = c[12], c[16], c[17], c[18]
    gain = packed_field(float(r) + p[19], p, 0) if len(p) > 33 else m.f_interp(float(r) + p[19], MODEL.rg, MODEL.fg[0])
    M = 1.0 + (gain - 1.0) * math.exp(p[20])
    ii_total = drain - seed - bbj - gidl - ch - iph
    ii_seed = (M - 1.0) * seed
    ii_ch = max(M - 1.0, 0.0) * ch * p[12]
    ii_ph = max(M - 1.0, 0.0) * iph
    iloc = ii_total - ii_seed - ii_ch - ii_ph
    return dict(vd=vd, drain=drain, ii_seed=ii_seed, ii_ch=ii_ch, ii_ph=ii_ph, iloc=iloc, bbj=bbj, gidl=gidl,
                iph=iph, diff=diff, bulk=bulk, junction=junction, racc=racc, rc=p[3], hole_drop=hole_drop, ch=ch,
                seed=seed, M=M)


def mechanism(ev0: dict, ev1: dict) -> dict | None:
    """Term changes at the baseline HRS (LU) and LRS (LD) fold rows, and at-fold component ratios."""
    z0 = ev0["_z"]
    if z0 is None:
        return None
    b, i, j, _ = z0
    out = {}
    for tag, ind in (("LU", i), ("LD", j)):
        u, r = b[ind, 17], b[ind, 18]
        t0, t1 = terms(u, r, ev0["_p"]), terms(u, r, ev1["_p"])
        if t0 is None or t1 is None:
            out[tag] = None
            continue
        gen0 = sum(t0[k] for k in TERM_LABEL if k not in LOSS_TERMS)
        change = {k: (100.0 * (t1[k] - t0[k]) / gen0 if gen0 > 0 else 0.0) for k in TERM_LABEL}
        dF = sum(-v if k in LOSS_TERMS else v for k, v in change.items())
        rec = dict(dF_pct=round(dF, 3),
                   terms_pct={k: round(v, 3) for k, v in change.items() if abs(v) >= 0.01},
                   dVD_series_mV=round(1e3 * (t1["vd"] - t0["vd"]), 3))
        out[tag] = rec
    # component ratios along the actual folds (branches columns), increased / baseline
    z1 = ev1["_z"]
    if z1 is not None:
        b1, i1, j1, _ = z1
        cols = dict(I_D=1, gidl=9, btbt_junction=8, channel=16, loss_diffusion=6, loss_bulk_srh=5,
                    loss_junction_srh=7, r_access=12, seed=3)
        rat = {}
        for tag, r0, r1 in (("LU", b[i], b1[i1]), ("LD", b[j], b1[j1])):
            iph0, iph1 = ev0["_p"][13], ev1["_p"][13]
            ii0 = r0[1] - r0[3] - r0[8] - r0[9] - r0[16] - iph0
            ii1 = r1[1] - r1[3] - r1[8] - r1[9] - r1[16] - iph1
            q = {k: sig(r1[c] / r0[c], 4) if r0[c] > 0 else None for k, c in cols.items()}
            q["ii_total"] = sig(ii1 / ii0, 4) if ii0 > 0 else None
            q["u_mV"] = round(1e3 * (r1[17] - r0[17]), 2)
            rat[tag] = q
        out["at_fold_ratio"] = rat
    return out


def mech_note(mech: dict | None, dlu: float | None, dld: float | None) -> str:
    if mech is None:
        return "baseline has no latch window"
    parts = []
    for tag, dv in (("LU", dlu), ("LD", dld)):
        rec = mech.get(tag)
        if rec is None:
            continue
        name = "V_LU" if tag == "LU" else "V_LD"
        if dv is not None and abs(dv) < NOEFF_MV and abs(rec["dVD_series_mV"]) < NOEFF_MV:
            parts.append(f"{name}: no effect")
            continue
        items = sorted(rec["terms_pct"].items(), key=lambda kv: -abs(kv[1]))
        lead = abs(items[0][1]) if items else 0.0
        top = [(k, v) for k, v in items if abs(v) >= max(0.25 * lead, 0.2)][:3]
        txt = ", ".join(f"{TERM_LABEL[k]} {v:+.1f} %" for k, v in top)
        s = f"{name} ({'n/a' if dv is None else f'{dv:+.1f} mV'}): "
        if txt:
            s += f"net holes at the fold state {rec['dF_pct']:+.1f} % of generation [{txt}]"
        if abs(rec["dVD_series_mV"]) >= 0.5:
            s += ("; " if txt else "") + f"series resistance/hole drop shifts V_D by {rec['dVD_series_mV']:+.1f} mV"
        if not txt and abs(rec["dVD_series_mV"]) < 0.5:
            s += "negligible change of the current terms"
        parts.append(s)
    return " | ".join(parts)


# ------------------------------------------------------------------------------------------------------------
# deterministic sensitivity of one parameter in one preset
# ------------------------------------------------------------------------------------------------------------
def with_ctx(dev: dict, spec: dict) -> dict:
    for path, v in (spec.get("ctx") or {}).items():
        dev = set_path(dev, path, v)
    return dev


def steps_for(spec: dict, x0: float) -> tuple[float | None, float | None, str | None, str | None]:
    lo, hi = spec.get("lo", -np.inf), spec.get("hi", np.inf)
    ok = lambda v: v is not None and lo - 1e-15 <= v <= hi + 1e-15
    if spec["kind"] == "mul":
        if x0 == 0:
            raise ValueError(f"{spec['key']}: multiplicative parameter with zero baseline")
        up, dn, su, sd = 2 * x0, 0.5 * x0, "×2", "×0.5"
    elif spec["kind"] == "add":
        st = spec["step"]
        up, dn = x0 + st, x0 - st
        disp = st * spec["scale"]
        u = spec["unit"]
        su, sd = f"+{disp:g} {u}".strip(), f"−{disp:g} {u}".strip()
    else:
        up, dn = spec.get("up"), spec.get("down")
        su = None if up is None else f"{up * spec['scale']:.4g} {spec['unit']}".strip()
        sd = None if dn is None else f"{dn * spec['scale']:.4g} {spec['unit']}".strip()
    if not ok(up):
        up, su = None, None
    if not ok(dn):
        dn, sd = None, None
    return up, dn, su, sd


def local_slope(spec: dict, dev0: dict, x0: float, vd_max: float) -> dict | None:
    path, kind = spec["path"], spec["kind"]
    lo, hi = spec.get("lo", -np.inf), spec.get("hi", np.inf)
    if kind == "mul":
        f = 1.05
        a, b = evaluate(set_path(dev0, path, x0 * f), vd_max=vd_max), evaluate(set_path(dev0, path, x0 / f), vd_max=vd_max)
        den = 2 * math.log(f) / math.log(1.1)
        per = "mV per +10 %"
    else:
        if spec.get("per") is None:
            return None
        h = (spec["step"] / 5) if kind == "add" else spec["per"] / 2
        xa, xb = x0 + h, x0 - h
        if xb < lo - 1e-15:           # one-sided at a bound (e.g. dibl = 0, gamma = 0)
            xb = x0
        if xa > hi + 1e-15:
            xa = x0
        a, b = evaluate(set_path(dev0, path, xa), vd_max=vd_max), evaluate(set_path(dev0, path, xb), vd_max=vd_max)
        den = (xa - xb) / spec["per"]
        per = spec.get("slope_unit") or "mV per unit"
    if not (a["window_exists"] and b["window_exists"]) or den == 0:
        return dict(per=per, VLU_mV=None, VLD_mV=None)
    return dict(per=per, VLU_mV=round(1e3 * (a["V_LU"] - b["V_LU"]) / den, 3),
                VLD_mV=round(1e3 * (a["V_LD"] - b["V_LD"]) / den, 3))


def window_edges(spec, dev0, x0, e0, eu, ed, up, dn, vd_max) -> dict | None:
    """Where the latch window (two-fold branch) disappears, by bisection between the baseline and a failing step."""
    if not e0["window_exists"]:
        return None
    path, sc = spec["path"], spec["scale"]
    out = {}
    cands = [(ev, xv) for ev, xv in ((eu, up), (ed, dn)) if ev is not None]
    for pv in spec.get("edge_probe", []):
        pv = x0 * pv if spec["kind"] == "mul" else pv
        cands.append((evaluate(set_path(dev0, path, pv), vd_max=vd_max), pv))
    for ev, xv in cands:
        side = "up" if xv > x0 else "down"
        if ev["window_exists"] or side in out:
            continue
        a, b = x0, xv          # a: window, b: none
        logscale = spec["kind"] == "mul" and a > 0 and b > 0
        for _ in range(24):
            mid = math.sqrt(a * b) if logscale else 0.5 * (a + b)
            if evaluate(set_path(dev0, path, mid), vd_max=vd_max)["window_exists"]:
                a = mid
            else:
                b = mid
            if abs(b - a) <= 2e-3 * max(abs(x0), 1e-12) or (not logscale and abs(b - a) < 1e-4):
                break
        last = evaluate(set_path(dev0, path, a), vd_max=vd_max)
        ev = evaluate(set_path(dev0, path, b), vd_max=vd_max)      # just beyond the edge: why the window is gone
        if ev.get("locus_gap"):
            why = "locus gap (V_LU beyond the traceable range)"
        elif ev.get("spurious"):
            why = "window closes (V_LU → V_LD, only spurious folds above 8 V remain)"
        else:
            why = "window closes (no hysteresis: V_LU → V_LD)"
        out[side] = dict(value=sig(a * sc, 4), factor=sig(a / x0, 4) if x0 else None, reason=why,
                         V_LU_at_edge=rnd(last["V_LU"]), V_LD_at_edge=rnd(last["V_LD"]))
    return out or None


def fold_currents(ev: dict) -> dict | None:
    """Absolute currents (A) of the engine terms at the HRS (LU) and LRS (LD) fold rows."""
    z = ev["_z"]
    if z is None:
        return None
    b, i, j, _ = z
    out = {}
    for tag, ind in (("LU", i), ("LD", j)):
        t = terms(b[ind, 17], b[ind, 18], ev["_p"])
        if t is None:
            continue
        out[tag] = {k: sig(t[k], 3) for k in ("drain", "seed", "ch", "gidl", "bbj", "iph", "ii_seed", "ii_ch", "ii_ph",
                                               "iloc", "diff", "bulk", "junction", "racc", "M")}
        out[tag].update(V_D=rnd(b[ind, 0]), u=rnd(b[ind, 17]), r=rnd(b[ind, 18]))
    return out


def hrs_fixed_vd(evs: dict, vds=(2.0, 3.0, 3.5)) -> dict:
    """HRS-branch currents at fixed V_D (log-interpolated), for each evaluation in evs."""
    out = {}
    cols = dict(I_D=1, gidl=9, channel=16, btbt_junction=8, loss_junction_srh=7)
    for name, ev in evs.items():
        if ev is None or ev["_z"] is None:
            out[name] = None
            continue
        b, i, _, _ = ev["_z"]
        hrs = D._monotone(b[: i + 1])
        rows = {}
        for v in vds:
            if not hrs[0, 0] <= v <= hrs[-1, 0]:
                continue
            rows[f"{v:g}"] = {k: sig(math.exp(np.interp(v, hrs[:, 0], np.log(np.maximum(hrs[:, c], 1e-300)))), 3)
                              for k, c in cols.items()}
        out[name] = rows
    return out


def direction_word(d: float | None) -> str:
    if d is None:
        return "n/a"
    if abs(d) < NOEFF_MV:
        return "≈0"
    return "↑" if d > 0 else "↓"


def summarize(block: dict, step_up: str | None, step_down: str | None) -> str:
    s = []
    if block["up"] is not None:
        s.append(f"{step_up}: V_LU {fmt_mv(block['dVLU_up_mV'], block['up'])}, V_LD {fmt_mv(block['dVLD_up_mV'], block['up'])}")
    if block["down"] is not None:
        s.append(f"{step_down}: V_LU {fmt_mv(block['dVLU_down_mV'], block['down'])}, V_LD {fmt_mv(block['dVLD_down_mV'], block['down'])}")
    return "; ".join(s)


def fmt_mv(d, ev) -> str:
    if ev is not None and not ev["window_exists"]:
        return "no latch window"
    if d is None:
        return "n/a"
    if abs(d) < NOEFF_MV:
        return "no change"
    return f"{d:+.1f} mV"


def device_block(spec: dict, which: str, dev_base: dict | None = None) -> dict:
    vd_max = VD_MAX[which]
    dev0 = with_ctx(dev_base if dev_base is not None else base_device(which), spec)
    path = spec["path"]
    x0 = get_path(dev0, path)
    up, dn, su, sd = steps_for(spec, x0)
    e0 = evaluate(dev0, vd_max=vd_max)
    eu = evaluate(set_path(dev0, path, up), vd_max=vd_max) if up is not None else None
    ed = evaluate(set_path(dev0, path, dn), vd_max=vd_max) if dn is not None else None
    sc = spec["scale"]
    blk = dict(base=pub(e0, x0 * sc), up=pub(eu, None if up is None else up * sc),
               down=pub(ed, None if dn is None else dn * sc))
    blk["step_up"], blk["step_down"] = su, sd
    blk["dVLU_up_mV"], blk["dVLD_up_mV"] = dmv(e0, eu, "V_LU"), dmv(e0, eu, "V_LD")
    blk["dVLU_down_mV"], blk["dVLD_down_mV"] = dmv(e0, ed, "V_LU"), dmv(e0, ed, "V_LD")
    blk["latch_up_exists"] = dict(base=e0["latch_up_exists"], up=None if eu is None else eu["latch_up_exists"],
                                  down=None if ed is None else ed["latch_up_exists"])
    blk["latch_down_exists"] = dict(base=e0["latch_down_exists"], up=None if eu is None else eu["latch_down_exists"],
                                    down=None if ed is None else ed["latch_down_exists"])
    blk["slope"] = local_slope(spec, dev0, x0, vd_max) if e0["window_exists"] else None
    nm = {}
    for k in ("VLU", "VLD"):
        a, b = blk[f"d{k}_up_mV"], blk[f"d{k}_down_mV"]
        nm[k] = bool(a is not None and b is not None and abs(a) >= NOEFF_MV and abs(b) >= NOEFF_MV
                     and np.sign(a) == np.sign(b))
    blk["nonmonotonic"] = nm
    mech = mechanism(e0, eu) if (eu is not None and e0["window_exists"]) else None
    if mech is None and ed is not None and e0["window_exists"] and eu is None:
        mech = mechanism(e0, ed)
        if mech is not None:
            mech["evaluated_for"] = "down step"
    blk["mechanism"] = mech
    blk["mechanism_note"] = mech_note(mech, blk["dVLU_up_mV"] if eu is not None else blk["dVLU_down_mV"],
                                      blk["dVLD_up_mV"] if eu is not None else blk["dVLD_down_mV"])
    extra = []
    for kind, val, label in spec.get("extra", []):
        if kind == "rel":
            val = x0 * val
        ee = evaluate(set_path(dev0, path, val), vd_max=vd_max)
        extra.append(dict(label=label, value=sig(val * sc, 5), **pub(ee), dVLU_mV=dmv(e0, ee, "V_LU"),
                          dVLD_mV=dmv(e0, ee, "V_LD")))
    if extra:
        blk["extra"] = extra
    edges = window_edges(spec, dev0, x0, e0, eu, ed, up, dn, vd_max)
    if edges:
        blk["window_edges"] = edges
    if spec.get("default_ctx_check") and spec.get("ctx"):
        # same step without the context: should be exactly zero
        d0 = dev_base if dev_base is not None else base_device(which)
        x00 = get_path(d0, path)
        upc, _, _, _ = steps_for(spec, x00) if spec["kind"] != "mul" else (2 * x00, None, None, None)
        if upc is not None:
            a, b = evaluate(d0, vd_max=vd_max), evaluate(set_path(d0, path, upc), vd_max=vd_max)
            blk["without_context"] = dict(dVLU_up_mV=dmv(a, b, "V_LU"), dVLD_up_mV=dmv(a, b, "V_LD"))
    blk["summary"] = summarize(blk, su, sd)
    return blk


# ------------------------------------------------------------------------------------------------------------
# physics notes (short, English; the UI writes its own plain-language text)
# ------------------------------------------------------------------------------------------------------------
NOTES = {
    "vg": "V_G enters the GIDL field E_G = (u + r − V_G − 1.42 V + φ_GIDL)/l_GIDL and the channel overdrive. A more "
          "positive V_G lowers the gate-drain field (less GIDL) and raises the channel current; at V_G = −2 / −1.8 V "
          "the channel is deep in subthreshold, so GIDL dominates.",
    "dphiG0": "Adds to the GIDL potential offset p[9] (same entry as phi_gidl0): +δφ_G raises the GIDL field like a more "
              "negative V_G.",
    "dphiE0": "Adds to the emitter offset p[10] (same entry as phi_emitter0): the emitter back-diffusion loss scales "
              "as exp(−φ_E/V_T); acts mainly on V_LD (high-injection LRS fold).",
    "beta": "Diffusion ratio: the emitter back-diffusion (hole) loss is ∝ 1/β. Larger β = smaller loss = holes stay in "
            "the body.",
    "tau_bulk": "SRH lifetime in the neutral body: longer τ = less bulk recombination loss (and a larger BJT gain).",
    "tau_junction": "SRH lifetime in the source depletion region: longer τ = less junction recombination loss "
                    "(the dominant loss at the HRS fold).",
    "r_contact": "Series contact resistance: V_D = u + r + (R_c + R_acc)·I_D. The preset is ~1 Ω, so ×2 / ×0.5 do "
                 "nothing; it matters only at ≳ 10 kΩ, mainly through the large LRS current (V_LD).",
    "l_gidl": "Effective length of the GIDL field (E_G ∝ 1/l_GIDL). Longer = weaker field = exponentially less GIDL.",
    "t_access": "Access-region thickness: sets R_acc (∝ 1/t) and the access charge; folds move only through the "
                "series voltage R_acc·I_D.",
    "na_access": "Access-region doping: sets R_acc (∝ 1/conductivity); folds move only through the series voltage.",
    "l_access": "Access-region length: R_acc ∝ L; folds move only through the series voltage.",
    "tau_ratio": "τ_p/τ_n of the body SRH kernel: changes the self-consistent BJT seed and the bulk loss.",
    "phi_gidl0": "Calibrated GIDL potential offset p[9] (the mean drain-edge local state); identical to dphiG0.",
    "phi_emitter0": "Calibrated emitter potential offset p[10] (the mean source-edge local state); identical to dphiE0.",
    "ch_ii": "Scale of impact ionisation by channel electrons. The channel current at the HRS fold is ~1e-18 A at "
             "V_G = −2 / −1.8 V, so this scale does nothing unless the channel is turned on (V_G, γ, DIBL, seed).",
    "dibl": "DIBL raises the channel overdrive by η·(u + r); with the channel in deep subthreshold small η does "
            "nothing, large η turns on the channel II path.",
    "gamma": "Body-to-channel coupling: overdrive + γ·u. It turns on the channel (and its II) as the body charges.",
    "kappa": "Slope degradation of the subthreshold ideality n(1 + κ(u + r)); only matters when the channel carries "
             "current.",
    "seed_ip": "Extra high-V_D channel seed I_p·10^((V_G + 1.8)/S) added to the channel electrons (multiplied by II).",
    "seed_S": "V_G slope of the seed current; at V_G = −1.8 V the factor 10^0 = 1, so S has no effect there.",
    "dj": "Junction potential offset (hypothesis lever): shifts the reverse bias seen by the multiplication and "
          "junction BTBT tables (M − 1 grows steeply with r).",
    "dm": "Log-scale of (M − 1) (hypothesis lever): multiplies all impact ionisation by e^Δ_M.",
    "aloc": "Local avalanche (microplasma-type) path, saturating at I_sat; off (0) in both presets.",
    "isat": "Saturation current of the local path (only with a_loc > 0).",
    "dloc": "Log fluctuation of the local path: strength × e^δ_loc (only with a_loc > 0).",
    "loc_carriers": "Which carriers the local path multiplies: 0 = edge (GIDL + BTBT + channel + photo), 1 = bulk "
                    "(BJT-injected + photo + BTBT).",
    "kappaF": "Field dependence of the local path: × exp(κ_F (V_GD − 5.6 V)) (only with a_loc > 0).",
    "light": "Photogeneration I_PH = R·P (R = 0.75 pA/mW) is a constant hole supply into the body (and its "
             "photo-electrons are multiplied at the drain).",
}


# ------------------------------------------------------------------------------------------------------------
# stochastic / sweep parameters
# ------------------------------------------------------------------------------------------------------------
N_MC = 200
MC_SEED = 20260925
STOCH_FIELD_NOTE = ("MC blocks: mean (V) and SD (mV) of V_LU / V_LD, n = 200 cycles, fixed seed, preset sweep (0 → 4 V "
                    "ref, 0 → 5 V photo) unless a block says otherwise (ls_sigma ref: 0 → 6 V so that no cycle is "
                    "censored); hazard blocks: carrier-noise first passage at the centre state (run_hazard)")


def mc(which: str, stoch: dict | None = None, sweep: dict | None = None, device: dict | None = None) -> dict:
    from server.compute import stochastic as ST
    st = dict(n_cycles=N_MC, seed=MC_SEED, n_traces=0)
    st.update(stoch or {})
    pl = dict(device=device or {"preset": PRESET_OF[which]}, stochastic=st)
    if sweep:
        pl["sweep"] = sweep
    t = time.perf_counter()
    o = ST.run_sweep_mc(pl)
    s = o["stats"]
    lu, ld = s["LU"], s["LD"]
    vlu = np.asarray(o["V_LU"], float)
    return dict(engine=o["engine"], mean_VLU=rnd(lu["mean"]), sd_VLU_mV=rnd(1e3 * lu["sd"], 2) if lu["sd"] else None,
                lag1_VLU=rnd(lu["lag1"], 3), mean_VLD=rnd(ld["mean"]),
                sd_VLD_mV=rnd(1e3 * ld["sd"], 2) if ld["sd"] else None, lag1_VLD=rnd(ld["lag1"], 3),
                censored=int(lu["censored"]), n=int(len(vlu)), runtime_s=round(time.perf_counter() - t, 1),
                warnings=[w for w in o["warnings"] if "10 mV read-out" not in w][:3], _vlu=vlu)


def strip(d: dict | None) -> dict | None:
    return None if d is None else {k: v for k, v in d.items() if not k.startswith("_")}


def mc_block(base: dict, up: dict | None, down: dict | None) -> dict:
    def dd(a, b, k, s=1.0):
        if a is None or b is None or a.get(k) is None or b.get(k) is None:
            return None
        return round(s * (b[k] - a[k]), 2)
    return dict(base=strip(base), up=strip(up), down=strip(down),
                dVLU_up_mV=dd(base, up, "mean_VLU", 1e3), dVLD_up_mV=dd(base, up, "mean_VLD", 1e3),
                dVLU_down_mV=dd(base, down, "mean_VLU", 1e3), dVLD_down_mV=dd(base, down, "mean_VLD", 1e3),
                dsdVLU_up_mV=dd(base, up, "sd_VLU_mV"), dsdVLD_up_mV=dd(base, up, "sd_VLD_mV"),
                dsdVLU_down_mV=dd(base, down, "sd_VLU_mV"), dsdVLD_down_mV=dd(base, down, "sd_VLD_mV"))


def hazard_stats(which: str, rate: float) -> dict:
    from server.compute import stochastic as ST
    from server.compute import stoch_core as C
    o = ST.run_hazard({"device": {"preset": PRESET_OF[which]}, "sweep": {"rate_V_per_s": rate}})
    dev = P.resolve_device({"preset": PRESET_OF[which]})
    rec = C.ld_hazard_curve(np.asarray(P.build_p(dev), float), grid=GRID)
    q = C.quantiles(rec, rate)
    return dict(rate_V_per_s=rate, fold_VLU=rnd(o["fold_V"]), mean_VLU=rnd(o["stats"]["mean"]),
                sd_VLU_mV=rnd(1e3 * o["stats"]["sd"], 2), below_fold_VLU_mV=rnd(1e3 * (o["stats"]["mean"] - o["fold_V"]), 2),
                fold_atom=rnd(o["fold_atom"], 4), fold_VLD=rnd(rec["fold_V"]),
                mean_VLD=rnd(float(np.mean(q))) if q is not None else None,
                sd_VLD_mV=rnd(1e3 * float(np.std(q)), 2) if q is not None else None,
                above_fold_VLD_mV=rnd(1e3 * (float(np.mean(q)) - rec["fold_V"]), 2) if q is not None else None)


LS_SIGMA_VD_MAX = 6.0   # ls_sigma (ref): sweep ceiling high enough that no cycle is censored at σ ×2


def ls_sigma_block(log) -> dict:
    """State SD σ_φ (frozen mode), ×2 / ×0.5.

    Reference preset: the main block raises the sweep ceiling to 6 V (general engine) so that every cycle latches
    up. At 0 → 4 V the cycles with the highest V_LU do not latch at ×2 and drop out of the statistics (censoring),
    which biases the mean V_LU low and σ small; that preset view (automatic engine = calibrated lookup, which also
    scales σ_φE with σ_φ) is kept in 'auto_engine_4V'. Illumination preset: 0 → 5 V, nothing is censored.
    """
    log("ls_sigma")
    blk = dict(key="ls_sigma", step_up="×2", step_down="×0.5", units="mV")
    note = ("The frozen drain-edge state shifts the GIDL offset of each cycle; V_LU follows it (dV_LU/dφ_G ≈ −0.8 V/V), "
            "so σ(V_LU) grows ~linearly with σ_φ on top of the carrier-noise floor ('no_state'); the mean hardly moves "
            "(V_LU is slightly convex in φ_G, so a wider state distribution shifts the uncensored mean by a few mV, "
            "within the statistical noise of n = 200).")
    s0 = P.SIGMA_PHI_G_V
    ls = lambda s, extra=None: dict(extra or {}, local_state={"mode": "frozen", "sigma": s})
    sw = {"vd_max_V": LS_SIGMA_VD_MAX}
    gen = {"engine": "general"}
    b, u, d = (mc("ref", ls(f * s0, gen), sweep=sw) for f in (1.0, 2.0, 0.5))
    z = mc("ref", dict(gen, local_state={"mode": "none"}), sweep=sw)
    blk["ref"] = mc_block(b, u, d)
    blk["ref"]["base"]["sigma_mV"] = rnd(1e3 * s0, 2)
    blk["ref"]["vd_max_V"] = LS_SIGMA_VD_MAX
    blk["ref"]["no_state"] = strip(z)
    blk["ref"]["mechanism_note"] = (
        note + f" Reference preset with the sweep ceiling raised to {LS_SIGMA_VD_MAX:g} V (general engine) so that no "
        "cycle is censored; the general engine keeps σ_φE fixed, so V_LD is unchanged. 'auto_engine_4V' = the preset "
        "view (automatic engine = calibrated lookup, 0 → 4 V): it scales σ_φE by the same factor (V_LD spread changes "
        "too) and, at ×2, the cycles with V_LU > 4 V do not latch and are left out (censored), which biases the mean "
        "V_LU low and σ(V_LU) small.")
    ab, au, ad = (mc("ref", ls(f * s0)) for f in (1.0, 2.0, 0.5))
    az = mc("ref", {"local_state": {"mode": "none"}})
    auto = mc_block(ab, au, ad)
    auto["no_state"] = strip(az)
    auto["vd_max_V"] = VD_MAX["ref"]
    blk["ref"]["auto_engine_4V"] = auto

    s0 = P.PHOTO_SIGMA_PHI_V
    b, u, d = (mc("photo", ls(f * s0)) for f in (1.0, 2.0, 0.5))
    z = mc("photo", {"local_state": {"mode": "none"}})
    blk["photo"] = mc_block(b, u, d)
    blk["photo"]["base"]["sigma_mV"] = rnd(1e3 * s0, 2)
    blk["photo"]["no_state"] = strip(z)
    blk["photo"]["mechanism_note"] = note + " Illumination preset (0 → 5 V): no cycle is censored."
    blk["notes"] = ("State SD mainly sets the cycle-to-cycle spread σ(V_LU) (×2 → σ ≈ ×2); it does not change the centre "
                    "folds or, without censoring, the mean V_LU.")
    return blk


# Stochastic blocks that can be recomputed on their own (--only KEY[,KEY]) and merged into the existing JSON.
STOCH_BLOCKS = {"ls_sigma": ls_sigma_block}


def run_stochastic(base_det: dict, log) -> dict:
    out: dict = {}
    # ---- rate: carrier-noise first passage at the centre state -----------------------------------------------
    log("rate (run_hazard)")
    blk = dict(key="rate", step_up="×10", step_down="×0.1", units="V/s")
    for which, r0 in (("ref", 0.4), ("photo", 1200.0)):
        b, u, d = hazard_stats(which, r0), hazard_stats(which, 10 * r0), hazard_stats(which, 0.1 * r0)
        blk[which] = dict(base=b, up=u, down=d,
                          dVLU_up_mV=round(1e3 * (u["mean_VLU"] - b["mean_VLU"]), 2),
                          dVLD_up_mV=round(1e3 * (u["mean_VLD"] - b["mean_VLD"]), 2) if u["mean_VLD"] and b["mean_VLD"] else None,
                          dVLU_down_mV=round(1e3 * (d["mean_VLU"] - b["mean_VLU"]), 2),
                          dVLD_down_mV=round(1e3 * (d["mean_VLD"] - b["mean_VLD"]), 2) if d["mean_VLD"] and b["mean_VLD"] else None,
                          dsdVLU_up_mV=round(u["sd_VLU_mV"] - b["sd_VLU_mV"], 2),
                          dsdVLU_down_mV=round(d["sd_VLU_mV"] - b["sd_VLU_mV"], 2),
                          latch_up_exists=dict(base=True, up=True, down=True),
                          latch_down_exists=dict(base=True, up=True, down=True),
                          mechanism_note="Carrier (first-passage) noise only, centre local state: the deterministic "
                                         "folds do not depend on the ramp rate. A slower ramp gives the body more "
                                         "time to escape by a rare fluctuation before the fold → switching earlier "
                                         "(V_LU lower, V_LD higher) and a slightly wider spread; a faster ramp pushes "
                                         "the switch toward the fold.")
    blk["notes"] = ("Ramp rate changes only the stochastic switching voltages (hazard integral ∫h dV/rate), not the "
                    "deterministic folds. 'fold_atom' = fraction of cycles that reach the fold without an earlier "
                    "escape.")
    out["rate"] = blk

    # ---- baseline MC (preset defaults) --------------------------------------------------------------------
    log("baseline MC")
    base_mc = {w: mc(w) for w in ("ref", "photo")}

    # ---- vd_max: threshold behaviour ----------------------------------------------------------------------
    log("vd_max")
    blk = dict(key="vd_max", step_up=None, step_down=None, units="V")
    for which in ("ref", "photo"):
        v = base_mc[which]["_vlu"]
        fold = base_det[which]["V_LU"]
        probes = sorted({round(x, 3) for x in (fold - 0.3, np.nanquantile(v, .05), np.nanmedian(v),
                                                np.nanquantile(v, .95), fold, VD_MAX[which])})
        rows = [dict(vd_max_V=rnd(x, 3), latched_fraction=round(float(np.mean(v <= x)), 3)) for x in probes]
        blk[which] = dict(base=dict(vd_max_V=VD_MAX[which], V_LU_fold=rnd(fold), V_LD_fold=rnd(base_det[which]["V_LD"]),
                                    mc_mean_VLU=base_mc[which]["mean_VLU"], mc_sd_VLU_mV=base_mc[which]["sd_VLU_mV"]),
                          threshold=rows,
                          latch_up_exists=dict(rule="only if V_D,max ≥ V_LU (per cycle)"),
                          latch_down_exists=dict(rule="only in the cycles that latched up"),
                          dVLU_up_mV=0.0, dVLD_up_mV=0.0, dVLU_down_mV=None, dVLD_down_mV=None,
                          mechanism_note="V_D,max does not move V_LU or V_LD; it only decides whether the sweep "
                                         "reaches V_LU. Below the fold, only the cycles whose (random) V_LU is lower "
                                         "than V_D,max latch (latched_fraction from the preset MC, n = 200).")
    blk["notes"] = ("Threshold parameter: no latch-up if V_D,max < V_LU; above V_LU increasing it changes nothing "
                    "(the LRS→HRS return happens at V_LD on the way down).")
    out["vd_max"] = blk

    # ---- dv: numerics -------------------------------------------------------------------------------------
    log("dv")
    blk = dict(key="dv", step_up="5 mV", step_down="1 mV", units="mV")
    for which in ("ref", "photo"):
        b = base_mc[which]
        u = mc(which, sweep={"dv_V": 0.005})
        d = mc(which, sweep={"dv_V": 0.001})
        blk[which] = mc_block(b, u, d)
        blk[which]["mechanism_note"] = ("numerics only: the voltage step sets the MC time step Δt = ΔV/rate and the "
                                        "resolution of the recorded switching voltage (the hazard integral always runs "
                                        "on ≤ 2 mV sub-steps); the reference-calibration lookup engine reads out on a "
                                        "10 mV grid like the measurement.")
    blk["notes"] = "Numerical resolution; changes of the mean/σ are within the statistical noise of n = 200 cycles."
    out["dv"] = blk

    # ---- ls_sigma: frozen mode (see ls_sigma_block) --------------------------------------------------------
    out["ls_sigma"] = ls_sigma_block(log)

    # ---- ls_tau: evolving mode ----------------------------------------------------------------------------
    log("ls_tau")
    blk = dict(key="ls_tau", step_up="×10", step_down="×0.1", units="s")
    for which in ("ref", "photo"):
        t0 = P.TAU_G_UP_S
        ev = lambda t: {"local_state": {"mode": "evolving", "tau_s": t}}
        b, u, d = mc(which, ev(t0)), mc(which, ev(10 * t0)), mc(which, ev(0.1 * t0))
        blk[which] = mc_block(b, u, d)
        blk[which]["base"]["tau_s"] = t0
        period = 2 * VD_MAX[which] / (0.4 if which == "ref" else 1200.0)
        blk[which]["cycle_period_s"] = rnd(period, 6)
        ex = []
        for f in (0.5, 2.0):
            r = mc(which, ev(f * t0))
            ex.append(dict(label=f"×{f:g}", tau_s=f * t0, **strip(r)))
        if which == "photo":
            for t in (0.01, 0.001):
                r = mc(which, ev(t))
                ex.append(dict(label=f"{t:g} s", tau_s=t, **strip(r)))
        blk[which]["extra"] = ex
        blk[which]["mechanism_note"] = (
            "The OU correlation time sets how similar consecutive cycles are: lag-1 correlation ≈ exp(−T_cycle/τ). "
            "τ ≫ record length → the state hardly moves during the record (high lag-1, and the SD of one record "
            "can be smaller than the stationary σ); τ ≪ T_cycle → independent cycles (lag-1 ≈ 0)." +
            (" Reference preset: the acquisition trend (on) also adds cycle-to-cycle correlation." if which == "ref" else ""))
    blk["notes"] = "Correlation time changes the ordering/correlation of cycles, not the fold positions."
    out["ls_tau"] = blk

    # ---- ls_sigmaE: emitter state SD -> sigma(V_LD) --------------------------------------------------------
    log("ls_sigmaE")
    blk = dict(key="ls_sigmaE", units="mV")
    se0 = P.SIGMA_PHI_E_V
    lsE = lambda s, extra=None: {"engine": "general", "local_state": dict({"mode": "frozen", "sigma_E_V": s}, **(extra or {}))}
    b = mc("ref", lsE(se0)); u = mc("ref", lsE(2 * se0)); d = mc("ref", lsE(0.5 * se0)); z = mc("ref", lsE(0.0))
    auto_u = mc("ref", {"local_state": {"mode": "frozen", "sigma_E_V": 2 * se0}})
    blk["ref"] = mc_block(b, u, d)
    blk["ref"]["base"]["sigma_E_mV"] = rnd(1e3 * se0, 4)
    blk["ref"]["zero"] = strip(z)
    blk["ref"]["auto_engine_up"] = strip(auto_u)
    blk["ref"]["mechanism_note"] = (
        "General engine (frozen): the source-edge state shifts the emitter offset φ_E of each cycle; V_LD moves with "
        "dV_LD/dφ_E ≈ −41 V/V, so σ(V_LD) grows with σ_φE while V_LU hardly changes. NOTE: with the preset's automatic "
        "engine (calibrated lookup) σ_φE is tied to σ_φ and a separate σ_φE is ignored ('auto_engine_up').")
    b = mc("photo", {"local_state": {"sigma_E_V": 0.0}})
    u = mc("photo", {"local_state": {"sigma_E_V": se0}})
    u2 = mc("photo", {"local_state": {"sigma_E_V": 2 * se0}})
    blk["photo"] = mc_block(b, u, None)
    blk["photo"]["base"]["sigma_E_mV"] = 0.0
    blk["photo"]["extra"] = [dict(label=f"{1e3 * 2 * se0:.3f} mV", **strip(u2))]
    blk["photo"]["mechanism_note"] = ("Illumination preset: σ_φE = 0 by default; switching it on (reference value "
                                      f"{1e3 * se0:.3f} mV) widens V_LD only.")
    blk["step_up"] = "×2 (ref) / 0 → 0.437 mV (photo)"
    blk["step_down"] = "×0.5 (ref)"
    blk["notes"] = "Emitter-state SD sets the spread of V_LD (σ(V_LD)); V_LU is unaffected."
    out["ls_sigmaE"] = blk

    # ---- carrier noise toggles ---------------------------------------------------------------------------
    log("carrier_noise / ld_carrier_noise")
    blk = dict(key="carrier_noise", step_up=None, step_down="off", units="")
    for which in ("ref", "photo"):
        eng = {"engine": "general"} if which == "ref" else {}   # same engine on/off (auto would switch engines)
        off = mc(which, dict(eng, carrier_noise=False, local_state={"mode": "frozen"}))
        on = mc(which, dict(eng, local_state={"mode": "frozen"})) if which == "ref" else base_mc[which]
        blk[which] = mc_block(on, None, off)
        blk[which]["mechanism_note"] = ("Off: every cycle switches exactly at its (state-dependent) fold → mean V_LU "
                                        "moves up to the fold, σ loses the carrier-noise part.")
    blk["notes"] = ("Carrier (first-passage) noise lowers the mean V_LU below the fold and adds spread (ref: general "
                    "engine, frozen states, for both on and off).")
    out["carrier_noise"] = blk
    blk = dict(key="ld_carrier_noise", step_up=None, step_down="off", units="")
    for which in ("ref", "photo"):
        eng = {"engine": "general"} if which == "ref" else {}
        on = mc(which, dict(eng, local_state={"mode": "frozen"}))
        off = mc(which, dict(eng, ld_carrier_noise=False, local_state={"mode": "frozen"}))
        blk[which] = mc_block(on, None, off)
        blk[which]["mechanism_note"] = ("Off: the down sweep switches exactly at the V_LD fold; on: an earlier escape "
                                        "(higher V_LD), larger at slow ramps.")
    blk["notes"] = "Latch-down first passage raises the mean V_LD above the fold (ref: general engine)."
    out["ld_carrier_noise"] = blk

    # ---- statistics / numerics ---------------------------------------------------------------------------
    log("n_cycles / seed / grid / fold_nodes / hazard_nodes")
    blk = dict(key="n_cycles", step_up="×4 (800)", step_down="×0.25 (50)", units="")
    for which in ("ref", "photo"):
        b = base_mc[which]
        u = mc(which, {"n_cycles": 4 * N_MC}); d = mc(which, {"n_cycles": N_MC // 4})
        blk[which] = mc_block(b, u, d)
        sd = (b["sd_VLU_mV"] or 0.0)
        blk[which]["standard_error_mean_VLU_mV"] = {str(n): round(sd / math.sqrt(n), 2) for n in (50, 100, 200, 400, 2000)}
        blk[which]["mechanism_note"] = "statistical precision only: SE(mean) = σ/√N, SE(σ) ≈ σ/√(2N)"
    blk["notes"] = "No physical shift of V_LU/V_LD; more cycles = smaller statistical error."
    out["n_cycles"] = blk

    blk = dict(key="seed", step_up="other seeds", step_down=None, units="")
    for which in ("ref", "photo"):
        runs = [base_mc[which]] + [mc(which, {"seed": MC_SEED + k}) for k in (1, 2, 3)]
        mu = [r["mean_VLU"] for r in runs]; sdv = [r["sd_VLU_mV"] for r in runs]
        mld = [r["mean_VLD"] for r in runs]
        blk[which] = dict(base=strip(runs[0]), seeds=[strip(r) for r in runs[1:]],
                          spread_mean_VLU_mV=round(1e3 * (max(mu) - min(mu)), 2),
                          spread_sd_VLU_mV=round(max(sdv) - min(sdv), 2),
                          spread_mean_VLD_mV=round(1e3 * (max(mld) - min(mld)), 2),
                          mechanism_note="random stream only: differences are sampling noise (n = 200)")
    blk["notes"] = "Same seed = same random stream; changing it changes the samples, not the physics."
    out["seed"] = blk

    blk = dict(key="grid", step_up="2001", step_down="201", units="")
    for which in ("ref", "photo"):
        dev = base_device(which)
        e0 = evaluate(dev, GRID, VD_MAX[which])
        eu = evaluate(dev, 2001, VD_MAX[which])
        ed = evaluate(dev, 201, VD_MAX[which])
        blk[which] = dict(base=pub(e0, GRID), up=pub(eu, 2001), down=pub(ed, 201),
                          dVLU_up_mV=dmv(e0, eu, "V_LU"), dVLD_up_mV=dmv(e0, eu, "V_LD"),
                          dVLU_down_mV=dmv(e0, ed, "V_LU"), dVLD_down_mV=dmv(e0, ed, "V_LD"),
                          mechanism_note="numerics only: u-grid of the steady-state locus; fold refined by a parabola")
    blk["notes"] = "Numerical resolution of the fold search; sub-mV effect."
    out["grid"] = blk

    blk = dict(key="fold_nodes", step_up="49", step_down="13", units="")
    for which in ("ref", "photo"):
        eng = {"engine": "general"} if which == "ref" else {}
        fr = {"local_state": {"mode": "frozen"}}
        b = mc(which, dict(eng, **fr)) if which == "ref" else base_mc[which]
        u = mc(which, dict(eng, fold_nodes=49, **fr)); d = mc(which, dict(eng, fold_nodes=13, **fr))
        blk[which] = mc_block(b, u, d)
        blk[which]["mechanism_note"] = ("numerics only: nodes of the fold-vs-state table (interpolated). The "
                                        "reference preset's automatic engine (calibrated lookup) does not use it; "
                                        "evaluated with the general engine." if which == "ref" else
                                        "numerics only: nodes of the fold-vs-state table (interpolated)")
    blk["notes"] = "Numerical interpolation of the fold table; no physical effect."
    out["fold_nodes"] = blk

    blk = dict(key="hazard_nodes", step_up="7", step_down="3", units="")
    for which in ("ref", "photo"):
        eng = {"engine": "general"} if which == "ref" else {}
        fr = {"local_state": {"mode": "frozen"}}
        b = mc(which, dict(eng, **fr)) if which == "ref" else base_mc[which]
        u = mc(which, dict(eng, hazard_nodes=7, **fr)); d = mc(which, dict(eng, hazard_nodes=3, **fr))
        blk[which] = mc_block(b, u, d)
        blk[which]["mechanism_note"] = ("numerics only: Gauss-Hermite nodes of the state-dependent first-passage "
                                        "hazard" + (" (general engine; the automatic calibrated lookup does not use "
                                                    "it)" if which == "ref" else ""))
    blk["notes"] = "Numerical quadrature of the hazard over the local state; no physical effect."
    out["hazard_nodes"] = blk
    out["_baseline_mc"] = {w: strip(base_mc[w]) for w in base_mc}
    return out


# ------------------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-only", action="store_true", help="deterministic folds only")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", default="", help="recompute only these stochastic blocks (comma-separated, e.g. "
                    "ls_sigma) and merge them into the existing --out JSON")
    ap.add_argument("--geometry", action="store_true", help="geometry and V_BG steps only (sensitivity.json untouched)")
    ap.add_argument("--geometry-out", default="", help="with --geometry: write the blocks to this JSON file")
    args = ap.parse_args()
    t0 = time.perf_counter()
    log = lambda s: print(f"[{time.perf_counter() - t0:6.1f} s] {s}", flush=True)

    if args.geometry:
        return geometry_main(args, log)

    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        bad = [k for k in keys if k not in STOCH_BLOCKS]
        if bad:
            ap.error(f"--only supports {sorted(STOCH_BLOCKS)}; got {bad}")
        out = Path(args.out)
        result = json.loads(out.read_text())
        for k in keys:
            result[k] = STOCH_BLOCKS[k](log)
        meta = result.setdefault("_meta", {})
        meta.setdefault("fields", {})["stochastic"] = STOCH_FIELD_NOTE
        meta["updated_blocks"] = {**meta.get("updated_blocks", {}), **{k: _dt.date.today().isoformat() for k in keys}}
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
        log(f"updated {', '.join(keys)} in {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
        return 0

    result: dict = {}
    base = {}
    for which in ("ref", "photo"):
        e = evaluate(base_device(which), vd_max=VD_MAX[which])
        base[which] = dict(preset=PRESET_OF[which], vg=float(base_device(which)["vg"]), iph_pA=0.0,
                           V_LU=rnd(e["V_LU"]), V_LD=rnd(e["V_LD"]), window_V=rnd(e["window_V"]),
                           vd_max_V=VD_MAX[which], fold_currents_A=fold_currents(e))
    dph = set_path(base_device("photo"), ("light", "iph_pA"), IPH_2P55)
    e = evaluate(dph, vd_max=5.0)
    base["photo_2p55mW"] = dict(preset="photo", vg=-1.8, iph_pA=IPH_2P55, power_mW=2.55, V_LU=rnd(e["V_LU"]),
                                V_LD=rnd(e["V_LD"]), window_V=rnd(e["window_V"]), vd_max_V=5.0,
                                fold_currents_A=fold_currents(e))
    result["_meta"] = dict(
        generated=_dt.date.today().isoformat(), script="scripts/param_sensitivity.py", grid=GRID,
        presets=dict(ref="paper: FDSOI reference calibration, V_G = −2 V, dark, 0.4 V/s",
                     photo="photo: illumination calibration, V_G = −1.8 V, P = 0 mW, 1200 V/s, γ = 0.2794, "
                           "δφ_G0 = +74.4 mV"),
        fields=dict(
            base_up_down="deterministic folds (V) at the baseline / increased / decreased value; value = UI display units",
            dV_mV="fold shift vs baseline in mV (up = increase step, down = decrease step)",
            slope="local derivative from small symmetric steps (×1.05 or step/5), in the stated unit",
            window_exists="two folds with V_LU > V_LD and V_LU ≤ 8 V (server sweep cap); spurious classify folds "
                          "above 8 V are rejected",
            latch_up_exists="window exists and V_LU ≤ the preset sweep maximum (4 V ref, 5 V photo)",
            latch_down_exists="window exists and V_LD > 0 (the device returns to HRS on the down sweep)",
            nonmonotonic="up and down steps shift the fold in the same direction",
            mechanism=("terms_pct: change of each term of the body-hole balance F at the baseline fold state (fixed u, r) "
                       "in % of the baseline hole generation there (sign = change of the term itself; for the loss terms "
                       "diff/bulk/junction a negative value means less loss). dF_pct: net change of F (+ = more net holes "
                       "= the body charges more easily). dVD_series_mV: "
                       "change of V_D needed for the same internal state (series R, hole drop). at_fold_ratio: "
                       "branch-column ratios increased/baseline along the actual folds."),
            stochastic=STOCH_FIELD_NOTE),
        no_effect_threshold_mV=NOEFF_MV)
    result["baseline"] = base
    log(f"baseline ref V_LU={base['ref']['V_LU']} V_LD={base['ref']['V_LD']}; photo V_LU={base['photo']['V_LU']} "
        f"V_LD={base['photo']['V_LD']}; photo 2.55 mW V_LU={base['photo_2p55mW']['V_LU']} V_LD={base['photo_2p55mW']['V_LD']}")

    for spec in PARAMS:
        k = spec["key"]
        blk = dict(key=k, units=spec["unit"])
        for which in ("ref", "photo"):
            blk[which] = device_block(spec, which)
        blk["step_up"], blk["step_down"] = blk["photo"].pop("step_up"), blk["photo"].pop("step_down")
        blk["step_up_ref"], blk["step_down_ref"] = blk["ref"].pop("step_up"), blk["ref"].pop("step_down")
        if blk["step_up_ref"] == blk["step_up"]:
            blk.pop("step_up_ref")
        if blk["step_down_ref"] == blk["step_down"]:
            blk.pop("step_down_ref")
        if k == "light":
            spec2 = dict(spec, kind="mul", extra=[])
            blk["photo_2p55mW"] = device_block(spec2, "photo", dev_base=dph)
            blk["photo_2p55mW"]["step_up"], blk["photo_2p55mW"]["step_down"] = "×2 (5.1 mW)", "×0.5 (1.275 mW)"
            blk["power_mW_per_pA"] = round(1 / RESP, 4)
        if spec.get("alt_vg") is not None:
            vg_alt = spec["alt_vg"]
            alt = dict(vg=vg_alt, note=f"same steps at V_G = {vg_alt:g} V, where the channel is less deeply off")
            for which in ("ref", "photo"):
                bd = set_path(base_device(which), ("vg",), vg_alt)
                a = device_block(spec, which, dev_base=bd)
                alt[which] = {kk: a[kk] for kk in ("base", "up", "down", "dVLU_up_mV", "dVLD_up_mV", "dVLU_down_mV",
                                                   "dVLD_down_mV", "slope", "mechanism_note", "summary")}
            blk[f"at_vg_{vg_alt:+.1f}".replace(".", "p").replace("-", "m").replace("+", "p")] = alt
        if k == "vg":
            fx = {}
            for which in ("ref", "photo"):
                d0 = base_device(which)
                v0 = float(d0["vg"])
                evs = {"base": evaluate(d0, vd_max=VD_MAX[which]),
                       "up": evaluate(set_path(d0, ("vg",), v0 + 0.1), vd_max=VD_MAX[which]),
                       "down": evaluate(set_path(d0, ("vg",), v0 - 0.1), vd_max=VD_MAX[which])}
                fx[which] = hrs_fixed_vd(evs)
            blk["hrs_at_fixed_VD_A"] = fx
        if spec.get("ctx_note"):
            blk["context"] = spec["ctx_note"]
        blk["notes"] = NOTES.get(k, "")
        result[k] = blk
        r, ph = blk["ref"], blk["photo"]
        log(f"{k:<13s} ref up {r['dVLU_up_mV']}/{r['dVLD_up_mV']} down {r['dVLU_down_mV']}/{r['dVLD_down_mV']} | "
            f"photo up {ph['dVLU_up_mV']}/{ph['dVLD_up_mV']} down {ph['dVLU_down_mV']}/{ph['dVLD_down_mV']}")

    if not args.det_only:
        det_base = {w: dict(V_LU=base[w]["V_LU"], V_LD=base[w]["V_LD"]) for w in ("ref", "photo")}
        st = run_stochastic(det_base, log)
        result["baseline"]["mc"] = st.pop("_baseline_mc")
        result.update(st)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    log(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    print_table(result)
    return 0


def geometry_main(args, log) -> int:
    """Geometry / V_BG blocks, same fields as the calibration-parameter blocks (reference and illumination presets)."""
    res: dict = dict(_meta=dict(generated=_dt.date.today().isoformat(), script="scripts/param_sensitivity.py --geometry",
                                grid=GRID, model=P.GEOMETRY_MODEL_ASSUMPTIONS,
                                note="deterministic folds of the geometry extension; not a fit to multi-geometry data"))
    for spec in GEOMETRY_PARAMS:
        k = spec["key"]
        blk = dict(key=k, units=spec["unit"])
        for which in ("ref", "photo"):
            blk[which] = device_block(spec, which)
            blk[which].pop("step_up", None), blk[which].pop("step_down", None)
        up, dn, su, sd = steps_for(spec, get_path(base_device("ref"), spec["path"]))
        blk["step_up"], blk["step_down"] = su, sd
        if spec.get("alt_vg") is not None:
            vg_alt = spec["alt_vg"]
            a = device_block(spec, "ref", dev_base=set_path(base_device("ref"), ("vg",), vg_alt))
            blk[f"ref_at_vg_{vg_alt:+.1f}"] = {kk: a[kk] for kk in ("base", "up", "down", "dVLU_up_mV", "dVLD_up_mV",
                                                                  "dVLU_down_mV", "dVLD_down_mV", "slope")}
        if spec.get("ctx_note"):
            blk["context"] = spec["ctx_note"]
        res[k] = blk
        r, ph = blk["ref"], blk["photo"]
        log(f"{k:<10s} {su!s:>10s}/{sd!s:<10s} ref up {r['dVLU_up_mV']}/{r['dVLD_up_mV']} down "
            f"{r['dVLU_down_mV']}/{r['dVLD_down_mV']} | photo up {ph['dVLU_up_mV']}/{ph['dVLD_up_mV']} down "
            f"{ph['dVLU_down_mV']}/{ph['dVLD_down_mV']}")
        for ex in r.get("extra") or []:
            log(f"{'':<10s} ref {ex}")
    if args.geometry_out:
        out = Path(args.geometry_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
        log(f"wrote {out}")
    return 0


def print_table(res: dict) -> None:
    f = lambda x: "   n/a" if x is None else f"{x:+7.1f}"
    print()
    print(f"{'key':<14s}{'step':>12s} | {'ref dVLU':>8s} {'ref dVLD':>8s} | {'ph dVLU':>8s} {'ph dVLD':>8s} || "
          f"{'step':>10s} | {'ref dVLU':>8s} {'ref dVLD':>8s} | {'ph dVLU':>8s} {'ph dVLD':>8s}")
    for spec in PARAMS:
        b = res[spec["key"]]
        r, p = b["ref"], b["photo"]
        print(f"{spec['key']:<14s}{str(b.get('step_up') or b.get('step_up_ref') or '-'):>12s} | "
              f"{f(r['dVLU_up_mV']):>8s} {f(r['dVLD_up_mV']):>8s} | {f(p['dVLU_up_mV']):>8s} {f(p['dVLD_up_mV']):>8s} || "
              f"{str(b.get('step_down') or b.get('step_down_ref') or '-'):>10s} | "
              f"{f(r['dVLU_down_mV']):>8s} {f(r['dVLD_down_mV']):>8s} | {f(p['dVLU_down_mV']):>8s} {f(p['dVLD_down_mV']):>8s}")
    if "photo_2p55mW" in res.get("light", {}):
        q = res["light"]["photo_2p55mW"]
        print(f"{'light@2.55mW':<14s}{'×2':>12s} | {'':>8s} {'':>8s} | {f(q['dVLU_up_mV']):>8s} {f(q['dVLD_up_mV']):>8s} || "
              f"{'×0.5':>10s} | {'':>8s} {'':>8s} | {f(q['dVLU_down_mV']):>8s} {f(q['dVLD_down_mV']):>8s}")
    for k in ("rate", "dv", "ls_sigma", "ls_tau", "ls_sigmaE", "carrier_noise", "ld_carrier_noise", "n_cycles",
              "fold_nodes", "hazard_nodes", "grid"):
        if k not in res:
            continue
        b = res[k]
        for which in ("ref", "photo"):
            x = b.get(which)
            if not x:
                continue
            g = lambda d, kk: None if d is None else d.get(kk)
            print(f"{k:<14s}{which:>6s}: base mean/sd LU {g(x['base'], 'mean_VLU')}/{g(x['base'], 'sd_VLU_mV')} "
                  f"LD {g(x['base'], 'mean_VLD')}/{g(x['base'], 'sd_VLD_mV')} | up dLU {x.get('dVLU_up_mV')} "
                  f"dLD {x.get('dVLD_up_mV')} dsdLU {x.get('dsdVLU_up_mV')} dsdLD {x.get('dsdVLD_up_mV')} | "
                  f"down dLU {x.get('dVLU_down_mV')} dLD {x.get('dVLD_down_mV')} dsdLU {x.get('dsdVLU_down_mV')} "
                  f"dsdLD {x.get('dsdVLD_down_mV')}")


if __name__ == "__main__":
    sys.exit(main())
