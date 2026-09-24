"""Measured data and the design map, served by GET /api/data/measured and GET /api/data/design_map.

Pure numpy/json: safe to import in the API process (never imports numba or the engine code).
All arrays are returned as numpy arrays (serialised by server.jsonutil, NaN → null).
Units: voltages in V, currents in A, optical power in mW, photocurrent in pA, times in s.
Keys and shapes are documented in docs/API.md.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from server import params

ENGINE = Path(__file__).resolve().parents[2] / "engine"
DATA = ENGINE / "data"
PAPER_IDVD = ENGINE / "model" / "janus_calibration_20260920" / "outputs" / "measured_idvd_parsed.npz"
PROTOCOL = ENGINE / "model" / "janus_calibration_20260920" / "inputs" / "measurement_protocol.json"
CONVERSION = ENGINE / "photo_extension" / "photo_conversion_fit.json"

PHOTO_RATE_V_PER_S = 1200.0
PHOTO_VD_MAX_V = 5.0


def _f(x: Any) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def lag1(x: np.ndarray) -> float | None:
    """Lag-1 autocorrelation sum(dx_t dx_{t+1}) / sum(dx_t^2) (definition used in measured_stats.json)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return None
    xc = x - x.mean()
    den = float(np.sum(xc * xc))
    return float(np.sum(xc[1:] * xc[:-1]) / den) if den > 0 else None


def stats(values: Any, censored: int | None = None) -> dict:
    """Contract `Stats`: n (finite values), mean, sd (ddof=1), median, p05, p95, min, max, censored, lag1.

    Non-finite entries count as censored unless `censored` is given explicitly.
    """
    v = np.asarray(values, float).ravel()
    ok = np.isfinite(v)
    x = v[ok]
    cens = int((~ok).sum()) if censored is None else int(censored)
    if len(x) == 0:
        return dict(n=0, mean=None, sd=None, median=None, p05=None, p95=None, min=None, max=None,
                    censored=cens, lag1=None)
    return dict(
        n=int(len(x)), mean=_f(x.mean()), sd=_f(x.std(ddof=1)) if len(x) > 1 else None,
        median=_f(np.median(x)), p05=_f(np.quantile(x, 0.05)), p95=_f(np.quantile(x, 0.95)),
        min=_f(x.min()), max=_f(x.max()), censored=cens, lag1=lag1(v[ok]),
    )


def _band(curves: np.ndarray, n_samples: int = 10) -> dict:
    """curves: (n_points, n_sweeps). Median, 10/90 % bands and a few chronological sample sweeps."""
    idx = np.unique(np.round(np.linspace(0, curves.shape[1] - 1, n_samples)).astype(int))
    return dict(
        median=np.nanmedian(curves, axis=1),
        p10=np.nanquantile(curves, 0.10, axis=1),
        p90=np.nanquantile(curves, 0.90, axis=1),
        sample_index=idx,
        samples=curves[:, idx].T,          # (n_samples, n_points)
    )


@lru_cache(maxsize=1)
def _measured_cached() -> dict:
    conv = json.loads(CONVERSION.read_text())
    resp_pA = 1e12 * float(conv["R_A_per_mW"])
    protocol = json.loads(PROTOCOL.read_text())

    # --- photo device: 400 cycles x 8 conditions at 1200 V/s ---------------------------------
    raw = np.load(DATA / "raw_VLU.npy")                     # (400, 8)
    mstats = json.loads((DATA / "measured_stats.json").read_text())
    conds = []
    for k, (vg, pw) in enumerate(params.MEASURED_PHOTO_CONDITIONS):
        rec = mstats[k]
        assert abs(rec["VG"] - vg) < 1e-9 and abs(rec["P_mW"] - pw) < 1e-9, "column order mismatch"
        conds.append(dict(index=k, label=rec["label"], vg=vg, power_mW=pw, iph_pA=resp_pA * pw,
                          stats=stats(raw[:, k]), file_stats=rec))
    photo = dict(
        description="Photo device, triangular 0->5 V sweeps at 1200 V/s, 400 cycles per condition; "
                    "V_LU per cycle (V). V_LU[k][c] = condition k, cycle c (chronological).",
        rate_V_per_s=PHOTO_RATE_V_PER_S, vd_max_V=PHOTO_VD_MAX_V, n_cycles=int(raw.shape[0]),
        responsivity_pA_per_mW=resp_pA,
        conditions=conds,
        V_LU=raw.T,                                          # (8, 400)
    )

    # --- photo device: light-dependent I_D-V_D (one sweep per power) ---------------------------
    iv = np.load(DATA / "idvd_light.npy")                   # (101, 7): V_D, then 6 powers
    powers = np.asarray(conv["P_mW"], float)
    light_iv = dict(
        description="Photo device I_D-V_D under illumination (V_G = -1.8 V assumed by the conversion fit); "
                    "id[k] is the trace at power_mW[k].",
        vg=float(conv.get("VG_assumed", -1.8)), vd=iv[:, 0], power_mW=powers, iph_pA=resp_pA * powers,
        id=iv[:, 1:].T,                                     # (6, 101)
        plateau_vd_V=float(conv["VD_plateau_V"]),
    )

    # --- dark I_D-V_G at V_D = 0.05 V ----------------------------------------------------------
    gv = np.load(DATA / "idvg_dark.npy")                    # (161, 3)
    idvg = dict(
        description="Dark I_D-V_G at V_D = 0.05 V, two traces (direction/repeat identity unconfirmed).",
        vd_V=float(protocol["IDVG"]["drain_voltage_V"]), vg=gv[:, 0], id=gv[:, 1:].T,   # (2, 161)
    )

    # --- paper device: 100 up + 100 down sweeps at V_G = -2 V, 0.4 V/s -------------------------
    z = np.load(PAPER_IDVD)
    vlu = (z["VLU_low"] + z["VLU_high"]) / 2
    vld = (z["VLD_low"] + z["VLD_high"]) / 2
    paper = dict(
        description="Paper device, V_G = -2 V, dark, 0->4 V (up) and 4->0 V (down), 10 mV steps, ~0.4 V/s, "
                    "100 separate up and 100 separate down sweeps (not paired cycles). "
                    "Currents summarised as median and 10/90 % quantile bands plus 10 chronological samples.",
        vg=-2.0, rate_V_per_s=0.4, n_sweeps=int(z["Iup"].shape[1]),
        up=dict(vd=z["Vup"], **_band(z["Iup"])),
        down=dict(vd=z["Vdown"], **_band(z["Idown"])),
        V_LU=vlu, V_LD=vld,
        V_LU_bracket=dict(low=z["VLU_low"], high=z["VLU_high"]),
        V_LD_bracket=dict(low=z["VLD_low"], high=z["VLD_high"]),
        stats=dict(LU=stats(vlu), LD=stats(vld)),
    )

    return dict(
        units=dict(voltage="V", current="A", power="mW", photocurrent="pA", rate="V/s"),
        photo=photo, light_iv=light_iv, idvg_dark=idvg, paper_idvd=paper,
    )


def measured() -> dict:
    """GET /api/data/measured.

    Keys: units; photo {rate_V_per_s, vd_max_V, n_cycles, responsivity_pA_per_mW, conditions[8]
    {index,label,vg,power_mW,iph_pA,stats,file_stats}, V_LU (8x400)}; light_iv {vg, vd(101), power_mW(6),
    iph_pA(6), id (6x101), plateau_vd_V}; idvg_dark {vd_V, vg(161), id (2x161)}; paper_idvd {vg, rate_V_per_s,
    n_sweeps, up/down {vd(401), median, p10, p90, sample_index(10), samples (10x401)}, V_LU(100), V_LD(100),
    V_LU_bracket/V_LD_bracket {low, high}, stats {LU, LD}}.
    """
    return _measured_cached()


DESIGN_MAP_DOC: dict[str, dict[str, str]] = {
    "length_nm": dict(unit="nm", meaning="lateral size L of the local (trap) region; x axis (81, log-spaced 3-100 nm)"),
    "depth_fraction": dict(unit="1", meaning="depth of the local region as a fraction of the film; y axis (61, 0-0.85)"),
    "sigma_phi_mV": dict(unit="mV", meaning="SD of the local drain-edge potential state sigma_phi (depth x length)"),
    "sigma_VLU_mV": dict(unit="mV", meaning="resulting cycle-to-cycle SD of V_LU for a 0-4 V sweep (depth x length)"),
    "latched_fraction": dict(unit="1", meaning="fraction of cycles that latch within the 0-4 V sweep (depth x length)"),
    "sigma_VLU_sweep5p2V_mV": dict(unit="mV", meaning="SD of V_LU for a 0-5.2 V sweep (depth x length)"),
    "expected_trap_count": dict(unit="1", meaning="expected number of traps N_t L^2 in the local region (depth x length)"),
    "Nt_cm2": dict(unit="cm^-2", meaning="trap areal density used for the map"),
    "device_sigma_phi_mV": dict(unit="mV", meaning="calibrated sigma_phi of the paper device (153.4 mV) - reference contour"),
    "phi_50mV": dict(unit="mV", meaning="reference contour value stored with the map (as provided)"),
    "line_Nt": dict(unit="cm^-2", meaning="trap densities of the reference lines"),
    "line_L0_device": dict(unit="nm", meaning="length L at which sigma_phi reaches the device value for each line_Nt"),
    "line_L0_50": dict(unit="nm", meaning="length L of the second reference contour for each line_Nt"),
}


@lru_cache(maxsize=1)
def _design_map_cached() -> dict:
    z = np.load(DATA / "design_map_filled.npz")
    arrays: dict[str, Any] = {}
    scalars: dict[str, float] = {}
    shapes: dict[str, list[int]] = {}
    for k in z.files:
        a = np.asarray(z[k])
        shapes[k] = list(a.shape)
        if a.ndim == 0:
            scalars[k] = float(a)
        else:
            arrays[k] = a
    return dict(
        description="Design map (design_map_filled.npz). 2-D arrays are indexed [depth_fraction][length_nm]. "
                    "Interpretation of the reference lines is inferred from the stored values.",
        axes=dict(x="length_nm", y="depth_fraction"),
        arrays=arrays, scalars=scalars, shapes=shapes, doc=DESIGN_MAP_DOC,
    )


def design_map() -> dict:
    """GET /api/data/design_map: {description, axes{x,y}, arrays{name: array}, scalars{name: float}, shapes, doc}."""
    return _design_map_cached()
