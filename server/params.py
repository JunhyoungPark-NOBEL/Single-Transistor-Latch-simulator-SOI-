"""Device parameter schema shared by every compute module (pure Python, no numba import).

A *device* object (JSON) is resolved against a preset. The reference uses the 26-element
parameter vector ``p`` used by ``engine/photo_extension/photo_mean.components``:

    p[0]  beta (diffusion ratio)          p[13] I_PH (A)
    p[1]  tau_bulk (s)                    p[14] DIBL eta (V/V)
    p[2]  tau_junction (s)                p[15] body-to-channel gamma
    p[3]  R_contact (ohm)                 p[16] slope kappa (1/V)
    p[4]  l_GIDL (nm)                     p[17] high-VD channel seed I_p at VG=-1.8 V (A)
    p[5]  t_access (nm)                   p[18] its slope S (V/dec)
    p[6]  N_A,access (cm^-3)              p[19] junction potential offset (V)
    p[7]  L_access (nm)                   p[20] log scale of (M-1)
    p[8]  tau_p/tau_n                     p[21] local avalanche strength
    p[9]  phi_GIDL offset (V)             p[22] local path saturation (A)
    p[10] phi_emitter offset (V)          p[23] local path log fluctuation
    p[11] V_G (V)                         p[24] local path carrier definition (0/1/2)
    p[12] channel-II scale                p[25] kappaF (1/V)

Changed geometry/backgate bias appends Lg, W, Tsi, EOT, Tbox, Nbody, VBG at p[26:33].
The first 26 entries keep their calibration meanings. Worker-only field tables are not public.

All paper-model calibration values are read from the engine JSON files so there is a
single source of truth.  With every extension at its neutral value the vector equals
``setup_photo.BASE`` (paper model, verified identical to gate_mean).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
_JM = ENGINE / "model" / "janus_calibration_20260920" / "claude_crosscheck_20260920" / "joint_model"
_PX = ENGINE / "photo_extension"

_REFIT = json.loads((_JM / "refit_3.json").read_text())
_DYN = json.loads((_JM / "gate_dynamic_calibration.json").read_text())
_C2C = json.loads((_PX / "c2c_calibration_m18_dark.json").read_text())
_CONV = json.loads((_PX / "photo_conversion_fit.json").read_text())
_GAMMA = float((_PX / "gamma_probe.txt").read_text().strip())

NA_CM3: float = float(_REFIT["NA_cm3"])
_P9 = [float(x) for x in _REFIT["prediction"]["parameters"]]  # p[0..8]
PHI_GIDL0_V: float = float(_DYN["parameters"][0])              # p[9]  (V)
PHI_EMITTER0_V: float = 1e-3 * float(_DYN["parameters"][1])    # p[10] (V)
SIGMA_PHI_G_V: float = float(_DYN["parameters"][2])            # drain-edge state SD (paper device)
SIGMA_PHI_E_V: float = 1e-3 * float(_DYN["parameters"][3])     # emitter state SD (paper device)
TAU_E_S: float = float(_DYN["kinetic_fit"]["tau_fast_s"])      # emitter OU tau (paper)
TAU_G_UP_S: float = float(_DYN["kinetic_fit"]["up_residual_tau_s"])  # drain-edge OU tau (paper)
PHOTO_DELTA_PHI_G0_V: float = float(_C2C["delta_phi_G0_V"])     # this (photo) device, -1.8 V dark calibration
PHOTO_SIGMA_PHI_V: float = float(_C2C["sigma_phi_V"])
PHOTO_GAMMA: float = _GAMMA                                      # body coupling set by the -1.1 V dark mean
RESPONSIVITY_PA_PER_MW: float = 1e12 * float(_CONV["R_A_per_mW"])  # 0.75 pA/mW

CALIB_KEYS = [
    "beta", "tau_bulk_s", "tau_junction_s", "r_contact_ohm", "l_gidl_nm",
    "t_access_nm", "na_access_cm3", "l_access_nm", "tau_ratio",
]

DEFAULT_CALIB: dict[str, float] = {k: v for k, v in zip(CALIB_KEYS, _P9)}
DEFAULT_CALIB.update(phi_gidl0_V=PHI_GIDL0_V, phi_emitter0_V=PHI_EMITTER0_V, channel_ii_scale=1.0)

DEFAULT_EXT: dict[str, float] = dict(
    dibl=0.0,          # p[14]
    gamma=0.0,         # p[15]
    kappa=0.0,         # p[16]
    seed_ip_pA=0.0,    # p[17] (stored in pA in JSON, A in p)
    seed_S=1.0,        # p[18]
    dj=0.0,            # p[19]
    dm=0.0,            # p[20]
    aloc=0.0,          # p[21]
    isat_pA=20.0,      # p[22] (pA in JSON)
    dloc=0.0,          # p[23]
    loc_carriers=0,    # p[24]  0 edge incl. channel, 1 bulk, 2 edge excl. channel
    kappaF=0.0,        # p[25]
)

# Values the UI offers for the open "-1.1 V channel seed" problem (not applied by default
# in the paper preset).  The photo preset applies the body-coupling option, as mc_cycles.py does.
CHANNEL_SEED_OPTIONS = {
    "none": {},
    "body_coupling": {"gamma": PHOTO_GAMMA},
    "high_vd_seed": {"seed_ip_pA": 1.33, "seed_S": 0.8},
}

# Calibration geometry. Tbox was absent from the handoff and is explicitly nominal.
# The server-owned geometry extension adds deterministic extrapolation around this device.
TECHNOLOGY = "FDSOI"
GEOMETRY: dict[str, float] = dict(Lg_nm=500.0, W_nm=200.0, Tsi_nm=50.0, EOT_nm=14.1, Tbox_nm=140.0, Nbody_cm3=NA_CM3)
GEOMETRY_KEYS = tuple(GEOMETRY)
# Nbody: 3e16 keeps a neutral base at the reference L (L_min = 412 nm there); above ~1.14e18 cm^-3 the finite
# avalanche-field table ends below 3 V reverse bias (geometry_model.pack_p), so 1.1e18 is the top of the input range.
GEOMETRY_LIMITS = dict(Lg_nm=(100., 2000.), W_nm=(20., 10000.), Tsi_nm=(5., 200.),
                       EOT_nm=(1., 100.), Tbox_nm=(10., 1000.), Nbody_cm3=(3e16, 1.1e18))
# |EOT/(Tbox + Tsi/3) * VBG| above this leaves the linear (depleted back-interface) coupling range
BACKGATE_SHIFT_MAX_V = 2.0
# smallest field-table reverse bias a changed Nbody must support (V_LU of the reference device is 3.70 V)
FIELD_MIN_REVERSE_V = 3.0
GEOMETRY_TEXT = "L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm"
TECHNOLOGIES = [
    dict(id="FDSOI", available=True),
    dict(id="PDSOI", available=False),
    dict(id="Bulk", available=False),
]

# engine constants (process_randomness/standard_mean.py, idvd_model/mean_model.py), kept here so the API process can
# check the geometry domain without importing the engine; server/tests/test_geometry_model.py compares them
_Q_C = 1.602176634e-19
_VT_V = 1.380649e-23 * 300.0 / _Q_C
_EPS0_F_M = 8.8541878128e-12
_NI_CM3 = 1e10
_ND_CM3 = 1e20


def min_length_nm(nbody_cm3: float) -> float:
    """L (nm) at and below which no neutral lateral base is left at zero bias: 2 w_d0(Nbody) + 1 nm, with w_d0 the
    source/drain depletion width at the built-in potential of the n+ junction (same criterion as geometry_model.pack_p)."""
    import math
    vbi = _VT_V * math.log(_ND_CM3 * nbody_cm3 / _NI_CM3 ** 2)
    wd0 = math.sqrt(2 * (11.7 * _EPS0_F_M / 100) * vbi / (_Q_C * nbody_cm3))
    return (2 * wd0 + 1e-7) * 1e7


_DEVICE_BASE: dict[str, Any] = dict(
    geometry=dict(GEOMETRY),
    vg=-2.0,
    vbg=0.0,
    light=dict(mode="iph", iph_pA=0.0, power_mW=0.0, responsivity_pA_per_mW=RESPONSIVITY_PA_PER_MW),
    calib=DEFAULT_CALIB,
    ext=DEFAULT_EXT,
    state=dict(delta_phi_G0_V=0.0, delta_phi_E0_V=0.0),
    numerics=dict(grid=601),
)

PRESETS: dict[str, dict[str, Any]] = {
    "paper": dict(
        label={"ko": f"FDSOI · {GEOMETRY_TEXT} — 기준 보정 (암조건, V_G = −2 V, 0.4 V/s)",
               "en": f"FDSOI · {GEOMETRY_TEXT} — reference calibration (dark, V_G = −2 V, 0.4 V/s)"},
        technology=TECHNOLOGY, geometry=dict(GEOMETRY),
        device=dict(copy.deepcopy(_DEVICE_BASE), preset="paper", vg=-2.0),
        sweep=dict(vd_max_V=4.0, rate_V_per_s=0.4, dv_V=0.002),
        stochastic=dict(
            n_cycles=100, seed=2026092920, carrier_noise=True, ld_carrier_noise=True,
            local_state=dict(mode="evolving", action="gidl", sigma=SIGMA_PHI_G_V, tau_s=TAU_G_UP_S,
                             sigma_E_V=SIGMA_PHI_E_V, tau_E_s=TAU_E_S, acquisition_trend=True),
            engine="auto", n_traces=12, fold_nodes=25, hazard_nodes=5,
        ),
    ),
    "photo": dict(
        label={"ko": f"FDSOI · {GEOMETRY_TEXT} — 광조사 보정 (V_G = −1.8 V, 1200 V/s)",
               "en": f"FDSOI · {GEOMETRY_TEXT} — illumination calibration (V_G = −1.8 V, 1200 V/s)"},
        technology=TECHNOLOGY, geometry=dict(GEOMETRY),
        device=dict(
            copy.deepcopy(_DEVICE_BASE), preset="photo", vg=-1.8,
            light=dict(mode="power", iph_pA=0.0, power_mW=0.0, responsivity_pA_per_mW=RESPONSIVITY_PA_PER_MW),
            ext=dict(DEFAULT_EXT, gamma=PHOTO_GAMMA),
            state=dict(delta_phi_G0_V=PHOTO_DELTA_PHI_G0_V, delta_phi_E0_V=0.0),
        ),
        sweep=dict(vd_max_V=5.0, rate_V_per_s=1200.0, dv_V=0.002),
        stochastic=dict(
            n_cycles=400, seed=20260922, carrier_noise=True, ld_carrier_noise=True,
            local_state=dict(mode="frozen", action="gidl", sigma=PHOTO_SIGMA_PHI_V, tau_s=TAU_G_UP_S,
                             sigma_E_V=0.0, tau_E_s=TAU_E_S, acquisition_trend=False),
            engine="auto", n_traces=12, fold_nodes=25, hazard_nodes=5,
        ),
    ),
}
PRESETS["custom"] = copy.deepcopy(PRESETS["paper"])
PRESETS["custom"]["label"] = {"ko": f"FDSOI · {GEOMETRY_TEXT} — 사용자 정의 (기준 보정 값에서 시작)",
                              "en": f"FDSOI · {GEOMETRY_TEXT} — custom (starts from the reference calibration)"}
PRESETS["custom"]["device"]["preset"] = "custom"

# Photo-device measured conditions (columns of data/raw_VLU.npy, same order as measured_stats.json).
MEASURED_PHOTO_CONDITIONS = [(-1.8, 0.0), (-1.8, 1.15), (-1.8, 2.55), (-1.8, 3.51),
                             (-1.1, 0.0), (-1.1, 1.15), (-1.1, 2.55), (-1.1, 3.51)]


def _merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


def resolve_device(device: dict | None) -> dict:
    """Fill a (possibly partial) device object from its preset's defaults."""
    device = device or {}
    preset = device.get("preset") or "paper"
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    return _merge(PRESETS[preset]["device"], device)


def resolve_section(preset: str | None, name: str, value: dict | None) -> dict:
    """Fill a sweep/stochastic section from the preset defaults."""
    return _merge(PRESETS[preset or "paper"][name], value)


def iph_A(device: dict) -> float:
    """Photogeneration current in ampere from the light block (I_PH = R * P in power mode)."""
    light = device["light"]
    if light.get("mode") == "power":
        return 1e-12 * float(light["responsivity_pA_per_mW"]) * float(light["power_mW"])
    return 1e-12 * float(light["iph_pA"])


def build_p(device: dict, dg: float = 0.0, de: float = 0.0, **ext_over: float) -> list[float]:
    """Reference vector (26) or geometry/backgate extension (33).

    dg, de: local-state deviations added to p[9], p[10] on top of the calibrated means and the
    device's state centre (delta_phi_G0_V, delta_phi_E0_V).  ext_over overrides extension keys
    (e.g. dloc=..., dj=..., dm=...) for the non-GIDL local-state action points.
    """
    d = resolve_device(device)
    c, e, s = d["calib"], dict(d["ext"], **ext_over), d["state"]
    p = [float(c[k]) for k in CALIB_KEYS]
    p += [
        float(c["phi_gidl0_V"]) + float(s["delta_phi_G0_V"]) + dg,   # 9
        float(c["phi_emitter0_V"]) + float(s["delta_phi_E0_V"]) + de,  # 10
        float(d["vg"]),                  # 11
        float(c["channel_ii_scale"]),    # 12
        iph_A(d),                        # 13
        float(e["dibl"]),                # 14
        float(e["gamma"]),               # 15
        float(e["kappa"]),               # 16
        1e-12 * float(e["seed_ip_pA"]),  # 17
        float(e["seed_S"]),              # 18
        float(e["dj"]),                  # 19
        float(e["dm"]),                  # 20
        float(e["aloc"]),                # 21
        1e-12 * float(e["isat_pA"]),     # 22
        float(e["dloc"]),                # 23
        float(e["loc_carriers"]),        # 24
        float(e["kappaF"]),              # 25
    ]
    g = d["geometry"]
    if uses_geometry_model(d):
        p += [float(g[k]) for k in GEOMETRY_KEYS] + [float(d.get("vbg", 0.0))]
    return p


def is_reference_geometry(device: dict) -> bool:
    g = device.get("geometry") or GEOMETRY
    return all(float(g.get(k, v)) == v for k, v in GEOMETRY.items())


def uses_geometry_model(device: dict) -> bool:
    return not is_reference_geometry(device) or float(device.get("vbg", 0.0)) != 0.0


# model assumptions reported with every geometry result (and in /api/meta)
GEOMETRY_MODEL_ASSUMPTIONS: dict[str, str] = dict(
    backgate_coupling=("channel overdrive += EOT / (Tbox + Tsi/3) * VBG; relative VBG=0 calibration; valid while "
                       f"|EOT / (Tbox + Tsi/3) * VBG| <= {BACKGATE_SHIFT_MAX_V:g} V (linear coupling, depleted back "
                       "interface); larger values are refused"),
    backgate_scope=("front-channel current only: VBG does not store holes in the body in this model, so with the "
                    "channel off (V_G below about -1.5 V) V_LU and V_LD hardly change"),
    emitter_injection_scaling="area only (fixed n+ source/drain)",
    junction_lifetime_scaling="tau_j_eff = tau_j_ref * Tsi / 50nm",
    body_capacitance_scaling="Cf + Cb - Cb0; Cb0 is reference-calibration counterterm",
    gate_charge="Cf*(psi-VG) + (Cb-Cb0)*psi - Cb*VBG",
    domain=("L > 2 w_d0(Nbody) + 1 nm (neutral lateral base); Nbody with an avalanche-field table reaching "
            f"{FIELD_MIN_REVERSE_V:g} V reverse bias; |EOT / (Tbox + Tsi/3) * VBG| <= {BACKGATE_SHIFT_MAX_V:g} V"),
)


def geometry_model_metadata(device: dict) -> dict:
    return dict(version="fdsoi-scaling-v1", calibrated_geometry=dict(GEOMETRY),
                reference_geometry=is_reference_geometry(device),
                tbox_source="nominal assumption; absent from calibration",
                validated=not uses_geometry_model(device),
                scope="reference-calibrated" if not uses_geometry_model(device) else "geometry-extrapolation",
                **GEOMETRY_MODEL_ASSUMPTIONS)


def is_paper_reference(device: dict) -> bool:
    """True when the device equals the calibrated paper device at V_G = -2 V, dark, no extensions
    (the only condition covered by the calibrated (phi_G, phi_E) lookup table gate_state_lookup.npz)."""
    d = resolve_device(device)
    base = resolve_device({"preset": "paper"})
    same_calib = all(abs(float(d["calib"][k]) - float(base["calib"][k])) <= 1e-12 * max(1.0, abs(float(base["calib"][k])))
                     for k in base["calib"])
    same_ext = all(float(d["ext"][k]) == float(DEFAULT_EXT[k]) for k in DEFAULT_EXT)
    return (not uses_geometry_model(d) and abs(float(d["vg"]) + 2.0) < 1e-9 and iph_A(d) == 0.0 and same_calib and same_ext
            and float(d["state"]["delta_phi_G0_V"]) == 0.0 and float(d["state"]["delta_phi_E0_V"]) == 0.0)


def match_photo_condition(device: dict, tol_vg: float = 1e-6, tol_p: float = 1e-6) -> int | None:
    """Column index into data/raw_VLU.npy when (V_G, P) equals a measured photo-device condition."""
    d = resolve_device(device)
    if uses_geometry_model(d):
        return None
    light = d["light"]
    p_mw = float(light["power_mW"]) if light.get("mode") == "power" else iph_A(d) * 1e12 / RESPONSIVITY_PA_PER_MW
    for k, (vg, pw) in enumerate(MEASURED_PHOTO_CONDITIONS):
        if abs(float(d["vg"]) - vg) < tol_vg and abs(p_mw - pw) < max(tol_p, 0.005):
            return k
    return None


def meta() -> dict:
    """Payload for GET /api/meta."""
    return dict(
        presets=PRESETS,
        constants=dict(NA_cm3=NA_CM3, sigma_phi_G_V=SIGMA_PHI_G_V, sigma_phi_E_V=SIGMA_PHI_E_V,
                       tau_E_s=TAU_E_S, tau_G_up_s=TAU_G_UP_S, photo_delta_phi_G0_V=PHOTO_DELTA_PHI_G0_V,
                       photo_sigma_phi_V=PHOTO_SIGMA_PHI_V, photo_gamma=PHOTO_GAMMA,
                       responsivity_pA_per_mW=RESPONSIVITY_PA_PER_MW,
                       geometry=dict(L_nm=500, W_nm=200, T_Si_nm=50, EOT_nm=14.1, Tbox_nm=140, Nbody_cm3=NA_CM3)),
        technology=TECHNOLOGY,
        geometry=dict(GEOMETRY),
        geometry_limits=GEOMETRY_LIMITS,
        geometry_model=dict(version="fdsoi-scaling-v1", calibrated_geometry=dict(GEOMETRY),
                            tbox_source="nominal assumption; absent from calibration", validated=False,
                            backgate_shift_max_V=BACKGATE_SHIFT_MAX_V, field_min_reverse_V=FIELD_MIN_REVERSE_V,
                            **GEOMETRY_MODEL_ASSUMPTIONS),
        technologies=TECHNOLOGIES,
        channel_seed_options=CHANNEL_SEED_OPTIONS,
        measured_photo_conditions=[dict(vg=vg, power_mW=p) for vg, p in MEASURED_PHOTO_CONDITIONS],
    )
