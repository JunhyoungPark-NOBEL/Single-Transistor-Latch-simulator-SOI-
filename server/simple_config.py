"""Pure-Python Simple Model contract; safe in the API process (no numerical engine import)."""
SIMPLE_VERSION = "simple-edl2026-paper-v2"
SIMPLE_MARKER = -1.0
SIMPLE_PACK_SIZE = 3038
# Published model the Simple Model implements (equations (1)-(4), (7) and Table I).
# The Detailed Model is the calibrated "updated accuracy" model that extends it.
SIMPLE_REFERENCE = dict(
    authors="J.-H. Park, H.-B. Noh, S.-W. Lee, S.-Y. Yun, and Y.-K. Choi",
    title="Analytical Model for Single Transistor Latch in MOSFETs",
    journal="IEEE Electron Device Letters",
    year=2026,
    doi="10.1109/LED.2026.3737574",
    citation=("J.-H. Park, H.-B. Noh, S.-W. Lee, S.-Y. Yun, and Y.-K. Choi, \"Analytical Model for Single "
              "Transistor Latch in MOSFETs,\" IEEE Electron Device Lett., 2026, doi: 10.1109/LED.2026.3737574."),
)
SIMPLE_KEYS = (
    "beta_ref", "tau_body_s", "cb_ref_F", "r_lrs_ref_ohm", "is_ref_A", "vbr_ref_V",
    "avalanche_eta", "gamma_fg", "gamma_bg", "vfb_V", "btbt_scale", "surface_fraction",
    "gidl_volume_scale",
)
# Manuscript Table I values rescaled from W=650 nm to the app's Wref=200 nm.
# Constant diffusion beta=2.3 replaces manuscript Eq.(8) (owner decision).
# gidl_volume_scale=100 reproduces the manuscript's GIDL generation volume
# (W * 5 nm overlap * tunnelling depth, times 100 as in the reference script);
# 1 would be the bare physical volume.  These are illustrative initial values,
# not a fit to Device 1's measurements.
SIMPLE_DEFAULTS = dict(
    beta_ref=2.3, tau_body_s=2e-7, cb_ref_F=0.86e-15*200/650,
    r_lrs_ref_ohm=44000.*650/200, is_ref_A=2e-16*200/650,
    vbr_ref_V=2.35, avalanche_eta=4., gamma_fg=.2, gamma_bg=.0525,
    vfb_V=-3.35, btbt_scale=1., surface_fraction=1., gidl_volume_scale=100.,
)
SIMPLE_LIMITS = dict(
    beta_ref=(.01, 1e4), tau_body_s=(1e-12, 10.), cb_ref_F=(1e-20, 1e-6),
    r_lrs_ref_ohm=(0., 1e12), is_ref_A=(1e-35, 1e-3), vbr_ref_V=(.1, 100.),
    avalanche_eta=(1., 12.), gamma_fg=(0., 1.), gamma_bg=(0., 1.),
    vfb_V=(-10., 10.), btbt_scale=(0., 1e6), surface_fraction=(0., 1.),
    gidl_volume_scale=(0., 1e6),
)


def is_simple_device(device):
    return isinstance(device, dict) and device.get("model", "detailed") == "simple"
