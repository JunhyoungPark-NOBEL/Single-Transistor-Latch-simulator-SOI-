"""Single import point for the numerical engine (engine/ = the handoff package, kept verbatim).

Importing this module triggers numba JIT loading (cached under engine/**/__pycache__) and should
only happen inside worker processes, never in the API process.

    from server.engine_bridge import A, S, m, MODEL, ct, np
    A      : engine/stl_api.py        (branches, folds, hazard, sweeps)
    S      : photo_extension/setup_photo.py (params, state, BASE, SIGG, SIGE)
    m      : photo_extension/photo_mean.py  (components, curve_grid, state_grid, constants)
    MODEL  : photo_mean.FastModel for the calibrated N_A (classify, branch, rg, fg)
    ct     : hypothesis_study_20260920/conditional_table.py (ct.cf = compound_fpt with the
             extended avalanche kernel installed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ENGINE = Path(__file__).resolve().parents[1] / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import stl_api as A  # noqa: E402  (sets up the remaining engine paths)

S = A.S
m = A.m
MODEL = A.MODEL
ct = S.ct
JOINT_MODEL_DIR = ENGINE / "model" / "janus_calibration_20260920" / "claude_crosscheck_20260920" / "joint_model"
PHOTO_DIR = ENGINE / "photo_extension"
DATA_DIR = ENGINE / "data"

__all__ = ["A", "S", "m", "MODEL", "ct", "np", "ENGINE", "JOINT_MODEL_DIR", "PHOTO_DIR", "DATA_DIR", "p_array"]


def p_array(p) -> np.ndarray:
    return np.asarray(p, dtype=float)
