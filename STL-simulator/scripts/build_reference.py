"""Rebuild the compact Reference ID–VD benchmark from supplied measurements and the real engine.

Run from repository root: python3 scripts/build_reference.py
This file is a fixed calibration benchmark, independent of the user's current device parameters.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import params
from server.compute.data import measured, PAPER_IDVD
from server.compute.deterministic import run_branches
from server.jsonutil import dumps


def assemble(result: dict) -> dict:
    ref = measured()["paper_idvd"]
    return {
        "provenance": {
            "kind": "fixed_reference_calibration",
            "measurement_source": str(PAPER_IDVD.relative_to(ROOT)),
            "measurement_sha256": hashlib.sha256(PAPER_IDVD.read_bytes()).hexdigest(),
            "model_source": "server.compute.deterministic.run_branches",
            "preset": "paper", "vg_V": -2, "iph_pA": 0, "n_sweeps_per_direction": ref["n_sweeps"],
            "sweep": params.PRESETS["paper"]["sweep"],
            "grid": params.PRESETS["paper"]["device"]["numerics"]["grid"],
        },
        "measured": {
            "vd_up": ref["up"]["vd"], "median_up": ref["up"]["median"],
            "vd_down": ref["down"]["vd"], "median_down": ref["down"]["median"],
        },
        "model": {"double_sweep": result["double_sweep"]},
    }


if __name__ == "__main__":
    selection = {"device": params.resolve_device({"preset": "paper"}), "sweep": params.PRESETS["paper"]["sweep"]}
    result = run_branches(selection)
    target = ROOT / "web/src/validation/data/reference-idvd.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(dumps(assemble(result)))
    print(f"Built {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)")
