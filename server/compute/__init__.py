"""Compute-kind registry. Functions are imported lazily (inside worker processes)."""
from __future__ import annotations

import importlib
from typing import Callable

KINDS: dict[str, str] = {
    "performance_calibrate": "server.compute.performance:run_performance_calibrate",
    "branches": "server.compute.deterministic:run_branches",
    "simple_calibrate": "server.compute.simple_calibration:run_simple_calibrate",
    "charge_balance": "server.compute.deterministic:run_charge_balance",
    "vg_curve": "server.compute.deterministic:run_vg_curve",
    "hazard": "server.compute.stochastic:run_hazard",
    "sweep_mc": "server.compute.stochastic:run_sweep_mc",
    "vg_curve_stochastic": "server.compute.stochastic:run_vg_curve_stochastic",
    "circuit": "server.compute.circuit:run_circuit",
    "validation": "server.compute.validation:run_validation",
}


def resolve(kind: str) -> Callable:
    if kind not in KINDS:
        raise ValueError(f"unknown compute kind {kind!r}; known: {sorted(KINDS)}")
    module, func = KINDS[kind].split(":")
    return getattr(importlib.import_module(module), func)
