"""Payload validation / clamping for every compute kind (API process; pure Python, no numba).

`normalize(kind, payload)` returns `(normalized_payload, warnings)`:
  * the `device` block is resolved against its preset (params.resolve_device), numeric fields are checked;
  * `sweep` (and, for the MC kinds, `stochastic`) are filled from the preset defaults;
  * values beyond the contract caps (docs/WEB_CONTRACT.md §1) are clamped and a warning is added
    (the worker prepends these warnings to the result's `warnings`);
  * malformed input raises ValueError (→ HTTP 422).
Unknown keys are passed through untouched (compute modules own their payload details).
"""
from __future__ import annotations

import copy
import math
from typing import Any

from server import params

CAPS: dict[str, Any] = {
    "grid": [201, 2001],
    "n_cycles": 2000,
    "vg_curve_points": 61,
    "fold_nodes": 61,
    "hazard_nodes": 9,
    "n_traces": 50,
    "circuit_max_steps": 2_000_000,
    "circuit_n_runs": 200,
    "vd_max_V": 8.0,
    "sweep_points_per_direction": 2001,
}

DEVICE_KINDS = {"branches", "folds", "charge_balance", "vg_curve", "hazard", "sweep_mc", "vg_curve_stochastic", "circuit"}
SWEEP_KINDS = {"branches", "hazard", "sweep_mc", "vg_curve_stochastic"}
STOCHASTIC_KINDS = {"sweep_mc", "vg_curve_stochastic"}
VG_RANGE_KINDS = {"vg_curve", "vg_curve_stochastic"}

# Structural limits of a request body (real payloads: < 200 values, depth <= 4, strings < 40 chars).
TREE_LIMITS = {"depth": 12, "nodes": 5000, "string": 1000}


def check_tree(payload: Any) -> None:
    """Reject bodies that are expensive to copy/hash/pickle or that would crash later: nesting deeper than
    TREE_LIMITS["depth"], more than TREE_LIMITS["nodes"] values, strings/keys longer than TREE_LIMITS["string"],
    non-finite numbers (NaN/Infinity are accepted by the stdlib JSON parser) and integers beyond float range.
    Iterative, so it cannot hit the recursion limit itself."""
    stack: list[tuple[Any, str, int]] = [(payload, "body", 0)]
    nodes = 0
    while stack:
        obj, path, depth = stack.pop()
        nodes += 1
        if nodes > TREE_LIMITS["nodes"]:
            raise ValueError(f"request body too large (more than {TREE_LIMITS['nodes']} values)")
        if isinstance(obj, (dict, list, tuple)):
            if depth >= TREE_LIMITS["depth"]:
                raise ValueError(f"request body nested too deeply at {path} (max depth {TREE_LIMITS['depth']})")
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if not isinstance(k, str) or len(k) > TREE_LIMITS["string"]:
                        raise ValueError(f"invalid key in {path}")
                    stack.append((v, f"{path}.{k}", depth + 1))
            else:
                stack.extend((v, f"{path}[{i}]", depth + 1) for i, v in enumerate(obj))
        elif isinstance(obj, str):
            if len(obj) > TREE_LIMITS["string"]:
                raise ValueError(f"{path}: string longer than {TREE_LIMITS['string']} characters")
        elif isinstance(obj, bool) or obj is None:
            pass
        elif isinstance(obj, (int, float)):
            try:
                ok = math.isfinite(float(obj))
            except OverflowError:
                ok = False
            if not ok:
                raise ValueError(f"{path} must be a finite number")
        else:
            raise ValueError(f"{path}: unsupported value type {type(obj).__name__}")


def _obj(block: dict, key: str, name: str) -> None:
    """A nested block must be a JSON object (or absent / null)."""
    if block.get(key) is not None and not isinstance(block[key], dict):
        raise ValueError(f"{name} must be an object")


def _num(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number (got {str(value)[:40]!r})") from None
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def _int(value: Any, name: str) -> int:
    """Integer field (e.g. a seed): exact for Python ints (no float round-trip), else via _num."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(_num(value, name))


def _clamp(block: dict, key: str, lo: float | None, hi: float | None, name: str, warnings: list[str],
           integer: bool = False) -> None:
    if key not in block or block[key] is None:
        return
    x = _num(block[key], name)
    y = x
    if lo is not None and y < lo:
        y = lo
    if hi is not None and y > hi:
        y = hi
    if integer:
        y = int(round(y))
    if y != x:
        warnings.append(f"{name} = {x:g} clamped to {y:g}")
    block[key] = y


def _check_numeric_tree(block: Any, prefix: str) -> None:
    if isinstance(block, dict):
        for k, v in block.items():
            _check_numeric_tree(v, f"{prefix}.{k}")
    elif isinstance(block, (int, float)) and not isinstance(block, bool):
        if not math.isfinite(float(block)):
            raise ValueError(f"{prefix} must be finite")


def normalize_device(device: Any, warnings: list[str]) -> dict:
    if device is not None and not isinstance(device, dict):
        raise ValueError("device must be an object")
    if device:
        if device.get("preset") is not None and not isinstance(device["preset"], str):
            raise ValueError("device.preset must be a string")
        for section in ("light", "calib", "ext", "state", "numerics"):
            _obj(device, section, f"device.{section}")
    d = params.resolve_device(device)          # raises ValueError for an unknown preset
    d["vg"] = _num(d["vg"], "device.vg")
    if not -10.0 <= d["vg"] <= 10.0:
        raise ValueError("device.vg must be within [-10, 10] V")
    light = d.get("light") or {}
    if light.get("mode") not in ("iph", "power"):
        raise ValueError("device.light.mode must be 'iph' or 'power'")
    for k in ("iph_pA", "power_mW", "responsivity_pA_per_mW"):
        light[k] = _num(light.get(k, 0.0), f"device.light.{k}")
        if light[k] < 0:
            raise ValueError(f"device.light.{k} must be >= 0")
    for section in ("calib", "ext", "state"):
        for k, v in list(d[section].items()):
            d[section][k] = _num(v, f"device.{section}.{k}")
    d["ext"]["loc_carriers"] = int(round(d["ext"]["loc_carriers"]))
    if d["ext"]["loc_carriers"] not in (0, 1, 2):
        raise ValueError("device.ext.loc_carriers must be 0, 1 or 2")
    if d["ext"]["seed_S"] == 0:
        raise ValueError("device.ext.seed_S must be non-zero")
    numerics = d.setdefault("numerics", {})
    numerics.setdefault("grid", 601)
    lo, hi = CAPS["grid"]
    _clamp(numerics, "grid", lo, hi, "device.numerics.grid", warnings, integer=True)
    _check_numeric_tree(d, "device")
    return d


def normalize_sweep(preset: str, sweep: Any, warnings: list[str]) -> dict:
    if sweep is not None and not isinstance(sweep, dict):
        raise ValueError("sweep must be an object")
    s = params.resolve_section(preset, "sweep", sweep)
    for k in ("vd_max_V", "rate_V_per_s", "dv_V"):
        s[k] = _num(s[k], f"sweep.{k}")
    if s["vd_max_V"] <= 0 or s["rate_V_per_s"] <= 0 or s["dv_V"] <= 0:
        raise ValueError("sweep.vd_max_V, sweep.rate_V_per_s and sweep.dv_V must be positive")
    _clamp(s, "vd_max_V", 0.1, CAPS["vd_max_V"], "sweep.vd_max_V", warnings)
    _clamp(s, "dv_V", 1e-4, 0.1, "sweep.dv_V", warnings)
    _clamp(s, "rate_V_per_s", 1e-4, 1e8, "sweep.rate_V_per_s", warnings)
    return s


def normalize_stochastic(preset: str, sto: Any, warnings: list[str]) -> dict:
    if sto is not None and not isinstance(sto, dict):
        raise ValueError("stochastic must be an object")
    if sto:
        _obj(sto, "local_state", "stochastic.local_state")
    s = params.resolve_section(preset, "stochastic", sto)
    _clamp(s, "n_cycles", 1, CAPS["n_cycles"], "stochastic.n_cycles", warnings, integer=True)
    _clamp(s, "fold_nodes", 3, CAPS["fold_nodes"], "stochastic.fold_nodes", warnings, integer=True)
    _clamp(s, "hazard_nodes", 1, CAPS["hazard_nodes"], "stochastic.hazard_nodes", warnings, integer=True)
    _clamp(s, "n_traces", 0, CAPS["n_traces"], "stochastic.n_traces", warnings, integer=True)
    if "seed" in s:
        s["seed"] = _int(s["seed"], "stochastic.seed")
    ls = s.get("local_state") or {}
    if ls.get("mode", "none") not in ("none", "frozen", "evolving"):
        raise ValueError("stochastic.local_state.mode must be none | frozen | evolving")
    if ls.get("action", "gidl") not in ("gidl", "local_avalanche", "junction", "multiplication"):
        raise ValueError("stochastic.local_state.action must be gidl | local_avalanche | junction | multiplication")
    for k in ("sigma", "tau_s", "sigma_E_V", "tau_E_s"):
        if k in ls:
            ls[k] = _num(ls[k], f"stochastic.local_state.{k}")
            if ls[k] < 0:
                raise ValueError(f"stochastic.local_state.{k} must be >= 0")
    if s.get("engine", "auto") not in ("auto", "general", "calibrated_lookup"):
        raise ValueError("stochastic.engine must be auto | general | calibrated_lookup")
    return s


def normalize(kind: str, payload: Any) -> tuple[dict, list[str]]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("the request body must be a JSON object")
    check_tree(payload)                        # before deepcopy/hash/pickle (depth, size, NaN, huge ints)
    p = copy.deepcopy(payload)
    warnings: list[str] = []
    if kind in DEVICE_KINDS:
        p["device"] = normalize_device(p.get("device"), warnings)
        preset = p["device"]["preset"]
    else:
        preset = "paper"
    if kind in SWEEP_KINDS:
        p["sweep"] = normalize_sweep(preset, p.get("sweep"), warnings)
    if kind in STOCHASTIC_KINDS:
        p["stochastic"] = normalize_stochastic(preset, p.get("stochastic"), warnings)
    if kind == "vg_curve":   # defaults of deterministic.run_vg_curve; vg_curve_stochastic keeps its own defaults
        p.setdefault("vg_min", -4.2)
        p.setdefault("vg_max", -0.6)
        p.setdefault("n", 37)
    if kind in VG_RANGE_KINDS:
        for k in ("vg_min", "vg_max"):
            if k in p:
                p[k] = _num(p[k], k)
                if not -10.0 <= p[k] <= 10.0:
                    raise ValueError("the V_G range must lie within [-10, 10] V")
        if "vg_min" in p and "vg_max" in p and not p["vg_min"] < p["vg_max"]:
            raise ValueError("vg_min must be smaller than vg_max")
        _clamp(p, "n", 2, CAPS["vg_curve_points"], "n", warnings, integer=True)
    if kind == "charge_balance":
        p.setdefault("vd", 3.2)
        _clamp(p, "vd", 0.05, CAPS["vd_max_V"], "vd", warnings)
        if "n_u" in p:
            _clamp(p, "n_u", 21, 2001, "n_u", warnings, integer=True)
    if kind == "hazard":
        for k in ("dg", "de"):
            if k in p and p[k] is not None:
                p[k] = _num(p[k], k)
    if kind == "circuit":
        for block in ("bench_params", "solver", "stochastic", "detect"):
            _obj(p, block, block)
        if p.get("bench") is not None and not isinstance(p["bench"], str):
            raise ValueError("bench must be a string")
        if p.get("mode") is not None and not isinstance(p["mode"], str):
            raise ValueError("mode must be a string")
        solver = p.get("solver")
        if solver is not None:
            _clamp(solver, "max_steps", 1, CAPS["circuit_max_steps"], "solver.max_steps", warnings, integer=True)
        sto = p.get("stochastic")
        if sto is not None:
            _obj(sto, "local_state", "stochastic.local_state")
            _clamp(sto, "n_runs", 1, CAPS["circuit_n_runs"], "stochastic.n_runs", warnings, integer=True)
    if kind == "validation":
        level = p.setdefault("level", "fast")
        if level not in ("fast", "full"):
            raise ValueError("validation level must be 'fast' or 'full'")
    return p, warnings
