"""JSON helpers (orjson): numpy arrays/scalars supported, NaN/±inf → null, canonical hashing form.

Used by the API process and the worker processes (no numba import here).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import orjson

_OPTS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS


def _default(o: Any) -> Any:
    """Fallback for objects orjson does not serialise natively (called recursively by orjson)."""
    if isinstance(o, np.ndarray):          # non-contiguous or unsupported dtype (float16, object, ...)
        return np.ascontiguousarray(o).tolist() if o.dtype.kind in "biuf" else o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def sanitize(obj: Any) -> Any:
    """Plain-Python copy with NaN/±inf → None (slow path; orjson already does this natively)."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return sanitize(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def dumps(obj: Any) -> bytes:
    """Serialise to JSON bytes. NaN/±inf become null (orjson behaviour for floats and numpy arrays)."""
    try:
        return orjson.dumps(obj, default=_default, option=_OPTS)
    except (orjson.JSONEncodeError, TypeError):
        return orjson.dumps(sanitize(obj), default=_default, option=_OPTS)


def loads(data: bytes | str) -> Any:
    return orjson.loads(data)


def _canon(obj: Any) -> Any:
    """Canonical form for hashing: ints → floats (so 2 and 2.0 hash alike), tuples → lists, numpy → python."""
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _canon(obj.tolist())
    if isinstance(obj, np.generic):
        return _canon(obj.item())
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):          # 2 and 2.0 hash alike; ints a float cannot hold exactly (seeds > 2**53) stay ints
        try:
            f = float(obj)
        except OverflowError:
            f = None
        if f is not None and int(f) == obj:
            return f
        return obj if -(1 << 63) <= obj < (1 << 64) else f"int:{obj}"   # orjson serialises 64-bit ints only
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def canonical(obj: Any) -> bytes:
    """Deterministic JSON bytes (sorted keys, ints as floats) for cache keys."""
    return orjson.dumps(_canon(obj), option=orjson.OPT_SORT_KEYS)


def sha256_hex(*parts: bytes | str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode() if isinstance(part, str) else part)
        h.update(b"\x00")
    return h.hexdigest()


Fragment = orjson.Fragment  # embed pre-serialised JSON bytes inside another document
