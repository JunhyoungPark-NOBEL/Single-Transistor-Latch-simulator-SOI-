"""Drop numba's on-disk caches under server/ whenever any server source changed.

numba re-compiles a cached ``@njit(cache=True)`` function only when the file that
*defines it* changes.  A cached caller in another file keeps the callee's code it
was compiled with: ``server/compute/circuit/element.py`` inlines
``server.simple_model.evaluate``, so after an update of ``simple_model.py`` the
circuit path would silently keep running the old Simple Model physics until
``element.py`` itself changed, and a freshly compiled function in the same
process can even link against that stale copy.  The caches therefore carry a
fingerprint of every ``server/**/*.py`` (tests excluded); when it moves, all
``*.nbi``/``*.nbc`` files below ``server/`` are deleted and rebuilt on demand.

``purge_if_stale()`` is called by the API process (``server.main``), the worker
import point (``server.engine_bridge``), the test session and ``scripts/warmup.py``
before numba loads anything from ``server/``.  The engine's own caches under
``engine/`` are untouched: engine code never calls server code.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

SERVER = Path(__file__).resolve().parent
STAMP = SERVER / "__pycache__" / "numba-sources.sha256"
CACHE_SUFFIXES = (".nbi", ".nbc")
log = logging.getLogger("stl.numba_cache")


def source_files(root: Path = SERVER) -> list[Path]:
    """Every server source that may define or inline a numba function (tests excluded)."""
    return sorted(p for p in root.rglob("*.py") if "tests" not in p.parts and "__pycache__" not in p.parts)


def fingerprint(root: Path = SERVER) -> str:
    h = hashlib.sha256()
    for p in source_files(root):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def cache_files(root: Path = SERVER) -> list[Path]:
    return [f for d in root.rglob("__pycache__") if d.is_dir() for f in d.iterdir() if f.suffix in CACHE_SUFFIXES]


def purge_if_stale(root: Path = SERVER, stamp: Path | None = None) -> int:
    """Delete stale caches; return the number of files removed (0 = caches were current)."""
    if os.environ.get("STL_KEEP_NUMBA_CACHE") == "1":
        return 0
    stamp = stamp or (root / "__pycache__" / "numba-sources.sha256")
    current = fingerprint(root)
    try:
        if stamp.read_text().strip() == current:
            return 0
    except OSError:
        pass
    removed = 0
    for f in cache_files(root):
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(current + "\n")
    except OSError as exc:            # read-only checkout: recompile every start, but keep running
        log.warning("numba cache stamp not writable (%s); caches will be rebuilt on each start", exc)
    if removed:
        log.info("server sources changed: removed %d numba cache file(s); they rebuild on first use", removed)
    return removed
