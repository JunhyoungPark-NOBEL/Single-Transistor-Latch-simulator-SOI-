"""STL web simulator backend.

Importing ``server`` (or any ``server.*`` module) pins numba's on-disk JIT cache to a directory stamped with the
sources numba compiles: ``server/__pycache__/numba/<stamp>``, where ``<stamp>`` is the first 12 hex digits of a
sha256 over ``server/**/*.py`` (tests excluded) and ``engine/**/*.py``.

Why: numba validates a cached kernel only against the modification time and size of the file that defines it, not
against the functions it calls from other modules. After a pull that changes, e.g., ``geometry_model.py`` alone, the
cached ``element.py``/``mna.py`` kernels would keep the old inlined code and silently return the old physics. A new
stamp is a new, empty cache directory, so every kernel is recompiled from the current sources.

* ``NUMBA_CACHE_DIR`` set in the environment wins (``os.environ.setdefault``).
* Older stamp directories next to the current one are removed (best effort).
* The directory lives in ``server/__pycache__``: ignored by git and by the Docker build context, rebuilt by
  ``scripts/warmup.py`` inside the image, and outside the result-cache volumes of the lab kit
  (``/app/server/.cache/results``) and the local kit (``/app/server/.cache``).
* Computed standalone (no ``server.jobs`` import) so that importing this package stays cheap and cycle-free.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
ROOT_DIR = SERVER_DIR.parent
ENGINE_DIR = ROOT_DIR / "engine"
NUMBA_CACHE_ROOT = SERVER_DIR / "__pycache__" / "numba"

_STAMP_RE = re.compile(r"^[0-9a-f]{12}$")
_SKIP_DIRS = {"__pycache__", "tests", "node_modules"}


def _sources(base: Path, skip_tests: bool) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        # prune caches, hidden folders (server/.cache holds thousands of result files) and, for server/, the tests
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d != "__pycache__" and not (skip_tests and d in _SKIP_DIRS))
        out.extend(Path(dirpath) / f for f in filenames if f.endswith(".py"))
    return sorted(out)


def source_stamp(server_dir: Path = SERVER_DIR, engine_dir: Path = ENGINE_DIR) -> str:
    """12 hex digits of sha256 over server/**/*.py (without tests) and engine/**/*.py (path + content)."""
    h = hashlib.sha256()
    for tag, base, skip_tests in (("server", server_dir, True), ("engine", engine_dir, False)):
        if not base.is_dir():
            continue
        for f in _sources(base, skip_tests):
            h.update(f"{tag}/{f.relative_to(base).as_posix()}".encode())
            h.update(b"\x00")
            h.update(f.read_bytes())
            h.update(b"\x00")
    return h.hexdigest()[:12]


def _prune(root: Path, keep: str) -> None:
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for d in entries:
        if d.name != keep and _STAMP_RE.match(d.name) and d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def _pin_numba_cache() -> None:
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    try:
        stamp = source_stamp()
    except OSError:
        return                       # unreadable tree: keep numba's default (next to the sources)
    target = NUMBA_CACHE_ROOT / stamp
    os.environ["NUMBA_CACHE_DIR"] = str(target)
    _prune(NUMBA_CACHE_ROOT, stamp)
    if "numba.core.config" in sys.modules:   # numba imported first: re-read the environment
        sys.modules["numba.core.config"].reload_config()


_pin_numba_cache()
