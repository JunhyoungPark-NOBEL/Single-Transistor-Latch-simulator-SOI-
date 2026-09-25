"""The numba JIT cache is stamped with the sources it was compiled from (server/__init__.py)."""
from __future__ import annotations

import os
import subprocess
import sys

import server
from server import NUMBA_CACHE_ROOT, ROOT_DIR, source_stamp


def _probe(code: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_CACHE_DIR"}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT_DIR, env=env, capture_output=True, text=True,
                         timeout=120, check=True)
    return out.stdout.strip()


def test_numba_cache_dir_is_the_stamped_directory():
    expected = str(NUMBA_CACHE_ROOT / source_stamp())
    # package import first (every entry point: uvicorn server.main, workers, scripts/warmup.py, pytest)
    assert _probe("import server, numba; print(numba.config.CACHE_DIR)") == expected
    # numba imported before the package: the configuration is re-read
    assert _probe("import numba; import server; print(numba.config.CACHE_DIR)") == expected
    # an explicit NUMBA_CACHE_DIR wins
    if os.environ.get("NUMBA_CACHE_DIR"):
        import numba
        assert numba.config.CACHE_DIR == os.environ["NUMBA_CACHE_DIR"]


def test_stamp_follows_compiled_sources_only(tmp_path):
    srv, eng = tmp_path / "server", tmp_path / "engine"
    for d in (srv / "compute", srv / "tests", srv / ".cache" / "results", srv / "__pycache__", eng / "photo_extension"):
        d.mkdir(parents=True)
    (srv / "geometry_model.py").write_text("a = 1\n")
    (srv / "compute" / "element.py").write_text("b = 1\n")
    (eng / "photo_extension" / "photo_mean.py").write_text("c = 1\n")
    s0 = source_stamp(srv, eng)
    assert len(s0) == 12 and int(s0, 16) >= 0
    # tests, caches and non-Python files do not count
    (srv / "tests" / "test_x.py").write_text("x = 2\n")
    (srv / ".cache" / "results" / "junk.py").write_text("x = 3\n")
    (srv / "__pycache__" / "y.py").write_text("x = 4\n")
    (eng / "photo_extension" / "table.json").write_text("{}")
    assert source_stamp(srv, eng) == s0
    # any compiled source counts: a module that only calls into another one, the engine, a new file
    for f in (srv / "geometry_model.py", srv / "compute" / "element.py", eng / "photo_extension" / "photo_mean.py"):
        old = f.read_text()
        f.write_text(old + "# edit\n")
        assert source_stamp(srv, eng) != s0, f
        f.write_text(old)
    assert source_stamp(srv, eng) == s0
    (srv / "compute" / "basic.py").write_text("d = 1\n")
    assert source_stamp(srv, eng) != s0


def test_older_stamps_are_pruned(tmp_path):
    root = tmp_path / "numba"
    for name in ("0123456789ab", "ba9876543210", "keepme", "fedcba987654"):
        (root / name).mkdir(parents=True)
    (root / "0123456789ab" / "index.nbi").write_bytes(b"x")
    server._prune(root, "fedcba987654")
    assert sorted(p.name for p in root.iterdir()) == ["fedcba987654", "keepme"]
