"""Server numba caches are dropped when any server source changes (stale inlined callees)."""
from __future__ import annotations

from server import numba_cache


def _tree(tmp_path):
    root = tmp_path / "server"
    (root / "compute" / "__pycache__").mkdir(parents=True)
    (root / "__pycache__").mkdir()
    (root / "tests").mkdir()
    (root / "simple_model.py").write_text("x = 1\n")
    (root / "compute" / "element.py").write_text("y = 2\n")
    (root / "tests" / "test_x.py").write_text("z = 3\n")            # tests never count
    for name in ("compute/__pycache__/element.f-1.py311.nbi", "compute/__pycache__/element.f-1.py311.1.nbc",
                 "__pycache__/simple_model.g-2.py311.nbi", "__pycache__/simple_model.g-2.py311.1.nbc",
                 "__pycache__/simple_model.cpython-311.pyc"):
        (root / name).write_bytes(b"cache")
    return root


def test_first_run_purges_and_stamps_then_unchanged_sources_keep_caches(tmp_path):
    root = _tree(tmp_path)
    assert numba_cache.purge_if_stale(root) == 4
    assert not numba_cache.cache_files(root)
    assert (root / "__pycache__" / "simple_model.cpython-311.pyc").exists()      # only numba files go
    assert (root / "__pycache__" / "numba-sources.sha256").read_text().strip() == numba_cache.fingerprint(root)
    (root / "__pycache__" / "simple_model.g-2.py311.nbi").write_bytes(b"fresh")
    assert numba_cache.purge_if_stale(root) == 0
    assert (root / "__pycache__" / "simple_model.g-2.py311.nbi").exists()


def test_changing_any_server_source_purges_again_but_tests_do_not(tmp_path):
    root = _tree(tmp_path)
    numba_cache.purge_if_stale(root)
    (root / "__pycache__" / "simple_model.g-2.py311.nbi").write_bytes(b"fresh")
    (root / "tests" / "test_x.py").write_text("z = 4\n")
    assert numba_cache.purge_if_stale(root) == 0
    (root / "simple_model.py").write_text("x = 2\n")
    assert numba_cache.purge_if_stale(root) == 1
    assert not (root / "__pycache__" / "simple_model.g-2.py311.nbi").exists()


def test_keep_switch_skips_the_purge(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setenv("STL_KEEP_NUMBA_CACHE", "1")
    assert numba_cache.purge_if_stale(root) == 0
    assert len(numba_cache.cache_files(root)) == 4


def test_real_server_tree_is_fingerprinted_without_tests():
    files = numba_cache.source_files()
    names = {p.name for p in files}
    assert "simple_model.py" in names and "element.py" in names
    assert not any("tests" in p.parts for p in files)
    assert len(numba_cache.fingerprint()) == 64
