"""Shared pytest configuration for the server tests.

* Environment for the in-process API: 2 workers (shared 4-CPU machine), a throw-away result cache.
* Fixtures: `client` (session-scoped FastAPI TestClient with the process pool running), `progress`
  (a no-op progress callback), `wait_job(client, job_id, timeout)` helper.
* Marker `slow`: deselect with `-m "not slow"`.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("STL_WORKERS", "2")
os.environ.setdefault("STL_CACHE_DIR", tempfile.mkdtemp(prefix="stl-test-cache-"))
os.environ.setdefault("STL_PREWARM", "1")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running test (deselect with -m 'not slow')")


@pytest.fixture(scope="session")
def client():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    from server.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def progress():
    from server.progress import null_progress
    return null_progress


def wait_job(client, job_id: str, timeout: float = 120.0) -> dict:
    """Poll GET /api/jobs/{id} (long-poll 1 s) until the job is final."""
    deadline = time.monotonic() + timeout
    while True:
        st = client.get(f"/api/jobs/{job_id}", params={"wait": 1.0}).json()
        if st["status"] in ("done", "error", "cancelled") or time.monotonic() > deadline:
            return st


@pytest.fixture
def wait():
    return wait_job
