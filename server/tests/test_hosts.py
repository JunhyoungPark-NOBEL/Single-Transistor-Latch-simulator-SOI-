"""Optional Host-header allow-list (STL_ALLOWED_HOSTS, server/main.py): off by default; when set (the local installer
kit sets "127.0.0.1,localhost") any other Host header gets 400, so a DNS-rebinding page cannot read the API.
The setting is read at import time, so each case runs in a fresh interpreter."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROBE = r"""
import json, warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient
from server.main import app
c = TestClient(app)                      # no lifespan: /api/meta needs no worker pool
out = {}
for host in ["127.0.0.1:8000", "localhost:8123", "127.0.0.1", "rebind.attacker.example", "rebind.attacker.example:8000"]:
    out[host] = c.get("/api/meta", headers={"host": host}).status_code
print(json.dumps(out))
"""


def _probe(allowed: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "STL_ALLOWED_HOSTS"}
    if allowed is not None:
        env["STL_ALLOWED_HOSTS"] = allowed
    res = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_hosts_unrestricted_by_default():
    assert set(_probe(None).values()) == {200}


def test_allowed_hosts_rejects_other_host_headers():
    got = _probe("127.0.0.1, localhost")
    assert got["127.0.0.1:8000"] == 200 and got["localhost:8123"] == 200 and got["127.0.0.1"] == 200
    assert got["rebind.attacker.example"] == 400 and got["rebind.attacker.example:8000"] == 400
