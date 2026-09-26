"""End-to-end HTTP smoke: a real Uvicorn process, authenticated API, and spawned numerical workers.

No TestClient, recorded curves, mock route, or engine import in the API process is used here.
"""
from __future__ import annotations

import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

import httpx


def test_live_http_solver_and_remote_session(tmp_path):
    root = Path(__file__).resolve().parents[2]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    password = secrets.token_urlsafe(24)
    origin = "http://127.0.0.1:8000"
    env = dict(os.environ, STL_WORKERS="1", STL_PREWARM="1", STL_ACCESS_PASSWORD=password,
               STL_REQUIRE_PASSWORD="1", STL_CORS_ORIGINS=origin, STL_CACHE_DIR=str(tmp_path / "cache"))
    log_path = tmp_path / "api.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=root, env=env, stdout=log, stderr=log,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=65, trust_env=False) as client:
                deadline = time.monotonic() + 60
                while True:
                    assert process.poll() is None, log_path.read_text()
                    try:
                        health = client.get("/api/health")
                        if health.status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    assert time.monotonic() < deadline, "API did not become ready"
                    time.sleep(.1)
                assert health.json()["product"] == "STL simulator"
                assert health.json()["auth_required"] is True
                assert client.get("/api/meta").status_code == 401
                preflight = client.options("/api/session", headers={"Origin": origin,
                    "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
                assert preflight.status_code == 200
                login = client.post("/api/session", json={"password": password}, headers={"Origin": origin})
                assert login.status_code == 200
                assert login.headers["access-control-allow-origin"] == origin
                token = login.json()["token"]
                client.headers.update({"Authorization": f"Bearer {token}", "Origin": origin})
                assert client.get("/api/meta").json()["kinds_available"]["branches"]

                def run(payload):
                    response = client.post("/api/compute/branches?wait=60", json=payload)
                    assert response.status_code == 200, response.text
                    status = response.json()
                    deadline = time.monotonic() + 120
                    while status["status"] in ("queued", "running") and time.monotonic() < deadline:
                        status = client.get(f"/api/jobs/{status['job_id']}?wait=2").json()
                    assert status["status"] == "done", status
                    assert status["cached"] is False
                    return status["result"]

                reference = run({"device": {"preset": "paper", "numerics": {"grid": 401}}})
                assert abs(reference["folds"]["V_LU"] - 3.7037) < .001
                assert abs(reference["folds"]["V_LD"] - 2.5979) < .001
                shorter = run({"device": {"preset": "paper", "geometry": {"Lg_nm": 400},
                                           "numerics": {"grid": 401}}})
                assert shorter["geometry"]["Lg_nm"] == 400
                assert shorter["folds"]["V_LU"] < reference["folds"]["V_LU"]
                assert shorter["folds"]["V_LD"] < reference["folds"]["V_LD"]
                slow = client.post("/api/compute/vg_curve?wait=0", json={
                    "device": {"preset": "paper", "numerics": {"grid": 2001}},
                    "vg_min": -4.1, "vg_max": -.7, "n": 61}).json()
                cancelled = client.delete(f"/api/jobs/{slow['job_id']}").json()
                assert cancelled["status"] == "cancelled"
                assert client.get(f"/api/jobs/{slow['job_id']}").json()["status"] == "cancelled"
                assert client.delete("/api/session").status_code == 200
                assert client.get("/api/meta").status_code == 401
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    assert password not in log_path.read_text()
