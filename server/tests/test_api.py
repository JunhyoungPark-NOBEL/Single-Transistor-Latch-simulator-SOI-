"""HTTP API tests (in-process TestClient, real process pool with STL_WORKERS=2)."""
from __future__ import annotations

import math
import subprocess
import sys
import time

import numpy as np
import pytest

from server import jsonutil

PAPER = {"device": {"preset": "paper"}}


def test_health(client):
    h = client.get("/api/health").json()
    assert h["ok"] is True
    assert h["workers"] == 2
    assert isinstance(h["version"], str) and len(h["version"]) == 12


def test_meta(client):
    m = client.get("/api/meta").json()
    assert {"paper", "photo", "custom"} <= set(m["presets"])
    assert set(m["kinds"]) == {"branches", "charge_balance", "vg_curve", "hazard", "sweep_mc",
                               "vg_curve_stochastic", "circuit", "validation"}
    assert m["caps"]["n_cycles"] == 2000 and m["caps"]["grid"] == [201, 2001]
    assert m["kinds_available"]["branches"] is True
    assert m["presets"]["paper"]["device"]["vg"] == -2.0


def test_compute_branches_wait(client):
    r = client.post("/api/compute/branches", params={"wait": 60}, json=PAPER)
    assert r.status_code == 200
    st = r.json()
    assert st["status"] == "done", st
    assert set(st) >= {"job_id", "kind", "status", "progress", "message", "cached", "elapsed_s", "result"}
    res = st["result"]
    assert res["latch"] is True
    assert abs(res["folds"]["V_LU"] - 3.7037) < 1e-3 and abs(res["folds"]["V_LD"] - 2.5979) < 1e-3
    for part in ("HRS", "unstable", "LRS", "full"):
        assert set(res[part]) == {"vd", "id", "u", "r", "comp"}
        assert len(res[part]["vd"]) == len(res[part]["comp"]["ii_total"])
    assert set(res["double_sweep"]) == {"up", "down"}
    assert isinstance(res["warnings"], list) and res["runtime_s"] > 0
    assert len(res["p"]) == 26


def test_polling_and_progress(client, wait):
    payload = {"device": {"preset": "paper", "vg": -1.9}, "vd": 3.3}
    st = client.post("/api/compute/charge_balance", params={"wait": 0}, json=payload).json()
    assert st["status"] in ("queued", "running", "done")
    final = wait(client, st["job_id"], 120)
    assert final["status"] == "done", final
    assert final["progress"] == 1.0
    assert [x["kind"] for x in final["result"]["roots"]] == ["stable", "unstable", "stable"]


def test_cache_hit(client):
    payload = {"device": {"preset": "paper", "vg": -2.05, "numerics": {"grid": 401}}}
    a = client.post("/api/compute/folds", params={"wait": 60}, json=payload).json()
    assert a["status"] == "done" and a["cached"] is False
    # same payload with ints/floats written differently → same cache key
    payload2 = {"device": {"preset": "paper", "vg": -2.05, "numerics": {"grid": 401.0}}}
    b = client.post("/api/compute/folds", params={"wait": 60}, json=payload2).json()
    assert b["status"] == "done" and b["cached"] is True
    assert b["result"]["folds"] == a["result"]["folds"]
    assert b["job_id"] != a["job_id"]


def test_cancel(client, wait):
    payload = {"device": {"preset": "paper", "numerics": {"grid": 2001}}, "vg_min": -4.1, "vg_max": -0.7, "n": 61}
    st = client.post("/api/compute/vg_curve", params={"wait": 0}, json=payload).json()
    jid = st["job_id"]
    deadline = time.monotonic() + 60
    while st["status"] != "running" and time.monotonic() < deadline:
        time.sleep(0.2)
        st = client.get(f"/api/jobs/{jid}").json()
    assert st["status"] == "running"
    c = client.delete(f"/api/jobs/{jid}")
    assert c.status_code == 200 and c.json()["status"] == "cancelled"
    assert client.get(f"/api/jobs/{jid}").json()["status"] == "cancelled"
    # the worker stops at its next progress call and the pool keeps serving
    ok = client.post("/api/compute/folds", params={"wait": 60}, json={"device": {"preset": "paper", "vg": -2.2}}).json()
    assert ok["status"] == "done"
    # resubmitting the cancelled payload starts a fresh job (not the cancelled one)
    again = client.post("/api/compute/vg_curve", params={"wait": 0}, json=payload).json()
    assert again["job_id"] != jid
    client.delete(f"/api/jobs/{again['job_id']}")


def test_unknown_kind_and_job(client):
    assert client.post("/api/compute/nope", json={}).status_code == 404
    assert client.get("/api/jobs/doesnotexist").status_code == 404
    assert client.delete("/api/jobs/doesnotexist").status_code == 404
    assert client.get("/api/unknown").status_code == 404


@pytest.mark.parametrize("body", [
    {"device": {"preset": "bogus"}},
    {"device": {"vg": "abc"}},
    {"device": {"light": {"mode": "laser"}}},
    {"sweep": {"vd_max_V": -1}},
])
def test_invalid_payload_422(client, body):
    r = client.post("/api/compute/branches", json=body)
    assert r.status_code == 422, r.text
    assert "detail" in r.json()


def test_value_error_in_job_is_error_status(client):
    # u_min > u_max passes the API-side validation; the compute function raises ValueError in the worker
    st = client.post("/api/compute/charge_balance", params={"wait": 60},
                     json={"device": {"preset": "paper"}, "vd": 3.0, "u_min": 0.9, "u_max": 0.5}).json()
    assert st["status"] == "error" and "u_min" in st["error"]


def test_clamping_warnings(client):
    st = client.post("/api/compute/folds", params={"wait": 60},
                     json={"device": {"preset": "paper", "numerics": {"grid": 99999}}}).json()
    assert st["status"] == "done"
    assert any("clamped" in w for w in st["result"]["warnings"])
    assert st["result"]["grid"] == 2001


def test_nan_serialisation(client):
    st = client.post("/api/compute/vg_curve", params={"wait": 60},
                     json={"device": {"preset": "paper"}, "vg_min": -0.8, "vg_max": -0.5, "n": 2, "refine": False}).json()
    assert st["status"] == "done"
    assert st["result"]["V_LU"] == [None, None]
    assert st["result"]["latch"] == [False, False]
    raw = jsonutil.dumps({"a": np.array([1.0, np.nan, np.inf]), "b": float("-inf"), "c": np.float64("nan")})
    assert jsonutil.loads(raw) == {"a": [1.0, None, None], "b": None, "c": None}


def test_data_measured(client):
    r = client.get("/api/data/measured")
    assert r.status_code == 200
    d = r.json()
    ph = d["photo"]
    assert len(ph["V_LU"]) == 8 and len(ph["V_LU"][0]) == 400
    assert abs(ph["conditions"][0]["stats"]["mean"] - 3.806) < 1e-3
    assert abs(ph["conditions"][0]["stats"]["sd"] * 1e3 - 173.2) < 0.1
    assert len(d["light_iv"]["id"]) == 6 and len(d["light_iv"]["vd"]) == 101
    assert len(d["idvg_dark"]["id"]) == 2 and len(d["idvg_dark"]["vg"]) == 161
    pp = d["paper_idvd"]
    assert len(pp["V_LU"]) == 100 and len(pp["up"]["median"]) == 401 and len(pp["up"]["samples"]) == 10
    assert abs(pp["stats"]["LU"]["sd"] * 1e3 - 123.1) < 0.1


def test_data_design_map(client):
    for path in ("/api/data/design_map", "/api/design_map"):
        d = client.get(path).json()
        assert len(d["arrays"]["sigma_VLU_mV"]) == 61 and len(d["arrays"]["sigma_VLU_mV"][0]) == 81
        assert len(d["arrays"]["length_nm"]) == 81
        assert math.isclose(d["scalars"]["Nt_cm2"], 1e12)


def test_aliases(client):
    r = client.get("/api/folds", params={"vg": -1.8, "wait": 60}).json()
    assert r["status"] == "done" and abs(r["result"]["folds"]["V_LU"] - 3.8644) < 1e-3
    r = client.post("/api/branches", params={"wait": 60}, json=PAPER).json()
    assert r["status"] == "done" and r["kind"] == "branches"
    r = client.get("/api/vg_curve", params={"vg_min": -0.8, "vg_max": -0.5, "n": 2, "wait": 60}).json()
    assert r["kind"] == "vg_curve"


def test_frontend_fallback(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


def test_api_process_never_imports_numba():
    code = ("import sys; import server.main; from server.compute import data; data.measured(); data.design_map();"
            "import server.jobs, server.payloads; server.payloads.normalize('sweep_mc', {});"
            "assert 'numba' not in sys.modules and 'stl_api' not in sys.modules, 'engine imported'")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr


@pytest.mark.slow
def test_validation_fast_via_api(client, wait):
    st = client.post("/api/compute/validation", params={"wait": 0}, json={"level": "fast"}).json()
    final = wait(client, st["job_id"], 300)
    assert final["status"] == "done", final
    checks = final["result"]["checks"]
    assert len(checks) >= 9
    failed = [c for c in checks if c["pass"] is not True]
    assert not failed, failed
    assert all("seconds" in c for c in checks)
