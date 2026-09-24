"""HTTP API tests (in-process TestClient, real process pool with STL_WORKERS=2)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

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


# ---------------------------------------------------------------------------------------------
# hardening (review of the API / ops layer)
# ---------------------------------------------------------------------------------------------
_BIG = int("1" + "0" * 400)          # beyond float range: used to raise OverflowError → HTTP 500


def _deep(n: int) -> dict:
    x: list = []
    for _ in range(n):
        x = [x]
    return {"junk": x}


@pytest.mark.parametrize("kind,body", [
    ("folds", {"device": {"vg": _BIG}}),
    ("folds", {"junk": _BIG}),
    ("sweep_mc", {"stochastic": {"n_cycles": _BIG}}),
    ("folds", _deep(800)),                                   # used to be a RecursionError in deepcopy → 500
    ("folds", {"device": {"calib": [1, 2]}}),
    ("folds", {"device": {"light": "x"}}),
    ("folds", {"device": {"numerics": "x"}}),
    ("folds", {"device": {"preset": [1]}}),
    ("sweep_mc", {"stochastic": {"local_state": "x"}}),
    ("circuit", {"detect": "x"}),
    ("circuit", {"stochastic": {"local_state": 3}}),
    ("folds", {"junk": ["x"] * 6000}),                       # too many values
    ("folds", {"junk": "y" * 2000}),                         # long string
])
def test_malformed_payload_is_422_not_500(client, kind, body):
    r = client.post(f"/api/compute/{kind}", params={"wait": 0}, content=json.dumps(body),
                    headers={"content-type": "application/json"})
    assert r.status_code == 422, r.text
    assert isinstance(r.json()["detail"], str)


def test_nan_literal_rejected(client):
    # the stdlib JSON parser accepts NaN/Infinity; they must not reach the cache key or the workers
    for raw in ('{"junk": NaN}', '{"device": {"state": {"delta_phi_G0_V": Infinity}}}'):
        r = client.post("/api/compute/folds", params={"wait": 0}, content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 422, r.text


def test_request_errors_are_plain_detail_strings(client):
    for raw in ("[1, 2]", "{bad"):
        r = client.post("/api/compute/folds", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 422
        d = r.json()["detail"]
        assert isinstance(d, str) and "input" not in r.json()


def test_body_size_limit_413(client):
    from server.main import MAX_BODY_BYTES
    big = b'{"junk": "' + b"x" * (MAX_BODY_BYTES + 10) + b'"}'
    r = client.post("/api/compute/folds", content=big, headers={"content-type": "application/json"})
    assert r.status_code == 413 and "too large" in r.json()["detail"]

    def chunks():                                             # chunked upload: no Content-Length header
        yield b'{"junk": "'
        for _ in range(MAX_BODY_BYTES // 65536 + 2):
            yield b"x" * 65536
        yield b'"}'
    r = client.post("/api/compute/folds", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert client.get("/api/health").status_code == 200


def test_alias_query_overflow_422(client):
    assert client.get("/api/sweeps", params={"n": "inf", "wait": 0}).status_code == 422
    assert client.get("/api/folds", params={"grid": "1e400", "wait": 0}).status_code == 422


def test_static_and_api_fallbacks(client):
    r = client.get("/api")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")
    assert client.get("/%00").status_code == 404
    assert client.get("/a%00b.js").status_code == 404
    for path in ("/..%2f..%2fserver/params.py", "/%2e%2e/%2e%2e/server/params.py", "/assets/..%2f..%2f..%2fserver%2fparams.py"):
        r = client.get(path)
        assert "def build_p" not in r.text and r.status_code in (200, 404)


def test_canonical_big_ints():
    c = jsonutil.canonical
    assert c({"a": 2}) == c({"a": 2.0})
    assert c({"seed": 2 ** 60}) != c({"seed": 2 ** 60 + 1})           # used to collide (int → float)
    assert c({"seed": 10 ** 30}) != c({"seed": 10 ** 30 + 1})         # beyond 64 bit: no orjson error


def test_seed_kept_exact():
    from server import payloads
    p, _ = payloads.normalize("sweep_mc", {"stochastic": {"seed": 2 ** 60 + 1}})
    assert p["stochastic"]["seed"] == 2 ** 60 + 1


def _fake_pending(manager, client_id: str):
    from server.jobs import Job
    job = Job(id="fake" + __import__("uuid").uuid4().hex[:12], kind="folds", key="fake", client=client_id)
    job.status = "running"
    manager._register(job)
    return job


def test_long_poll_does_not_block_the_server(client):
    """40+ concurrent long-polls used to hold every thread of the pool: /api/health then hung for `wait` s."""
    from concurrent.futures import ThreadPoolExecutor

    from server.main import manager
    job = _fake_pending(manager, "other-client")
    try:
        with ThreadPoolExecutor(48) as ex:
            futs = [ex.submit(client.get, f"/api/jobs/{job.id}", params={"wait": 8}) for _ in range(48)]
            time.sleep(0.5)
            t0 = time.monotonic()
            assert client.get("/api/health").status_code == 200
            assert time.monotonic() - t0 < 3.0
            manager._finish(job, "cancelled")                 # releases the long-polls
            assert all(f.result(timeout=20).json()["status"] == "cancelled" for f in futs)
    finally:
        manager._finish(job, "cancelled")


def test_identical_jobs_cancel_independently(client, wait):
    payload = {"device": {"preset": "paper", "numerics": {"grid": 2001}}, "vg_min": -4.05, "vg_max": -0.75, "n": 61}
    a = client.post("/api/compute/vg_curve", params={"wait": 0}, json=payload).json()
    b = client.post("/api/compute/vg_curve", params={"wait": 0}, json=payload).json()
    assert a["job_id"] != b["job_id"]                        # separate handles on one computation
    assert client.delete(f"/api/jobs/{a['job_id']}").json()["status"] == "cancelled"
    time.sleep(0.5)
    assert client.get(f"/api/jobs/{b['job_id']}").json()["status"] in ("queued", "running", "done")
    assert client.delete(f"/api/jobs/{b['job_id']}").json()["status"] in ("cancelled", "done")
    c = client.post("/api/compute/vg_curve", params={"wait": 0}, json=payload).json()
    assert c["status"] in ("queued", "running", "done")      # a fresh run, not the cancelled one
    client.delete(f"/api/jobs/{c['job_id']}")


def test_admission_limits_429(client):
    from server.main import manager
    fakes = []
    old = manager.max_pending_client, manager.max_pending
    try:
        manager.max_pending_client = 1
        fakes.append(_fake_pending(manager, "testclient"))    # TestClient's client address
        r = client.post("/api/compute/folds", params={"wait": 0}, json={"device": {"preset": "paper", "vg": -2.61}})
        assert r.status_code == 429 and "Retry-After" in r.headers
        manager.max_pending_client, manager.max_pending = 100, len([j for j in manager.list() if j.status not in
                                                                    ("done", "error", "cancelled")])
        r = client.post("/api/compute/folds", params={"wait": 0}, json={"device": {"preset": "paper", "vg": -2.62}})
        assert r.status_code == 429 and "busy" in r.json()["detail"]
    finally:
        manager.max_pending_client, manager.max_pending = old
        for f in fakes:
            manager._finish(f, "cancelled")
    ok = client.post("/api/compute/folds", params={"wait": 60}, json={"device": {"preset": "paper", "vg": -2.61}})
    assert ok.status_code == 200 and ok.json()["status"] == "done"


def test_job_list_is_per_client(client):
    from server.main import manager
    other = _fake_pending(manager, "10.9.8.7")
    try:
        ids = {j["job_id"] for j in client.get("/api/jobs").json()}
        assert other.id not in ids
    finally:
        manager._finish(other, "cancelled")


def test_abandoned_jobs_are_cancelled():
    from server.jobs import JobManager
    import tempfile
    m = JobManager(workers=1, directory=Path(tempfile.mkdtemp(prefix="stl-test-")))
    job = _fake_pending(m, "c")
    m.abandon_s = 30.0
    m._reap_abandoned()
    assert job.status == "running"
    job.seen -= 60.0
    m._reap_abandoned()
    assert job.status == "cancelled" and "abandoned" in job.message


def test_worker_crash_is_retried_and_pool_recovers(client, wait):
    import os
    import signal

    from server.main import manager
    slow = {"device": {"preset": "paper", "numerics": {"grid": 2001}}, "vg_min": -4.0, "vg_max": -0.8, "n": 61}
    st = client.post("/api/compute/vg_curve", params={"wait": 0}, json=slow).json()
    jid = st["job_id"]
    deadline = time.monotonic() + 60
    while st["status"] != "running" and time.monotonic() < deadline:
        time.sleep(0.2)
        st = client.get(f"/api/jobs/{jid}").json()
    assert st["status"] == "running"
    for proc in list(manager._pool._processes.values()):     # simulate a segfault / OOM kill
        os.kill(proc.pid, signal.SIGKILL)
    time.sleep(2.0)
    st = client.get(f"/api/jobs/{jid}").json()
    assert st["status"] in ("queued", "running"), st         # retried on a fresh pool, not failed
    client.delete(f"/api/jobs/{jid}")
    ok = client.post("/api/compute/folds", params={"wait": 60}, json={"device": {"preset": "paper", "vg": -2.63}}).json()
    assert ok["status"] == "done", ok


def test_workers_exit_when_the_api_process_dies(tmp_path):
    """SIGKILL of the API process used to leave the Manager process and the workers running for ever."""
    import os
    import signal

    root = Path(__file__).resolve().parents[2]
    code = ("import sys, time, multiprocessing as mp; sys.path.insert(0, %r)\n"
            "from pathlib import Path\nfrom server.jobs import JobManager\n"
            "m = JobManager(workers=1, directory=Path(%r)); m.start(prewarm=True); time.sleep(4)\n"
            "print(' '.join(str(c.pid) for c in mp.active_children()), flush=True); time.sleep(120)\n"
            % (str(root), str(tmp_path)))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        pids = [int(x) for x in p.stdout.readline().split()]
        assert len(pids) >= 2                                  # Manager process + worker
    finally:
        p.send_signal(signal.SIGKILL)
        p.wait()

    def alive(pid: int) -> bool:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            return False
        return state != "Z"                                    # a zombie has exited (waiting for init to reap it)

    deadline = time.monotonic() + 15
    while any(alive(q) for q in pids) and time.monotonic() < deadline:
        time.sleep(0.2)
    left = [q for q in pids if alive(q)]
    for q in left:
        os.kill(q, signal.SIGKILL)
    assert not left, f"orphaned processes: {left}"


def test_result_cache_disk_budget_enforced_while_running(tmp_path):
    import os

    from server.jobs import ResultCache
    c = ResultCache(tmp_path, disk_max_bytes=60_000)
    for i in range(30):
        c.put(f"k{i:02d}", os.urandom(10_000))                 # incompressible
        size = sum(f.stat().st_size for f in tmp_path.glob("*.json.gz"))
        assert size <= 60_000 + 11_000
    assert c.get("k29") is not None
    c.clear_memory()
    assert c.get("k00") is None                                # oldest entries were pruned from disk


def test_prune_dir_oldest_first(tmp_path):
    import os

    from server.jobs import prune_dir
    for i in range(10):
        f = tmp_path / "fold" / f"{i}.json"
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(b"x" * 1000)
        os.utime(f, (1000 + i, 1000 + i))
    left = prune_dir(tmp_path, 4500, "*.json")
    assert left <= 4500
    assert sorted(p.name for p in (tmp_path / "fold").glob("*.json")) == ["6.json", "7.json", "8.json", "9.json"]


def test_engine_version_covers_engine_and_serialisation(tmp_path):
    from server.jobs import engine_version
    srv, eng = tmp_path / "server", tmp_path / "engine"
    (srv / "compute").mkdir(parents=True)
    for name in ("params.py", "engine_bridge.py", "jsonutil.py", "jobs.py", "compute/a.py"):
        (srv / name).write_text("x = 1\n")
    (eng / "photo_extension" / "photo_nodes").mkdir(parents=True)
    (eng / "__pycache__").mkdir()
    (eng / "stl_api.py").write_text("y = 1\n")
    (eng / "table.npz").write_bytes(b"\x00\x01")
    v0 = engine_version(srv, eng)
    (eng / "photo_extension" / "photo_nodes" / "node.npz").write_bytes(b"cache")   # run-time caches do not count
    (eng / "__pycache__" / "x.nbi").write_bytes(b"cache")
    assert engine_version(srv, eng) == v0
    for f, data in ((eng / "stl_api.py", b"y = 2\n"), (eng / "table.npz", b"\x00\x02"), (srv / "jsonutil.py", b"x = 2\n")):
        old = f.read_bytes()
        f.write_bytes(data)
        assert engine_version(srv, eng) != v0, f
        f.write_bytes(old)
    assert engine_version(srv, eng) == v0


def test_job_result_memory_budget(tmp_path):
    from server.jobs import Job, JobManager
    m = JobManager(workers=1, directory=tmp_path)
    m.result_budget = 1500
    jobs = []
    for i in range(3):
        j = Job(id=f"j{i}", kind="folds", key=f"key{i}")
        m._register(j)
        data = jsonutil.dumps({"i": i, "pad": "x" * 600})
        m.cache.put(j.key, data)
        m._finish(j, "done", result=data)
        jobs.append(j)
    assert jobs[0].result is None and jobs[2].result is not None     # older results are not pinned in memory
    assert m.status(jobs[0])["result"] is not None                    # ... but still served from the cache
    m.cache.clear_memory()
    (tmp_path / "key0.json.gz").unlink()
    st = m.status(jobs[0])
    assert st["status"] == "error" and "run it again" in st["error"]
