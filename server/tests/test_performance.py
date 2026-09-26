"""Timing-estimator behavior, independent of fabricated benchmark performance."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest

from server import jsonutil, payloads
from server.jobs import Job, JobManager, _Run
from server.performance import PerformanceStore, describe, schedule


def write_catalog(path):
    cases = []
    for model, value in (("detailed", .4), ("simple", .04)):
        p, _ = payloads.normalize("branches", {"device": {"model": model, "vg": -3}})
        _, family, features = describe("branches", p)
        cases.append(dict(id=model, model=model, family=family, kind="branches", payload=p,
                          features=features, supported=True, timings={"median_s": value}))
    path.write_text(json.dumps({"schema_version": 1, "cases": cases}))
    return cases


@pytest.fixture
def store(tmp_path):
    catalog = tmp_path / "catalog.json"
    write_catalog(catalog)
    return PerformanceStore(tmp_path / "host", "test-engine", 1, 1, catalog)


def device(model="detailed", **options):
    p, _ = payloads.normalize("branches", {"device": {"model": model, **options}})
    return p


def test_first_call_is_not_a_warm_sample_and_models_are_isolated(store):
    p = device()
    store.record("branches", p, "a", dict(first_for_family=True, compute_s=40, worker_pid=1))
    assert not store.snapshot()["observations"]
    store.record("branches", p, "a", dict(first_for_family=False, compute_s=.1, worker_pid=1))
    estimate = store.estimate_one("branches", p, "a")
    assert estimate["source"] == "observed"
    assert estimate["estimate"]["seconds"] == .1
    assert not estimate["setup_unknown"]
    simple = store.estimate_one("branches", device("simple"), "b")
    assert simple["source"] == "reference"
    assert simple["setup_unknown"]


def test_calibration_is_family_and_model_specific(store):
    sample = dict(case_id="detailed", model="detailed", family="idvd", warm_s=.2)
    store.record("performance_calibrate", {}, "cal", {"worker_pid": 5}, {"samples": [sample]})
    result = store.estimate_one("branches", device(vg=-3), "fresh")
    assert result["source"] == "calibrated"
    assert result["estimate"]["seconds"] == pytest.approx(.2)
    assert store.estimate_one("branches", device("simple"), "other")["source"] == "reference"
    new = PerformanceStore(store.directory, "new-engine", 1, 1, store.catalog_path)
    assert new.snapshot()["calibration"] is None
    assert new.host["id"] == store.host["id"]
    assert new.host["instance_id"] != store.host["instance_id"]


def test_old_calibration_expires(store):
    sample = dict(case_id="detailed", model="detailed", family="idvd", warm_s=.2)
    store.record("performance_calibrate", {}, "cal", {"worker_pid": 5}, {"samples": [sample]})
    saved = json.loads(store.path.read_text())
    saved["calibration"]["measured_at"] = 0
    store.path.write_text(json.dumps(saved))
    new = PerformanceStore(store.directory, "test-engine", 1, 1, store.catalog_path)
    assert new.snapshot()["calibration"] is None


def item(key, duration, deps=(), **flags):
    return dict(key=key, estimate={"seconds": duration}, depends_on=list(deps), **flags)


def test_dependency_parallel_and_cached_scheduling():
    group = [item("a", 2), item("b", 3), item("c", 4, ["a"])]
    assert schedule(group, 2) == 6
    assert schedule(group, 1) == 9
    assert schedule([item("cache", .01, cached=True)], 1, [10]) == .01
    joined = item("joined", 1, joined=True, inflight_finish={"seconds": 11})
    assert schedule([joined], 1) == 1
    assert schedule([joined], 1, [11]) == 11
    with pytest.raises(ValueError):
        schedule([item("a", 1, ["b"]), item("b", 1, ["a"])], 1)


def test_signed_bias_changes_uncertainty(store):
    from server.performance import _distance
    assert _distance({"vg_V": -3, "vbg_V": 0}, {"vg_V": -1, "vbg_V": 1}) > 1


def test_stochastic_preparation_tracks_conditions_not_seed_or_cycle_count(store):
    from server.performance import preparation_key
    base, _ = payloads.normalize("sweep_mc", {"stochastic": {"seed": 1, "n_cycles": 10}})
    more = copy.deepcopy(base)
    more["stochastic"].update(seed=42, n_cycles=100)
    changed = copy.deepcopy(base)
    changed["device"]["vg"] -= .1
    assert preparation_key("sweep_mc", base) == preparation_key("sweep_mc", more)
    assert preparation_key("sweep_mc", base) != preparation_key("sweep_mc", changed)
    store.record("sweep_mc", base, "a", dict(worker_pid=1, compute_s=4, first_for_family=False,
                                             first_preparation=True))
    assert store.snapshot()["observations"] == []
    assert not store.estimate_one("sweep_mc", more, "b")["setup_unknown"]
    assert store.estimate_one("sweep_mc", changed, "c")["setup_unknown"]


def test_estimate_exact_cache_and_unsupported_simple(tmp_path):
    manager = JobManager(workers=1, directory=tmp_path / "cache")
    catalog = tmp_path / "table.json"
    write_catalog(catalog)
    manager.performance.catalog_path = catalog
    raw = {"device": {"model": "simple"}}
    p, warns = payloads.normalize("branches", raw)
    key = manager.cache_key("branches", p, warns)
    manager.cache.put(key, jsonutil.dumps({"runtime_s": 999}))
    result = manager.estimate({"jobs": [{"kind": "branches", "payload": raw}]})
    assert result["cached"] and result["total"]["seconds"] == 0
    assert not result["setup"]["unknown"]
    assert manager.performance.snapshot()["observations"] == []
    result = manager.estimate({"jobs": [{"kind": "sweep_mc", "payload": raw}]})
    assert not result["supported"] and result["total"]["seconds"] is None
    with pytest.raises(ValueError):
        manager.estimate({"jobs": [{"key": "x", "depends_on": ["x"], "kind": "branches", "payload": raw}]})


def test_repeated_calibration_never_hits_result_cache(tmp_path, monkeypatch):
    manager = JobManager(workers=1, directory=tmp_path / "cache")
    calls = []
    monkeypatch.setattr(manager.cache, "get", lambda key: pytest.fail("calibration must not read result cache"))
    monkeypatch.setattr(manager.cache, "put", lambda *args: pytest.fail("calibration must not write result cache"))
    def dispatch(run):
        calls.append(run.id)
        manager._complete(run, ("ok", jsonutil.dumps({"samples": []}), {"worker_pid": 1}))
    monkeypatch.setattr(manager, "_dispatch", dispatch)
    assert manager.submit("performance_calibrate", {}).status == "done"
    assert manager.submit("performance_calibrate", {}).status == "done"
    assert len(calls) == 2
    with pytest.raises(ValueError):
        manager.submit("performance_calibrate", {"unexpected": 1})


@pytest.mark.parametrize("status,complete,learn", [("queued", True, True), ("cancelled", True, False),
                                                   ("queued", False, False)])
def test_only_successful_complete_live_runs_are_learned(tmp_path, monkeypatch, status, complete, learn):
    manager = JobManager(workers=1, directory=tmp_path / "cache")
    records = []
    monkeypatch.setattr(manager.performance, "record", lambda *args: records.append(args))
    job = Job(id="test", kind="branches", key="a", status=status)
    run = _Run(id="test", kind="branches", key="a", payload=device(), warns=[], jobs=[job])
    manager._complete(run, ("ok", b"{}", {"complete": complete, "compute_s": 1}))
    assert bool(records) is learn


def test_worker_reports_compute_separately_from_resolve_and_result_runtime(monkeypatch):
    import server.jobs as jobs
    monkeypatch.setattr(jobs, "_W_PROGRESS", {})
    monkeypatch.setattr(jobs, "_W_CANCEL", {})
    monkeypatch.setattr(jobs, "_W_FAMILIES", set())
    def resolve(kind):
        time.sleep(.01)
        return lambda payload, progress: {"runtime_s": 999}
    monkeypatch.setattr(jobs, "_resolve", resolve)
    first = jobs._run_job("1", "branches", device(), [])
    second = jobs._run_job("2", "branches", device(), [])
    assert first[0] == "ok" and first[2]["first_for_family"]
    assert not second[2]["first_for_family"]
    assert first[2]["resolve_s"] >= .009
    assert first[2]["compute_s"] < .1
    assert jsonutil.loads(first[1])["runtime_s"] == 999


def test_api_performance_requires_no_worker(client):
    response = client.get("/api/performance")
    assert response.status_code == 200
    assert "host" in response.json() and "catalog" in response.json()
    response = client.post("/api/performance/estimate", json={"jobs": [{"kind": "branches", "payload": {}}]})
    assert response.status_code == 200
    assert "setup" in response.json()
    assert client.post("/api/performance/estimate", json={"jobs": []}).status_code == 422


@pytest.mark.slow
def test_real_calibration_repeats_and_fresh_job_learning(client, wait):
    before = client.get("/api/performance").json()
    completed = []
    for _ in range(2):
        started = client.post("/api/performance/calibrate", params={"wait": 0}).json()
        final = wait(client, started["job_id"], timeout=180)
        assert final["status"] == "done", final
        assert not final["cached"]
        assert len(final["result"]["samples"]) == 4
        assert all(row["warm_s"] > 0 and len(row["warm_samples_s"]) == 2 for row in final["result"]["samples"])
        completed.append(final)
    assert completed[0]["job_id"] != completed[1]["job_id"]
    profile = client.get("/api/performance").json()
    assert profile["host"]["id"] == before["host"]["id"]
    assert len(profile["calibration"]["samples"]) == 4
    p = {"device": {"model": "simple", "vg": -3.0037}}
    fresh = client.post("/api/compute/branches", params={"wait": 60}, json=p).json()
    assert fresh["status"] == "done" and not fresh["cached"], fresh
    count = sum(row["count"] for row in client.get("/api/performance").json()["observations"])
    assert count >= 1
    hit = client.post("/api/compute/branches", params={"wait": 60}, json=p).json()
    assert hit["cached"]
    assert sum(row["count"] for row in client.get("/api/performance").json()["observations"]) == count
    estimate = client.post("/api/performance/estimate", json={"jobs": [{"kind": "branches", "payload": p}]}).json()
    assert estimate["cached"] and estimate["total"]["seconds"] == 0
