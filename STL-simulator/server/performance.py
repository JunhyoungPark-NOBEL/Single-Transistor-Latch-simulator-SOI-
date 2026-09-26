"""Host-local timing estimates; pure Python (never imports a numerical kernel).

The benchmark table is a reference, not a CPU specification. Host scaling is
learned only from successful computations on this API server. First calls,
cache hits, queue waits, cancellation and JSON transfer are not warm samples.
"""
from __future__ import annotations

import hashlib
import copy
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
import platform
import statistics
import threading
import time
import uuid
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).parent / "data" / "performance_catalog.json"
SCHEMA_VERSION = 1
MAX_GROUP = 8
MAX_AGE_S = 30 * 24 * 3600


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def describe(kind: str, payload: dict) -> tuple[str, str, dict]:
    from server.performance_cases import family_for_payload, model_for_payload, workload_features
    return model_for_payload(kind, payload), family_for_payload(kind, payload), workload_features(kind, payload)


def family_key(kind: str, payload: dict) -> str:
    model, family, _ = describe(kind, payload)
    return model + ":" + family


def preparation_key(kind: str, payload: dict) -> str | None:
    """Stochastic conditional tables change with device/bias/node configuration.

    Repeating the same conditions with another random seed or cycle count does
    not make a new table; otherwise conservatively treat conditions as unseen.
    """
    if kind not in ("hazard", "sweep_mc", "vg_curve_stochastic") and not (
            kind == "circuit" and payload.get("mode") == "stochastic"):
        return None
    condition = copy.deepcopy(payload)
    condition.pop("seed", None)
    stochastic = condition.get("stochastic") or {}
    for name in ("seed", "n_cycles", "n_traces", "n_runs"):
        stochastic.pop(name, None)
    condition["stochastic"] = stochastic
    return hashlib.sha256((kind + json.dumps(condition, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()


def _range(seconds: float | None, lo: float = 0.5, hi: float = 2.0) -> dict:
    if seconds is None:
        return {"low_s": None, "seconds": None, "high_s": None}
    return {"low_s": max(0.0, seconds * lo), "seconds": max(0.0, seconds), "high_s": max(0.0, seconds * hi)}


def _units(features: dict) -> float:
    return max(1e-12, _finite(features.get("work_units"), 1.0))


def _distance(left: dict, right: dict) -> float:
    terms = []
    for key in set(left) & set(right):
        a, b = _finite(left[key]), _finite(right[key])
        if key in ("vg_V", "vbg_V"):
            terms.append(abs(a - b) / (0.5 if key == "vg_V" else 0.1))
        elif a > 0 and b > 0:
            terms.append(abs(math.log(a / b)))
        elif a != b:
            terms.append(1.0)
    return sum(terms) / max(1, len(terms)) + .25 * max(terms, default=0.0)


def _scaled(seconds: float, target: dict, reference: dict) -> float:
    # Workload extrapolation is intentionally not capped: a 100x longer transient
    # must not be presented as only 10x longer. Solver adaptation remains uncertain.
    return max(0.0001, seconds * _units(target) / _units(reference))


class PerformanceStore:
    def __init__(self, directory: Path, engine_version: str, workers: int, cpus: int,
                 catalog_path: Path = CATALOG_PATH) -> None:
        self.directory, self.catalog_path = directory, catalog_path
        self._lock = threading.RLock()
        self.engine_version, self.workers = engine_version, workers
        self.instance_id = uuid.uuid4().hex
        self._warm: dict[str, set[int]] = {}
        self._prepared: dict[str, set[int]] = {}
        self._observations: list[dict] = []
        self._calibration: dict | None = None
        self._catalog_mtime = None
        self._catalog: dict = {}
        try:
            directory.mkdir(parents=True, exist_ok=True)
            identity = directory / "host-id"
            if not identity.exists():
                identity.write_text(uuid.uuid4().hex)
            self.host_id = identity.read_text().strip()[:64]
        except OSError:
            self.host_id = uuid.uuid4().hex
        self.host_id = hashlib.sha256((self.host_id + platform.node() + platform.machine()).encode()).hexdigest()[:32]
        self.host = dict(id=self.host_id, instance_id=self.instance_id,
                         engine_version=engine_version, label="계산 서버",
                         os=platform.system(), architecture=platform.machine(),
                         python=platform.python_version(), workers=workers, cpu_count=cpus)
        packages = {}
        for name in ("numpy", "scipy", "numba", "llvmlite"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = "unavailable"
        self.host["packages"] = packages
        # Separate records after a Python/environment/CPU capacity change as well
        # as an engine change; do not expose a machine name or network identifier.
        identity_material = {**self.host, "instance_id": None,
                             "processor": platform.processor(),
                             "threads": {v: os.environ.get(v, "1") for v in
                                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
        revision = hashlib.sha256()
        for source in (Path(__file__), Path(__file__).with_name("performance_cases.py")):
            try:
                revision.update(source.read_bytes())
            except OSError:
                revision.update(b"unavailable")
        identity_material.update(estimator_schema=SCHEMA_VERSION, estimator_revision=revision.hexdigest())
        fingerprint = hashlib.sha256(json.dumps(identity_material, sort_keys=True).encode()).hexdigest()
        self.host["profile_id"] = fingerprint[:24]
        self.path = directory / (fingerprint + ".json")
        try:
            data = json.loads(self.path.read_text())
            self._observations = [o for o in data.get("observations", [])[-240:]
                                  if time.time() - _finite(o.get("measured_at")) < MAX_AGE_S]
            self._calibration = data.get("calibration")
            if self._calibration and time.time() - _finite(self._calibration.get("measured_at")) >= MAX_AGE_S:
                self._calibration = None
        except (OSError, ValueError, TypeError):
            pass

    def catalog(self) -> dict:
        try:
            stamp = self.catalog_path.stat().st_mtime_ns
            if stamp != self._catalog_mtime:
                catalog = json.loads(self.catalog_path.read_text())
                if not isinstance(catalog, dict) or not isinstance(catalog.get("cases"), list):
                    raise ValueError("invalid benchmark catalog")
                self._catalog, self._catalog_mtime = catalog, stamp
        except (OSError, ValueError):
            if not self._catalog:
                return {"schema_version": 1, "cases": [], "unavailable": True}
        return self._catalog

    def _save(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"schema_version": SCHEMA_VERSION,
                                            "calibration": self._calibration,
                                            "observations": self._observations[-240:]}))
            os.replace(temporary, self.path)
        except OSError:
            pass  # A read-only installation can still learn within this session.

    def reset_workers(self) -> None:
        with self._lock:
            self._warm.clear()
            self._prepared.clear()

    def _expire(self) -> None:
        cutoff = time.time() - MAX_AGE_S
        self._observations = [row for row in self._observations if row.get("measured_at", 0) >= cutoff]
        if self._calibration and self._calibration.get("measured_at", 0) < cutoff:
            self._calibration = None

    def record(self, kind: str, payload: dict, key: str, timing: dict, result: dict | None = None) -> None:
        """Called once per successful run, never once per joined client handle."""
        with self._lock:
            if kind == "performance_calibrate":
                samples = list((result or {}).get("samples") or [])
                valid = [s for s in samples if _finite(s.get("warm_s")) > 0]
                if valid:
                    self._calibration = {"measured_at": time.time(), "samples": valid,
                                         "engine_version": self.engine_version}
                    for s in valid:
                        self._warm.setdefault(s["model"] + ":" + s["family"], set()).add(int(timing.get("worker_pid", 0)))
                    self._save()
                return
            model, family, features = describe(kind, payload)
            tag = model + ":" + family
            self._warm.setdefault(tag, set()).add(int(timing.get("worker_pid", 0)))
            preparation = preparation_key(kind, payload)
            if preparation:
                self._prepared.setdefault(preparation, set()).add(int(timing.get("worker_pid", 0)))
            seconds = _finite(timing.get("compute_s"))
            # First kernel execution may load disk cache or compile. Neither can
            # be inferred from a single elapsed time, so do not train on it.
            if timing.get("first_for_family", True) or timing.get("first_preparation", False) or seconds <= 0:
                return
            self._observations.append(dict(kind=kind, model=model, family=family, features=features,
                                           key=key, seconds=seconds, measured_at=time.time()))
            self._observations = self._observations[-240:]
            self._save()

    def snapshot(self) -> dict:
        with self._lock:
            self._expire()
            groups: dict[str, dict] = {}
            for obs in self._observations:
                tag = obs["model"] + ":" + obs["family"]
                group = groups.setdefault(tag, {"model": obs["model"], "family": obs["family"],
                                               "count": 0, "last_seconds": 0})
                group.update(count=group["count"] + 1, last_seconds=obs["seconds"], measured_at=obs["measured_at"])
            return dict(catalog=self.catalog(), host=self.host, calibration=self._calibration,
                        observations=list(groups.values()),
                        warm_workers={key: len(pids) for key, pids in self._warm.items()})

    def estimate_one(self, kind: str, payload: dict, key: str) -> dict:
        model, family, features = describe(kind, payload)
        cases = [c for c in self.catalog().get("cases", []) if c.get("supported", True) and c.get("group") != "holdout"
                 and c.get("model") == model and c.get("family") == family
                 and _finite((c.get("timings") or {}).get("median_s")) > 0]
        with self._lock:
            self._expire()
            observations = [o for o in self._observations if o.get("model") == model and o.get("family") == family]
            exact = [o for o in observations if o.get("key") == key]
            calibrated = list((self._calibration or {}).get("samples") or [])
            warm_workers = len(self._warm.get(model + ":" + family, set()))
            preparation = preparation_key(kind, payload)
            preparation_unknown = bool(preparation and len(self._prepared.get(preparation, set())) < self.workers)
        base = dict(kind=kind, model=model, family=family, features=features, supported=True,
                    cached=False, joined=False, setup_unknown=warm_workers < self.workers or preparation_unknown,
                    preparation_unknown=preparation_unknown,
                    warm_workers=warm_workers)
        if exact:
            values = [o["seconds"] for o in exact[-9:]]
            center = statistics.median(values)
            # Host load and adaptive solvers vary even for an identical payload.
            estimate = {"seconds": center, "low_s": min(values) * .75,
                        "high_s": max(values) * 1.5}
            return {**base, "estimate": estimate, "source": "observed",
                    "confidence": "high" if len(values) >= 3 else "medium", "samples": len(values)}
        if observations:
            nearest = sorted(observations, key=lambda o: _distance(features, o["features"]))[:5]
            distance = _distance(features, nearest[0]["features"])
            # A different branch/grid family is never borrowed. Large geometry or
            # transient changes are still marked as extrapolation below.
            values = [_scaled(o["seconds"], features, o["features"]) for o in nearest]
            return {**base, "estimate": _range(statistics.median(values), .35 if distance < 1 else .15,
                                               min(20.0, 3.0 + 2 * distance)),
                    "source": "observed", "confidence": "low", "samples": len(nearest),
                    "extrapolated": True}
        if not cases:
            return {**base, "estimate": _range(None), "source": "reference", "confidence": "low",
                    "reason": "이 조합의 측정 기준이 없습니다."}
        reference = min(cases, key=lambda c: _distance(features, c.get("features") or {}))
        center = _scaled(float(reference["timings"]["median_s"]), features, reference.get("features") or {})
        ratios = []
        cases_by_id = {c.get("id"): c for c in cases}
        for sample in calibrated:
            ref = cases_by_id.get(sample.get("case_id"))
            if ref and sample.get("model") == model and sample.get("family") == family:
                ratios.append(sample["warm_s"] / ref["timings"]["median_s"] *
                              _units(ref.get("features") or {}) / _units(sample.get("features") or ref.get("features") or {}))
        source = "calibrated" if ratios else "reference"
        if ratios:
            center *= statistics.median(ratios)
        broad = family.startswith("circuit") or "csvm" in family or "stochastic" in family
        distance = _distance(features, reference.get("features") or {})
        return {**base, "estimate": _range(center, .15 if distance > 1 else (.3 if broad else .5),
                                           min(20.0, (4 if broad else 2.5) + 2 * distance)),
                "source": source, "confidence": "low" if source == "reference" or broad else "medium",
                "reference_id": reference["id"], "extrapolated": features != reference.get("features")}


def schedule(items: list[dict], workers: int, initial: list[float] | None = None,
             field: str = "seconds") -> float | None:
    """Conservative list scheduling for a small dependency DAG on the actual pool."""
    lanes = list(initial or [0.0] * workers)
    finishes: dict[str, float] = {}
    pending = list(items)
    while pending:
        ready = next((item for item in pending if all(dep in finishes for dep in item.get("depends_on", []))), None)
        if ready is None:
            raise ValueError("performance jobs contain an unknown or cyclic dependency")
        seconds = ready["estimate"].get(field)
        if seconds is None:
            return None
        available = max((finishes[dep] for dep in ready.get("depends_on", [])), default=0.0)
        lane = min(range(len(lanes)), key=lambda index: lanes[index])
        shared_finish = ready.get("inflight_finish", {}).get(field, seconds) if initial is not None else seconds
        end = (available + seconds if ready.get("cached") else
               max(available, shared_finish) if ready.get("joined") else
               max(lanes[lane], available) + seconds)
        finishes[ready["key"]] = end
        if not ready.get("cached") and not ready.get("joined"):
            lanes[lane] = end
        pending.remove(ready)
    return max(finishes.values(), default=0.0)
