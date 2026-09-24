"""Job manager: a process pool running the compute kinds, with progress, cancellation and a result cache.

* Workers: ``ProcessPoolExecutor`` with a ``spawn`` context (the API process runs uvicorn threads, so no
  fork); the initializer imports ``server.engine_bridge`` once per worker (numba cache load).
  Number of workers: env ``STL_WORKERS`` or ``max(1, cpu_count - 1)``.
* Progress / cancellation: a ``multiprocessing.Manager`` holds two dicts shared with the workers
  (``progress[job_id] = (fraction, message)`` and ``cancel[job_id] = True``).  The progress callback given to
  the compute function writes at most every 0.2 s and raises ``JobCancelled`` once the cancel flag is set.
* Results are serialised to JSON bytes inside the worker (``server.jsonutil``: NaN/inf → null) and cached
  in memory (LRU) and on disk as gzip JSON under ``server/.cache/results/`` (env ``STL_CACHE_DIR``).
  Cache key = sha256(kind + canonical JSON of the normalised payload + clamp warnings + ENGINE_VERSION),
  ENGINE_VERSION = hash of ``server/compute/**/*.py``, ``server/params.py`` and ``server/engine_bridge.py``.
* The API process never imports numba or the engine: compute modules are imported only in workers.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib
import logging
import multiprocessing as mp
import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from server import jsonutil, payloads
from server.compute import KINDS
from server.progress import JobCancelled

log = logging.getLogger("stl.jobs")

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parent

# Kinds served in addition to the contract registry (thin aliases, backend-core owned).
EXTRA_KINDS: dict[str, str] = {"folds": "server.compute.deterministic:run_folds"}
ALL_KINDS: dict[str, str] = {**KINDS, **EXTRA_KINDS}

STATUSES = ("queued", "running", "done", "error", "cancelled")
FINAL = ("done", "error", "cancelled")


def engine_version() -> str:
    """sha256 over the compute sources (+ params.py, engine_bridge.py): any code edit invalidates the cache."""
    files = sorted((SERVER_DIR / "compute").rglob("*.py")) + [SERVER_DIR / "params.py", SERVER_DIR / "engine_bridge.py"]
    h = hashlib.sha256()
    for f in files:
        if "__pycache__" in f.parts or not f.is_file():
            continue
        h.update(str(f.relative_to(SERVER_DIR)).encode())
        h.update(b"\x00")
        h.update(f.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()


def available_cpus() -> int:
    """CPUs usable by this process: affinity mask, capped by a cgroup v2/v1 CPU quota (containers)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    quota = None
    try:
        q, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if q != "max":
            quota = float(q) / float(period)
    except (OSError, ValueError):
        try:
            q = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0 and period > 0:
                quota = q / period
        except (OSError, ValueError):
            pass
    if quota is not None:
        n = min(n, max(1, int(quota + 0.5)))
    return max(1, n)


def default_workers() -> int:
    """env STL_WORKERS, else max(1, available CPUs - 1)."""
    env = os.environ.get("STL_WORKERS", "").strip()
    if env:
        return max(1, int(env))
    return max(1, available_cpus() - 1)


def cache_dir() -> Path:
    return Path(os.environ.get("STL_CACHE_DIR") or (SERVER_DIR / ".cache" / "results"))


# =============================================================================================
# worker side (runs in the pool processes)
# =============================================================================================
_W_PROGRESS: Any = None
_W_CANCEL: Any = None


def _init_worker(progress_dict: Any, cancel_dict: Any) -> None:
    global _W_PROGRESS, _W_CANCEL
    _W_PROGRESS, _W_CANCEL = progress_dict, cancel_dict
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    importlib.import_module("server.engine_bridge")      # loads the engine + numba caches once per worker


def _resolve(kind: str) -> Callable:
    module, func = ALL_KINDS[kind].split(":")
    return getattr(importlib.import_module(module), func)


def _warm() -> bool:
    return True


def _run_job(job_id: str, kind: str, payload: dict, pre_warnings: list[str]) -> tuple:
    """Executed in a worker. Returns ("ok", json_bytes) | ("error", message, traceback) | ("cancelled",)."""
    state = {"last": -1e9}

    def progress(fraction: float = 0.0, message: str = "") -> None:
        now = time.monotonic()
        if now - state["last"] < 0.2 and fraction < 1.0:
            return
        state["last"] = now
        try:
            _W_PROGRESS[job_id] = (float(min(max(fraction, 0.0), 1.0)), str(message)[:300])
            cancelled = bool(_W_CANCEL.get(job_id, False))
        except (OSError, EOFError, BrokenPipeError, ConnectionError, TypeError):
            return
        if cancelled:
            raise JobCancelled(job_id)

    t0 = time.perf_counter()
    try:
        progress(0.0, "started")
        fn = _resolve(kind)
        result = fn(payload, progress)
        if not isinstance(result, dict):
            result = {"value": result}
        result["warnings"] = list(pre_warnings) + list(result.get("warnings") or [])
        result.setdefault("runtime_s", time.perf_counter() - t0)
        return ("ok", jsonutil.dumps(result))
    except JobCancelled:
        return ("cancelled",)
    except ValueError as exc:
        return ("error", str(exc) or "invalid input", traceback.format_exc())
    except ModuleNotFoundError as exc:
        return ("error", f"compute kind {kind!r} is not available yet ({exc})", traceback.format_exc())
    except Exception as exc:  # noqa: BLE001 - reported to the client
        return ("error", f"{type(exc).__name__}: {exc}", traceback.format_exc())


# =============================================================================================
# API side
# =============================================================================================
@dataclass
class Job:
    id: str
    kind: str
    key: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "queued"
    result: bytes | None = None
    error: str | None = None
    traceback: str | None = None
    cached: bool = False
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    future: Future | None = None
    event: threading.Event = field(default_factory=threading.Event)

    def elapsed(self) -> float:
        if self.cached:
            return 0.0
        end = self.finished if self.finished is not None else time.time()
        return max(0.0, end - self.created)


class ResultCache:
    """In-memory LRU (bytes budget) in front of a gzip-JSON directory."""

    def __init__(self, directory: Path, max_items: int = 256, max_bytes: int = 256 << 20,
                 disk_max_bytes: int = 1 << 30) -> None:
        self.dir = directory
        self.max_items, self.max_bytes, self.disk_max_bytes = max_items, max_bytes, disk_max_bytes
        self._mem: OrderedDict[str, bytes] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._prune_disk()
        except OSError as exc:
            log.warning("result cache directory unavailable: %s", exc)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json.gz"

    def get(self, key: str) -> bytes | None:
        with self._lock:
            data = self._mem.get(key)
            if data is not None:
                self._mem.move_to_end(key)
                return data
        path = self._path(key)
        try:
            data = gzip.decompress(path.read_bytes())
        except (OSError, EOFError, gzip.BadGzipFile):
            return None
        self._put_mem(key, data)
        try:
            os.utime(path)
        except OSError:
            pass
        return data

    def _put_mem(self, key: str, data: bytes) -> None:
        with self._lock:
            if key in self._mem:
                self._size -= len(self._mem.pop(key))
            self._mem[key] = data
            self._size += len(data)
            while self._mem and (len(self._mem) > self.max_items or self._size > self.max_bytes):
                _, old = self._mem.popitem(last=False)
                self._size -= len(old)

    def put(self, key: str, data: bytes) -> None:
        self._put_mem(key, data)
        try:
            tmp = self.dir / f".{key}.{uuid.uuid4().hex}.tmp"
            tmp.write_bytes(gzip.compress(data, compresslevel=5))
            os.replace(tmp, self._path(key))
        except OSError as exc:
            log.warning("could not write cache entry %s: %s", key, exc)

    def _prune_disk(self) -> None:
        files = sorted(self.dir.glob("*.json.gz"), key=lambda f: f.stat().st_mtime)
        total = sum(f.stat().st_size for f in files)
        while files and total > self.disk_max_bytes:
            f = files.pop(0)
            total -= f.stat().st_size
            f.unlink(missing_ok=True)

    def clear_memory(self) -> None:
        with self._lock:
            self._mem.clear()
            self._size = 0


class JobManager:
    def __init__(self, workers: int | None = None, max_jobs: int = 200, directory: Path | None = None,
                 mp_context: str | None = None) -> None:
        self.workers = workers or default_workers()
        self.max_jobs = max_jobs
        self.mp_context = mp_context or os.environ.get("STL_MP_CONTEXT", "spawn")
        self.engine_version = engine_version()
        self.cache = ResultCache(directory or cache_dir(),
                                 disk_max_bytes=int(float(os.environ.get("STL_DISK_CACHE_MB", "1024")) * (1 << 20)))
        self._lock = threading.RLock()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._inflight: dict[str, str] = {}
        self._pool: ProcessPoolExecutor | None = None
        self._manager: Any = None
        self._progress: Any = None
        self._cancel: Any = None
        self._broken = False

    # ---- lifecycle ---------------------------------------------------------------------------
    def start(self, prewarm: bool = False) -> None:
        with self._lock:
            if self._pool is not None:
                return
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")      # inherited by the spawned workers
            ctx = mp.get_context(self.mp_context)
            if self._manager is None:
                self._manager = ctx.Manager()
                self._progress = self._manager.dict()
                self._cancel = self._manager.dict()
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx, initializer=_init_worker,
                                             initargs=(self._progress, self._cancel))
            self._broken = False
        if prewarm:
            for _ in range(self.workers):
                self._pool.submit(_warm)

    def shutdown(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
            manager, self._manager = self._manager, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:  # noqa: BLE001
                pass

    def _restart_pool(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass
        self.start()

    # ---- submission --------------------------------------------------------------------------
    def cache_key(self, kind: str, payload: dict, warnings: list[str]) -> str:
        return jsonutil.sha256_hex(kind, jsonutil.canonical(payload), jsonutil.canonical(warnings), self.engine_version)

    def submit(self, kind: str, payload: Any) -> Job:
        """Validate + submit. Raises KeyError (unknown kind) or ValueError (invalid payload)."""
        if kind not in ALL_KINDS:
            raise KeyError(kind)
        norm, warns = payloads.normalize(kind, payload)
        key = self.cache_key(kind, norm, warns)
        with self._lock:
            jid = self._inflight.get(key)
            if jid and jid in self._jobs and self._jobs[jid].status in ("queued", "running"):
                return self._jobs[jid]
        data = self.cache.get(key)
        job = Job(id=uuid.uuid4().hex[:16], kind=kind, key=key)
        if data is not None:
            job.status, job.progress, job.message, job.result, job.cached = "done", 1.0, "cached", data, True
            job.started = job.finished = job.created
            job.event.set()
            self._register(job)
            return job
        self._register(job)
        with self._lock:
            self._inflight[key] = job.id
        self._dispatch(job, norm, warns)
        return job

    def _dispatch(self, job: Job, payload: dict, warns: list[str]) -> None:
        for attempt in range(2):
            if self._pool is None or self._broken:
                self._restart_pool()
            try:
                fut = self._pool.submit(_run_job, job.id, job.kind, payload, warns)
                break
            except (BrokenProcessPool, RuntimeError):
                self._broken = True
                if attempt:
                    self._finish(job, "error", error="the worker pool is unavailable")
                    return
        job.future = fut
        fut.add_done_callback(lambda f, j=job: self._on_done(j, f))

    def _register(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._evict()

    def _evict(self) -> None:
        finished = [jid for jid, j in self._jobs.items() if j.status in FINAL]
        excess = len(self._jobs) - self.max_jobs
        for jid in finished[: max(0, excess)]:
            self._jobs.pop(jid, None)

    # ---- completion --------------------------------------------------------------------------
    def _on_done(self, job: Job, fut: Future) -> None:
        try:
            out = fut.result()
        except CancelledError:
            out = ("cancelled",)
        except BrokenProcessPool as exc:
            self._broken = True
            out = ("error", f"the worker process crashed ({exc}); the pool will be restarted", None)
        except Exception as exc:  # noqa: BLE001
            out = ("error", f"{type(exc).__name__}: {exc}", traceback.format_exc())
        try:
            if self._progress is not None:
                self._progress.pop(job.id, None)
            if self._cancel is not None:
                self._cancel.pop(job.id, None)
        except Exception:  # noqa: BLE001 - manager gone during shutdown
            pass
        if out[0] == "ok":
            self.cache.put(job.key, out[1])
            if job.status != "cancelled":
                self._finish(job, "done", result=out[1])
        elif out[0] == "cancelled":
            self._finish(job, "cancelled")
        else:
            if out[2]:
                log.warning("job %s (%s) failed: %s\n%s", job.id, job.kind, out[1], out[2])
            self._finish(job, "error", error=out[1], tb=out[2])

    def _finish(self, job: Job, status: str, result: bytes | None = None, error: str | None = None,
                tb: str | None = None) -> None:
        with self._lock:
            if job.status in FINAL:        # e.g. cancelled by the user, the worker finishing later
                return
            job.status = status
            job.finished = job.finished or time.time()
            if status == "done":
                job.progress, job.message, job.result = 1.0, "done", result
            elif status == "error":
                job.message, job.error, job.traceback = "error", error, tb
            else:
                job.message = "cancelled"
            if self._inflight.get(job.key) == job.id:
                self._inflight.pop(job.key, None)
            self._evict()
        job.event.set()

    # ---- queries -----------------------------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            self._refresh(job)
        return job

    def _refresh(self, job: Job) -> None:
        if job.status not in ("queued", "running") or self._progress is None:
            return
        try:
            rec = self._progress.get(job.id)
        except Exception:  # noqa: BLE001
            return
        if rec is None:
            return
        with self._lock:
            if job.status in ("queued", "running"):
                job.status = "running"
                job.started = job.started or time.time()
                job.progress, job.message = float(rec[0]), str(rec[1])

    def wait(self, job: Job, timeout: float) -> Job:
        if timeout > 0 and job.status not in FINAL:
            job.event.wait(timeout)
        self._refresh(job)
        return job

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None or job.status in FINAL:
            return job
        if job.future is not None and job.future.cancel():
            self._finish(job, "cancelled")
            return job
        try:
            if self._cancel is not None:
                self._cancel[job.id] = True
        except Exception:  # noqa: BLE001
            pass
        self._finish(job, "cancelled")    # reported immediately; the worker stops at its next progress call
        return job

    def list(self) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        for j in jobs:
            self._refresh(j)
        return jobs

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for j in self.list():
            out[j.status] = out.get(j.status, 0) + 1
        return out

    # ---- serialisation -----------------------------------------------------------------------
    @staticmethod
    def status(job: Job, include_result: bool = True) -> dict:
        """Contract `JobStatus` (result embedded as pre-serialised JSON)."""
        out: dict[str, Any] = dict(job_id=job.id, kind=job.kind, status=job.status, progress=job.progress,
                                   message=job.message, cached=job.cached, elapsed_s=round(job.elapsed(), 3))
        if job.status == "done" and include_result and job.result is not None:
            out["result"] = jsonutil.Fragment(job.result)
        if job.status == "error":
            out["error"] = job.error or "unknown error"
        return out
