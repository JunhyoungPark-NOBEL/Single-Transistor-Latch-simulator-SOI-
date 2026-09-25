"""Job manager: a process pool running the compute kinds, with progress, cancellation and a result cache.

* Workers: ``ProcessPoolExecutor`` with a ``spawn`` context (the API process runs uvicorn threads, so no
  fork); the initializer imports ``server.engine_bridge`` once per worker (numba cache load).
  Number of workers: env ``STL_WORKERS`` or ``max(1, cpu_count - 1)``.  Workers and the Manager process
  exit on their own when the API process dies (parent-death watchdog), so a SIGKILL/OOM kill of the API
  leaves no orphans.  A crashed worker (segfault, OOM kill) breaks the pool: it is restarted at once and the
  runs that were in it are retried once (the crash may have been caused by another job).
* Jobs and runs: a ``Job`` is one client's handle (its own id, status, cancel); a ``_Run`` is one
  computation.  Identical in-flight payloads share a run, so one client cancelling its handle does not
  cancel the other's; the run itself is cancelled when its last handle is cancelled.  Handles no client
  has asked about for ``STL_ABANDON_S`` (600 s) are cancelled (closed browser tab).
* Admission: at most ``STL_MAX_PENDING`` (64) queued/running handles in total and
  ``STL_MAX_PENDING_PER_CLIENT`` (16) per client address; beyond that ``submit`` raises ``QueueFull`` (HTTP 429).
* Progress / cancellation: a ``multiprocessing.Manager`` holds two dicts shared with the workers
  (``progress[run_id] = (fraction, message)`` and ``cancel[run_id] = True``).  The progress callback given to
  the compute function writes at most every 0.2 s and raises ``JobCancelled`` once the cancel flag is set.
* Results are serialised to JSON bytes inside the worker (``server.jsonutil``: NaN/inf → null) and cached
  in memory (LRU, ``STL_MEM_CACHE_MB``) and on disk as gzip JSON under ``server/.cache/results/``
  (env ``STL_CACHE_DIR``, kept below ``STL_DISK_CACHE_MB`` while running).  Finished handles keep at most
  ``STL_JOB_RESULTS_MB`` of result bytes; older results are re-read from the cache on demand.
  The stochastic package's node cache (``server/.cache/stochastic``) is pruned to ``STL_NODE_CACHE_MB``.
  Cache key = sha256(kind + canonical JSON of the normalised payload + clamp warnings + ENGINE_VERSION),
  ENGINE_VERSION = hash of ``server/compute/**/*.py``, ``params.py``, ``engine_bridge.py``, ``jsonutil.py``,
  ``jobs.py`` and the engine's code + data files (``engine/**``, run-time cache folders excluded).
* The API process never imports numba or the engine: compute modules are imported only in workers.
"""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import importlib
import logging
import math
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
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Any, Callable

from server import jsonutil, payloads
from server.compute import KINDS
from server.progress import JobCancelled

log = logging.getLogger("stl.jobs")

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parent
ENGINE_DIR = ROOT / "engine"

# Kinds served in addition to the contract registry (thin aliases, backend-core owned).
EXTRA_KINDS: dict[str, str] = {"folds": "server.compute.deterministic:run_folds"}
ALL_KINDS: dict[str, str] = {**KINDS, **EXTRA_KINDS}

STATUSES = ("queued", "running", "done", "error", "cancelled")
FINAL = ("done", "error", "cancelled")

# engine inputs that shape results (code + data); run-time cache folders are not inputs
_ENGINE_SUFFIXES = {".py", ".json", ".npz", ".npy", ".csv", ".txt"}
_ENGINE_CACHE_DIRS = {"__pycache__", "photo_nodes", "fpt_nodes", "conditional_table", "fast_fpt"}
# server sources that never shape a result (HTTP layer, login gate); every other server/**/*.py is hashed
_SERVER_EXCLUDED = ("main.py", "auth.py")


def _server_sources(server_dir: Path) -> list[Path]:
    """server/**/*.py except tests, caches (__pycache__, hidden folders such as .cache) and _SERVER_EXCLUDED."""
    out = []
    for f in server_dir.rglob("*.py"):
        rel = f.relative_to(server_dir).parts
        if "__pycache__" in rel or "tests" in rel or any(x.startswith(".") for x in rel[:-1]):
            continue
        if len(rel) == 1 and rel[0] in _SERVER_EXCLUDED:
            continue
        out.append(f)
    return sorted(out)


def _env_float(name: str, default: float) -> float:
    try:
        x = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return x if math.isfinite(x) else default


def engine_version(server_dir: Path = SERVER_DIR, engine_dir: Path = ENGINE_DIR) -> str:
    """sha256 over everything that shapes a cached result: every server/**/*.py except the tests and the HTTP/login
    layer (main.py, auth.py) -- compute/**, params.py, engine_bridge.py, geometry_model.py, payloads.py (normalisation),
    jsonutil.py (serialisation), jobs.py (result post-processing), and any module added later -- plus the engine's
    code + data files.  Any edit invalidates the result cache."""
    groups: list[tuple[str, Path, list[Path]]] = [("server", server_dir, _server_sources(server_dir))]
    if engine_dir.is_dir():
        groups.append(("engine", engine_dir, sorted(
            f for f in engine_dir.rglob("*")
            if f.suffix in _ENGINE_SUFFIXES and not _ENGINE_CACHE_DIRS.intersection(f.relative_to(engine_dir).parts))))
    h = hashlib.sha256()
    for tag, base, files in groups:
        for f in files:
            if "__pycache__" in f.parts or not f.is_file():
                continue
            h.update(f"{tag}/{f.relative_to(base).as_posix()}".encode())
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


def node_cache_dir() -> Path:
    """The stochastic package's fold/hazard node cache (server/compute/stoch_core.py CACHE_DIR)."""
    return Path(os.environ.get("STL_STOCH_CACHE_DIR") or (SERVER_DIR / ".cache" / "stochastic"))


def prune_dir(directory: Path, max_bytes: int, pattern: str = "*") -> int:
    """Delete the oldest files (by mtime) matching ``pattern`` below ``directory`` until their total size is
    <= max_bytes; stale temp files (> 1 h) go too.  Returns the remaining total.  Safe with concurrent
    readers: every cache reader treats a missing file as a miss."""
    entries: list[tuple[float, int, Path]] = []
    now = time.time()
    for f in directory.rglob(pattern):
        try:
            st = f.stat()
        except OSError:
            continue
        entries.append((st.st_mtime, st.st_size, f))
    for f in directory.rglob("*.tmp"):
        try:
            if now - f.stat().st_mtime > 3600:
                f.unlink(missing_ok=True)
        except OSError:
            pass
    entries.sort(key=lambda e: e[0])
    total = sum(e[1] for e in entries)
    for _, size, f in entries:
        if total <= max_bytes:
            break
        try:
            f.unlink(missing_ok=True)
            total -= size
        except OSError:
            pass
    return total


# =============================================================================================
# worker side (runs in the pool processes)
# =============================================================================================
_W_PROGRESS: Any = None
_W_CANCEL: Any = None


def _parent_watchdog() -> None:
    """Exit this child process as soon as the API process is gone.  Without it a SIGKILL / OOM kill of the
    API process leaves the spawned workers and the Manager process re-parented to init for ever."""
    parent = mp.parent_process()
    ppid = os.getppid()

    def watch() -> None:
        if parent is not None:
            parent.join()               # returns when the parent's sentinel pipe closes (the parent exited)
        else:
            while os.getppid() == ppid:
                time.sleep(2.0)
        os._exit(0)

    threading.Thread(target=watch, name="stl-parent-watchdog", daemon=True).start()


def _init_worker(progress_dict: Any, cancel_dict: Any) -> None:
    global _W_PROGRESS, _W_CANCEL
    _parent_watchdog()
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
class QueueFull(Exception):
    """Too many queued/running jobs (globally or for this client) → HTTP 429."""


@dataclass(eq=False)
class Job:
    """One client's handle on a computation (identical in-flight submissions share one ``_Run``)."""
    id: str
    kind: str
    key: str
    client: str = ""
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
    seen: float = field(default_factory=time.monotonic)     # last client contact (abandoned-job reaper)
    run: _Run | None = None
    event: threading.Event = field(default_factory=threading.Event)

    def elapsed(self) -> float:
        if self.cached:
            return 0.0
        end = self.finished if self.finished is not None else time.time()
        return max(0.0, end - self.created)


@dataclass(eq=False)
class _Run:
    """One computation in the pool; ``jobs`` are the handles still interested in it."""
    id: str                      # key of the Manager progress/cancel dicts
    kind: str
    key: str
    payload: dict
    warns: list[str]
    jobs: list[Job] = field(default_factory=list)
    future: Future | None = None
    pool: ProcessPoolExecutor | None = None
    attempts: int = 0
    done: bool = False


class ResultCache:
    """In-memory LRU (bytes budget) in front of a gzip-JSON directory (size budget enforced while running)."""

    def __init__(self, directory: Path, max_items: int = 256, max_bytes: int = 256 << 20,
                 disk_max_bytes: int = 1 << 30) -> None:
        self.dir = directory
        self.max_items, self.max_bytes, self.disk_max_bytes = max_items, max_bytes, disk_max_bytes
        self._mem: OrderedDict[str, bytes] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()
        self._disk_bytes = 0
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
            os.utime(path)          # LRU on disk: pruning removes the least recently used first
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
            blob = gzip.compress(data, compresslevel=5)
            tmp = self.dir / f".{key}.{uuid.uuid4().hex}.tmp"
            tmp.write_bytes(blob)
            os.replace(tmp, self._path(key))
        except OSError as exc:
            log.warning("could not write cache entry %s: %s", key, exc)
            return
        with self._lock:
            self._disk_bytes += len(blob)
            over = self._disk_bytes > self.disk_max_bytes
        if over:
            self._prune_disk()

    def _prune_disk(self) -> None:
        """Oldest entries first down to 80 % of the budget (so a full cache is not rescanned on every put)."""
        target = self.disk_max_bytes if self._disk_bytes == 0 else int(0.8 * self.disk_max_bytes)
        try:
            remaining = prune_dir(self.dir, target, "*.json.gz")
        except OSError as exc:
            log.warning("result cache pruning failed: %s", exc)
            return
        with self._lock:
            self._disk_bytes = remaining

    def clear_memory(self) -> None:
        with self._lock:
            self._mem.clear()
            self._size = 0


class JobManager:
    def __init__(self, workers: int | None = None, max_jobs: int = 200, directory: Path | None = None,
                 mp_context: str | None = None) -> None:
        self.workers = workers or default_workers()
        self.max_jobs = max_jobs
        self.max_pending = max(1, int(_env_float("STL_MAX_PENDING", 64)))
        self.max_pending_client = max(1, int(_env_float("STL_MAX_PENDING_PER_CLIENT", 16)))
        self.result_budget = int(_env_float("STL_JOB_RESULTS_MB", 64) * (1 << 20))
        self.abandon_s = _env_float("STL_ABANDON_S", 600.0)
        self.node_cache_dir = node_cache_dir()
        self.node_cache_max = int(_env_float("STL_NODE_CACHE_MB", 1024) * (1 << 20))
        self.mp_context = mp_context or os.environ.get("STL_MP_CONTEXT", "spawn")
        self.engine_version = engine_version()
        self.cache = ResultCache(directory or cache_dir(),
                                 max_bytes=int(_env_float("STL_MEM_CACHE_MB", 256) * (1 << 20)),
                                 disk_max_bytes=int(_env_float("STL_DISK_CACHE_MB", 1024) * (1 << 20)))
        self._lock = threading.RLock()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._inflight: dict[str, _Run] = {}
        self._pool: ProcessPoolExecutor | None = None
        self._manager: Any = None
        self._progress: Any = None
        self._cancel: Any = None
        self._broken = False
        self._closing = False
        self._stop = threading.Event()
        self._housekeeper: threading.Thread | None = None

    # ---- lifecycle ---------------------------------------------------------------------------
    def start(self, prewarm: bool = False) -> None:
        with self._lock:
            if self._pool is not None:
                return
            self._closing = False
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")      # inherited by the spawned workers
            ctx = mp.get_context(self.mp_context)
            if self._manager is None:
                mgr = SyncManager(ctx=ctx)
                mgr.start(initializer=_parent_watchdog)
                self._manager = mgr
                self._progress = mgr.dict()
                self._cancel = mgr.dict()
            self._pool = pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx,
                                                    initializer=_init_worker, initargs=(self._progress, self._cancel))
            self._broken = False
            if self._housekeeper is None:
                self._stop.clear()
                self._housekeeper = threading.Thread(target=self._housekeeping, name="stl-jobs-housekeeping",
                                                     daemon=True)
                self._housekeeper.start()
        if prewarm:
            for _ in range(self.workers):
                pool.submit(_warm)

    def shutdown(self) -> None:
        """Stop promptly: cancel queued jobs and terminate the workers (a running job would otherwise keep the
        interpreter alive until it finishes, because concurrent.futures joins its workers at exit)."""
        self._closing = True
        self._stop.set()
        with self._lock:
            pool, self._pool = self._pool, None
            manager, self._manager = self._manager, None
            self._housekeeper = None
            pending = [j for j in self._jobs.values() if j.status in ("queued", "running")]
        for job in pending:
            self._finish(job, "cancelled")
        if pool is not None:
            procs = list((getattr(pool, "_processes", None) or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for proc in procs:
                try:
                    proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:  # noqa: BLE001
                pass

    def _ensure_pool(self) -> ProcessPoolExecutor | None:
        """The current pool, restarted first when it is missing or broken."""
        with self._lock:
            pool = self._pool
            if pool is not None and not self._broken:
                return pool
            if self._closing:
                return None
            self._pool = None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass
            log.warning("restarting the worker pool")
        self.start()
        return self._pool

    def _mark_broken(self, pool: ProcessPoolExecutor | None) -> None:
        with self._lock:
            if pool is not None and pool is self._pool:
                self._broken = True

    # ---- submission --------------------------------------------------------------------------
    def cache_key(self, kind: str, payload: dict, warnings: list[str]) -> str:
        return jsonutil.sha256_hex(kind, jsonutil.canonical(payload), jsonutil.canonical(warnings), self.engine_version)

    def submit(self, kind: str, payload: Any, client: str = "") -> Job:
        """Validate + submit. Raises KeyError (unknown kind), ValueError (invalid payload), QueueFull."""
        if kind not in ALL_KINDS:
            raise KeyError(kind)
        norm, warns = payloads.normalize(kind, payload)
        key = self.cache_key(kind, norm, warns)
        job = Job(id=uuid.uuid4().hex[:16], kind=kind, key=key, client=client)
        with self._lock:
            if self._join(job):
                return job
        data = self.cache.get(key)
        if data is not None:
            job.status, job.progress, job.message, job.result, job.cached = "done", 1.0, "cached", data, True
            job.started = job.finished = job.created
            job.event.set()
            with self._lock:
                self._register(job)
            return job
        with self._lock:                      # check-and-register atomically (concurrent identical submits)
            if self._join(job):
                return job
            self._check_capacity(client)
            run = _Run(id=job.id, kind=kind, key=key, payload=norm, warns=warns, jobs=[job])
            job.run = run
            self._inflight[key] = run
            self._register(job)
        self._dispatch(run)
        return job

    def _join(self, job: Job) -> bool:
        """Attach a new handle to an identical in-flight run (lock held)."""
        run = self._inflight.get(job.key)
        if run is None or run.done or not run.jobs:
            return False
        self._check_capacity(job.client)
        lead = run.jobs[0]
        job.run = run
        job.status, job.progress, job.message, job.started = lead.status, lead.progress, lead.message, lead.started
        run.jobs.append(job)
        self._register(job)
        return True

    def _check_capacity(self, client: str) -> None:
        """Admission control (lock held): bounded queue globally and per client."""
        pending = [j for j in self._jobs.values() if j.status not in FINAL]
        if len(pending) >= self.max_pending:
            raise QueueFull(f"the server is busy ({len(pending)} jobs queued or running); try again shortly")
        if client and sum(1 for j in pending if j.client == client) >= self.max_pending_client:
            raise QueueFull(f"too many jobs in progress for this client (limit {self.max_pending_client}); "
                            "wait for them to finish or cancel some")

    def _dispatch(self, run: _Run) -> None:
        for attempt in range(2):
            pool = self._ensure_pool()
            if pool is None:
                self._complete(run, ("cancelled",))
                return
            try:
                fut = pool.submit(_run_job, run.id, run.kind, run.payload, run.warns)
            except (BrokenProcessPool, RuntimeError):
                self._mark_broken(pool)
                if attempt:
                    self._complete(run, ("error", "the worker pool is unavailable", None))
                    return
                continue
            run.future, run.pool = fut, pool
            fut.add_done_callback(lambda f, r=run: self._on_done(r, f))
            return

    def _register(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._evict()

    def _evict(self) -> None:
        """Drop the oldest finished handles beyond max_jobs; finished handles keep at most result_budget bytes of
        results (newest first), older results are re-read from the cache on demand."""
        finished = [jid for jid, j in self._jobs.items() if j.status in FINAL]
        excess = len(self._jobs) - self.max_jobs
        for jid in finished[: max(0, excess)]:
            self._jobs.pop(jid, None)
        budget = self.result_budget
        for j in reversed(self._jobs.values()):
            if j.result is None:
                continue
            if len(j.result) <= budget:
                budget -= len(j.result)
            else:
                j.result = None

    # ---- completion --------------------------------------------------------------------------
    def _on_done(self, run: _Run, fut: Future) -> None:
        try:
            out = fut.result()
        except CancelledError:
            out = ("cancelled",)
        except BrokenProcessPool:
            self._mark_broken(run.pool)
            try:                          # did this attempt start? (only a started run can have caused the crash)
                started = self._progress is not None and self._progress.pop(run.id, None) is not None
            except Exception:  # noqa: BLE001
                started = True
            with self._lock:
                run.attempts += int(started)
                retry = run.attempts <= 1 and bool(run.jobs) and not run.done and not self._closing
            # restart the pool right away (not at the next submit) and retry: the crash may have been caused by
            # another job that shared the pool; a run that crashes twice itself is reported as an error
            threading.Thread(target=self._recover, args=(run if retry else None,), daemon=True).start()
            if retry:
                return
            out = ("error", "the worker process crashed while running this job (out of memory or a crash in the "
                            "numerical code); the worker pool was restarted", None)
        except Exception as exc:  # noqa: BLE001
            out = ("error", f"{type(exc).__name__}: {exc}", traceback.format_exc())
        self._complete(run, out)

    def _recover(self, run: _Run | None) -> None:
        if self._closing:
            return
        try:
            self._ensure_pool()
        except Exception:  # noqa: BLE001
            log.exception("could not restart the worker pool")
        if run is None:
            return
        with self._lock:
            alive = bool(run.jobs) and not run.done
        if alive:
            log.warning("retrying run %s (%s) after a worker crash", run.id, run.kind)
            self._dispatch(run)
        else:
            self._complete(run, ("cancelled",))

    def _complete(self, run: _Run, out: tuple) -> None:
        try:
            if self._progress is not None:
                self._progress.pop(run.id, None)
            if self._cancel is not None:
                self._cancel.pop(run.id, None)
        except Exception:  # noqa: BLE001 - manager gone during shutdown
            pass
        if out[0] == "ok":
            self.cache.put(run.key, out[1])
        elif out[0] == "error" and out[2]:
            log.warning("job %s (%s) failed: %s\n%s", run.id, run.kind, out[1], out[2])
        with self._lock:
            run.done = True
            if self._inflight.get(run.key) is run:
                self._inflight.pop(run.key, None)
            jobs, run.jobs, run.payload = list(run.jobs), [], {}     # release references (finished handles stay)
        for job in jobs:
            if out[0] == "ok":
                self._finish(job, "done", result=out[1])
            elif out[0] == "cancelled":
                self._finish(job, "cancelled")
            else:
                self._finish(job, "error", error=out[1], tb=out[2])

    def _finish(self, job: Job, status: str, result: bytes | None = None, error: str | None = None,
                tb: str | None = None, message: str | None = None) -> None:
        with self._lock:
            if job.status in FINAL:        # e.g. cancelled by the user, the worker finishing later
                return
            job.status, job.run = status, None
            job.finished = job.finished or time.time()
            if status == "done":
                job.progress, job.message, job.result = 1.0, "done", result
            elif status == "error":
                job.message, job.error, job.traceback = "error", error, tb
            else:
                job.message = message or "cancelled"
            self._evict()
        job.event.set()

    # ---- queries -----------------------------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            job.seen = time.monotonic()
            self._refresh(job)
        return job

    def _refresh(self, job: Job) -> None:
        run = job.run
        if job.status not in ("queued", "running") or run is None or self._progress is None:
            return
        try:
            rec = self._progress.get(run.id)
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

    async def wait_async(self, job: Job, timeout: float) -> Job:
        """Long-poll without holding a thread (a blocking wait would exhaust the server's thread pool)."""
        timeout = min(timeout, 3600.0) if math.isfinite(timeout) else 0.0
        deadline = time.monotonic() + timeout
        delay = 0.01
        while job.status not in FINAL:
            job.seen = now = time.monotonic()
            if now >= deadline:
                break
            await asyncio.sleep(min(delay, deadline - now))
            delay = min(delay * 1.5, 0.1)
        return job

    def cancel(self, job_id: str, message: str = "cancelled") -> Job | None:
        """Cancel one handle; the computation stops only when no other handle is waiting for it."""
        job = self.get(job_id)
        if job is None or job.status in FINAL:
            return job
        run = job.run
        self._finish(job, "cancelled", message=message)
        if run is None:
            return job
        with self._lock:
            if job in run.jobs:
                run.jobs.remove(job)
            orphan = not run.jobs and not run.done
            if orphan and self._inflight.get(run.key) is run:
                self._inflight.pop(run.key, None)       # an identical new submission starts a fresh run
        if orphan:
            fut = run.future
            if fut is None or not fut.cancel():
                try:
                    if self._cancel is not None:
                        self._cancel[run.id] = True     # the worker stops at its next progress call
                except Exception:  # noqa: BLE001
                    pass
        return job

    def list(self, client: str | None = None) -> list[Job]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if client is None or j.client == client]
        for j in jobs:
            self._refresh(j)
        return jobs

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for j in self.list():
            out[j.status] = out.get(j.status, 0) + 1
        return out

    # ---- housekeeping ------------------------------------------------------------------------
    def _housekeeping(self) -> None:
        """Every 5 s: cancel abandoned handles; every 10 min (and at start): prune the node cache."""
        last_prune = -1e18
        while True:
            try:
                self._reap_abandoned()
                if time.monotonic() - last_prune > 600.0:
                    last_prune = time.monotonic()
                    if self.node_cache_dir.is_dir():
                        prune_dir(self.node_cache_dir, self.node_cache_max, "*.json")
            except Exception:  # noqa: BLE001
                log.exception("job housekeeping failed")
            if self._stop.wait(5.0):
                return

    def _reap_abandoned(self) -> None:
        if self.abandon_s <= 0:
            return
        now = time.monotonic()
        with self._lock:
            stale = [j.id for j in self._jobs.values() if j.status not in FINAL and now - j.seen > self.abandon_s]
        for jid in stale:
            log.info("cancelling job %s: no client contact for %.0f s", jid, self.abandon_s)
            self.cancel(jid, message="cancelled (abandoned: nobody polled it)")

    # ---- serialisation -----------------------------------------------------------------------
    def snapshot(self, job: Job, include_result: bool = True) -> dict:
        """Refreshed progress + `status()` (blocking IPC / disk: call it from a worker thread in async code)."""
        self._refresh(job)
        return self.status(job, include_result)

    def status(self, job: Job, include_result: bool = True) -> dict:
        """Contract `JobStatus` (result embedded as pre-serialised JSON)."""
        out: dict[str, Any] = dict(job_id=job.id, kind=job.kind, status=job.status, progress=job.progress,
                                   message=job.message, cached=job.cached, elapsed_s=round(job.elapsed(), 3))
        if job.status == "done" and include_result:
            data = job.result if job.result is not None else self.cache.get(job.key)
            if data is not None:
                out["result"] = jsonutil.Fragment(data)
            else:
                out.update(status="error", message="error",
                           error="the result is no longer cached on the server; please run it again")
        if job.status == "error":
            out["error"] = job.error or "unknown error"
        return out
