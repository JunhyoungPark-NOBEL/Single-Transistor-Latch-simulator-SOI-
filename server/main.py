"""FastAPI service for the STL web simulator (docs/WEB_CONTRACT.md §3, docs/API.md).

Run:  uvicorn server.main:app --port 8000        (one uvicorn worker: the compute pool lives in-process)
The API process never imports numba / the engine; compute kinds run in a process pool (server.jobs).
"""
from __future__ import annotations

import importlib.util
import logging
import math
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import FileResponse, HTMLResponse, Response

from server import auth, jsonutil, params
from server.compute import KINDS
from server.compute import data as data_mod
from server.jobs import ALL_KINDS, EXTRA_KINDS, JobManager, QueueFull
from server.payloads import CAPS

APP_VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parents[1]
WEB_DIST = Path(os.environ.get("STL_WEB_DIST") or (ROOT / "web" / "dist"))
MAX_WAIT_S = 60.0
MAX_BODY_BYTES = int(float(os.environ.get("STL_MAX_BODY_KB", "256")) * 1024)   # real payloads are a few kB

log = logging.getLogger("stl.api")


class JSONResponse(Response):
    """orjson-backed response: numpy arrays supported, NaN/±inf → null, pre-serialised fragments embedded."""
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return jsonutil.dumps(content)


class BodySizeLimit:
    """ASGI middleware: reject request bodies larger than `max_bytes` with 413 *before* they are buffered and
    parsed (uvicorn has no limit; a 100 MB JSON body used to cost ~0.4 GB in the API and in a worker)."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers") or ():
            if name == b"content-length":
                try:
                    too_big = int(value) > self.max_bytes
                except ValueError:
                    too_big = True
                if too_big:
                    await self._reject(send)
                    return
        chunks, size = [], 0
        while True:                              # also covers chunked uploads without Content-Length
            message = await receive()
            if message["type"] != "http.request":
                return                           # client went away
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                await self._reject(send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body, replayed = b"".join(chunks), False

        async def replay() -> dict:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    async def _reject(self, send: Any) -> None:
        data = jsonutil.dumps({"detail": f"request body too large (limit {self.max_bytes // 1024} KiB)"})
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": data})


manager = JobManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.start(prewarm=os.environ.get("STL_PREWARM", "1") != "0")
    log.info("STL API: %d workers, engine %s, cache %s", manager.workers, manager.engine_version[:12], manager.cache.dir)
    auth.announce()                                      # stderr: visible under plain uvicorn / host logs
    try:
        yield
    finally:
        manager.shutdown()


app = FastAPI(title="STL simulator API", version=APP_VERSION, default_response_class=JSONResponse, lifespan=lifespan)

_origins = [o.strip() for o in os.environ.get(
    "STL_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173"
).split(",") if o.strip()]
app.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)
# Optional password gate (server/auth.py): pass-through unless STL_ACCESS_PASSWORD is set; then everything except
# /login, /logout, /api/health and the favicon — API, SPA, static files, /docs, /redoc, /openapi.json — needs a
# session cookie. Outside BodySizeLimit so unauthenticated bodies are refused before they are buffered.
app.add_middleware(auth.AccessGate)
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2048)
# Optional Host-header allow-list (outermost): STL_ALLOWED_HOSTS="127.0.0.1,localhost" (the local installer kit sets it)
# answers 400 to any other Host, so a web page that rebinds its own DNS name to 127.0.0.1 cannot read this server
# from the browser (DNS rebinding). Unset = no check (deployments behind a proxy set their own domain here).
_allowed_hosts = [h.strip() for h in os.environ.get("STL_ALLOWED_HOSTS", "").split(",") if h.strip()]
if _allowed_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts, www_redirect=False)


@app.exception_handler(ValueError)
async def _value_error(request: Request, exc: ValueError):
    return JSONResponse({"detail": str(exc) or "invalid input"}, status_code=422)


@app.exception_handler(RequestValidationError)
async def _request_invalid(request: Request, exc: RequestValidationError):
    """Body/query parse errors as {"detail": "message"} like every other error (FastAPI's default is a list that
    echoes the whole input back)."""
    parts = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body" and not isinstance(x, int))
        msg = str(err.get("msg", "invalid"))
        ctx = err.get("ctx") or {}
        if isinstance(ctx, dict) and ctx.get("error"):
            msg += f" ({str(ctx['error'])[:120]})"
        parts.append(f"{loc}: {msg}" if loc else msg)
    return JSONResponse({"detail": "invalid request: " + "; ".join(parts)}, status_code=422)


@app.exception_handler(QueueFull)
async def _queue_full(request: Request, exc: QueueFull):
    return JSONResponse({"detail": str(exc)}, status_code=429, headers={"Retry-After": "10"})


@app.exception_handler(Exception)
async def _internal_error(request: Request, exc: Exception):
    """Unexpected errors: JSON body without internals (the traceback goes to the server log)."""
    return JSONResponse({"detail": "internal server error"}, status_code=500)


def _client(request: Request) -> str:
    """Client identity for per-client admission limits and the job list (proxy-aware when uvicorn runs with
    --proxy-headers, as in the Docker image)."""
    return request.client.host if request.client else ""


def _wait_s(wait: float) -> float:
    return min(max(float(wait), 0.0), MAX_WAIT_S) if math.isfinite(wait) else 0.0


# ---------------------------------------------------------------------------------------------
# service info
# ---------------------------------------------------------------------------------------------
def _kind_available(kind: str) -> bool:
    module = ALL_KINDS[kind].split(":")[0]
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@app.get("/api/health")
def health() -> Any:
    """{ok, version, workers} (+ app_version, engine_version, job counts, access_gate "on" | "off")."""
    return JSONResponse(dict(ok=True, version=manager.engine_version[:12], app_version=APP_VERSION,
                             engine_version=manager.engine_version, workers=manager.workers, jobs=manager.counts(),
                             access_gate=auth.state_name()))


@app.get("/api/meta")
def meta() -> Any:
    """params.meta() + compute kinds + caps."""
    out = params.meta()
    out.update(kinds=list(KINDS), extra_kinds=list(EXTRA_KINDS),
               kinds_available={k: _kind_available(k) for k in ALL_KINDS}, caps=CAPS,
               engine_version=manager.engine_version, app_version=APP_VERSION, workers=manager.workers)
    return JSONResponse(out)


# ---------------------------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------------------------
async def _submit(kind: str, payload: Any, wait: float, client: str) -> Any:
    """Validate + submit in a worker thread, then long-poll on the event loop (no thread held while waiting:
    blocking waits used to exhaust the 40-thread pool and stall every other endpoint)."""
    if kind not in ALL_KINDS:
        raise HTTPException(404, f"unknown compute kind {kind!r}; known: {sorted(ALL_KINDS)}")
    job = await run_in_threadpool(manager.submit, kind, payload, client)   # ValueError → 422, QueueFull → 429
    await manager.wait_async(job, _wait_s(wait))
    return JSONResponse(await run_in_threadpool(manager.snapshot, job))


@app.post("/api/compute/{kind}")
async def compute(request: Request, kind: str, payload: dict[str, Any] | None = Body(default=None),
                  wait: float = Query(2.0, description="seconds to wait for completion before returning")) -> Any:
    """Submit a compute job; returns a JobStatus (with `result` when it finished within `wait`)."""
    return await _submit(kind, payload or {}, wait, _client(request))


@app.get("/api/jobs")
def list_jobs(request: Request) -> Any:
    """The caller's recent jobs (other clients' job ids are not listed: they would allow cancelling them)."""
    return JSONResponse([manager.status(j, include_result=False) for j in manager.list(client=_client(request))])


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, wait: float = Query(0.0, description="optional long-poll seconds")) -> Any:
    job = await run_in_threadpool(manager.get, job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id}")
    await manager.wait_async(job, _wait_s(wait))
    return JSONResponse(await run_in_threadpool(manager.snapshot, job))


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str) -> Any:
    job = manager.cancel(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id}")
    return JSONResponse(manager.status(job))


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _data_bytes(name: str) -> bytes:
    return jsonutil.dumps(data_mod.measured() if name == "measured" else data_mod.design_map())


def _data_response(name: str) -> Response:
    return Response(_data_bytes(name), media_type="application/json", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/data/measured")
def data_measured() -> Response:
    """Measured data: photo raw V_LU (8 conditions × 400 cycles) + stats, light I_D-V_D, dark I_D-V_G,
    paper-device 100-sweep I_D-V_D (median/10-90 % bands/10 samples) + V_LU/V_LD.  Keys: docs/API.md."""
    return _data_response("measured")


@app.get("/api/data/design_map")
def data_design_map() -> Response:
    """design_map_filled.npz: {description, axes, arrays, scalars, shapes, doc}.  Keys: docs/API.md."""
    return _data_response("design_map")


# ---------------------------------------------------------------------------------------------
# handoff aliases (engine/docs/00_START_HERE_websim_KO.md endpoint names)
# ---------------------------------------------------------------------------------------------
def _query_payload(kind: str, q: dict[str, str]) -> dict:
    """Map simple query parameters to a payload (GET aliases)."""
    def f(name: str) -> float | None:
        return float(q[name]) if name in q and q[name] != "" else None

    device: dict[str, Any] = {}
    if "preset" in q:
        device["preset"] = q["preset"]
    if f("vg") is not None:
        device["vg"] = f("vg")
    if f("iph_pA") is not None:
        device["light"] = {"mode": "iph", "iph_pA": f("iph_pA")}
    elif f("power_mW") is not None:
        device["light"] = {"mode": "power", "power_mW": f("power_mW")}
    if f("grid") is not None:
        device["numerics"] = {"grid": int(f("grid"))}
    if f("dg") is not None or f("de") is not None:
        device["state"] = {"delta_phi_G0_V": f("dg") or 0.0, "delta_phi_E0_V": f("de") or 0.0}
    payload: dict[str, Any] = {"device": device}
    sweep = {k2: f(k1) for k1, k2 in (("vd_max", "vd_max_V"), ("rate", "rate_V_per_s"), ("dv", "dv_V")) if f(k1) is not None}
    if sweep:
        payload["sweep"] = sweep
    if kind == "sweep_mc":
        sto: dict[str, Any] = {}
        if f("n") is not None:
            sto["n_cycles"] = int(f("n"))
        if f("seed") is not None:
            sto["seed"] = int(f("seed"))
        if sto:
            payload["stochastic"] = sto
    if kind == "vg_curve":
        for k in ("vg_min", "vg_max"):
            if f(k) is not None:
                payload[k] = f(k)
        if f("n") is not None:
            payload["n"] = int(f("n"))
    if kind == "charge_balance" and f("vd") is not None:
        payload["vd"] = f("vd")
    return payload


def _alias(path: str, kind: str) -> None:
    async def post_alias(request: Request, wait: float = Query(2.0)) -> Any:
        body = await request.body()
        payload = jsonutil.loads(body) if body.strip() else {}      # orjson errors are ValueErrors → 422
        return await _submit(kind, payload, wait, _client(request))

    async def get_alias(request: Request, wait: float = Query(2.0)) -> Any:
        try:
            payload = _query_payload(kind, dict(request.query_params))
        except (ValueError, OverflowError) as exc:
            raise HTTPException(422, f"invalid query parameter: {exc}") from None
        return await _submit(kind, payload, wait, _client(request))

    app.add_api_route(path, post_alias, methods=["POST"], name=f"alias_post_{kind}_{path}",
                      summary=f"alias of POST /api/compute/{kind}")
    app.add_api_route(path, get_alias, methods=["GET"], name=f"alias_get_{kind}_{path}",
                      summary=f"alias of POST /api/compute/{kind} with query parameters")


for _path, _kind in (("/api/branches", "branches"), ("/api/folds", "folds"), ("/api/hazard", "hazard"),
                     ("/api/sweeps", "sweep_mc"), ("/api/vg_curve", "vg_curve")):
    _alias(_path, _kind)


@app.get("/api/design_map")
def design_map_alias() -> Response:
    return _data_response("design_map")


@app.api_route("/api", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
def api_not_found(rest: str = "") -> Any:
    raise HTTPException(404, f"unknown API path /api/{rest}")


# ---------------------------------------------------------------------------------------------
# frontend (web/dist) with SPA fallback
# ---------------------------------------------------------------------------------------------
_NO_FRONTEND = """<!doctype html><html><head><meta charset="utf-8"><title>STL simulator API</title></head>
<body style="font-family:system-ui;max-width:40rem;margin:3rem auto">
<h1>STL simulator API</h1><p>The frontend is not built (<code>web/dist</code> missing).
Run <code>cd web &amp;&amp; npm ci &amp;&amp; npm run build</code>, or use the Vite dev server on :5173.</p>
<p><a href="/api/health">/api/health</a> · <a href="/api/meta">/api/meta</a> · <a href="/docs">/docs</a></p></body></html>"""


@app.get("/{full_path:path}", include_in_schema=False)
def frontend(full_path: str) -> Response:
    if "\x00" in full_path:                             # invalid path, whether or not the frontend is built
        raise HTTPException(404, "not found")
    index = WEB_DIST / "index.html"
    if not index.is_file():
        return HTMLResponse(_NO_FRONTEND)
    if full_path:
        try:
            target = (WEB_DIST / full_path).resolve()
            dist = WEB_DIST.resolve()
            is_file = target.is_file()
        except (OSError, ValueError):                   # e.g. an embedded NUL byte
            raise HTTPException(404, "not found") from None
        if is_file and (target == dist or dist in target.parents):
            headers = {"Cache-Control": "public, max-age=31536000, immutable"} if "/assets/" in f"/{full_path}" else None
            return FileResponse(target, headers=headers)
        last = full_path.rsplit("/", 1)[-1]
        if "." in last and not last.endswith(".html"):   # missing asset → 404, not the SPA shell
            raise HTTPException(404, "not found")
    return FileResponse(index, headers={"Cache-Control": "no-cache"})
