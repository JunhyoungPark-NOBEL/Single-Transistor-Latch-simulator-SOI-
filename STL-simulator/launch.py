#!/usr/bin/env python3
"""STL simulator launcher. Standard library only; never imports the numerical engine.

Default: prepare .venv, start one API process, open the browser after /api/health.
--check inspects the selected interpreter and bundled web build without installing or serving.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import webbrowser

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "server" / "requirements.txt"
VENV = ROOT / ".venv"
STAMP = VENV / ".studio-requirements.sha256"
MODULES = ("numpy", "scipy", "numba", "fastapi", "uvicorn", "orjson", "httpx")


@contextmanager
def instance_lock():
    """One launcher per project, including first-time dependency setup; released on exit/crash."""
    with (ROOT / ".studio-launch.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another STL simulator launcher is already running in this project.") from exc
        yield


def environment_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def probe(python: Path) -> dict:
    """Imports happen in a disposable process, never in the API process."""
    script = """
import importlib, json, sys
result = {'python': sys.version.split()[0], 'errors': [], 'versions': {}}
for name in %r:
    try:
        module = importlib.import_module(name)
        result['versions'][name] = getattr(module, '__version__', 'available')
    except Exception as exc:
        result['errors'].append(name + ': ' + str(exc))
print(json.dumps(result))
""" % (MODULES,)
    completed = subprocess.run([str(python), "-c", script], cwd=ROOT,
                               capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode:
        raise RuntimeError("Python dependency check failed: " + completed.stderr.strip()[-1500:])
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Python dependency check returned an invalid result.") from exc


def frontend_path() -> Path:
    return Path(os.environ.get("STL_WEB_DIST") or (ROOT / "web" / "dist"))


def check(python: Path) -> bool:
    info = probe(python)
    print(f"Python {info['python']}: {python}", flush=True)
    for name, version in info["versions"].items():
        print(f"  {name}: {version}", flush=True)
    for error in info["errors"]:
        print("  MISSING/BROKEN: " + error, flush=True)
    web_ok = (frontend_path() / "index.html").is_file()
    print(f"Frontend: {'ready' if web_ok else 'MISSING'} ({frontend_path()})", flush=True)
    if not info["errors"]:
        # Check API import separately so dependency probing cannot mask accidental engine imports.
        completed = subprocess.run(
            [str(python), "-c", "import sys; import server.main; "
             "assert 'numba' not in sys.modules, 'API imported the numerical engine'; "
             "print('API import: ready (engine isolated in workers)')"],
            cwd=ROOT, capture_output=True, text=True, timeout=60, check=False,
        )
        print(completed.stdout.strip() if completed.returncode == 0 else
              "API import: FAILED\n" + completed.stderr.strip()[-1500:], flush=True)
        return web_ok and completed.returncode == 0
    return False


def prepare(use_current: bool, no_install: bool) -> Path:
    python = Path(sys.executable) if use_current else environment_python()
    if not python.is_file():
        if no_install:
            raise RuntimeError(".venv is missing. Run launch.py once without --no-install.")
        print("Creating project .venv ...", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV)
    fingerprint = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    info = probe(python)
    stale = not use_current and (not STAMP.is_file() or STAMP.read_text().strip() != fingerprint)
    if info["errors"] or stale:
        if no_install or use_current:
            if info["errors"]:
                raise RuntimeError("Dependencies are missing/broken. Install server/requirements.txt in "
                                   "this interpreter, or run without --use-current-python/--no-install.")
            # Existing installed environments may be used without a launcher stamp.
        else:
            print("Installing server/requirements.txt (first run or changed requirements) ...", flush=True)
            subprocess.run([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
                           cwd=ROOT, check=True)
            info = probe(python)
            if info["errors"]:
                raise RuntimeError("Dependency import check failed after installation: " + "; ".join(info["errors"]))
            STAMP.write_text(fingerprint + "\n")
    return python


def port_available(host: str, port: int) -> None:
    """Fail visibly instead of opening an unrelated existing server."""
    last_error = None
    for family, socktype, proto, _, address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        try:
            with socket.socket(family, socktype, proto) as candidate:
                if os.name == "nt":
                    candidate.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                candidate.bind(address)
            return
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"Cannot bind {host}:{port}: {last_error}. Stop the existing server or use --port.")


def local_url(host: str, port: int) -> str:
    target = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    return f"http://{'[' + target + ']' if ':' in target else target}:{port}"


def is_ready(url: str) -> bool:
    # Loopback readiness must not be routed through an HTTP proxy from the user's shell.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url + "/api/health", timeout=1) as response:
            data = json.load(response)
        return data.get("ok") is True and bool(data.get("engine_version"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    print("Stopping STL simulator ...", flush=True)
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=20)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)


def serve(python: Path, args: argparse.Namespace) -> int:
    if not (frontend_path() / "index.html").is_file():
        raise RuntimeError("web/dist/index.html is missing. In web/, run npm ci and npm run build.")
    port_available(args.host, args.port)
    env = os.environ.copy()
    env["STL_WORKERS"] = str(args.workers)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(name, "1")
    command = [str(python), "-m", "uvicorn", "server.main:app", "--host", args.host,
               "--port", str(args.port), "--proxy-headers", "--forwarded-allow-ips",
               env.get("FORWARDED_ALLOW_IPS", "127.0.0.1")]
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(command, cwd=ROOT, env=env, **options)
    try:
        url = local_url(args.host, args.port)
        deadline = time.monotonic() + args.startup_timeout
        while process.poll() is None:
            if is_ready(url):
                print(f"\nSTL simulator ready: {url}\nKeep this terminal open. Stop: Ctrl+C.\n", flush=True)
                if not args.no_browser:
                    webbrowser.open(url)
                return process.wait()
            if time.monotonic() >= deadline:
                raise RuntimeError("Server did not become healthy before the startup timeout. "
                                   "Check logs above and STL_REQUIRE_PASSWORD/STL_ACCESS_PASSWORD; "
                                   "use --startup-timeout for slow machines.")
            time.sleep(0.25)
        raise RuntimeError(f"Server exited before startup (exit code {process.returncode}).")
    except KeyboardInterrupt:
        return 0
    finally:
        stop(process)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STL simulator locally or on a lab server.")
    parser.add_argument("--check", action="store_true", help="Check dependencies/API imports/web build; do not install or serve.")
    parser.add_argument("--setup-only", action="store_true", help="Prepare .venv/dependencies without starting the server.")
    parser.add_argument("--use-current-python", action="store_true", help="Use the current provisioned interpreter; do not install packages.")
    parser.add_argument("--no-install", action="store_true", help="Never create an environment or install packages.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser (lab/server mode).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; default 127.0.0.1, use 0.0.0.0 for explicit LAN access.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("STL_WORKERS", "2")), help="Numerical worker count, not uvicorn workers.")
    parser.add_argument("--startup-timeout", type=float, default=120, help="Seconds to wait for health (default 120).")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be 1..65535")
    if args.workers < 1 or args.startup_timeout <= 0:
        parser.error("--workers and --startup-timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < (3, 11):
        print("Python 3.11 or newer is required (3.11/3.12 recommended).", file=sys.stderr)
        return 1
    args = parse_args(argv)
    try:
        if args.check:
            python = Path(sys.executable) if args.use_current_python or not environment_python().is_file() else environment_python()
            return 0 if check(python) else 1
        with instance_lock():
            python = prepare(args.use_current_python, args.no_install)
            if args.setup_only:
                return 0 if check(python) else 1
            def interrupted(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, interrupted)
            return serve(python, args)
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"STL simulator: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
