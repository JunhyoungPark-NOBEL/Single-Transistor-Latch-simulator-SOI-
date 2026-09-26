"""Launcher behavior that can otherwise leave duplicate or orphaned servers."""
from pathlib import Path
import socket
import sys

import pytest

import launch


def test_occupied_port_is_not_reused():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(RuntimeError, match="Cannot bind"):
            launch.port_available("127.0.0.1", listener.getsockname()[1])


def test_two_launchers_cannot_prepare_or_serve_the_same_project(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    with launch.instance_lock():
        with pytest.raises(RuntimeError, match="already running"):
            with launch.instance_lock():
                pytest.fail("Second launcher acquired the project lock")
    # A normal or exceptional process exit releases the OS lock; no stale PID cleanup needed.
    with launch.instance_lock():
        pass


def test_prepared_environment_does_not_reinstall(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy>=2.0\n")
    stamp = tmp_path / "stamp"
    stamp.write_text(launch.hashlib.sha256(requirements.read_bytes()).hexdigest())
    monkeypatch.setattr(launch, "REQUIREMENTS", requirements)
    monkeypatch.setattr(launch, "STAMP", stamp)
    monkeypatch.setattr(launch, "environment_python", lambda: Path(sys.executable))
    monkeypatch.setattr(launch, "probe", lambda _: {"errors": []})
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected pip installation"))
    assert launch.prepare(False, False) == Path(sys.executable)


def test_ready_then_browser_then_shutdown(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("test frontend")
    monkeypatch.setattr(launch, "frontend_path", lambda: tmp_path)
    monkeypatch.setattr(launch, "port_available", lambda *a: None)
    events = []

    class Child:
        def poll(self):
            return None

        def wait(self):
            events.append("wait")
            raise KeyboardInterrupt

    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(launch, "is_ready", lambda _: events.append("health") or True)
    monkeypatch.setattr(launch.webbrowser, "open", lambda _: events.append("browser"))
    monkeypatch.setattr(launch, "stop", lambda _: events.append("stop"))
    assert launch.serve(Path(sys.executable), launch.parse_args([])) == 0
    assert events == ["health", "browser", "wait", "stop"]


def test_failed_startup_stops_child_and_does_not_open_browser(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("test frontend")
    monkeypatch.setattr(launch, "frontend_path", lambda: tmp_path)
    monkeypatch.setattr(launch, "port_available", lambda *a: None)
    stopped = []

    class Child:
        def poll(self):
            return None

    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(launch, "is_ready", lambda _: False)
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(launch.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(launch.webbrowser, "open", lambda _: pytest.fail("Opened unhealthy server"))
    monkeypatch.setattr(launch, "stop", lambda child: stopped.append(child))
    with pytest.raises(RuntimeError, match="did not become healthy"):
        launch.serve(Path(sys.executable), launch.parse_args(["--startup-timeout", "1"]))
    assert len(stopped) == 1


def test_current_python_missing_dependencies_never_mutates_environment(monkeypatch):
    monkeypatch.setattr(launch, "probe", lambda _: {"errors": ["numpy: missing"]})
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected environment mutation"))
    with pytest.raises(RuntimeError, match="Dependencies are missing"):
        launch.prepare(True, False)


@pytest.mark.parametrize("host,url", [("0.0.0.0", "http://127.0.0.1:8000"), ("::", "http://[::1]:8000")])
def test_wildcard_bind_opens_concrete_loopback_address(host, url):
    assert launch.local_url(host, 8000) == url
