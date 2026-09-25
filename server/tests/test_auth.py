"""Password gate (server/auth.py): off by default; with a password every route except /login, /logout, /api/health
and the favicon needs a session cookie.  Dummy passwords only (the real one is a deploy-time secret)."""
from __future__ import annotations

import time
import warnings
from urllib.parse import parse_qs, urlsplit

import pytest

from server import auth

PW = "test-pass-123"
OTHER_PW = "other-pass-456"
SECRET = "test-session-secret-0123456789abcdef"


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _client(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    from server.main import app
    return TestClient(app, follow_redirects=False, **kwargs)     # no lifespan: the gate answers before any route


@pytest.fixture
def restore_gate():
    saved = auth.STATE.config, auth.STATE.misconfigured
    try:
        yield
    finally:
        auth.STATE.config, auth.STATE.misconfigured = saved


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def gate(restore_gate, clock):
    """Gate on with a dummy password (cheap PBKDF2 for speed) and a fake monotonic clock for the limiter."""
    return auth.configure(PW, iterations=1000, monotonic=clock)


@pytest.fixture
def c(gate):
    return _client()


def _login(client, password=PW, headers=None, **fields):
    return client.post("/login", data={"password": password, **fields}, headers=headers)


def _asset_path() -> str:
    from server.main import WEB_DIST
    assets = WEB_DIST / "assets"
    if assets.is_dir():
        for p in sorted(assets.iterdir()):
            if p.is_file():
                return f"/assets/{p.name}"
    return "/assets/index-deadbeef.js"


# ---------------------------------------------------------------------------------------------
# off by default
# ---------------------------------------------------------------------------------------------
def test_off_unless_password_set():
    assert auth.config_from_env({}) is None
    assert auth.config_from_env({"STL_ACCESS_PASSWORD": ""}) is None
    assert auth.config_from_env({"STL_SESSION_SECRET": SECRET, "STL_TRUST_PROXY": "1"}) is None
    cfg = auth.config_from_env({"STL_ACCESS_PASSWORD": PW + "\n", "STL_TRUST_PROXY": "1"})
    assert cfg is not None and cfg.trust_proxy == 1 and cfg.iterations == auth.PBKDF2_ITERATIONS
    assert cfg.check_password(PW) and not cfg.check_password(PW + "\n") and not cfg.check_password("")
    assert cfg.secret_source == "per-process random"
    assert auth.config_from_env({"STL_ACCESS_PASSWORD": PW, "STL_SESSION_SECRET": SECRET}).secret_source == \
        "STL_SESSION_SECRET"
    assert [auth._parse_trust(v) for v in ("", "0", "false", "1", "true", "2", "junk")] == [0, 0, 0, 1, 1, 2, 1]


def test_gate_off_behaviour_unchanged(restore_gate):
    auth.configure(None)
    c = _client()
    r = c.get("/api/meta")
    assert r.status_code == 200 and "presets" in r.json()
    r = c.get("/login")                              # no login route: the SPA fallback answers as before
    assert r.status_code == 200 and "set-cookie" not in r.headers
    assert "Incorrect password" not in r.text and 'name="password"' not in r.text
    r = c.get("/api/health")
    assert r.status_code == 200 and r.json()["access_gate"] == "off"
    assert c.get("/openapi.json").status_code == 200


def test_password_value_cleaned(caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="stl.auth")
    cfg = auth.config_from_env({"STL_ACCESS_PASSWORD": "  " + PW + " \r\n"})     # dashboard paste
    assert cfg.check_password(PW) and not cfg.check_password(PW + " ")
    assert "whitespace" in caplog.text and PW not in caplog.text
    assert auth.config_from_env({"STL_ACCESS_PASSWORD": " \t \n"}) is None          # whitespace only = not set
    caplog.clear()
    cfg = auth.config_from_env({"STL_ACCESS_PASSWORD": f'"{PW}"'})                  # docker --env-file keeps quotes
    assert cfg.check_password(f'"{PW}"') and "quote" in caplog.text and PW not in caplog.text


# ---------------------------------------------------------------------------------------------
# fail closed / visible state
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", [None, "", "  ", "\n"])
def test_require_password_fails_closed(restore_gate, value):
    env = {"STL_REQUIRE_PASSWORD": "1"}
    if value is not None:
        env["STL_ACCESS_PASSWORD"] = value
    with pytest.raises(auth.GateMisconfigured):
        auth.config_from_env(env)
    assert auth.configure_from_env(env) is None and auth.state_name() == "misconfigured"
    assert "MISCONFIGURED" in auth.describe()
    c = _client()
    for method, path in [("GET", "/api/health"), ("GET", "/"), ("GET", "/login"), ("POST", "/login"),
                         ("GET", "/api/meta"), ("GET", "/favicon.svg"), ("GET", _asset_path()), ("GET", "/logout")]:
        r = c.request(method, path)
        assert r.status_code == 503, (method, path, r.status_code)
        assert r.json()["error"] == "access control not configured" and "set-cookie" not in r.headers
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws"):
            pass


def test_require_password_with_password_is_on(restore_gate):
    cfg = auth.configure_from_env({"STL_REQUIRE_PASSWORD": "1", "STL_ACCESS_PASSWORD": PW})
    assert cfg is not None and cfg.check_password(PW) and auth.state_name() == "on"
    assert auth.configure_from_env({"STL_REQUIRE_PASSWORD": "0"}) is None and auth.state_name() == "off"


def test_announce_prints_state(restore_gate):
    import io
    out = io.StringIO()
    auth.configure(None)
    auth.announce(out)
    assert "access gate OFF" in out.getvalue() and "anyone with the URL" in out.getvalue()
    out = io.StringIO()
    auth.configure(PW, session_secret=SECRET, iterations=1000)
    auth.announce(out)
    assert "access gate ON" in out.getvalue() and PW not in out.getvalue() and SECRET not in out.getvalue()


def test_password_not_kept_in_config(gate):
    assert PW not in repr(vars(gate)) and PW.encode() not in b"".join(
        v for v in vars(gate).values() if isinstance(v, bytes))


# ---------------------------------------------------------------------------------------------
# everything protected
# ---------------------------------------------------------------------------------------------
API_DENIED = [
    ("GET", "/api/meta"), ("POST", "/api/compute/branches"), ("POST", "/api/compute/circuit"),
    ("GET", "/api/jobs"), ("GET", "/api/jobs/abc"), ("DELETE", "/api/jobs/abc"),
    ("GET", "/api/data/measured"), ("GET", "/api/data/design_map"), ("GET", "/api/design_map"),
    ("GET", "/api/branches"), ("POST", "/api/sweeps"), ("GET", "/api/vg_curve"), ("GET", "/api"),
    ("GET", "/api/unknown"), ("GET", "/openapi.json"), ("GET", "/API/meta.json"),
    ("POST", "/"), ("PUT", "/index.html"), ("DELETE", "/docs"), ("OPTIONS", "/api/meta"),
    ("GET", "/api/health/"), ("GET", "/api/healthz"),
]
PAGES_DENIED = ["/", "/index.html", "/docs", "/redoc", "/docs/oauth2-redirect", "/circuit", "/some/spa/route",
                "/favicon.svg/x", "/login.html"]


@pytest.mark.parametrize("method,path", API_DENIED)
def test_api_denied_without_session(c, method, path):
    r = c.request(method, path, json={"device": {"preset": "paper"}} if method in ("POST", "PUT") else None)
    assert r.status_code == 401, (method, path, r.status_code)
    assert r.json()["error"] == "login required"
    assert r.headers["cache-control"] == "no-store" and "set-cookie" not in r.headers


def test_static_asset_denied(c):
    r = c.get(_asset_path())
    assert r.status_code == 401 and r.json()["error"] == "login required"
    assert len(r.content) < 200                          # nothing of the file leaks


@pytest.mark.parametrize("path", PAGES_DENIED)
@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_pages_redirect_to_login(c, method, path):
    r = c.request(method, path)
    assert r.status_code == 303, (method, path, r.status_code)
    loc = urlsplit(r.headers["location"])
    assert loc.path == "/login" and not loc.netloc
    assert parse_qs(loc.query)["next"] == [path]
    assert r.content == b"" and r.headers["cache-control"] == "no-store"


def test_denied_compute_does_not_submit(c):
    from server.main import manager
    before = len(manager.list())
    big = {"device": {"preset": "paper"}, "pad": "x" * (1 << 20)}       # also: no 413, body never buffered
    assert c.post("/api/compute/branches", params={"wait": 5}, json=big).status_code == 401
    assert c.post("/api/branches", json={}).status_code == 401
    assert len(manager.list()) == before


def test_websocket_refused(c):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/ws"):
            pass
    assert exc.value.code == 1008


def test_health_and_favicon_open(c):
    r = c.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True and r.json()["access_gate"] == "on"
    for path in ("/favicon.svg", "/favicon.ico"):
        assert c.get(path).status_code not in (303, 401)


def test_query_string_preserved_in_next(c):
    r = c.get("/circuit", params={"tab": "schematic", "x": "1"})
    nxt = parse_qs(urlsplit(r.headers["location"]).query)["next"][0]
    assert nxt == "/circuit?tab=schematic&x=1"
    r = _login(c, next=nxt)
    assert r.status_code == 303 and r.headers["location"] == "/circuit?tab=schematic&x=1"


# ---------------------------------------------------------------------------------------------
# login page and login
# ---------------------------------------------------------------------------------------------
def test_login_page(c):
    r = c.get("/login", params={"next": "/circuit"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    t = r.text
    import re
    assert re.sub(r"</?span>", "", t).count("KAIST · NOBEL 연구실") == 1
    assert "지도교수" not in t and "개발" not in t          # reduced credits (owner decision D2), as in the app
    assert 'method="post" action="/login"' in t and 'name="password"' in t and 'type="password"' in t
    assert 'name="next" value="/circuit"' in t and 'name="remember"' in t
    assert "비밀번호" in t and "Password" in t and "prefers-color-scheme:dark" in t
    assert "<script" not in t.lower()
    assert r.headers["cache-control"] == "no-store" and "noindex" in r.headers["x-robots-tag"]
    assert "form-action 'self'" in r.headers["content-security-policy"]
    assert c.head("/login").status_code == 200
    assert c.put("/login").status_code == 405


def test_login_page_escapes_next(c):
    t = c.get("/login", params={"next": '/"><script>alert(1)</script>'}).text
    assert "<script>" not in t and 'name="next" value="/"' in t


@pytest.mark.parametrize("password", ["wrong", "", PW.upper(), PW + " ", " " + PW, PW[:-1]])
def test_login_wrong(c, password):
    r = _login(c, password)
    assert r.status_code == 401 and "set-cookie" not in r.headers
    assert "Incorrect password" in r.text and "비밀번호가 올바르지 않습니다" in r.text
    assert PW not in r.text
    assert c.get("/api/meta").status_code == 401


def test_login_right_session_cookie(c):
    r = _login(c)
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("stl_session=v1.")
    for attr in ("HttpOnly", "SameSite=Lax", "Path=/"):
        assert attr in cookie
    assert "Max-Age" not in cookie and "Expires" not in cookie and "Secure" not in cookie    # browser session
    token = cookie.split(";")[0].split("=", 1)[1]
    issued, expires = (int(x) for x in token.split(".")[1:3])
    assert expires - issued == auth.SESSION_S and abs(issued - time.time()) < 5
    for path in ("/api/meta", "/", "/index.html", "/docs", "/redoc", "/openapi.json", "/api/health"):
        assert c.get(path).status_code == 200, path
    assert c.get("/api/jobs/unknown-job").status_code == 404                  # reaches the route
    assert c.get("/login", params={"next": "/circuit"}).headers["location"] == "/circuit"   # already signed in


def test_login_remember(c):
    r = _login(c, remember="1")
    cookie = r.headers["set-cookie"]
    assert f"Max-Age={auth.REMEMBER_S}" in cookie and auth.REMEMBER_S == 30 * 86400
    token = cookie.split(";")[0].split("=", 1)[1]
    issued, expires = (int(x) for x in token.split(".")[1:3])
    assert expires - issued == auth.REMEMBER_S


def test_authenticated_cache_headers_private(c):
    _login(c)
    r = c.get("/api/data/design_map")
    assert r.status_code == 200
    assert r.headers["cache-control"].startswith("private") and "public" not in r.headers["cache-control"]


def test_secure_cookie_on_https(gate):
    r = _login(_client(base_url="https://testserver"))
    assert r.status_code == 303 and "Secure" in r.headers["set-cookie"]
    r = _login(_client(), headers={"X-Forwarded-Proto": "https"})             # behind Render / HF proxies
    assert "Secure" in r.headers["set-cookie"]
    r = _login(_client(), headers={"X-Forwarded-Proto": "http"})
    assert "Secure" not in r.headers["set-cookie"]


def test_https_client_uses_session(gate):
    c = _client(base_url="https://testserver")
    assert _login(c).status_code == 303
    assert c.get("/api/meta").status_code == 200


def test_login_body_limits(c):
    r = c.post("/login", content=b"password=" + b"x" * (auth.LOGIN_BODY_MAX + 10),
               headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 413 and "set-cookie" not in r.headers
    r = c.post("/login", content=b"password=\xff\xfe", headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 400
    assert _login(c, "x" * (auth.PASSWORD_MAX + 1)).status_code == 401


# ---------------------------------------------------------------------------------------------
# tokens: tamper, expiry, password change
# ---------------------------------------------------------------------------------------------
def test_token_tamper(gate):
    tok = gate.make_token(False)
    assert gate.check_token(tok)
    v, iat, exp, nonce, mac = tok.split(".")
    flipped = mac[:-1] + ("A" if mac[-1] != "A" else "B")
    bad = [
        f"{v}.{iat}.{exp}.{nonce}.{flipped}",
        f"{v}.{iat}.{int(exp) + 86400 * 365}.{nonce}.{mac}",
        f"{v}.{int(iat) - 1}.{exp}.{nonce}.{mac}",
        f"{v}.{iat}.{exp}.{nonce[::-1]}.{mac}",
        f"v2.{iat}.{exp}.{nonce}.{mac}",
        f"{v}.{iat}.{exp}.{mac}",
        tok + ".x", "", "garbage", "v1....", "v1.a.b.c.d", f"{v}.{iat}.{exp}.{nonce}.{mac}é", "v1." + "9" * 300,
        f"{v}.-{iat}.{exp}.{nonce}.{mac}",
    ]
    for t in bad:
        assert not gate.check_token(t), t


def test_token_expiry(gate):
    t0 = 2_000_000_000
    tok = gate.make_token(False, now=t0)
    assert gate.check_token(tok, now=t0) and gate.check_token(tok, now=t0 + auth.SESSION_S - 1)
    assert not gate.check_token(tok, now=t0 + auth.SESSION_S)
    tok = gate.make_token(True, now=t0)
    assert gate.check_token(tok, now=t0 + auth.REMEMBER_S - 1) and not gate.check_token(tok, now=t0 + auth.REMEMBER_S)
    assert not gate.check_token(gate.make_token(False, now=t0 + 3600), now=t0)          # issued in the future


def test_tampered_or_expired_cookie_over_http(c, gate):
    c.cookies.set("stl_session", gate.make_token(False, now=time.time() - auth.SESSION_S - 5))
    assert c.get("/api/meta").status_code == 401
    r = c.get("/circuit")
    q = parse_qs(urlsplit(r.headers["location"]).query)
    assert r.status_code == 303 and q["next"] == ["/circuit"] and q["expired"] == ["1"]
    assert "session has expired" in c.get(r.headers["location"]).text
    good = gate.make_token(False)
    c.cookies.set("stl_session", good[:-2] + ("xx" if not good.endswith("xx") else "yy"))
    assert c.get("/api/meta").status_code == 401
    c.cookies.set("stl_session", good)
    assert c.get("/api/meta").status_code == 200


def test_password_change_invalidates_sessions(restore_gate):
    cfg = auth.configure(PW, session_secret=SECRET, iterations=1000)
    tok = cfg.make_token(True)
    assert auth.configure(PW, session_secret=SECRET, iterations=1000).check_token(tok)      # restart, same secret
    assert not auth.configure(OTHER_PW, session_secret=SECRET, iterations=1000).check_token(tok)  # password changed
    assert not auth.configure(PW, session_secret=SECRET + "x", iterations=1000).check_token(tok)  # secret rotated
    per_process = auth.configure(PW, iterations=1000)
    assert not auth.configure(PW, iterations=1000).check_token(per_process.make_token(True))    # restart w/o secret
    # over HTTP: a cookie issued before the change is refused afterwards
    auth.configure(PW, session_secret=SECRET, iterations=1000)
    c = _client()
    assert _login(c).status_code == 303 and c.get("/api/meta").status_code == 200
    auth.configure(OTHER_PW, session_secret=SECRET, iterations=1000)
    assert c.get("/api/meta").status_code == 401
    assert _login(c).status_code == 401 and _login(c, OTHER_PW).status_code == 303
    assert c.get("/api/meta").status_code == 200


# ---------------------------------------------------------------------------------------------
# redirect safety
# ---------------------------------------------------------------------------------------------
BAD_NEXT = ["https://evil.example", "//evil.example", "//evil.example/path", "/\\evil.example", "\\\\evil.example",
            "/\t/evil.example", "/\n/evil.example", " /x", "javascript:alert(1)", "http:/evil.example",
            "evil.example", "", "/login", "/login?next=//evil.example", "/logout", "/é", "/" + "a" * 3000,
            "///evil.example", "https:evil.example", "/%6cogout", "/%6Cogin", "/x/../logout", "/./logout",
            "/%2e%2e/logout", "/logout/", "/%2Flogin/../logout"]


@pytest.mark.parametrize("nxt", BAD_NEXT)
def test_no_open_redirect(c, nxt):
    assert auth.safe_next(nxt) == "/"
    r = _login(c, next=nxt)
    assert r.status_code == 303 and r.headers["location"] == "/", (nxt, r.headers["location"])


@pytest.mark.parametrize("nxt", ["/", "/circuit", "/a/b?c=d&e=%2F%2Fx", "/?lang=en#physics", "/%2F%2Fevil.example"])
def test_relative_next_kept(nxt):
    assert auth.safe_next(nxt) == nxt


# ---------------------------------------------------------------------------------------------
# brute-force protection
# ---------------------------------------------------------------------------------------------
def test_backoff_over_http(c, clock):
    for _ in range(auth.FREE_FAILURES):
        assert _login(c, "wrong").status_code == 401
    r = _login(c)                                          # even the right password waits
    assert r.status_code == 429 and r.headers["retry-after"] == "30" and "set-cookie" not in r.headers
    assert "Too many attempts" in r.text and "30" in r.text
    clock.advance(29)
    assert _login(c).headers.get("retry-after") == "1"
    clock.advance(2)
    assert _login(c, "wrong").status_code == 401            # 6th failure → 60 s
    r = _login(c)
    assert r.status_code == 429 and r.headers["retry-after"] == "60"
    clock.advance(61)
    assert _login(c).status_code == 303                     # success resets the count
    assert _login(_client(), "wrong").status_code == 401    # same client IP, fresh count


def test_backoff_doubles_and_caps(clock):
    lim = auth.Limiter(clock=clock)
    locks = []
    for _ in range(12):
        clock.advance(lim.retry_after("a") + 0.001)
        lim.reserve("a")
        locks.append(round(lim.retry_after("a")))
        clock.advance(auth.GLOBAL_WINDOW_S)                 # keep the global window out of the way
    assert locks[:4] == [0, 0, 0, 0]
    assert locks[4:] == [30, 60, 120, 240, 480, 900, 900, 900]


def test_parallel_attempts_are_counted_before_the_check(clock):
    lim = auth.Limiter(clock=clock)
    tickets = [lim.reserve("a") for _ in range(auth.FREE_FAILURES)]    # 5 in flight, none finished yet
    assert lim.retry_after("a") == pytest.approx(30)
    lim.success("a", tickets[-1])
    assert lim.retry_after("a") == 0


def test_global_cap(clock, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="stl.auth")
    lim = auth.Limiter(clock=clock)
    for i in range(auth.GLOBAL_MAX_FAILURES):
        assert lim.retry_after(f"ip{i}") == 0
        lim.reserve(f"ip{i}")
        clock.advance(0.1)
    # window full: clients that already failed wait until it has room …
    assert lim.retry_after("ip0") == pytest.approx(auth.GLOBAL_WINDOW_S - 0.1 * auth.GLOBAL_MAX_FAILURES)
    assert "server-wide" not in caplog.text
    # … but a client without recent failures still gets one check (GRACE_MAX of them per window)
    for i in range(auth.GRACE_MAX):
        assert lim.retry_after(f"fresh{i}") == 0
        lim.reserve(f"fresh{i}")
        assert lim.retry_after(f"fresh{i}") > 0                     # its second try waits for the window
    assert caplog.text.count("server-wide") == 1                   # warned once per window
    assert lim.retry_after("fresh-late") > 0                        # grace budget spent
    clock.advance(auth.GLOBAL_WINDOW_S + 0.1)
    assert lim.retry_after("fresh-late") == 0 and lim.retry_after("ip0") == 0
    assert len(lim.recent) == 0 and len(lim.grace) == 0


def test_global_cap_does_not_lock_out_everyone(restore_gate, clock):
    """An attacker who fills the window (rotating addresses) cannot keep a legitimate user from signing in."""
    auth.configure(PW, trust_proxy=1, iterations=1000, monotonic=clock)
    c = _client()
    for i in range(auth.GLOBAL_MAX_FAILURES):
        r = c.post("/login", data={"password": "wrong"}, headers={"X-Forwarded-For": f"198.51.100.{i}"})
        assert r.status_code == 401
    r = c.post("/login", data={"password": "wrong"}, headers={"X-Forwarded-For": "198.51.100.0"})
    assert r.status_code == 429                                          # attacker address: waits for the window
    r = c.post("/login", data={"password": PW}, headers={"X-Forwarded-For": "203.0.113.50"})
    assert r.status_code == 303 and "stl_session=v1." in r.headers["set-cookie"]     # new visitor gets in
    cfg = auth.STATE.config
    assert len(cfg.limiter.grace) == 0                                   # a successful check gives its slot back
    total = auth.GLOBAL_MAX_FAILURES
    for i in range(50):                                                  # fresh addresses: ≤ GRACE_MAX more checks
        r = c.post("/login", data={"password": "wrong"}, headers={"X-Forwarded-For": f"192.0.2.{i}"})
        total += r.status_code == 401
    assert total == auth.GLOBAL_MAX_FAILURES + auth.GRACE_MAX


def test_limiter_forgets_idle_clients(clock):
    lim = auth.Limiter(clock=clock)
    for i in range(300):
        lim.reserve(f"ip{i}")
    clock.advance(auth.FORGET_S + 1)
    lim.reserve("new")
    assert len(lim.entries) == 1


def test_limiter_forgets_old_typos_in_small_deployments(clock):
    lim = auth.Limiter(clock=clock)
    for _ in range(auth.FREE_FAILURES - 1):
        lim.reserve("lab-pc")
    clock.advance(7 * 86400)                                       # a week later, one typo
    lim.reserve("lab-pc")
    assert lim.retry_after("lab-pc") == 0 and lim.entries["lab-pc"].failures == 1
    for _ in range(auth.FREE_FAILURES - 1):                        # but a burst still locks
        lim.reserve("lab-pc")
    assert lim.retry_after("lab-pc") == pytest.approx(auth.BASE_LOCK_S)


def test_limiter_lock_exponent_cannot_overflow(clock):
    lim = auth.Limiter(clock=clock)
    lim.entries["a"] = auth._Entry(failures=5000, last=clock())
    lim.reserve("a")                                                # 2.0 ** 4996 would raise OverflowError
    assert lim.retry_after("a") == pytest.approx(auth.MAX_LOCK_S)


def test_forwarded_for_ignored_unless_trusted(c):
    for i in range(auth.FREE_FAILURES):                      # rotating the header gives no fresh count …
        r = c.post("/login", data={"password": "wrong"}, headers={"X-Forwarded-For": f"203.0.113.{i}"})
        assert r.status_code == 401
    r = c.post("/login", data={"password": PW}, headers={"X-Forwarded-For": "198.51.100.1"})
    assert r.status_code == 429
    r = c.post("/login", data={"password": PW}, headers={"X-Forwarded-For": "testclient"})
    assert r.status_code == 429
    assert _login(c).status_code == 303                     # … and naming another address cannot lock it out


def test_uvicorn_rewritten_client_not_trusted():
    """uvicorn --forwarded-allow-ips='*' puts the client-supplied left-most X-Forwarded-For entry into
    scope["client"]; without STL_TRUST_PROXY such requests all share one key."""
    for xff in (b"192.0.2.55", b"10.9.1.1", b"127.0.0.1, 10.0.0.1"):
        scope = {"client": (xff.split(b",")[0].decode(), 0), "headers": [(b"x-forwarded-for", xff)]}
        assert auth.client_key(scope, 0) == auth.UNTRUSTED_FORWARDED_KEY
    assert auth.client_key({"client": ("192.0.2.55", 51000), "headers": []}, 0) == "192.0.2.55"


def test_ipv6_grouped_by_64():
    k = auth._norm_ip
    assert k("2001:db8:1:2:aaaa::1") == k("2001:db8:1:2:bbbb:cccc:dddd:9") == "2001:db8:1:2::/64"
    assert k("2001:db8:1:3::1") != k("2001:db8:1:2::1")
    assert k("[2001:db8::1]:443") == "2001:db8::/64" and k("::ffff:192.0.2.1") == "192.0.2.1"
    assert k("192.0.2.1") == "192.0.2.1" and k("192.0.2.1:8080") == "192.0.2.1" and k(" 192.0.2.1 ") == "192.0.2.1"
    assert k("fe80::1%eth0") == "fe80::/64" and k("unknown-host") == "unknown-host" and k("") == "unknown"
    scope = {"client": ("2001:db8:5:6::7", 1), "headers": []}
    assert auth.client_key(scope, 0) == "2001:db8:5:6::/64"
    scope = {"client": ("10.0.0.1", 1), "headers": [(b"x-forwarded-for", b"2001:db8:5:6::1, 2001:db8:9:9::abcd")]}
    assert auth.client_key(scope, 1) == "2001:db8:9:9::/64"


def test_trusted_proxy_hop(restore_gate, clock):
    auth.configure(PW, trust_proxy=1, iterations=1000, monotonic=clock)
    c = _client()
    for _ in range(auth.FREE_FAILURES):
        c.post("/login", data={"password": "wrong"}, headers={"X-Forwarded-For": "1.1.1.1, 203.0.113.9"})
    # the client-supplied left part changes, the proxy-appended right-most address does not → still locked
    r = c.post("/login", data={"password": PW}, headers={"X-Forwarded-For": "8.8.8.8, 203.0.113.9"})
    assert r.status_code == 429
    r = c.post("/login", data={"password": PW}, headers={"X-Forwarded-For": "203.0.113.9, 198.51.100.7"})
    assert r.status_code == 303
    scope = {"client": ("10.0.0.1", 1), "headers": [(b"x-forwarded-for", b"a, b"), (b"x-forwarded-for", b"c")]}
    assert auth.client_key(scope, 0) == auth.UNTRUSTED_FORWARDED_KEY
    assert auth.client_key({"client": ("10.0.0.1", 1), "headers": []}, 1) == "10.0.0.1"
    assert auth.client_key(scope, 1) == "c" and auth.client_key(scope, 2) == "b" and auth.client_key(scope, 9) == "a"


# ---------------------------------------------------------------------------------------------
# logout
# ---------------------------------------------------------------------------------------------
def test_logout(c):
    assert _login(c, remember="1").status_code == 303
    assert c.get("/api/meta").status_code == 200
    r = c.get("/logout")
    assert r.status_code == 303 and r.headers["location"] == "/login?out=1"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("stl_session=;") and "Max-Age=0" in cookie and "HttpOnly" in cookie
    assert c.get("/api/meta").status_code == 401
    assert "You have signed out" in c.get("/login?out=1").text
    assert c.post("/logout").status_code == 303
    assert c.put("/logout").status_code == 405


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_logout_from_another_site_asks_first(c, method, site):
    assert _login(c, remember="1").status_code == 303
    r = c.request(method, "/logout", headers={"Sec-Fetch-Site": site})
    assert r.status_code == 200 and "set-cookie" not in r.headers
    assert 'method="post" action="/logout"' in r.text and "Sign out" in r.text and "<script" not in r.text.lower()
    assert "KAIST</span> · <span>NOBEL 연구실" in r.text and r.headers["cache-control"] == "no-store"
    assert c.get("/api/meta").status_code == 200                    # still signed in
    for own in ("same-origin", "none"):                              # own page's form / typed URL or bookmark
        r = c.request(method, "/logout", headers={"Sec-Fetch-Site": own})
        assert r.status_code == 303 and "Max-Age=0" in r.headers["set-cookie"]
    assert c.get("/api/meta").status_code == 401


def test_no_password_in_logs(c, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    _login(c, "wrong")
    _login(c)
    assert PW not in caplog.text and "wrong" not in caplog.text.replace("wrong password", "")
