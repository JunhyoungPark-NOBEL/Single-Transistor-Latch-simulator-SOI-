"""Optional password gate for a deployed simulator ("link + password"; docs/DEPLOY.md §4).

OFF unless the environment variable STL_ACCESS_PASSWORD is set (non-empty). When it is set, every request except
the login page (GET/POST /login), /logout, /api/health and the favicon needs a valid session cookie:

* API paths (/api/*, /openapi.json) and file-like paths (/assets/x.js …) → 401 JSON {"error": "login required"};
* page requests (GET/HEAD /, /index.html, SPA routes, /docs, /redoc) → 303 to /login?next=<path>.

Password check: PBKDF2-HMAC-SHA256 of the submitted password vs. the stored digest, compared with
hmac.compare_digest (the plain password is kept only inside the verifier closure at start-up, never logged).

Session cookie `stl_session` = "v1.<issued>.<expires>.<nonce>.<mac>", mac = HMAC-SHA256(key, "v1.<issued>.<expires>.<nonce>"),
key = HMAC-SHA256(secret, "stl-websim session v1" ‖ PBKDF2(password)).  secret = STL_SESSION_SECRET if set, else
32 random bytes per process (sessions then end when the server restarts).  Binding the password digest into the
key means a password change invalidates every old cookie.  Lifetime: 30 days with "remember me" (Max-Age), else a
browser-session cookie whose token still expires after 24 h.  Attributes: HttpOnly; SameSite=Lax; Path=/; Secure
when the request is https (scope scheme or X-Forwarded-Proto).

Brute force: per-client exponential backoff (5 failures → 30 s, doubling, capped at 15 min) plus a global window
of 20 failed checks per 60 s; while it is full, only clients with no recent failures may try (≤ 10 such "grace"
checks per 60 s), so an attacker who fills the window cannot lock everyone out and at most 30 checks per minute run
server-wide.  Failures of a client that stays idle (and unlocked) for an hour are forgotten.  The client key is the
socket peer (IPv6 grouped by /64), or with STL_TRUST_PROXY=N the N-th address from the right of X-Forwarded-For (the
one appended by the N-th trusted proxy).  Without STL_TRUST_PROXY, requests that carry X-Forwarded-For share ONE key:
uvicorn with --forwarded-allow-ips='*' (the Dockerfile) has already replaced the socket peer by the client-supplied
left-most entry, so that address cannot be trusted.  Every attempt is recorded as a failure *before* the check (and
cleared on success), so parallel requests cannot bypass the limiter.

Fail-closed switch: with STL_REQUIRE_PASSWORD=1 and no STL_ACCESS_PASSWORD, every request (including /api/health, so
the host's health check fails visibly) answers 503 instead of silently serving an open site.  `announce()` prints the
gate state to stderr at start-up (visible under plain uvicorn, Render and HF logs).

/logout: a request that a browser marks as coming from another site (Sec-Fetch-Site: cross-site / same-site) gets a
confirmation page instead of being signed out.

Tests configure the gate in-process with `configure(...)` (tests/test_auth.py); `configure_from_env()` restores it.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import html
import logging
import ipaddress
import os
import secrets
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, quote, unquote, urlsplit

from starlette.concurrency import run_in_threadpool

log = logging.getLogger("stl.auth")

COOKIE_NAME = "stl_session"
LOGIN_PATH = "/login"
LOGOUT_PATH = "/logout"
OPEN_PATHS = frozenset({"/api/health", "/favicon.svg", "/favicon.ico"})
REMEMBER_S = 30 * 86400                  # "keep me signed in": cookie Max-Age and token lifetime
SESSION_S = 24 * 3600                    # browser-session cookie: the token still expires after 24 h
MAX_TOKEN_S = REMEMBER_S + 86400         # tokens claiming a longer lifetime are rejected
CLOCK_SKEW_S = 300
PBKDF2_ITERATIONS = 200_000              # ≈ 60 ms per login check on one core
LOGIN_BODY_MAX = 8192
PASSWORD_MAX = 1024

# brute-force limiter
FREE_FAILURES = 5                        # failures before the first lock
BASE_LOCK_S = 30.0
MAX_LOCK_S = 900.0
MAX_DOUBLINGS = 16                       # 30 s · 2**16 ≫ MAX_LOCK_S (also keeps 2.0 ** n finite)
FORGET_S = 3600.0                        # an idle, unlocked client's failure count is forgotten
MAX_CLIENTS = 10_000
GLOBAL_WINDOW_S = 60.0
GLOBAL_MAX_FAILURES = 20                 # failed checks per window before only "clean" clients may try
GRACE_MAX = 10                           # checks per window for clients without recent failures once it is full
UNTRUSTED_FORWARDED_KEY = "x-forwarded-for (untrusted)"

_TOKEN_VERSION = "v1"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


# ---------------------------------------------------------------------------------------------
# limiter
# ---------------------------------------------------------------------------------------------
@dataclass
class _Entry:
    failures: int = 0
    locked_until: float = 0.0
    last: float = 0.0


@dataclass
class Limiter:
    """In-memory per-client exponential backoff + global sliding window (single process, event-loop only).

    While the global window holds GLOBAL_MAX_FAILURES failed checks, a client that already failed recently waits for
    the window, but a client without recent failures still gets a check — at most GRACE_MAX of those per window —
    so filling the window does not lock every legitimate user out."""
    clock: Callable[[], float] = time.monotonic
    entries: dict[str, _Entry] = field(default_factory=dict)
    recent: deque = field(default_factory=deque)          # timestamps of recorded (reserved) failures
    grace: deque = field(default_factory=deque)           # timestamps of checks granted while the window was full
    tripped_until: float = 0.0                            # rate-limits the "global window full" warning

    def _trim(self, now: float) -> None:
        cutoff = now - GLOBAL_WINDOW_S
        for q in (self.recent, self.grace):
            while q and q[0] <= cutoff:
                q.popleft()

    def _entry(self, key: str, now: float) -> _Entry | None:
        entry = self.entries.get(key)
        if entry is not None and entry.locked_until <= now and entry.last <= now - FORGET_S:
            del self.entries[key]                          # idle and unlocked for an hour: old typos are forgotten
            return None
        return entry

    def retry_after(self, key: str) -> float:
        """Seconds the client must wait before its next password check (0 = may try now)."""
        now = self.clock()
        self._trim(now)
        entry = self._entry(key, now)
        wait = entry.locked_until - now if entry is not None else 0.0
        if len(self.recent) >= GLOBAL_MAX_FAILURES:
            room = self.recent[-GLOBAL_MAX_FAILURES] + GLOBAL_WINDOW_S      # when the window has room again
            if entry is not None:                          # failed recently: wait until the window has room
                wait = max(wait, room - now)
            elif len(self.grace) >= GRACE_MAX:             # clean client, but the grace budget is spent too
                wait = max(wait, min(room, self.grace[-GRACE_MAX] + GLOBAL_WINDOW_S) - now)
        return max(wait, 0.0)

    def reserve(self, key: str) -> float:
        """Record an attempt as a failure before it is checked; returns a ticket for `success`."""
        now = self.clock()
        self._trim(now)
        self._prune(now)
        entry = self._entry(key, now)
        if len(self.recent) >= GLOBAL_MAX_FAILURES:
            self.grace.append(now)
            if now >= self.tripped_until:
                self.tripped_until = now + GLOBAL_WINDOW_S
                log.warning("login: %d failed checks within %.0f s server-wide; only clients without recent failures "
                            "may try (at most %d per %.0f s) — possible password guessing",
                            len(self.recent), GLOBAL_WINDOW_S, GRACE_MAX, GLOBAL_WINDOW_S)
        if entry is None:
            entry = self.entries[key] = _Entry()
        entry.failures += 1
        entry.last = now
        if entry.failures >= FREE_FAILURES:
            doublings = min(entry.failures - FREE_FAILURES, MAX_DOUBLINGS)
            lock = min(BASE_LOCK_S * 2.0 ** doublings, MAX_LOCK_S)
            entry.locked_until = now + lock
            log.warning("login: %d failed attempts from %s, locked for %.0f s", entry.failures, key, lock)
        self.recent.append(now)
        return now

    def success(self, key: str, ticket: float) -> None:
        self.entries.pop(key, None)
        for q in (self.recent, self.grace):
            try:
                q.remove(ticket)
            except ValueError:
                pass

    def _prune(self, now: float) -> None:
        if len(self.entries) < 256:
            return
        stale = [k for k, e in self.entries.items() if e.locked_until <= now and e.last < now - FORGET_S]
        for k in stale:
            del self.entries[k]
        if len(self.entries) >= MAX_CLIENTS:                # still full: drop the oldest unlocked entries
            for k, _ in sorted(((k, e.last) for k, e in self.entries.items() if e.locked_until <= now),
                               key=lambda kv: kv[1])[: len(self.entries) - MAX_CLIENTS + 1]:
                del self.entries[k]


# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------
class GateConfig:
    """Password verifier + session key.  The password itself is not stored (only salted PBKDF2 digests)."""

    def __init__(self, password: str, session_secret: str | None = None, trust_proxy: int = 0,
                 iterations: int = PBKDF2_ITERATIONS, clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        if not password:
            raise ValueError("empty password")
        pw = password.encode("utf-8")
        self.iterations = int(iterations)
        self._salt = secrets.token_bytes(16)
        self._verifier = hashlib.pbkdf2_hmac("sha256", pw, self._salt, self.iterations)
        bind = hashlib.pbkdf2_hmac("sha256", pw, b"stl-websim/session-bind/v1", self.iterations)
        self.secret_source = "STL_SESSION_SECRET" if session_secret else "per-process random"
        secret = session_secret.encode("utf-8") if session_secret else secrets.token_bytes(32)
        self._key = hmac.new(secret, b"stl-websim session v1\x00" + bind, hashlib.sha256).digest()
        self.trust_proxy = max(0, int(trust_proxy))
        self.clock = clock
        self.limiter = Limiter(clock=monotonic)

    # ---- password --------------------------------------------------------------------------
    def check_password(self, candidate: str) -> bool:
        """Constant-time comparison of PBKDF2 digests (slow on purpose; call from a worker thread)."""
        digest = hashlib.pbkdf2_hmac("sha256", candidate.encode("utf-8", "surrogatepass"), self._salt, self.iterations)
        return hmac.compare_digest(digest, self._verifier)

    # ---- session tokens --------------------------------------------------------------------
    def _mac(self, body: str) -> str:
        return _b64(hmac.new(self._key, body.encode("ascii"), hashlib.sha256).digest())

    def make_token(self, remember: bool, now: float | None = None) -> str:
        issued = int(self.clock() if now is None else now)
        expires = issued + (REMEMBER_S if remember else SESSION_S)
        body = f"{_TOKEN_VERSION}.{issued}.{expires}.{_b64(secrets.token_bytes(12))}"
        return f"{body}.{self._mac(body)}"

    def check_token(self, token: str, now: float | None = None) -> bool:
        if not token or len(token) > 200:
            return False
        parts = token.split(".")
        if len(parts) != 5 or parts[0] != _TOKEN_VERSION:
            return False
        body, mac = ".".join(parts[:4]), parts[4]
        try:
            expected = self._mac(body)
        except UnicodeEncodeError:
            return False
        if not hmac.compare_digest(mac.encode("ascii", "replace"), expected.encode("ascii")):
            return False
        issued_s, expires_s = parts[1], parts[2]
        if not (issued_s.isdigit() and expires_s.isdigit() and len(issued_s) <= 12 and len(expires_s) <= 12):
            return False
        issued, expires = int(issued_s), int(expires_s)
        t = self.clock() if now is None else now
        return issued <= t + CLOCK_SKEW_S and t < expires and 0 < expires - issued <= MAX_TOKEN_S


def _parse_trust(value: str | None) -> int:
    v = (value or "").strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return 0
    if v in ("true", "yes", "on"):
        return 1
    try:
        return max(0, int(v))
    except ValueError:
        log.warning("STL_TRUST_PROXY=%r is not a number of proxy hops; using 1", value)
        return 1


class GateMisconfigured(RuntimeError):
    """STL_REQUIRE_PASSWORD is set but no usable STL_ACCESS_PASSWORD: the server must not run open."""


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _clean_password(raw: str) -> str:
    password = raw.strip()                 # dashboard pastes: a trailing space/newline would make it untypeable
    if password != raw.rstrip("\r\n"):
        log.warning("STL_ACCESS_PASSWORD had leading/trailing whitespace; it was removed")
    if len(password) >= 2 and password[0] == password[-1] and password[0] in "\"'":
        log.warning("STL_ACCESS_PASSWORD starts and ends with a quote character; the quotes are part of the password "
                    "(docker --env-file keeps them literally) — remove them if that is not intended")
    return password


def config_from_env(env: Mapping[str, str] | None = None) -> GateConfig | None:
    env = os.environ if env is None else env
    password = _clean_password(env.get("STL_ACCESS_PASSWORD") or "")
    if not password:
        if _truthy(env.get("STL_REQUIRE_PASSWORD")):
            raise GateMisconfigured("STL_REQUIRE_PASSWORD is set but STL_ACCESS_PASSWORD is empty or missing")
        return None
    secret = (env.get("STL_SESSION_SECRET") or "").strip() or None
    if secret and len(secret) < 16:
        log.warning("STL_SESSION_SECRET is shorter than 16 characters; use a long random string")
    return GateConfig(password, session_secret=secret, trust_proxy=_parse_trust(env.get("STL_TRUST_PROXY")))


class _State:
    config: GateConfig | None = None
    misconfigured: str | None = None       # set → every request answers 503 (fail closed)


STATE = _State()


def configure(password: str | None, session_secret: str | None = None, trust_proxy: int = 0,
              **kwargs: Any) -> GateConfig | None:
    """Switch the gate on (password) or off (None) in this process — used by tests."""
    STATE.config = GateConfig(password, session_secret, trust_proxy, **kwargs) if password else None
    STATE.misconfigured = None
    return STATE.config


def configure_from_env(env: Mapping[str, str] | None = None) -> GateConfig | None:
    try:
        STATE.config, STATE.misconfigured = config_from_env(env), None
    except GateMisconfigured as exc:
        STATE.config, STATE.misconfigured = None, str(exc)
        log.error("access gate: %s — every request answers 503 until the password is set", exc)
    return STATE.config


def state_name() -> str:
    """"on" | "off" | "misconfigured" (reported by /api/health so a deploy can be checked without logging in)."""
    return "misconfigured" if STATE.misconfigured else "off" if STATE.config is None else "on"


def describe() -> str:
    cfg = STATE.config
    if STATE.misconfigured:
        return f"access gate MISCONFIGURED ({STATE.misconfigured}): every request answers 503"
    if cfg is None:
        return ("access gate OFF: anyone with the URL can use the simulator "
                "(set STL_ACCESS_PASSWORD to require a password)")
    return (f"access gate ON (session secret: {cfg.secret_source}; "
            f"client IP: {'X-Forwarded-For hop ' + str(cfg.trust_proxy) if cfg.trust_proxy else 'socket peer'})")


def announce(stream: Any = None) -> None:
    """Print the gate state at start-up.  stderr, not logging: uvicorn's default log config drops INFO records of
    other loggers, and an open-by-accident deployment must be visible in the host's logs."""
    print(f"STL: {describe()}", file=stream or sys.stderr, flush=True)


_warned: set[str] = set()


def _warn_once(tag: str, msg: str, *args: Any) -> None:
    if tag not in _warned:
        _warned.add(tag)
        log.warning(msg, *args)


# ---------------------------------------------------------------------------------------------
# request helpers
# ---------------------------------------------------------------------------------------------
def _headers(scope: Mapping[str, Any]) -> list[tuple[bytes, bytes]]:
    return list(scope.get("headers") or ())


def _header_values(scope: Mapping[str, Any], name: bytes) -> list[str]:
    return [v.decode("latin-1") for k, v in _headers(scope) if k.lower() == name]


def _route_path(scope: Mapping[str, Any]) -> str:
    """The path the router matches (scope path without root_path, as Starlette's get_route_path)."""
    path: str = scope.get("path") or "/"
    root = scope.get("root_path") or ""
    if root and path.startswith(root) and (len(path) == len(root) or path[len(root)] == "/"):
        return path[len(root):] or "/"
    return path


def _session_cookies(scope: Mapping[str, Any]) -> list[str]:
    out = []
    for header in _header_values(scope, b"cookie"):
        for chunk in header.split(";"):
            name, _, value = chunk.strip().partition("=")
            if name.strip() == COOKIE_NAME:
                out.append(value.strip().strip('"'))
    return out


def _is_https(scope: Mapping[str, Any]) -> bool:
    if scope.get("scheme") in ("https", "wss"):
        return True
    for value in _header_values(scope, b"x-forwarded-proto"):
        if value.split(",")[0].strip().lower() == "https":
            return True
    return False


def _norm_ip(value: str) -> str:
    """Limiter key for one address: IPv4 as is, IPv4-mapped IPv6 as IPv4, other IPv6 grouped by /64 (one host
    usually owns a whole /64, so per-/128 keys would give it 2**64 fresh counters)."""
    v = value.strip().strip('"')
    if v.startswith("["):                                   # "[2001:db8::1]:443"
        v = v[1:].split("]", 1)[0]
    elif v.count(":") == 1:                                 # "192.0.2.1:443"
        v = v.split(":", 1)[0]
    try:
        ip = ipaddress.ip_address(v)
    except ValueError:
        return v[:64] or "unknown"
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.IPv6Network(((int(ip) >> 64) << 64, 64)))
    return str(ip)


def client_key(scope: Mapping[str, Any], trust_proxy: int) -> str:
    hops = [h.strip() for v in _header_values(scope, b"x-forwarded-for") for h in v.split(",") if h.strip()]
    if trust_proxy > 0 and hops:
        extra = [h for h in ("cf-connecting-ip", "true-client-ip", "x-real-ip") if _header_values(scope, h.encode())]
        _warn_once("xff-shape", "login limiter: X-Forwarded-For has %d entr%s, using entry %d from the right "
                   "(STL_TRUST_PROXY=%d)%s — every visitor must get a different key; if the host adds more than one "
                   "proxy hop, raise STL_TRUST_PROXY", len(hops), "y" if len(hops) == 1 else "ies",
                   min(trust_proxy, len(hops)), trust_proxy, f"; also present: {', '.join(extra)}" if extra else "")
        return _norm_ip(hops[-trust_proxy] if len(hops) >= trust_proxy else hops[0])
    if hops:
        # No trusted proxy configured, yet the request names a client address.  uvicorn run with
        # --forwarded-allow-ips='*' (the Dockerfile) has already replaced scope["client"] by the client-supplied
        # left-most entry, so neither value can be trusted: all such requests share one limiter key (no bypass by
        # rotating the header, no lock-out of another address by naming it).
        _warn_once("xff-untrusted", "login limiter: a request carries X-Forwarded-For but STL_TRUST_PROXY is not set; "
                   "all such requests share one limiter key (set STL_TRUST_PROXY=1 behind a reverse proxy)")
        return UNTRUSTED_FORWARDED_KEY
    client = scope.get("client")
    return _norm_ip(str(client[0])) if client else "unknown"


_URL_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%")


def _resolved_path(path: str) -> str:
    """Dot segments removed as a browser does (%2e counts as "."), then percent-decoded as the router does."""
    out: list[str] = []
    for seg in path.split("/")[1:]:
        dots = seg.replace("%2e", ".").replace("%2E", ".")
        if dots == ".":
            continue
        if dots == "..":
            if out:
                out.pop()
            continue
        out.append(seg)
    return unquote("/" + "/".join(out))


def safe_next(value: str | None) -> str:
    """Same-site relative path to return to after login; anything else → "/" (no open redirect)."""
    if not value or len(value) > 2048:
        return "/"
    if not _URL_CHARS.issuperset(value):
        return "/"                         # whitespace/control chars (browsers drop tabs), backslash, quotes, non-ASCII
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    parts = urlsplit(value)
    if parts.scheme or parts.netloc:
        return "/"
    target = _resolved_path(parts.path)                   # as browser + router see it: /%6cogout, /x/../logout
    if target in (LOGIN_PATH, LOGOUT_PATH) or target.startswith((LOGIN_PATH + "/", LOGOUT_PATH + "/")):
        return "/"
    return value


def _requested(scope: Mapping[str, Any]) -> str:
    raw = scope.get("raw_path")
    path = raw.decode("latin-1") if isinstance(raw, (bytes, bytearray)) else quote(scope.get("path") or "/")
    qs = scope.get("query_string") or b""
    return path + ("?" + qs.decode("latin-1") if qs else "")


def _is_api(path: str) -> bool:
    return path == "/api" or path.startswith("/api/") or path == "/openapi.json"


def _is_page(path: str) -> bool:
    last = path.rsplit("/", 1)[-1]
    return "." not in last or last.endswith(".html")


# ---------------------------------------------------------------------------------------------
# responses
# ---------------------------------------------------------------------------------------------
_PAGE_HEADERS = [
    (b"cache-control", b"no-store"),
    (b"x-robots-tag", b"noindex, nofollow"),
    (b"referrer-policy", b"same-origin"),
    (b"x-content-type-options", b"nosniff"),
    (b"content-security-policy",
     b"default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; form-action 'self'; base-uri 'none'"),
]


async def _send(send: Any, status: int, body: bytes, content_type: bytes,
                headers: list[tuple[bytes, bytes]] | None = None) -> None:
    hdrs = [(b"content-type", content_type), (b"content-length", str(len(body)).encode())] + (headers or [])
    await send({"type": "http.response.start", "status": status, "headers": hdrs})
    await send({"type": "http.response.body", "body": body})


async def _redirect(send: Any, location: str, headers: list[tuple[bytes, bytes]] | None = None) -> None:
    await _send(send, 303, b"", b"text/plain; charset=utf-8",
                [(b"location", location.encode("latin-1")), (b"cache-control", b"no-store")] + (headers or []))


_DENIED = b'{"error":"login required","detail":"login required \xe2\x80\x94 reload the page to sign in"}'


async def _deny_api(send: Any) -> None:
    await _send(send, 401, _DENIED, b"application/json", [(b"cache-control", b"no-store")])


def _cookie(value: str, max_age: int | None, secure: bool) -> bytes:
    parts = [f"{COOKIE_NAME}={value}", "Path=/", "HttpOnly", "SameSite=Lax"]
    if max_age is not None:
        parts.append(f"Max-Age={max_age}")
        if max_age == 0:
            parts.append("Expires=Thu, 01 Jan 1970 00:00:00 GMT")
    if secure:
        parts.append("Secure")
    return "; ".join(parts).encode("latin-1")


_MESSAGES = {
    "wrong": ("err", "비밀번호가 올바르지 않습니다.", "Incorrect password."),
    "locked": ("warn", "시도가 너무 많습니다. {s}초 후에 다시 시도하세요.", "Too many attempts. Try again in {s} s."),
    "expired": ("info", "세션이 만료되었습니다. 다시 로그인하세요.", "Your session has expired. Please sign in again."),
    "out": ("info", "로그아웃했습니다.", "You have signed out."),
    "bad": ("err", "잘못된 요청입니다. 다시 시도하세요.", "Invalid request. Please try again."),
}

_LOGIN_CSS = """
:root{color-scheme:light;--bg:#f4f5f7;--surface:#fff;--surface-2:#f8f9fb;--border:#e2e5eb;--border-strong:#cfd4dc;
--text:#0f172a;--text-2:#3b4658;--muted:#6b7587;--accent:#0d9488;--accent-strong:#0f766e;--on-accent:#fff;
--err:#b91c1c;--err-soft:rgba(220,38,38,.09);--warn:#b45309;--warn-soft:rgba(217,119,6,.12);--info:#1d4ed8;
--info-soft:rgba(37,99,235,.08);--shadow:0 4px 16px rgba(15,23,42,.08),0 1px 3px rgba(15,23,42,.06);
--font:"Pretendard","Pretendard Variable","Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#0e1015;--surface:#161920;--surface-2:#1b1f27;
--border:#2a303b;--border-strong:#3a4250;--text:#e7eaf0;--text-2:#b7bfcc;--muted:#8a93a3;--accent:#2dd4bf;
--accent-strong:#5eead4;--on-accent:#04201d;--err:#f87171;--err-soft:rgba(248,113,113,.12);--warn:#fbbf24;
--warn-soft:rgba(251,191,36,.12);--info:#93c5fd;--info-soft:rgba(96,165,250,.1);--shadow:0 6px 20px rgba(0,0,0,.35)}}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;
padding:24px 16px;font-family:var(--font);font-size:14px;line-height:1.5;color:var(--text);
background:radial-gradient(1200px 600px at 100% -10%,rgba(99,102,241,.06),transparent 60%),var(--bg);
-webkit-font-smoothing:antialiased;word-break:keep-all;overflow-wrap:break-word}
main{width:100%;max-width:400px;background:var(--surface);border:1px solid var(--border);border-radius:14px;
box-shadow:var(--shadow);padding:24px}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:16px}
.brand img{width:40px;height:40px;border-radius:10px;flex:none}
h1{font-size:18px;line-height:1.3;margin:0;letter-spacing:-.01em}
.tag{margin:0;color:var(--muted);font-size:12.5px}
.lead{margin:0 0 16px;color:var(--text-2)}
.en{display:block;color:var(--muted);font-size:12px;font-weight:400}
.msg{margin:0 0 16px;padding:8px 12px;border-radius:8px;font-size:13px}
.msg.err{background:var(--err-soft);color:var(--err)}
.msg.warn{background:var(--warn-soft);color:var(--warn)}
.msg.info{background:var(--info-soft);color:var(--info)}
.msg .en{color:inherit;opacity:.85}
label.pw{display:block;font-weight:600;font-size:13px;margin-bottom:6px}
label.pw .en{display:inline;margin-left:6px}
input[type=password]{display:block;width:100%;height:42px;padding:0 12px;border-radius:8px;
border:1px solid var(--border-strong);background:var(--surface-2);color:var(--text);font:inherit;font-size:16px}
input[type=password]:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px rgba(13,148,136,.25)}
.check{display:flex;gap:8px;align-items:flex-start;margin:14px 0 18px;font-size:13px;color:var(--text-2);cursor:pointer}
.check input{margin:3px 0 0;width:16px;height:16px;accent-color:var(--accent);flex:none}
button{display:block;width:100%;height:42px;border:0;border-radius:8px;background:var(--accent-strong);
color:var(--on-accent);font:inherit;font-size:15px;font-weight:650;cursor:pointer}
button:hover{filter:brightness(1.08)}
button:focus-visible{outline:none;box-shadow:0 0 0 3px rgba(13,148,136,.35)}
.scope{margin:16px 0 0;font-size:12px;color:var(--muted)}
a{color:var(--accent-strong)}
footer{max-width:400px;text-align:center;font-size:12px;color:var(--muted)}
footer span{white-space:nowrap}
"""


def _page(title: str, content: str, root: str = "") -> bytes:
    """Self-contained page shell (inline CSS, no JS) shared by the login and logout-confirmation pages."""
    root_e = html.escape(root, quote=True)
    page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<meta name="color-scheme" content="light dark">
<title>{title} · STL Simulator</title>
<link rel="icon" type="image/svg+xml" href="{root_e}/favicon.svg">
<style>{_LOGIN_CSS}</style>
</head>
<body>
<main>
<div class="brand"><img src="{root_e}/favicon.svg" alt="" width="40" height="40">
<div><h1>STL Simulator</h1><p class="tag">SOI 단일 트랜지스터 래치 시뮬레이터
<span class="en" lang="en">Single-transistor latch simulator for SOI</span></p></div></div>
{content}
</main>
<footer><span>KAIST 전기및전자공학부</span> · <span>NOBEL 연구실 (지도교수 최양규)</span> · <span>개발 박준형</span></footer>
</body>
</html>
"""
    return page.encode("utf-8")


def login_page(next_path: str = "/", message: str | None = None, retry_s: int = 0, root: str = "") -> bytes:
    msg_html = ""
    if message in _MESSAGES:
        cls, ko, en = _MESSAGES[message]
        msg_html = (f'<p class="msg {cls}" role="alert">{html.escape(ko.format(s=retry_s))}'
                    f'<span class="en" lang="en">{html.escape(en.format(s=retry_s))}</span></p>')
    root_e = html.escape(root, quote=True)
    content = f"""<p class="lead">비공개 연구용 페이지입니다. 공유받은 비밀번호를 입력하세요.
<span class="en" lang="en">This is a private research page. Enter the password you were given.</span></p>
{msg_html}
<form method="post" action="{root_e}{LOGIN_PATH}">
<input type="hidden" name="next" value="{html.escape(next_path, quote=True)}">
<label class="pw" for="pw">비밀번호<span class="en" lang="en">Password</span></label>
<input id="pw" name="password" type="password" autocomplete="current-password" maxlength="{PASSWORD_MAX}" required autofocus>
<label class="check"><input type="checkbox" name="remember" value="1">
<span>이 기기에서 30일 동안 로그인 유지<span class="en" lang="en">Keep me signed in on this device for 30 days</span></span></label>
<button type="submit">들어가기 · Sign in</button>
</form>
<p class="scope">미발표 모델 — 연구용으로만 사용하세요.<span class="en" lang="en">Unpublished model — for research use only.</span></p>"""
    return _page("로그인", content, root)


def logout_page(root: str = "") -> bytes:
    """Shown when /logout is reached from another site (a link or form elsewhere must not sign the user out)."""
    root_e = html.escape(root, quote=True)
    content = f"""<p class="lead">로그아웃할까요?
<span class="en" lang="en">Sign out of the simulator?</span></p>
<form method="post" action="{root_e}{LOGOUT_PATH}">
<button type="submit">로그아웃 · Sign out</button>
</form>
<p class="scope"><a href="{root_e}/">시뮬레이터로 돌아가기<span class="en" lang="en">Back to the simulator</span></a></p>"""
    return _page("로그아웃", content, root)


async def _send_login_page(send: Any, status: int, next_path: str, message: str | None = None, retry_s: int = 0,
                           root: str = "", extra: list[tuple[bytes, bytes]] | None = None) -> None:
    await _send(send, status, login_page(next_path, message, retry_s, root), b"text/html; charset=utf-8",
                list(_PAGE_HEADERS) + (extra or []))


_UNAVAILABLE = (b'{"error":"access control not configured","detail":"The server is not configured yet '
                b'(access control). Try again later."}')


async def _read_body(receive: Any, limit: int) -> bytes | None:
    chunks, size = [], 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            return None
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
        if not message.get("more_body", False):
            return b"".join(chunks)


# ---------------------------------------------------------------------------------------------
# middleware
# ---------------------------------------------------------------------------------------------
class AccessGate:
    """ASGI middleware; pass-through while `STATE.config` is None (STL_ACCESS_PASSWORD unset)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        cfg = STATE.config
        if scope["type"] not in ("http", "websocket") or (cfg is None and not STATE.misconfigured):
            await self.app(scope, receive, send)
            return
        if cfg is None:                                      # fail closed: STL_REQUIRE_PASSWORD without a password
            if scope["type"] == "websocket":
                if (await receive())["type"] == "websocket.connect":
                    await send({"type": "websocket.close", "code": 1011})
                return
            await _send(send, 503, _UNAVAILABLE, b"application/json",
                        [(b"cache-control", b"no-store"), (b"retry-after", b"300")])
            return
        path = _route_path(scope)
        cookies = _session_cookies(scope)
        authed = any(cfg.check_token(c) for c in cookies)
        if scope["type"] == "websocket":
            if authed:
                await self.app(scope, receive, send)
            elif (await receive())["type"] == "websocket.connect":
                await send({"type": "websocket.close", "code": 1008})      # before accept → HTTP 403
            return
        method = scope.get("method", "GET")
        root = scope.get("root_path") or ""
        if path == LOGIN_PATH:
            await self._login(scope, receive, send, cfg, authed, root)
            return
        if path == LOGOUT_PATH:
            await self._logout(scope, send, root)
            return
        if path in OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        if authed:
            await self.app(scope, receive, self._private(send))
            return
        if method in ("GET", "HEAD") and not _is_api(path) and _is_page(path):
            query = "next=" + quote(safe_next(_requested(scope)), safe="")
            if cookies:
                query += "&expired=1"
            await _redirect(send, f"{root}{LOGIN_PATH}?{query}")
            return
        await _deny_api(send)

    @staticmethod
    def _private(send: Any) -> Any:
        """Authenticated responses must not be stored by shared caches (CDN): Cache-Control public → private."""
        async def wrapped(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v.replace(b"public", b"private")) if k.lower() == b"cache-control" else (k, v)
                           for k, v in message.get("headers") or ()]
                message = {**message, "headers": headers}
            await send(message)
        return wrapped

    @staticmethod
    async def _logout(scope: dict, send: Any, root: str) -> None:
        method = scope.get("method", "GET")
        if method not in ("GET", "HEAD", "POST"):
            await _send(send, 405, b"method not allowed", b"text/plain; charset=utf-8",
                        [(b"allow", b"GET, HEAD, POST")])
            return
        site = next(iter(_header_values(scope, b"sec-fetch-site")), "").strip().lower()
        if site in ("cross-site", "same-site"):              # a link/form on another site: ask instead
            await _send(send, 200, logout_page(root), b"text/html; charset=utf-8", list(_PAGE_HEADERS))
            return
        await _redirect(send, f"{root}{LOGIN_PATH}?out=1", [(b"set-cookie", _cookie("", 0, _is_https(scope)))])

    async def _login(self, scope: dict, receive: Any, send: Any, cfg: GateConfig, authed: bool, root: str) -> None:
        method = scope.get("method", "GET")
        if method in ("GET", "HEAD"):
            query = parse_qs((scope.get("query_string") or b"").decode("latin-1"), max_num_fields=10)
            next_path = safe_next((query.get("next") or ["/"])[0])
            if authed:
                await _redirect(send, next_path)
                return
            message = "out" if "out" in query else "expired" if "expired" in query else None
            await _send_login_page(send, 200, next_path, message, root=root)
            return
        if method != "POST":
            await _send(send, 405, b"method not allowed", b"text/plain; charset=utf-8",
                        [(b"allow", b"GET, HEAD, POST")])
            return
        body = await _read_body(receive, LOGIN_BODY_MAX)
        if body is None:
            await _send_login_page(send, 413, "/", "bad", root=root)
            return
        try:
            form = parse_qs(body.decode("utf-8"), keep_blank_values=True, max_num_fields=10)
        except (UnicodeDecodeError, ValueError):
            await _send_login_page(send, 400, "/", "bad", root=root)
            return
        next_path = safe_next((form.get("next") or ["/"])[0])
        password = (form.get("password") or [""])[0]
        remember = (form.get("remember") or [""])[0].lower() in ("1", "on", "true", "yes")
        key = client_key(scope, cfg.trust_proxy)
        wait = cfg.limiter.retry_after(key)
        if wait > 0:
            secs = int(wait + 0.999)
            await _send_login_page(send, 429, next_path, "locked", secs, root=root,
                                   extra=[(b"retry-after", str(secs).encode())])
            return
        ticket = cfg.limiter.reserve(key)            # counted as a failure until the check succeeds
        ok = bool(password) and len(password) <= PASSWORD_MAX and await run_in_threadpool(cfg.check_password, password)
        if not ok:
            log.info("login: wrong password from %s", key)
            await _send_login_page(send, 401, next_path, "wrong", root=root)
            return
        cfg.limiter.success(key, ticket)
        token = cfg.make_token(remember)
        log.info("login: session issued to %s (%s)", key, "30 days" if remember else "browser session")
        await _redirect(send, next_path,
                        [(b"set-cookie", _cookie(token, REMEMBER_S if remember else None, _is_https(scope)))])


configure_from_env()
