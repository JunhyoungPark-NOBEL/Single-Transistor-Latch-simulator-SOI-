"""Remote browser authentication uses revocable, memory-only bearer sessions, never a third-party cookie."""
from __future__ import annotations

import pytest

from server import auth

PW = "test-studio-secret"
ORIGIN = "http://127.0.0.1:4173"


@pytest.fixture
def session_client():
    from fastapi.testclient import TestClient
    from server.main import app
    previous = auth.STATE.config, auth.STATE.misconfigured
    cfg = auth.configure(PW, iterations=1000)
    try:
        with TestClient(app, follow_redirects=False) as client:
            yield client, cfg
    finally:
        auth.STATE.config, auth.STATE.misconfigured = previous


def test_remote_login_meta_and_revoke(session_client):
    client, cfg = session_client
    assert client.get("/api/session").json()["authenticated"] is False
    assert client.get("/api/health").json()["auth_required"] is True
    assert client.get("/api/meta").status_code == 401
    login = client.post("/api/session", json={"password": PW}, headers={"Origin": ORIGIN})
    assert login.status_code == 200
    assert login.headers["cache-control"] == "no-store"
    assert "set-cookie" not in login.headers
    assert login.headers["access-control-allow-origin"] == ORIGIN
    assert "access-control-allow-credentials" not in login.headers
    data = login.json()
    assert data["authenticated"] and data["expires_in"] == 86400
    token = data["token"]
    headers = {"Authorization": f"Bearer {token}", "Origin": ORIGIN}
    assert cfg.check_api_token(token) and not cfg.check_token(token)
    assert token not in cfg._api_sessions
    assert client.get("/api/session", headers=headers).json()["authenticated"]
    assert client.get("/api/meta", headers=headers).status_code == 200
    # Bearer tokens grant API access only; they are not a substitute for the browser's page cookie.
    assert client.get("/", headers=headers).status_code == 303
    assert client.delete("/api/session", headers=headers).json()["authenticated"] is False
    assert client.get("/api/meta", headers=headers).status_code == 401


def test_session_uses_same_limiter_as_cookie_login(session_client):
    client, cfg = session_client
    for _ in range(auth.FREE_FAILURES):
        assert client.post("/api/session", json={"password": "wrong"}).status_code == 401
    response = client.post("/login", data={"password": PW})
    assert response.status_code == 429 and int(response.headers["retry-after"]) > 0
    response = client.post("/api/session", json={"password": PW})
    assert response.status_code == 429 and int(response.headers["retry-after"]) > 0
    assert not cfg._api_sessions


def test_session_origin_content_type_and_payload_guards(session_client):
    client, cfg = session_client
    assert client.post("/api/session", json={"password": PW}, headers={"Origin": "https://unknown.example"}).status_code == 403
    assert client.post("/api/session", json={"password": PW}, headers={"Origin": "null"}).status_code == 403
    assert client.post("/api/session", data={"password": PW}).status_code == 415
    assert client.post("/api/session", json={"password": [PW]}).status_code == 400
    assert client.post("/api/session", content="{", headers={"Content-Type": "application/json"}).status_code == 400
    assert client.post("/api/session", json={"password": "x" * 9000}).status_code == 413
    assert not cfg._api_sessions
    preflight = {"Origin": ORIGIN, "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "Content-Type, Authorization"}
    assert client.options("/api/compute/branches", headers=preflight).status_code == 200
    preflight["Origin"] = "https://unknown.example"
    assert client.options("/api/compute/branches", headers=preflight).status_code == 400


def test_api_token_expiration_restart_and_bounded_storage():
    clock = [100.0]
    cfg = auth.GateConfig(PW, iterations=1000, clock=lambda: clock[0])
    token = cfg.make_api_token()
    assert cfg.check_api_token(token)
    assert not cfg.check_api_token(token + "a") and not cfg.check_api_token("api1." + "a" * 43)
    restarted = auth.GateConfig(PW, iterations=1000)
    assert not restarted.check_api_token(token)
    clock[0] += auth.SESSION_S
    assert not cfg.check_api_token(token)
    cfg.make_api_token()
    assert len(cfg._api_sessions) == 1
    for _ in range(auth.MAX_API_SESSIONS + 1):
        cfg.make_api_token()
    assert len(cfg._api_sessions) == auth.MAX_API_SESSIONS


@pytest.mark.parametrize("origin", ["*", "https://*.example", "https://example/path", "https://user:pass@example", "null"])
def test_wildcard_or_nonorigin_cors_configuration_refused(origin):
    with pytest.raises(ValueError, match="exact http"):
        auth.cors_origins({"STL_CORS_ORIGINS": origin})


def test_sameorigin_cookie_and_local_open_mode(session_client):
    client, _ = session_client
    assert client.post("/login", data={"password": PW}).status_code == 303
    assert client.get("/api/session").json()["authenticated"] is True
    assert client.get("/api/meta").status_code == 200
    client.delete("/api/session", headers={"Origin": "http://testserver"})
    assert client.get("/api/meta").status_code == 401
    auth.configure(None)
    assert client.get("/api/session").json() == {"authenticated": True, "access_gate": "off"}
    assert client.post("/api/session", json={}).json()["token"] is None
