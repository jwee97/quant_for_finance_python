"""The dashboard's HTTP server: what it refuses (a request that would make it wait, a page without the access code), what it adds to every answer (security headers) and what it saves (compression, a
second download only when a script changed)."""

from __future__ import annotations

import gzip
import http.client
import json
import socket
import time
from urllib.parse import urlparse

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.test_webapp import _default, _fetch


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=_default), root=tmp_path)


@pytest.fixture()
def live(app):
    running = server.start(app, "127.0.0.1", 0)
    yield running
    running.stop()


def _call(running, method, path, headers=None, body=None, timeout=10):
    """One request over a plain connection (no redirects followed, no automatic decompression): ``(status, body, headers)`` with lower-cased header names."""
    u = urlparse(running.url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        r = conn.getresponse()
        return r.status, r.read(), {k.lower(): v for k, v in r.getheaders()}
    finally:
        conn.close()


# ------------------------------------------------------------------------------------------------------ what it refuses
def test_a_content_length_that_is_negative_or_not_a_number_is_refused_instead_of_waiting_for_the_client(live):
    for length in ("-5", "abc", "1e3"):
        u = urlparse(live.url)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=5)
        conn.putrequest("POST", "/api/tickers")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("X-Token", live.token)
        conn.putheader("Content-Length", length)
        conn.endheaders()
        t0 = time.time()
        r = conn.getresponse()
        assert r.status == 400 and json.loads(r.read())["error"].startswith("the Content-Length"), length
        assert time.time() - t0 < 4, "it answered at once and did not wait for a body that was never coming"
        conn.close()
    status, body, _ = _call(live, "POST", "/api/tickers", {"Content-Type": "application/json", "X-Token": live.token}, b'{"tickers": ["' + b"x" * 300_000 + b'"]}')
    assert status == 413


def test_a_client_that_connects_and_says_nothing_is_dropped_after_the_timeout(app, monkeypatch):
    monkeypatch.setattr(server, "REQUEST_TIMEOUT", 0.4)
    running = server.start(app, "127.0.0.1", 0)
    try:
        u = urlparse(running.url)
        with socket.create_connection((u.hostname, u.port), timeout=5) as s:
            t0 = time.time()
            assert s.recv(10) == b"", "the server closed the silent connection"
            assert time.time() - t0 < 4
        assert _call(running, "GET", "/")[0] == 200, "and goes on serving"
    finally:
        running.stop()


# ------------------------------------------------------------------------------------------------------ what it adds
def test_every_answer_carries_the_security_headers_and_hides_the_python_version(live):
    for path, token in (("/", False), ("/static/app.js", False), ("/api/catalog", True), ("/nothing", False)):
        status, _, h = _call(live, "GET", path, {"X-Token": live.token} if token else {})
        assert status in (200, 404), path
        policy = h["content-security-policy"]
        assert "base-uri 'none'" in policy and "object-src 'none'" in policy and "form-action 'self'" in policy and "frame-ancestors 'none'" in policy, path
        assert h["referrer-policy"] == "no-referrer" and h["x-content-type-options"] == "nosniff" and h["cross-origin-opener-policy"] == "same-origin" and h["cross-origin-resource-policy"] == "same-origin", path
        assert "camera=()" in h["permissions-policy"] and "python" not in h.get("server", "").lower(), path


def test_the_page_and_the_answers_are_never_stored_and_the_scripts_are_checked_against_their_etag(live):
    assert _call(live, "GET", "/")[2]["cache-control"] == "no-store", "the page carries the token"
    assert _call(live, "GET", "/api/catalog", {"X-Token": live.token})[2]["cache-control"] == "no-store"
    status, body, h = _call(live, "GET", "/static/app.js")
    assert status == 200 and h["cache-control"] == "no-cache" and h["etag"].startswith('"')
    again = _call(live, "GET", "/static/app.js", {"If-None-Match": h["etag"]})
    assert again[0] == 304 and again[1] == b"" and again[2]["etag"] == h["etag"]
    assert _call(live, "GET", "/static/app.js", {"If-None-Match": '"something-else"'})[0] == 200
    assert _call(live, "GET", "/static/app.js", {"If-None-Match": f'W/{h["etag"]}, "x"'})[0] == 304, "a weak validator or a list of them is understood"


def test_big_answers_and_scripts_are_compressed_for_a_browser_that_accepts_it_and_only_then(live):
    plain = _call(live, "GET", "/api/catalog", {"X-Token": live.token})
    packed = _call(live, "GET", "/api/catalog", {"X-Token": live.token, "Accept-Encoding": "gzip, deflate"})
    assert "content-encoding" not in plain[2] and packed[2]["content-encoding"] == "gzip" and packed[2]["vary"] == "Accept-Encoding"
    assert gzip.decompress(packed[1]) == plain[1] and len(packed[1]) < 0.5 * len(plain[1])
    assert int(packed[2]["content-length"]) == len(packed[1])
    js = _call(live, "GET", "/static/ml.js", {"Accept-Encoding": "gzip"})
    assert js[2]["content-encoding"] == "gzip" and gzip.decompress(js[1]) == _call(live, "GET", "/static/ml.js")[1]
    tiny = _call(live, "GET", "/api/jobs", {"X-Token": live.token, "Accept-Encoding": "gzip"})
    assert "content-encoding" not in tiny[2], "a small answer is not worth compressing"
    etag = js[2]["etag"]
    assert _call(live, "GET", "/static/ml.js", {"Accept-Encoding": "gzip", "If-None-Match": etag})[0] == 304, "the ETag is of the script, so a compressed and a plain request agree"


# ------------------------------------------------------------------------------------------------------ the access code
@pytest.fixture()
def locked(app):
    running = server.start(app, "127.0.0.1", 0, access_code="open sesame")
    yield running
    running.stop()


def _login(running, code, headers=None):
    return _call(running, "POST", "/login", {"Content-Type": "application/x-www-form-urlencoded", **(headers or {})}, ("code=" + code.replace(" ", "+")).encode())


def test_with_an_access_code_nothing_is_served_before_it_is_entered(locked):
    status, _, h = _call(locked, "GET", "/")
    assert status == 303 and h["location"] == "/login"
    status, body, _ = _call(locked, "GET", "/login")
    assert status == 200 and b"Access code" in body and locked.token.encode() not in body, "the form does not carry the token"
    assert _call(locked, "GET", "/static/app.js")[0] == 303
    status, body, _ = _call(locked, "GET", "/api/catalog", {"X-Token": locked.token})
    assert status == 401 and "access code" in json.loads(body)["error"], "knowing the token is not enough"
    assert _call(locked, "POST", "/api/tickers", {"Content-Type": "application/json", "X-Token": locked.token}, b"{}")[0] == 401


def test_the_right_code_opens_the_app_for_that_browser_through_an_httponly_cookie_and_a_wrong_one_does_not(locked):
    status, body, _ = _login(locked, "wrong")
    assert status == 401 and b"not right" in body and b"Set-Cookie" not in body
    status, _, h = _login(locked, "open sesame")
    assert status == 303 and h["location"] == "/"
    cookie = h["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/" in cookie and "Secure" not in cookie
    sid = cookie.split(";")[0]
    status, body, _ = _call(locked, "GET", "/", {"Cookie": sid})
    assert status == 200 and locked.token.encode() in body
    assert _call(locked, "GET", "/api/catalog", {"X-Token": locked.token, "Cookie": sid})[0] == 200
    assert _call(locked, "GET", "/api/catalog", {"X-Token": locked.token, "Cookie": "ql_session=forged"})[0] == 401
    assert "Secure" in _login(locked, "open sesame", {"X-Forwarded-Proto": "https"})[2]["set-cookie"], "behind an HTTPS proxy the cookie is marked Secure"
    status, _, h = _call(locked, "GET", "/logout", {"Cookie": sid})
    assert status == 303 and "Max-Age=0" in h["set-cookie"]


def test_wrong_codes_are_limited_so_the_code_cannot_be_guessed(locked):
    for _ in range(server.LOGIN_ATTEMPTS):
        assert _login(locked, "nope")[0] == 401
    status, body, h = _login(locked, "open sesame")
    assert status == 429 and b"Too many" in body and int(h["retry-after"]) > 0, "even the right code waits while guesses are being refused"


def test_without_an_access_code_nothing_changes_and_the_login_route_does_not_exist(live):
    assert _call(live, "GET", "/login")[0] == 404 and _call(live, "POST", "/login", {"Content-Type": "application/x-www-form-urlencoded"}, b"code=x")[0] == 404
    assert _call(live, "GET", "/")[0] == 200 and _call(live, "GET", "/api/catalog", {"X-Token": live.token})[0] == 200


def test_an_empty_access_code_means_no_code(app):
    running = server.start(app, "127.0.0.1", 0, access_code="")
    try:
        assert _call(running, "GET", "/")[0] == 200
    finally:
        running.stop()
