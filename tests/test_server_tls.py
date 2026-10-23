"""The dashboard's server over HTTPS, when a certificate and its key are given: it answers over TLS and only over TLS, a client that stalls in the handshake holds nothing but its own thread, the session
cookie is marked Secure, the oldest version offered is TLS 1.2, and a missing or mismatched file is told in words before anything starts."""

from __future__ import annotations

import http.client
import shutil
import socket
import ssl
import subprocess
import time
from urllib.parse import urlparse

import pytest

from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.test_webapp import _default, _fetch

needs_openssl = pytest.mark.skipif(shutil.which("openssl") is None, reason="the openssl command is needed to make a throwaway certificate")


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=_default), root=tmp_path)


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    """A certificate for 127.0.0.1, made for this test and thrown away with it."""
    folder = tmp_path_factory.mktemp("tls")
    cert, key = folder / "cert.pem", folder / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(cert), "-days", "2", "-subj", "/CN=localhost",
                    "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"], check=True, capture_output=True)
    return cert, key


def _client_context(cert) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=str(cert))                                          # trust exactly this certificate, and check the name on it
    return context


def _get(running, cert, path="/", method="GET", headers=None, body=None, timeout=10):
    u = urlparse(running.url)
    conn = http.client.HTTPSConnection("127.0.0.1", u.port, timeout=timeout, context=_client_context(cert))
    try:
        conn.request(method, path, body=body, headers=headers or {})
        r = conn.getresponse()
        return r.status, r.read(), {k.lower(): v for k, v in r.getheaders()}
    finally:
        conn.close()


@needs_openssl
def test_the_page_is_served_over_tls_with_the_same_headers_and_not_over_plain_http(app, certificate):
    cert, key = certificate
    running = server.start(app, "127.0.0.1", 0, None, server.tls_context(str(cert), str(key)))
    try:
        assert running.url.startswith("https://127.0.0.1:") and running.secure is True
        status, body, headers = _get(running, cert)
        assert status == 200 and b"<html" in body.lower() and headers["x-content-type-options"] == "nosniff" and "default-src 'self'" in headers["content-security-policy"]
        assert "strict-transport-security" not in headers                                     # on localhost it would lock the browser out of every other local server
        status, body, _ = _get(running, cert, "/api/jobs", headers={"X-Token": running.token})
        assert status == 200
        u = urlparse(running.url)
        plain = http.client.HTTPConnection("127.0.0.1", u.port, timeout=5)                   # plain HTTP to the TLS port: no page
        with pytest.raises((http.client.HTTPException, ConnectionError, OSError)):
            plain.request("GET", "/")
            plain.getresponse().read()
        plain.close()
        wrong = ssl.create_default_context()                                                 # a client that does not trust the certificate is refused by its own check
        with pytest.raises(ssl.SSLError):
            http.client.HTTPSConnection("127.0.0.1", u.port, timeout=5, context=wrong).request("GET", "/")
    finally:
        running.stop()


@needs_openssl
def test_a_client_that_connects_and_never_finishes_the_handshake_does_not_hold_up_the_others(app, certificate, monkeypatch):
    cert, key = certificate
    monkeypatch.setattr(server, "REQUEST_TIMEOUT", 2)
    running = server.start(app, "127.0.0.1", 0, None, server.tls_context(str(cert), str(key)))
    port = urlparse(running.url).port
    stalled = [socket.create_connection(("127.0.0.1", port), timeout=5) for _ in range(3)]       # three connections that say nothing at all
    try:
        t0 = time.time()
        status, _, _ = _get(running, cert, timeout=5)
        assert status == 200 and time.time() - t0 < 4.0                                       # the handshake is done in each connection's own thread: the listener never waits for it (three stalled ones, one after another, would take six seconds)
        mid = socket.create_connection(("127.0.0.1", port), timeout=5)                         # and one that starts a handshake with rubbish
        mid.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        assert _get(running, cert, timeout=5)[0] == 200
        mid.close()
    finally:
        for s in stalled:
            s.close()
        running.stop()


@needs_openssl
def test_the_session_cookie_is_marked_secure_over_tls_and_the_login_works(app, certificate):
    cert, key = certificate
    running = server.start(app, "127.0.0.1", 0, "open sesame", server.tls_context(str(cert), str(key)))
    try:
        status, _, headers = _get(running, cert)
        assert status == 303 and headers["location"] == "/login"
        status, _, headers = _get(running, cert, "/login", "POST", {"Content-Type": "application/x-www-form-urlencoded"}, b"code=open+sesame")
        cookie = headers["set-cookie"]
        assert status == 303 and "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Secure" in cookie
        status, body, _ = _get(running, cert, "/", headers={"Cookie": cookie.split(";")[0]})
        assert status == 200 and b"<html" in body.lower()
    finally:
        running.stop()


def test_only_tls_one_point_two_and_newer_is_offered_and_a_bad_file_is_told_in_words(tmp_path):
    for name in ("missing.pem",):
        with pytest.raises(ValueError, match="missing"):
            server.tls_context(str(tmp_path / name), str(tmp_path / "key.pem"))
    junk = tmp_path / "junk.pem"
    junk.write_text("not a certificate")
    with pytest.raises(ValueError, match="PEM files that belong together"):
        server.tls_context(str(junk), str(junk))


@needs_openssl
def test_the_context_offers_nothing_older_than_tls_one_point_two_and_a_key_that_does_not_belong_is_refused(tmp_path, certificate):
    cert, key = certificate
    context = server.tls_context(str(cert), str(key))
    assert context.minimum_version == ssl.TLSVersion.TLSv1_2 and context.protocol == ssl.PROTOCOL_TLS_SERVER
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(other / "k.pem"), "-out", str(other / "c.pem"), "-days", "2", "-subj", "/CN=other"],
                   check=True, capture_output=True)
    with pytest.raises(ValueError, match="belong together"):
        server.tls_context(str(cert), str(other / "k.pem"))


def test_serve_refuses_to_start_with_only_one_of_the_two_files_or_with_files_that_do_not_work(monkeypatch, capsys, tmp_path):
    config = load_config()
    monkeypatch.setenv("QUANT_LAB_TLS_CERT", str(tmp_path / "cert.pem"))
    monkeypatch.delenv("QUANT_LAB_TLS_KEY", raising=False)
    assert server.serve(config, "127.0.0.1", 0, open_browser=False) == 2
    assert "needs both QUANT_LAB_TLS_CERT and QUANT_LAB_TLS_KEY" in capsys.readouterr().out
    monkeypatch.setenv("QUANT_LAB_TLS_KEY", str(tmp_path / "key.pem"))
    assert server.serve(config, "127.0.0.1", 0, open_browser=False) == 2
    assert "missing" in capsys.readouterr().out
