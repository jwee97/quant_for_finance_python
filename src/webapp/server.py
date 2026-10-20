"""A small threaded HTTP server for the dashboard.

It listens on localhost only by default. Every API call needs the per-launch token that is embedded in the page it served (so another web page you have open
cannot drive it), the Host and Origin headers must be this server, and POST bodies must be JSON of at most 200 KB.

Where the server is reachable by others (a Replit preview, a container), the token alone is no gate: anyone who can load the page receives it. An **access code** closes that: with
``QUANT_LAB_ACCESS_CODE`` set, the page, the scripts and the API are served only to a browser that has entered the code (a session cookie that is HttpOnly and SameSite=Strict, kept in
memory, so a restart signs everyone out; wrong guesses are limited to ten in five minutes).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import secrets
import threading
import time
import webbrowser
from collections import deque
from html import escape
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .api import ApiError, App

STATIC = Path(__file__).parent / "static"
JS = "text/javascript; charset=utf-8"
FILES = {"app.js": JS, "style.css": "text/css; charset=utf-8", "charts.js": JS, "workspace.js": JS, "options.js": JS, "futures.js": JS, "crypto.js": JS, "ml.js": JS, "tune.js": JS, "forecast.js": JS, "fundamentals.js": JS, "fundamentals_lab.js": JS}
MAX_BODY = 200_000
MAX_QUERY_FIELDS = 40
LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}
ACCESS_ENV = "QUANT_LAB_ACCESS_CODE"
SESSION_COOKIE, SESSION_SECONDS = "ql_session", 24 * 3600
LOGIN_ATTEMPTS, LOGIN_WINDOW = 10, 300.0
REQUEST_TIMEOUT = 30                                          # seconds a client may take to send its request or take the answer: a stalled connection is dropped, not kept for ever
MIN_GZIP = 1024
COMPRESSIBLE = ("text/", "application/json")

LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Quant Lab: access code</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;margin:0;display:grid;place-items:center;min-height:100vh;background:#fcfcfb;color:#0b0b0b}}@media(prefers-color-scheme:dark){{body{{background:#0b0b0b;color:#f2f2f0}}}}
main{{width:min(360px,calc(100vw - 32px))}}h1{{font-size:1.3rem;margin:0 0 .25rem}}p{{margin:.25rem 0 1rem;opacity:.8}}label{{display:block;font-weight:600;margin-bottom:.25rem}}
input{{width:100%;box-sizing:border-box;font:inherit;padding:.6rem .7rem;border:1px solid #888;border-radius:8px;background:transparent;color:inherit}}
button{{margin-top:.8rem;font:inherit;padding:.6rem 1rem;border:0;border-radius:8px;background:#226fcb;color:#fff;cursor:pointer}}.err{{color:#b3261e;margin:.5rem 0 0}}@media(prefers-color-scheme:dark){{.err{{color:#ff8a80}}}}
:focus-visible{{outline:2px solid #226fcb;outline-offset:2px}}</style></head>
<body><main><h1>Quant Lab</h1><p>This app asks for its access code before it shows anything.</p>
<form method="post" action="/login"><label for="code">Access code</label><input id="code" name="code" type="password" autocomplete="current-password" autofocus required>{message}
<button type="submit">Open the app</button></form></main></body></html>"""


def make_handler(app: App, token: str, port_holder: list, access: str | None = None):
    sessions: dict[str, float] = {}                             # session id -> when it expires
    attempts: deque = deque()                                   # times of recent wrong guesses
    gate = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "QuantLab"
        timeout = REQUEST_TIMEOUT

        def version_string(self) -> str:                        # do not announce which Python this is
            return self.server_version

        def log_message(self, fmt, *args):                      # quiet: the terminal belongs to the backtests
            pass

        # ------------------------------------------------------------------------------------------ helpers
        def _headers(self, extra: dict | None = None) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=(), usb=()")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            # Only the exposed Replit preview may be embedded; local use stays frame-blocked.
            ancestors = "'none'"
            if self.server.server_address[0] not in LOOPBACK and os.environ.get("REPLIT_DEV_DOMAIN"):
                ancestors = "https://replit.com https://*.replit.com https://*.replit.dev"
            self.send_header("Content-Security-Policy", f"default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; object-src 'none'; "
                                                        f"frame-ancestors {ancestors}")
            for k, v in (extra or {}).items():
                self.send_header(k, v)

        def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None, revalidate: bool = False) -> None:
            """Send ``body``, compressed when the browser accepts it and it is worth it. A page that carries the token or an answer is ``no-store``; a script or stylesheet (``revalidate``) is kept
            by the browser but checked against its ETag on every use, so an updated app is picked up at once and an unchanged one is not sent again."""
            headers = dict(extra or {})
            if revalidate:
                etag = '"' + hashlib.sha1(body).hexdigest()[:20] + '"'
                headers["ETag"], headers["Cache-Control"] = etag, "no-cache"
                theirs = [v.strip().removeprefix("W/") for v in (self.headers.get("If-None-Match") or "").split(",")]
                if etag in theirs:
                    self.send_response(HTTPStatus.NOT_MODIFIED)
                    self._headers(headers)
                    self.end_headers()
                    return
            else:
                headers["Cache-Control"] = "no-store"
            if len(body) >= MIN_GZIP and content_type.startswith(COMPRESSIBLE) and "gzip" in (self.headers.get("Accept-Encoding") or "").lower():
                body = gzip.compress(body, 5, mtime=0)
                headers["Content-Encoding"] = "gzip"
            headers["Vary"] = "Accept-Encoding"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._headers(headers)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload, allow_nan=False).encode(), "application/json")

        def _redirect(self, where: str, extra: dict | None = None) -> None:
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", where)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self._headers(extra)
            self.end_headers()

        def _host_ok(self) -> bool:
            if self.server.server_address[0] not in LOOPBACK:         # deliberately exposed: the token (and the access code, if one is set) is the only gate
                return True
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0] if not (self.headers.get("Host") or "").startswith("[") else "[::1]"
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).hostname not in LOOPBACK:
                return False
            return host in LOOPBACK

        # ------------------------------------------------------------------------------------------ the access code
        def _signed_in(self) -> bool:
            if access is None:
                return True
            jar = SimpleCookie()
            try:
                jar.load(self.headers.get("Cookie") or "")
            except Exception:
                return False
            morsel = jar.get(SESSION_COOKIE)
            with gate:
                now = time.time()
                for sid in [s for s, until in sessions.items() if until < now]:
                    sessions.pop(sid, None)
                return bool(morsel and morsel.value in sessions)

        def _login_page(self, status: int = HTTPStatus.OK, message: str = "") -> None:
            note = f'<p class="err" role="alert">{escape(message)}</p>' if message else ""
            self._send(status, LOGIN_PAGE.format(message=note).encode(), "text/html; charset=utf-8")

        def _login(self) -> None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 <= length <= 2048:
                self._login_page(HTTPStatus.BAD_REQUEST, "That request could not be read.")
                return
            code = parse_qs(self.rfile.read(length).decode("utf-8", "replace")).get("code", [""])[0]
            with gate:
                now = time.time()
                while attempts and attempts[0] < now - LOGIN_WINDOW:
                    attempts.popleft()
                blocked = len(attempts) >= LOGIN_ATTEMPTS
                good = not blocked and secrets.compare_digest(code.encode(), (access or "").encode())
                if not good:
                    attempts.append(now)
                else:
                    sid = secrets.token_urlsafe(32)
                    sessions[sid] = now + SESSION_SECONDS
            if blocked:
                self._send(HTTPStatus.TOO_MANY_REQUESTS, LOGIN_PAGE.format(message='<p class="err" role="alert">Too many wrong codes. Wait a few minutes and try again.</p>').encode(), "text/html; charset=utf-8",
                           {"Retry-After": str(int(LOGIN_WINDOW))})
            elif not good:
                self._login_page(HTTPStatus.UNAUTHORIZED, "That code is not right.")
            else:
                secure = "; Secure" if (self.headers.get("X-Forwarded-Proto") or "").lower() == "https" else ""
                self._redirect("/", {"Set-Cookie": f"{SESSION_COOKIE}={sid}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_SECONDS}{secure}"})

        def _guard(self, api: bool) -> bool:
            if not self._host_ok():
                self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host or Origin"})
                return False
            if not self._signed_in():
                if api:
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "this app is locked: open the page and enter the access code"})
                else:
                    self._redirect("/login")
                return False
            if api and not secrets.compare_digest(self.headers.get("X-Token", ""), token):
                self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong token: reload the page"})
                return False
            return True

        def _body(self) -> dict | None:
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send application/json"})
                return None
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0:                                          # a negative or unreadable length would make the read wait for the client to hang up
                self._json(HTTPStatus.BAD_REQUEST, {"error": "the Content-Length is not a valid size"})
                return None
            if length > MAX_BODY:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
                return None
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "the request body is not valid JSON"})
                return None
            if not isinstance(body, dict):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "the request body must be a JSON object"})
                return None
            return body

        def _call(self, fn, *args) -> None:
            try:
                self._json(HTTPStatus.OK, fn(*args))
            except ApiError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:                             # reach the page as a message, not a dropped connection
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})

        # ------------------------------------------------------------------------------------------ routes
        def _query(self) -> dict:
            """The query string as ``{name: last value}`` (at most ``MAX_QUERY_FIELDS`` fields, so a crafted URL cannot make the parser do unbounded work)."""
            try:
                fields = parse_qs(urlparse(self.path).query, keep_blank_values=True, max_num_fields=MAX_QUERY_FIELDS)
            except ValueError:
                raise ApiError("too many query fields") from None
            return {k: v[-1] for k, v in fields.items()}

        def do_GET(self):
            path = urlparse(self.path).path
            if access is not None and path == "/login":
                if self._host_ok():
                    self._login_page()
                else:
                    self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host or Origin"})
            elif access is not None and path == "/logout":
                self._redirect("/login", {"Set-Cookie": f"{SESSION_COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"})
            elif path in ("/", "/index.html"):
                if self._guard(False):
                    html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", token)
                    self._send(HTTPStatus.OK, html.encode(), "text/html; charset=utf-8")
            elif path.startswith("/static/") and path[8:] in FILES:
                if self._guard(False):
                    self._send(HTTPStatus.OK, (STATIC / path[8:]).read_bytes(), FILES[path[8:]], revalidate=True)
            elif path.startswith("/api/"):
                if not self._guard(True):
                    return
                if path == "/api/catalog":
                    self._call(app.catalog)
                elif path == "/api/docs":
                    self._call(app.docs_index)
                elif path.startswith("/api/docs/"):
                    self._call(app.doc, path[len("/api/docs/"):])
                elif path.startswith("/api/job/"):
                    self._call(app.job, path[len("/api/job/"):])
                elif path in app.extra_get:
                    self._call(lambda: app.extra_get[path](self._query()))
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            if access is not None and path == "/login":
                if self._host_ok():
                    self._login()
                else:
                    self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host or Origin"})
                return
            if not path.startswith("/api/") or not self._guard(True):
                if not path.startswith("/api/"):
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            routes = {"/api/tickers": app.check_tickers, "/api/backtest": app.submit, "/api/formula": app.check_formula, "/api/requirements": app.requirements,
                      "/api/source/test": app.test_source, "/api/exec": app.exec_lab, "/api/cash": app.cash_lab, **app.extra_post}
            if path in routes:
                body = self._body()
                if body is not None:
                    self._call(routes[path], body)
            elif path == "/api/reload":
                self._call(app.reload_strategies)
            elif path == "/api/trials/reset":
                self._call(app.reset_trials)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    return Handler


class Running:
    """A started server: ``url``, ``token`` and ``stop()``."""

    def __init__(self, httpd: ThreadingHTTPServer, token: str, thread: threading.Thread):
        self.httpd, self.token, self.thread = httpd, token, thread
        self.url = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}/"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def start(app: App, host: str = "127.0.0.1", port: int = 8765, access_code: str | None = None) -> Running:
    """Start serving in a background thread (port 0 picks a free one). With an ``access_code`` nothing is served to a browser that has not entered it."""
    token = secrets.token_urlsafe(16)
    holder: list = []
    httpd = ThreadingHTTPServer((host, port), make_handler(app, token, holder, access_code or None))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return Running(httpd, token, thread)


def serve(config, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    app = App(config)
    access = os.environ.get(ACCESS_ENV) or None
    running = start(app, host, port, access)
    print(f"Quant Lab is running at {running.url}   (Ctrl+C to stop)")
    if access:
        print("An access code is required (the QUANT_LAB_ACCESS_CODE secret).")
    if host not in LOOPBACK and not access:
        print("WARNING: listening on a non-local address; anyone who can reach this port and obtains the page can run backtests on this machine. "
              f"Set {ACCESS_ENV} (a Replit secret) to ask for a code first.")
    if open_browser:
        webbrowser.open(running.url)
    try:
        running.thread.join()
    except KeyboardInterrupt:
        print("\nstopping")
        running.stop()
    return 0
