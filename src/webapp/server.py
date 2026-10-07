"""A small threaded HTTP server for the dashboard.

It listens on localhost only by default. Every API call needs the per-launch token that is embedded in the page it served (so another web page you have open
cannot drive it), the Host and Origin headers must be this server, and POST bodies must be JSON of at most 200 KB.
"""

from __future__ import annotations

import json
import secrets
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .api import ApiError, App

STATIC = Path(__file__).parent / "static"
FILES = {"app.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8", "charts.js": "text/javascript; charset=utf-8"}
MAX_BODY = 200_000
LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


def make_handler(app: App, token: str, port_holder: list):
    class Handler(BaseHTTPRequestHandler):
        server_version = "QuantLab"

        def log_message(self, fmt, *args):                      # quiet: the terminal belongs to the backtests
            pass

        # ------------------------------------------------------------------------------------------ helpers
        def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload, allow_nan=False).encode(), "application/json")

        def _host_ok(self) -> bool:
            if self.server.server_address[0] not in LOOPBACK:         # deliberately exposed: the token is the only gate
                return True
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0] if not (self.headers.get("Host") or "").startswith("[") else "[::1]"
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).hostname not in LOOPBACK:
                return False
            return host in LOOPBACK

        def _guard(self, api: bool) -> bool:
            if not self._host_ok():
                self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host or Origin"})
                return False
            if api and not secrets.compare_digest(self.headers.get("X-Token", ""), token):
                self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong token: reload the page"})
                return False
            return True

        def _body(self) -> dict | None:
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send application/json"})
                return None
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
                return None
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
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
        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                if self._guard(False):
                    html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", token)
                    self._send(HTTPStatus.OK, html.encode(), "text/html; charset=utf-8")
            elif path.startswith("/static/") and path[8:] in FILES:
                if self._guard(False):
                    self._send(HTTPStatus.OK, (STATIC / path[8:]).read_bytes(), FILES[path[8:]])
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
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            if not path.startswith("/api/") or not self._guard(True):
                if not path.startswith("/api/"):
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            routes = {"/api/tickers": app.check_tickers, "/api/backtest": app.submit, "/api/formula": app.check_formula}
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


def start(app: App, host: str = "127.0.0.1", port: int = 8765) -> Running:
    """Start serving in a background thread (port 0 picks a free one)."""
    token = secrets.token_urlsafe(16)
    holder: list = []
    httpd = ThreadingHTTPServer((host, port), make_handler(app, token, holder))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return Running(httpd, token, thread)


def serve(config, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    app = App(config)
    running = start(app, host, port)
    print(f"Quant Lab is running at {running.url}   (Ctrl+C to stop)")
    if host not in LOOPBACK:
        print("WARNING: listening on a non-local address; anyone who can reach this port and obtains the page can run backtests on this machine.")
    if open_browser:
        webbrowser.open(running.url)
    try:
        running.thread.join()
    except KeyboardInterrupt:
        print("\nstopping")
        running.stop()
    return 0
