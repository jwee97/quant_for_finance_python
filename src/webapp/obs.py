"""Observability of the web layer: a correlation id on every request, a structured request log, and error messages that carry no file paths.

* Every response has an ``X-Request-Id`` header (the client's own id when it sent a sane one, else a new one). The same id is in the request log line, in the log of an unexpected error with its traceback, and in the
  message the page shows ("... (ref 3fa9c2d1b0e4)"), so a failure on the page can be found in the log.
* The request log (``data/user/logs/requests.jsonl``) holds, per request: time, id, method, route, the NAMES of the query fields (never their values), status, duration in milliseconds, response size, and an error category.
  Bodies, query values, cookies and keys are never logged. The file rotates at 5 MB and keeps three generations.
* An unexpected error's text is scrubbed of absolute paths and clipped before it reaches the page; the traceback goes only to the log.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
import traceback
from pathlib import Path

ROTATE_BYTES = 5_000_000
KEEP = 3
_ID_OK = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_PATH = re.compile(r"(?:[A-Za-z]:\\|/)(?:[\w.\-+]+[\\/])+[\w.\-+]*")

CATEGORIES = {400: "bad_request", 401: "unauthorised", 403: "forbidden", 404: "not_found", 405: "method_not_allowed", 413: "too_large", 429: "rate_limited", 500: "internal_error", 503: "unavailable"}


def new_id() -> str:
    return secrets.token_hex(6)


def request_id(header: str | None) -> str:
    """The client's id if it is a short token of safe characters (so it cannot inject a header or a log line), else a new one."""
    return header if header and _ID_OK.fullmatch(header) else new_id()


def scrub(text: str, roots=(), limit: int = 300) -> str:
    """The message without absolute paths (the repository and home folders become ``...``), clipped."""
    out = str(text)
    for r in sorted({str(x) for x in roots if x}, key=len, reverse=True):
        out = out.replace(r, "...")
    out = _PATH.sub("...", out)
    return out[:limit] + ("..." if len(out) > limit else "")


def category(status: int, error_kind: str = "") -> str:
    return error_kind or CATEGORIES.get(int(status), "ok" if int(status) < 400 else "error")


class RequestLog:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _rotate(self) -> None:
        try:
            if self.path.exists() and self.path.stat().st_size > ROTATE_BYTES:
                for i in range(KEEP - 1, 0, -1):
                    a, b = self.path.with_name(f"{self.path.name}.{i}"), self.path.with_name(f"{self.path.name}.{i + 1}")
                    if a.exists():
                        os.replace(a, b)
                os.replace(self.path, self.path.with_name(self.path.name + ".1"))
        except OSError:
            pass

    def write(self, **fields) -> None:
        """One JSON line. A log that cannot be written is skipped: logging never fails a request."""
        line = json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}, default=str)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate()
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:
            pass

    def error(self, rid: str, route: str, exc: BaseException) -> None:
        self.write(kind="error", request_id=rid, route=route, error=type(exc).__name__, message=str(exc)[:500], traceback="".join(traceback.format_exception(exc))[-4000:])

    def tail(self, n: int = 50, only_errors: bool = False) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines()[-2000:]:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not only_errors or e.get("kind") == "error" or int(e.get("status", 0) or 0) >= 500:
                out.append(e)
        return out[-n:]
