"""Helpers shared by the keyed data providers (Massive, Finzdata, the fundamentals sources): reading a secret from the environment or a git-ignored ``.env``, tidying a pasted key, redacting it from every
message, one rate limiter shared by all threads, and a GET that never follows a redirect (the key travels in a header and must not be re-sent to wherever a redirect points).

**The API key.** It lives only in the environment (on Replit: the Secrets tool) or in a git-ignored ``.env`` file. It is sent in one request header to its own provider and nowhere else, and never put in a
response, a log line, an error message or a file: every message that leaves goes through :func:`redact`.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import parse, request

MAX_BODY_BYTES = 8_000_000


# ------------------------------------------------------------------------------------------------------------------------------- secrets
def read_secret(name: str, root: Path | None = None) -> str:
    """The environment variable ``name``, else the same name in ``<root>/.env`` (a git-ignored file of ``NAME=value`` lines), else an empty string."""
    value = os.environ.get(name, "").strip()
    if value or root is None:
        return value
    try:
        lines = (Path(root) / ".env").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, rest = line.partition("=")
        if key.strip().removeprefix("export ").strip() == name:
            return rest.strip().strip("'\"")
    return ""


_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00a0"))       # what a copy and paste can carry along that nobody can see
_PLACEHOLDER = re.compile(r"(?i)^<?\s*(?:your|my)[ _-]*(?:massive[ _-]*)?(?:api[ _-]*)?(?:key|token)\s*>?$|^(?:key|token|xxx+|\*+)$")


def tidy_key(raw: object, name: str) -> tuple[str, list[str]]:
    """``(key, notes)``: the key without the wrapping a paste often adds (quotes, spaces, a line break, invisible characters, ``MASSIVE_API_KEY=`` or ``Bearer`` in front of it), and a note for
    everything removed or still wrong. The notes never contain the key."""
    notes: list[str] = []
    text = str(raw or "")
    cleaned = text.translate(_INVISIBLE)
    if cleaned != text:
        notes.append("invisible characters (a zero-width or non-breaking space) were removed from it")
    text, cleaned = cleaned, cleaned.strip()
    if cleaned != text:
        notes.append("spaces or a line break around it were removed")
    text = cleaned

    def unquote(value: str) -> str:
        while len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"`":
            value = value[1:-1].strip()
            if "quotation marks around it were removed" not in notes:
                notes.append("quotation marks around it were removed")
        return value

    text = unquote(text)
    for pattern, what in ((rf"(?i)^(?:export\s+)?{re.escape(name)}\s*=\s*", f"the secret's name ({name}=) was pasted in front of the key and was removed"),
                          (r"(?i)^(?:token|api[_-]?key)\s*[=:]\s*", "a label (token= or apikey=) was pasted in front of the key and was removed"),
                          (r"(?i)^bearer\s+", "the word Bearer was pasted in front of the key and was removed")):
        match = re.match(pattern, text)
        if match:
            text = unquote(text[match.end():].strip())
            notes.append(what)
    if re.search(r"\s", text):
        notes.append("it has a space or a line break inside, which no key has: only the key itself should be pasted")
    if text and not text.isascii():
        notes.append("it has characters outside plain ASCII (often a copied smart quote)")
    if _PLACEHOLDER.match(text):
        notes.append("it looks like a placeholder, not a key")
    return text, notes


def redact(text: object, *secrets: str) -> str:
    """``text`` with every secret (and its URL-encoded form) replaced by ``***``."""
    out = str(text)
    for secret in secrets:
        if secret and len(secret) >= 6:
            out = out.replace(secret, "***").replace(parse.quote(secret, safe=""), "***")
    return out


# ------------------------------------------------------------------------------------------------------------------------- rate limiting
class RateLimitTimeout(RuntimeError):
    """The wait for a free call slot would be longer than the caller allows."""


class RateLimiter:
    """At most ``calls`` calls in any ``window`` seconds (plus a safety ``margin``), shared by every thread.

    The times of recent calls are kept in ``path`` (when given) so a restart does not reset the allowance; :meth:`hold` blocks everything for a while, which is what an HTTP 429 asks for.
    """

    def __init__(self, calls: int = 5, window: float = 60.0, margin: float = 1.0, path: Path | None = None, clock=time.time, sleep=time.sleep):
        if calls < 1 or window <= 0 or margin < 0:
            raise ValueError("calls >= 1, window > 0 and margin >= 0")
        self.calls, self.window, self.margin = int(calls), float(window), float(margin)
        self._clock, self._sleep, self._path = clock, sleep, (Path(path) if path else None)
        self._lock = threading.Lock()
        self._stamps: list[float] = []
        self._blocked_until = 0.0
        self._load()

    @property
    def span(self) -> float:
        return self.window + self.margin

    def _load(self) -> None:
        if self._path is None:
            return
        try:
            state = json.loads(self._path.read_text(encoding="utf-8"))
            now = self._clock()
            self._stamps = sorted(float(t) for t in state.get("calls", []) if -5.0 <= now - float(t) < self.span)          # a clock that moved back must not block for hours
            blocked = float(state.get("blocked_until", 0.0))
            self._blocked_until = blocked if now < blocked <= now + 3600 else 0.0
        except (OSError, ValueError, TypeError, AttributeError):
            pass                                                                    # no history (or a damaged file) just means a fresh start

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"calls": self._stamps, "blocked_until": self._blocked_until}), encoding="utf-8")
            tmp.replace(self._path)
        except OSError:
            pass                                                                    # the limit then holds for this process only

    def acquire(self, on_wait=None, max_wait: float = 900.0) -> float:
        """Wait until a call is allowed, record it and return the seconds waited. ``on_wait(delay, reason)`` is called about once a second while waiting."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._stamps = [t for t in self._stamps if now - t < self.span]
                if now < self._blocked_until:
                    delay, why = self._blocked_until - now, "the provider asked for a pause"
                elif len(self._stamps) < self.calls:
                    self._stamps.append(now)
                    self._save()
                    return waited
                else:
                    delay, why = self._stamps[0] + self.span - now, f"the limit allows {self.calls} call{'s' if self.calls != 1 else ''} a minute"
            if waited + delay > max_wait:
                raise RateLimitTimeout(f"the next call would have to wait {int(waited + delay)} s ({why})")
            if on_wait is not None:
                on_wait(delay, why)
            step = min(max(delay, 0.0), 1.0)
            self._sleep(step)
            waited += step

    def hold(self, seconds: float) -> None:
        """Allow no call for ``seconds`` (the provider answered HTTP 429)."""
        with self._lock:
            self._blocked_until = max(self._blocked_until, self._clock() + min(max(float(seconds), 0.0), 3600.0))
            self._save()

    def remaining(self) -> int:
        """Calls that could be made right now without waiting."""
        with self._lock:
            now = self._clock()
            if now < self._blocked_until:
                return 0
            return max(0, self.calls - sum(1 for t in self._stamps if now - t < self.span))

    def estimate_seconds(self, n_calls: int) -> float:
        """An upper bound on the seconds ``n_calls`` calls would take from now: free slots first, then one full window per ``calls`` further calls."""
        free = self.remaining()
        return 0.0 if n_calls <= free else self.span * math.ceil((n_calls - free) / self.calls)




# ------------------------------------------------------------------------------------------------------------------------------ transport
class _NoRedirect(request.HTTPRedirectHandler):
    """Never follow a redirect: the key travels in a header and must not be re-sent to wherever a redirect points."""

    def redirect_request(self, *args, **kwargs):
        return None


def urllib_transport(url: str, headers: dict, timeout: float) -> tuple[int, dict, bytes]:
    """``(status, headers, body)`` of a GET. Network failures raise ``OSError`` (including ``URLError`` and timeouts)."""
    opener = request.build_opener(_NoRedirect)
    req = request.Request(url, headers=headers, method="GET")
    try:
        with opener.open(req, timeout=timeout) as response:
            return int(response.status), dict(response.headers), response.read(MAX_BODY_BYTES)
    except urlerror.HTTPError as error:                                             # 4xx and 5xx (and an unfollowed redirect) are answers, not failures
        return int(error.code), dict(error.headers or {}), error.read(MAX_BODY_BYTES)

