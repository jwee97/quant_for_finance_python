"""A small HTTP client for public, key-less data feeds.

Every source in this package is a public endpoint that needs no account and no key. The client therefore does only what such a feed needs: a ``GET`` that follows redirects (Cboe serves its quotes
from a redirected address), a browser-like ``User-Agent`` (several of the feeds refuse the default of urllib), a timeout, a few polite retries after a timeout, an HTTP 429 or a 5xx answer, and a
pause between calls to the same host so a loop over many symbols does not hammer one server. Its only output is parsed JSON, or a :class:`FetchError` whose message is meant to be read by the
person who pressed the button: it says what was asked, what came back and, for the answers that mean "not from where you are", what that means.

The transport is a plain function ``(url, headers, timeout) -> (status, headers, body)`` so a test can replace the network with a dictionary.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable
from urllib import error as urlerror
from urllib import parse, request

USER_AGENT = "Mozilla/5.0"                              # the feeds refuse urllib's default; Yahoo also answers HTTP 429 to full browser strings (and to Python's agent) from many addresses but serves this bare one
MAX_BODY_BYTES = 64_000_000                              # the Cboe chain of an index with weekly options is about 40 MB
Transport = Callable[[str, dict, float], "tuple[int, dict, bytes]"]


class FetchError(Exception):
    """A download that did not work. ``status`` is the HTTP status when there was an answer; ``blocked`` marks the answers that mean this server will not talk to this address."""

    def __init__(self, message: str, status: int | None = None, blocked: bool = False):
        super().__init__(message)
        self.status, self.blocked = status, blocked


def urllib_transport(url: str, headers: dict, timeout: float) -> tuple[int, dict, bytes]:
    """``(status, headers, body)`` of a GET that follows redirects. A 4xx or 5xx answer is returned, not raised; network failures raise ``OSError`` (``URLError`` and timeouts included)."""
    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return int(response.status), dict(response.headers), response.read(MAX_BODY_BYTES)
    except urlerror.HTTPError as error:
        return int(error.code), dict(error.headers or {}), error.read(MAX_BODY_BYTES)


def _host(url: str) -> str:
    return parse.urlsplit(url).netloc.lower()


def _reading(status: int) -> str:
    return {403: "refused the request (HTTP 403)", 404: "has no such item (HTTP 404)", 429: "is rate limiting this address (HTTP 429)", 451: "does not serve this location (HTTP 451)"}.get(status, f"answered HTTP {status}")


class Http:
    """``get_json(url, params)`` with retries, per-host pacing and readable errors. ``pause`` is the minimum number of seconds between two calls to one host (``pauses`` overrides it per host)."""

    def __init__(self, transport: Transport | None = None, timeout: float = 25.0, retries: int = 2, backoff: float = 1.5, pause: float = 0.2, pauses: dict | None = None,
                 cooldown: float = 90.0, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.transport = transport or urllib_transport
        self.timeout, self.retries, self.backoff, self.pause, self.cooldown = float(timeout), int(retries), float(backoff), float(pause), float(cooldown)
        self.pauses = dict(pauses or {})
        self._sleep, self._clock = sleep, clock
        self._last: dict[str, float] = {}
        self._tripped: dict[str, tuple[float, FetchError]] = {}
        self._lock = threading.Lock()
        self.calls = 0

    def _wait(self, host: str) -> None:
        gap = self.pauses.get(host, self.pause)
        with self._lock:
            wait = self._last.get(host, -1e9) + gap - self._clock()
            self._last[host] = self._clock() + max(wait, 0.0)
        if wait > 0:
            self._sleep(wait)

    def _trip(self, host: str, error: FetchError) -> FetchError:
        """Remember that ``host`` refused (rate limit, or a block on this address): for ``cooldown`` seconds further calls to it fail at once, so a loop over fifty symbols does not wait out fifty retries."""
        with self._lock:
            self._tripped[host] = (self._clock() + self.cooldown, error)
        return error

    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None) -> Any:
        if params:
            url = f"{url}{'&' if '?' in url else '?'}{parse.urlencode(params)}"
        host = _host(url)
        with self._lock:
            until, earlier = self._tripped.get(host, (0.0, None))
        if earlier is not None and self._clock() < until:
            raise FetchError(f"{earlier} (not asked again: it refused a moment ago)", status=earlier.status, blocked=earlier.blocked)
        head = {"User-Agent": USER_AGENT, "Accept": "application/json, */*", **(headers or {})}
        last = "no answer"
        for attempt in range(self.retries + 1):
            self._wait(host)
            self.calls += 1
            try:
                status, answer_headers, body = self.transport(url, head, self.timeout)
            except OSError as exc:
                last = f"{host} could not be reached ({type(exc).__name__}: {exc})"
                if attempt < self.retries:
                    self._sleep(self.backoff * (2 ** attempt))
                continue
            if status == 200:
                try:
                    return json.loads(body)
                except ValueError:
                    raise FetchError(f"{host} answered with something that is not JSON") from None
            last = f"{host} {_reading(status)}"
            if status in (429, 500, 502, 503, 504) and attempt < self.retries:
                try:
                    delay = float({str(k).lower(): v for k, v in answer_headers.items()}.get("retry-after", ""))
                except (TypeError, ValueError):
                    delay = self.backoff * (2 ** attempt)
                self._sleep(min(max(delay, 0.0), 10.0))
                continue
            error = FetchError(last, status=status, blocked=status in (403, 451))
            raise self._trip(host, error) if status in (403, 429, 451) else error
        raise FetchError(last)
