"""The one HTTP client for Massive (formerly Polygon.io): ``https://api.massive.com``.

* **Key.** ``MASSIVE_API_KEY`` from the environment or a git-ignored ``.env``; sent only in the ``Authorization: Bearer`` header (never in a URL, never to the page, never in a message or a file:
  every message goes through ``redact``). A ``next_url`` is fetched as given, with the header; an ``apiKey`` parameter in it is removed first.
* **Pooling.** One ``requests.Session`` with a connection pool shared by every thread.
* **Pagination.** ``paginate`` follows ``next_url`` until it is absent, a page limit is reached or the caller cancels.
* **Rate limiting.** ``MASSIVE_CALLS_PER_MINUTE`` (default: none, because paid plans are unlimited; set 5 for a free plan). One limiter, remembered on disk, honours ``Retry-After`` after an HTTP 429.
* **Retries.** Exponential backoff with jitter for 429, 5xx and connection errors, at most ``MASSIVE_RETRIES`` (default 4); 400, 401, 403 and 404 are never retried.
* **Usage.** Requests, retries, 429s, errors and bytes per endpoint family, kept in ``data/user/massive/usage.json``.

Massive's documentation does not show the body of an HTTP 403 or 429, so this client reads the status first and then whatever ``message`` or ``error`` text the body has; it never assumes a format.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from requests.adapters import HTTPAdapter

from ..webapp.providers import RateLimiter, RateLimitTimeout, read_secret, redact, tidy_key
from .models import MarketDataError

KEY_VARIABLE = "MASSIVE_API_KEY"
DEFAULT_BASE_URL = "https://api.massive.com"
NOT_CONFIGURED = "Massive is not set up: add your key as the MASSIVE_API_KEY secret (or a line MASSIVE_API_KEY=... in the git-ignored .env file) and restart the app"
USER_AGENT = "QuantLab/1.0 (market-data client)"
MAX_BODY_BYTES = 64_000_000


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return min(max(int(os.environ.get(name, "")), low), high)
    except ValueError:
        return default


def family(path: str) -> str:
    """A short name for the endpoint, for the usage table: the path with ids, tickers and dates removed."""
    p = urlsplit(path).path
    p = re.sub(r"/range/\d+/\w+/[^/]+/[^/]+$", "/range", p)
    p = re.sub(r"/(?:O|X|C|I):[A-Za-z0-9._-]+|/[A-Z]{1,6}\d{0,2}[A-Z]?\d?(?=/|$)", "/{id}", p)
    return p[:80]


class MassiveClient:
    def __init__(self, key: str | None = None, root: Path | None = None, base_url: str | None = None, calls_per_minute: int | None = None, session=None, sleep=time.sleep, clock=time.time,
                 retries: int | None = None, timeout: float = 30.0):
        self.root = Path(root) if root else None
        self._key_given = key
        self.base_url = (base_url or os.environ.get("MASSIVE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        if not self.base_url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise MarketDataError("the Massive address must be https", "bad_request")
        self.timeout, self._sleep, self._clock = timeout, sleep, clock
        self.retries = retries if retries is not None else _env_int("MASSIVE_RETRIES", 4, 0, 10)
        calls = calls_per_minute if calls_per_minute is not None else _env_int("MASSIVE_CALLS_PER_MINUTE", 0, 0, 100000)
        state = (self.root / "data" / "user" / "massive" / "rate.json") if self.root else None
        self.limiter = RateLimiter(calls=calls, window=60.0, margin=0.5, path=state, clock=clock, sleep=sleep) if calls else None
        self.session = session or self._make_session()
        self._lock = threading.Lock()
        self._usage: dict = {"since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "requests": 0, "retries": 0, "rate_limited": 0, "errors": 0, "bytes": 0, "by_endpoint": {}}
        self._load_usage()

    # ------------------------------------------------------------------------------------------------------------ setup
    @staticmethod
    def _make_session() -> requests.Session:
        s = requests.Session()
        s.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=16, max_retries=0))
        s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        return s

    def _raw_key(self) -> str:
        return self._key_given if self._key_given is not None else read_secret(KEY_VARIABLE, self.root)

    @property
    def key(self) -> str:
        return tidy_key(self._raw_key(), KEY_VARIABLE)[0]

    def key_notes(self) -> list[str]:
        return tidy_key(self._raw_key(), KEY_VARIABLE)[1]

    def available(self) -> tuple[bool, str]:
        if not self.key:
            return False, NOT_CONFIGURED
        notes = self.key_notes()
        return (not any("placeholder" in n for n in notes)), ("; ".join(notes) if notes else "")

    def _clean(self, text: object) -> str:
        return redact(text, self.key, self._raw_key())

    # ------------------------------------------------------------------------------------------------------------ usage
    def _usage_path(self) -> Path | None:
        return (self.root / "data" / "user" / "massive" / "usage.json") if self.root else None

    def _load_usage(self) -> None:
        p = self._usage_path()
        try:
            if p and p.is_file():
                old = json.loads(p.read_text())
                if isinstance(old, dict) and "requests" in old:
                    self._usage = old
        except (OSError, ValueError):
            pass

    def _save_usage(self) -> None:
        p = self._usage_path()
        if not p:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._usage))
            tmp.replace(p)
        except OSError:
            pass

    def _count(self, fam: str, **kw) -> None:
        with self._lock:
            self._usage["requests"] += kw.get("requests", 0)
            for k in ("retries", "rate_limited", "errors", "bytes"):
                self._usage[k] += kw.get(k, 0)
            e = self._usage["by_endpoint"].setdefault(fam, {"requests": 0, "errors": 0})
            e["requests"] += kw.get("requests", 0)
            e["errors"] += kw.get("errors", 0)
            if kw.get("requests"):
                self._usage["last_request"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._save_usage()

    def usage(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._usage))

    # ------------------------------------------------------------------------------------------------------------ requests
    @staticmethod
    def _strip_key(url: str) -> str:
        parts = urlsplit(url)
        q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() != "apikey"]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), ""))

    def _message(self, resp) -> str:
        try:
            body = resp.json()
            if isinstance(body, dict):
                text = body.get("message") or body.get("error") or body.get("status") or ""
                return self._clean(str(text))[:300]
        except ValueError:
            pass
        return self._clean((getattr(resp, "text", "") or "")[:200])

    def _classify(self, status: int, text: str, path: str) -> MarketDataError:
        if status == 401:
            return MarketDataError(f"Massive refused the key (HTTP 401){': ' + text if text else ''}. Check MASSIVE_API_KEY.", "auth", status, path)
        if status == 403:
            return MarketDataError(f"your Massive plan does not include this data (HTTP 403){': ' + text if text else ''}", "entitlement", status, path)
        if status == 404:
            return MarketDataError(f"Massive has nothing at this address or symbol (HTTP 404){': ' + text if text else ''}", "not_found", status, path)
        if status == 429:
            return MarketDataError("Massive rate limit reached (HTTP 429)", "rate_limit", status, path)
        if 400 <= status < 500:
            return MarketDataError(f"Massive rejected the request (HTTP {status}){': ' + text if text else ''}", "bad_request", status, path)
        return MarketDataError(f"Massive answered HTTP {status}{': ' + text if text else ''}", "server", status, path)

    def get(self, path_or_url: str, params: dict | None = None, cancel=None, on_wait=None) -> dict:
        """One JSON request, with limiter, retries and backoff. ``path_or_url`` is a path (``/v3/...``) or a ``next_url``."""
        ok, why = self.available()
        if not self.key:
            raise MarketDataError(NOT_CONFIGURED, "not_configured")
        if not ok:
            raise MarketDataError(f"the Massive key looks wrong: {why}", "auth")
        url = self._strip_key(path_or_url) if path_or_url.startswith("http") else self.base_url + path_or_url
        if path_or_url.startswith("http") and urlsplit(path_or_url).netloc != urlsplit(self.base_url).netloc:
            raise MarketDataError("refusing to send the key to another host than Massive's", "bad_request")
        clean = {k: v for k, v in (params or {}).items() if v is not None and k.lower() != "apikey"}
        fam = family(path_or_url if not path_or_url.startswith("http") else urlsplit(path_or_url).path)
        headers = {"Authorization": f"Bearer {self.key}"}
        attempt = 0
        while True:
            if cancel is not None and cancel():
                raise MarketDataError("cancelled", "cancelled")
            if self.limiter is not None:
                try:
                    self.limiter.acquire(on_wait=on_wait)
                except RateLimitTimeout as exc:
                    raise MarketDataError(self._clean(exc), "rate_limit") from None
            try:
                resp = self.session.get(url, params=clean or None, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                self._count(fam, requests=1, errors=1)
                if attempt >= self.retries:
                    raise MarketDataError(f"could not reach Massive: {self._clean(type(exc).__name__)}", "network", 0, fam) from None
                attempt += 1
                self._count(fam, retries=1)
                self._sleep(min(30.0, (2 ** attempt) * 0.5 + random.random() * 0.25))
                continue
            status = resp.status_code
            self._count(fam, requests=1, bytes=len(getattr(resp, "content", b"") or b""), errors=0 if status < 400 else 1)
            if status == 200:
                if len(resp.content) > MAX_BODY_BYTES:
                    raise MarketDataError("the answer is larger than the safety limit", "server", status, fam)
                try:
                    body = resp.json()
                except ValueError:
                    raise MarketDataError("Massive's answer was not JSON", "server", status, fam) from None
                if isinstance(body, dict) and str(body.get("status", "OK")).upper() in ("ERROR", "NOT_AUTHORIZED"):
                    msg = self._clean(str(body.get("message") or body.get("error") or body.get("status")))
                    raise MarketDataError(f"Massive: {msg}", "entitlement" if "AUTHORIZ" in str(body.get("status", "")).upper() else "bad_request", status, fam)
                return body
            if status in (429, 500, 502, 503, 504) and attempt < self.retries:
                attempt += 1
                self._count(fam, retries=1, rate_limited=1 if status == 429 else 0)
                wait = None
                try:
                    wait = float(resp.headers.get("Retry-After", ""))
                except (TypeError, ValueError):
                    pass
                delay = min(60.0, wait if wait is not None else (2 ** attempt) * 0.5 + random.random() * 0.25)
                if status == 429 and self.limiter is not None:
                    self.limiter.hold(delay)
                self._sleep(delay)
                continue
            if status == 429:
                self._count(fam, rate_limited=1)
            raise self._classify(status, self._message(resp), fam)

    def paginate(self, path: str, params: dict | None = None, max_pages: int = 100, cancel=None, on_page=None, on_wait=None):
        """Yield ``(results, body)`` per page, following ``next_url``. Stops after ``max_pages`` or when ``cancel()`` is true (raising ``cancelled``)."""
        page, nxt = 0, None
        while page < max_pages:
            body = self.get(nxt or path, None if nxt else params, cancel=cancel, on_wait=on_wait)
            page += 1
            results = body.get("results") if isinstance(body, dict) else None
            if results is None:
                results = []
            elif isinstance(results, dict):
                results = [results]
            if on_page:
                on_page(page, len(results))
            yield results, body
            nxt = body.get("next_url") if isinstance(body, dict) else None
            if not nxt:
                return
