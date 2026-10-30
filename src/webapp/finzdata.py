"""Finzdata (``api.finzdata.com``) as a place to download futures from, and the record of what it was asked.

**The API key.** It lives only in the environment (on Replit: the Secrets tool, which sets the ``FINZDATA_API_KEY`` variable) or in a git-ignored ``.env`` file. It is read when a call is made, sent in one
request header (``X-API-Key``) to Finzdata and nowhere else, and never put in a response, a log line, an error message or a file: every message that leaves this module goes through ``redact``. The page
can see only whether a key exists, never the key. A redirect is never followed, so the header cannot be re-sent to another address. Finzdata's keys begin with ``fz_``; one that does not is reported in a
note and still tried.

**The allowance.** The free plan allows 1,000 calls a day. Every call is counted in ``data/user/finzdata_calls.json`` (by UTC day, so restarting the app does not hand out a fresh allowance) and the
client refuses to send more than ``FINZDATA_DAILY_CALLS`` of them a day (900 unless set), so a download cannot use up the day by itself. A 429 from the service stops the run at once.

**Errors** are shown in Finzdata's own words (its answers carry a ``detail``): a 401 is a missing or refused key, a 403 a plan without this data, a 429 the day's limit, a 422 a request it did not
understand.

**What is and is not verified.** The paths, parameters and error codes follow Finzdata's OpenAPI document and public pages (see ``assets.free_data.finzdata``). Nothing here has been run against the live
service with a key: ``experiments/live_feed_smoke.py`` (the ``finzdata`` check) is how that is done, and prints the columns the service really sends.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import parse

from ..assets.free_data.finzdata import BASE_URL, KEY_VARIABLE
from ..assets.free_data.http import FetchError
from .providers import read_secret, redact, tidy_key, urllib_transport

URL_VARIABLE = "FINZDATA_URL"                             # the vendor's own Python client reads the same variable
BUDGET_VARIABLE = "FINZDATA_DAILY_CALLS"
DEFAULT_BUDGET = 900
DAILY_ALLOWANCE = 1000                                    # the free plan's calls a day, as published
USER_AGENT = "QuantLab/1.0 (futures data client)"
NOT_CONFIGURED = "Finzdata is not set up: add your key as the FINZDATA_API_KEY secret (on Replit: Tools > Secrets) and restart the app"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
LEDGER_DAYS = 8


class FinzdataError(FetchError):
    """A call that did not work, in words fit for the page. ``kind`` is ``key``, ``plan``, ``quota``, ``budget``, ``request``, ``server`` or ``network``; ``fatal`` marks the ones after which asking again for
    something else in the same run would fail the same way (the key, the day's allowance)."""

    def __init__(self, message: str, status: int | None = None, kind: str = "request", fatal: bool = False):
        super().__init__(message, status=status)
        self.kind, self.fatal = kind, fatal


class CallLedger:
    """How many calls were made on each UTC day, kept on disk so a restart does not forget them."""

    def __init__(self, path: Path | None, clock=time.time):
        self.path, self._clock = (Path(path) if path else None), clock
        self._lock = threading.Lock()
        self._memory: dict[str, int] = {}

    def today(self) -> str:
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc).date().isoformat()

    def _load(self) -> dict[str, int]:
        if self.path is None:
            return dict(self._memory)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): int(v) for k, v in raw.items() if isinstance(v, int)} if isinstance(raw, dict) else {}

    def _save(self, counts: dict[str, int]) -> None:
        keep = dict(sorted(counts.items())[-LEDGER_DAYS:])
        if self.path is None:
            self._memory = keep
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".part")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(keep, handle)
            os.replace(tmp, self.path)
        except OSError:
            pass                                                                   # a ledger that cannot be written must never stop a download

    def count(self) -> int:
        with self._lock:
            return self._load().get(self.today(), 0)

    def add(self, n: int = 1) -> int:
        with self._lock:
            counts = self._load()
            counts[self.today()] = counts.get(self.today(), 0) + n
            self._save(counts)
            return counts[self.today()]


def _budget(value=None) -> int:
    try:
        raw = int(value if value is not None else os.environ.get(BUDGET_VARIABLE, DEFAULT_BUDGET))
    except (TypeError, ValueError):
        raw = DEFAULT_BUDGET
    return max(1, min(raw, 1_000_000))


def _detail(body: bytes) -> str:
    """Finzdata's own words for a refusal: the ``detail`` of its JSON answer (a text, or the messages of a validation error)."""
    try:
        payload = json.loads(body[:20_000].decode("utf-8", "replace"))
    except ValueError:
        return body[:200].decode("utf-8", "replace").strip()
    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    if isinstance(detail, list):
        parts = []
        for item in detail[:3]:
            if isinstance(item, dict):
                where = ".".join(str(p) for p in item.get("loc", []) if p not in ("query", "body"))
                parts.append(f"{where}: {item.get('msg', '')}".strip(": "))
            else:
                parts.append(str(item))
        return "; ".join(parts)
    return " ".join(str(detail).split())


class FinzdataClient:
    """One client for the app: one key, one ledger. ``contracts`` and ``daily`` return the parsed answers; the readers in ``assets.free_data.finzdata`` turn them into tables."""

    def __init__(self, key: str | None = None, root: Path | None = None, base_url: str | None = None, transport=None, sleep=time.sleep, clock=time.time, budget: int | None = None,
                 pause: float = 0.15, timeout: float = 60.0, retries: int = 2):
        self.root = Path(root) if root is not None else None
        self._given = key
        self.base_url = self._check_url(base_url or os.environ.get(URL_VARIABLE) or BASE_URL)
        self.transport = transport or urllib_transport
        self._sleep, self.pause, self.timeout, self.retries = sleep, float(pause), float(timeout), int(retries)
        self.budget = _budget(budget)
        self.ledger = CallLedger(self.root / "data" / "user" / "finzdata_calls.json" if self.root is not None else None, clock)
        self._last = 0.0
        self._lock = threading.Lock()
        self.calls = 0                                                             # the calls this client made (the ledger has the day's)

    @staticmethod
    def _check_url(url: str) -> str:
        parts = parse.urlsplit(url)
        if parts.scheme != "https" and not (parts.scheme == "http" and (parts.hostname or "") in LOOPBACK):
            raise ValueError("FINZDATA_URL must be an https address")
        return url.rstrip("/")

    # ------------------------------------------------------------------------------------------------------------ the key
    def _raw_key(self) -> str:
        return str(self._given) if self._given is not None else read_secret(KEY_VARIABLE, self.root)

    def _key(self) -> str:
        return tidy_key(self._raw_key(), KEY_VARIABLE)[0]

    def key_notes(self) -> list[str]:
        """What is wrong with the key as it was pasted (never the key itself)."""
        text, notes = tidy_key(self._raw_key(), KEY_VARIABLE)
        if text and not text.startswith("fz_"):
            notes.append("Finzdata's keys begin with fz_; this one does not")
        return notes

    def available(self) -> tuple[bool, str]:
        return (True, "") if self._key() else (False, NOT_CONFIGURED)

    def describe(self) -> dict:
        """What the page may know: whether a key exists, notes on it, the host, and the day's calls. Never the key."""
        ok, why = self.available()
        used = self.ledger.count()
        return {"configured": ok, "problem": why, "setup": "" if ok else "key", "key_variable": KEY_VARIABLE, "key_notes": self.key_notes() if ok else [], "host": parse.urlsplit(self.base_url).netloc,
                "calls_today": used, "budget": self.budget, "calls_left": max(self.budget - used, 0), "daily_allowance": DAILY_ALLOWANCE}

    def _clean(self, text: object) -> str:
        return redact(text, self._raw_key(), self._key())

    # ------------------------------------------------------------------------------------------------------------ one call
    def _wait(self) -> None:
        with self._lock:
            gap = self._last + self.pause - time.monotonic()
            self._last = time.monotonic() + max(gap, 0.0)
        if gap > 0:
            self._sleep(gap)

    def get(self, path: str, params: dict | None = None):
        """The parsed JSON answer of a GET. Raises :class:`FinzdataError` for everything else."""
        key = self._key()
        if not key:
            raise FinzdataError(NOT_CONFIGURED, kind="key", fatal=True)
        if self.ledger.count() >= self.budget:
            raise FinzdataError(f"{self.ledger.count():,} Finzdata calls have been made today (UTC), the most this app sends in a day ({self.budget:,}; the free plan allows {DAILY_ALLOWANCE:,}). "
                                f"Try again tomorrow, or raise {BUDGET_VARIABLE}.", kind="budget", fatal=True)
        query = {k: ("true" if v is True else "false" if v is False else v) for k, v in (params or {}).items() if v is not None}
        url = f"{self.base_url}{path}" + (f"?{parse.urlencode(query)}" if query else "")
        headers = {"X-API-Key": key, "Accept": "application/json", "User-Agent": USER_AGENT}
        last = "no answer"
        for attempt in range(self.retries + 1):
            self._wait()
            try:
                status, answer_headers, body = self.transport(url, headers, self.timeout)
            except OSError as exc:
                last = self._clean(f"Finzdata could not be reached ({type(exc).__name__}: {exc})")
                if attempt < self.retries:
                    self._sleep(1.5 * (2 ** attempt))
                continue
            self.calls += 1
            self.ledger.add()
            if status == 200:
                try:
                    return json.loads(body.decode("utf-8"))
                except ValueError:
                    raise FinzdataError("Finzdata answered with something that is not JSON", status=200, kind="server") from None
            said = self._clean(_detail(body))
            if status in (500, 502, 503, 504) and attempt < self.retries:
                self._sleep(1.5 * (2 ** attempt))
                last = f"Finzdata had an error (HTTP {status})"
                continue
            raise self._refusal(status, said, path)
        raise FinzdataError(last, kind="network")

    @staticmethod
    def _refusal(status: int, said: str, path: str) -> FinzdataError:
        words = f": {said}" if said else ""
        if status == 401:
            return FinzdataError(f"Finzdata does not accept the key (HTTP 401{words}). Check the FINZDATA_API_KEY secret, then stop and run the app again.", status=401, kind="key", fatal=True)
        if status == 403:
            return FinzdataError(f"Finzdata's plan for this key does not include this data (HTTP 403{words})", status=403, kind="plan")
        if status == 429:
            return FinzdataError(f"Finzdata's limit of calls for the day is reached (HTTP 429{words}). Try again tomorrow.", status=429, kind="quota", fatal=True)
        if status in (400, 404, 422):
            return FinzdataError(f"Finzdata did not understand the request for {path} (HTTP {status}{words})", status=status, kind="request")
        return FinzdataError(f"Finzdata answered HTTP {status}{words}", status=status, kind="server")

    # ------------------------------------------------------------------------------------------------------------ the endpoints
    def roots(self):
        return self.get("/v1/futures/roots")

    def contracts(self, root: str):
        return self.get("/v1/futures/contracts", {"root": str(root).upper(), "format": "json"})

    def daily(self, root: str, contract: str | None = None, continuous: bool = False, start=None, end=None, limit: int | None = None):
        return self.get("/v1/futures/daily", {"root": str(root).upper(), "contract": contract, "continuous": True if continuous else None, "adjust": "none", "start": _day(start), "end": _day(end),
                                              "format": "json", "limit": limit})

    def me(self):
        """Your plan and usage (one call). The answer is returned as it is, for the page to show the fields it knows."""
        return self.get("/v1/me")


def _day(value) -> str | None:
    if value is None:
        return None
    text = str(value)[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(f"'{value}' is not a date (write YYYY-MM-DD)")
    return text
