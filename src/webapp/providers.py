"""iTick as a place to download prices from (Yahoo Finance is the other; see ``universe.yahoo_fetch``).

**The API key.** It lives only in the environment (on Replit: the Secrets tool, which sets the ``ITICK_API_KEY`` variable) or in a git-ignored ``.env`` file. It is read when a download
starts, sent in one request header to iTick and nowhere else, and never put in a response, a log line, an error message or a file: every message that leaves this module goes through
:func:`redact`. The page can see only whether a key exists, never the key.

**The rate limit.** iTick's free plan allows 5 calls a minute. Every call goes through one :class:`RateLimiter` shared by all threads. It remembers the time of recent calls on disk, so
restarting the app does not hand out a fresh allowance, and it backs off for as long as iTick asks after an HTTP 429. A download that has to wait says so in its progress message.

**What is and is not verified.** The request and response formats follow iTick's published documentation (``/stock/kline``: ``region``, ``code``, ``kType``, ``limit``, ``et``). The author
could not call the live API, so the client checks what comes back instead of trusting it: the bars must be about a day apart (the documentation and community code disagree on which
``kType`` is daily; ``ITICK_DAILY_KTYPE`` overrides it), a history that does not reach back as far as asked is reported, and a one-day price jump that looks like an unadjusted stock
split is flagged (iTick does not document whether its bars are adjusted). :meth:`ITickProvider.test` spends one call to show what your account actually returns.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error as urlerror
from urllib import parse, request

import numpy as np
import pandas as pd

KEY_VARIABLE = "ITICK_API_KEY"
DEFAULT_BASE_URL = "https://api-free.itick.org"          # the documented host for free accounts; ``ITICK_BASE_URL`` overrides it
LOOPBACK = ("127.0.0.1", "localhost", "::1")
MAX_BODY_BYTES = 8_000_000
USER_AGENT = "QuantLab/1.0 (daily-price client)"        # some gateways refuse urllib's default agent
NOT_CONFIGURED = "iTick is not set up: add your key as the ITICK_API_KEY secret (on Replit: Tools > Secrets) and restart the app"


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


def redact(text: object, *secrets: str) -> str:
    """``text`` with every secret (and its URL-encoded form) replaced by ``***``."""
    out = str(text)
    for secret in secrets:
        if secret and len(secret) >= 6:
            out = out.replace(secret, "***").replace(parse.quote(secret, safe=""), "***")
    return out


def _int_env(name: str, default: int, low: int, high: int) -> int:
    try:
        return min(max(int(os.environ.get(name, "")), low), high)
    except ValueError:
        return default


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
                    delay, why = self._blocked_until - now, "iTick asked for a pause"
                elif len(self._stamps) < self.calls:
                    self._stamps.append(now)
                    self._save()
                    return waited
                else:
                    delay, why = self._stamps[0] + self.span - now, f"the plan allows {self.calls} call{'s' if self.calls != 1 else ''} a minute"
            if waited + delay > max_wait:
                raise RateLimitTimeout(f"the next iTick call would have to wait {int(waited + delay)} s ({why})")
            if on_wait is not None:
                on_wait(delay, why)
            step = min(max(delay, 0.0), 1.0)
            self._sleep(step)
            waited += step

    def hold(self, seconds: float) -> None:
        """Allow no call for ``seconds`` (iTick answered HTTP 429)."""
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


# ----------------------------------------------------------------------------------------------------------------------------- symbols
STOCK_MARKETS = {                                  # Yahoo suffix -> (iTick region, the exchange's time zone)
    "": ("US", "America/New_York"), "HK": ("HK", "Asia/Hong_Kong"), "SS": ("SH", "Asia/Shanghai"), "SH": ("SH", "Asia/Shanghai"), "SZ": ("SZ", "Asia/Shanghai"),
    "T": ("JP", "Asia/Tokyo"), "SI": ("SG", "Asia/Singapore"), "TW": ("TW", "Asia/Taipei"), "NS": ("IN", "Asia/Kolkata"), "BO": ("IN", "Asia/Kolkata"),
    "BK": ("TH", "Asia/Bangkok"), "DE": ("DE", "Europe/Berlin"), "L": ("GB", "Europe/London"), "MX": ("MX", "America/Mexico_City"), "KL": ("MY", "Asia/Kuala_Lumpur"),
    "IS": ("TR", "Europe/Istanbul"), "MC": ("ES", "Europe/Madrid"), "AS": ("NL", "Europe/Amsterdam"), "JK": ("ID", "Asia/Jakarta"), "VN": ("VN", "Asia/Ho_Chi_Minh"),
}
SYMBOL_HELP = ("US stocks as AAPL, Hong Kong as 0700.HK, Shanghai 600519.SS, Shenzhen 000001.SZ, crypto as BTC-USD, forex as EURUSD=X, indices as ^SPX "
               "(other markets use Yahoo's suffix: .T Tokyo, .L London, .DE Germany, .SI Singapore, .TW Taiwan, .NS India)")


@dataclass(frozen=True)
class Instrument:
    kind: str                                      # stock | crypto | forex | indices
    region: str
    code: str
    tz: str

    @property
    def label(self) -> str:
        return f"{self.region}:{self.code}"


class ITickError(Exception):
    """A problem worth telling the user about; ``fatal`` means no other ticker will fare better (a rejected key, an exhausted plan)."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


def map_symbol(ticker: str) -> Instrument:
    """The iTick instrument for a ticker written the way Yahoo writes it. Raises :class:`ITickError` for a symbol it cannot place."""
    t = ticker.strip().upper()
    if t.startswith("^") and len(t) > 1:
        return Instrument("indices", "GB", t[1:], "UTC")
    if t.endswith("=X") and len(t) > 2:
        return Instrument("forex", "GB", t[:-2], "UTC")
    for quote in ("-USDT", "-USD"):
        if t.endswith(quote) and len(t) > len(quote):
            return Instrument("crypto", "BA", t[: -len(quote)] + "USDT", "UTC")
    suffix = t.rsplit(".", 1)[1] if "." in t else ""
    if suffix in STOCK_MARKETS and suffix:
        region, tz = STOCK_MARKETS[suffix]
        code = t[: -len(suffix) - 1]
        if not code:
            raise ITickError(f"'{ticker}' has no code before the .{suffix} market suffix")
        if region == "HK":
            code = code.lstrip("0") or code                                       # iTick writes Tencent as 700, Yahoo as 0700.HK
        return Instrument("stock", region, code, tz)
    if not t or not t[0].isalnum() or not all(ch.isalnum() or ch in ".-" for ch in t):
        raise ITickError(f"iTick cannot place '{ticker}'. Write symbols as: {SYMBOL_HELP}")
    region, tz = STOCK_MARKETS[""]
    return Instrument("stock", region, t, tz)


# --------------------------------------------------------------------------------------------------------------------------- bar parsing
def session_dates(stamps_ms: np.ndarray, tz: str, weekdays_only: bool) -> pd.DatetimeIndex:
    """The calendar date of each daily bar. iTick does not say when in the day a daily bar is stamped, so the exchange-local date is used unless that puts bars on weekends for
    a market that is closed then, in which case the UTC date is used if it fits better."""
    utc = pd.to_datetime(np.asarray(stamps_ms, dtype="int64"), unit="ms", utc=True)
    local = utc.tz_convert(tz).tz_localize(None).normalize()
    if weekdays_only:
        weekend = int((local.dayofweek >= 5).sum())
        if weekend:
            flat = utc.tz_localize(None).normalize()
            if int((flat.dayofweek >= 5).sum()) < weekend:
                local = flat
    return pd.DatetimeIndex(local)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["close", "high", "low", "volume"], index=pd.DatetimeIndex([], name="date"), dtype="float64")


def bars_to_frame(rows: list, inst: Instrument) -> tuple[pd.DataFrame, int | None]:
    """``(frame, oldest_timestamp_ms)``: close, high, low, volume by date, sorted, one row a day, for the raw ``t, o, h, l, c, v`` rows of a daily kline answer."""
    if not rows:
        return _empty(), None
    df = pd.DataFrame(rows)
    if "t" not in df or "c" not in df:
        raise ITickError("iTick's answer has no 't' (time) and 'c' (close) fields, so its format is not what this client expects")
    t, c = pd.to_numeric(df["t"], errors="coerce"), pd.to_numeric(df["c"], errors="coerce")
    keep = (t.notna() & c.notna() & (c > 0)).to_numpy()
    df, t, c = df[keep], t[keep], c[keep]
    if df.empty:
        return _empty(), None
    order = np.argsort(t.to_numpy(dtype="float64"), kind="stable")
    df, t, c = df.iloc[order], t.iloc[order], c.iloc[order]
    stamps = t.to_numpy(dtype="int64")
    if float(np.median(stamps)) < 1e11:                                             # seconds since 1970 (about 1.7e9 now), not the documented milliseconds (1e11 ms is 1973)
        stamps = stamps * 1000
    if len(stamps) >= 5:
        gap_hours = float(np.median(np.diff(np.unique(stamps))) / 3_600_000.0)
        if not 18.0 <= gap_hours <= 36.0:
            raise ITickError(f"iTick returned bars about {gap_hours:.1f} hours apart, not daily bars. The daily interval code is 8 in iTick's documentation; if your account uses another, "
                             "set ITICK_DAILY_KTYPE to it", fatal=True)
    out = pd.DataFrame({"close": c.to_numpy(dtype="float64")}, index=pd.DatetimeIndex(session_dates(stamps, inst.tz, inst.kind == "stock"), name="date"))
    for source, name in (("h", "high"), ("l", "low"), ("v", "volume")):
        if source in df:
            out[name] = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype="float64")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if inst.kind == "stock":
        out = out[out.index.dayofweek < 5]
    return out, int(stamps[0])


SPLIT_RATIOS = sorted({1 / k for k in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50)} | {2 / 3, 3 / 4, 4 / 5} | {float(k) for k in (2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 50, 100)} | {1.5, 4 / 3})


def suspected_splits(close: pd.Series, tolerance: float = 0.03) -> list[tuple[pd.Timestamp, float]]:
    """Days on which the close moved by (almost exactly) a common split ratio: ``(date, close / previous close)``. A real one-day crash of exactly half is far rarer than a split."""
    ratio = (close / close.shift(1)).dropna()
    out = []
    for date, x in ratio[(ratio < 0.72) | (ratio > 1.4)].items():
        nearest = min(SPLIT_RATIOS, key=lambda s: abs(math.log(x / s)))
        if abs(x / nearest - 1.0) <= tolerance:
            out.append((date, float(x)))
    return out


def split_warnings(close: pd.Series, limit: int = 3) -> list[str]:
    found = suspected_splits(close)
    notes = []
    for date, x in found[:limit]:
        what = f"a {1 / x:.1f}-for-1 split" if x < 1 else f"a 1-for-{x:.1f} reverse split"
        notes.append(f"{date.date()}: the price moved {x:.2f}x in one day, which looks like {what} that iTick did not adjust for. Backtests treat it as a real loss or gain; check the ticker or use Yahoo Finance")
    if len(found) > limit:
        notes.append(f"... and {len(found) - limit} more days like that")
    return notes


# ------------------------------------------------------------------------------------------------------------------------------- HTTP
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


def _problem(payload) -> tuple[str, bool] | None:
    """``(message, fatal)`` when an answer's own status says it failed, else ``None``."""
    if not isinstance(payload, dict):
        return "iTick's answer was not a JSON object", False
    code = payload.get("code")
    if code in (0, "0", 200, "200", None) and "data" in payload:
        return None
    text = f"{code} {payload.get('msg') or ''}".strip()
    lowered = text.lower()
    if code == "E002" or "auth" in lowered:
        return f"iTick rejected the API key ({text}): check the ITICK_API_KEY secret", True
    if code == "E003" or "exceeding" in lowered or "subscription limit" in lowered:
        return f"your iTick plan does not allow this request ({text})", True
    if code == "E001" or "not found" in lowered:
        return f"iTick does not have this symbol ({text})", False
    return f"iTick reported an error: {text}", False


class ITickProvider:
    """Downloads daily bars from iTick, one rate-limited call at a time. Call the instance like ``fetch(tickers, start, known=None, progress=None)``: it returns
    ``{ticker: DataFrame | error string}``, never raising for a single ticker's problem."""

    name, label = "itick", "iTick"
    supports_context = True                           # the store passes ``known`` (cached frames) and ``progress``

    def __init__(self, key: str | None = None, root: Path | None = None, base_url: str | None = None, calls_per_minute: int | None = None, limiter: RateLimiter | None = None,
                 transport=None, timeout: float = 20.0, page_size: int | None = None, max_pages: int | None = None, daily_ktype: int | None = None,
                 max_calls_per_run: int | None = None, sleep=time.sleep, today=None):
        self.root = Path(root) if root else None
        self._key_override = key
        self.base_url = (base_url or os.environ.get("ITICK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.config_error = self._check_url(self.base_url)
        state = (self.root / "data" / "user" / "itick_calls.json") if self.root else None
        self.limiter = limiter or RateLimiter(calls_per_minute or _int_env("ITICK_CALLS_PER_MINUTE", 5, 1, 1200), path=state, sleep=sleep)
        self.page_size = page_size or _int_env("ITICK_PAGE_SIZE", 1000, 10, 5000)
        self.max_pages = max_pages or _int_env("ITICK_MAX_PAGES", 12, 1, 60)
        self.daily_ktype = daily_ktype or _int_env("ITICK_DAILY_KTYPE", 8, 1, 10)
        self.max_calls_per_run = max_calls_per_run or _int_env("ITICK_MAX_CALLS_PER_RUN", 60, 1, 5000)
        self.transport, self.timeout, self._sleep = transport or urllib_transport, timeout, sleep
        self._today = today or (lambda tz: pd.Timestamp.now(tz=tz).normalize().tz_localize(None))

    # ------------------------------------------------------------------------------------------------------------------------ status
    @staticmethod
    def _check_url(url: str) -> str:
        parts = parse.urlparse(url)
        if parts.scheme == "https" and parts.hostname:
            return ""
        if parts.scheme == "http" and parts.hostname in LOOPBACK:
            return ""
        return "ITICK_BASE_URL must start with https:// (the key travels in a request header)"

    def _key(self) -> str:
        return (self._key_override or read_secret(KEY_VARIABLE, self.root)).strip()

    def available(self) -> tuple[bool, str]:
        """``(usable, why not)``. Only whether a key exists is ever reported, never the key."""
        if self.config_error:
            return False, self.config_error
        if not self._key():
            return False, NOT_CONFIGURED
        return True, ""

    def describe(self) -> dict:
        ok, why = self.available()
        return {"configured": ok, "problem": why, "calls_per_minute": self.limiter.calls, "calls_available_now": self.limiter.remaining(),
                "host": parse.urlparse(self.base_url).netloc, "symbols": SYMBOL_HELP}

    def _clean(self, text: object) -> str:
        return redact(text, self._key())

    def preflight(self, ticker: str) -> str | None:
        """Why ``ticker`` could not be downloaded (no key, a symbol iTick cannot place), found without making a call; ``None`` when it looks fine."""
        ok, why = self.available()
        if not ok:
            return why
        try:
            map_symbol(ticker)
        except ITickError as error:
            return str(error)
        return None

    def calls_for(self, ticker: str, start) -> int:
        """About how many calls downloading ``ticker`` from ``start`` takes (one per page of ``page_size`` daily bars)."""
        return self.calls_needed(map_symbol(ticker), start)

    # ---------------------------------------------------------------------------------------------------------------------- the call
    def _get(self, path: str, params: dict, say=None) -> dict:
        key = self._key()
        if not key:
            raise ITickError(NOT_CONFIGURED, fatal=True)
        if self.config_error:
            raise ITickError(self.config_error, fatal=True)
        url = f"{self.base_url}{path}?{parse.urlencode(params)}"

        def waiting(delay: float, why: str) -> None:
            if say:
                say(f"waiting {int(delay) + 1} s for a free call ({why})")

        failure = "iTick did not answer"
        for attempt in range(3):
            self.limiter.acquire(waiting)
            try:
                status, headers, body = self.transport(url, {"accept": "application/json", "token": key, "User-Agent": USER_AGENT}, self.timeout)
            except Exception as error:                                              # no connection, a timeout, a certificate problem
                failure = f"could not reach iTick ({type(error).__name__}); check the network and ITICK_BASE_URL"
                self._sleep(min(2.0 ** attempt, 5.0))
                continue
            if status == 429:
                after = next((v for k, v in headers.items() if k.lower() == "retry-after"), "")
                try:
                    pause = float(after)
                except ValueError:
                    pause = self.limiter.window
                self.limiter.hold(pause)
                failure = "iTick said the call limit was exceeded (HTTP 429)"
                continue
            if status in (401, 403):
                raise ITickError(f"iTick rejected the API key (HTTP {status}): check the ITICK_API_KEY secret", fatal=True)
            if status >= 500:
                failure = f"iTick had a server problem (HTTP {status})"
                self._sleep(min(2.0 ** attempt, 5.0))
                continue
            if status != 200:
                raise ITickError(f"iTick answered HTTP {status} for {path}; check ITICK_BASE_URL and the symbol. {self._clean(body[:160].decode('utf-8', 'replace'))}")
            try:
                payload = json.loads(body)
            except ValueError:
                raise ITickError("iTick's answer was not JSON; check ITICK_BASE_URL") from None
            problem = _problem(payload)
            if problem:
                raise ITickError(self._clean(problem[0]), fatal=problem[1])
            return payload
        raise ITickError(failure, fatal=True)

    def _bars(self, inst: Instrument, limit: int, end_ms: int | None, say=None) -> tuple[pd.DataFrame, int | None]:
        params = {"region": inst.region, "code": inst.code, "kType": self.daily_ktype, "limit": int(limit)}
        if end_ms is not None:
            params["et"] = int(end_ms)
        payload = self._get(f"/{inst.kind}/kline", params, say)
        data = payload.get("data")
        if data is None:
            data = []
        if not isinstance(data, list):
            raise ITickError("iTick's answer had no list of bars under 'data'")
        return bars_to_frame(data, inst)

    # ------------------------------------------------------------------------------------------------------------------------ history
    def _need(self, inst: Instrument, since: pd.Timestamp) -> int:
        today = self._today(inst.tz)
        days = max(int((today - since).days), 1)
        return int(days * (1.0 if inst.kind == "crypto" else 0.71)) + 10                      # bars between ``since`` and today, with a margin

    def calls_needed(self, inst: Instrument, start, known: pd.DataFrame | None = None) -> int:
        """About how many calls a download takes: one per page of ``page_size`` bars from ``start`` (or, when a saved history exists, from just before its last bar)."""
        since = known.index[-1] - pd.Timedelta(days=10) if known is not None and len(known) >= 20 else pd.Timestamp(start)
        return max(1, min(self.max_pages, math.ceil(self._need(inst, since) / self.page_size)))

    def _pages(self, inst: Instrument, since: pd.Timestamp, first_limit: int, max_pages: int, say=None) -> tuple[pd.DataFrame, bool]:
        """Daily bars back to ``since``, newest page first, each page ending just before the oldest bar of the last. ``(frame, stopped_early)``."""
        parts: list[pd.DataFrame] = []
        end_ms: int | None = None
        earliest: pd.Timestamp | None = None
        for page in range(max_pages):
            if say:
                say("downloading" + (f" (page {page + 1}, back to {earliest.date()})" if earliest is not None else ""))
            frame, oldest_ms = self._bars(inst, first_limit if page == 0 else self.page_size, end_ms, say)
            if frame.empty or (earliest is not None and frame.index[0] >= earliest):
                break                                                           # no more history, or the server ignored ``et``
            parts.insert(0, frame)
            earliest = frame.index[0]
            if earliest <= since:
                return self._join(parts), False
            end_ms = (oldest_ms or 0) - 1
        else:
            return self._join(parts), True
        return self._join(parts), False

    @staticmethod
    def _join(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return _empty()
        frame = pd.concat(parts)
        return frame[~frame.index.duplicated(keep="last")].sort_index()

    @staticmethod
    def _merge(known: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame | None:
        """The cached history extended with ``fresh`` bars, or ``None`` when they do not line up (too little overlap, or iTick restated its history)."""
        overlap = known.index.intersection(fresh.index)
        if len(overlap) < 3:
            return None
        if float((fresh.loc[overlap, "close"] / known.loc[overlap, "close"] - 1.0).abs().max()) > 0.002:
            return None
        return pd.concat([known[known.index < fresh.index[0]], fresh])

    def _download(self, ticker: str, inst: Instrument, start, known: pd.DataFrame | None, say=None) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        today = self._today(inst.tz)
        frame = None
        if known is not None and len(known) >= 20:
            since = known.index[-1] - pd.Timedelta(days=10)
            need = self._need(inst, since)
            fresh, _ = self._pages(inst, since, min(max(need, 30), self.page_size), max(1, min(self.max_pages, math.ceil(need / self.page_size))), say)
            frame = self._merge(known, fresh) if not fresh.empty else None
            if frame is None and say:
                say("the saved history does not match what iTick returns now: downloading it again")
        stopped = False
        if frame is None:
            frame, stopped = self._pages(inst, start_ts, min(self._need(inst, start_ts), self.page_size), self.max_pages, say)
        frame = frame[frame.index < today]                                          # today's bar is still being drawn
        if frame.empty:
            raise ITickError(f"iTick returned no daily bars for {inst.label}. {SYMBOL_HELP}")
        notes = []
        if stopped and frame.index[0] > start_ts + pd.Timedelta(days=30):
            notes.append(f"the history stops at {frame.index[0].date()} because the page limit of {self.max_pages} calls a ticker was reached; raise ITICK_MAX_PAGES or choose a later start")
        if inst.kind == "stock":
            notes += split_warnings(frame["close"])
        frame.attrs["warnings"] = notes
        return frame

    # ---------------------------------------------------------------------------------------------------------------------------- main
    def __call__(self, tickers, start, known: dict | None = None, progress=None) -> dict:
        tickers, known = list(tickers), known or {}
        out: dict = {}
        ok, why = self.available()
        if not ok:
            return {t: why for t in tickers}
        plan: list[tuple[str, Instrument]] = []
        for t in tickers:
            try:
                plan.append((t, map_symbol(t)))
            except ITickError as error:
                out[t] = str(error)
        calls = sum(self.calls_needed(inst, start, known.get(t)) for t, inst in plan)
        if calls > self.max_calls_per_run:
            message = (f"this download needs about {calls} iTick calls (around {int(self.limiter.estimate_seconds(calls) / 60) + 1} minutes at {self.limiter.calls} calls a minute), "
                       f"more than the {self.max_calls_per_run} allowed in one run: choose a later start date or add fewer new tickers at a time")
            return {**out, **{t: message for t, _ in plan}}
        for i, (t, inst) in enumerate(plan):
            def say(text: str, _i=i, _t=t) -> None:
                if progress:
                    progress(f"iTick {_t} ({_i + 1} of {len(plan)}): {text}")

            try:
                out[t] = self._download(t, inst, start, known.get(t), say)
            except ITickError as error:
                out[t] = self._clean(error)
                if error.fatal:
                    out.update({rest: out[t] for rest, _ in plan[i + 1:]})
                    break
            except RateLimitTimeout as error:
                out[t] = self._clean(error)
                out.update({rest: out[t] for rest, _ in plan[i + 1:]})
                break
            except Exception as error:                                              # a bug here must reach the page as a message, without ever echoing the key
                out[t] = self._clean(f"unexpected {type(error).__name__}: {str(error)[:160]}")
        return out

    # ----------------------------------------------------------------------------------------------------------------------- self-test
    def test(self, symbol: str = "AAPL") -> dict:
        """Spend one call on the last few daily bars of ``symbol`` and report what came back (never the key): the quickest way to learn whether the key, the host and the daily
        interval code work for your account."""
        started = time.time()
        ok, why = self.available()
        if not ok:
            return {"ok": False, "message": why}
        if self.limiter.remaining() < 1:
            return {"ok": False, "message": f"the {self.limiter.calls} calls a minute your plan allows are in use right now; try again in a minute"}
        try:
            inst = map_symbol(symbol)
            frame, _ = self._bars(inst, 10, None)
        except (ITickError, RateLimitTimeout) as error:
            return {"ok": False, "message": self._clean(error)}
        except Exception as error:
            return {"ok": False, "message": self._clean(f"unexpected {type(error).__name__}: {str(error)[:160]}")}
        if frame.empty:
            return {"ok": False, "message": f"iTick answered but returned no daily bars for {inst.label}"}
        return {"ok": True, "message": f"iTick works: {len(frame)} daily bars for {inst.label}, the latest dated {frame.index[-1].date()} (close {frame['close'].iloc[-1]:g}). "
                                      f"Prices come as delivered; iTick does not say whether they are adjusted for splits and dividends.",
                "bars": int(len(frame)), "first": str(frame.index[0].date()), "last": str(frame.index[-1].date()), "seconds": round(time.time() - started, 1)}
