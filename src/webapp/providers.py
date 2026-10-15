"""iTick as a place to download prices from (Yahoo Finance is the other; see ``universe.yahoo_fetch``).

**The API key.** It lives only in the environment (on Replit: the Secrets tool, which sets the ``ITICK_API_KEY`` variable) or in a git-ignored ``.env`` file. It is read when a download
starts, sent in one request header to iTick and nowhere else, and never put in a response, a log line, an error message or a file: every message that leaves this module goes through
:func:`redact`. The page can see only whether a key exists, never the key.

**The rate limit.** iTick's free plan allows 5 calls a minute. Every call goes through one :class:`RateLimiter` shared by all threads. It remembers the time of recent calls on disk, so
restarting the app does not hand out a fresh allowance, and it backs off for as long as iTick asks after an HTTP 429. A download that has to wait says so in its progress message.

**Hosts.** iTick documents a free-trial host (``api-free.itick.org``, the default here) and a production host (``api.itick.org``). A key that one refuses (HTTP 401, ``Invalid API key in request``)
is tried once on the other, and the host that knows it is used from then on; ``ITICK_BASE_URL`` chooses an address and switches that off. An HTTP 403 is reported as a plan or address problem, not as a
wrong key. Every refusal is shown in iTick's own words.

**What is and is not verified.** The request and response formats follow iTick's published documentation (``/stock/kline``: ``region``, ``code``, ``kType``, ``limit``, ``et``). The author
could not call the live API with a working key (only its refusals of a request without one were seen), so the client checks what comes back instead of trusting it: the bars must be about a day apart (the documentation and community code disagree on which
``kType`` is daily; ``ITICK_DAILY_KTYPE`` overrides it), a history that does not reach back as far as asked is reported, and a one-day price jump that looks like an unadjusted stock
split is flagged (iTick does not document whether its bars are adjusted). :meth:`ITickProvider.test` spends one call to show what your account actually returns.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error as urlerror
from urllib import parse, request

import numpy as np
import pandas as pd

KEY_VARIABLE = "ITICK_API_KEY"
FREE_HOST = "https://api-free.itick.org"                  # "Free trial api url" in iTick's documentation
PAID_HOST = "https://api.itick.org"                       # "Production api url", and the host its own examples use
DEFAULT_BASE_URL = FREE_HOST                              # ``ITICK_BASE_URL`` overrides it; with it set, the other host is never tried
LOOPBACK = ("127.0.0.1", "localhost", "::1")
TEST_SYMBOLS = {"futures": "ES=F", "crypto": "BTC-USD"}  # what the connection test asks for on a tab whose data a plan may not include
CONTROL_SYMBOLS = {"futures": "NQ=F", "crypto": "ETH-USD"}   # asked next when that product is not found: iTick's own documentation uses NQ (and BTCUSDT) as its examples
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


_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00a0"))       # what a copy and paste can carry along that nobody can see
_PLACEHOLDER = re.compile(r"(?i)^<?\s*(?:your|my)[ _-]*(?:itick[ _-]*)?(?:api[ _-]*)?(?:key|token)\s*>?$|^(?:key|token|xxx+|\*+)$")


def tidy_key(raw: object, name: str = KEY_VARIABLE) -> tuple[str, list[str]]:
    """``(key, notes)``: the key without the wrapping a paste often adds (quotes, spaces, a line break, invisible characters, ``ITICK_API_KEY=`` or ``Bearer`` in front of it), and a note for
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
SYMBOL_HELP = ("US stocks as AAPL, Hong Kong as 0700.HK, Shanghai 600519.SS, Shenzhen 000001.SZ, crypto as BTC-USD, forex as EURUSD=X, indices as ^SPX, "
               "US futures as ES=F (the price series of the product, not of one contract) "
               "(other markets use Yahoo's suffix: .T Tokyo, .L London, .DE Germany, .SI Singapore, .TW Taiwan, .NS India)")


FUTURE_REGIONS = {"US": "America/New_York", "HK": "Asia/Hong_Kong", "CN": "Asia/Shanghai"}      # iTick's futures regions and the exchange time zone of each


@dataclass(frozen=True)
class Instrument:
    kind: str                                      # stock | crypto | forex | indices | future
    region: str
    code: str
    tz: str

    @property
    def label(self) -> str:
        return f"{self.region}:{self.code}"


class ITickError(Exception):
    """A problem worth telling the user about; ``fatal`` means no other ticker will fare better (a rejected key, an exhausted plan)."""

    def __init__(self, message: str, fatal: bool = False, kind: str = ""):
        super().__init__(message)
        self.fatal = fatal
        self.kind = kind                                                         # "auth", "plan" or "symbol" when iTick said so; "" for everything else


def map_symbol(ticker: str) -> Instrument:
    """The iTick instrument for a ticker written the way Yahoo writes it. Raises :class:`ITickError` for a symbol it cannot place."""
    t = ticker.strip().upper()
    if t.startswith("^") and len(t) > 1:
        return Instrument("indices", "GB", t[1:], "UTC")
    if t.endswith("=X") and len(t) > 2:
        return Instrument("forex", "GB", t[:-2], "UTC")
    if t.endswith("=F") and len(t) > 2:
        return future_instrument("US", t[:-2])                                   # Yahoo's front-month series of a US future: CL=F, ES=F
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


def future_instrument(region: str, code: str) -> Instrument:
    """The iTick instrument of a futures product: ``region`` is US, HK or CN and ``code`` the product (``NQ``, ``CL``, ``HSI``), not a delivery month. Raises :class:`ITickError` for anything else."""
    region, code = str(region or "").strip().upper(), str(code or "").strip().upper()
    if region not in FUTURE_REGIONS:
        raise ITickError(f"iTick lists futures in the regions {', '.join(FUTURE_REGIONS)}, not '{region}'")
    if not code or not code[0].isalnum() or not all(ch.isalnum() or ch == "." for ch in code) or len(code) > 12:
        raise ITickError(f"'{code}' is not a futures product code (letters and digits, such as NQ or HSI)")
    return Instrument("future", region, code, FUTURE_REGIONS[region])


def crypto_instrument(coin: str) -> Instrument:
    """The iTick instrument of a coin's market against the US-dollar stablecoin USDT (``BTCUSDT``)."""
    coin = str(coin or "").strip().upper()
    if not coin or not coin.isalnum() or len(coin) > 12:
        raise ITickError(f"'{coin}' is not a coin symbol (letters and digits, such as BTC)")
    return Instrument("crypto", "BA", coin + "USDT", "UTC")


# --------------------------------------------------------------------------------------------------------------------------- bar parsing
def session_dates(stamps_ms: np.ndarray, tz: str, weekdays_only: bool, trade_day: bool = False) -> pd.DatetimeIndex:
    """The calendar date of each daily bar. iTick does not say when in the day a daily bar is stamped, so the exchange-local date is used unless that puts bars on weekends for
    a market that is closed then, in which case the UTC date is used if it fits better. ``trade_day`` is for futures, whose trading day starts the evening before (Sunday 18:00 in
    New York for Monday): a date six hours after the stamp is then a third candidate, and the one with the fewest weekend dates wins (the first of the three on a tie)."""
    utc = pd.to_datetime(np.asarray(stamps_ms, dtype="int64"), unit="ms", utc=True)
    local_time = utc.tz_convert(tz).tz_localize(None)
    candidates = [local_time.normalize()]
    if weekdays_only:
        if trade_day:
            candidates.append((local_time + pd.Timedelta(hours=6)).normalize())
        candidates.append(utc.tz_localize(None).normalize())
    best = min(candidates, key=lambda days: int((days.dayofweek >= 5).sum())) if weekdays_only else candidates[0]
    return pd.DatetimeIndex(best)


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
    weekdays = inst.kind in ("stock", "future")
    out = pd.DataFrame({"close": c.to_numpy(dtype="float64")}, index=pd.DatetimeIndex(session_dates(stamps, inst.tz, weekdays, trade_day=inst.kind == "future"), name="date"))
    for source, name in (("h", "high"), ("l", "low"), ("v", "volume")):
        if source in df:
            out[name] = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype="float64")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if weekdays:
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


def _problem(payload) -> tuple[str, bool, str] | None:
    """``(message, fatal, kind)`` when an answer's own status says it failed, else ``None``; ``kind`` is ``auth`` (the key), ``plan`` (the account's allowance), ``symbol`` or ``other``."""
    if not isinstance(payload, dict):
        return "iTick's answer was not a JSON object", False, "other"
    code = payload.get("code")
    if code in (0, "0", 200, "200", None) and "data" in payload:
        return None
    text = f"{code} {payload.get('msg') or ''}".strip()
    lowered = text.lower()
    if code == "E002" or "auth" in lowered:
        return text, True, "auth"
    if code == "E003" or "exceeding" in lowered or "subscription limit" in lowered:
        return f"your iTick plan does not allow this request ({text})", True, "plan"
    if code == "E001" or "not found" in lowered:
        return text, False, "symbol"                                            # the caller adds what it means (see ``ITickProvider._not_found``)
    return f"iTick reported an error: {text}", False, "other"


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
        explicit = base_url or os.environ.get("ITICK_BASE_URL")
        self.base_url = (explicit or DEFAULT_BASE_URL).rstrip("/")
        self.auto_host = not explicit                                         # no address was chosen: iTick's two documented hosts are tried in turn when a key is not recognised
        self._learned: tuple[str, str] | None = None                          # (host, key) of the last host that accepted the key (kept in memory only)
        self._access: dict[str, dict] = {}                                    # "future:US:ES" -> {"ok", "at", "why"}: what iTick returned for this key lately (product codes and dates, never the key)
        self._access_path = (Path(root) / "data" / "user" / "itick_access.json") if root else None
        self.config_error = self._check_url(self.base_url)
        self._load_access()
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

    ACCESS_DAYS = 30

    def _load_access(self) -> None:
        if self._access_path is None:
            return
        try:
            data = json.loads(self._access_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            self._access = {str(k): v for k, v in data.items() if isinstance(v, dict) and isinstance(v.get("ok"), bool) and isinstance(v.get("at"), str)}

    def note_access(self, kind: str, item: str, ok: bool, why: str = "") -> None:
        """Remember whether iTick returned a product for this key, so that the finder can mark it (``kind`` is iTick's own: ``future``, ``crypto``...; ``item`` like ``US:ES``)."""
        self._access[f"{kind}:{item}"] = {"ok": bool(ok), "at": pd.Timestamp.now().date().isoformat(), "why": self._clean(why)[:160]}
        if self._access_path is None:
            return
        try:
            self._access_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._access_path.with_suffix(".part")
            tmp.write_text(json.dumps(self._access), encoding="utf-8")
            tmp.replace(self._access_path)
        except OSError:
            pass

    def access(self, kind: str) -> dict[str, dict]:
        """``{item: {"ok", "at", "why"}}`` for the products of this kind that iTick returned or refused in the last ``ACCESS_DAYS`` days."""
        cutoff = (pd.Timestamp.now() - pd.Timedelta(days=self.ACCESS_DAYS)).date().isoformat()
        prefix = f"{kind}:"
        return {k[len(prefix):]: dict(v) for k, v in self._access.items() if k.startswith(prefix) and v["at"] >= cutoff}

    def _raw_key(self) -> str:
        return self._key_override or read_secret(KEY_VARIABLE, self.root)

    def _key(self) -> str:
        return tidy_key(self._raw_key())[0]

    def key_notes(self) -> list[str]:
        """What was removed from, or looks wrong with, the secret as it was pasted (never the key itself)."""
        return tidy_key(self._raw_key())[1]

    @property
    def host(self) -> str:
        """The address calls go to: the one chosen with ``ITICK_BASE_URL``, else the host that last accepted this key, else the free host."""
        if self.auto_host and self._learned and self._learned[1] == self._key():
            return self._learned[0]
        return self.base_url

    @staticmethod
    def other_host(host: str) -> str | None:
        """The other of iTick's two documented hosts, or ``None`` for any other address."""
        return {FREE_HOST: PAID_HOST, PAID_HOST: FREE_HOST}.get(host.rstrip("/"))

    def available(self) -> tuple[bool, str]:
        """``(usable, why not)``. Only whether a key exists is ever reported, never the key."""
        if self.config_error:
            return False, self.config_error
        if not self._key():
            return False, NOT_CONFIGURED
        return True, ""

    def describe(self) -> dict:
        ok, why = self.available()
        setup = "url" if self.config_error else ("" if self._key() else "key")                # what is missing: the address is wrong, or there is no key yet
        return {"configured": ok, "problem": why, "setup": setup, "calls_per_minute": self.limiter.calls, "calls_available_now": self.limiter.remaining(),
                "host": parse.urlparse(self.host).netloc, "symbols": SYMBOL_HELP, "page_size": self.page_size}

    def _clean(self, text: object) -> str:
        return redact(text, self._key(), self._raw_key().strip())

    def local_today(self, tz: str) -> pd.Timestamp:
        """Today's date in the exchange's time zone (a day's bar is complete only after it)."""
        return self._today(tz)

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
    def _reason(self, body: bytes) -> str:
        """What iTick itself said about a refusal, on one line and without the key. Its gateway answers like ``{"message": "Invalid API key in request"}``."""
        text = body[:600].decode("utf-8", "replace").strip()
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict):
            for field in ("message", "msg", "error_msg", "error", "detail"):
                if isinstance(data.get(field), str) and data[field].strip():
                    text = data[field]
                    break
        return self._clean(" ".join(str(text).split()))[:160]

    def _refused(self, tried: list[tuple[str, int, str]], path: str) -> str:
        """The message for a key iTick would not accept: what each host said, and the usual causes. ``tried`` holds ``(host, HTTP status, iTick's words)`` for every host asked."""
        said = "; ".join(f'{parse.urlparse(host).netloc} said "{reason}" (HTTP {status})' if reason else f"{parse.urlparse(host).netloc} answered HTTP {status}" for host, status, reason in tried)
        if any(status == 403 for _, status, _ in tried):
            kind = {"future": "futures"}.get(path.strip("/").split("/")[0], path.strip("/").split("/")[0] or "this")
            return (f"iTick refused this request: {said}. HTTP 403 usually means the key was recognised but is not allowed this data: your plan may not include {kind} prices "
                    "(iTick's pricing page lists stocks, forex, indices and crypto), or the key may be limited to certain IP addresses (a Repl's address changes). The key itself may be fine.")
        unchosen = " ITICK_BASE_URL is set, so iTick's other host was not tried." if not self.auto_host and self.other_host(tried[0][0]) else ""
        return (f"iTick rejected the API key: {said}.{unchosen} Copy the key again from the iTick dashboard into the ITICK_API_KEY secret with no quotes or spaces, and stop and run the app "
                "again (a changed secret is read only when the app starts). A key from a time-limited free trial stops working when the trial ends.")

    @staticmethod
    def _not_found(text: str, path: str) -> str:
        """iTick's "not found this produce" (E001), with what it means: its documentation says the code is wrong or the product has not been added to the plan."""
        kind = path.strip("/").split("/")[0]
        more = " iTick's pricing page does not list futures among the free plan's features, so futures may need a paid plan." if kind == "future" else ""
        check = " Test the iTick connection on the Futures or Crypto card also tries a product iTick uses in its own examples, which tells a plan without that data from one missing product." if kind in ("future", "crypto") else ""
        return f"iTick does not have this symbol for your key ({text}). iTick says that means the product code is wrong or the product has not been added to your plan (its customer service adds products).{more}{check}"

    def _get(self, path: str, params: dict, say=None) -> dict:
        key = self._key()
        if not key:
            raise ITickError(NOT_CONFIGURED, fatal=True)
        if self.config_error:
            raise ITickError(self.config_error, fatal=True)
        query = parse.urlencode(params)

        def waiting(delay: float, why: str) -> None:
            if say:
                say(f"waiting {int(delay) + 1} s for a free call ({why})")

        host, tried, attempt = self.host, [], 0                                     # tried: (host, status, iTick's words) of every host that did not accept the key
        failure = "iTick did not answer"
        while attempt < 3:
            self.limiter.acquire(waiting)
            try:
                status, headers, body = self.transport(f"{host}{path}?{query}", {"accept": "application/json", "token": key, "User-Agent": USER_AGENT}, self.timeout)
            except Exception as error:                                              # no connection, a timeout, a certificate problem
                failure = f"could not reach iTick ({type(error).__name__}); check the network and ITICK_BASE_URL"
                self._sleep(min(2.0 ** attempt, 5.0))
                attempt += 1
                continue
            if status == 429:
                after = next((v for k, v in headers.items() if k.lower() == "retry-after"), "")
                try:
                    pause = float(after)
                except ValueError:
                    pause = self.limiter.window
                self.limiter.hold(pause)
                failure = "iTick said the call limit was exceeded (HTTP 429)"
                attempt += 1
                continue
            if status in (401, 403):
                tried.append((host, status, self._reason(body)))
                other = self.other_host(host) if status == 401 and self.auto_host and len(tried) == 1 else None
                if other is None:
                    raise ITickError(self._refused(tried, path), fatal=True)
                if say:
                    say(f"{parse.urlparse(host).netloc} did not recognise the key; trying {parse.urlparse(other).netloc}")
                host = other                                                         # a second host, not another attempt
                continue
            if status >= 500:
                failure = f"iTick had a server problem (HTTP {status})"
                self._sleep(min(2.0 ** attempt, 5.0))
                attempt += 1
                continue
            if status != 200:
                raise ITickError(f"iTick answered HTTP {status} for {path}; check ITICK_BASE_URL and the symbol. {self._clean(body[:160].decode('utf-8', 'replace'))}")
            try:
                payload = json.loads(body)
            except ValueError:
                raise ITickError("iTick's answer was not JSON; check ITICK_BASE_URL") from None
            problem = _problem(payload)
            if problem and problem[2] == "auth":                                    # the application's own refusal ("E002 auth failed") is handled like the gateway's
                tried.append((host, status, self._clean(problem[0])))
                other = self.other_host(host) if self.auto_host and len(tried) == 1 else None
                if other is None:
                    raise ITickError(self._refused(tried, path), fatal=True)
                if say:
                    say(f"{parse.urlparse(host).netloc} did not recognise the key; trying {parse.urlparse(other).netloc}")
                host = other
                continue
            if problem:
                text = self._not_found(problem[0], path) if problem[2] == "symbol" else problem[0]
                raise ITickError(self._clean(text), fatal=problem[1], kind=problem[2])
            if self.auto_host and host != self.host:
                self._learned = (host, key)                                         # the other host knows this key: use it from now on
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

    def series(self, inst: Instrument, start, known: pd.DataFrame | None = None, progress=None) -> pd.DataFrame:
        """The daily bars of one instrument from ``start`` (extending ``known``, an earlier result, when given): ``close``, ``high``, ``low`` and ``volume`` by date, today's unfinished bar left out.
        Raises :class:`ITickError` (``fatal`` when no other instrument would fare better) or :class:`RateLimitTimeout`; no message contains the key."""
        ok, why = self.available()
        if not ok:
            raise ITickError(why, fatal=True)

        def say(text: str) -> None:
            if progress:
                progress(f"iTick {inst.label}: {text}")

        item = f"{inst.region}:{inst.code}"
        try:
            frame = self._download(inst.label, inst, start, known, say)
        except ITickError as error:
            if error.kind == "symbol":
                self.note_access(inst.kind, item, False, str(error).split(". ", 1)[0])
            raise ITickError(self._clean(error), fatal=error.fatal, kind=error.kind) from None
        except RateLimitTimeout as error:
            raise RateLimitTimeout(self._clean(error)) from None
        except Exception as error:                                                  # a bug here must reach the page as a message, without ever echoing the key
            raise ITickError(self._clean(f"unexpected {type(error).__name__}: {str(error)[:160]}")) from None
        self.note_access(inst.kind, item, True)
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
    def test(self, symbol: str = "AAPL", kind: str | None = None) -> dict:
        """Spend one call on the last few daily bars of ``symbol`` (two when the first host does not recognise the key) and report what came back, in words that say what to do next, and never the key:
        the quickest way to learn whether the key, the host, the plan and the daily interval code work for your account. ``kind`` (``futures`` or ``crypto``) tests that tab's own data instead of a stock,
        because a plan can include one and not the other. When that product is not found, one more call asks for a product iTick uses in its own documentation, which tells a plan that lacks the data
        from a product that is merely missing."""
        started = time.time()
        notes = self.key_notes()
        ok, why = self.available()
        if not ok:
            return {"ok": False, "message": why, "notes": notes}
        if self.limiter.remaining() < 1:
            return {"ok": False, "message": f"the {self.limiter.calls} calls a minute your plan allows are in use right now; try again in a minute", "notes": notes}
        what = TEST_SYMBOLS.get(kind or "", symbol)
        asked = f" ({kind} data)" if kind in TEST_SYMBOLS else ""
        fixed = f" Also: {'; '.join(notes)}." if notes else ""
        try:
            inst = map_symbol(what)
            frame = self._tested(inst)
        except (ITickError, RateLimitTimeout) as error:
            out = {"ok": False, "message": self._clean(error) + fixed, "notes": notes, "host": parse.urlparse(self.host).netloc}
            if getattr(error, "kind", "") == "symbol" and kind in CONTROL_SYMBOLS:
                out.update(self._control(inst, kind, error, fixed))
            return out
        except Exception as error:
            return {"ok": False, "message": self._clean(f"unexpected {type(error).__name__}: {str(error)[:160]}") + fixed, "notes": notes}
        host = parse.urlparse(self.host).netloc
        if frame.empty:
            return {"ok": False, "message": f"iTick ({host}) answered but returned no daily bars for {inst.label}{asked}", "notes": notes, "host": host}
        moved = f" Your key was not recognised at {parse.urlparse(self.base_url).netloc}, so the app now uses {host}." if self.auto_host and self.host != self.base_url else ""
        return {"ok": True, "message": f"iTick works at {host}{asked}: {len(frame)} daily bars for {inst.label}, the latest dated {frame.index[-1].date()} (close {frame['close'].iloc[-1]:g}).{moved} "
                                      f"Prices come as delivered; iTick does not say whether they are adjusted for splits and dividends.{fixed}",
                "bars": int(len(frame)), "first": str(frame.index[0].date()), "last": str(frame.index[-1].date()), "seconds": round(time.time() - started, 1), "host": host, "notes": notes}

    def _tested(self, inst: Instrument) -> pd.DataFrame:
        """The last ten daily bars of ``inst``, remembering whether iTick had it (for the finder's marks)."""
        item = f"{inst.region}:{inst.code}"
        try:
            frame, _ = self._bars(inst, 10, None)
        except ITickError as error:
            if error.kind == "symbol":
                self.note_access(inst.kind, item, False, str(error).split(". ", 1)[0])
            raise
        if not frame.empty:
            self.note_access(inst.kind, item, True)
        return frame

    def _control(self, inst: Instrument, kind: str, error: ITickError, fixed: str) -> dict:
        """The product was not found: ask for the one iTick's own documentation uses and say what the two answers together mean."""
        control = map_symbol(CONTROL_SYMBOLS[kind])
        if control.label == inst.label:
            return {}
        try:
            frame = self._tested(control)
        except (ITickError, RateLimitTimeout) as second:
            if getattr(second, "kind", "") == "symbol":
                return {"message": f"iTick returns neither {inst.label} nor {control.label}, the product its own documentation uses as an example: your plan probably does not include {kind} data "
                                   f"(iTick's pricing page lists it separately from the free plan's features). Yahoo Finance has futures without a key: choose it as the source, or ask iTick's customer service what your plan "
                                   f"includes.{fixed}", "control": {"item": control.label, "ok": False}}
            return {}
        if frame.empty:
            return {}
        return {"message": f"iTick has no {inst.label} for your key, but it does return {control.label}, the product its own documentation uses as an example. So the key and the plan do work for "
                           f"{kind} data: {inst.label} has probably not been added to your plan. Choose another product (the search marks the ones that worked), or ask iTick's customer service to add it.{fixed}",
                "control": {"item": control.label, "ok": True}}
