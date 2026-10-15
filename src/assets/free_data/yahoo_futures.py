"""The futures Yahoo Finance lists, found by name so that nobody has to guess a symbol.

Yahoo has no page that lists its futures, but its lookup service (``/v1/finance/lookup?type=future&query=...``) answers a query with up to 100 matching futures a page and about 300 in all: continuous
front-month series (``ES=F``, ``CL=F``, ``6E=F``) and single contracts (``CLZ26.NYM``). It is a search, not a listing (the query ``=F`` does not return every continuous series), so this module works
the way a person would: ``yahoo_futures.csv`` is a bundled snapshot of about 340 continuous series (October 2026), which the finder searches with no call, and :func:`lookup` asks Yahoo about
the words you typed (one to three calls) and keeps what it finds in ``data/user/yahoo_futures.json``, where :func:`catalog` reads it from then on. Everything is a CME Group market (CME, CBOT,
NYMEX, COMEX) or ICE US sugar, coffee, cotton and cocoa; Yahoo has no European, Asian or ICE Europe futures. Yahoo does not say which group a contract belongs to, so :func:`classify` guesses from
the name and the guess is shown, not relied on.

A continuous series is Yahoo's own roll of the front contract, not back-adjusted: :func:`fetch_series` downloads its daily history (one call) for the History view, with the notes that say so.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .http import FetchError, Http
from .store import Download

CSV = Path(__file__).with_name("yahoo_futures.csv")
LOOKUP = "https://query1.finance.yahoo.com/v1/finance/lookup"
CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
PAGE = 100
PAGES = 3                                                                       # Yahoo answers about 300 matches for a query, in pages of 100
SYMBOL = re.compile(r"^(?:[A-Z0-9]{1,9}=F|[A-Z0-9]{1,6}[FGHJKMNQUVXZ]\d{2}\.[A-Z]{2,4})$")   # ES=F (continuous) or CLZ26.NYM (one contract)
YEARS = {1: "1y", 2: "2y", 3: "5y", 5: "5y", 10: "10y", 20: "max"}
GROUPS = ("Equity indexes", "Rates", "Currencies", "Energy", "Metals", "Grains and oilseeds", "Livestock and dairy", "Softs", "Crypto", "Single stocks", "Other")

_RULES = (                                                                      # first match wins; patterns are lower case words or phrases of the contract's name
    ("Single stocks", r"\bstock\b"),
    ("Crypto", r"bitcoin|\bbtc\b|ether\b|\beth\b|solana|\bsol\b|\bxrp\b|avax|cardano|\bada\b|chainlink|\blink futures|\bsui\b|lumens|litecoin|stellar|polkadot|dogecoin|crypto"),
    ("Rates", r"treasury|t-note|t-bond|t-bill|ultra u|yield|\bsofr\b|fed fund|federal funds|eurodollar|\bumbs\b|\bestr\b|interest rate|short-term rate|\bsonia\b|\bswap|mortgage rate"),
    ("Grains and oilseeds", r"\bcorn\b|wheat|soybean|\bsoy\b|\boats?\b|rough rice|canola|sorghum|\bmilo\b|rapeseed|palm oil|\bmeal\b|oilseed|oilshare|\blumber\b|\bpulp\b|\bwood\b|yellow pine|\burea\b"),
    ("Currencies", r"dollar|\byen\b|euro fx|\beuro\b|pound|peso|franc|rand\b|krona|krone|koruna|forint|zloty|\breal\b|rupee|shekel|renminbi|yuan|\bwon\b|\bfx\b|currency|\b(?:usd|eur|gbp|jpy|aud|cad|chf|inr|rmb|cnh)\b|offshore"),
    ("Equity indexes", r"s&p|nasdaq|\bdow\b|russell|nikkei|\bmsci\b|\bvix\b|volatility|equity index|midcap|smallcap|\bnyse\b|\besg\b|bloomberg commodity|\bftse\b|stoxx|\bdax\b|\bbtic\b|\btaco\b|\btmac\b|select sector"),
    ("Energy", r"crude|brent|\boil\b|natural gas|gasoline|rbob|heating oil|\bgasoil\b|diesel|\bulsd\b|ethanol|propane|\bcoal\b|electric|\bpower\b|\bwti\b|naphtha|\bfuel\b|\blng\b|\bngl\b|butane|\bjet\b|henry hub|emission|carbon"),
    ("Metals", r"gold|silver|copper|platinum|palladium|aluminum|aluminium|\bzinc\b|\blead\b|nickel|\btin\b|iron ore|steel|busheling|cobalt|lithium|\bhrc\b|hot-rolled|\bcoil\b|molybdenum|scrap|\bounce\b|uranium"),
    ("Livestock and dairy", r"cattle|\bhogs?\b|\blean\b|feeder|\bbeef\b|\bmilk\b|cheese|butter|\bwhey\b|\bdairy\b|pork|\bbelly\b|\btrim\b"),
    ("Softs", r"sugar|coffee|cocoa|cotton|orange juice|\bjuice\b"),
)
_MONTH_TAIL = re.compile(r"\s*(?:,\s*[A-Z][a-z]{2}-\d{0,4}|\b[A-Za-z]{3}\s+\d{2,4}|\b[A-Z]{3}\s+\d{4}\s+Trading.*|\bFuture\s+[A-Z]{3}\s+\d{4}.*)$")


@dataclass(frozen=True)
class Product:
    symbol: str
    name: str
    exchange: str
    group: str
    kind: str                                                                   # "continuous" (ES=F) or "contract" (CLZ26.NYM)

    @property
    def item(self) -> str:
        return self.symbol

    def view(self) -> dict:
        return {"item": self.symbol, "code": self.symbol, "name": self.name, "exchange": self.exchange, "group": self.group, "kind": self.kind, "region": "YF"}


def clean_name(text: str) -> str:
    """The name of a contract without its delivery month, which Yahoo puts at the end (``Crude Oil Nov 26``, ``Euro FX Futures,Dec-2026``) and truncates to about thirty characters."""
    name = str(text or "").strip()
    for _ in range(2):
        name = _MONTH_TAIL.sub("", name).strip(" ,-")
    return name or str(text or "").strip()


def classify(symbol: str, name: str) -> str:
    """A guess at the group of a contract from its name: ``Energy``, ``Metals``, ``Rates`` ... or ``Other`` (the symbol alone says nothing: ``BCH`` is scrap steel at one exchange)."""
    text = str(name).lower()
    for group, pattern in _RULES:
        if re.search(pattern, text):
            return group
    return "Other"


def kind_of(symbol: str) -> str:
    return "continuous" if symbol.endswith("=F") else "contract"


def parse_lookup(payload) -> list[Product]:
    """The futures in one answer of Yahoo's lookup service (anything that is not a future, or has a symbol this app does not read, is skipped)."""
    result = (((payload or {}).get("finance") or {}).get("result") or [{}])[0] if isinstance(payload, dict) else {}
    out: dict[str, Product] = {}
    for doc in (result or {}).get("documents") or []:
        symbol = str(doc.get("symbol") or "").upper()
        if str(doc.get("quoteType") or "").lower() != "future" or not SYMBOL.match(symbol):
            continue
        name = clean_name(doc.get("shortName") or doc.get("longName") or symbol)
        out[symbol] = Product(symbol, name, str(doc.get("exchange") or ""), classify(symbol, name), kind_of(symbol))
    return list(out.values())


# ------------------------------------------------------------------------------------------------------------------ the list
@lru_cache(maxsize=1)
def bundled() -> tuple[Product, ...]:
    with CSV.open(encoding="utf-8", newline="") as handle:
        return tuple(Product(r["symbol"], r["name"], r["exchange"], r["group"], kind_of(r["symbol"])) for r in csv.DictReader(handle))


def _read_cache(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and isinstance(data.get("products"), list) else {}


def catalog(path: Path | None = None) -> list[Product]:
    """Every future the app knows: the bundled continuous series, with the refreshed list (when there is one) added to it and winning where both have a symbol."""
    merged = {p.symbol: p for p in bundled()}
    for row in _read_cache(path).get("products", []):
        try:
            merged[row["symbol"]] = Product(row["symbol"], row["name"], row.get("exchange", ""), row.get("group", "Other"), kind_of(row["symbol"]))
        except (KeyError, TypeError):
            continue
    return list(merged.values())


def find(symbol: str, path: Path | None = None) -> Product | None:
    symbol = str(symbol).upper()
    return next((p for p in catalog(path) if p.symbol == symbol), None)


def _score(p: Product, word: str) -> int | None:
    sym, name = p.symbol.lower(), p.name.lower()
    base = sym[:-2] if sym.endswith("=f") else sym
    if word in (sym, base):
        return 0
    if base.startswith(word):
        return 1
    if any(w.startswith(word) for w in re.split(r"[^a-z0-9]+", name)):
        return 2
    if word in name or word in sym:
        return 3
    return None


def search(query: str = "", group: str | None = None, kind: str | None = None, limit: int = 30, path: Path | None = None) -> dict:
    """Futures whose symbol or name contains every word of ``query``, best match first (continuous series before single contracts); with no query, the continuous series of the best-known roots first.
    ``{results, total, shown}``."""
    words = [w for w in re.split(r"\s+", str(query or "").strip().lower()) if w]
    from .futures import FREE_ROOTS                                             # the roots this app builds curves for come first: they are the ones most people look for

    known = {r: i for i, r in enumerate(FREE_ROOTS)}
    ranked = []
    for p in catalog(path):
        if group and p.group != group:
            continue
        if kind and p.kind != kind:
            continue
        scores = [_score(p, w) for w in words]
        if words and any(s is None for s in scores):
            continue
        base = p.symbol[:-2] if p.symbol.endswith("=F") else p.symbol
        ranked.append((max(scores) if words else 0, 0 if p.kind == "continuous" else 1, known.get(base, 999), p.exchange, p.symbol, p))
    ranked.sort(key=lambda t: t[:5])
    limit = max(1, min(int(limit), 200))
    return {"results": [t[5].view() for t in ranked[:limit]], "total": len(ranked), "shown": min(limit, len(ranked))}


def groups(path: Path | None = None) -> list[str]:
    present = {p.group for p in catalog(path)}
    return [g for g in GROUPS if g in present]


# ------------------------------------------------------------------------------------------------------------------ asking Yahoo
def lookup(http: Http, query: str, path: Path | None = None, pages: int = PAGES) -> dict:
    """Ask Yahoo's lookup service for the futures matching ``query`` (a name or a symbol: ``lumber``, ``nikkei``, ``6E``) and remember what it found in ``path`` so that :func:`catalog` has it from now
    on. Returns ``{results, found, new, pages}``; the results are in Yahoo's order, continuous series first. A page that fails ends the asking and keeps what was read; with nothing read it raises."""
    query = str(query or "").strip()
    if not 1 <= len(query) <= 40:
        raise ValueError("ask Yahoo for a word or two: a name (lumber, nikkei) or a symbol (6E, ES)")
    found: dict[str, Product] = {}
    read, failure = 0, None
    for page in range(max(1, int(pages))):
        try:
            payload = http.get_json(LOOKUP, {"query": query, "type": "future", "start": page * PAGE, "count": PAGE, "formatted": "false", "fetchPricingData": "false"})
        except FetchError as error:
            failure = error
            break
        read += 1
        rows = parse_lookup(payload)
        for p in rows:
            found.setdefault(p.symbol, p)
        documents = (((payload or {}).get("finance") or {}).get("result") or [{}])[0].get("documents") or [] if isinstance(payload, dict) else []
        if len(documents) < PAGE:
            break
    if read == 0:
        raise failure
    before = {p.symbol for p in catalog(path)}
    new = [p for p in found.values() if p.symbol not in before]
    if path is not None and found:
        _remember(Path(path), found.values())
    ordered = sorted(found.values(), key=lambda p: (0 if p.kind == "continuous" else 1, p.exchange, p.symbol))
    return {"results": [p.view() for p in ordered], "found": len(found), "new": len(new), "pages": read, "incomplete": failure is not None}


def _remember(path: Path, products) -> None:
    """Add products to the file of what Yahoo has been asked about (symbols, names, exchanges; nothing secret)."""
    have = {row["symbol"]: row for row in _read_cache(path).get("products", []) if isinstance(row, dict) and "symbol" in row}
    for p in products:
        have[p.symbol] = {"symbol": p.symbol, "name": p.name, "exchange": p.exchange, "group": p.group}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": "Yahoo Finance lookup", "products": sorted(have.values(), key=lambda r: r["symbol"])}), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------------------------ a price history
def series_name(symbol: str) -> str:
    """The file name of a futures price history from Yahoo: ``YF_ES`` for ``ES=F``, ``YF_CLZ26_NYM`` for ``CLZ26.NYM``."""
    return "YF_" + re.sub(r"[^A-Z0-9]+", "_", symbol.upper().replace("=F", "")).strip("_")


def yahoo_series(payload: dict) -> pd.DataFrame:
    """``date, close, high, low, volume`` from Yahoo's chart answer: the exchange's calendar day of each bar."""
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict) or chart.get("error") or not chart.get("result"):
        detail = ((chart or {}).get("error") or {}).get("description", "no result")
        raise FetchError(f"Yahoo Finance has no history for this symbol ({detail})", status=404)
    result = chart["result"][0]
    stamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    close = quote.get("close") or []
    if not stamps or len(close) != len(stamps):
        raise FetchError("Yahoo Finance's answer has no prices")
    offset = int((result.get("meta") or {}).get("gmtoffset") or 0)
    dates = [datetime.fromtimestamp(t + offset, tz=timezone.utc).date() for t in stamps]

    def column(name: str) -> pd.Series:
        values = quote.get(name)
        return pd.to_numeric(pd.Series(values if values is not None and len(values) == len(stamps) else [np.nan] * len(stamps)), errors="coerce")

    frame = pd.DataFrame({"date": pd.to_datetime(dates), "close": column("close"), "high": column("high"), "low": column("low"), "volume": column("volume")})
    frame = frame.dropna(subset=["close"])
    frame = frame[frame["close"] > 0].drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


def fetch_series(http: Http, symbol: str, years: float = 5.0, path: Path | None = None) -> Download:
    """The daily price history of one Yahoo future (``ES=F``: the front month rolled by Yahoo; ``CLZ26.NYM``: one contract) as a series the History view reads. One call."""
    symbol = str(symbol).strip().upper()
    if not SYMBOL.match(symbol):
        raise ValueError(f"{symbol} is not a futures symbol Yahoo uses: ES=F (the front month) or CLZ26.NYM (one contract)")
    span = YEARS[min((k for k in YEARS if k >= years), default=20)]
    frame = yahoo_series(http.get_json(CHART.format(symbol=symbol), {"range": span, "interval": "1d"}))
    if len(frame) < 30:
        raise FetchError(f"Yahoo Finance has only {len(frame)} daily prices of {symbol}; at least 30 are needed")
    product = find(symbol, path)
    name = product.name if product else symbol
    first, last = frame["date"].iloc[0], frame["date"].iloc[-1]
    continuous = symbol.endswith("=F")
    notes = [f"Yahoo Finance's daily prices of {name} ({symbol}): {len(frame):,} bars, {first.date()} to {last.date()}; the last bar can be a session in progress, not a settlement.",
             ("Yahoo's =F series is the front-month contract, rolled by Yahoo on a schedule it does not publish and not back-adjusted: expect price jumps on roll days (the History view lists the largest moves), and treat "
              "it as a price history of the product, not as the P&L of holding one contract. There is no curve, roll or carry in it.") if continuous else
             "This is one contract: it stops when the contract expires, and Yahoo stops serving it some time after. There is no curve, roll or carry in a single contract's history.",
             "Yahoo has no open interest for these series."]
    meta = {"source_id": "yahoo", "source": "Yahoo Finance", "url": CHART.format(symbol=symbol), "as_of": str(last.date()), "first_date": str(first.date()), "region": "YF", "code": symbol, "root": symbol,
            "root_name": name, "exchange": product.exchange if product else "", "group": product.group if product else "", "bars": int(len(frame)), "years": float(years), "notes": notes}
    return Download("futures", series_name(symbol), "series", frame=frame[["date", "close", "high", "low", "volume"]], meta=meta)
