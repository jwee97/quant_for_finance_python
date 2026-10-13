"""What the Futures, Crypto and Options tabs download from iTick, as :class:`~assets.free_data.store.Download` values the store can save.

All of it goes through the one :class:`~webapp.providers.ITickProvider` the Backtest tab uses, so a single rate limiter (the plan's calls a minute, remembered on disk) covers every download in the
app, the key stays in the environment and every message that leaves is redacted. Nothing here opens a connection itself.

* **Futures** (:func:`futures_series`): the daily price series iTick keeps for a futures *product* (``NQ``, ``CL``, ``HSI``), saved as ``<REGION>_<CODE>`` (``US_NQ``). iTick has no per-contract
  histories, so there is no curve, roll or carry in it: the Futures tab reads it in the History view.
* **Crypto** (:func:`coin_history`): a coin's daily closes against USDT, saved as the coin's spot-only history (the same file the exchange downloads write, without perpetual and funding).
* **Options** (:func:`underlying_prices`): iTick has no option data at all. What it can add is the underlying's daily prices, which give a downloaded chain a realised volatility to compare with.

The request and response formats follow iTick's published documentation; the author could not call the live API, so these functions check what comes back, and the notes of each download say what iTick
does not document (how a futures series is joined across contract expiries, whether prices are adjusted).
"""

from __future__ import annotations

import pandas as pd

from ..assets.free_data import itick_catalog as catalog
from ..assets.free_data.store import Download
from .providers import ITickError, ITickProvider, crypto_instrument, future_instrument

MIN_COIN_DAYS = 250                                                         # the Crypto tab needs at least this many daily prices of a coin
INDEX_UNDERLYINGS = {"SPX": "^SPX", "VIX": "^VIX", "DJI": "^DJI", "IXIC": "^IXIC"}   # iTick's names for the index underlyings of the Options tab's presets (it has no NDX or RUT)


def series_name(region: str, code: str) -> str:
    """The file name of a futures price history: ``US_NQ``."""
    return f"{region}_{code}"


def _frame(daily: pd.DataFrame, columns=("close", "high", "low", "volume")) -> pd.DataFrame:
    """``date`` and the columns of a daily frame that exist, as a plain table."""
    out = daily.reset_index()
    return out[["date", *[c for c in columns if c in out]]]


def futures_series(provider: ITickProvider, region: str, code: str, years: float = 5.0, known: pd.DataFrame | None = None, progress=None) -> Download:
    """The daily price history of a futures product from iTick. ``known`` is an earlier download of the same product (``date`` index, ``close`` column): when it reaches back as far as asked, only
    the newest bars are fetched (a call or two instead of a dozen), unless iTick's old prices no longer match it, in which case the whole history is fetched again."""
    inst = future_instrument(region, code)
    today = provider.local_today(inst.tz)
    start = today - pd.Timedelta(days=int(round(years * 365.25)))
    if known is not None and (known.empty or known.index[0] > start + pd.Timedelta(days=45)):
        known = None                                                         # the saved file does not reach back as far as this request wants
    daily = provider.series(inst, start, known, progress)
    product = catalog.find(inst.region, inst.code)
    name = product.name if product else inst.code
    first, last = daily.index[0], daily.index[-1]
    notes = [f"iTick's daily prices of the {name} futures product ({inst.code}, {inst.region} market): {len(daily):,} bars, {first.date()} to {last.date()}. iTick keeps one series per product, "
             "not one per contract month, so there is no forward curve, roll or carry in it. Today's unfinished bar is left out.",
             "iTick does not document how this series is joined when one contract expires and the next becomes the front: expect price jumps on roll days, and treat the series as a price history of "
             "the product, not as the P&L of holding one contract. Whether the prices are adjusted is not documented either; the largest one-day moves are listed in the History view so you can check them.",
             "iTick's daily bar is dated by the exchange's trading day as this app reads it (a bar stamped on the evening before is moved to the next weekday). The series has no open interest."]
    if product is None:
        notes.append(f"{inst.code} is not in the product list bundled with the app, but iTick answered for it.")
    notes += [str(w) for w in daily.attrs.get("warnings") or []]
    frame = _frame(daily)
    meta = {"source_id": "itick", "source": "iTick", "url": f"{provider.host}/future/kline?region={inst.region}&code={inst.code}", "as_of": str(last.date()), "first_date": str(first.date()),
            "region": inst.region, "code": inst.code, "root": inst.code, "root_name": name, "exchange": product.exchange if product else "", "group": product.group if product else "", "bars": int(len(frame)),
            "years": float(years), "notes": notes}
    return Download("futures", series_name(inst.region, inst.code), "series", frame=frame, meta=meta)


def coin_history(provider: ITickProvider, coin: str, days: int = 400, known: pd.DataFrame | None = None, progress=None) -> Download:
    """A coin's daily closes against USDT from iTick, as the spot-only history the Crypto tab reads (``time, spot``)."""
    coin = str(coin).strip().upper()
    if coin not in catalog.coins():
        raise ITickError(f"iTick's list of coins quoted against USDT does not have {coin}. Choose the exchanges as the source, or another coin.")
    if not MIN_COIN_DAYS <= int(days) <= 1500:
        raise ValueError(f"days must be between {MIN_COIN_DAYS} and 1500 (the Crypto tab needs at least {MIN_COIN_DAYS} daily prices)")
    inst = crypto_instrument(coin)
    start = provider.local_today(inst.tz) - pd.Timedelta(days=int(days))
    if known is not None and (known.empty or known.index[0] > start + pd.Timedelta(days=45)):
        known = None
    daily = provider.series(inst, start, known, progress)
    daily = daily.tail(int(days))
    if len(daily) < MIN_COIN_DAYS:
        raise ITickError(f"iTick has only {len(daily)} daily bars of {coin}; the Crypto tab needs at least {MIN_COIN_DAYS}")
    frame = pd.DataFrame({"time": daily.index, "spot": daily["close"].to_numpy(dtype="float64")})
    first, last = daily.index[0], daily.index[-1]
    notes = [f"Daily closes of {inst.code} from iTick (the {inst.region} market, quoted in USDT): {len(frame):,} days, {first.date()} to {last.date()}. The day in progress is left out.",
             "This history is spot only: iTick has no perpetual prices or funding rates, so the basis and funding views have nothing from it (download the exchanges' history for those).",
             "iTick quotes in USDT, not dollars: a few basis points of difference from a dollar price can be the USDT/USD rate. Which exchange's prices iTick's region code stands for is not documented; "
             "compare a few closes with an exchange you trust before relying on them.",
             "The platform's own BTC and ETH series use the 08:00 UTC mark; iTick's daily bar is read as a UTC day, so the two are not on the same clock."]
    notes += [str(w) for w in daily.attrs.get("warnings") or []]
    meta = {"source_id": "itick", "source": "iTick (USDT)", "url": f"{provider.host}/crypto/kline?region={inst.region}&code={inst.code}", "as_of": str(last.date()), "coin": coin,
            "sources": {"spot": "iTick (USDT)"}, "first_date": str(first.date()), "notes": notes}
    return Download("crypto", coin, "history", frame=frame, meta=meta)


def itick_underlying(ticker: str) -> str:
    """The symbol to give iTick for an option underlying: an index preset (SPX) is written ``^SPX``, anything else as it is."""
    t = str(ticker).strip().upper()
    return INDEX_UNDERLYINGS.get(t, t)


def underlying_prices(provider: ITickProvider, ticker: str, years: float = 3.0, progress=None) -> Download:
    """The daily prices of an option underlying from iTick (``date, close, volume``), saved beside its chain as ``<TICKER>_prices`` so the Options tab can measure realised volatility."""
    symbol = itick_underlying(ticker)
    start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=int(round(years * 365.25)))).strftime("%Y-%m-%d")
    answer = provider([symbol], start, progress=progress)[symbol]
    if isinstance(answer, str):
        raise ITickError(answer)
    daily = answer
    frame = _frame(daily, ("close", "volume"))
    first, last = daily.index[0], daily.index[-1]
    notes = [f"iTick's daily prices of {symbol} ({len(frame):,} days, {first.date()} to {last.date()}), used for the realised volatility beside the implied one. Prices come as iTick delivers them: it does not say "
             "whether they are adjusted for splits and dividends."] + [str(w) for w in daily.attrs.get("warnings") or []]
    name = str(ticker).strip().upper()
    meta = {"source_id": "itick", "source": "iTick", "url": f"{provider.host}/stock/kline", "as_of": str(last.date()), "first_date": str(first.date()), "underlying": name, "bars": int(len(frame)), "notes": notes}
    return Download("options", f"{name}_prices", "prices", frame=frame, meta=meta)
