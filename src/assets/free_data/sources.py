"""The free public sources, described once, for the page, the status endpoint and the documentation.

None of them needs an account or a key, except iTick and Finzdata (``needs_key``): iTick's key is the ``ITICK_API_KEY`` secret that the Backtest tab's iTick source uses too, Finzdata's the ``FINZDATA_API_KEY`` secret. All of them are somebody else's service: they can change a field, refuse an address or go away, and what they publish is delayed or unaudited. The
dashboard says so beside the data; a download is for your own research on your own machine, and each provider's terms of use are theirs.
"""

from __future__ import annotations

SOURCES: dict[str, dict] = {
    "cboe": {"label": "Cboe delayed quotes", "url": "https://www.cboe.com/delayed_quotes/", "provides": "US stock, ETF and index option chains: bid, ask, volume, open interest",
             "delay": "about 15 minutes during the session; after the close, the last session", "kinds": ["options"]},
    "yahoo": {"label": "Yahoo Finance", "url": "https://finance.yahoo.com/", "provides": "daily prices of each individual futures contract that has not expired (for the curve: as many as it lists, up to ten years out for crude oil) and the price history of any future it lists, found by name (CME Group and ICE US only); option chains when Cboe cannot be reached",
              "delay": "end-of-day settlements; delayed during the session; no open interest; expired contracts are dropped", "kinds": ["futures", "options"]},
    "deribit": {"label": "Deribit", "url": "https://docs.deribit.com/", "provides": "Bitcoin and Ether option chains, dated futures (today's curve, or every monthly and quarterly future since 2019 from its daily charts, expired ones included) and the perpetual",
                "delay": "live", "kinds": ["options", "futures", "crypto"]},
    "cboe_cfe": {"label": "Cboe Futures Exchange", "url": "https://www.cboe.com/us/futures/market_statistics/historical_data/", "provides": "the VIX futures (VX) of every month since 2013, expired ones included: settlement, volume and open interest",
                 "delay": "end of day, published the evening of the trading day or the next morning", "kinds": ["futures"]},
    "coinbase": {"label": "Coinbase Exchange", "url": "https://docs.cdp.coinbase.com/exchange/docs/welcome", "provides": "spot quotes and order books in US dollars; daily candles", "delay": "live", "kinds": ["crypto"]},
    "kraken": {"label": "Kraken", "url": "https://docs.kraken.com/api/", "provides": "spot quotes and order books in US dollars; two years of daily candles", "delay": "live", "kinds": ["crypto"]},
    "bitstamp": {"label": "Bitstamp", "url": "https://www.bitstamp.net/api/", "provides": "spot quotes and order books in US dollars; daily candles", "delay": "live", "kinds": ["crypto"]},
    "itick": {"label": "iTick", "url": "https://docs.itick.org/", "provides": "the daily price history of any futures product (United States, Hong Kong, mainland China) and of coins against USDT, and of stocks and indexes; "
              "no option data, no prices of single futures contracts (so no curves), no open interest", "delay": "end of day; the free plan allows 5 calls a minute", "kinds": ["futures", "crypto", "options"], "needs_key": True},
    "finzdata": {"label": "Finzdata", "url": "https://finzdata.com/docs", "provides": "the daily bars of every individual contract of 139 futures markets since 2007 (expired ones included), with volume and open interest; "
                 "the free plan shows only the last 30 days and allows 1,000 calls a day, so each download is merged into the file of the last one", "delay": "end of day",
                 "kinds": ["futures"], "needs_key": True, "key_variable": "FINZDATA_API_KEY"},
    "okx": {"label": "OKX", "url": "https://www.okx.com/docs-v5/en/", "provides": "spot and perpetual prices, order books, daily candles and funding-rate history", "delay": "live", "kinds": ["crypto"]},
}


def describe(ids) -> list[dict]:
    """The records of the given source ids, in the order given (unknown ids are skipped)."""
    return [{"id": i, **SOURCES[i]} for i in ids if i in SOURCES]
