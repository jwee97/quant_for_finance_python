"""The free public sources, described once, for the page, the status endpoint and the documentation.

None of them needs an account or a key. All of them are somebody else's service: they can change a field, refuse an address or go away, and what they publish is delayed or unaudited. The
dashboard says so beside the data; a download is for your own research on your own machine, and each provider's terms of use are theirs.
"""

from __future__ import annotations

SOURCES: dict[str, dict] = {
    "cboe": {"label": "Cboe delayed quotes", "url": "https://www.cboe.com/delayed_quotes/", "provides": "US stock, ETF and index option chains: bid, ask, volume, open interest",
             "delay": "about 15 minutes during the session; after the close, the last session", "kinds": ["options"]},
    "yahoo": {"label": "Yahoo Finance", "url": "https://finance.yahoo.com/", "provides": "daily prices of each individual futures contract (for the curve); option chains when Cboe cannot be reached",
              "delay": "end-of-day settlements; delayed during the session; no open interest", "kinds": ["futures", "options"]},
    "deribit": {"label": "Deribit", "url": "https://docs.deribit.com/", "provides": "Bitcoin and Ether option chains, dated futures and the perpetual", "delay": "live", "kinds": ["options", "futures", "crypto"]},
    "coinbase": {"label": "Coinbase Exchange", "url": "https://docs.cdp.coinbase.com/exchange/docs/welcome", "provides": "spot quotes and order books in US dollars; daily candles", "delay": "live", "kinds": ["crypto"]},
    "kraken": {"label": "Kraken", "url": "https://docs.kraken.com/api/", "provides": "spot quotes and order books in US dollars; two years of daily candles", "delay": "live", "kinds": ["crypto"]},
    "bitstamp": {"label": "Bitstamp", "url": "https://www.bitstamp.net/api/", "provides": "spot quotes and order books in US dollars; daily candles", "delay": "live", "kinds": ["crypto"]},
    "okx": {"label": "OKX", "url": "https://www.okx.com/docs-v5/en/", "provides": "spot and perpetual prices, order books, daily candles and funding-rate history", "delay": "live", "kinds": ["crypto"]},
}


def describe(ids) -> list[dict]:
    """The records of the given source ids, in the order given (unknown ids are skipped)."""
    return [{"id": i, **SOURCES[i]} for i in ids if i in SOURCES]
