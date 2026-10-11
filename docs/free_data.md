# Free data for the Options, Futures and Crypto tabs

The repository holds no option quotes, no futures contract prices and no multi-exchange crypto quotes, so those three tabs start on a model and say so. Each of them has a **Real data (free)** card in its left panel that downloads the real thing from public sources that need no account and no key, saves it on your machine and switches the tab to it. You do not find, clean or format anything.

## How to use it

1. Open the Options, Futures or Crypto tab. The card is open while nothing has been downloaded.
2. Tick what you want (tickers, futures roots, coins) and press **Download real data**. A line under the button counts the seconds and a hint beside it estimates the time (ten tickers or roots take a minute or two), because the providers are asked politely, one request after another.
3. The result lists each item with a tick or a cross and the reason for the cross. The tab selects the first download, the badge above every table reads **Downloaded: free data** (or **Mixed sources** when part of the page is still a model, and the parts are listed beside it), and the "How this was made" note names the provider, the time of the download and what the data is not.
4. **Update** repeats a download (the same request, with fresh prices); **Delete** removes it (it asks twice). A futures download is merged with the earlier one, so the contracts that have expired since keep their history.

| Tab | Offers | From | Saved as |
|---|---|---|---|
| Options | US stock, ETF and index chains (SPY, QQQ, AAPL, SPX, any ticker with listed options; up to 15 at a time), and Bitcoin and Ether chains | [Cboe delayed quotes](https://www.cboe.com/delayed_quotes/) (about 15 minutes late); [Yahoo Finance](https://finance.yahoo.com/) when Cboe cannot be reached and the `yfinance` package is installed; [Deribit](https://docs.deribit.com/) (live) for BTC and ETH | `data/user/downloads/options/<TICKER>.csv`; the Underlying menu gains "SPY (real chain, …)" |
| Futures | The curve of CL, NG, GC, SI, HG, ES, NQ, ZN, ZB, 6E, ZC, ZS and ZW (2 to 16 contracts, 6 months to 5 years of history), and Bitcoin and Ether futures | Yahoo Finance, one daily series per contract (settlements; delayed in the session); Deribit (live, with open interest) | `data/user/downloads/futures/<ROOT>.csv`; the Root menu marks "(downloaded)" |
| Crypto | A coin's daily history (spot, perpetual, funding) for BTC, ETH, SOL, XRP, BNB, DOGE, ADA, AVAX, LINK, LTC or any coin the exchanges list; the real quotes and order books of the coin on six venues; OKX's funding history | OKX (spot, perpetual, funding), Coinbase Exchange, Kraken, Bitstamp and Deribit (live); spot alone from Kraken, Bitstamp or Coinbase when OKX does not list the coin | `data/user/downloads/crypto/` (`<COIN>.csv`, `venues_<COIN>.json`, `funding_<COIN>_OKX.csv`) |

Every download is a file you can open and a `.json` record beside it (`source`, the address, `fetched_at`, `as_of`, the number of rows, the notes, and the request that repeats it). Nothing is written anywhere else: `data/user/options/`, `data/user/futures/` and `data/user/crypto/` are for your own files, which always win over a download of the same name.

## What the tabs do with it

- **Options.** The downloaded chain is analysed exactly like a model chain: smile, skew, term structure, surface, Greeks and risk, the strategy builder priced off real quotes. Because a vendor chain gives prices but not the rate and the dividend yield they were struck with, the page assumes the risk-free rate (4% a year, stored with the download) and reads **each expiry's dividend yield from put-call parity at the money**: the forward the market itself implies. That keeps the implied volatilities and the parity checks consistent. Deribit publishes each expiry's forward, which is read directly. The arbitrage view checks only quotes that have a live bid and lie within 5% of the forward for stock and ETF options (15% for the European-style index, Bitcoin and Ether chains): a delayed, wide quote far out of the money, or an in-the-money American option that is worth more than parity says, would otherwise be reported as a violation. Stock and ETF options can be exercised early and are priced as European, which understates the value of that right a little.
- **Futures.** Yahoo's contract histories are stacked into a curve by expiry date. The expiry dates are computed from each exchange's rule (the third-last business day for gold, the third Friday for the equity indexes, and so on) without a holiday calendar, so one can be a day off. The feed has **no open interest**, so the open-interest chart and crossover are left out and say why; margins stay the illustrative figures of the model. Deribit's futures carry open interest and volume.
- **Crypto.** A downloaded coin gets its own history, funding and the real venues: the overview, funding, basis and liquidity views read them and say which numbers are data. The cross-exchange view compares venues only within one kind of market (spot with spot, perpetual with perpetual), on real bids and asks at the moment of the download, net of entry-level taker fees that are **assumptions** (they depend on your volume and change). Liquidity walks the real order book (up to 400 levels a side, within 5% of the mid) for the cost of $100 thousand, $1 million and $10 million; a size the saved book cannot fill is shown as n/a, not extrapolated. OKX keeps about three months of funding history whatever you ask. Bitcoin and Ether keep the platform's longer Deribit history; the card adds their venues and funding.

## What this data is not

- **Delayed or unaudited, and somebody else's service.** Cboe's quotes are 15 minutes late, Yahoo's last bar can be a session still in progress, and a provider can change a field, refuse an address or disappear. Use the data for research on your own machine; each provider's terms of use are theirs. It is not a feed to trade on, and a backtest on it is still a backtest.
- **Not a history of implied volatility.** A chain is one day. The overview's volatility rank needs a year of implied volatility the downloads do not have, so it says "n/a" instead of borrowing the model's.
- **Not the venues you may expect.** Binance and Bybit refuse US and many cloud addresses (HTTP 451 and 403) and CME's site forbids scripts, so none of them is used.
- **Shared cloud hosts get refused.** Yahoo and the exchanges rate-limit addresses that many people share, and Replit's is one of them. When a provider answers "HTTP 403" or "HTTP 429", the card says so by name and the app waits a while before asking that host again instead of hammering it. Try again in a few minutes, download fewer items at a time, or run the dashboard on your own computer.

## Where the code is

`src/assets/free_data/` holds one module per tab (`options.py`, `futures.py`, `crypto.py`), the HTTP client (`http.py`: paced per host, retried with a back-off, a cool-down after a refusal, readable errors), the store (`store.py`) and the list of sources (`sources.py`). The web side is `src/webapp/data_api.py`:

- `GET /api/data/status`: what is downloaded and with which provenance, what each tab can ask for, and which downloads are running.
- `POST /api/data/fetch` (a job; poll `/api/job/<id>`): `{"kind": "options" | "futures" | "crypto", "items": [...], ...settings}`. The browser names a ticker, a root or a coin; every address and file name is fixed on the server, so a request cannot reach another host or another file. One download per tab at a time, on a worker of its own, so a slow feed never delays a backtest.
- `POST /api/data/delete`: `{"kind", "name", "confirm": true}`.

The tests replace the network with a table of answers in the shape of the live feeds (`tests/free_data_helpers.py`), so none of them goes to the internet. See also [the dashboard guide](dashboard.md), and the guides to [option chains](techniques/option-chains-and-greeks.md), [futures curves](techniques/futures-curves-carry-and-rolls.md) and [crypto funding and venues](techniques/crypto-funding-basis-and-venues.md).

After an update, stop the app and press Run again (on Replit, Stop and then Run): the Free data card needs the app's API version 5, and an older app says so instead of leaving the card empty.
