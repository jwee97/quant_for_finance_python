"""A zero-intelligence limit order book: Poisson arrivals of limit orders, cancellations and market orders on a price grid (Cont, Stoikov & Talreja 2010; Smith, Farmer et al. 2003).

State: ``book[p]`` on integer price ticks, positive = resting buy orders at ``p``, negative = resting sell orders. Best bid = highest positive tick, best ask = lowest negative tick.
Events (continuous time, Gillespie algorithm, so inter-arrival times are exponential):

* a buy limit order arrives ``k`` ticks below the best ask (``k = 1 .. K``) at rate ``lambda_k = lam0 * k^-alpha`` (more orders near the touch) and a sell limit order ``k`` ticks above the best bid likewise;
* each resting order is cancelled at rate ``theta``;
* a market buy (sell) arrives at rate ``mu`` and consumes one order at the best ask (bid).

No participant has information or intent, yet the model reproduces the main shape of real books: a bounded spread, mean-reverting depth, mid-price moves driven by the imbalance between the two
sides, and the linear relation between order-flow imbalance and price change. It is a testbed for execution and market-making code, not a model of any market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_lob(n_events: int = 50000, K: int = 5, lam0: float = 1.0, alpha: float = 0.6, theta: float = 0.05, mu: float = 0.5, initial_depth: int = 8, grid: int = 400, seed: int = 0) -> pd.DataFrame:
    """Simulate ``n_events`` events and return one row per event with the time, event type, trade sign and price, and the best bid/ask and their sizes AFTER the event."""
    rng = np.random.default_rng(seed)
    book = np.zeros(grid, dtype=np.int64)
    mid0 = grid // 2
    book[mid0 - 3: mid0] = initial_depth
    book[mid0 + 1: mid0 + 4] = -initial_depth
    lam_k = lam0 * np.arange(1, K + 1, dtype=float) ** (-alpha)
    t = 0.0
    rows = []
    best_bid, best_ask = mid0 - 1, mid0 + 1
    for _ in range(n_events):
        n_orders = np.abs(book).sum()
        total_limit = 2.0 * lam_k.sum()
        rate_cancel = theta * n_orders
        total = total_limit + rate_cancel + 2.0 * mu
        t += rng.exponential(1.0 / total)
        u = rng.random() * total
        kind, sign, price = "", 0, np.nan
        if u < total_limit:                                                        # limit order
            side_buy = u < lam_k.sum()
            idx = int(np.searchsorted(np.cumsum(lam_k), u if side_buy else u - lam_k.sum()))
            k = min(idx + 1, K)
            if side_buy:
                p = best_ask - k
                if 0 < p < grid:
                    book[p] += 1
                kind = "limit_buy"
            else:
                p = best_bid + k
                if 0 < p < grid - 1:
                    book[p] -= 1
                kind = "limit_sell"
        elif u < total_limit + rate_cancel:                                        # cancellation: choose an order uniformly
            pick = rng.integers(n_orders)
            levels = np.flatnonzero(book)
            sizes = np.abs(book[levels])
            j = int(np.searchsorted(np.cumsum(sizes), pick, side="right"))
            lvl = levels[j]
            book[lvl] += -1 if book[lvl] > 0 else 1
            kind = "cancel"
        else:                                                                       # market order
            buy = (u - total_limit - rate_cancel) < mu
            if buy and book[best_ask] < 0:
                price = best_ask
                book[best_ask] += 1
                kind, sign = "market_buy", 1
            elif not buy and book[best_bid] > 0:
                price = best_bid
                book[best_bid] -= 1
                kind, sign = "market_sell", -1
            else:
                kind = "none"
        bids, asks = np.flatnonzero(book > 0), np.flatnonzero(book < 0)
        if len(bids) == 0:                                                          # an emptied side is replenished one tick away from the other side's touch
            book[asks.min() - 1] = initial_depth
            bids = np.flatnonzero(book > 0)
        if len(asks) == 0:
            book[bids.max() + 1] = -initial_depth
            asks = np.flatnonzero(book < 0)
        best_bid, best_ask = int(bids.max()), int(asks.min())
        if best_ask <= best_bid:                                                    # never cross: remove the offending orders (cannot happen with the entry rules; defensive)
            book[best_ask] = 0
            asks = np.flatnonzero(book < 0)
            best_ask = int(asks.min())
        if best_bid < 40 or best_ask > grid - 40:                                   # recentre the book to stay away from the grid edges
            shift = mid0 - (best_bid + best_ask) // 2
            book = np.roll(book, shift)
            best_bid, best_ask = best_bid + shift, best_ask + shift
        rows.append((t, kind, sign, price - 0 if np.isnan(price) else price, best_bid, int(book[best_bid]), best_ask, int(-book[best_ask])))
    df = pd.DataFrame(rows, columns=["time", "event", "sign", "trade_price", "bid", "bid_size", "ask", "ask_size"])
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread"] = df["ask"] - df["bid"]
    return df
