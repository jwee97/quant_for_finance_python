"""Information-driven bars (Lopez de Prado, chapter 2).

A bar is a summary of trades between two moments. Time bars sample at a fixed clock, which oversamples quiet periods and undersamples busy ones. These bars close when *something has happened* instead:
a number of trades (tick bars), a quantity traded (volume bars), money changing hands (dollar bars), or a lopsided stretch of buying or selling (imbalance and runs bars).

**They need trades.** Every function takes a frame of individual trades with ``price`` and ``volume`` columns and a time index. A day's open, high, low, close and volume cannot be turned into them: there is no
order of trades inside the day to cut. Nothing here is fed by the daily data the rest of the platform uses, and no result should be described as evidence about a daily strategy.

Bars are *not* independent observations: a bar that closes after a burst of activity is followed by another, and overlapping labels built on them overlap just the same. The bars' output carries ``ticks``,
``volume`` and ``dollar`` per bar so that the sampling can be inspected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check(trades: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    if not {"price", "volume"} <= set(trades.columns):
        raise ValueError("trades need 'price' and 'volume' columns")
    p, v = trades["price"].to_numpy(dtype=float), trades["volume"].to_numpy(dtype=float)
    if len(p) == 0 or not np.isfinite(p).all() or not np.isfinite(v).all() or (p <= 0).any() or (v < 0).any():
        raise ValueError("trades must be non-empty with positive prices and non-negative volumes")
    return p, v, trades.index


def _frame(starts: list[int], p: np.ndarray, v: np.ndarray, idx: pd.Index) -> pd.DataFrame:
    rows, bounds = [], starts + [len(p)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        seg = p[a:b]
        rows.append((idx[a], idx[b - 1], seg[0], seg.max(), seg.min(), seg[-1], b - a, v[a:b].sum(), (seg * v[a:b]).sum()))
    out = pd.DataFrame(rows, columns=["start", "end", "open", "high", "low", "close", "ticks", "volume", "dollar"])
    return out.set_index("end")


def _by_threshold(trades: pd.DataFrame, size: float, measure: str, complete_only: bool) -> pd.DataFrame:
    if size <= 0:
        raise ValueError("size must be positive")
    p, v, idx = _check(trades)
    x = {"tick": np.ones_like(p), "volume": v, "dollar": p * v}[measure]
    starts, acc = [0], 0.0
    for i in range(len(p)):
        acc += x[i]
        if acc >= size:
            starts.append(i + 1)
            acc = 0.0
    if starts[-1] >= len(p):
        starts.pop()
    out = _frame(starts, p, v, idx)
    if complete_only and acc > 0 and len(out):
        out = out.iloc[:-1]                                                                  # the last bar has not closed: it never reached the threshold
    return out


def tick_bars(trades: pd.DataFrame, ticks: int) -> pd.DataFrame:
    """A bar every ``ticks`` trades. The last, unfinished bar is dropped."""
    return _by_threshold(trades, float(ticks), "tick", True)


def volume_bars(trades: pd.DataFrame, volume: float) -> pd.DataFrame:
    """A bar every ``volume`` units traded (a trade is never split across bars, so a bar can exceed it by up to one trade). The last, unfinished bar is dropped."""
    return _by_threshold(trades, float(volume), "volume", True)


def dollar_bars(trades: pd.DataFrame, dollars: float) -> pd.DataFrame:
    """A bar every ``dollars`` of traded value. The last, unfinished bar is dropped."""
    return _by_threshold(trades, float(dollars), "dollar", True)


def tick_rule(price: np.ndarray) -> np.ndarray:
    """+1 for a trade above the previous price, -1 below, and the previous sign when the price is unchanged (+1 for the first)."""
    d = np.diff(price, prepend=price[0])
    b = np.sign(d)
    last = 1.0
    for i in range(len(b)):
        if b[i] == 0:
            b[i] = last
        last = b[i]
    return b


def imbalance_bars(trades: pd.DataFrame, kind: str = "tick", expected_ticks: int = 50, span: int = 10, min_ticks: int = 5, max_ticks: int | None = None, floor: float = 0.1) -> pd.DataFrame:
    """Close a bar when the signed imbalance ``|sum b_t x_t|`` reaches what is expected of a bar: ``E[T] x |E[b_t x_t]|``, with ``x`` 1, the volume or the dollar value of a trade and ``b`` the tick-rule sign.
    ``E[T]`` (ticks per bar) and ``E[b x]`` (signed value per trade) are exponentially weighted averages over the *previous* bars (``span`` bars); the first bar uses the first ``expected_ticks`` trades for them, so
    that one bar's threshold is set from the data it is cut from. Bounds ``min_ticks`` and ``max_ticks`` (default ``10 x expected_ticks``) keep a threshold that has drifted to nothing or to infinity from
    producing one-trade bars or none.

    ``floor`` is a stabilisation that is not in the book: the expected imbalance per trade is never taken below ``floor`` times the average ``x``. Without it, when buying and selling balance, ``|E[b x]|`` goes to zero,
    so does the threshold, and every bar closes at ``min_ticks``; with it, a balanced stretch gives bars of ordinary length and a one-sided stretch gives shorter ones. ``floor = 0`` is the book's rule."""
    return _imbalance_runs(trades, kind, expected_ticks, span, min_ticks, max_ticks, runs=False, floor=floor)


def runs_bars(trades: pd.DataFrame, kind: str = "tick", expected_ticks: int = 50, span: int = 10, min_ticks: int = 5, max_ticks: int | None = None) -> pd.DataFrame:
    """Close a bar when the longer of the buy run and the sell run, ``max(sum of buy x, sum of sell x)``, reaches ``E[T] x max(E[buy x], E[sell x])`` (per-trade expectations over previous bars). Bounds and initial
    values as for ``imbalance_bars``. A bar closes sooner when one side dominates for long."""
    return _imbalance_runs(trades, kind, expected_ticks, span, min_ticks, max_ticks, runs=True, floor=0.0)


def _imbalance_runs(trades: pd.DataFrame, kind: str, expected_ticks: int, span: int, min_ticks: int, max_ticks: int | None, runs: bool, floor: float) -> pd.DataFrame:
    if kind not in ("tick", "volume", "dollar"):
        raise ValueError("kind must be tick, volume or dollar")
    p, v, idx = _check(trades)
    n = len(p)
    if n < 2 * expected_ticks:
        raise ValueError("too few trades for the expected bar size")
    x = {"tick": np.ones(n), "volume": v, "dollar": p * v}[kind]
    b = tick_rule(p)
    buy, sell = np.where(b > 0, x, 0.0), np.where(b < 0, x, 0.0)
    max_ticks = max_ticks or 10 * expected_ticks
    alpha = 2.0 / (span + 1.0)
    e_t = float(expected_ticks)
    e_imb = float(np.mean(b[:expected_ticks] * x[:expected_ticks]))
    e_buy, e_sell = float(buy[:expected_ticks].mean()), float(sell[:expected_ticks].mean())
    e_x = float(x[:expected_ticks].mean())
    starts, a, theta_b, theta_s, theta = [0], 0, 0.0, 0.0, 0.0
    for i in range(n):
        theta += b[i] * x[i]
        theta_b += buy[i]
        theta_s += sell[i]
        t = i - a + 1
        limit = e_t * (max(e_buy, e_sell) if runs else max(abs(e_imb), floor * e_x))
        stat = max(theta_b, theta_s) if runs else abs(theta)
        if (t >= min_ticks and stat >= limit) or t >= max_ticks:
            e_t = (1 - alpha) * e_t + alpha * t
            seg = slice(a, i + 1)
            e_imb = (1 - alpha) * e_imb + alpha * float(np.mean(b[seg] * x[seg]))
            e_x = (1 - alpha) * e_x + alpha * float(np.mean(x[seg]))
            e_buy = (1 - alpha) * e_buy + alpha * float(buy[seg].mean())
            e_sell = (1 - alpha) * e_sell + alpha * float(sell[seg].mean())
            starts.append(i + 1)
            a, theta_b, theta_s, theta = i + 1, 0.0, 0.0, 0.0
    if starts[-1] >= n:
        starts.pop()
    out = _frame(starts, p, v, idx)
    if a < n and len(out):
        out = out.iloc[:-1]                                                                  # the last bar has not closed
    return out


def simulate_trades(n: int = 20_000, seed: int = 0, regimes: bool = True) -> pd.DataFrame:
    """Synthetic trades for tests and demonstrations: a random-walk price, log-normal sizes, and (with ``regimes``) a stretch of heavy one-sided volume in the middle. Not market data."""
    rng = np.random.default_rng(seed)
    drift = np.zeros(n)
    if regimes:
        drift[n // 2:n // 2 + n // 10] = 6e-4
    price = 100 * np.exp(np.cumsum(rng.normal(0, 4e-4, n) + drift))
    vol = np.exp(rng.normal(3.0, 0.8, n))
    if regimes:
        vol[n // 2:n // 2 + n // 10] *= 3.0
    return pd.DataFrame({"price": price, "volume": vol}, index=pd.date_range("2024-01-02 09:30", periods=n, freq="s"))
