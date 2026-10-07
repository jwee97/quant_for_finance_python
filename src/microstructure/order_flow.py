"""Order flow and liquidity measures from trades and quotes.

Direction of trade: ``tick_rule`` (Hasbrouck's tick test) and ``lee_ready`` (Lee & Ready 1991: compare the trade with the quote midpoint, fall back to the tick rule at the midpoint).

Order-flow imbalance (Cont, Kukanov & Stoikov 2014): from changes in the best bid and ask and their sizes, ``ofi`` gives the net pressure of each event. The mid-price change over an interval is
approximately linear in the interval's OFI divided by market depth (``ofi_price_impact``), which is the cleanest empirical regularity in limit-order-book data.

Impact and spread estimators:
* ``kyle_lambda``: slope of the price change on signed volume (Kyle 1985): price impact per unit traded.
* ``amihud``: mean of ``|return| / dollar volume`` (Amihud 2002): daily illiquidity from daily data.
* ``roll_spread``: ``2 sqrt(-cov(dp_t, dp_{t-1}))`` (Roll 1984): the bid-ask bounce implies negative serial covariance of price changes.
* ``corwin_schultz``: the high-low spread estimator (Corwin & Schultz 2012) from two consecutive daily ranges.
* ``effective_spread`` / ``realized_spread``: twice the distance of the trade from the midpoint, and the part of it the liquidity provider keeps after prices move ``horizon`` trades later (the rest is adverse selection).
* ``vpin``: volume-synchronised probability of informed trading (Easley, Lopez de Prado & O'Hara 2012) with bulk volume classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..stats.regression import ols


def tick_rule(prices: pd.Series) -> pd.Series:
    """+1 (buyer-initiated) if the price rose from the previous trade, -1 if it fell, and the previous sign if unchanged (the first trade is undefined)."""
    d = np.sign(prices.diff())
    return d.replace(0.0, np.nan).ffill()


def lee_ready(trade_price: pd.Series, bid: pd.Series, ask: pd.Series) -> pd.Series:
    """Quote rule: above the midpoint = buy, below = sell; trades at the midpoint use the tick rule. ``bid`` and ``ask`` must be the quotes prevailing JUST BEFORE each trade."""
    mid = 0.5 * (bid + ask)
    diff = trade_price - mid
    quote = np.sign(diff).where(diff.abs() > 1e-9 * mid)                          # within rounding of the midpoint counts as AT the midpoint
    return quote.fillna(tick_rule(trade_price))


def ofi(bid_price: pd.Series, bid_size: pd.Series, ask_price: pd.Series, ask_size: pd.Series) -> pd.Series:
    """Event-by-event order-flow imbalance at the best quotes: ``e_n = 1[Pb_n >= Pb_{n-1}] qb_n - 1[Pb_n <= Pb_{n-1}] qb_{n-1} - 1[Pa_n <= Pa_{n-1}] qa_n + 1[Pa_n >= Pa_{n-1}] qa_{n-1}``."""
    pb, pa = bid_price.to_numpy(), ask_price.to_numpy()
    qb, qa = bid_size.to_numpy(), ask_size.to_numpy()
    e = np.zeros(len(pb))
    e[1:] = ((pb[1:] >= pb[:-1]) * qb[1:] - (pb[1:] <= pb[:-1]) * qb[:-1] - (pa[1:] <= pa[:-1]) * qa[1:] + (pa[1:] >= pa[:-1]) * qa[:-1])
    return pd.Series(e, index=bid_price.index, name="ofi")


def ofi_price_impact(ofi_events: pd.Series, mid: pd.Series, interval: int = 100, depth: pd.Series | None = None) -> dict:
    """Aggregate OFI and the mid-price change over blocks of ``interval`` events and regress ``dmid = a + b OFI/depth`` (``depth`` defaults to the average best-quote size of the block).
    Returns the slope ``b`` (price impact per unit of normalised flow), its t statistic and the R-squared."""
    n = (len(ofi_events) // interval) * interval
    o = ofi_events.iloc[:n].to_numpy().reshape(-1, interval).sum(axis=1)
    m = mid.iloc[:n].to_numpy().reshape(-1, interval)
    dm = m[:, -1] - m[:, 0]
    d = np.ones_like(o, dtype=float) if depth is None else depth.iloc[:n].to_numpy().reshape(-1, interval).mean(axis=1)
    x = pd.DataFrame({"ofi_over_depth": o / d})
    res = ols(pd.Series(dm), x, "HC1")
    return {"beta": float(res.params["ofi_over_depth"]), "t": float(res.tvalues["ofi_over_depth"]), "r2": float(res.r2), "n_blocks": int(len(o))}


def kyle_lambda(price_change: pd.Series, signed_volume: pd.Series) -> dict:
    """``dp = a + lambda * signed volume + e``: impact per unit of net buying. Returns lambda, its HC1 t statistic and R-squared."""
    res = ols(price_change, signed_volume.rename("sv").to_frame(), "HC1")
    return {"lambda": float(res.params["sv"]), "t": float(res.tvalues["sv"]), "r2": float(res.r2), "nobs": res.nobs}


def amihud(returns: pd.Series, dollar_volume: pd.Series, window: int | None = None) -> pd.Series | float:
    """``|r| / dollar volume`` averaged over the sample (a float) or a rolling ``window`` (a series). Higher = less liquid."""
    ratio = (returns.abs() / dollar_volume.replace(0.0, np.nan))
    return float(ratio.mean()) if window is None else ratio.rolling(window, min_periods=window // 2).mean()


def roll_spread(prices: pd.Series) -> float:
    """Roll (1984): ``S = 2 sqrt(-cov(dp_t, dp_{t-1}))``; 0 if the covariance is not negative (no detectable bounce)."""
    dp = prices.diff().dropna()
    c = float(np.cov(dp.iloc[1:].to_numpy(), dp.iloc[:-1].to_numpy())[0, 1])
    return 2.0 * np.sqrt(-c) if c < 0 else 0.0


def corwin_schultz(high: pd.Series, low: pd.Series) -> pd.Series:
    """The high-low spread estimator from consecutive days: ``S = 2 (e^alpha - 1) / (1 + e^alpha)`` with
    ``alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2))``; negative estimates are set to zero."""
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)
    gamma = np.log(high.rolling(2).max() / low.rolling(2).min()) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return s.clip(lower=0.0)


def effective_spread(trade_price: pd.Series, mid: pd.Series, sign: pd.Series, relative: bool = True) -> pd.Series:
    """``2 x sign x (trade - mid)``, in price units or relative to the midpoint."""
    es = 2.0 * sign * (trade_price - mid)
    return es / mid if relative else es


def realized_spread(trade_price: pd.Series, mid: pd.Series, sign: pd.Series, horizon: int = 5) -> dict:
    """The liquidity provider's revenue after ``horizon`` trades: ``2 sign (trade - mid_{t+h})``. The gap between effective and realized spread is the adverse-selection cost (price impact)."""
    future_mid = mid.shift(-horizon)
    ok = future_mid.notna() & sign.notna()                                          # the same trades enter all three averages
    eff = (2.0 * sign * (trade_price - mid))[ok]
    real = (2.0 * sign * (trade_price - future_mid))[ok]
    return {"effective": float(eff.mean()), "realized": float(real.mean()), "adverse_selection": float((eff - real).mean())}


def vpin(price: pd.Series, volume: pd.Series, bucket_volume: float | None = None, n_buckets: int = 50) -> pd.Series:
    """VPIN with bulk volume classification: split each bar's volume into buy and sell by ``V Phi(dp / sigma)``, fill equal-volume buckets, and report the rolling mean over
    ``n_buckets`` buckets of ``|V_buy - V_sell| / V_bucket``. Values near 1 mean one-sided (toxic) flow. The result is indexed by the time each bucket completes."""
    dp = price.diff().fillna(0.0)
    sigma = dp.std()
    buy = volume * norm.cdf(dp / sigma if sigma > 0 else dp)
    bucket = bucket_volume or float(volume.sum() / max(len(volume) // 5, 1))
    imbalance, times = [], []
    cum_buy = cum_vol = 0.0
    for t, v, b in zip(volume.index, volume.to_numpy(), buy.to_numpy()):
        remaining, buy_left = float(v), float(b)
        while remaining > 1e-12:
            take = min(remaining, bucket - cum_vol)
            share = take / remaining
            cum_buy += buy_left * share
            cum_vol += take
            buy_left *= 1.0 - share
            remaining -= take
            if cum_vol >= bucket - 1e-12:
                imbalance.append(abs(2.0 * cum_buy - cum_vol) / bucket)
                times.append(t)
                cum_buy = cum_vol = 0.0
    s = pd.Series(imbalance, index=pd.DatetimeIndex(times) if len(times) and isinstance(volume.index, pd.DatetimeIndex) else times, name="bucket_imbalance")
    return s.rolling(n_buckets, min_periods=n_buckets).mean().rename("vpin")
