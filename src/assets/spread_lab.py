"""A research bench for futures spreads: calendar, intercommodity, cross-market and user-defined baskets.

Everything here treats a statistical relationship between prices as a **hypothesis to be tested**, never as an arbitrage: a spread can stay wide, a cointegrating relation estimated on the past can break, and
a mean-reversion rule that works in one sample can fail in the next. So each tool reports the evidence against the hypothesis next to the evidence for it.

* The legs are the **excess-return indices** of rolling the front contract of each root (``futures.excess_return_series``): the price of holding and rolling, which includes the roll yield and is free of the
  jumps at expiry. The spread of two legs is ``ln I_A - beta ln I_B``.
* ``hedge_ratio`` estimates ``beta`` by ordinary least squares on log levels (the cointegrating regression) and by total least squares, and reports how it moves through time (rolling windows): a hedge ratio
  that is not stable is a finding.
* ``diagnostics``: Augmented Dickey-Fuller on the spread (the null is a unit root: a high p-value means no evidence of stationarity), the Engle-Granger cointegration test (MacKinnon p-values), the half-life
  of mean reversion from an AR(1) fit, rolling correlation, spread volatility, seasonality by calendar month and a simple two-state regime split on volatility.
* ``backtest``: walk-forward. At each date the hedge ratio and the z-score's mean and standard deviation are estimated on the ``lookback`` days **before** it; the position is taken at the next day's return
  (no look-ahead); entries and exits are on z-score thresholds; costs are the roots' calendar and outright costs per traded notional leg. The result carries bootstrap intervals and a realism label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..derivatives import validation as val
from .futures import excess_return_series


def leg_index(data, roll_days: int = 5) -> pd.Series:
    """The log excess-return index of rolling the front contract of a root: the cumulative sum of ``ln(1 + r)``."""
    r = excess_return_series(data.long[["date", "contract", "expiry", "price"]], roll_days)
    return np.log1p(r).cumsum().rename(data.root)


def build_spread(legs: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """``sum_i w_i * log index_i`` over the dates every leg has: a user-defined basket (weights may be negative)."""
    unknown = set(weights) - set(legs)
    if unknown:
        raise ValueError(f"no series for {sorted(unknown)}")
    frame = pd.concat([legs[k] for k in weights], axis=1, keys=list(weights)).dropna()
    if len(frame) < 60:
        raise ValueError(f"only {len(frame)} dates are common to the legs: too few to study")
    return sum(frame[k] * w for k, w in weights.items()).rename("spread")


def hedge_ratio(a: pd.Series, b: pd.Series, window: int = 252) -> dict:
    """The ratio ``beta`` in ``ln I_A = alpha + beta ln I_B + e``: by ordinary least squares and by total least squares (the first principal axis, symmetric in the two legs), and the OLS ratio over rolling windows."""
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(df) < max(60, window // 2):
        raise ValueError("too few common dates to estimate a hedge ratio")
    x, y = df["b"].to_numpy(), df["a"].to_numpy()
    ols = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    xc, yc = x - x.mean(), y - y.mean()
    _, _, vt = np.linalg.svd(np.c_[xc, yc], full_matrices=False)
    tls = float(vt[0, 1] / vt[0, 0])
    roll = [float(np.cov(x[i - window:i], y[i - window:i], ddof=1)[0, 1] / np.var(x[i - window:i], ddof=1)) for i in range(window, len(x) + 1, max(1, window // 12))]
    return {"ols": ols, "tls": tls, "rolling_ols": roll, "rolling_min": float(min(roll)) if roll else None, "rolling_max": float(max(roll)) if roll else None,
            "stable": bool(roll and (max(roll) - min(roll)) < 0.5 * abs(ols)), "n": int(len(df)),
            "reading": "the ratio moves by more than half of its own size across windows: it is not a constant" if roll and (max(roll) - min(roll)) >= 0.5 * abs(ols) else "the ratio is reasonably steady across windows"}


def half_life(spread: pd.Series) -> float | None:
    """Days for a deviation to halve, from ``dS_t = a + b S_{t-1} + e`` (``-ln 2 / ln(1 + b)``); None unless the spread mean-reverts (b < 0)."""
    s = spread.dropna()
    ds, lag = s.diff().dropna(), s.shift(1).dropna()
    lag = lag.loc[ds.index]
    b = float(np.cov(lag, ds, ddof=1)[0, 1] / np.var(lag, ddof=1))
    return float(-np.log(2.0) / np.log1p(b)) if -1.0 < b < 0.0 else None


def diagnostics(a: pd.Series, b: pd.Series | None, spread: pd.Series, beta: float | None = None) -> dict:
    """Stationarity and cointegration evidence, plus the descriptive statistics a spread needs. With ``b`` None (a basket) only the spread itself is tested."""
    from statsmodels.tsa.stattools import adfuller, coint
    s = spread.dropna()
    adf = adfuller(s.to_numpy(), regression="c", autolag="AIC", result_object=False)
    out = {"n": int(len(s)), "adf": {"statistic": float(adf[0]), "p_value": float(adf[1]), "lags": int(adf[2]), "critical": {k: float(v) for k, v in adf[4].items()}, "null": "the spread has a unit root (not stationary)"},
           "stationary_at_5pct": bool(adf[1] < 0.05), "half_life_days": half_life(s), "spread_vol_annualised": float(s.diff().std(ddof=1) * np.sqrt(252.0)), "mean": float(s.mean()), "std": float(s.std(ddof=1))}
    if b is not None:
        df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        stat, p, crit = coint(df["a"].to_numpy(), df["b"].to_numpy(), trend="c", autolag="aic")
        out["engle_granger"] = {"statistic": float(stat), "p_value": float(p), "critical": [float(x) for x in crit], "null": "no cointegration"}
        out["cointegrated_at_5pct"] = bool(p < 0.05)
        ra, rb = a.diff(), b.diff()
        rc = ra.rolling(63).corr(rb).dropna()
        out["rolling_correlation"] = {"dates": [str(d.date()) for d in rc.index[-400:]], "values": [round(float(v), 4) for v in rc.tail(400)], "mean": float(rc.mean()), "min": float(rc.min()), "max": float(rc.max())}
    m = s.groupby(s.index.month).agg(["mean", "std", "count"])
    out["seasonality"] = {"by_month": [{"month": int(i), "mean_deviation": float(r["mean"] - s.mean()), "std": float(r["std"]) if np.isfinite(r["std"]) else None, "days": int(r["count"])} for i, r in m.iterrows()],
                          "caveat": "with a few years of history the monthly pattern describes the sample, not a calendar to trade"}
    vol = s.diff().rolling(21).std().dropna()
    cut = float(vol.median())
    reg = (vol > cut).astype(int)
    out["regimes"] = {"split": "21-day spread volatility above or below its median", "high_vol_share": float(reg.mean()), "current": "high volatility" if bool(reg.iloc[-1]) else "low volatility", "median_vol": cut,
                      "mean_reversion_in_high_vol": half_life(s.loc[reg[reg == 1].index]) if reg.sum() > 60 else None, "mean_reversion_in_low_vol": half_life(s.loc[reg[reg == 0].index]) if (reg == 0).sum() > 60 else None}
    out["verdict"] = ("evidence for stationarity and cointegration at the 5% level: still a hypothesis about the past" if out["stationary_at_5pct"] and out.get("cointegrated_at_5pct", True) else
                      "no statistical evidence that this spread mean-reverts: treat any apparent reversion as chance until shown otherwise")
    return out


def zscore(spread: pd.Series, lookback: int = 126) -> pd.Series:
    """The spread against the mean and standard deviation of the ``lookback`` days BEFORE each date (the date itself excluded)."""
    mu = spread.shift(1).rolling(lookback).mean()
    sd = spread.shift(1).rolling(lookback).std(ddof=1)
    return ((spread - mu) / sd).rename("z")


def backtest(a: pd.Series, b: pd.Series, lookback: int = 126, entry: float = 2.0, exit_: float = 0.5, stop: float = 4.0, cost_bps: float = 3.0, max_leverage: float = 1.0) -> dict:
    """Walk-forward mean reversion on ``a - beta b``. Each day the hedge ratio is the OLS ratio over the ``lookback`` days before it; the position (+1 long the spread, -1 short, 0 flat) set at the close of day t
    earns the spread return of day t+1: ``r_a - beta r_b`` on the log indices, scaled by ``max_leverage / (1 + |beta|)``. Entry at ``|z| > entry``, exit at ``|z| < exit_``, a stop at ``|z| > stop``
    (a deviation that large is read as a break, not an opportunity). Trading costs are ``cost_bps`` of the traded notional of each leg."""
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(df) < lookback + 60:
        raise ValueError(f"not enough dates: {len(df)} available, {lookback + 60} needed")
    ra, rb = df["a"].diff(), df["b"].diff()
    beta = pd.Series(np.nan, index=df.index)
    cov = ra.shift(1).rolling(lookback).cov(rb.shift(1))
    var = rb.shift(1).rolling(lookback).var()
    beta = (df["a"].diff().shift(1).rolling(lookback).cov(df["b"].diff().shift(1)) / var).clip(-5, 5)
    spread = df["a"] - beta * df["b"]
    z = zscore(spread, lookback)
    pos = pd.Series(0.0, index=df.index)
    cur = 0.0
    for i in range(len(df)):
        zi = z.iloc[i]
        if not np.isfinite(zi):
            pos.iloc[i] = 0.0
            continue
        if cur == 0.0 and abs(zi) > entry and abs(zi) < stop:
            cur = -np.sign(zi)                                                         # a spread far above its mean is shorted
        elif cur != 0.0 and (abs(zi) < exit_ or abs(zi) > stop):
            cur = 0.0
        pos.iloc[i] = cur
    held = pos.shift(1).fillna(0.0)                                                    # decided at yesterday's close, earned today
    scale = max_leverage / (1.0 + beta.abs())
    gross = (held * scale * (ra - beta.shift(1) * rb)).fillna(0.0)
    traded = (held.diff().abs().fillna(0.0)) * scale * (1.0 + beta.abs())
    costs = traded * cost_bps / 1e4
    net = (gross - costs).iloc[lookback + 1:]
    trades = int((held.diff().abs() > 0).sum())
    summary = {"sharpe": val._sharpe(net.to_numpy(), 252.0), "annual_return": float(net.mean() * 252), "annual_vol": float(net.std(ddof=1) * np.sqrt(252)), "max_drawdown": val._maxdd(net.to_numpy()),
               "trades": trades, "share_in_market": float((held.iloc[lookback + 1:] != 0).mean()), "turnover_per_year": float(traded.iloc[lookback + 1:].sum() / (len(net) / 252.0)), "cost_drag_annual": float(costs.iloc[lookback + 1:].mean() * 252)}
    modelled = {"bid-ask spread crossed on every fill": False, "commissions": cost_bps > 0, "no look-ahead: signal at one close, trade at a later price": True, "contract multiplier and tick size": False,
                "daily variation margin": False, "initial margin and collateral financing": False, "contract rolling on a stated rule": True, "roll execution cost (calendar-spread half-spread)": False,
                "liquidity limits (volume, open interest, size)": False, "expiry and delivery conventions": True}
    return {"summary": summary, "dates": [str(d.date()) for d in net.index], "equity": [round(float(v), 5) for v in (1 + net).cumprod()], "z": [None if not np.isfinite(v) else round(float(v), 3) for v in z.reindex(net.index)],
            "position": [float(v) for v in held.reindex(net.index)], "beta": [None if not np.isfinite(v) else round(float(v), 3) for v in beta.reindex(net.index)],
            "validation": {"bootstrap": val.bootstrap(net, n=500), "realism": val.realism("futures", modelled, {"bid-ask spread crossed on every fill": "costs are one flat figure in basis points of traded notional"})},
            "settings": {"lookback": lookback, "entry": entry, "exit": exit_, "stop": stop, "cost_bps": cost_bps, "max_leverage": max_leverage},
            "caveat": "Walk-forward on one sample: the hedge ratio and the thresholds were never tuned on the data they are judged on, but they were chosen before seeing it only if you did not look. A statistical relationship is a hypothesis."}
