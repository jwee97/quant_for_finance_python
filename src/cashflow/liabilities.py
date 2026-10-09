"""Liabilities: a stream of promised payments, what it is worth, how it moves with interest rates, and a portfolio built to keep up with it.

A pension plan or an insurer owes dated payments. Their present value is the liability, and it rises when interest rates fall (the same payments are discounted less), so assets that do not
rise with it leave the plan worse funded exactly when rates fall. The funding ratio is assets over liabilities. Liability-driven investing (LDI) holds a bucket of long bonds whose dollar duration
offsets a share of the liabilities' (the hedge ratio) and puts the rest in return-seeking assets; a glide path raises the hedge ratio as the plan becomes better funded, locking in the gain.

* :class:`Liability`: payments at dates in years from an origin, with ``pv``, ``duration`` and ``dv01`` at any yield (annual compounding);
* :func:`liability_value`: the value on every date from a yield series, time passing and paid-out payments leaving;
* :func:`hedge_ratio_glide`: the hedge ratio as a rising function of the funding ratio;
* :func:`simulate_ldi`: assets and liabilities followed together with a rebalance to the LDI weights, against the choices it is compared with (all return-seeking, fully hedged).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Liability:
    """Payments of ``amounts`` dollars due ``times`` years after the origin."""

    times: tuple
    amounts: tuple

    def __post_init__(self):
        if len(self.times) != len(self.amounts) or not len(self.times):
            raise ValueError("a liability needs equal, non-empty lists of times and amounts")
        if min(self.times) <= 0 or min(self.amounts) < 0:
            raise ValueError("payment times must be positive and amounts non-negative")

    @classmethod
    def level(cls, amount: float, years: int, first: float = 1.0) -> "Liability":
        """``amount`` a year for ``years`` years, the first ``first`` years from now."""
        return cls(tuple(first + k for k in range(int(years))), (float(amount),) * int(years))

    @classmethod
    def growing(cls, amount: float, years: int, growth: float, first: float = 1.0) -> "Liability":
        """Payments that start at ``amount`` and grow by ``growth`` a year (pensions indexed to inflation)."""
        return cls(tuple(first + k for k in range(int(years))), tuple(float(amount) * (1.0 + growth) ** k for k in range(int(years))))

    def _arrays(self):
        return np.asarray(self.times, float), np.asarray(self.amounts, float)

    def pv(self, rate: float, at: float = 0.0) -> float:
        """The value at time ``at`` (years from the origin) of the payments still to come, discounted at the annual rate ``rate`` (0.04 = 4%)."""
        t, a = self._arrays()
        left = t > at + 1e-12
        return float((a[left] * (1.0 + rate) ** -(t[left] - at)).sum())

    def duration(self, rate: float, at: float = 0.0) -> float:
        """Modified duration: the fall in value, in percent, for a one-point rise in the rate."""
        t, a = self._arrays()
        left = t > at + 1e-12
        pv = a[left] * (1.0 + rate) ** -(t[left] - at)
        return float((pv * (t[left] - at)).sum() / pv.sum() / (1.0 + rate)) if pv.sum() > 0 else 0.0

    def dv01(self, rate: float, at: float = 0.0) -> float:
        """The dollar fall in value for a one-basis-point rise in the rate."""
        return self.pv(rate, at) * self.duration(rate, at) * 1e-4

    def payments(self, index: pd.DatetimeIndex, origin=None) -> pd.Series:
        """The payments as dollars on the first trading day on or after each due date (``origin`` defaults to the first date of ``index``); payments after the last date are left out."""
        index = pd.DatetimeIndex(index).sort_values()
        start = index[0] if origin is None else pd.Timestamp(origin)
        out = pd.Series(0.0, index=index)
        for t, a in zip(self.times, self.amounts):
            day = start + pd.Timedelta(days=round(t * 365.25))
            pos = index.searchsorted(day)
            if pos < len(index):
                out.iloc[pos] += a
        return out


def liability_value(index: pd.DatetimeIndex, yields: pd.Series, liability: Liability, origin=None, spread: float = 0.0, static: bool = False) -> pd.Series:
    """The value of the liability on every date: the payments still to come, discounted at that day's yield (a percentage such as 4.2) plus ``spread`` (also a percentage).

    Time passes, so the value accretes toward each payment, and a payment leaves the liability once its date has been reached. With ``static`` it does not: the payments are always as far ahead as
    on the first date (an open plan that keeps accruing as it pays), so the value moves with the yield alone."""
    index = pd.DatetimeIndex(index)
    start = index[0] if origin is None else pd.Timestamp(origin)
    at = np.zeros(len(index)) if static else np.asarray((index - start).days, float) / 365.25
    rate = (pd.Series(yields).reindex(index).ffill().to_numpy(dtype=float) + spread) / 100.0
    t, a = liability._arrays()
    gap = t[None, :] - at[:, None]
    pv = np.where(gap > 1e-12, a[None, :] * (1.0 + rate[:, None]) ** -np.where(gap > 1e-12, gap, 0.0), 0.0)
    return pd.Series(pv.sum(axis=1), index=index)


def estimate_duration(returns: pd.Series, yields: pd.Series, window: int = 252) -> pd.Series:
    """A bond fund's duration from data: minus the covariance of its returns with yield changes over the trailing ``window`` days, divided by the variance of those changes (yields in percent)."""
    dy = pd.Series(yields).reindex(returns.index).ffill().diff() / 100.0
    cov = returns.rolling(window, min_periods=window // 2).cov(dy)
    var = dy.rolling(window, min_periods=window // 2).var()
    return (-cov / var.replace(0.0, np.nan)).clip(lower=0.0)


def hedge_ratio_glide(funding_ratio, low: float = 0.30, high: float = 1.0, fr_low: float = 0.80, fr_high: float = 1.10):
    """The hedge ratio for a funding ratio: ``low`` at or below ``fr_low``, ``high`` at or above ``fr_high`` and a straight line between, so the plan hedges more as it becomes better funded."""
    if not 0 <= low <= high <= 1.5 or fr_low >= fr_high:
        raise ValueError("0 <= low <= high <= 1.5 and fr_low < fr_high")
    return np.interp(funding_ratio, [fr_low, fr_high], [low, high])


def simulate_ldi(prices: pd.DataFrame, yields: pd.Series, liability: Liability, hedge_assets: list, seeking_assets: list, *, initial_funding_ratio: float = 0.85,
                 hedge_ratio="glide", glide: dict | None = None, durations: dict | None = None, rebalance: str = "monthly", pay: bool = True, cost_bps: float = 2.0) -> dict:
    """Follow a plan with ``liability`` through ``prices`` (daily, total-return series) and ``yields`` (a percentage, the discount rate).

    Assets start at ``initial_funding_ratio`` times the liability and are held in two buckets: ``hedge_assets`` (equal weight; long bonds) and ``seeking_assets`` (equal weight). At each rebalance the
    hedge bucket gets the weight that matches ``hedge_ratio`` times the liability's dollar duration with its own: ``hedge_ratio * D_L / (funding_ratio * D_hedge)``, at most one. ``hedge_ratio`` is a
    number (0 = all return-seeking, 1 = fully hedged) or ``"glide"`` (see :func:`hedge_ratio_glide`, with settings from ``glide``). Hedge durations are those in ``durations`` or, failing that,
    estimated from the data up to each date (``estimate_duration``; 8 before there is enough history). With ``pay`` the liability is a closed plan: its payments are paid out of the assets (pro rata)
    on their dates and leave the liability. Without it the plan is open (the liability is valued as if its payments were always as far ahead as at the start, ``static``), which isolates what interest
    rates and the asset mix do to the funding ratio. Returns a frame with ``assets``, ``liabilities``, ``funding_ratio``, ``hedge_ratio`` and ``w_hedge`` by date and a ``summary``."""
    index = prices.index
    if not hedge_assets or not seeking_assets:
        raise ValueError("hedge_assets and seeking_assets must both name at least one asset")
    ret = prices[hedge_assets + seeking_assets].pct_change().fillna(0.0)
    liab = liability_value(index, yields, liability, static=not pay)
    payments = liability.payments(index) if pay else pd.Series(0.0, index=index)
    if durations is not None:
        dur_h = pd.Series(float(np.mean([durations[a] for a in hedge_assets])), index=index)
    else:
        dur_h = pd.concat([estimate_duration(ret[a], yields) for a in hedge_assets], axis=1).mean(axis=1).reindex(index).fillna(8.0)
    marks = set(pd.DatetimeIndex(pd.Series(index, index=index.to_period({"monthly": "M", "quarterly": "Q", "annual": "Y"}[rebalance])).groupby(level=0).max().to_numpy()))
    rate = pd.Series(yields).reindex(index).ffill() / 100.0
    at = np.zeros(len(index)) if not pay else (index - index[0]).days / 365.25
    h, s = len(hedge_assets), len(seeking_assets)
    A = initial_funding_ratio * float(liab.iloc[0])
    values = np.zeros(h + s)
    cost = cost_bps * 1e-4
    out = {k: np.zeros(len(index)) for k in ("assets", "funding_ratio", "hedge_ratio", "w_hedge")}
    settings = dict(glide or {})
    for i, day in enumerate(index):
        if i > 0:
            values = values * (1.0 + ret.iloc[i].to_numpy())
            if payments.iloc[i] > 0 and values.sum() > 0:
                values = values * max(1.0 - payments.iloc[i] / values.sum(), 0.0)
        A = float(values.sum()) if i > 0 else A
        L = float(liab.iloc[i])
        fr = A / L if L > 0 else np.inf
        if i == 0 or day in marks:
            ratio = float(hedge_ratio_glide(fr, **settings)) if hedge_ratio == "glide" else float(hedge_ratio)
            d_l = liability.duration(float(rate.iloc[i]), float(at[i]))
            w_h = float(np.clip(ratio * d_l / (max(fr, 1e-9) * max(float(dur_h.iloc[i]), 1e-9)), 0.0, 1.0)) if L > 0 else 0.0
            weights = np.r_[np.full(h, w_h / h), np.full(s, (1.0 - w_h) / s)]
            new = weights * A
            A -= cost * float(np.abs(new - values).sum()) if i > 0 else 0.0
            values = weights * A
            out["hedge_ratio"][i], out["w_hedge"][i] = ratio, w_h
        elif i > 0:
            out["hedge_ratio"][i], out["w_hedge"][i] = out["hedge_ratio"][i - 1], float(values[:h].sum() / values.sum()) if values.sum() > 0 else 0.0
        out["assets"][i], out["funding_ratio"][i] = float(values.sum()), (float(values.sum()) / L if L > 0 else np.inf)
    frame = pd.DataFrame(out, index=index)
    frame.insert(1, "liabilities", liab)
    fr = frame["funding_ratio"].replace(np.inf, np.nan).dropna()
    change = np.log(fr).diff().dropna()
    frame.attrs["summary"] = {"funding_ratio_start": float(fr.iloc[0]), "funding_ratio_end": float(fr.iloc[-1]), "funding_ratio_min": float(fr.min()), "funding_ratio_vol": float(change.std() * np.sqrt(252)),
                              "max_funding_drawdown": float((fr / fr.cummax() - 1.0).min()), "share_below_one": float((fr < 1.0).mean())}
    return {"frame": frame, "summary": frame.attrs["summary"]}
