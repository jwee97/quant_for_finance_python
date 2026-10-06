"""Regime detectors: the market's state as probabilities, causally.

Each detector turns a ``MarketBundle`` into a ``RegimeSeries``: one probability per regime name on every date, using
only information available at that date's close. The names the allocator and the risk policy react to are::

    Crisis      high and rising stress: volatility in its top decile with a drawdown under way
    HighVol     elevated volatility without a crisis
    LowVol      the calm default
    Inflation   consumer-price inflation high (US CPI year-on-year, published with its lag)
    Deflation   consumer-price inflation near or below zero

A trend flag (Bull / Bear: the market proxy above or below its 200-day average) is attached as an annotation so a
regime can be reported as ``HighVolBear``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register_detector
from .types import RegimeSeries


def ramp(x, low: float, high: float):
    """0 below ``low``, 1 above ``high``, linear between: a soft threshold."""
    return ((x - low) / (high - low)).clip(0.0, 1.0)


def market_proxy(bundle) -> pd.Series:
    """Daily return of an equal-weight portfolio of the assets that are investable that day (no look-ahead: weights are 1/N)."""
    r = bundle.returns.where(bundle.investable.reindex_like(bundle.returns).fillna(False))
    return r.mean(axis=1).fillna(0.0)


def _trend(proxy: pd.Series, window: int = 200) -> pd.Series:
    level = (1.0 + proxy).cumprod()
    ma = level.rolling(window, min_periods=window // 2).mean()
    flag = pd.Series(np.where(level >= ma, "Bull", "Bear"), index=proxy.index)
    return flag.where(ma.notna())


class _Base:
    name = "base"

    def detect(self, bundle) -> RegimeSeries:
        raise NotImplementedError


@register_detector("vol_state", "Volatility-state model: where today's EWMA volatility ranks in its own trailing history, with a drawdown test for crises")
class VolStateDetector(_Base):
    name = "vol_state"

    def __init__(self, halflife: float = 21.0, rank_window: int = 1260, min_history: int = 504, high_ramp=(0.60, 0.85),
                 crisis_ramp=(0.90, 0.98), drawdown_ramp=(0.05, 0.15), drawdown_window: int = 252, trend_window: int = 200):
        self.halflife, self.rank_window, self.min_history = halflife, rank_window, min_history
        self.high_ramp, self.crisis_ramp, self.drawdown_ramp = tuple(high_ramp), tuple(crisis_ramp), tuple(drawdown_ramp)
        self.drawdown_window, self.trend_window = drawdown_window, trend_window

    def detect(self, bundle) -> RegimeSeries:
        proxy = market_proxy(bundle)
        vol = np.sqrt((proxy ** 2).ewm(halflife=self.halflife, adjust=False, min_periods=21).mean() * 252.0)
        rank = vol.rolling(self.rank_window, min_periods=self.min_history).apply(lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else np.nan, raw=True)
        level = (1.0 + proxy).cumprod()
        drawdown = 1.0 - level / level.rolling(self.drawdown_window, min_periods=21).max()
        p_crisis = ramp(rank, *self.crisis_ramp) * ramp(drawdown, *self.drawdown_ramp)
        p_high = (1.0 - p_crisis) * ramp(rank, *self.high_ramp)
        p_low = 1.0 - p_crisis - p_high
        probabilities = pd.DataFrame({"LowVol": p_low, "HighVol": p_high, "Crisis": p_crisis}).where(rank.notna())
        series = RegimeSeries(probabilities, self.name, pd.DataFrame({"trend": _trend(proxy, self.trend_window), "vol": vol, "rank": rank}))
        series.validate()
        return series


@register_detector("macro", "Inflation and deflation from US CPI year-on-year, published-by-date (soft thresholds)")
class MacroDetector(_Base):
    name = "macro"

    def __init__(self, series: str = "CPI_YOY", inflation_ramp=(0.035, 0.050), deflation_ramp=(0.010, 0.000)):
        self.series, self.inflation_ramp, self.deflation_ramp = series, tuple(inflation_ramp), tuple(deflation_ramp)

    def detect(self, bundle) -> RegimeSeries:
        cpi = bundle.macro_series(self.series)
        p_inflation = ramp(cpi, *self.inflation_ramp)
        lo, hi = self.deflation_ramp                                  # deflation_ramp is (start, full): inflation falling from `start` to `full`
        p_deflation = ((cpi - lo) / (hi - lo)).clip(0.0, 1.0)
        p_neutral = (1.0 - p_inflation - p_deflation).clip(lower=0.0)
        total = p_inflation + p_deflation + p_neutral
        probabilities = pd.DataFrame({"Neutral": p_neutral / total, "Inflation": p_inflation / total, "Deflation": p_deflation / total}).where(cpi.notna())
        series = RegimeSeries(probabilities, self.name)
        series.validate()
        return series


@register_detector("composite", "Priority chain Crisis > Inflation > Deflation > HighVol > LowVol, built from the volatility-state and macro detectors")
class CompositeDetector(_Base):
    name = "composite"

    def __init__(self, vol_params: dict | None = None, macro_params: dict | None = None):
        self.vol = VolStateDetector(**(vol_params or {}))
        self.macro = MacroDetector(**(macro_params or {}))

    def detect(self, bundle) -> RegimeSeries:
        v, m = self.vol.detect(bundle), self.macro.detect(bundle)
        index = v.probabilities.index
        pv = v.probabilities
        pm = m.probabilities.reindex(index)
        p_crisis = pv["Crisis"]
        rest = 1.0 - p_crisis
        p_infl = rest * pm["Inflation"]
        p_defl = rest * (1.0 - pm["Inflation"]) * pm["Deflation"]
        remainder = rest * (1.0 - pm["Inflation"]) * (1.0 - pm["Deflation"])
        calm = (pv["HighVol"] + pv["LowVol"]).replace(0.0, np.nan)
        high_share = (pv["HighVol"] / calm).fillna(0.0)
        probabilities = pd.DataFrame({"LowVol": remainder * (1.0 - high_share), "HighVol": remainder * high_share, "Crisis": p_crisis,
                                      "Inflation": p_infl, "Deflation": p_defl}).where(pv["LowVol"].notna() & pm["Neutral"].notna())
        series = RegimeSeries(probabilities, self.name, v.annotations)
        series.validate()
        return series


@register_detector("hmm", "Two-state Gaussian hidden Markov model on the market proxy, refit walk-forward; the high-variance state is HighVol")
class HMMDetector(_Base):
    name = "hmm"

    def __init__(self, min_train: int = 750, refit_every: int = 126, n_init: int = 3, seed: int = 11):
        self.min_train, self.refit_every, self.n_init, self.seed = min_train, refit_every, n_init, seed

    def detect(self, bundle) -> RegimeSeries:
        from ..models.regimes import walk_forward_hmm

        proxy = market_proxy(bundle)
        wf = walk_forward_hmm(proxy.to_frame("proxy"), 2, self.min_train, self.refit_every, self.n_init, 1, seed=self.seed)
        p_high = wf.p_high
        probabilities = pd.DataFrame({"LowVol": 1.0 - p_high, "HighVol": p_high}).where(p_high.notna())
        series = RegimeSeries(probabilities, self.name, pd.DataFrame({"trend": _trend(proxy)}))
        series.validate()
        return series


@register_detector("static", "A single regime that is always on: the control for any regime-based rule")
class StaticDetector(_Base):
    name = "static"

    def __init__(self, regime: str = "LowVol"):
        self.regime = regime

    def detect(self, bundle) -> RegimeSeries:
        return RegimeSeries(pd.DataFrame({self.regime: 1.0}, index=bundle.index), self.name)


@register_detector("gmm", "Two-component Gaussian mixture on the market proxy, refit walk-forward, each day classified on its own; the high-variance component is HighVol")
class GMMDetector(_Base):
    name = "gmm"

    def __init__(self, min_train: int = 750, refit_every: int = 126, n_init: int = 3, seed: int = 11):
        self.min_train, self.refit_every, self.n_init, self.seed = min_train, refit_every, n_init, seed

    def detect(self, bundle) -> RegimeSeries:
        from ..models.regimes import walk_forward_gmm

        proxy = market_proxy(bundle)
        probs = walk_forward_gmm(proxy.to_frame("proxy"), 2, self.min_train, self.refit_every, self.n_init, self.seed)
        p_high = probs["state_1"]                                      # components are ordered by variance: state_1 is the high-variance one
        probabilities = pd.DataFrame({"LowVol": 1.0 - p_high, "HighVol": p_high}).where(p_high.notna())
        series = RegimeSeries(probabilities, self.name, pd.DataFrame({"trend": _trend(proxy)}))
        series.validate()
        return series


@register_detector("bocpd", "Bayesian online change-point detection on the market proxy: Shock is the probability the current run of stable returns is at most ten days old")
class BOCPDDetector(_Base):
    name = "bocpd"

    def __init__(self, hazard_lambda: float = 250.0, burn_in: int = 250, short_run: int = 10):
        self.hazard_lambda, self.burn_in, self.short_run = hazard_lambda, burn_in, short_run

    def detect(self, bundle) -> RegimeSeries:
        from ..models.regimes import bocpd

        proxy = market_proxy(bundle)
        result = bocpd(proxy, self.hazard_lambda, self.burn_in, short_runs=(self.short_run,))
        shock = result.short_mass.iloc[:, 0].reindex(proxy.index).clip(0.0, 1.0)
        probabilities = pd.DataFrame({"Stable": 1.0 - shock, "Shock": shock}).where(shock.notna())
        series = RegimeSeries(probabilities, self.name, pd.DataFrame({"trend": _trend(proxy)}))
        series.validate()
        return series
