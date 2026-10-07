"""Cross-asset strategies that read per-instrument signals from the macro panel: carry, basis momentum and long-term reversal (a value proxy).

These need a bundle that carries ``CARRY_<asset>`` (and, for commodities, ``BASISMOM_<asset>``) series, such as those built by ``assets.bundles``; on a bundle without them they raise a clear
error rather than inventing a signal. All are causal: the signal on date ``t`` is what a trader could compute at the close of ``t``.

``carry_xs``            rank assets by carry per unit of volatility WITHIN each asset class (Koijen, Moskowitz, Pedersen & Vrugt 2018): long high carry, short low carry
``carry_ts``            time series: long an asset whose carry is positive, short if negative, sized by carry over volatility
``basis_momentum``      commodities: long where the front of the curve has been outperforming the back (Boons & Prado 2019), within the class
``long_term_reversal``  value proxy (Asness, Moskowitz & Pedersen 2013): fade the 5-year return, within each asset class
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import daily_vol


def _signal_frame(data, prefix: str) -> pd.DataFrame:
    cols = {a: data.macro[f"{prefix}_{a}"] for a in data.assets if f"{prefix}_{a}" in data.macro.columns}
    if not cols:
        raise KeyError(f"needs macro series named {prefix}_<asset>; this bundle has none (build one with src.assets.bundles)")
    return pd.DataFrame(cols).reindex(index=data.index, columns=data.assets)


def _within_class_z(score: pd.DataFrame, data, min_members: int = 2, clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score computed separately inside each asset class (classes with fewer than ``min_members`` live assets on a date get NaN)."""
    out = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    groups: dict[str, list[str]] = {}
    for a in score.columns:
        groups.setdefault(data.asset_class.get(a, "unknown"), []).append(a)
    for members in groups.values():
        block = score[members]
        n = block.notna().sum(axis=1)
        mean, sd = block.mean(axis=1), block.std(axis=1, ddof=0).replace(0.0, np.nan)
        z = block.sub(mean, axis=0).div(sd, axis=0).clip(-clip, clip)
        out[members] = z.where(n >= min_members, np.nan)
    return out


@register_model("carry_xs", "carry", "Cross-asset carry: rank instruments by carry per unit of volatility within their asset class and hold the high-carry ones against the low-carry ones")
class CarryCrossSection(ForecastModel):
    """Carry is the return an instrument earns if prices do not move (roll yield, interest differential, yield over cash): a premium for bearing risk that has appeared in every liquid asset class."""

    name, family, position_mode = "carry_xs", "carry", "cross_sectional"
    rebalance = "weekly"

    def __init__(self, vol_halflife: float = 60.0, clip: float = 3.0, per_class: bool = True):
        if vol_halflife <= 0 or clip <= 0:
            raise ValueError("vol_halflife and clip must be positive")
        self.vol_halflife, self.clip, self.per_class = vol_halflife, clip, per_class

    def score(self, data):
        carry = _signal_frame(data, "CARRY")
        vol = daily_vol(data.returns, self.vol_halflife) * np.sqrt(252.0)
        sharpe_carry = carry / vol.replace(0.0, np.nan)
        z = _within_class_z(sharpe_carry, data, 2, self.clip) if self.per_class else (sharpe_carry.sub(sharpe_carry.mean(axis=1), axis=0).div(sharpe_carry.std(axis=1), axis=0)).clip(-self.clip, self.clip)
        return z.where(data.investable & carry.notna())


@register_model("carry_ts", "carry", "Time-series carry: long an instrument while its carry is positive and short while negative, sized by carry over volatility")
class CarryTimeSeries(ForecastModel):
    """Each instrument is its own bet: an upward-sloping reward-for-waiting signal, scaled so a unit of carry-to-risk has the same weight everywhere."""

    name, family, position_mode = "carry_ts", "carry", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, vol_halflife: float = 60.0, scale: float = 0.5, smooth: int = 21):
        if vol_halflife <= 0 or scale <= 0 or smooth < 1:
            raise ValueError("vol_halflife and scale must be positive, smooth >= 1")
        self.vol_halflife, self.scale, self.smooth = vol_halflife, scale, smooth

    def score(self, data):
        carry = _signal_frame(data, "CARRY").rolling(self.smooth, min_periods=1).mean()
        vol = daily_vol(data.returns, self.vol_halflife) * np.sqrt(252.0)
        return np.tanh(carry / vol.replace(0.0, np.nan) / self.scale).where(data.investable & carry.notna())


@register_model("basis_momentum", "carry", "Basis momentum: within commodities, favour instruments whose front contract has outperformed the second contract over the last year")
class BasisMomentum(ForecastModel):
    """When the nearby contract beats the deferred one for months, the market is tightening; the effect has predicted returns beyond ordinary momentum and carry."""

    name, family, position_mode = "basis_momentum", "carry", "cross_sectional"
    rebalance = "weekly"

    def __init__(self, classes: tuple = ("commodity",)):
        self.classes = tuple(classes)

    def score(self, data):
        sig = _signal_frame(data, "BASISMOM")
        members = [a for a in data.assets if data.asset_class.get(a) in self.classes]
        out = pd.DataFrame(np.nan, index=data.index, columns=data.assets)
        if len(members) >= 2:
            out[members] = _within_class_z(sig[members], data, 2)
        return out.where(data.investable)


@register_model("long_term_reversal", "cross-sectional", "Long-term reversal (value proxy): within each asset class, buy what has fallen over five years and sell what has risen")
class LongTermReversal(ForecastModel):
    """Prices revert toward fundamental value over multi-year horizons, so a five-year loser is cheap against its own history."""

    name, family, position_mode = "long_term_reversal", "value", "cross_sectional"

    def __init__(self, lookback: int = 1260, skip: int = 252):
        if lookback <= skip + 21 or skip < 0:
            raise ValueError("lookback must exceed skip + 21")
        self.lookback, self.skip = lookback, skip

    def score(self, data):
        past = np.log(data.prices.shift(self.skip) / data.prices.shift(self.lookback))
        return (-_within_class_z(past, data, 2)).where(data.investable & past.notna())
