"""Momentum signal (Ch. 22 §22.3.1, §22.3.9, §22.3.10).

A ``Signal`` is a small object with one job: turn ``MarketData`` into a
forecast frame, causally and reproducibly. It does *not* decide position
sizes -- ``signals.transform`` does that -- and it does not know anything
about portfolio construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..features.momentum import ranked_momentum, total_return_momentum, volatility_scaled_momentum
from ..features.volatility import ewma_volatility, rolling_volatility
from .transform import signal_to_positions

VARIANTS = ("raw", "vol_scaled", "ranked")


@dataclass
class MomentumSignal:
    """Cross-sectional / time-series momentum.

    ``skip`` days are dropped from the end of the lookback window. One day is
    the default for two reasons: the well-documented one-day reversal
    contaminates the trend measurement, and you cannot trade on the same close
    you measure.
    """

    lookback: int = 126
    variant: str = "vol_scaled"
    skip: int = 1
    vol_lookback: int = 63
    vol_method: str = "rolling"
    name: str = field(default="")

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got '{self.variant}'")
        if not self.name:
            self.name = f"mom_{self.variant}_{self.lookback}"

    def compute(self, market) -> pd.DataFrame:
        prices, returns = market.prices, market.returns()
        if self.variant == "raw":
            signal = total_return_momentum(prices, self.lookback, self.skip)
        elif self.variant == "vol_scaled":
            signal = volatility_scaled_momentum(
                prices, returns, self.lookback, self.skip, self.vol_lookback, self.vol_method
            )
        else:
            signal = ranked_momentum(prices, self.lookback, self.skip)
        return signal.where(market.investable)

    def positions(self, market, transform_config: dict | None = None,
                  volatility: pd.DataFrame | None = None) -> pd.DataFrame:
        config = dict(transform_config or {})
        if volatility is None and config.pop("risk_scale", True):
            volatility = rolling_volatility(market.returns(), self.vol_lookback)
        return signal_to_positions(
            self.compute(market), volatility, investable=market.investable, **config
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "family": "momentum",
            "variant": self.variant,
            "lookback": self.lookback,
            "skip": self.skip,
            "vol_lookback": self.vol_lookback,
            "hypothesis": (
                f"Total return over the past {self.lookback} days (skipping the last "
                f"{self.skip}) carries information about subsequent returns."
            ),
        }
