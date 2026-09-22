"""Mean-reversion signal (Ch. 22 §22.3.1).

Sign convention, stated once and enforced here. The *feature* is the z-score

    Z = (P - mu) / sigma

and the mean-reversion *hypothesis* is that ``corr(Z_t, r_{t+h}) < 0``. The
signal therefore returns ``-Z`` (via ``sign = -1``): a stretched-high asset
becomes a negative forecast. Stage 4 tests the hypothesis on the raw feature
first; this class exists to trade it only once it survives that test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..features.mean_reversion import price_zscore, short_term_reversal
from ..features.volatility import rolling_volatility
from .transform import cross_sectional_rank, signal_to_positions

VARIANTS = ("zscore", "zscore_ranked", "reversal")


@dataclass
class MeanReversionSignal:
    lookback: int = 21
    variant: str = "zscore"
    price_basis: str = "log"
    sign: int = -1
    vol_lookback: int = 63
    name: str = field(default="")

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got '{self.variant}'")
        if self.sign not in (-1, 1):
            raise ValueError("sign must be -1 (mean reversion) or +1 (trend)")
        if not self.name:
            self.name = f"mr_{self.variant}_{self.lookback}"

    def feature(self, market) -> pd.DataFrame:
        """The raw, sign-neutral feature -- what Stage 4 regresses."""
        if self.variant == "reversal":
            feature = short_term_reversal(market.returns(), self.lookback)
        else:
            feature = price_zscore(market.prices, self.lookback, self.price_basis)
            if self.variant == "zscore_ranked":
                feature = cross_sectional_rank(feature)
        return feature.where(market.investable)

    def compute(self, market) -> pd.DataFrame:
        """The forecast: the feature with the hypothesis sign applied."""
        return self.sign * self.feature(market)

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
            "family": "mean_reversion",
            "variant": self.variant,
            "lookback": self.lookback,
            "sign": self.sign,
            "hypothesis": (
                f"Assets stretched away from their {self.lookback}-day mean revert, "
                f"i.e. the z-score is negatively related to subsequent returns."
            ),
        }
