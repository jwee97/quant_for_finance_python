"""The optional crypto branch: three strategies on Deribit perpetual-futures data and the stablecoin supply.

All use the bundle from ``framework.crypto_bundle.load_crypto_bundle``. Not built: cross-exchange spreads (they need synchronised quotes
from several venues; only one venue is reachable here) and the Binance/Bybit funding books (blocked from this environment).
"""

from __future__ import annotations

import numpy as np

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import expanding_z, zero_except


@register_model("funding_carry", "crypto", "Funding-rate carry: hold long-spot / short-perpetual when trailing funding is high, reverse when it is negative")
class FundingCarry(ForecastModel):
    """Funding is paid by whichever side is crowded; it persists, so trailing funding forecasts the carry earned next month."""

    name, family, position_mode = "funding_carry", "crypto", "time_series"
    requires = ("FUNDING_BTC", "FUNDING_ETH")

    def score(self, data):
        self.require(data)
        return zero_except(data.index, data.assets, {f"{a}_CARRY": np.tanh(data.macro[f"FUNDING_{a}"] / 0.10) for a in ("BTC", "ETH")}).where(data.investable)


@register_model("basis_reversion", "crypto", "Basis trading: hold the carry position when the perpetual trades rich to spot relative to its own history")
class BasisReversion(ForecastModel):
    """A wide perpetual-over-spot basis converges; the short-perpetual leg earns the convergence."""

    name, family, position_mode = "basis_reversion", "crypto", "time_series"
    requires = ("BASIS_BTC", "BASIS_ETH")

    def score(self, data):
        self.require(data)
        values = {f"{a}_CARRY": np.tanh(expanding_z(data.macro[f"BASIS_{a}"].dropna(), 90).reindex(data.index) / 2.0) for a in ("BTC", "ETH")}
        return zero_except(data.index, data.assets, values).where(data.investable)


@register_model("stablecoin_flow", "crypto", "Stablecoin flow factor: be long BTC and ETH when the total stablecoin supply has been growing quickly")
class StablecoinFlow(ForecastModel):
    """Stablecoin issuance is dry powder entering the crypto market; supply growth should lead spot prices."""

    name, family, position_mode = "stablecoin_flow", "crypto", "time_series"
    requires = ("STABLE_SUPPLY",)

    def __init__(self, window: int = 30):
        self.window = window

    def score(self, data):
        self.require(data)
        growth = np.log(data.macro["STABLE_SUPPLY"]).diff(self.window)
        s = np.tanh(expanding_z(growth.dropna(), 120).reindex(data.index) / 2.0)
        return zero_except(data.index, data.assets, {"BTC": s, "ETH": s}).where(data.investable)
