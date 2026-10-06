"""A ``MarketBundle`` for the crypto branch: spot BTC and ETH plus two synthetic delta-neutral CARRY assets.

``BTC_CARRY`` is long spot and short the perpetual: its daily return is the funding it collects plus the change in spot minus the change in the
perpetual (the basis). The macro frame carries the signals (trailing funding, the basis, stablecoin supply growth) as known at each close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.crypto import build_crypto_frames, crypto_version, ensure_crypto_raw
from .data import MarketBundle, bundle_from_prices


def load_crypto_bundle(config) -> MarketBundle:
    frames = build_crypto_frames(ensure_crypto_raw(config))
    spot, perp, funding, stable = frames["spot"], frames["perp"], frames["funding"], frames["stable"]
    idx = spot.index.intersection(perp.index).intersection(funding.index)
    spot, perp, funding = spot.loc[idx], perp.loc[idx], funding.loc[idx]
    carry_returns = funding + spot.pct_change() - perp.pct_change()                   # short the perpetual, long spot (linear approximation)
    levels = {}
    macro = {}
    for asset in ("BTC", "ETH"):
        levels[asset] = spot[asset]
        levels[f"{asset}_CARRY"] = (1.0 + carry_returns[asset].fillna(0.0)).cumprod() * 100.0
        macro[f"FUNDING_{asset}"] = funding[asset].rolling(7, min_periods=7).mean() * 365.0         # annualised trailing funding earned by a short
        macro[f"BASIS_{asset}"] = perp[asset] / spot[asset] - 1.0
    macro["STABLE_SUPPLY"] = stable.reindex(idx).ffill()
    prices = pd.DataFrame(levels)
    macro = pd.DataFrame(macro)
    weekdays = prices.index[prices.index.dayofweek < 5]                                    # sample weekday marks so 252-day annualisation applies
    prices, macro = prices.loc[weekdays], macro.loc[weekdays]
    prices.index = pd.DatetimeIndex(prices.index.tz_localize(None))
    macro.index = prices.index
    classes = {"BTC": "crypto", "ETH": "crypto", "BTC_CARRY": "crypto_carry", "ETH_CARRY": "crypto_carry"}
    bundle = bundle_from_prices(prices, asset_class=classes, macro=macro, name=f"crypto-{crypto_version(config)}", min_history=60)
    return bundle
