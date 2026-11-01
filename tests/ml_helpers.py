"""Synthetic markets for the machine-learning tests: small, fast and with a known, plantable signal."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.framework import bundle_from_prices


def make_bundle(n_assets: int = 10, n: int = 1400, seed: int = 0, alpha: float = 0.0, volume: bool = True, macro: bool = True, start: str = "2015-01-05"):
    """A bundle of random-walk prices. ``alpha`` plants a persistent, slowly varying expected return per asset (an AR(1) with coefficient 0.97) that trailing returns reveal, so a model that reads momentum
    features has something to find; ``alpha = 0`` is pure noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    assets = [f"A{i:02d}" for i in range(n_assets)]
    a = np.zeros((n, n_assets))
    for t in range(1, n):
        a[t] = 0.97 * a[t - 1] + 0.0004 * rng.standard_normal(n_assets)
    market = 0.0004 + 0.006 * rng.standard_normal(n)
    r = np.zeros((n, n_assets))
    r[1:] = alpha * a[:-1] + 0.9 * market[1:, None] + 0.010 * rng.standard_normal((n - 1, n_assets))
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + r, axis=0), index=idx, columns=assets)
    vol = pd.DataFrame(np.exp(13 + 0.3 * rng.standard_normal((n, n_assets))), index=idx, columns=assets) if volume else None
    high = prices * (1.0 + np.abs(rng.normal(0, 0.004, prices.shape)))
    low = prices * (1.0 - np.abs(rng.normal(0, 0.004, prices.shape)))
    levels = None
    if macro:
        levels = pd.DataFrame({"DGS10": 2.0 + np.cumsum(rng.normal(0, 0.02, n)), "VIX": 18.0 + np.cumsum(rng.normal(0, 0.3, n)).clip(-8, 20), "T10Y2Y": 1.0 + np.cumsum(rng.normal(0, 0.01, n))}, index=idx)
    classes = {c: ("equity" if i % 2 == 0 else "rates") for i, c in enumerate(assets)}
    return bundle_from_prices(prices, volume=vol, high=high, low=low, asset_class=classes, macro=levels, name=f"synthetic{seed}", min_history=60)


def probe_inputs(n: int = 620, seed: int = 5):
    """The inputs of every feature group at once: volume, high and low, an unadjusted close, a macro panel with the series the macro, futures, options and crypto features read, and assets named so that the
    implied-volatility and funding features have something to read. For the feature audit, which needs every feature to be computable."""
    import types

    from src.ml import features as feat

    b = make_bundle(8, n, seed=seed)
    ren = {"A00": "SPY", "A01": "GLD", "A02": "BTC"}
    b.prices, b.returns, b.investable = b.prices.rename(columns=ren), b.returns.rename(columns=ren), b.investable.rename(columns=ren)
    for field in ("volume", "high", "low"):
        setattr(b, field, getattr(b, field).rename(columns=ren))
    macro, rng = b.macro.copy(), np.random.default_rng(1)
    for a in ("SPY", "GLD", "A03"):
        macro[f"CARRY_{a}"] = 0.02 * np.sin(np.arange(n) / 40.0 + len(a))
        macro[f"SLOPE_{a}"] = 0.01 * np.cos(np.arange(n) / 55.0)
        macro[f"BASISMOM_{a}"] = 0.03 * np.sin(np.arange(n) / 70.0)
    macro["GVZ"] = 16 + np.cumsum(rng.normal(0, 0.2, n)).clip(-6, 12)
    macro["FUNDING_BTC"] = 0.05 + 0.03 * np.sin(np.arange(n) / 30.0)
    macro["BASIS_BTC"] = 0.001 * np.cos(np.arange(n) / 20.0)
    macro["STABLE_SUPPLY"] = 1e11 * np.exp(0.0005 * np.arange(n))
    for name in ("T10Y3M", "BAA10Y", "MOVE", "DFF", "DTWEXBGS", "T10YIE"):
        macro[name] = 1.0 + np.cumsum(rng.normal(0, 0.02, n))
    b.macro = macro
    b.market = types.SimpleNamespace(close=b.prices * 0.97)
    return feat.inputs_from_bundle(b)


def changed_after(data, p, parts=("prices", "returns", "X", "Y")):
    """The same dataset with the ``parts`` (prices, returns, features ``X``, labels ``Y``) of every date after position ``p`` replaced. All four is a changed future; ``("prices", "returns", "Y")`` changes only the
    outcomes and leaves the features of the later rows as they were."""
    rng = np.random.default_rng(7)
    late = data.dates > data.dates[p]
    prices, returns, X, Y = data.prices.copy(), data.returns.copy(), data.X.astype("float64"), data.Y.astype("float64")
    if "prices" in parts:
        prices.loc[late] = prices.loc[late] * rng.uniform(0.6, 1.5, prices.loc[late].shape)
    if "returns" in parts:
        returns.loc[late] = rng.normal(0.0, 0.04, returns.loc[late].shape)
    rows = X.index.get_level_values(0) > data.dates[p]
    for name, frame, scale in (("X", X, 1.0), ("Y", Y, 0.1)):                  # the pattern of missing values is part of the calendar (the last labels are missing because the data ends), so it is kept; only the numbers change
        if name in parts:
            block = frame.loc[rows]
            frame.loc[rows] = np.where(block.isna().to_numpy(), np.nan, rng.normal(0.0, scale, block.shape))
    return dataclasses.replace(data, prices=prices, returns=returns, X=X, Y=Y)




NET = {"epochs": 2, "max_rows": 1500, "hidden": 8, "batch_size": 128, "lookback": 8, "patience": 2}
FAMILIES = {                                                  # name: (kind, params, tolerance, packages)
    "ridge": ("ridge", {"alpha": 10.0}, 1e-9, ()),
    "hmm": ("hmm_regime", {"n_states": 2}, 1e-9, ("hmmlearn",)),
    "lstm": ("lstm", NET, 1e-5, ("torch",)),
    "gcn": ("gcn", {"epochs": 2, "hidden": 8, "layers": 1, "batch_size": 64, "patience": 2}, 1e-5, ("torch",)),
    "ensemble": ("model_ensemble", {"members": "ridge,ridge[alpha=100]", "weighting": "dynamic"}, 1e-9, ()),
    "ppo": ("ppo", {"total_timesteps": 256, "n_steps": 64, "n_epochs": 1, "batch_size": 32, "hidden": 8, "episode_days": 60}, 1e-5, ("gymnasium", "stable_baselines3")),
}
