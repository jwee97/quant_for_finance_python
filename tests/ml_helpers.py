"""Synthetic markets for the machine-learning tests: small, fast and with a known, plantable signal."""

from __future__ import annotations

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
