"""The data bundle every model receives, and how to build one from the platform's data or from your own.

A model sees a ``MarketBundle``: prices, returns, investability and, if available, high/low/volume and a panel of
macro levels already shifted by their publication lags. Nothing in a bundle is allowed to be known before its date:
that is checked for every registered model by ``validate.check_causality``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MarketBundle:
    prices: pd.DataFrame
    returns: pd.DataFrame
    investable: pd.DataFrame
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None
    macro: pd.DataFrame = field(default_factory=pd.DataFrame)
    asset_class: dict = field(default_factory=dict)
    group: dict = field(default_factory=dict)
    name: str = "bundle"
    market: object | None = None             # the underlying MarketData when there is one (the Generation 1-4 builders want it)

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.prices.index)

    @property
    def assets(self) -> list[str]:
        return list(self.prices.columns)

    def as_market(self):
        """The ``MarketData``-like view the Generation 1-4 portfolio builders expect (``prices``, ``returns()``, ``investable``)."""
        if self.market is not None:
            return self.market
        from types import SimpleNamespace

        digest = pd.util.hash_pandas_object(self.prices.tail(250).fillna(0.0), index=True).sum()
        return SimpleNamespace(prices=self.prices, returns=lambda: self.returns, investable=self.investable, high=self.high, low=self.low,
                               volume=self.volume, close=self.prices, asset_class=self.asset_class, group=self.group,
                               data_version=f"{self.name}-{self.prices.shape}-{digest}")

    def macro_series(self, name: str) -> pd.Series:
        if name not in self.macro.columns:
            raise KeyError(f"macro series '{name}' is not in this bundle; available: {list(self.macro.columns)}")
        return self.macro[name]

    def assets_in(self, *classes: str) -> list[str]:
        return [a for a in self.assets if self.asset_class.get(a) in classes or self.group.get(a) in classes]

    def perturbed_after(self, cutoff, scale: float = 2.5, seed: int = 0) -> "MarketBundle":
        """A copy whose data AFTER ``cutoff`` is replaced by noise: the input to the causality test."""
        rng = np.random.default_rng(seed)
        after = self.index > pd.Timestamp(cutoff)
        returns = self.returns.copy()
        noise = rng.normal(0.0, 0.02, size=(int(after.sum()), returns.shape[1]))
        returns.loc[after] = noise * scale
        base = self.prices.loc[self.index <= pd.Timestamp(cutoff)].iloc[-1]
        prices = self.prices.copy()
        prices.loc[after] = base.to_numpy() * (1.0 + returns.loc[after]).cumprod().to_numpy()
        high = self.high.copy() if self.high is not None else None
        low = self.low.copy() if self.low is not None else None
        volume = self.volume.copy() if self.volume is not None else None
        if high is not None:
            high.loc[after] = prices.loc[after] * 1.01
        if low is not None:
            low.loc[after] = prices.loc[after] * 0.99
        if volume is not None:
            volume.loc[after] = volume.loc[after] * rng.uniform(0.2, 5.0, size=(int(after.sum()), volume.shape[1]))
        macro = self.macro.copy()
        if not macro.empty:
            macro.loc[macro.index > pd.Timestamp(cutoff)] = rng.normal(size=(int((macro.index > pd.Timestamp(cutoff)).sum()), macro.shape[1])) * 10.0
        return MarketBundle(prices, returns, self.investable, high, low, volume, macro, self.asset_class, self.group, self.name + "+noise", None)


def bundle_from_market(market, macro: pd.DataFrame | None = None, name: str = "platform") -> MarketBundle:
    """From the platform's cleaned ``MarketData`` (plus an optional macro-levels panel)."""
    returns = market.returns()
    prices = market.prices
    macro = macro if macro is not None else pd.DataFrame(index=prices.index)
    return MarketBundle(prices=prices, returns=returns, investable=market.investable, high=getattr(market, "high", None), low=getattr(market, "low", None),
                        volume=getattr(market, "volume", None), macro=macro.reindex(prices.index), asset_class=dict(market.asset_class or {}),
                        group=dict(market.group or {}), name=name, market=market)


def bundle_from_prices(prices: pd.DataFrame, volume: pd.DataFrame | None = None, high: pd.DataFrame | None = None, low: pd.DataFrame | None = None,
                       asset_class: dict | None = None, macro: pd.DataFrame | None = None, name: str = "custom", min_history: int = 60) -> MarketBundle:
    """From YOUR prices: a wide frame (date index, one column per asset) of adjusted closes.

    An asset is investable from ``min_history`` observations after its first price and while it has a price. The data
    are not altered: missing prices stay missing (never filled), and returns are simple returns.
    """
    prices = prices.sort_index().astype(float)
    prices.index = pd.DatetimeIndex(prices.index)
    if prices.index.has_duplicates:
        raise ValueError("duplicate dates in the price frame")
    returns = prices.pct_change()
    seen = prices.notna().cumsum()
    investable = prices.notna() & (seen >= min_history)
    asset_class = asset_class or {c: "unknown" for c in prices.columns}
    return MarketBundle(prices, returns, investable, high, low, volume, macro if macro is not None else pd.DataFrame(index=prices.index),
                        asset_class, dict(asset_class), name, None)


def load_prices_csv(path: str | Path, date_column: str = "date") -> pd.DataFrame:
    """A wide CSV of prices: one date column and one column per asset."""
    frame = pd.read_csv(path, parse_dates=[date_column]).set_index(date_column).sort_index()
    return frame.apply(pd.to_numeric, errors="coerce")


def load_default_bundle(config, with_macro: bool = True, download: bool = False) -> MarketBundle:
    """The 15-ETF platform bundle with point-in-time macro levels (and the framework's extra FRED series)."""
    from ..data.loader import MarketData

    processed = config.path("processed")
    if (processed / "prices_adjusted.csv").exists():
        market = MarketData.from_processed(processed, config.path("metadata"), config.asset_class_map, config.group_map)
    else:
        market, _, cleaned = MarketData.build(config.path("raw"), config.tickers, config.data, config.path("metadata"), config.asset_class_map, config.group_map)
        cleaned.write(processed)
    macro = load_macro_levels(config, market.prices.index, download=download) if with_macro else None
    return bundle_from_market(market, macro)


def load_macro_levels(config, index: pd.DatetimeIndex, download: bool = False) -> pd.DataFrame:
    """Macro and extra FRED series as published-by-date levels on the trading calendar."""
    from ..data.macro import MacroDownloader, MacroSeriesSpec, asof_series, ensure_macro_raw, load_macro_raw
    from ..features.macro import macro_feature_panel

    specs, raw = ensure_macro_raw(config)
    levels = macro_feature_panel(raw, specs, index, config)["levels"]
    extra = [MacroSeriesSpec.from_config(s) for s in (config.get("framework.extra_series", []) or [])]
    if extra:
        raw_dir = config.root / "data" / "raw" / "framework"
        missing = [s for s in extra if not (raw_dir / f"{s.id}.csv").exists()]
        if missing:
            class _Downloader(MacroDownloader):
                manifest_name = "framework_manifest.json"
            _Downloader(raw_dir, config.path("metadata")).download_all(extra, force=False)
        series = load_macro_raw(raw_dir, extra)
        calendar = pd.DatetimeIndex(index)
        for spec in extra:
            levels[spec.id] = asof_series(series[spec.id], spec, calendar).reindex(index)
    return levels
