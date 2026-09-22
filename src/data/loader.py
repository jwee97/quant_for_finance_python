"""Stage 1 - loading (spec §7).

Two entry points:

``load_raw_panel``    reads the immutable per-ticker raw CSVs.
``MarketData``        the object every downstream stage consumes: wide frames
                      on a single calendar, plus the investability mask and the
                      provenance/data version that produced them.

Nothing downstream is allowed to read ``data/raw`` directly. That keeps one
definition of "the cleaned dataset" in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.dates import slice_dates
from ..utils.logging import get_logger
from .clean import CleaningResult, clean_panel
from .download import data_version, load_manifest
from .validation import ValidationReport, validate_panel

LOGGER = get_logger(__name__)

WIDE_FILES = {
    "prices": "prices_adjusted.csv",
    "close": "prices_close.csv",
    "adjustment_ratio": "adjustment_ratio.csv",
    "volume": "volume.csv",
    "high": "high.csv",
    "low": "low.csv",
    "open_": "open.csv",
    "filled_mask": "filled_mask.csv",
    "investable": "investable.csv",
}


def load_raw_panel(raw_dir: str | Path, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Read raw CSVs into ``{ticker: frame}`` indexed by date."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.csv"))
    panel: dict[str, pd.DataFrame] = {}
    for path in files:
        ticker = path.stem.upper()
        if tickers is not None and ticker not in {t.upper() for t in tickers}:
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        panel[ticker] = frame.set_index("date").sort_index()
    if tickers is not None:
        missing = [t for t in tickers if t.upper() not in panel]
        if missing:
            LOGGER.warning("no raw file for: %s", ", ".join(missing))
        panel = {t: panel[t] for t in tickers if t in panel}
    LOGGER.info("loaded %d raw series from %s", len(panel), raw_dir)
    return panel


@dataclass
class MarketData:
    """The cleaned dataset. The single input contract for Stages 2-13."""

    prices: pd.DataFrame           # adjusted close (total-return basis)
    close: pd.DataFrame            # unadjusted close
    adjustment_ratio: pd.DataFrame
    volume: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    open_: pd.DataFrame
    filled_mask: pd.DataFrame
    investable: pd.DataFrame
    data_version: str = "unversioned"
    asset_class: dict[str, str] | None = None
    group: dict[str, str] | None = None

    # -- construction -----------------------------------------------------
    @classmethod
    def from_cleaning_result(cls, result: CleaningResult, version: str = "unversioned",
                             asset_class: dict | None = None, group: dict | None = None) -> "MarketData":
        return cls(
            prices=result.prices, close=result.close, adjustment_ratio=result.adjustment_ratio,
            volume=result.volume, high=result.high, low=result.low, open_=result.open_,
            filled_mask=result.filled_mask, investable=result.investable,
            data_version=version, asset_class=asset_class, group=group,
        )

    @classmethod
    def from_processed(cls, directory: str | Path, metadata_dir: str | Path | None = None,
                       asset_class: dict | None = None, group: dict | None = None) -> "MarketData":
        directory = Path(directory)
        frames = {}
        for attribute, filename in WIDE_FILES.items():
            path = directory / filename
            if not path.exists():
                raise FileNotFoundError(f"processed file missing: {path}. Run stage 1 first.")
            frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
            if attribute in {"filled_mask", "investable"}:
                frame = frame.astype(bool)
            frames[attribute] = frame
        version = data_version(metadata_dir) if metadata_dir is not None else "unversioned"
        return cls(**frames, data_version=version, asset_class=asset_class, group=group)

    @classmethod
    def build(cls, raw_dir: str | Path, tickers: list[str], data_config: dict,
              metadata_dir: str | Path | None = None, asset_class: dict | None = None,
              group: dict | None = None) -> tuple["MarketData", ValidationReport, CleaningResult]:
        """Raw -> validate -> clean, in one call."""
        panel = load_raw_panel(raw_dir, tickers)
        report = validate_panel(panel, data_config.get("validation", {}))
        blocking = report.blocking_tickers()
        if blocking:
            LOGGER.error("excluding tickers with blocking data errors: %s", ", ".join(blocking))
            panel = {t: f for t, f in panel.items() if t not in set(blocking)}
        cleaned = clean_panel(panel, data_config.get("missing_data", {}) | data_config.get("corporate_actions", {}))
        version = data_version(metadata_dir) if metadata_dir is not None else "unversioned"
        return cls.from_cleaning_result(cleaned, version, asset_class, group), report, cleaned

    # -- views ------------------------------------------------------------
    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.prices.index)

    def returns(self, kind: str = "simple", exclude_filled: bool = True) -> pd.DataFrame:
        """Total returns from adjusted prices.

        ``exclude_filled=True`` blanks returns whose price was forward filled,
        so a provider gap never enters the research set as a fabricated 0%.
        """
        if kind == "simple":
            out = self.prices.pct_change()
        elif kind == "log":
            out = np.log(self.prices).diff()
        else:
            raise ValueError("kind must be 'simple' or 'log'")
        if exclude_filled:
            touched = self.filled_mask | self.filled_mask.shift(1, fill_value=False)
            out = out.mask(touched)
        return out.where(self.investable)

    def price_returns(self) -> pd.DataFrame:
        """Price-only returns (no distributions), for the dividend comparison."""
        return self.close.pct_change().where(self.investable)

    def dividend_yield(self) -> pd.Series:
        """Annualised contribution of distributions, per asset.

        The gap between total and price returns. Reported in the data chapter
        as evidence that the corporate-action treatment matters.
        """
        total = self.returns().mean() * 252
        price = self.price_returns().mean() * 252
        return (total - price).rename("distribution_contribution")

    def slice(self, start=None, end=None) -> "MarketData":
        kwargs = {}
        for name in ("prices", "close", "adjustment_ratio", "volume", "high", "low", "open_",
                     "filled_mask", "investable"):
            kwargs[name] = slice_dates(getattr(self, name), start, end)
        return MarketData(**kwargs, data_version=self.data_version,
                          asset_class=self.asset_class, group=self.group)

    def subset(self, tickers: list[str]) -> "MarketData":
        keep = [t for t in tickers if t in self.tickers]
        kwargs = {}
        for name in ("prices", "close", "adjustment_ratio", "volume", "high", "low", "open_",
                     "filled_mask", "investable"):
            kwargs[name] = getattr(self, name).loc[:, keep]
        return MarketData(**kwargs, data_version=self.data_version,
                          asset_class=self.asset_class, group=self.group)

    def common_start(self, min_assets: int | None = None) -> pd.Timestamp:
        """First date on which the required number of assets is investable."""
        counts = self.investable.sum(axis=1)
        target = min_assets if min_assets is not None else self.investable.shape[1]
        eligible = counts[counts >= target]
        if eligible.empty:
            raise ValueError(f"no date has {target} investable assets")
        return pd.Timestamp(eligible.index[0])

    def inception_dates(self) -> pd.Series:
        out = {}
        for ticker in self.tickers:
            live = self.investable[ticker]
            out[ticker] = live.index[live.to_numpy().argmax()] if live.any() else pd.NaT
        return pd.Series(out, name="inception")

    def describe(self) -> pd.DataFrame:
        returns = self.returns()
        return pd.DataFrame(
            {
                "inception": self.inception_dates(),
                "observations": self.investable.sum(),
                "mean_ann": returns.mean() * 252,
                "vol_ann": returns.std() * np.sqrt(252),
                "filled_values": self.filled_mask.sum(),
            }
        )

    def manifest(self, metadata_dir: str | Path) -> dict:
        return load_manifest(metadata_dir)
