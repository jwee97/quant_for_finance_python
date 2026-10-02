"""Macro data with point-in-time availability (Generation 2, Priority 5).

The central fact about macro data is that a number has two dates: the
*reference* date it describes and the *availability* date on which anyone could
first have known it. June CPI is stamped 2022-06-01 and published on
2022-07-13. A feature built from the reference date uses a number six weeks
before it existed, and a backtest on it measures how well a strategy can read
the future.

So every series carries an explicit publication lag, and ``asof_panel`` builds
the daily panel from availability dates. For each trading date it holds the
most recent value that had actually been published.

Revisions are the second-order problem. FRED serves the latest vintage, so
seasonally adjusted series (unemployment) carry a small look-ahead from
revision that no lag removes. The series are chosen to limit this, CPI being
the not-seasonally-adjusted index which is never revised, and the remaining
exposure is stated in the report instead of being ignored.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"


@dataclass(frozen=True)
class MacroSeriesSpec:
    """One macro series and, crucially, when it becomes known."""

    id: str
    source: str                      # "fred" | "yahoo"
    frequency: str                   # "daily" | "monthly"
    release_lag_days: int
    symbol: str = ""                 # yahoo ticker when it differs from the id
    feature: str = ""

    @classmethod
    def from_config(cls, node: dict) -> "MacroSeriesSpec":
        return cls(
            id=str(node["id"]), source=str(node["source"]),
            frequency=str(node["frequency"]),
            release_lag_days=int(node["release_lag_days"]),
            symbol=str(node.get("symbol", "")), feature=str(node.get("feature", "")),
        )


def load_specs(config) -> list[MacroSeriesSpec]:
    return [MacroSeriesSpec.from_config(n) for n in (config.get("macro.series", []) or [])]


# ---------------------------------------------------------------------------
# Download, with provenance
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MacroDownloader:
    """Download macro series once, keep them immutable, record provenance.

    Same contract as the ETF downloader: raw files are written once and never
    overwritten without ``force=True``, and a manifest records what was fetched,
    when, from where, with a checksum. The macro manifest is separate from the
    ETF one on purpose, so adding macro data cannot change the ETF
    ``data_version`` that every Generation 1 result is stamped with.
    """

    def __init__(self, raw_dir: str | Path, metadata_dir: str | Path,
                 max_retries: int = 4, backoff_seconds: float = 2.0):
        self.raw_dir = Path(raw_dir)
        self.metadata_dir = Path(metadata_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = int(max_retries)
        self.backoff = float(backoff_seconds)

    def raw_path(self, spec: MacroSeriesSpec) -> Path:
        return self.raw_dir / f"{spec.id}.csv"

    def _fetch_fred(self, spec: MacroSeriesSpec) -> pd.DataFrame:
        frame = pd.read_csv(FRED_URL.format(id=spec.id), na_values=[".", ""])
        frame.columns = ["date", "value"]
        frame["date"] = pd.to_datetime(frame["date"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        return frame.dropna(subset=["value"])

    def _fetch_yahoo(self, spec: MacroSeriesSpec) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(spec.symbol or spec.id, start="1985-01-01", auto_adjust=False,
                          progress=False, threads=False)
        if raw is None or len(raw) == 0:
            raise RuntimeError("provider returned an empty frame")
        close = raw["Close"]
        close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
        frame = close.rename("value").to_frame()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
        frame.index.name = "date"
        return frame.reset_index().dropna(subset=["value"])

    def download_one(self, spec: MacroSeriesSpec, force: bool = False) -> dict:
        path = self.raw_path(spec)
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        status, message, attempts = "ok", "", 0

        if path.exists() and not force:
            frame = pd.read_csv(path, parse_dates=["date"])
            status = "cached"
            download_time = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        else:
            frame, download_time = None, started
            for attempts in range(1, self.max_retries + 1):
                try:
                    frame = self._fetch_fred(spec) if spec.source == "fred" else self._fetch_yahoo(spec)
                    break
                except Exception as exc:                       # network / provider errors
                    message = f"{type(exc).__name__}: {exc}"
                    LOGGER.warning("macro %s attempt %d/%d failed: %s", spec.id, attempts,
                                   self.max_retries, message)
                    if attempts < self.max_retries:
                        time.sleep(self.backoff * 2 ** (attempts - 1))
            if frame is None or len(frame) == 0:
                return {"id": spec.id, "status": "failed", "message": message,
                        "attempts": attempts, "observation_count": 0}
            frame.sort_values("date").to_csv(path, index=False)

        return {
            **asdict(spec),
            "status": status, "message": message, "attempts": attempts,
            "download_time": download_time,
            "actual_start": str(frame["date"].min().date()),
            "actual_end": str(frame["date"].max().date()),
            "observation_count": int(len(frame)),
            "file": str(path.name), "sha256": _sha256(path),
        }

    def download_all(self, specs: list[MacroSeriesSpec], force: bool = False) -> dict:
        records = {spec.id: self.download_one(spec, force) for spec in specs}
        combined = "".join(r.get("sha256", "") for _, r in sorted(records.items()))
        manifest = {
            "manifest_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_version": hashlib.sha256(combined.encode()).hexdigest()[:12],
            "note": ("FRED serves the latest vintage, not the as-first-published value. "
                     "Seasonally adjusted series therefore carry a small revision look-ahead "
                     "that publication lags do not remove."),
            "series": records,
        }
        (self.metadata_dir / "macro_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest


def macro_data_version(metadata_dir: str | Path) -> str:
    path = Path(metadata_dir) / "macro_manifest.json"
    return json.loads(path.read_text())["data_version"] if path.exists() else "unversioned"


def load_macro_raw(raw_dir: str | Path, specs: list[MacroSeriesSpec]) -> dict[str, pd.Series]:
    """Raw series indexed by REFERENCE date."""
    out: dict[str, pd.Series] = {}
    for spec in specs:
        path = Path(raw_dir) / f"{spec.id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"macro series {spec.id} missing: run stage 15 first")
        frame = pd.read_csv(path, parse_dates=["date"])
        out[spec.id] = frame.set_index("date")["value"].sort_index()
    return out


# ---------------------------------------------------------------------------
# Point-in-time availability
# ---------------------------------------------------------------------------
def availability_index(reference: pd.DatetimeIndex, spec: MacroSeriesSpec) -> pd.DatetimeIndex:
    """The date each observation first became known.

    Daily series are shifted by business days (a FRED daily value for day d
    appears on the next business day). Monthly series are shifted by calendar
    days from the date stamp, since the stamp is the first of the reference
    month and the lag is measured from it.
    """
    lag = int(spec.release_lag_days)
    if lag == 0:
        return pd.DatetimeIndex(reference)
    if spec.frequency == "daily":
        return pd.DatetimeIndex(reference) + pd.offsets.BDay(lag)
    return pd.DatetimeIndex(reference) + pd.Timedelta(days=lag)


def asof_series(series: pd.Series, spec: MacroSeriesSpec, calendar: pd.DatetimeIndex) -> pd.Series:
    """The latest PUBLISHED value as of each calendar date.

    Re-indexes the series by availability date, then carries each value
    forward until the next one is published. A value is never visible before
    its availability date: that is the invariant the tests pin down.
    """
    available = availability_index(series.index, spec)
    published = pd.Series(series.to_numpy(), index=available).sort_index()
    published = published[~published.index.duplicated(keep="last")]
    union = published.index.union(pd.DatetimeIndex(calendar)).sort_values()
    return published.reindex(union).ffill().reindex(pd.DatetimeIndex(calendar))


def asof_panel(series: dict[str, pd.Series], specs: list[MacroSeriesSpec],
               calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily panel of what was known on each trading date."""
    by_id = {s.id: s for s in specs}
    return pd.DataFrame({k: asof_series(v, by_id[k], calendar) for k, v in series.items()},
                        index=pd.DatetimeIndex(calendar))


def staleness(series: dict[str, pd.Series], specs: list[MacroSeriesSpec],
              calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Days since the value held on each date was published (a diagnostic)."""
    by_id = {s.id: s for s in specs}
    out = {}
    for key, values in series.items():
        available = availability_index(values.index, by_id[key])
        stamp = pd.Series(available, index=available).sort_index()
        stamp = stamp[~stamp.index.duplicated(keep="last")]
        union = stamp.index.union(pd.DatetimeIndex(calendar)).sort_values()
        last = stamp.reindex(union).ffill().reindex(pd.DatetimeIndex(calendar))
        out[key] = (pd.DatetimeIndex(calendar) - pd.DatetimeIndex(last)).days
    return pd.DataFrame(out, index=pd.DatetimeIndex(calendar))
