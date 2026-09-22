"""Stage 1 - data acquisition with provenance (spec §7-§8, Ch. 7 §7.2-§7.4).

Design rules
------------
1. **Raw is immutable.** A raw file is written once. Re-downloading requires an
   explicit ``force=True``; otherwise the existing file is kept and reported.
   Everything reproducible is rebuilt from raw, never in place.
2. **Every download carries provenance.** provider, requested window, actual
   window, download timestamp, observation count, checksum. Without this you
   cannot answer "which data produced this number?" six months later.
3. **The provider's adjustment convention is recorded, not assumed** -- both
   ``close`` and ``adj_close`` are stored (Ch. 7 §7.5.1).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

RAW_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

# Yahoo's adjustment convention, recorded in metadata so the research report
# can state it rather than guess (spec §10).
PROVIDER_NOTES = {
    "yahoo": (
        "Yahoo Finance via yfinance. 'close' is the exchange closing price; "
        "'adj_close' is back-adjusted for splits and cash distributions "
        "(dividends reinvested at the ex-date close). Splits affect both "
        "series; distributions affect only adj_close. Adjustment is applied "
        "retroactively, so the adjusted history for a given date can change "
        "after a future distribution -- which is why the download date is part "
        "of the data version."
    )
}


@dataclass
class Provenance:
    """Provenance record for one downloaded series (spec §8)."""

    provider: str
    ticker: str
    requested_start: str
    requested_end: str
    actual_start: str | None
    actual_end: str | None
    download_time: str
    observation_count: int
    columns: list[str] = field(default_factory=list)
    file: str = ""
    sha256: str = ""
    provider_notes: str = ""
    attempts: int = 1
    status: str = "ok"
    message: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_yahoo(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance output to the canonical long format."""
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=RAW_COLUMNS)

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance returns (field, ticker); keep the requested ticker only.
        level = 1 if ticker in data.columns.get_level_values(1) else 0
        data = data.xs(ticker, axis=1, level=level)
    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    data = data.rename(columns={"adj_close": "adj_close", "adjclose": "adj_close"})

    if "adj_close" not in data.columns and "close" in data.columns:
        # auto_adjust=True was applied upstream; record the fact explicitly.
        data["adj_close"] = data["close"]

    data.index = pd.DatetimeIndex(pd.to_datetime(data.index)).tz_localize(None).normalize()
    data.index.name = "date"
    data = data.reset_index()
    data["ticker"] = ticker
    for column in RAW_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    return data.loc[:, RAW_COLUMNS].sort_values("date").reset_index(drop=True)


class YahooDownloader:
    """Download daily OHLCV + adjusted close, one file per ticker."""

    provider = "yahoo"

    def __init__(
        self,
        raw_dir: str | Path,
        metadata_dir: str | Path,
        max_retries: int = 4,
        backoff_seconds: float = 2.0,
    ):
        self.raw_dir = Path(raw_dir)
        self.metadata_dir = Path(metadata_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)

    # -- single ticket ----------------------------------------------------
    def raw_path(self, ticker: str) -> Path:
        return self.raw_dir / f"{ticker.upper()}.csv"

    def _fetch(self, ticker: str, start: str, end: str | None) -> tuple[pd.DataFrame, int, str]:
        """Fetch with bounded exponential backoff; network errors are retried."""
        import yfinance as yf

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                frame = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    threads=False,
                )
                normalised = _normalise_yahoo(frame, ticker)
                if len(normalised) == 0:
                    raise RuntimeError("provider returned an empty frame")
                return normalised, attempt, ""
            except Exception as exc:  # network / provider errors
                last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning("download %s attempt %d/%d failed: %s", ticker, attempt, self.max_retries, last_error)
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        return pd.DataFrame(columns=RAW_COLUMNS), self.max_retries, last_error

    def download_one(self, ticker: str, start: str, end: str | None, force: bool = False) -> Provenance:
        path = self.raw_path(ticker)
        requested_end = end or datetime.now(timezone.utc).date().isoformat()

        if path.exists() and not force:
            # Raw data is immutable: report what is on disk instead of refetching.
            existing = pd.read_csv(path, parse_dates=["date"])
            LOGGER.info("raw file for %s exists (%d rows) - keeping it (use force=True to refetch)", ticker, len(existing))
            return Provenance(
                provider=self.provider,
                ticker=ticker,
                requested_start=start,
                requested_end=requested_end,
                actual_start=str(existing["date"].min().date()) if len(existing) else None,
                actual_end=str(existing["date"].max().date()) if len(existing) else None,
                download_time=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                observation_count=int(len(existing)),
                columns=list(existing.columns),
                file=str(path.relative_to(self.raw_dir.parent.parent)),
                sha256=_sha256(path),
                provider_notes=PROVIDER_NOTES.get(self.provider, ""),
                status="cached",
            )

        frame, attempts, error = self._fetch(ticker, start, end)
        download_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if len(frame) == 0:
            LOGGER.error("no data for %s after %d attempts", ticker, attempts)
            return Provenance(
                provider=self.provider, ticker=ticker, requested_start=start, requested_end=requested_end,
                actual_start=None, actual_end=None, download_time=download_time, observation_count=0,
                provider_notes=PROVIDER_NOTES.get(self.provider, ""), attempts=attempts,
                status="failed", message=error,
            )

        frame.to_csv(path, index=False)
        provenance = Provenance(
            provider=self.provider,
            ticker=ticker,
            requested_start=start,
            requested_end=requested_end,
            actual_start=str(frame["date"].min().date()),
            actual_end=str(frame["date"].max().date()),
            download_time=download_time,
            observation_count=int(len(frame)),
            columns=list(frame.columns),
            file=str(path.relative_to(self.raw_dir.parent.parent)),
            sha256=_sha256(path),
            provider_notes=PROVIDER_NOTES.get(self.provider, ""),
            attempts=attempts,
            status="ok",
        )
        LOGGER.info(
            "%s: %d obs %s .. %s (attempt %d)",
            ticker, provenance.observation_count, provenance.actual_start, provenance.actual_end, attempts,
        )
        return provenance

    # -- universe ---------------------------------------------------------
    def download_universe(
        self, tickers: list[str], start: str, end: str | None = None, force: bool = False
    ) -> dict[str, Provenance]:
        records: dict[str, Provenance] = {}
        for ticker in tickers:
            records[ticker] = self.download_one(ticker, start, end, force=force)
        self.write_manifest(records, start, end)
        return records

    def write_manifest(self, records: dict[str, Provenance], start: str, end: str | None) -> Path:
        """Write the dataset manifest: the data version of this project."""
        payload = {
            "provider": self.provider,
            "provider_notes": PROVIDER_NOTES.get(self.provider, ""),
            "requested_start": start,
            "requested_end": end or datetime.now(timezone.utc).date().isoformat(),
            "manifest_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "series": {ticker: prov.to_json() for ticker, prov in sorted(records.items())},
        }
        combined = "".join(prov.sha256 for _, prov in sorted(records.items()))
        payload["data_version"] = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:12]
        path = self.metadata_dir / "manifest.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        LOGGER.info("manifest written: %s (data_version=%s)", path, payload["data_version"])
        return path


def load_manifest(metadata_dir: str | Path) -> dict:
    path = Path(metadata_dir) / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def data_version(metadata_dir: str | Path) -> str:
    return load_manifest(metadata_dir).get("data_version", "unversioned")
