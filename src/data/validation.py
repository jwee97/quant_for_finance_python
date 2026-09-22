"""Stage 1 - data validation (spec §9, Ch. 7 §7.5-§7.6).

The governing rule of this module is::

    flag anomaly  !=  delete anomaly

A 20% one-day move in an ETF is either a data error or the most informative
observation in the sample. Deleting it automatically destroys exactly the tail
behaviour the risk engine (Ch. 21) exists to measure. So every check here
*records* an issue with enough context to be investigated by a human, and the
cleaner then applies an explicit, configured policy.

Checks implemented
------------------
missing observations          - gaps against the universe trading calendar
duplicate dates               - repeated rows for one (ticker, date)
invalid dates                 - unparseable, future-dated, or weekend stamps
non-positive prices           - price <= 0 or NaN in a required field
OHLC inconsistencies          - low > high, close/open outside [low, high]
static-arbitrage violations   - Ch. 7 §7.5.3: internal no-arbitrage identities
suspicious price jumps        - |r| above the configured thresholds
adjustment-factor breaks      - adj_close/close ratio jumps => corporate action
stale prices                  - identical closes over a run of days
zero / missing volume         - liquidity or provider artefacts
differing inception dates     - survivorship / backfill exposure (§7.5.2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

SEVERITIES = ("info", "warning", "error")
REQUIRED_PRICE_FIELDS = ("open", "high", "low", "close", "adj_close")


@dataclass
class Issue:
    """One flagged observation or condition."""

    check: str
    ticker: str
    date: str | None
    severity: str
    detail: str
    value: float | None = None

    def as_row(self) -> dict:
        return {
            "check": self.check,
            "ticker": self.ticker,
            "date": self.date,
            "severity": self.severity,
            "value": self.value,
            "detail": self.detail,
        }


@dataclass
class ValidationReport:
    """Collection of issues plus per-ticker summary statistics."""

    issues: list[Issue] = field(default_factory=list)
    summary: dict[str, dict] = field(default_factory=dict)

    # -- mutation ---------------------------------------------------------
    def add(self, check: str, ticker: str, date, severity: str, detail: str, value=None) -> None:
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        stamp = None
        if date is not None and not (isinstance(date, float) and np.isnan(date)):
            stamp = str(pd.Timestamp(date).date())
        self.issues.append(
            Issue(check=check, ticker=ticker, date=stamp, severity=severity, detail=detail,
                  value=None if value is None else float(value))
        )

    def extend(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)
        self.summary.update(other.summary)

    # -- views ------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["check", "ticker", "date", "severity", "value", "detail"])
        return pd.DataFrame([issue.as_row() for issue in self.issues])

    def summary_frame(self) -> pd.DataFrame:
        if not self.summary:
            return pd.DataFrame()
        return pd.DataFrame(self.summary).T.sort_index()

    def counts(self) -> pd.DataFrame:
        frame = self.to_frame()
        if frame.empty:
            return pd.DataFrame(columns=["check", "severity", "n"])
        out = frame.groupby(["check", "severity"]).size().reset_index(name="n")
        return out.sort_values(["severity", "n"], ascending=[False, False]).reset_index(drop=True)

    @property
    def n_errors(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def blocking_tickers(self) -> list[str]:
        """Tickers with an error severe enough to exclude them from research."""
        blocking = {"insufficient_history", "empty_series", "no_price_column"}
        return sorted({i.ticker for i in self.issues if i.severity == "error" and i.check in blocking})


# ---------------------------------------------------------------------------
# Individual checks. Each takes a single-ticker frame indexed by date.
# ---------------------------------------------------------------------------
def check_duplicates(frame: pd.DataFrame, ticker: str, report: ValidationReport) -> None:
    duplicated = frame.index.duplicated(keep=False)
    for stamp in pd.DatetimeIndex(frame.index[duplicated]).unique():
        report.add("duplicate_date", ticker, stamp, "error", "more than one row for this date")


def check_invalid_dates(frame: pd.DataFrame, ticker: str, report: ValidationReport,
                        today: pd.Timestamp | None = None) -> None:
    index = pd.DatetimeIndex(frame.index)
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    for stamp in index[index > today]:
        report.add("future_date", ticker, stamp, "error", "observation dated in the future")
    weekend = index[index.dayofweek >= 5]
    for stamp in weekend:
        report.add("weekend_date", ticker, stamp, "warning", "observation on a Saturday/Sunday")
    if not index.is_monotonic_increasing:
        report.add("unsorted_index", ticker, None, "warning", "dates are not monotonically increasing")


def check_prices(frame: pd.DataFrame, ticker: str, report: ValidationReport) -> None:
    for column in REQUIRED_PRICE_FIELDS:
        if column not in frame.columns:
            report.add("no_price_column", ticker, None, "error", f"missing required column '{column}'")
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        for stamp in series.index[series.isna()]:
            report.add("missing_price", ticker, stamp, "warning", f"{column} is NaN")
        nonpositive = series[(series <= 0) & series.notna()]
        for stamp, value in nonpositive.items():
            report.add("nonpositive_price", ticker, stamp, "error", f"{column}={value:.6g} <= 0", value)


def check_ohlc(frame: pd.DataFrame, ticker: str, report: ValidationReport, tolerance: float = 1e-8) -> None:
    """Internal consistency of the OHLC bar (Ch. 7 §7.5.3, static arbitrage).

    The bar must satisfy ``low <= min(open, close) <= max(open, close) <= high``.
    A violation is a *self-contradictory* observation: it would allow a
    riskless intraday profit at the recorded prices, so it cannot be a true
    market bar and is a data error rather than a market event.
    """
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(frame.columns):
        return
    numeric = frame.loc[:, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    ok = numeric.notna().all(axis=1)
    numeric = numeric[ok]

    violations = {
        "low_above_high": numeric["low"] > numeric["high"] + tolerance,
        "open_outside_range": (numeric["open"] < numeric["low"] - tolerance)
        | (numeric["open"] > numeric["high"] + tolerance),
        "close_outside_range": (numeric["close"] < numeric["low"] - tolerance)
        | (numeric["close"] > numeric["high"] + tolerance),
    }
    for name, mask in violations.items():
        for stamp in numeric.index[mask]:
            row = numeric.loc[stamp]
            report.add(
                "ohlc_inconsistency", ticker, stamp, "error",
                f"{name}: o={row['open']:.4f} h={row['high']:.4f} l={row['low']:.4f} c={row['close']:.4f}",
            )


def check_jumps(frame: pd.DataFrame, ticker: str, report: ValidationReport,
                threshold: float = 0.20, extreme: float = 0.50, basis: str = "adj_close") -> None:
    """Flag large one-day moves for investigation -- never remove them."""
    if basis not in frame.columns:
        return
    prices = pd.to_numeric(frame[basis], errors="coerce")
    returns = prices.pct_change()
    flagged = returns[returns.abs() > threshold].dropna()
    for stamp, value in flagged.items():
        severity = "error" if abs(value) > extreme else "warning"
        report.add(
            "price_jump", ticker, stamp, severity,
            f"{basis} return {value:+.2%} exceeds {threshold:.0%} threshold "
            f"({'investigate as probable data error' if severity == 'error' else 'investigate, may be genuine'})",
            value,
        )


def check_calendar_gaps(frame: pd.DataFrame, ticker: str, report: ValidationReport, max_gap_days: int = 5) -> None:
    index = pd.DatetimeIndex(frame.index).sort_values()
    if len(index) < 2:
        return
    gaps = pd.Series(index[1:], index=index[:-1]) - pd.Series(index[:-1], index=index[:-1])
    for stamp, delta in gaps.items():
        days = delta.days
        if days > max_gap_days:
            report.add("trading_gap", ticker, stamp, "warning",
                       f"{days} calendar days to the next observation", float(days))


def check_missing_vs_calendar(frame: pd.DataFrame, ticker: str, calendar: pd.DatetimeIndex,
                              report: ValidationReport) -> pd.DatetimeIndex:
    """Dates on which the universe traded but this ticker has no row.

    Only dates at or after the ticker's own inception are counted: absence
    before inception is not missing data, it is non-existence (see §7.5.2).
    """
    index = pd.DatetimeIndex(frame.index)
    if len(index) == 0:
        return pd.DatetimeIndex([])
    live = calendar[(calendar >= index.min()) & (calendar <= index.max())]
    missing = live.difference(index)
    for stamp in missing:
        report.add("missing_observation", ticker, stamp, "warning",
                   "universe traded on this date but this series has no row")
    return missing


def check_stale_prices(frame: pd.DataFrame, ticker: str, report: ValidationReport, run_length: int = 5) -> None:
    if "close" not in frame.columns:
        return
    close = pd.to_numeric(frame["close"], errors="coerce")
    changed = close.ne(close.shift())
    run_id = changed.cumsum()
    runs = close.groupby(run_id).size()
    for group, length in runs[runs >= run_length].items():
        stamps = close.index[run_id == group]
        report.add("stale_price", ticker, stamps[-1], "warning",
                   f"close unchanged at {close.loc[stamps[-1]]:.4f} for {int(length)} consecutive observations",
                   float(length))


def check_volume(frame: pd.DataFrame, ticker: str, report: ValidationReport, min_volume: float = 0.0) -> None:
    if "volume" not in frame.columns:
        return
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    for stamp in volume.index[volume.isna()]:
        report.add("missing_volume", ticker, stamp, "info", "volume is NaN")
    zero = volume[(volume <= min_volume) & volume.notna()]
    for stamp, value in zero.items():
        report.add("zero_volume", ticker, stamp, "warning",
                   f"volume={value:.0f} <= {min_volume:.0f}; price may be stale or a provider artefact", value)


def check_adjustment_factor(frame: pd.DataFrame, ticker: str, report: ValidationReport,
                            threshold: float = 0.001) -> pd.Series:
    """Track the adj_close/close ratio.

    A change in this ratio is a corporate action (distribution or split), not
    an investment loss (spec §10). The ratio is returned so the cleaner can
    store it and so the report can count actions per ticker.
    """
    if not {"close", "adj_close"}.issubset(frame.columns):
        return pd.Series(dtype=float)
    close = pd.to_numeric(frame["close"], errors="coerce")
    adjusted = pd.to_numeric(frame["adj_close"], errors="coerce")
    ratio = (adjusted / close).replace([np.inf, -np.inf], np.nan)
    change = ratio.pct_change().abs()
    events = change[change > threshold].dropna()
    for stamp, value in events.items():
        report.add("adjustment_break", ticker, stamp, "info",
                   f"adj_close/close ratio moved {value:+.3%}: corporate action, not a return", value)
    # A ratio that rises through time would mean adjusted prices above raw
    # prices for a dividend payer, which contradicts back-adjustment.
    clean_ratio = ratio.dropna()
    if len(clean_ratio) > 1 and clean_ratio.iloc[-1] < clean_ratio.iloc[0] - 1e-9:
        report.add("adjustment_direction", ticker, None, "warning",
                   "adj_close/close ratio decreases through time; check provider convention")
    return ratio


def check_history_length(frame: pd.DataFrame, ticker: str, report: ValidationReport, minimum: int = 500) -> None:
    if len(frame) == 0:
        report.add("empty_series", ticker, None, "error", "no observations at all")
    elif len(frame) < minimum:
        report.add("insufficient_history", ticker, None, "error",
                   f"{len(frame)} observations < required {minimum}", float(len(frame)))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def universe_calendar(panel: dict[str, pd.DataFrame], min_tickers: int = 2) -> pd.DatetimeIndex:
    """Trading calendar implied by the data itself.

    A date is a universe trading day when at least ``min_tickers`` series have
    an observation on it. This avoids treating a single provider glitch as a
    market holiday and avoids depending on an exchange-calendar package.
    """
    counts: dict[pd.Timestamp, int] = {}
    for frame in panel.values():
        for stamp in pd.DatetimeIndex(frame.index):
            counts[stamp] = counts.get(stamp, 0) + 1
    if not counts:
        return pd.DatetimeIndex([])
    kept = [stamp for stamp, n in counts.items() if n >= min(min_tickers, len(panel))]
    return pd.DatetimeIndex(sorted(kept))


def validate_panel(panel: dict[str, pd.DataFrame], config: dict | None = None) -> ValidationReport:
    """Run every check over a ``{ticker: frame}`` panel indexed by date."""
    config = config or {}
    report = ValidationReport()
    calendar = universe_calendar(panel)

    for ticker, raw in panel.items():
        frame = raw.copy()
        if "date" in frame.columns:
            frame = frame.set_index("date")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
        frame = frame.sort_index()

        check_history_length(frame, ticker, report, int(config.get("min_observations", 500)))
        check_duplicates(frame, ticker, report)
        check_invalid_dates(frame, ticker, report)
        check_prices(frame, ticker, report)
        check_ohlc(frame, ticker, report, float(config.get("ohlc_tolerance", 1e-8)))
        check_jumps(frame, ticker, report,
                    float(config.get("jump_threshold", 0.20)),
                    float(config.get("extreme_jump_threshold", 0.50)))
        check_calendar_gaps(frame, ticker, report, int(config.get("max_gap_days", 5)))
        missing = check_missing_vs_calendar(frame, ticker, calendar, report)
        check_stale_prices(frame, ticker, report, int(config.get("stale_price_run", 5)))
        check_volume(frame, ticker, report, float(config.get("min_volume", 0.0)))
        ratio = check_adjustment_factor(frame, ticker, report)

        index = pd.DatetimeIndex(frame.index)
        adjusted = pd.to_numeric(frame.get("adj_close", pd.Series(dtype=float)), errors="coerce")
        returns = adjusted.pct_change().dropna()
        report.summary[ticker] = {
            "observations": int(len(frame)),
            "first_date": None if len(index) == 0 else str(index.min().date()),
            "last_date": None if len(index) == 0 else str(index.max().date()),
            "missing_vs_calendar": int(len(missing)),
            "duplicate_dates": int(pd.Index(index).duplicated().sum()),
            "nan_adj_close": int(adjusted.isna().sum()),
            "zero_volume_days": int((pd.to_numeric(frame.get("volume", pd.Series(dtype=float)),
                                                   errors="coerce") <= 0).sum()),
            "jumps_gt_20pct": int((returns.abs() > 0.20).sum()),
            "max_abs_daily_return": float(returns.abs().max()) if len(returns) else float("nan"),
            "corporate_actions_detected": int((ratio.pct_change().abs() > 0.001).sum()) if len(ratio) else 0,
            "total_adjustment_ratio": float(ratio.dropna().iloc[0] / ratio.dropna().iloc[-1])
            if len(ratio.dropna()) > 1 else float("nan"),
        }

    # Survivorship / backfill exposure: differing inception dates (§7.5.2).
    inceptions = {t: s["first_date"] for t, s in report.summary.items() if s["first_date"]}
    if inceptions:
        latest = max(inceptions.values())
        for ticker, first in sorted(inceptions.items()):
            if first != min(inceptions.values()):
                report.add("late_inception", ticker, first, "info",
                           f"series starts {first}, after the earliest universe start "
                           f"{min(inceptions.values())}; asset is not investable before this date")
        LOGGER.info("common history for all tickers starts %s", latest)

    LOGGER.info("validation complete: %d issues (%d errors, %d warnings)",
                len(report.issues), report.n_errors, report.n_warnings)
    return report


def write_report(report: ValidationReport, directory: str | Path, prefix: str = "data_quality") -> dict[str, Path]:
    """Persist the issue log, the per-ticker summary and the check counts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "issues": directory / f"{prefix}_issues.csv",
        "summary": directory / f"{prefix}_summary.csv",
        "counts": directory / f"{prefix}_counts.csv",
    }
    report.to_frame().to_csv(paths["issues"], index=False)
    report.summary_frame().to_csv(paths["summary"], index_label="ticker")
    report.counts().to_csv(paths["counts"], index=False)
    return paths


# ---------------------------------------------------------------------------
# Jump investigation (spec §9: "flag anomaly != delete anomaly")
# ---------------------------------------------------------------------------
def investigate_jumps(
    returns: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    asset_class: dict[str, str] | None = None,
    threshold: float = 0.20,
    peer_confirm_ratio: float = 0.25,
    volume_z_threshold: float = 2.0,
) -> pd.DataFrame:
    """Decide, from evidence, whether each flagged jump is market or error.

    A single large return tells you nothing on its own. Three pieces of
    corroborating evidence are available inside the panel itself:

    ``peer_move``       the same-day median return of the asset's peers. A real
                        macro shock moves the peer group; a bad tick does not.
    ``cross_sectional`` the same-day median of the whole universe.
    ``volume_z``        z-score of that day's volume against its trailing
                        60-day history. Genuine repricing trades heavy.

    The verdict is advisory and deliberately conservative: nothing is deleted.
    ``market_event_confirmed`` observations stay in the sample -- they are the
    tail the risk engine exists to measure.
    """
    asset_class = asset_class or {}
    rows = []
    universe_median = returns.median(axis=1)

    for ticker in returns.columns:
        series = returns[ticker].dropna()
        flagged = series[series.abs() > threshold]
        if flagged.empty:
            continue
        peers = [c for c in returns.columns
                 if c != ticker and asset_class.get(c) == asset_class.get(ticker, "__none__")]
        peer_median = returns[peers].median(axis=1) if peers else pd.Series(index=returns.index, dtype=float)

        vol_z = pd.Series(index=returns.index, dtype=float)
        if volume is not None and ticker in volume.columns:
            v = pd.to_numeric(volume[ticker], errors="coerce")
            roll_mean = v.rolling(60, min_periods=20).mean()
            roll_std = v.rolling(60, min_periods=20).std()
            vol_z = (v - roll_mean) / roll_std.replace(0.0, np.nan)

        for stamp, value in flagged.items():
            peer = float(peer_median.get(stamp, np.nan))
            cross = float(universe_median.get(stamp, np.nan))
            zscore = float(vol_z.get(stamp, np.nan))
            same_sign_peer = np.isfinite(peer) and np.sign(peer) == np.sign(value) \
                and abs(peer) >= peer_confirm_ratio * abs(value)
            same_sign_cross = np.isfinite(cross) and np.sign(cross) == np.sign(value)
            heavy_volume = np.isfinite(zscore) and zscore > volume_z_threshold

            if same_sign_peer or (same_sign_cross and heavy_volume):
                verdict, action = "market_event_confirmed", "retain"
            elif heavy_volume:
                verdict, action = "likely_market_event", "retain"
            elif not peers:
                verdict, action = "no_peer_available", "retain_and_review"
            else:
                verdict, action = "unconfirmed_by_peers", "review_before_use"

            rows.append(
                {
                    "ticker": ticker,
                    "date": pd.Timestamp(stamp).date().isoformat(),
                    "return": float(value),
                    "peer_median_return": peer,
                    "universe_median_return": cross,
                    "volume_z": zscore,
                    "peer_confirms": bool(same_sign_peer),
                    "heavy_volume": bool(heavy_volume),
                    "verdict": verdict,
                    "action": action,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("return", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return out
