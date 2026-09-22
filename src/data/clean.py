"""Stage 1 - cleaning (spec §10-§11, Ch. 7 §7.5-§7.6).

What this module does *not* do is as important as what it does. It does not
winsorise returns, it does not drop outliers, and it does not impute prices
that were never traded. What it does is:

1. **Preserve both price series.** ``close`` (exchange close) and ``adj_close``
   (split- and distribution-adjusted) are both carried through, together with
   the adjustment ratio, so that a mechanical price change caused by a
   distribution is never read as an investment loss (spec §10).

2. **Apply an explicit missing-data policy.** Leading NaNs (pre-inception) are
   never filled -- the asset simply is not investable yet. Short interior gaps
   are forward filled up to a configured limit and the fill is recorded in a
   mask, so every downstream consumer can exclude filled days. Longer runs are
   left missing and reported (spec §11).

3. **Record everything it did** in a ``CleaningResult`` so the data-quality
   report can state exactly how many values were filled, per ticker.

Alternative imputation methods from Ch. 7 §7.6 (linear interpolation,
cross-sectional regression, bootstrap, KNN) are implemented in
``compare_imputation_methods`` for *comparison on artificially masked data*.
They are deliberately not used in the production path for prices: on a daily
ETF panel the honest answer to "why is this price missing?" is almost always
"the fund did not trade", and inventing a price there manufactures a return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .validation import universe_calendar

LOGGER = get_logger(__name__)

PRICE_FIELDS = ("open", "high", "low", "close", "adj_close")


@dataclass
class CleaningResult:
    """Cleaned panel plus a full account of what was changed."""

    prices: pd.DataFrame                      # adjusted close, wide (date x ticker)
    close: pd.DataFrame                       # unadjusted close, wide
    adjustment_ratio: pd.DataFrame            # adj_close / close, wide
    volume: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    open_: pd.DataFrame
    filled_mask: pd.DataFrame                 # True where a value was forward filled
    investable: pd.DataFrame                  # True where the asset is tradable
    calendar: pd.DatetimeIndex
    actions: pd.DataFrame = field(default_factory=pd.DataFrame)   # detected corporate actions
    log: dict[str, dict] = field(default_factory=dict)

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    def log_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log).T.sort_index()

    def write(self, directory: str | Path) -> dict[str, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "prices_adjusted": self.prices,
            "prices_close": self.close,
            "adjustment_ratio": self.adjustment_ratio,
            "volume": self.volume,
            "high": self.high,
            "low": self.low,
            "open": self.open_,
            "filled_mask": self.filled_mask,
            "investable": self.investable,
        }
        written: dict[str, Path] = {}
        for name, frame in payload.items():
            path = directory / f"{name}.csv"
            frame.to_csv(path, index_label="date")
            written[name] = path
        if not self.actions.empty:
            path = directory / "corporate_actions.csv"
            self.actions.to_csv(path, index=False)
            written["corporate_actions"] = path
        path = directory / "cleaning_log.csv"
        self.log_frame().to_csv(path, index_label="ticker")
        written["cleaning_log"] = path
        return written


# ---------------------------------------------------------------------------
# Corporate actions
# ---------------------------------------------------------------------------
def adjustment_ratio(close: pd.Series, adj_close: pd.Series) -> pd.Series:
    """adj_close / close. Constant between corporate actions."""
    ratio = (adj_close / close).replace([np.inf, -np.inf], np.nan)
    return ratio


def detect_corporate_actions(close: pd.Series, adj_close: pd.Series, threshold: float = 0.001,
                             split_threshold: float = 0.20) -> pd.DataFrame:
    """Infer corporate actions from the two price series.

    A change in the adjustment ratio means cash was distributed or the share
    count changed. Distinguishing the two matters because a split is
    price-neutral in the raw series too, whereas a distribution is not::

        raw close jumps AND ratio jumps  -> split-like
        raw close smooth, ratio jumps    -> distribution-like
    """
    ratio = adjustment_ratio(close, adj_close)
    ratio_change = ratio.pct_change()
    raw_change = close.pct_change()
    flagged = ratio_change.abs() > threshold
    rows = []
    for stamp in ratio.index[flagged.fillna(False)]:
        rc, pc = float(ratio_change.loc[stamp]), float(raw_change.loc[stamp])
        kind = "split_like" if abs(pc) > split_threshold else "distribution_like"
        implied_yield = -rc if kind == "distribution_like" else float("nan")
        rows.append(
            {
                "date": stamp,
                "kind": kind,
                "ratio_change": rc,
                "raw_price_change": pc,
                "implied_distribution_yield": implied_yield,
            }
        )
    return pd.DataFrame(rows)


def total_return_from_adjusted(adj_close: pd.Series) -> pd.Series:
    """Total return series. Uses adjusted prices, i.e. dividends reinvested."""
    return adj_close.pct_change()


def price_return_from_close(close: pd.Series) -> pd.Series:
    """Price-only return. Differs from the total return by the dividend yield."""
    return close.pct_change()


# ---------------------------------------------------------------------------
# Missing data (Ch. 7 §7.6)
# ---------------------------------------------------------------------------
def classify_missing(series: pd.Series) -> pd.Series:
    """Label every missing value by *reason*, which drives the policy.

    Returns a string series with values in
    ``{"present", "pre_inception", "post_delisting", "interior_gap"}``.
    """
    present = series.notna()
    labels = pd.Series("present", index=series.index, dtype=object)
    if not present.any():
        return pd.Series("pre_inception", index=series.index, dtype=object)
    first, last = present.idxmax(), present[::-1].idxmax()
    labels.loc[(series.index < first)] = "pre_inception"
    labels.loc[(series.index > last)] = "post_delisting"
    interior = (~present) & (series.index >= first) & (series.index <= last)
    labels.loc[interior] = "interior_gap"
    return labels


def limited_ffill(series: pd.Series, max_days: int = 3) -> tuple[pd.Series, pd.Series]:
    """Forward fill interior gaps up to ``max_days``; never fill the edges.

    Returns ``(filled_series, filled_mask)``. Runs longer than ``max_days``
    are left missing so that they show up in the report instead of quietly
    becoming a flat price (which would look like a zero return).
    """
    labels = classify_missing(series)
    interior = labels.eq("interior_gap")
    if not interior.any() or max_days <= 0:
        return series.copy(), pd.Series(False, index=series.index)

    filled = series.ffill(limit=int(max_days))
    # Only keep fills that land on interior gaps.
    mask = interior & series.isna() & filled.notna()
    out = series.copy()
    out[mask] = filled[mask]
    return out, mask


def clean_one(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    max_ffill_days: int = 3,
    action_threshold: float = 0.001,
) -> dict:
    """Clean a single ticker onto the universe calendar."""
    data = frame.copy()
    if "date" in data.columns:
        data = data.set_index("date")
    data.index = pd.DatetimeIndex(pd.to_datetime(data.index)).tz_localize(None).normalize()
    data = data[~data.index.duplicated(keep="first")].sort_index()

    for column in PRICE_FIELDS + ("volume",):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        else:
            data[column] = np.nan

    # Non-positive prices cannot be prices. They are set to NaN so that the
    # missing-data policy handles them explicitly rather than propagating a
    # nonsensical return; the original value stays in the raw file.
    invalidated = 0
    for column in PRICE_FIELDS:
        bad = data[column].le(0)
        invalidated += int(bad.sum())
        data.loc[bad, column] = np.nan

    # Reindex onto the universe calendar: rows the provider omitted become
    # explicit NaNs and are then subject to the fill policy.
    data = data.reindex(calendar)

    actions = detect_corporate_actions(data["close"], data["adj_close"], threshold=action_threshold)
    ratio = adjustment_ratio(data["close"], data["adj_close"])

    filled: dict[str, pd.Series] = {}
    masks: dict[str, pd.Series] = {}
    for column in PRICE_FIELDS:
        series, mask = limited_ffill(data[column], max_days=max_ffill_days)
        filled[column] = series
        masks[column] = mask

    labels = classify_missing(data["adj_close"])
    investable = filled["adj_close"].notna() & labels.ne("pre_inception")

    log = {
        "observations_raw": int(data["adj_close"].notna().sum()),
        "calendar_days": int(len(calendar)),
        "pre_inception_days": int(labels.eq("pre_inception").sum()),
        "post_delisting_days": int(labels.eq("post_delisting").sum()),
        "interior_gaps": int(labels.eq("interior_gap").sum()),
        "values_forward_filled": int(masks["adj_close"].sum()),
        "unfilled_interior_gaps": int((labels.eq("interior_gap") & ~masks["adj_close"]).sum()),
        "nonpositive_prices_invalidated": invalidated,
        "corporate_actions_detected": int(len(actions)),
        "distribution_like_actions": int((actions["kind"] == "distribution_like").sum()) if len(actions) else 0,
        "split_like_actions": int((actions["kind"] == "split_like").sum()) if len(actions) else 0,
        "investable_days": int(investable.sum()),
    }
    return {
        "adj_close": filled["adj_close"],
        "close": filled["close"],
        "open": filled["open"],
        "high": filled["high"],
        "low": filled["low"],
        "volume": data["volume"],
        "ratio": ratio,
        "filled_mask": masks["adj_close"],
        "investable": investable,
        "actions": actions,
        "log": log,
    }


def clean_panel(panel: dict[str, pd.DataFrame], config: dict | None = None) -> CleaningResult:
    """Clean every ticker and assemble wide frames on a common calendar."""
    config = config or {}
    max_ffill = int(config.get("max_ffill_days", 3))
    action_threshold = float(config.get("detect_action_threshold", 0.001))

    calendar = universe_calendar(panel)
    LOGGER.info("universe calendar: %d trading days %s .. %s",
                len(calendar), calendar.min().date(), calendar.max().date())

    columns: dict[str, dict[str, pd.Series]] = {
        key: {} for key in ("adj_close", "close", "open", "high", "low", "volume", "ratio", "filled_mask", "investable")
    }
    actions_frames = []
    logs: dict[str, dict] = {}

    for ticker, frame in panel.items():
        cleaned = clean_one(frame, calendar, max_ffill_days=max_ffill, action_threshold=action_threshold)
        for key in columns:
            columns[key][ticker] = cleaned[key]
        if len(cleaned["actions"]):
            actions = cleaned["actions"].copy()
            actions.insert(0, "ticker", ticker)
            actions_frames.append(actions)
        logs[ticker] = cleaned["log"]

    def wide(key: str, dtype=float) -> pd.DataFrame:
        out = pd.DataFrame(columns[key], index=calendar)
        out.index.name = "date"
        return out.astype(dtype)

    result = CleaningResult(
        prices=wide("adj_close"),
        close=wide("close"),
        adjustment_ratio=wide("ratio"),
        volume=wide("volume"),
        high=wide("high"),
        low=wide("low"),
        open_=wide("open"),
        filled_mask=wide("filled_mask", bool),
        investable=wide("investable", bool),
        calendar=calendar,
        actions=pd.concat(actions_frames, ignore_index=True) if actions_frames else pd.DataFrame(),
        log=logs,
    )
    LOGGER.info("cleaned %d tickers; %d values forward filled in total",
                len(result.tickers), int(result.filled_mask.to_numpy().sum()))
    return result


# ---------------------------------------------------------------------------
# Imputation comparison (Ch. 7 §7.6) -- research tool, not production path
# ---------------------------------------------------------------------------
def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) < 1e-15 or np.nanstd(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare_imputation_methods(
    prices: pd.DataFrame,
    n_masked: int = 200,
    seed: int = 11,
    methods: tuple[str, ...] = ("ffill", "linear_interpolate", "knn", "cross_sectional_regression"),
    knn_k: int = 5,
) -> pd.DataFrame:
    """Mask observed returns at random, impute them, and score the error.

    This is how the book's methods should be judged: not by whether they run,
    but by how much damage they do to the *return* series -- because returns,
    not prices, are the research input. Errors are reported in basis points.
    """
    rng = np.random.default_rng(seed)
    returns = np.log(prices).diff()
    complete = returns.dropna(how="any")
    if len(complete) < 300:
        return pd.DataFrame()

    rows, cols = complete.shape
    flat_positions = rng.choice(rows * cols, size=min(n_masked, rows * cols // 10), replace=False)
    row_idx, col_idx = np.unravel_index(flat_positions, (rows, cols))
    truth = complete.to_numpy()[row_idx, col_idx]

    masked = complete.copy()
    masked_values = masked.to_numpy(dtype=float).copy()
    masked_values[row_idx, col_idx] = np.nan
    masked = pd.DataFrame(masked_values, index=complete.index, columns=complete.columns)

    results = []
    for method in methods:
        if method == "ffill":
            imputed = masked.ffill()
        elif method == "linear_interpolate":
            imputed = masked.interpolate(method="linear", limit_direction="both")
        elif method == "zero":
            imputed = masked.fillna(0.0)
        elif method == "knn":
            from sklearn.impute import KNNImputer

            imputer = KNNImputer(n_neighbors=knn_k)
            imputed = pd.DataFrame(imputer.fit_transform(masked), index=masked.index, columns=masked.columns)
        elif method == "cross_sectional_regression":
            # Regress each asset's return on the equal-weighted cross-section
            # of the observed assets that day (a one-factor fill).
            market = masked.mean(axis=1, skipna=True)
            imputed = masked.copy()
            for column in masked.columns:
                observed = masked[column].notna() & market.notna()
                if observed.sum() < 100:
                    continue
                x = market[observed].to_numpy()
                y = masked.loc[observed, column].to_numpy()
                beta = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
                alpha = float(y.mean() - beta * x.mean())
                gaps = masked[column].isna() & market.notna()
                imputed.loc[gaps, column] = alpha + beta * market[gaps]
        else:
            raise ValueError(f"unknown imputation method '{method}'")

        estimate = imputed.to_numpy()[row_idx, col_idx]
        error = estimate - truth
        finite = np.isfinite(error)
        results.append(
            {
                "method": method,
                "n_imputed": int(finite.sum()),
                "mae_bps": float(np.nanmean(np.abs(error[finite])) * 1e4),
                "rmse_bps": float(np.sqrt(np.nanmean(error[finite] ** 2)) * 1e4),
                "bias_bps": float(np.nanmean(error[finite]) * 1e4),
                # A constant reconstruction (e.g. zero-fill) has no variance, so
                # the correlation is undefined rather than zero.
                "corr_with_truth": _safe_corr(estimate[finite], truth[finite]),
            }
        )
    return pd.DataFrame(results).set_index("method")
