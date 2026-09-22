"""Data validation and cleaning (Ch. 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.clean import (
    adjustment_ratio,
    classify_missing,
    clean_panel,
    compare_imputation_methods,
    detect_corporate_actions,
    limited_ffill,
)
from src.data.validation import (
    investigate_jumps,
    universe_calendar,
    validate_panel,
)


def _panel(n: int = 400) -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(11)
    out = {}
    for ticker in ("AAA", "BBB"):
        close = 100.0 * (1.0 + rng.normal(0.0004, 0.01, n)).cumprod()
        out[ticker] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "adj_close": close * 0.98,
                "volume": np.full(n, 1e6),
            },
            index=index,
        )
    return out


def test_clean_panel_is_a_no_op_on_clean_data():
    panel = _panel()
    result = clean_panel(panel, {"max_ffill_days": 3})
    assert not result.filled_mask.to_numpy().any()
    assert result.investable.to_numpy().all()
    assert len(result.calendar) == 400


def test_nonpositive_prices_are_invalidated_not_propagated():
    panel = _panel()
    panel["AAA"].iloc[100, panel["AAA"].columns.get_loc("close")] = -5.0
    result = clean_panel(panel, {"max_ffill_days": 3})
    assert result.log["AAA"]["nonpositive_prices_invalidated"] >= 1
    assert (result.close["AAA"] > 0).all() or result.close["AAA"].isna().any()


def test_validation_flags_an_ohlc_inconsistency():
    panel = _panel()
    panel["AAA"].iloc[50, panel["AAA"].columns.get_loc("low")] = 1e6   # low above high
    report = validate_panel(panel, {})
    checks = report.to_frame()["check"].tolist()
    assert "ohlc_inconsistency" in checks
    assert report.n_errors > 0


def test_validation_flags_a_large_jump_without_removing_it():
    panel = _panel()
    column = panel["AAA"].columns.get_loc("adj_close")
    panel["AAA"].iloc[200, column] = panel["AAA"].iloc[199, column] * 1.5
    report = validate_panel(panel, {"jump_threshold": 0.20})
    frame = report.to_frame()
    assert "price_jump" in frame["check"].tolist()
    # The observation survives cleaning: flagged, not deleted.
    result = clean_panel(panel, {"max_ffill_days": 3})
    assert result.prices["AAA"].iloc[200] == pytest.approx(panel["AAA"].iloc[200, column])


def test_validation_flags_duplicate_dates():
    panel = _panel()
    panel["AAA"] = pd.concat([panel["AAA"], panel["AAA"].iloc[[10]]]).sort_index()
    report = validate_panel(panel, {})
    assert "duplicate_date" in report.to_frame()["check"].tolist()


def test_leading_gaps_are_never_filled():
    """Pre-inception is non-existence, not missing data."""
    series = pd.Series([np.nan, np.nan, 100.0, 101.0, np.nan, 103.0],
                       index=pd.bdate_range("2020-01-01", periods=6))
    filled, mask = limited_ffill(series, 3)
    assert np.isnan(filled.iloc[0]) and np.isnan(filled.iloc[1])
    assert filled.iloc[4] == 101.0        # an interior gap is filled
    assert bool(mask.iloc[4]) and not bool(mask.iloc[0])


def test_fill_limit_is_respected():
    values = [100.0] + [np.nan] * 6 + [107.0]
    series = pd.Series(values, index=pd.bdate_range("2020-01-01", periods=8))
    filled, mask = limited_ffill(series, 2)
    assert int(mask.sum()) == 2
    assert filled.isna().sum() == 4       # the rest stay missing and get reported


def test_missing_values_are_classified_by_reason():
    series = pd.Series([np.nan, 1.0, np.nan, 2.0, np.nan],
                       index=pd.bdate_range("2020-01-01", periods=5))
    labels = classify_missing(series)
    assert labels.iloc[0] == "pre_inception"
    assert labels.iloc[2] == "interior_gap"
    assert labels.iloc[4] == "post_delisting"


def test_adjustment_ratio_is_constant_without_corporate_actions():
    index = pd.bdate_range("2020-01-01", periods=50)
    close = pd.Series(np.linspace(100.0, 150.0, 50), index=index)
    ratio = adjustment_ratio(close, close * 0.95)
    assert ratio.std() == pytest.approx(0.0, abs=1e-12)
    assert detect_corporate_actions(close, close * 0.95).empty


def test_distribution_is_detected_and_classified():
    index = pd.bdate_range("2020-01-01", periods=50)
    close = pd.Series(np.full(50, 100.0), index=index)
    adjusted = close * 0.95
    adjusted.iloc[25:] = close.iloc[25:] * 0.94      # a distribution on day 25
    actions = detect_corporate_actions(close, adjusted)
    assert len(actions) == 1
    assert actions["kind"].iloc[0] == "distribution_like"


def test_split_is_distinguished_from_a_distribution():
    index = pd.bdate_range("2020-01-01", periods=50)
    close = pd.Series(np.full(50, 100.0), index=index)
    close.iloc[25:] = 50.0                            # 2-for-1 split
    adjusted = pd.Series(np.full(50, 50.0), index=index)
    actions = detect_corporate_actions(close, adjusted)
    assert len(actions) == 1
    assert actions["kind"].iloc[0] == "split_like"


def test_universe_calendar_requires_a_quorum_of_tickers():
    """A date is a trading day when enough series observe it.

    This is deliberate: one provider glitch should not create a market
    holiday. With a two-ticker panel the quorum is both of them, so the days
    before the second ticker lists are not universe trading days.
    """
    panel = _panel(100)
    panel["BBB"] = panel["BBB"].iloc[10:]
    calendar = universe_calendar(panel)
    assert len(calendar) == 90
    assert calendar.min() == panel["BBB"].index.min()


def test_universe_calendar_keeps_every_date_for_a_single_ticker():
    panel = {"AAA": _panel(100)["AAA"]}
    assert len(universe_calendar(panel)) == 100


def test_universe_calendar_keeps_dates_a_minority_misses():
    """With a real universe, one late-listing asset must not truncate history."""
    panel = _panel(100)
    panel["CCC"] = panel["AAA"].copy()
    panel["BBB"] = panel["BBB"].iloc[10:]
    assert len(universe_calendar(panel)) == 100


def test_jump_investigation_confirms_a_market_wide_move():
    index = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(rng.normal(0.0, 0.01, (300, 3)), index=index,
                           columns=["A", "B", "C"])
    returns.iloc[150] = [-0.30, -0.25, -0.22]        # everything falls together
    volume = pd.DataFrame(1e6, index=index, columns=returns.columns)
    volume.iloc[150] = 1e8                            # on huge volume
    frame = investigate_jumps(returns, volume, {c: "equity" for c in returns.columns}, 0.20)
    assert len(frame) >= 1
    assert frame["verdict"].iloc[0] == "market_event_confirmed"
    assert frame["action"].iloc[0] == "retain"


def test_cross_sectional_imputation_beats_time_series_fills(synthetic_prices):
    """The Ch. 7 methods, scored on returns rather than assumed."""
    scores = compare_imputation_methods(synthetic_prices, n_masked=200, seed=5,
                                        methods=("zero", "ffill", "knn"))
    assert scores.loc["knn", "rmse_bps"] < scores.loc["ffill", "rmse_bps"]


def test_returns_exclude_forward_filled_prices(synthetic_market):
    """A filled price must never enter the research set as a real return."""
    market = synthetic_market
    mask = market.filled_mask.copy()
    mask.iloc[100, 0] = True
    patched = type(market)(
        prices=market.prices, close=market.close, adjustment_ratio=market.adjustment_ratio,
        volume=market.volume, high=market.high, low=market.low, open_=market.open_,
        filled_mask=mask, investable=market.investable, data_version="t",
    )
    returns = patched.returns(exclude_filled=True)
    assert np.isnan(returns.iloc[100, 0])
    assert np.isnan(returns.iloc[101, 0])   # the day after is contaminated too
