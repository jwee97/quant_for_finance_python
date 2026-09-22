"""Look-ahead detection (spec §44, §59).

The most important tests in the suite. They check two things: that an honest
strategy passes, and -- equally important -- that a deliberately broken one
fails. A leakage test that cannot fail proves nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.momentum import total_return_momentum
from src.features.volatility import rolling_volatility
from src.signals.transform import signal_to_positions
from src.validation.leakage import (
    check_no_lookahead,
    detect_suspicious_alignment,
    run_leakage_suite,
    scramble_future,
)
from src.validation.walk_forward import WalkForwardSplitter, purged_kfold_indices

SPLIT = "2018-06-29"


def honest_builder(market):
    signal = total_return_momentum(market.prices, 63, 1).where(market.investable)
    return signal_to_positions(signal, rolling_volatility(market.returns(), 63),
                               investable=market.investable)


def cheating_builder(market):
    """Uses tomorrow's return. The suite must catch this."""
    return signal_to_positions(market.returns().shift(-1),
                               rolling_volatility(market.returns(), 63),
                               investable=market.investable)


def subtly_cheating_builder(market):
    """A centred rolling mean: leaks only a few days of future, by accident."""
    signal = market.prices.rolling(21, center=True).mean() / market.prices - 1.0
    return signal_to_positions(signal.where(market.investable),
                               rolling_volatility(market.returns(), 63),
                               investable=market.investable)


def test_scrambling_leaves_the_past_untouched(synthetic_market):
    scrambled = scramble_future(synthetic_market, SPLIT)
    past = synthetic_market.prices.index <= pd.Timestamp(SPLIT)
    pd.testing.assert_frame_equal(synthetic_market.prices.loc[past], scrambled.prices.loc[past])


def test_scrambling_actually_changes_the_future(synthetic_market):
    scrambled = scramble_future(synthetic_market, SPLIT)
    future = synthetic_market.prices.index > pd.Timestamp(SPLIT)
    difference = (synthetic_market.prices.loc[future] - scrambled.prices.loc[future]).abs()
    assert float(difference.max().max()) > 0.0


def test_honest_strategy_passes(synthetic_market):
    result = check_no_lookahead(synthetic_market, honest_builder, SPLIT)
    assert result.passed, result.message
    assert result.n_differing_cells == 0
    assert result.future_weights_changed      # the test was not vacuous


def test_blatant_cheating_is_caught(synthetic_market):
    result = check_no_lookahead(synthetic_market, cheating_builder, SPLIT)
    assert not result.passed
    assert result.n_differing_cells > 0
    assert "LEAKAGE" in result.message


def test_subtle_cheating_is_caught(synthetic_market):
    """A centred window leaks only ten days -- the test must still catch it."""
    result = check_no_lookahead(synthetic_market, subtly_cheating_builder, SPLIT)
    assert not result.passed
    assert result.n_differing_cells > 0


def test_a_strategy_that_ignores_the_data_is_inconclusive_not_a_pass(synthetic_market):
    def constant(market):
        return pd.DataFrame(0.2, index=market.prices.index, columns=market.prices.columns)

    result = check_no_lookahead(synthetic_market, constant, SPLIT)
    assert not result.passed
    assert not result.future_weights_changed
    assert "INCONCLUSIVE" in result.message


def test_suite_reports_both_outcomes(synthetic_market):
    frame = run_leakage_suite(
        synthetic_market, {"honest": honest_builder, "cheating": cheating_builder},
        split_dates=(SPLIT,), modes=("shock", "reverse"),
    )
    assert bool(frame[frame["strategy"] == "honest"]["passed"].all())
    assert not bool(frame[frame["strategy"] == "cheating"]["passed"].any())


def test_alignment_detector_flags_a_future_correlated_signal(synthetic_returns):
    frame = detect_suspicious_alignment(synthetic_returns.shift(-1), synthetic_returns)
    assert bool(frame["leakage"].iloc[0])
    assert frame["worst_forward_correlation"].iloc[0] > 0.9


def test_alignment_detector_clears_a_momentum_signal(synthetic_prices, synthetic_returns):
    frame = detect_suspicious_alignment(total_return_momentum(synthetic_prices, 63, 1),
                                        synthetic_returns)
    assert not bool(frame["leakage"].iloc[0])


def test_walk_forward_folds_do_not_overlap(dates):
    splitter = WalkForwardSplitter(scheme="expanding", train_years_min=2, test_months=12,
                                   embargo_days=21)
    folds = splitter.split(dates)
    assert len(folds) >= 2
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end < later.test_start


def test_walk_forward_embargo_is_respected(dates):
    splitter = WalkForwardSplitter(scheme="expanding", train_years_min=2, test_months=12,
                                   embargo_days=30)
    for fold in splitter.split(dates):
        assert (fold.test_start - fold.train_end).days >= 30
        assert fold.train_end < fold.test_start


def test_purged_kfold_leaves_a_gap_around_each_test_block(dates):
    folds = purged_kfold_indices(dates, 5, embargo_days=21, horizon=21)
    assert folds
    for train_idx, test_idx in folds:
        assert not set(train_idx) & set(test_idx)
        gap = (dates[test_idx[0]] - dates[max(train_idx[train_idx < test_idx[0]], default=0)]).days \
            if (train_idx < test_idx[0]).any() else 999
        assert gap >= 21
