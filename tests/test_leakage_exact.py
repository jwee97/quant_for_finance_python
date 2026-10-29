"""The look-ahead detector, checked against what it must report, number by number.

A planted-bug audit (``experiments/plant_bugs.py``) found ``validation/leakage.py`` at 35% (19 of 55): the tests show that an honest strategy passes and a cheating one is caught, and nothing pins the
instrument itself: what each way of scrambling the future does, what the tolerance is, what is counted, what a failed run looks like in the suite, and where the correlation of a signal with returns
is called implausible. A detector that is wrong in either direction is worse than none, so each of those is stated here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import MarketData
from src.validation.leakage import (
    LeakageTestResult,
    check_no_lookahead,
    check_signal_causality,
    detect_suspicious_alignment,
    run_leakage_suite,
    scramble_future,
)

SPLIT = "2020-03-31"


def toy_market(n: int = 90, seed: int = 4) -> MarketData:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = ["A", "B", "C"]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, (n, 3)), axis=0)), index=idx, columns=cols)
    return MarketData(prices=prices, close=prices * 0.9, adjustment_ratio=pd.DataFrame(1.0, index=idx, columns=cols), volume=pd.DataFrame(1e6, index=idx, columns=cols), high=prices * 1.01,
                      low=prices * 0.99, open_=prices * 0.95, filled_mask=pd.DataFrame(False, index=idx, columns=cols), investable=pd.DataFrame(True, index=idx, columns=cols),
                      data_version="toy", asset_class={c: "equity" for c in cols}, group={c: "g" for c in cols})


def honest(market: MarketData) -> pd.DataFrame:
    return market.prices.pct_change().rolling(5).mean().shift(1)                              # yesterday's five-day mean return: knows nothing of tomorrow


# ------------------------------------------------------------------------------------------------------------------ scrambling the future
def test_every_way_of_scrambling_leaves_the_past_alone_and_changes_the_future():
    m = toy_market()
    split = pd.Timestamp(SPLIT)
    past = m.prices.index <= split
    for mode in ("shock", "reverse", "zero"):
        s = scramble_future(m, SPLIT, mode=mode)
        for field in ("prices", "close", "high", "low", "open_"):
            assert getattr(s, field).loc[past].equals(getattr(m, field).loc[past]), (mode, field, "the past, byte for byte")
        assert not s.prices.loc[~past].equals(m.prices.loc[~past]), mode
        assert s.data_version == f"toy_scrambled_{mode}" and s.asset_class is m.asset_class and s.investable.equals(m.investable) and s.volume.equals(m.volume)
    assert scramble_future(m, SPLIT).data_version == "toy_scrambled_shock", "a multiplicative shock unless told otherwise"


def test_the_shock_multiplies_the_future_by_a_random_path_of_a_chosen_size_that_a_seed_repeats():
    m = toy_market()
    future = m.prices.index > pd.Timestamp(SPLIT)
    s = scramble_future(m, SPLIT)
    rng = np.random.default_rng(99)
    block = m.prices.loc[future]
    path = pd.DataFrame(rng.lognormal(0.0, 0.5, size=block.shape), index=block.index, columns=block.columns).cumprod()
    assert s.prices.loc[future].to_numpy() == pytest.approx((block.to_numpy() * path.to_numpy())), "lognormal steps of size 0.5 by default, compounded, seed 99"
    assert scramble_future(m, SPLIT, seed=99).prices.equals(s.prices) and not scramble_future(m, SPLIT, seed=5).prices.equals(s.prices)
    mild = scramble_future(m, SPLIT, shock=0.01)
    assert (mild.prices.loc[future] / m.prices.loc[future] - 1.0).abs().to_numpy().max() < 0.5 < (s.prices.loc[future] / m.prices.loc[future] - 1.0).abs().to_numpy().max()
    scale = (s.prices / m.prices).fillna(1.0)
    assert s.close.to_numpy() == pytest.approx((m.close * scale).to_numpy()) and s.high.to_numpy() == pytest.approx((m.high * scale).to_numpy())
    assert s.low.to_numpy() == pytest.approx((m.low * scale).to_numpy()) and s.open_.to_numpy() == pytest.approx((m.open_ * scale).to_numpy()), "the other price fields move with the price"
    assert (s.close.loc[~future] == m.close.loc[~future]).all().all() and (s.close.loc[future] / m.close.loc[future]).to_numpy() == pytest.approx((s.prices.loc[future] / m.prices.loc[future]).to_numpy())


def test_reversing_and_flattening_the_future_and_the_two_ways_it_can_be_refused():
    m = toy_market()
    future = m.prices.index > pd.Timestamp(SPLIT)
    rev = scramble_future(m, SPLIT, mode="reverse")
    assert rev.prices.loc[future].to_numpy() == pytest.approx(m.prices.loc[future].to_numpy()[::-1]), "the same future prices in the opposite order"
    zero = scramble_future(m, SPLIT, mode="zero")
    level = float(m.prices.loc[~future].iloc[-1].mean())
    assert (zero.prices.loc[future].to_numpy() == level).all(), "a flat line at the mean of the last prices before the split"
    with pytest.raises(ValueError, match="unknown scramble mode 'wobble'"):
        scramble_future(m, SPLIT, mode="wobble")
    with pytest.raises(ValueError, match="split date 2020-12-31 leaves no future observations"):
        scramble_future(m, "2020-12-31")
    assert scramble_future(m, m.prices.index[-2]).prices.index.equals(m.prices.index), "one day of future is enough"
    with pytest.raises(ValueError, match="leaves no future"):
        scramble_future(m, m.prices.index[-1])


# ------------------------------------------------------------------------------------------------------------------ the test itself
def test_an_honest_book_passes_and_reports_how_much_it_checked():
    m = toy_market()
    result = check_no_lookahead(m, honest, SPLIT)
    past_days = int((m.prices.index <= pd.Timestamp(SPLIT)).sum())
    assert isinstance(result, LeakageTestResult) and result.passed and result.message == "PASS: no pre-split weight responded to future data"
    assert result.n_differing_cells == 0 and result.max_weight_difference == 0.0 and result.future_weights_changed and result.first_difference_date is None
    assert result.n_cells_checked == past_days * 3 and result.split_date == pd.Timestamp(SPLIT)
    row = result.to_row()
    assert row == {"split_date": SPLIT, "passed": True, "max_weight_difference": 0.0, "n_differing_cells": 0, "n_cells_checked": past_days * 3, "future_weights_changed": True, "first_difference_date": "",
                   "message": result.message}
    for mode in ("reverse", "zero"):
        assert check_no_lookahead(m, honest, SPLIT, mode=mode).passed, mode


def test_a_book_that_reads_the_future_is_caught_with_where_and_by_how_much():
    m = toy_market()

    def peeks(market):
        return market.prices.pct_change().shift(-1)                                          # tomorrow's return, today

    result = check_no_lookahead(m, peeks, SPLIT)
    split = pd.Timestamp(SPLIT)
    assert not result.passed and result.n_differing_cells > 0 and result.max_weight_difference > 1e-3
    assert result.first_difference_date == m.prices.index[m.prices.index <= split][-1], "only the last day before the split looks at the first day after it"
    assert result.message.startswith(f"LEAKAGE: {result.n_differing_cells} pre-split weights changed when future data changed (max ") and f"first at {result.first_difference_date.date()}" in result.message
    assert result.to_row()["first_difference_date"] == str(result.first_difference_date.date()) and result.to_row()["passed"] is False

    def peeks_far(market):
        return market.prices.pct_change().shift(-10)

    far = check_no_lookahead(m, peeks_far, SPLIT)
    assert far.n_differing_cells > result.n_differing_cells and far.first_difference_date == m.prices.index[m.prices.index <= split][-10], "ten days ahead is seen ten days before the split"


def test_the_tolerance_decides_what_is_a_change():
    m = toy_market()
    split = pd.Timestamp(SPLIT)

    def past_nudged(market):
        weights = honest(market)
        if market.data_version.endswith("shock"):                                               # in the scrambled world the past moves by 1.2e-10
            weights = weights.add(pd.Series(np.where(weights.index <= split, 1.2e-10, 0.0), index=weights.index), axis=0)
        return weights

    leak = check_no_lookahead(m, past_nudged, SPLIT)
    assert not leak.passed and leak.message.startswith("LEAKAGE"), "a change of 1.2e-10 is more than the default tolerance of 1e-10"
    loose = check_no_lookahead(m, past_nudged, SPLIT, tolerance=1.5e-10)
    assert loose.passed and loose.max_weight_difference == pytest.approx(1.2e-10, rel=1e-3) and loose.n_differing_cells == 0

    def future_nudged(market):
        flat = pd.DataFrame(0.2, index=market.prices.index, columns=market.prices.columns)
        if market.data_version.endswith("shock"):                                               # the future moves by 0.9e-10, the past not at all
            flat = flat.add(pd.Series(np.where(flat.index > split, 0.9e-10, 0.0), index=flat.index), axis=0)
        return flat

    inconclusive = check_no_lookahead(m, future_nudged, SPLIT)
    assert not inconclusive.passed and not inconclusive.future_weights_changed and inconclusive.message.startswith("INCONCLUSIVE: the perturbation did not change any post-split weight either")
    tighter = check_no_lookahead(m, future_nudged, SPLIT, tolerance=0.5e-10)
    assert tighter.passed and tighter.future_weights_changed, "a smaller tolerance sees the future move"


def test_a_book_that_ignores_the_data_is_inconclusive_and_one_with_nothing_before_the_split_cannot_be_judged():
    m = toy_market()
    flat = check_no_lookahead(m, lambda market: pd.DataFrame(0.2, index=market.prices.index, columns=market.prices.columns), SPLIT)
    assert not flat.passed and not flat.future_weights_changed and flat.n_differing_cells == 0 and flat.max_weight_difference == 0.0
    late = check_no_lookahead(m, honest, "2019-12-31")
    assert late.passed is False and late.message == "no pre-split weights to compare" and np.isnan(late.max_weight_difference) and late.n_cells_checked == 0 and late.future_weights_changed is False
    assert check_no_lookahead(m, honest, "2020-01-02").passed, "two days before the split are enough to compare"
    def vanishes(market):
        weights = honest(market)
        weights.loc[weights.index > pd.Timestamp(SPLIT)] = np.nan
        return weights

    missing = check_no_lookahead(m, vanishes, SPLIT)
    assert not missing.passed and not missing.future_weights_changed, "a book that vanishes after the split is not a book that responded"


def test_columns_and_dates_are_compared_where_both_books_have_them_and_a_missing_weight_is_a_zero():
    m = toy_market()

    def partial(market):
        w = honest(market)
        return w.drop(columns=["C"]) if market.data_version.endswith("reverse") else w

    result = check_no_lookahead(m, partial, SPLIT, mode="reverse")
    assert result.n_cells_checked == int((m.prices.index <= pd.Timestamp(SPLIT)).sum()) * 2, "only the columns both worlds have"
    assert check_no_lookahead(m, lambda market: honest(market).fillna(0.0), SPLIT).passed, "a missing weight and a weight of zero are the same book"


def test_the_signal_level_test_is_the_same_test_with_its_own_arguments(monkeypatch):
    m = toy_market()
    direct = check_no_lookahead(m, honest, SPLIT, 1e-10, 99)
    via = check_signal_causality(honest, m, SPLIT)
    assert via.passed == direct.passed and via.n_cells_checked == direct.n_cells_checked and via.message == direct.message
    seen = {}

    def spy(market, build, split, tolerance, seed):
        seen.update(tolerance=tolerance, seed=seed, split=split)
        return direct

    monkeypatch.setattr("src.validation.leakage.check_no_lookahead", spy)
    check_signal_causality(honest, m, SPLIT, tolerance=3e-9, seed=7)
    assert seen == {"tolerance": 3e-9, "seed": 7, "split": SPLIT}


# ------------------------------------------------------------------------------------------------------------------ the suite
def test_the_suite_runs_every_strategy_at_every_split_and_mode_and_reports_a_failed_run_as_a_row():
    m = toy_market(150)

    def broken(market):
        raise RuntimeError("no data")

    frame = run_leakage_suite(m, {"honest": honest, "broken": broken}, split_dates=("2020-03-31", "2020-05-29"), modes=("shock", "zero"), seed=3)
    assert len(frame) == 8 and list(frame.columns[:8]) == ["strategy", "mode", "split_date", "passed", "max_weight_difference", "n_differing_cells", "future_weights_changed", "message"]
    assert set(frame["strategy"]) == {"honest", "broken"} and set(frame["mode"]) == {"shock", "zero"} and set(frame["split_date"]) == {"2020-03-31", "2020-05-29"}
    good, bad = frame[frame["strategy"] == "honest"], frame[frame["strategy"] == "broken"]
    assert good["passed"].all() and (good["n_differing_cells"] == 0).all()
    assert not bad["passed"].any() and (bad["message"] == "error: no data").all() and (bad["n_differing_cells"] == -1).all() and (bad["n_cells_checked"] == 0).all() and bad["max_weight_difference"].isna().all()
    assert not bad["future_weights_changed"].any() and (bad["first_difference_date"] == "").all()
    defaults = run_leakage_suite(toy_market(1200), {"honest": honest})
    assert sorted(defaults["split_date"].unique()) == ["2012-06-29", "2017-06-30", "2021-06-30"] and len(defaults) == 6 and set(defaults["mode"]) == {"shock", "reverse"}, "three splits, two ways of scrambling"
    assert defaults[defaults["split_date"] == "2021-06-30"]["passed"].all() and (~defaults[defaults["split_date"] == "2012-06-29"]["passed"]).all(), "a split before the data starts has nothing to compare"


# ------------------------------------------------------------------------------------------------------------------ the correlation profile
def aligned_pair(n: int = 400, seed: int = 6):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    returns = pd.DataFrame(rng.normal(0, 0.01, (n, 6)), index=idx, columns=list("ABCDEF"))
    return idx, returns


def test_the_profile_lists_each_lead_and_flags_only_a_large_correlation_at_or_after_the_present():
    idx, returns = aligned_pair()
    signal = returns.shift(-1)                                                                  # tomorrow's return, today
    frame = detect_suspicious_alignment(signal, returns)
    assert frame["lead"].tolist() == [-3, -2, -1, 0, 1, 2, 3] and list(frame.columns) == ["lead", "interpretation", "mean_correlation", "abs_mean_correlation", "implausible", "suspicious", "leakage",
                                                                                           "worst_forward_correlation", "threshold", "diagnosis"]
    assert frame.loc[frame["lead"] == 1, "mean_correlation"].iloc[0] == pytest.approx(1.0) and frame.loc[frame["lead"] == 1, "implausible"].iloc[0]
    assert frame["interpretation"].tolist() == ["signal vs PAST return (expected for momentum)"] * 3 + ["signal vs SAME-DAY return"] + ["signal vs FUTURE return"] * 3
    assert frame["leakage"].all() and frame["suspicious"].all() and frame["threshold"].iloc[0] == 0.30 and frame["worst_forward_correlation"].iloc[0] == pytest.approx(1.0)
    assert frame["diagnosis"].iloc[0] == "LEAKAGE: the signal is implausibly correlated with a strictly future return, so a future observation was used to build it"
    assert (frame["abs_mean_correlation"] == frame["mean_correlation"].abs()).all()


def test_a_signal_built_from_the_same_days_return_is_contemporaneous_and_one_built_from_the_past_is_clean():
    idx, returns = aligned_pair()
    same_day = detect_suspicious_alignment(returns.copy(), returns)
    assert same_day.loc[same_day["lead"] == 0, "implausible"].iloc[0] and not same_day["leakage"].iloc[0] and same_day["suspicious"].iloc[0]
    assert same_day["diagnosis"].iloc[0].startswith("CONTEMPORANEOUS: the signal contains the same day's return") and same_day["worst_forward_correlation"].iloc[0] == pytest.approx(1.0)
    past = detect_suspicious_alignment(returns.shift(1), returns)
    assert past.loc[past["lead"] == -1, "mean_correlation"].iloc[0] == pytest.approx(1.0) and not past["implausible"].any() and not past["suspicious"].iloc[0] and not past["leakage"].iloc[0]
    assert past["diagnosis"].iloc[0] == "CLEAN: no implausible correlation at a non-negative lead" and past["worst_forward_correlation"].iloc[0] < 0.3, "a perfect correlation with the past is a momentum signal, not a leak"


def test_the_threshold_is_thirty_percent_and_the_range_of_leads_is_a_setting():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2019-01-01", periods=120)
    cols = [f"S{i}" for i in range(300)]                                                        # many assets, so that each day's cross-sectional correlation is close to the one built in
    returns = pd.DataFrame(rng.normal(0, 0.01, (120, 300)), index=idx, columns=cols)
    noise = pd.DataFrame(rng.normal(0, 1, (120, 300)), index=idx, columns=cols)
    tomorrow = returns.shift(-1) / 0.01

    def with_correlation(rho: float) -> pd.DataFrame:
        return rho * tomorrow + np.sqrt(1 - rho ** 2) * noise

    mild = detect_suspicious_alignment(with_correlation(0.27), returns)
    strong = detect_suspicious_alignment(with_correlation(0.33), returns)
    c_mild = mild.loc[mild["lead"] == 1, "abs_mean_correlation"].iloc[0]
    c_strong = strong.loc[strong["lead"] == 1, "abs_mean_correlation"].iloc[0]
    assert 0.24 < c_mild < 0.30 < c_strong < 0.37, (c_mild, c_strong)
    assert not mild["leakage"].iloc[0] and strong["leakage"].iloc[0] and mild["diagnosis"].iloc[0].startswith("CLEAN") and strong["diagnosis"].iloc[0].startswith("LEAKAGE")
    low_bar = detect_suspicious_alignment(with_correlation(0.27), returns, implausible=0.2)
    assert low_bar["leakage"].iloc[0] and low_bar["threshold"].iloc[0] == 0.2 and not detect_suspicious_alignment(with_correlation(0.33), returns, implausible=0.6)["leakage"].iloc[0]
    assert strong["worst_forward_correlation"].iloc[0] == pytest.approx(c_strong)
    five = returns.shift(-5)
    assert not detect_suspicious_alignment(five, returns)["leakage"].iloc[0], "a leak five days ahead is not within the default three leads"
    wide = detect_suspicious_alignment(five, returns, max_lead=5)
    assert wide["lead"].tolist() == list(range(-5, 6)) and wide["leakage"].iloc[0] and detect_suspicious_alignment(returns.shift(-3), returns)["leakage"].iloc[0], "but three days ahead is, and a wider range finds the five"
    assert len(detect_suspicious_alignment(returns, returns, max_lead=0)) == 1
