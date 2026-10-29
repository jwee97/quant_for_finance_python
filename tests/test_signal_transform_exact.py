"""The step from a forecast to a position, checked against numbers worked out by hand.

A planted-bug audit (``experiments/plant_bugs.py``) found ``signals/transform.py`` weakly pinned: of the places where one operator, comparison or number was changed the tests noticed 26% (14 of 54).
Every strategy in the library goes through ``signal_to_positions``, so what each step does to a row of signals, the cap on a position and the order of the steps are stated here. Reading what the audit
left alone found three faults, each fixed and pinned below: a long-only book whose cap was undone by the renormalisation that followed it (a single positive forecast became 100% of the book against a
cap of 25%), a weighted average that counted the weight of assets that had no signal, and a book with no positions reported as having infinitely many.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.transform import (
    apply_weight_cap,
    clip_signal,
    cross_sectional_demean,
    cross_sectional_rank,
    cross_sectional_zscore,
    effective_number_of_positions,
    long_only,
    neutralise,
    normalise_gross,
    normalise_net,
    risk_scale,
    signal_to_positions,
    tanh_squash,
    turnover,
    winsorize,
)


def rows(*values, columns="ABCDEFGH") -> pd.DataFrame:
    return pd.DataFrame(list(values), index=pd.bdate_range("2021-01-04", periods=len(values)), columns=list(columns[: len(values[0])]), dtype=float)


# ------------------------------------------------------------------------------------------------------------------ one date at a time
def test_winsorizing_clips_each_date_to_its_own_quantiles():
    s = rows([1, 2, 3, 4, 100], [10, 20, 30, 40, 50])
    out = winsorize(s, 0.2)
    first = np.array([1, 2, 3, 4, 100.0])
    low, high = np.quantile(first, 0.2), np.quantile(first, 0.8)
    assert out.iloc[0].tolist() == pytest.approx(np.clip(first, low, high).tolist()) and out.iloc[0, 4] == pytest.approx(23.2) and out.iloc[0, 0] == pytest.approx(1.8)
    assert out.iloc[1].tolist() == pytest.approx(np.clip([10, 20, 30, 40, 50.0], 18, 42).tolist()), "each date on its own: the second row's quantiles are not the first's"
    assert winsorize(s, 0.0) is s and winsorize(s, -0.1) is s, "a quantile of zero or less leaves the signal alone"
    assert winsorize(s).equals(winsorize(s, 0.02)), "two percent each side by default"
    assert not winsorize(s, 0.02).equals(s) and winsorize(s, 0.02).iloc[0, 4] < 100.0 and winsorize(s, 0.02).iloc[0, 4] == pytest.approx(np.quantile(first, 0.98))
    gappy = rows([1, np.nan, 3, 4, 100])
    assert np.isnan(winsorize(gappy, 0.2).iloc[0, 1]), "a missing signal stays missing"


def test_demeaning_makes_a_signal_relative_and_a_weighted_mean_ignores_assets_without_a_signal():
    s = rows([1, 2, 6], [4, 4, 4])
    assert cross_sectional_demean(s).iloc[0].tolist() == pytest.approx([-2, -1, 3]) and cross_sectional_demean(s).iloc[1].tolist() == [0, 0, 0]
    w = rows([1, 1, 2], [1, 1, 2])
    assert cross_sectional_demean(s, w).iloc[0].tolist() == pytest.approx([1 - 3.75, 2 - 3.75, 6 - 3.75]), "the mean weighted 1, 1 and 2 is (1 + 2 + 12) / 4"
    gap = rows([1, 2, np.nan], [1, 2, 3])
    out = cross_sectional_demean(gap, rows([1, 1, 2], [1, 1, 2]))
    assert out.iloc[0, :2].tolist() == pytest.approx([-0.5, 0.5]), "the asset with no signal has no weight in the average of the other two"
    assert np.isnan(out.iloc[0, 2]) and out.iloc[1].tolist() == pytest.approx([1 - 2.25, 2 - 2.25, 3 - 2.25])


def test_a_z_score_has_mean_zero_and_unit_deviation_and_a_row_that_does_not_vary_has_none():
    s = rows([1, 2, 3, 4], [5, 5, 5, 5], [0, 0, 0, 8])
    z = cross_sectional_zscore(s)
    assert z.iloc[0].tolist() == pytest.approx(((np.array([1, 2, 3, 4.0]) - 2.5) / np.std([1, 2, 3, 4], ddof=1)).tolist()) and z.iloc[0].std(ddof=1) == pytest.approx(1.0) and abs(z.iloc[0].mean()) < 1e-12
    assert z.iloc[1].isna().all(), "no spread to standardise by"
    assert cross_sectional_zscore(s, ddof=0).iloc[0].tolist() == pytest.approx(((np.array([1, 2, 3, 4.0]) - 2.5) / np.std([1, 2, 3, 4], ddof=0)).tolist())
    assert cross_sectional_zscore(s).iloc[2, 3] == pytest.approx((8 - 2) / np.std([0, 0, 0, 8], ddof=1))


def test_a_percentile_rank_keeps_the_order_and_not_the_size_centred_or_not():
    s = rows([10, 30, 20, 40], [5, 5, 7, np.nan])
    plain = cross_sectional_rank(s, centred=False)
    assert plain.iloc[0].tolist() == [0.25, 0.75, 0.5, 1.0] and cross_sectional_rank(s).iloc[0].tolist() == pytest.approx([-0.5, 0.5, 0.0, 1.0]), "centred: (rank - 1/2) * 2"
    assert plain.iloc[1].tolist()[:3] == pytest.approx([0.5 * (1 / 3 + 2 / 3), 0.5 * (1 / 3 + 2 / 3), 1.0]) and np.isnan(plain.iloc[1, 3]), "ties share the average rank and a missing signal has none"
    assert cross_sectional_rank(s, centred=True).equals(cross_sectional_rank(s)), "centred by default"
    assert cross_sectional_rank(s).iloc[1, 2] == pytest.approx(1.0) and cross_sectional_rank(s).iloc[1, 0] == pytest.approx(2 * (0.5 * (1 / 3 + 2 / 3)) - 1.0)


def test_squashing_and_clipping_bound_a_signal():
    s = rows([-10, -1, 0, 1, 10])
    assert tanh_squash(s).iloc[0].tolist() == pytest.approx(np.tanh([-10, -1, 0, 1, 10]).tolist()) and tanh_squash(s, 2.0).iloc[0].tolist() == pytest.approx(np.tanh(np.array([-10, -1, 0, 1, 10]) / 2.0).tolist())
    assert clip_signal(s).iloc[0].tolist() == [-3, -1, 0, 1, 3] and clip_signal(s, 0.5).iloc[0].tolist() == [-0.5, -0.5, 0, 0.5, 0.5], "three by default"
    assert clip_signal(s, 0).equals(s) and clip_signal(s, None).equals(s), "no limit, no clip"


def test_neutralising_removes_each_groups_own_mean_and_assets_without_a_group_form_one():
    s = rows([1.0, 3.0, 10.0, 20.0, 7.0])
    out = neutralise(s, {"A": "stocks", "B": "stocks", "C": "bonds", "D": "bonds"})                  # E has no group: it is alone in "other"
    assert out.iloc[0].tolist() == pytest.approx([-1.0, 1.0, -5.0, 5.0, 0.0]), "within each group the mean is taken off; a group of one has nothing left"
    assert s.iloc[0].tolist() == [1.0, 3.0, 10.0, 20.0, 7.0], "the input is not changed"
    assert neutralise(s, {}).iloc[0].tolist() == pytest.approx((s.iloc[0] - s.iloc[0].mean()).tolist()), "with no groups everything is one group, 'other'"


def test_risk_scaling_divides_by_volatility_with_a_floor_and_normalising_sets_the_exposure():
    s = rows([1.0, -1.0, 2.0])
    vol = rows([0.1, 0.2, 0.0])
    out = risk_scale(s, vol)
    assert out.iloc[0].tolist() == pytest.approx([1 * 0.10 / 0.1, -1 * 0.10 / 0.2, 2 * 0.10 / 1e-4]), "a volatility of zero is floored at 1e-4, and the default target is 10%"
    assert risk_scale(s, vol, target_vol=0.2, floor=0.05).iloc[0].tolist() == pytest.approx([2.0, -1.0, 2 * 0.2 / 0.05])
    assert np.isnan(risk_scale(s, rows([0.1, np.nan, 0.1])).iloc[0, 1]), "no volatility known, no position"
    w = rows([2.0, -1.0, 1.0], [0.0, 0.0, 0.0])
    assert normalise_gross(w).iloc[0].tolist() == pytest.approx([0.5, -0.25, 0.25]) and normalise_gross(w, 2.0).iloc[0].tolist() == pytest.approx([1.0, -0.5, 0.5]) and normalise_gross(w).iloc[1].isna().all()
    assert normalise_net(w).iloc[0].tolist() == pytest.approx([1.0, -0.5, 0.5]) and normalise_net(w, 2.0).iloc[0].tolist() == pytest.approx([2.0, -1.0, 1.0]) and normalise_net(rows([1.0, -1.0])).iloc[0].isna().all()
    assert long_only(rows([2.0, -1.0, 6.0])).iloc[0].tolist() == pytest.approx([0.25, 0.0, 0.75]) and long_only(rows([-1.0, -2.0])).iloc[0].isna().all()


# ------------------------------------------------------------------------------------------------------------------ the cap
def test_a_cap_hands_what_it_cuts_to_the_names_under_the_cap_until_nobody_is_over():
    w = rows([0.6, 0.2, 0.1, 0.05, 0.05])
    capped = apply_weight_cap(w, 0.3, 1.0)
    assert capped.iloc[0].tolist() == pytest.approx([0.3, 0.3, 0.2, 0.1, 0.1]), "0.3 off the first goes to the rest in proportion, which lifts the second to the cap, and its excess goes on"
    assert capped.iloc[0].sum() == pytest.approx(1.0) and apply_weight_cap(w, 0.3, None).iloc[0].tolist() == pytest.approx([0.3, 0.2, 0.1, 0.05, 0.05]), "no gross to hold: only clipped"
    short = apply_weight_cap(rows([-0.6, 0.2, 0.1, 0.05, 0.05]), 0.3, 1.0)
    assert short.iloc[0].tolist() == pytest.approx([-0.3, 0.3, 0.2, 0.1, 0.1]) and short.iloc[0].abs().sum() == pytest.approx(1.0), "a short is capped like a long"
    assert apply_weight_cap(rows([0.5, 0.3, 0.2]), 0.25, 1.0).iloc[0].tolist() == pytest.approx([0.25, 0.25, 0.25]), "three names at 25% cannot make a whole book: capped, and 0.75 gross, not a breach"
    assert apply_weight_cap(rows([0.3, 0.3, 0.2, 0.1, 0.1]), 0.3, 1.0).iloc[0].tolist() == pytest.approx([0.3, 0.3, 0.2, 0.1, 0.1]), "already as it should be"
    assert apply_weight_cap(w).iloc[0].max() <= 0.25 + 1e-12 and apply_weight_cap(w).iloc[0].sum() == pytest.approx(1.0), "25% and a gross of one by default"
    assert apply_weight_cap(rows([0.8, 0.5, 0.3, 0.2, 0.2]), 0.5, 2.0).iloc[0].tolist() == pytest.approx([0.5, 0.5, 3 / 7, 2 / 7, 2 / 7]), "a gross of two in five names capped at one half"
    nan = apply_weight_cap(rows([0.6, np.nan, 0.2, 0.1, 0.1]), 0.3, 1.0)
    assert np.isnan(nan.iloc[0, 1]) and nan.iloc[0].sum() == pytest.approx(1.0) and nan.iloc[0].max() <= 0.3 + 1e-12, "a missing weight stays missing and the others fill the book"
    flat = apply_weight_cap(rows([0.0, 0.0, 0.0], [0.5, 0.3, 0.2]), 0.4, 1.0)
    assert flat.iloc[0].tolist() == [0.0, 0.0, 0.0] and flat.iloc[1].tolist() == pytest.approx([0.4, 0.36, 0.24]), "an empty row stays empty while the next is capped, and the 0.1 cut goes to the other two in proportion"
    assert apply_weight_cap(rows([0.6, 0.3, 0.1]), 0.5, 1.0, iterations=1).iloc[0].tolist() == pytest.approx([0.5, 0.375, 0.125]), "one pass of redistribution: the 0.1 cut is spread over the other two in proportion (a factor of 1.25)"


def test_a_cap_that_already_fills_the_gross_scales_the_whole_book_down_instead_of_leaving_it_over():
    four = apply_weight_cap(rows([0.4, 0.4, 0.4, 0.4, 0.05]), 0.25, 1.0)
    assert four.iloc[0].abs().sum() == pytest.approx(1.0) and four.iloc[0].abs().max() <= 0.25 + 1e-12, "four names cut to a quarter hold the whole gross: the fifth cannot be added to it"
    assert four.iloc[0].tolist() == pytest.approx([0.25 / 1.05] * 4 + [0.05 / 1.05]), "everything is scaled down together: no cap is breached by a smaller book"
    five = apply_weight_cap(rows([0.3, 0.3, 0.3, 0.3, 0.3]), 0.25, 1.0)
    assert five.iloc[0].tolist() == pytest.approx([0.2] * 5), "five names cut to a quarter hold 1.25: they come down to a fifth each"
    mixed = apply_weight_cap(rows([-0.6, -0.6, 0.6, 0.6, 0.6, 0.01]), 0.25, 1.0)
    assert mixed.iloc[0].abs().sum() == pytest.approx(1.0) and mixed.iloc[0].tolist()[:2] == pytest.approx([-0.25 / 1.26] * 2) and mixed.iloc[0].iloc[5] == pytest.approx(0.01 / 1.26), "shorts keep their sign"
    exact = apply_weight_cap(rows([0.4, 0.4, 0.4, 0.4]), 0.25, 1.0)
    assert exact.iloc[0].tolist() == pytest.approx([0.25] * 4), "four at a quarter are a whole book and are left alone"
    assert apply_weight_cap(rows([0.4, 0.4, 0.4, 0.4, 0.05], [0.3, 0.3, 0.2, 0.1, 0.1]), 0.25, 1.0).iloc[1].sum() == pytest.approx(1.0), "a row that needs nothing of this is not disturbed by one that does"


# ------------------------------------------------------------------------------------------------------------------ the whole stack
def test_the_stack_applies_investable_then_winsorising_then_neutralising_then_scaling_then_the_cap():
    s = rows([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    plain = signal_to_positions(s, winsorize_quantile=0.0, scale="none", clip=None, cross_sectional=False, max_abs_weight=1.0)
    assert plain.iloc[0].tolist() == pytest.approx((np.arange(1, 7) / 21).tolist()), "no step but the normalisation: weights in proportion to the signal, gross one"
    centred = signal_to_positions(s, winsorize_quantile=0.0, scale="none", clip=None, cross_sectional=True, max_abs_weight=1.0)
    assert centred.iloc[0].tolist() == pytest.approx((np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) / 9.0).tolist()), "demeaned, then gross one: nine halves"
    z = signal_to_positions(s, winsorize_quantile=0.0, clip=None, max_abs_weight=1.0)
    assert z.iloc[0].tolist() == pytest.approx(centred.iloc[0].tolist()), "z-scoring only rescales a row that is then rescaled to gross one"
    ranked = signal_to_positions(s, winsorize_quantile=0.0, scale="rank", clip=None, max_abs_weight=1.0)
    assert ranked.iloc[0].tolist() == pytest.approx((np.array([-2, -1, 0, 1, 2, 3]) / 9.0).tolist()), "centred percentile ranks (r - 3) / 3 for six assets, then gross one"
    squashed = signal_to_positions(rows([-2.0, 0.0, 0.5, 1.0, 3.0]), winsorize_quantile=0.0, scale="tanh", clip=None, cross_sectional=False, max_abs_weight=1.0).iloc[0]
    assert squashed.tolist() == pytest.approx((np.tanh([-2, 0, 0.5, 1, 3]) / np.abs(np.tanh([-2, 0, 0.5, 1, 3])).sum()).tolist())
    assert signal_to_positions(s, winsorize_quantile=0.0, scale=None, clip=None, max_abs_weight=1.0).iloc[0].tolist() == pytest.approx(centred.iloc[0].tolist())
    with pytest.raises(ValueError, match="unknown scale 'sigmoid'"):
        signal_to_positions(s, scale="sigmoid")


def test_the_stack_clips_scales_by_risk_caps_and_leaves_a_flat_book_where_nothing_is_investable():
    s = rows([5.0, -5.0, 0.2, -0.2])
    clipped = signal_to_positions(s, winsorize_quantile=0.0, scale="none", clip=1.0, cross_sectional=False, max_abs_weight=1.0)
    assert clipped.iloc[0].tolist() == pytest.approx((np.array([1.0, -1.0, 0.2, -0.2]) / 2.4).tolist()), "each signal limited to one before the rows are normalised"
    vol = rows([0.1, 0.1, 0.4, 0.4])
    risky = signal_to_positions(rows([1.0, -1.0, 1.0, -1.0]), vol, winsorize_quantile=0.0, scale="none", clip=None, cross_sectional=False, max_abs_weight=1.0)
    assert risky.iloc[0].tolist() == pytest.approx((np.array([1.0, -1.0, 0.25, -0.25]) / 2.5).tolist()), "equal conviction, equal risk: the volatile pair gets a quarter of the weight"
    assert signal_to_positions(rows([1.0, -1.0, 1.0, -1.0]), vol, target_vol=0.5, winsorize_quantile=0.0, scale="none", clip=None, cross_sectional=False, max_abs_weight=1.0).equals(risky), "the target is a scale the gross normalisation takes out"
    big = signal_to_positions(rows([1.0, 2.0, 3.0, 30.0, -20.0, -3.0]), winsorize_quantile=0.0, scale="none", clip=None, cross_sectional=False, gross_leverage=1.5, max_abs_weight=0.4).iloc[0]
    assert big.abs().max() <= 0.4 + 1e-12 and big.abs().sum() == pytest.approx(1.5) and big.iloc[3] == pytest.approx(0.4) and big.iloc[4] == pytest.approx(-0.4), "a gross of 1.5 with a cap of 0.4"
    two = rows([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    investable = pd.DataFrame([[True] * 4, [False] * 4], index=two.index, columns=list("ABCD"))
    out = signal_to_positions(two, winsorize_quantile=0.0, investable=investable, max_abs_weight=1.0)
    assert out.index.equals(two.index) and list(out.columns) == list("ABCD") and (out.iloc[1] == 0.0).all() and out.iloc[0].abs().sum() == pytest.approx(1.0), "nothing investable: a flat book of zeros, not missing data"
    partial = pd.DataFrame([[True, True, False, np.nan]], index=two.index[:1], columns=list("ABCD"), dtype=object)
    some = signal_to_positions(two.iloc[:1], winsorize_quantile=0.0, investable=partial, max_abs_weight=1.0)
    assert some.iloc[0, 2] == 0.0 and some.iloc[0, 3] == 0.0 and some.iloc[0, 0] == pytest.approx(-0.5) and some.iloc[0, 1] == pytest.approx(0.5), "an asset not known to be investable holds nothing"
    grouped = signal_to_positions(rows([1.0, 3.0, 10.0, 20.0]), winsorize_quantile=0.0, scale="none", clip=None, groups={"A": "x", "B": "x", "C": "y", "D": "y"}, cross_sectional=False, max_abs_weight=1.0)
    assert grouped.iloc[0].tolist() == pytest.approx((np.array([-1.0, 1.0, -5.0, 5.0]) / 12.0).tolist()), "no bet between the groups"
    assert signal_to_positions(rows([1.0, 2.0, 3.0, 40.0, 5.0]), max_abs_weight=1.0).iloc[0].abs().max() < signal_to_positions(rows([1.0, 2.0, 3.0, 40.0, 5.0]), winsorize_quantile=0.0, max_abs_weight=1.0).iloc[0].abs().max(), "two percent winsorising by default pulls an outlier in"


def test_the_defaults_are_two_percent_winsorising_a_clip_of_three_and_a_cap_of_a_quarter_of_the_book():
    twelve = "ABCDEFGHIJKL"
    s = rows([10.0] + [0.0] * 11, columns=twelve)                                             # one extreme forecast among twelve
    wide = dict(max_abs_weight=1.0)
    default = signal_to_positions(s, **wide)
    assert default.equals(signal_to_positions(s, winsorize_quantile=0.02, clip=3.0, **wide)) and not default.equals(signal_to_positions(s, winsorize_quantile=0.0, **wide)) and not default.equals(signal_to_positions(s, clip=None, **wide))
    no_winsor = signal_to_positions(s, winsorize_quantile=0.0, **wide).iloc[0]
    z_outlier = (10.0 - 10.0 / 12) / np.std([10.0] + [0.0] * 11, ddof=1)                      # 3.175: over the limit of three
    assert z_outlier > 3.0 and no_winsor["A"] == pytest.approx(3.0 / (3.0 + 11 * (10.0 / 12) / np.std([10.0] + [0.0] * 11, ddof=1))), "the extreme z-score is cut to three"
    assert signal_to_positions(s, winsorize_quantile=0.0, clip=None, **wide).iloc[0, 0] == pytest.approx(0.5), "uncut it would be half the book"
    assert signal_to_positions(s).iloc[0].abs().max() == pytest.approx(0.25) and signal_to_positions(s, max_abs_weight=0.25).equals(signal_to_positions(s)) and signal_to_positions(s, **wide).iloc[0].abs().max() > 0.3
    assert signal_to_positions(s).iloc[0].abs().sum() == pytest.approx(1.0) and signal_to_positions(s, gross_leverage=1.0).equals(signal_to_positions(s)), "a gross of one by default"


def test_a_long_only_book_is_fully_invested_in_the_longs_and_never_over_the_cap():
    s = rows([3.0, 0.2, 0.1, 0.05, -0.2, -0.3, 0.0, 0.1], columns="ABCDEFGH")
    one_long = signal_to_positions(s, long_only_book=True, max_abs_weight=0.25, winsorize_quantile=0.0)
    assert one_long.iloc[0].max() == pytest.approx(0.25) and one_long.iloc[0].sum() == pytest.approx(0.25) and (one_long.iloc[0] >= 0).all(), "one positive forecast cannot be 100% of a book capped at 25%: it is 25% and the rest is cash"
    many = rows([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1], columns="ABCDEFGH")
    book = signal_to_positions(many, long_only_book=True, max_abs_weight=0.25, winsorize_quantile=0.0, scale="none", clip=None, cross_sectional=False).iloc[0]
    assert book.sum() == pytest.approx(1.0) and book.max() <= 0.25 + 1e-12 and (book >= 0).all() and book.iloc[0] == pytest.approx(0.25) and book.iloc[1] == pytest.approx(0.25), "fully invested where the cap leaves room"
    assert book.iloc[2] > book.iloc[3] > book.iloc[4], "the order of the forecasts is kept among the names under the cap"
    assert signal_to_positions(s, long_only_book=False, max_abs_weight=0.25, winsorize_quantile=0.0).iloc[0].abs().sum() == pytest.approx(1.0), "a long-short book is held to its gross"


def test_turnover_and_the_effective_number_of_positions():
    w = rows([0.5, 0.5, 0.0], [0.4, 0.4, 0.2], [0.0, 0.0, 0.0])
    assert turnover(w).iloc[1] == pytest.approx(0.1 + 0.1 + 0.2) and turnover(w).iloc[2] == pytest.approx(1.0) and turnover(w).iloc[0] == 0.0, "the first row has nothing to differ from"
    book = rows([0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0], [0.5, -0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.7, 0.1, 0.1, 0.1])
    n = effective_number_of_positions(book)
    assert n.iloc[0] == pytest.approx(4.0) and n.iloc[1] == pytest.approx(1.0) and n.iloc[2] == pytest.approx(2.0), "equal weights give their count, and a short counts as a position"
    assert np.isnan(n.iloc[3]), "a book with no positions is not infinitely diversified"
    assert n.iloc[4] == pytest.approx(1.0 / (0.7 ** 2 + 3 * 0.1 ** 2))
