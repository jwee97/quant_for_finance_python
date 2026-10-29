"""The equal-weight and inverse-volatility books, checked against numbers worked out by hand.

These two books are what every strategy is compared with (the Backtest tab's "beats equal weight" and "beats risk parity" flags), so a slip in either changes a verdict. A planted-bug audit
(``experiments/plant_bugs.py``) found ``inverse_vol_book`` untested: of the eight places where one operator, comparison or number was changed in ``portfolio/inverse_vol.py`` the tests noticed two,
and none of them in the book through time (only in the weights of one day). What each book holds on each day is stated here with the arithmetic written out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.volatility import ewma_volatility, rolling_volatility
from src.portfolio.constraints import Constraints
from src.portfolio.covariance import estimate_covariance
from src.portfolio.equal_weight import equal_weight_book, equal_weights
from src.portfolio.inverse_vol import inverse_vol_book, inverse_volatility_weights
from src.portfolio.risk_parity import (
    marginal_risk_contributions,
    risk_concentration,
    risk_contribution_frame,
    risk_contributions,
    risk_parity_book,
    risk_parity_weights,
)


def days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-04", periods=n)


# ------------------------------------------------------------------------------------------------------------------ equal weight
def test_equal_weights_split_the_value_evenly_and_a_book_splits_each_day_over_what_can_be_held():
    assert equal_weights(["A", "B", "C", "D"]).tolist() == [0.25] * 4 and equal_weights(["A", "B"], 2.0).tolist() == [1.0, 1.0] and equal_weights([]).empty
    assert list(equal_weights(iter("XY")).index) == ["X", "Y"], "any iterable of names"
    investable = pd.DataFrame({"A": [True, True, True, False], "B": [True, True, False, False], "C": [False, True, True, False]}, index=days(4))
    book = equal_weight_book(investable)
    assert book.iloc[0].tolist() == [0.5, 0.5, 0.0] and book.iloc[1].tolist() == pytest.approx([1 / 3] * 3) and book.iloc[2].tolist() == [0.5, 0.0, 0.5]
    assert book.iloc[3].tolist() == [0.0, 0.0, 0.0], "a day on which nothing can be held is flat: zeros, not missing"
    assert not book.isna().any().any() and book.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0, 0.0])
    assert equal_weight_book(investable.astype(int)).equals(book), "ones and zeros are an investable mask too"


def test_an_equal_weight_book_is_projected_onto_the_limits_when_given_them():
    investable = pd.DataFrame(True, index=days(2), columns=list("ABCDEF"))
    loose = equal_weight_book(investable, Constraints(max_weight=0.25))
    assert loose.iloc[0].tolist() == pytest.approx([1 / 6] * 6), "a sixth is under the cap of a quarter: nothing to do"
    tight = equal_weight_book(investable, Constraints(max_weight=0.1))
    assert tight.iloc[0].tolist() == pytest.approx([0.1] * 6) and tight.iloc[0].sum() == pytest.approx(0.6), "six names capped at a tenth cannot make a whole book: capped, short of the target"
    few = pd.DataFrame({"A": [True], "B": [True], "C": [False], "D": [False]}, index=days(1))
    assert equal_weight_book(few, Constraints(max_weight=0.4)).iloc[0].tolist() == pytest.approx([0.4, 0.4, 0.0, 0.0]), "half each is over the cap of 0.4"


# ------------------------------------------------------------------------------------------------------------------ inverse volatility
def test_inverse_volatility_weights_are_the_reciprocal_of_each_volatility_over_the_sum_of_them():
    vol = pd.Series({"A": 0.10, "B": 0.20, "C": 0.40})
    w = inverse_volatility_weights(vol)
    total = 1 / 0.10 + 1 / 0.20 + 1 / 0.40
    assert w.to_dict() == pytest.approx({"A": (1 / 0.10) / total, "B": (1 / 0.20) / total, "C": (1 / 0.40) / total}) and w.sum() == pytest.approx(1.0)
    with_gaps = pd.Series({"A": 0.10, "B": np.nan, "C": 0.0, "D": 0.20})
    assert inverse_volatility_weights(with_gaps).to_dict() == pytest.approx({"A": 2 / 3, "D": 1 / 3}), "a missing volatility and a volatility of zero get no weight"
    assert inverse_volatility_weights(pd.Series({"A": np.nan, "B": 0.0})).empty and inverse_volatility_weights(pd.Series(dtype=float)).empty
    tiny = pd.Series({"A": 1e-9, "B": 1e-3})
    assert inverse_volatility_weights(tiny).to_dict() == pytest.approx({"A": 1e6 / (1e6 + 1e3), "B": 1e3 / (1e6 + 1e3)}), "a volatility under a millionth is taken as a millionth: no weight of 1 / 1e-9"
    assert inverse_volatility_weights(tiny, floor=1e-4).to_dict() == pytest.approx({"A": 1e4 / (1e4 + 1e3), "B": 1e3 / (1e4 + 1e3)}) and inverse_volatility_weights(tiny, floor=0.0).iloc[0] > 0.999999


def make_returns(n: int = 160, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"A": rng.normal(0, 0.01, n), "B": rng.normal(0, 0.02, n), "C": rng.normal(0, 0.04, n)}, index=days(n))


def test_the_book_weights_each_day_by_the_inverse_of_the_volatility_known_that_day():
    r = make_returns()
    investable = pd.DataFrame(True, index=r.index, columns=r.columns)
    k = 120
    ewma = inverse_vol_book(r, investable)
    sigma = ewma_volatility(r, 40.0).iloc[k]
    assert ewma.iloc[k].tolist() == pytest.approx((1 / sigma / (1 / sigma).sum()).tolist()), "an exponentially weighted volatility with a half-life of 40 days, by default"
    assert ewma.iloc[k]["A"] > ewma.iloc[k]["B"] > ewma.iloc[k]["C"], "the steadiest asset gets the most"
    slow = inverse_vol_book(r, investable, halflife=10.0)
    quick = ewma_volatility(r, 10.0).iloc[k]
    assert slow.iloc[k].tolist() == pytest.approx((1 / quick / (1 / quick).sum()).tolist())
    rolling = inverse_vol_book(r, investable, method="rolling")
    spread = rolling_volatility(r, 63).iloc[k]
    assert rolling.iloc[k].tolist() == pytest.approx((1 / spread / (1 / spread).sum()).tolist()), "a rolling volatility over 63 days when asked for it"
    short = inverse_vol_book(r, investable, lookback=20, method="rolling")
    brief = rolling_volatility(r, 20).iloc[k]
    assert short.iloc[k].tolist() == pytest.approx((1 / brief / (1 / brief).sum()).tolist()) and not np.allclose(short.iloc[k], rolling.iloc[k])
    assert not np.allclose(ewma.iloc[k], rolling.iloc[k]), "the method makes a difference"
    assert (ewma.iloc[:29] == 0.0).all().all() and ewma.iloc[29:].sum(axis=1).tolist() == pytest.approx([1.0] * (len(r) - 29)), "no book until the estimate has thirty days, a whole one after"
    changed = r.copy()
    changed.iloc[k + 1] = [0.3, -0.3, 0.3]
    again = inverse_vol_book(changed, investable)
    assert again.iloc[:k + 1].equals(ewma.iloc[:k + 1]) and not again.iloc[k + 1].equals(ewma.iloc[k + 1]), "a return reaches the weights of its own day and none before"


def test_the_book_leaves_out_what_cannot_be_held_or_has_no_volatility_and_obeys_the_limits_when_given_them():
    r = make_returns()
    r["Z"] = 0.0                                                                         # a series that never moves has no volatility to take the inverse of
    investable = pd.DataFrame(True, index=r.index, columns=r.columns)
    k = 100
    book = inverse_vol_book(r, investable)
    assert book.iloc[k]["Z"] == 0.0 and book.iloc[k].sum() == pytest.approx(1.0), "a volatility of zero gets no weight, not an infinite one"
    investable.loc[investable.index[k], "B"] = False
    held = inverse_vol_book(r, investable)
    sigma = ewma_volatility(r, 40.0).iloc[k]
    inverse = (1 / sigma).drop(["B", "Z"])
    assert held.iloc[k]["B"] == 0.0 and held.iloc[k][["A", "C"]].tolist() == pytest.approx((inverse / inverse.sum()).tolist()), "what cannot be held on a day gets nothing that day and the rest share the whole"
    capped = inverse_vol_book(r.drop(columns="Z"), pd.DataFrame(True, index=r.index, columns=["A", "B", "C"]), constraints=Constraints(max_weight=0.5))
    free = inverse_vol_book(r.drop(columns="Z"), pd.DataFrame(True, index=r.index, columns=["A", "B", "C"]))
    assert free.iloc[k]["A"] > 0.5 and capped.iloc[k]["A"] == pytest.approx(0.5) and capped.iloc[k].sum() == pytest.approx(1.0), "the steadiest asset is cut to the cap and the rest take what it gave up"
    rest = free.iloc[k][["B", "C"]]
    room = 0.5 - rest                                                                    # what A gave up is shared in proportion to the room under the cap, not to the weight
    assert capped.iloc[k][["B", "C"]].tolist() == pytest.approx((rest + (free.iloc[k]["A"] - 0.5) * room / room.sum()).tolist())


def test_a_limit_that_binds_before_every_asset_has_listed_hands_its_excess_to_the_assets_that_have():
    """Before HYG listed the book had to be spread over the other names. A cap that clipped the steadiest asset used to hand what it freed to every name with room under the cap, listed or not, and the engine dropped
    what went to the unlisted: the book was left partly in cash on exactly the days the cap bound."""
    rng = np.random.default_rng(9)
    n = 200
    r = pd.DataFrame({"A": rng.normal(0, 0.002, n), "B": rng.normal(0, 0.02, n), "C": rng.normal(0, 0.02, n), "D": rng.normal(0, 0.02, n), "E": rng.normal(0, 0.02, n), "F": rng.normal(0, 0.02, n)}, index=days(n))
    investable = pd.DataFrame(True, index=r.index, columns=r.columns)
    investable.loc[investable.index[:120], ["E", "F"]] = False
    r.loc[r.index[:120], ["E", "F"]] = np.nan
    k = 60
    free = inverse_vol_book(r, investable)
    assert free.iloc[k]["A"] > 0.25 and free.iloc[k][["E", "F"]].tolist() == [0.0, 0.0], "four listed names, the steadiest of them with more than a quarter"
    capped = inverse_vol_book(r, investable, constraints=Constraints(max_weight=0.25))
    assert capped.iloc[k][["A", "B", "C", "D"]].tolist() == pytest.approx([0.25] * 4) and capped.iloc[k][["E", "F"]].tolist() == [0.0, 0.0], "four names at a quarter are a whole book and the two that have not listed hold nothing"
    assert capped.iloc[k].sum() == pytest.approx(1.0)
    later = 150
    assert capped.iloc[later].sum() == pytest.approx(1.0) and capped.iloc[later].max() <= 0.25 + 1e-9 and (capped.iloc[later] > 0).all(), "once all six are listed they share it"


# ------------------------------------------------------------------------------------------------------------------ risk parity
COV = np.array([[0.04, 0.006], [0.006, 0.09]])                                          # volatilities of 20% and 30%, a correlation of 0.1


def test_risk_contributions_are_the_weight_times_the_marginal_contribution_and_add_up_to_the_volatility():
    w = np.array([0.6, 0.4])
    sigma = float(np.sqrt(w @ COV @ w))
    mrc = COV @ w / sigma
    assert marginal_risk_contributions(w, COV).tolist() == pytest.approx(mrc.tolist()) and risk_contributions(w, COV).tolist() == pytest.approx((w * mrc).tolist())
    assert risk_contributions(w, COV).sum() == pytest.approx(sigma), "Euler: the contributions add up to the portfolio volatility"
    assert risk_contributions(w, COV, normalise=True).tolist() == pytest.approx((w * mrc / sigma).tolist()) and risk_contributions(w, COV, normalise=True).sum() == pytest.approx(1.0)
    assert marginal_risk_contributions([0.0, 0.0], COV).tolist() == [0.0, 0.0] and risk_contributions([0.0, 0.0], COV, normalise=True).tolist() == [0.0, 0.0], "a book of nothing has no risk to share"
    table = risk_contribution_frame(pd.Series({"A": 0.6, "B": 0.4}), pd.DataFrame(COV, index=["A", "B"], columns=["A", "B"]))
    assert list(table.columns) == ["weight", "marginal_risk", "risk_contribution", "risk_share"] and table["risk_share"].sum() == pytest.approx(1.0)
    assert table["marginal_risk"].tolist() == pytest.approx(mrc.tolist()) and table["risk_share"].tolist() == pytest.approx((w * mrc / sigma).tolist())
    assert risk_contribution_frame(pd.Series({"B": 0.4, "A": 0.6}), pd.DataFrame(COV, index=["A", "B"], columns=["A", "B"])).index.tolist() == ["B", "A"], "the covariance is read in the order of the weights"
    assert risk_contribution_frame(pd.Series({"A": 0.0, "B": 0.0}), pd.DataFrame(COV, index=["A", "B"], columns=["A", "B"]))["risk_share"].isna().all()
    equal = np.array([0.5, 0.5])
    shares = risk_contributions(equal, COV, normalise=True)
    assert risk_concentration(equal, COV) == pytest.approx(float(np.sqrt(np.sum((shares - 0.5) ** 2)))) and risk_concentration(equal, COV) > 0.0
    parity = risk_parity_weights(pd.DataFrame(COV, index=["A", "B"], columns=["A", "B"]))
    assert risk_concentration(parity.to_numpy(), COV) == pytest.approx(0.0, abs=1e-8), "zero for a perfect risk-parity book"
    three = np.array([[0.04, 0, 0], [0, 0.04, 0], [0, 0, 0.04]])
    assert risk_concentration([1.0, 0.0, 0.0], three) == pytest.approx(float(np.sqrt((1 - 1 / 3) ** 2 + 2 * (1 / 3) ** 2))), "all the risk in one of three names"


def test_two_uncorrelated_assets_weigh_in_proportion_to_the_root_of_their_budget_over_their_volatility():
    cov = pd.DataFrame(np.diag([0.01, 0.04]), index=["A", "B"], columns=["A", "B"])         # volatilities of 10% and 20%
    equal = risk_parity_weights(cov)
    assert equal.to_dict() == pytest.approx({"A": 2 / 3, "B": 1 / 3}), "equal risk from assets of 10% and 20%: twice as much of the first"
    budget = pd.Series({"A": 0.75, "B": 0.25})
    expected = np.array([np.sqrt(0.75) / 0.1, np.sqrt(0.25) / 0.2])
    expected = expected / expected.sum()
    for method in ("newton", "slsqp"):
        got = risk_parity_weights(cov, budget, method=method)
        assert got.tolist() == pytest.approx(expected.tolist(), abs=1e-5), method
        shares = risk_contributions(got.to_numpy(), cov.to_numpy(), normalise=True)
        assert shares.tolist() == pytest.approx([0.75, 0.25], abs=1e-5), f"the risk is shared as the budget says ({method})"
    scaled = risk_parity_weights(cov, pd.Series({"A": 3.0, "B": 1.0}), method="slsqp")
    assert scaled.tolist() == pytest.approx(expected.tolist(), abs=1e-5), "a budget is read as shares: 3 and 1 are 0.75 and 0.25"
    assert risk_parity_weights(cov, pd.Series({"A": 0.0, "B": 0.0})).to_dict() == pytest.approx({"A": 2 / 3, "B": 1 / 3}) and risk_parity_weights(cov, pd.Series({"A": 1.0})).iloc[0] > 0.99, "no budget at all is an equal one; a name left out of the budget has none"
    assert risk_parity_weights(cov, method="slsqp").tolist() == pytest.approx([2 / 3, 1 / 3], abs=1e-5)
    with pytest.raises(ValueError, match="unknown risk-parity method 'gradient'"):
        risk_parity_weights(cov, method="gradient")


def test_with_equal_correlations_risk_parity_is_inverse_volatility_and_a_limit_is_applied_to_the_solution():
    vols = np.array([0.10, 0.15, 0.30, 0.20])
    corr = np.full((4, 4), 0.3) + 0.7 * np.eye(4)
    cov = pd.DataFrame(np.outer(vols, vols) * corr, index=list("ABCD"), columns=list("ABCD"))
    parity = risk_parity_weights(cov)
    inverse = 1 / vols / (1 / vols).sum()
    assert parity.tolist() == pytest.approx(inverse.tolist(), abs=1e-6), "a known result: when every pair has the same correlation, equal risk is weight in proportion to 1 / volatility"
    limited = risk_parity_weights(cov, constraints=Constraints(max_weight=0.3))
    assert limited.iloc[0] == pytest.approx(0.3) and limited.sum() == pytest.approx(1.0) and (limited.to_numpy() <= 0.3 + 1e-9).all() and limited.iloc[0] < parity.iloc[0], "the steadiest asset is cut to the cap and the rest take what it gave up"
    room = 0.3 - parity.iloc[1:]
    assert limited.iloc[1:].tolist() == pytest.approx((parity.iloc[1:] + (parity.iloc[0] - 0.3) * room / room.sum()).tolist(), abs=1e-6), "in proportion to their room under the cap"


def make_panel(n: int = 420, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scale = np.array([0.006, 0.01, 0.015, 0.02, 0.03, 0.012])
    return pd.DataFrame(rng.normal(0.0002, 1.0, (n, 6)) * scale, index=days(n), columns=list("ABCDEF"))


def test_the_book_solves_risk_parity_at_each_rebalance_date_on_the_window_before_it_and_holds_it_until_the_next():
    r = make_panel()
    investable = pd.DataFrame(True, index=r.index, columns=r.columns)
    marks = r.index[[150, 220, 300]]
    book = risk_parity_book(r, investable, lookback=100, rebalance_index=marks, covariance_method="sample", min_assets=5)
    assert (book.iloc[:150] == 0.0).all().all() and not book.isna().any().any(), "no book until a rebalance date with a full window behind it, and flat is zeros"
    for position in (150, 220, 300):
        window = r.iloc[position - 100:position]                                         # the 100 days before the date: its own return is not in it
        expected = risk_parity_weights(estimate_covariance(window, "sample", 100, 60.0, annualise=True))
        assert book.iloc[position].tolist() == pytest.approx(expected.tolist()), position
    assert book.iloc[151:220].eq(book.iloc[150]).all().all() and book.iloc[221:300].eq(book.iloc[220]).all().all() and book.iloc[301:].eq(book.iloc[300]).all().all(), "held between dates"
    assert not book.iloc[150].equals(book.iloc[220]) and book.iloc[150].sum() == pytest.approx(1.0)
    assert book.iloc[150]["A"] > book.iloc[150]["E"], "the quietest asset holds the most"
    changed = r.copy()
    changed.iloc[150] = 0.2
    again = risk_parity_book(changed, investable, lookback=100, rebalance_index=marks, covariance_method="sample", min_assets=5)
    assert again.iloc[:151].equals(book.iloc[:151]) and not again.iloc[220].equals(book.iloc[220]), "a return reaches the weights of the next date and not of its own"
    daily = risk_parity_book(r.iloc[:200], investable.iloc[:200], lookback=100, covariance_method="sample", min_assets=5)
    assert (daily.iloc[:100] == 0.0).all().all() and daily.iloc[100:].sum(axis=1).tolist() == pytest.approx([1.0] * 100), "with no dates given it is solved every day, from the first with a window"
    assert risk_parity_book(r.iloc[:200], investable.iloc[:200], lookback=100, rebalance_index=pd.DatetimeIndex(["2030-01-02"]), covariance_method="sample", min_assets=5).eq(0.0).all().all(), "a date that is not in the data is passed over"


def test_the_book_holds_only_what_can_be_held_with_enough_history_and_enough_names():
    r = make_panel()
    investable = pd.DataFrame(True, index=r.index, columns=r.columns)
    marks = r.index[[150, 300]]
    investable.loc[r.index[150], "F"] = False
    r.loc[r.index[:70], "E"] = np.nan                                                   # 80 of the 100 days: not more than 80%, so not enough history
    book = risk_parity_book(r, investable, lookback=100, rebalance_index=marks, covariance_method="sample", min_assets=4)
    assert book.iloc[150][["E", "F"]].tolist() == [0.0, 0.0] and book.iloc[150].sum() == pytest.approx(1.0), "a name that is not investable that day, or has too few returns in the window, is left out"
    window = r.iloc[50:150][["A", "B", "C", "D"]]
    expected = risk_parity_weights(estimate_covariance(window, "sample", 100, 60.0, annualise=True))
    assert book.iloc[150][["A", "B", "C", "D"]].tolist() == pytest.approx(expected.tolist())
    enough = make_panel()
    enough.loc[enough.index[:69], "E"] = np.nan                                         # 81 of the 100 days
    full = risk_parity_book(enough, investable, lookback=100, rebalance_index=marks, covariance_method="sample", min_assets=4)
    assert full.iloc[150]["E"] > 0.0, "more than 80% of the window is history enough"
    few = risk_parity_book(r, investable, lookback=100, rebalance_index=marks, covariance_method="sample", min_assets=5)
    assert (few.iloc[150:300] == 0.0).all().all() and few.iloc[300].sum() == pytest.approx(1.0), "four names are fewer than the five asked for: nothing is held that day; the next date has them"
    capped = risk_parity_book(r, investable, lookback=100, rebalance_index=marks, covariance_method="sample", min_assets=4, constraints=Constraints(max_weight=0.3))
    assert capped.iloc[300].max() <= 0.3 + 1e-9 and capped.iloc[300].sum() == pytest.approx(1.0)
    shrunk = risk_parity_book(make_panel(), pd.DataFrame(True, index=r.index, columns=r.columns), lookback=100, rebalance_index=marks, min_assets=5)
    assert not np.allclose(shrunk.iloc[150], risk_parity_book(make_panel(), pd.DataFrame(True, index=r.index, columns=r.columns), lookback=100, rebalance_index=marks, covariance_method="sample", min_assets=5).iloc[150]), "covariance is shrunk by default"


def test_the_book_asks_for_a_year_of_history_five_names_and_the_shrunk_covariance_unless_told_otherwise():
    r = make_panel(n=300)
    investable = pd.DataFrame(True, index=r.index, columns=r.columns)
    book = risk_parity_book(r, investable, rebalance_index=r.index[[251, 252, 290]], covariance_method="sample")
    assert (book.iloc[:252] == 0.0).all().all(), "252 days of history are needed before a date can be solved: the 252nd day has only 251 before it"
    window = r.iloc[:252]
    expected = risk_parity_weights(estimate_covariance(window, "sample", 252, 60.0, annualise=True))
    assert book.iloc[252].tolist() == pytest.approx(expected.tolist()), "the window is the 252 days before the date"
    four = risk_parity_book(r[["A", "B", "C", "D"]], investable[["A", "B", "C", "D"]], rebalance_index=r.index[[260]], covariance_method="sample")
    assert (four == 0.0).all().all(), "four names are fewer than five"
    five = risk_parity_book(r[["A", "B", "C", "D", "E"]], investable[["A", "B", "C", "D", "E"]], rebalance_index=r.index[[260]], covariance_method="sample")
    assert five.iloc[260].sum() == pytest.approx(1.0)
    slow, quick = (risk_parity_book(r, investable, rebalance_index=r.index[[260]], covariance_method="ewma", halflife=h).iloc[260] for h in (60.0, 10.0))
    default = risk_parity_book(r, investable, rebalance_index=r.index[[260]], covariance_method="ewma").iloc[260]
    assert default.tolist() == pytest.approx(slow.tolist()) and not np.allclose(slow, quick), "a half-life of 60 days for the exponentially weighted covariance, by default"
