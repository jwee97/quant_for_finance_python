"""Brinson-Fachler effects, Carino linking and risk contributions, checked against a worked example.

A planted-bug audit (``experiments/plant_bugs.py``) found ``backtest/attribution.py`` at 81% (38 of 47), and the survivors showed what the existing tests cannot see: they check that the three effects
*sum* to the active return, and a wrong allocation formula that sums to the same total (the sector's return plus the benchmark's instead of less it, which differs by a term that is zero when both
sets of weights sum to one) passes. The numbers of each sector are therefore stated here, from a portfolio small enough to do by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.attribution import brinson_fachler, carino_factor, cost_by_asset, cumulative, euler_risk_contributions, link_effects

SECTORS = {"A1": "S1", "A2": "S1", "B1": "S2"}


def one_period(wp, wb, returns) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.DatetimeIndex(["2021-01-29"])
    return (pd.DataFrame([wp], index=idx, columns=["A1", "A2", "B1"]), pd.DataFrame([wb], index=idx, columns=["A1", "A2", "B1"]), pd.DataFrame([returns], index=idx, columns=["A1", "A2", "B1"]))


def test_the_effects_of_each_sector_are_worked_out_by_hand():
    wp, wb, r = one_period([0.3, 0.2, 0.5], [0.2, 0.2, 0.6], [0.10, 0.02, -0.04])
    out = brinson_fachler(wp, wb, r, SECTORS)
    # S1: portfolio 0.5 at (0.03 + 0.004) / 0.5 = 6.8%, benchmark 0.4 at (0.02 + 0.004) / 0.4 = 6%.  S2: 0.5 against 0.6 at -4% on both sides.  Benchmark return 0.
    assert out.portfolio_return.iloc[0] == pytest.approx(0.014) and out.benchmark_return.iloc[0] == pytest.approx(0.0)
    assert list(out.allocation.columns) == ["S1", "S2"]
    assert out.allocation.iloc[0].tolist() == pytest.approx([(0.5 - 0.4) * (0.06 - 0.0), (0.5 - 0.6) * (-0.04 - 0.0)]) and out.allocation.iloc[0].tolist() == pytest.approx([0.006, 0.004]), "being over-weight a sector that beat the benchmark, and under-weight one that lost to it"
    assert out.selection.iloc[0].tolist() == pytest.approx([0.4 * (0.068 - 0.06), 0.0]) and out.selection.iloc[0].tolist() == pytest.approx([0.0032, 0.0])
    assert out.interaction.iloc[0].tolist() == pytest.approx([0.1 * 0.008, 0.0]) and out.interaction.iloc[0].tolist() == pytest.approx([0.0008, 0.0])
    assert out.total.iloc[0].tolist() == pytest.approx([0.006 + 0.0032 + 0.0008, 0.004]) and out.active_return().iloc[0] == pytest.approx(0.014) and out.identity_error() < 1e-15
    assert out.portfolio_sector_weight.iloc[0].tolist() == pytest.approx([0.5, 0.5]) and out.benchmark_sector_weight.iloc[0].tolist() == pytest.approx([0.4, 0.6])


def test_a_sector_the_other_side_does_not_hold_makes_no_allocation_or_selection_effect_for_the_benchmark_side():
    idx = pd.DatetimeIndex(["2021-01-29"])
    cols = ["A1", "B1", "C1"]
    sectors = {"A1": "S1", "B1": "S2", "C1": "S3"}
    wp = pd.DataFrame([[0.5, 0.2, 0.3]], index=idx, columns=cols)
    wb = pd.DataFrame([[0.6, 0.4, 0.0]], index=idx, columns=cols)
    r = pd.DataFrame([[0.05, -0.02, 0.10]], index=idx, columns=cols)
    out = brinson_fachler(wp, wb, r, sectors)
    rb = 0.6 * 0.05 + 0.4 * -0.02
    assert out.benchmark_return.iloc[0] == pytest.approx(rb)
    assert out.allocation["S3"].iloc[0] == pytest.approx(0.0) and out.selection["S3"].iloc[0] == pytest.approx(0.0), "the benchmark holds none of S3, so its return there is taken to be the benchmark's own: nothing to be right about"
    assert out.interaction["S3"].iloc[0] == pytest.approx(0.3 * (0.10 - rb)), "what the portfolio earned over the benchmark in the sector it alone holds is interaction"
    assert out.identity_error() < 1e-15
    no_portfolio = brinson_fachler(pd.DataFrame([[0.0, 0.4, 0.6]], index=idx, columns=cols), wb.assign(A1=0.4, B1=0.0, C1=0.6), r, sectors)
    assert np.isfinite(no_portfolio.total.to_numpy()).all() and no_portfolio.identity_error() < 1e-15


def test_weights_that_do_not_sum_to_one_are_refused_and_the_tolerance_is_a_millionth():
    wp, wb, r = one_period([0.3, 0.2, 0.5], [0.2, 0.2, 0.6], [0.10, 0.02, -0.04])
    with pytest.raises(ValueError, match=r"portfolio weights must sum to one in every period \(largest gap 1\.20e-06\)"):
        brinson_fachler(wp.assign(A1=0.3 + 1.2e-6), wb, r, SECTORS)
    with pytest.raises(ValueError, match="benchmark weights must sum to one"):
        brinson_fachler(wp, wb.assign(B1=0.6 + 1.2e-6), r, SECTORS)
    assert brinson_fachler(wp.assign(A1=0.3 + 5e-7), wb, r, SECTORS).identity_error() < 1e-5, "a gap of half a millionth is rounding"
    with pytest.raises(ValueError, match="largest gap 1.00e-01"):
        brinson_fachler(wp.assign(A1=0.4), wb, r, SECTORS)


def test_the_carino_factor_scales_a_periods_effects_so_they_link_to_the_cumulative_difference():
    assert carino_factor([0.10], [0.02])[0] == pytest.approx((np.log(1.10) - np.log(1.02)) / 0.08)
    assert carino_factor([0.05], [0.05])[0] == pytest.approx(1 / 1.05) and carino_factor(0.05, 0.05) == pytest.approx(1 / 1.05), "the limit when the two are equal"
    assert carino_factor([0.05], [0.05 + 1e-13])[0] == pytest.approx(1 / 1.05, abs=1e-9), "a gap of rounding noise is the limit, not a division by nothing"
    assert carino_factor([0.10, -0.03])[1] == pytest.approx(np.log(0.97) / -0.03), "no benchmark is a benchmark that earned nothing"
    assert cumulative([0.1, -0.1, 0.05]) == pytest.approx(1.1 * 0.9 * 1.05 - 1.0)
    idx = pd.date_range("2021-01-31", periods=3, freq="ME")
    rp, rb = pd.Series([0.04, -0.01, 0.03], index=idx), pd.Series([0.02, 0.01, 0.01], index=idx)
    effects = pd.DataFrame({"allocation": [0.01, 0.0, 0.005], "other": [0.01, -0.02, 0.015]}, index=idx)
    assert effects.sum(axis=1).tolist() == pytest.approx((rp - rb).tolist())
    linked = link_effects(effects, rp, rb)
    assert linked.sum() == pytest.approx(cumulative(rp) - cumulative(rb)), "linked, they add up to the cumulative active return"
    k = carino_factor(rp.to_numpy(), rb.to_numpy())
    big_k = float(carino_factor(cumulative(rp), cumulative(rb)))
    assert linked["allocation"] == pytest.approx(float((effects["allocation"].to_numpy() * k / big_k).sum()))
    contributions = link_effects(pd.DataFrame({"a": [0.02, 0.01], "b": [0.02, -0.02]}, index=idx[:2]), pd.Series([0.04, -0.01], index=idx[:2]))
    assert contributions.sum() == pytest.approx(cumulative([0.04, -0.01])), "with no benchmark, contributions link to the total return"


def test_risk_contributions_are_the_euler_components_and_a_book_with_no_risk_has_none():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
    w = pd.Series({"A": 0.6, "B": 0.4})
    out = euler_risk_contributions(w, cov)
    variance = 0.6 ** 2 * 0.04 + 0.4 ** 2 * 0.09 + 2 * 0.6 * 0.4 * 0.01
    sigma = np.sqrt(variance)
    assert out["weight"].tolist() == [0.6, 0.4] and out["marginal"].tolist() == pytest.approx([(0.04 * 0.6 + 0.01 * 0.4) / sigma, (0.01 * 0.6 + 0.09 * 0.4) / sigma])
    assert out["component"].tolist() == pytest.approx([0.6 * (0.04 * 0.6 + 0.01 * 0.4) / sigma, 0.4 * (0.01 * 0.6 + 0.09 * 0.4) / sigma]) and out["component"].sum() == pytest.approx(sigma)
    assert out["share"].tolist() == pytest.approx((out["component"] / sigma).tolist()) and out["share"].sum() == pytest.approx(1.0)
    flat = euler_risk_contributions(pd.Series({"A": 0.0, "B": 0.0}), cov)
    assert (flat[["marginal", "component", "share"]].to_numpy() == 0.0).all(), "a book with no positions has no risk to divide"
    reordered = euler_risk_contributions(pd.Series({"B": 0.4, "A": 0.6}), cov)
    assert reordered["weight"].tolist() == [0.6, 0.4], "weights are matched to the covariance by name"
    assert euler_risk_contributions(pd.Series({"A": 0.6}), cov)["weight"].tolist() == [0.6, 0.0], "an asset not in the book has no weight"


def test_the_cost_of_each_asset_is_its_traded_weight_times_its_rate():
    trades = pd.DataFrame({"A": [0.5, -0.25], "B": [0.0, 0.1]}, index=pd.date_range("2021-01-04", periods=2))
    out = cost_by_asset(trades, pd.Series({"A": 0.001, "B": 0.002}))
    assert out["A"].tolist() == pytest.approx([0.0005, 0.00025]) and out["B"].tolist() == pytest.approx([0.0, 0.0002])
