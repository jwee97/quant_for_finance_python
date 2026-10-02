"""Walk-forward forecasting: causality, alignment, power and size.

The Clark-West test and the ridge learner were validated separately. What these
tests pin down is the glue between them, where look-ahead can hide: how month-end
features are paired with next-month targets, which rows are allowed into each
training window, and whether the full pipeline rejects a true null at about the
nominal rate while finding a planted signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.sleeves import month_end_dates, monthly_compound, sleeve_returns
from src.models.macro_forecast import (
    build_monthly_panel,
    evaluate_nested,
    walk_forward_forecasts,
)

COMPARISONS = [("hist", "price"), ("hist", "macro"), ("price", "both")]
KWARGS = dict(min_train_months=36, alphas=[30.0, 300.0], cv_splits=3)


def _synthetic(seed: int, signal: float = 0.0, years: int = 14, n_sleeves: int = 2):
    """Daily sleeve returns whose NEXT-month drift is signal * s_j, with s_j public at month-end j.

    ``signal`` is the monthly drift per unit of the macro state (1.5% gives roughly 10% R-squared).
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2005-01-03", periods=252 * years)
    months = days.to_period("M")
    labels = pd.PeriodIndex(sorted(set(months)))
    state = pd.Series(rng.standard_normal(len(labels)), index=labels)
    macro_daily = pd.DataFrame({
        "m1": state.reindex(months).to_numpy(),
        "m2": pd.Series(rng.standard_normal(len(labels)), index=labels).reindex(months).to_numpy(),
    }, index=days)
    drift = state.shift(1).fillna(0.0).reindex(months).to_numpy()          # last month's state moves THIS month
    per_day = np.array([(months == m).sum() for m in months])
    returns = pd.DataFrame(index=days)
    for j in range(n_sleeves):
        noise = 0.010 * rng.standard_normal(len(days))
        returns[f"s{j}"] = noise + (signal * drift / per_day if j == 0 else 0.0)
    cash = pd.Series(0.0001, index=days)
    return returns, cash, macro_daily


def _forecast(returns, cash, macro):
    panel = build_monthly_panel(returns, cash, macro)
    return panel, walk_forward_forecasts(panel, **KWARGS)


@pytest.fixture(scope="module")
def planted():
    """One walk-forward on data with a real macro signal in sleeve s0, shared by several tests."""
    returns, cash, macro = _synthetic(4, signal=0.015, years=12)
    panel, result = _forecast(returns, cash, macro)
    return returns, cash, macro, panel, result


def test_features_at_month_end_k_are_paired_with_the_return_over_month_k_plus_1(planted):
    _, _, _, panel, _ = planted
    pd.testing.assert_series_equal(panel.target["s0"].iloc[:-1],
                                   panel.realised["s0"].shift(-1).iloc[:-1], check_names=False)
    assert np.isnan(panel.target["s0"].iloc[-1])                           # the last month has no future yet
    assert panel.macro.index.equals(panel.dates)


def test_history_forecast_is_the_mean_of_targets_already_realised(planted):
    _, _, _, panel, result = planted
    hist = result["forecasts"]["hist"]["s0"].dropna()
    assert len(hist) > 60
    for when in (hist.index[0], hist.index[len(hist) // 2], hist.index[-1]):
        k = panel.dates.get_loc(when)
        realised_by_k = panel.realised["s0"].iloc[1:k + 1]                 # months 1..k, all known at month-end k
        assert hist.loc[when] == pytest.approx(realised_by_k.mean(), abs=1e-14)


def test_forecasts_do_not_depend_on_anything_after_the_forecast_date(planted):
    """Replace every return and macro reading after a cut-off. Forecasts made on or before it must
    not move by a single bit, for every model; forecasts after it must move."""
    returns, cash, macro, panel, base = planted
    cut = panel.dates[100]
    rng = np.random.default_rng(99)
    altered_returns, altered_macro = returns.copy(), macro.copy()
    after = returns.index > cut
    altered_returns.loc[after] = 0.02 * rng.standard_normal(altered_returns.loc[after].shape)
    altered_macro.loc[after] = 5.0 * rng.standard_normal(altered_macro.loc[after].shape)
    _, changed = _forecast(altered_returns, cash, altered_macro)
    for model in ("hist", "price", "macro", "both"):
        before = base["forecasts"][model].loc[:cut]
        assert before.notna().any().any(), model
        pd.testing.assert_frame_equal(before, changed["forecasts"][model].loc[:cut])
        assert not np.allclose(base["forecasts"][model].loc[cut:].dropna().to_numpy(),
                               changed["forecasts"][model].loc[cut:].dropna().to_numpy()), model


def test_a_planted_macro_signal_is_found_beyond_price_and_pure_noise_is_not(planted):
    *_, result = planted
    table = evaluate_nested(result, COMPARISONS, fdr=0.10).set_index(["sleeve", "comparison"])
    found = table.loc[("s0", "both_vs_price")]
    assert found["oos_r2"] > 0.02 and found["p_value"] < 0.01 and bool(found["bh_significant"])
    clean = table.loc[("s1", "both_vs_price")]                             # the sleeve with no signal
    assert clean["p_value"] > 0.05


def test_the_whole_pipeline_rejects_a_true_null_at_about_the_nominal_rate():
    """A leak anywhere in features, alignment or training would push this far above 10%."""
    p_values = []
    for seed in range(8):
        returns, cash, macro = _synthetic(100 + seed, signal=0.0, years=9, n_sleeves=1)
        panel = build_monthly_panel(returns, cash, macro)
        result = walk_forward_forecasts(panel, min_train_months=36, alphas=[100.0, 1000.0],
                                        cv_splits=3, models=("price", "both"))
        table = evaluate_nested(result, [("price", "both")], fdr=0.10)
        p_values.append(float(table["p_value"].iloc[0]))
    rejections = int(np.sum(np.array(p_values) < 0.10))
    assert rejections <= 4, f"{rejections}/8 rejections at the 10% level: {np.round(p_values, 3)}"
    assert np.mean(p_values) > 0.30                                        # not piled up near zero


def test_sleeve_returns_average_available_members_and_monthly_compounding_matches_by_hand():
    days = pd.bdate_range("2020-01-01", periods=45)
    frame = pd.DataFrame({"A": 0.01, "B": 0.03}, index=days)
    frame.loc[days[:10], "B"] = np.nan                                     # B has not started trading
    sleeve = sleeve_returns(frame, {"mix": ["A", "B"]})["mix"]
    assert sleeve.iloc[0] == pytest.approx(0.01) and sleeve.iloc[20] == pytest.approx(0.02)
    monthly = monthly_compound(frame["A"])
    january = frame["A"].loc["2020-01"]
    assert monthly.iloc[0] == pytest.approx((1 + january).prod() - 1)
    assert monthly.index[0] == january.index[-1]                           # stamped on the month's last trading day
    assert month_end_dates(days)[0] == january.index[-1]
