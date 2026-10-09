"""Portfolio-rebalance styles: the rebalancing rule does what it says, the month-end flow sits where it should, and flight to quality and the market outlook react to a planted crash and a planted bull market."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, bundle_from_prices, load_library
from src.framework.validate import check_causality
from src.strategies._common import month_end_flags
from src.strategies.rebalance_styles import HAVENS, stress_gauge

load_library()
CLASSES = {"EQ1": "equity", "EQ2": "equity", "EQ3": "equity", "EQ4": "real_estate", "BD1": "rates", "BD2": "fixed_income"}


def _bundle(returns: pd.DataFrame, classes: dict | None = None):
    prices = 100.0 * (1.0 + returns.fillna(0.0)).cumprod()
    return bundle_from_prices(prices, asset_class=classes or {c: CLASSES.get(c, "equity") for c in returns.columns}, name="synthetic")


def _world(n: int = 2600, seed: int = 3, columns=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-02", periods=n)
    columns = columns or list(CLASSES)
    common = rng.normal(0.0004, 0.006, (n, 1))
    r = common * np.array([1.0 if CLASSES.get(c) in ("equity", "real_estate") else -0.1 for c in columns]) + rng.normal(0, 0.004, (n, len(columns)))
    return pd.DataFrame(r, index=idx, columns=columns)


# ------------------------------------------------------------------------------------------------------------------------------- policy portfolio
def _two_asset(growth_a: float = 0.002, n: int = 900):
    idx = pd.bdate_range("2012-01-02", periods=n)
    return _bundle(pd.DataFrame({"A": growth_a, "B": 0.0}, index=idx), {"A": "equity", "B": "rates"})


def _month_end_weights(model, bundle) -> pd.DataFrame:
    w = model.weights(bundle)
    return w[month_end_flags(bundle.index).to_numpy()].dropna(how="all")


def test_policy_portfolio_never_rebalancing_lets_the_winner_take_over_exactly_as_the_market_moved():
    b = _two_asset()
    w = _month_end_weights(MODELS.create("policy_portfolio", targets={"A": 0.6, "B": 0.4}, calendar="never", band=10.0), b)
    assert np.allclose(w.iloc[0].to_numpy(), [0.6, 0.4])                                                    # bought at the policy weights
    grown = (b.prices.loc[w.index] / b.prices.loc[w.index[0]])
    expected = 0.6 * grown["A"] / (0.6 * grown["A"] + 0.4 * grown["B"])
    assert np.allclose(w["A"].to_numpy(), expected.to_numpy(), atol=1e-9)
    assert w["A"].iloc[-1] > 0.85


def test_policy_portfolio_calendar_restores_the_policy_on_schedule_and_between_dates_the_weights_drift():
    b = _two_asset()
    w = _month_end_weights(MODELS.create("policy_portfolio", targets={"A": 0.6, "B": 0.4}, calendar="quarterly", band=10.0), b)
    restored = np.isclose(w["A"].to_numpy(), 0.6)
    assert restored[0] and restored[3] and restored[6] and restored[9]                                       # months 0, 3, 6, 9 ...
    assert not restored[[1, 2, 4, 5, 7, 8]].any()
    monthly = _month_end_weights(MODELS.create("policy_portfolio", targets={"A": 0.6, "B": 0.4}, calendar="monthly", band=10.0), b)
    assert np.allclose(monthly["A"].to_numpy(), 0.6)


def test_policy_portfolio_band_restores_the_policy_only_when_a_weight_has_drifted_beyond_it():
    b = _two_asset(0.002)
    band = 0.05
    model = MODELS.create("policy_portfolio", targets={"A": 0.6, "B": 0.4}, calendar="never", band=band)
    w = _month_end_weights(model, b)
    assert ((w["A"] - 0.6).abs() <= band + 1e-12).all()                                                      # no month-end is left outside the band
    restores = np.isclose(w["A"].to_numpy(), 0.6)
    assert restores.sum() >= 3 and restores.sum() < len(w)                                                   # restored sometimes, not always
    tighter = _month_end_weights(MODELS.create("policy_portfolio", targets={"A": 0.6, "B": 0.4}, calendar="never", band=0.02), b)
    assert np.isclose(tighter["A"].to_numpy(), 0.6).sum() > restores.sum()                                   # a narrower band restores more often


def test_policy_portfolio_presets_follow_model_portfolio_and_missing_assets_are_left_out():
    b = _bundle(_world(700), None)
    w = MODELS.create("policy_portfolio", preset="60_40", calendar="monthly").weights(b).dropna(how="all")
    last = w.iloc[-1]
    assert last["EQ1"] + last["EQ2"] + last["EQ3"] + last["EQ4"] == pytest.approx(0.6)                      # SPY is not in the universe: the equity class stands in for it
    assert last["BD1"] + last["BD2"] == pytest.approx(0.4)
    assert last.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("kwargs", [{"preset": "nope"}, {"calendar": "daily"}, {"band": 0.0}, {"targets": {"A": -1.0}}, {"targets": {}}, {"targets": {"A": 0.0}}])
def test_policy_portfolio_rejects_nonsense(kwargs):
    with pytest.raises(ValueError):
        MODELS.create("policy_portfolio", **kwargs)


def test_policy_portfolio_is_causal():
    b = _bundle(_world(1800))
    assert check_causality(MODELS.create("policy_portfolio", calendar="quarterly", band=0.03), b, cutoff=b.index[1400])["ok"]


# ------------------------------------------------------------------------------------------------------------------------------- the month-end flow
def _flow_world():
    """Five years of months in which the risky asset beats the safe one by a lot in half of them and lags by a lot in the others, in the first fifteen days of each month; quiet otherwise."""
    idx = pd.bdate_range("2010-01-04", periods=1300)
    month = idx.year * 12 + idx.month
    first_days = pd.Series(idx.to_series().groupby(month).cumcount().to_numpy(), index=idx)
    sign = pd.Series(np.where(pd.Series(month, index=idx).map(lambda m: m % 2 == 0), 1.0, -1.0), index=idx)
    rng = np.random.default_rng(2)
    risky = np.where(first_days < 15, 0.004 * sign, 0.0) + rng.normal(0, 0.001, len(idx))
    safe = rng.normal(0, 0.001, len(idx))
    return _bundle(pd.DataFrame({"EQ": risky, "BD": safe}, index=idx), {"EQ": "equity", "BD": "rates"}), sign


def test_rebalancing_flow_leans_against_the_expected_flow_in_the_window_before_the_month_end_and_is_flat_otherwise():
    b, sign = _flow_world()
    s = MODELS.create("rebalancing_flow", days=3, lead=2, min_months=12).score(b)
    idx = b.index
    month = pd.Series(idx.year * 12 + idx.month, index=idx)
    left = month.groupby(month).cumcount(ascending=False) + 1
    window = (left > 2) & (left <= 5) & (month < month.iloc[-1]) & (month > month.iloc[0] + 14)               # after a year of monthly history
    outside = ~((left > 2) & (left <= 5))
    assert (s.loc[outside & (month > month.iloc[0] + 14), "EQ"] == 0).all()                                   # nothing outside the three days
    assert (s.loc[window, "EQ"] * sign[window] < 0).mean() > 0.95                                             # a strong month for risky assets: short risky ...
    assert (s.loc[window, "BD"] * sign[window] > 0).mean() > 0.95                                             # ... and long safe
    assert s.loc[window, "EQ"].abs().min() > 0.3
    assert (s.loc[month == month.iloc[-1], "EQ"].fillna(0) == 0).all()                                        # the data's last month may be unfinished: its end is unknown


def test_rebalancing_flow_can_drop_the_safe_leg_and_needs_both_kinds_of_asset():
    b, _ = _flow_world()
    s = MODELS.create("rebalancing_flow", min_months=12, safe_leg=False).score(b)
    assert (s["BD"].fillna(0) == 0).all()
    only_risky = _bundle(_world(500, columns=["EQ1", "EQ2"]))
    with pytest.raises(KeyError, match="asset of class"):
        MODELS.create("rebalancing_flow").score(only_risky)


@pytest.mark.parametrize("kwargs", [{"days": 0}, {"days": 11}, {"lead": -1}, {"scale": 0.0}, {"min_months": 6}])
def test_rebalancing_flow_rejects_nonsense(kwargs):
    with pytest.raises(ValueError):
        MODELS.create("rebalancing_flow", **kwargs)


# ------------------------------------------------------------------------------------------------------------------------------- flight to quality
def _crash_world(with_gold: bool = False):
    """1500 calm days, a 60-day crash (risky -0.8%/day with big swings, bonds +0.15%/day), then calm again."""
    n = 1700
    idx = pd.bdate_range("2009-01-05", periods=n)
    rng = np.random.default_rng(8)
    risky = rng.normal(0.0004, 0.006, (n, 3))
    safe = rng.normal(0.0001, 0.002, (n, 2))
    crash = slice(1500, 1560)
    risky[crash] = rng.normal(-0.008, 0.02, (60, 3))
    safe[crash] = rng.normal(0.0015, 0.004, (60, 2))
    cols = ["EQ1", "EQ2", "EQ3", "BD1", "BD2"]
    data = np.hstack([risky, safe])
    if with_gold:
        data = np.hstack([data, rng.normal(0.0, 0.008, (n, 1))])
        cols.append("GLD")
    classes = {"EQ1": "equity", "EQ2": "equity", "EQ3": "equity", "BD1": "rates", "BD2": "rates", "GLD": "commodity"}
    return _bundle(pd.DataFrame(data, index=idx, columns=cols), classes)


def test_flight_to_quality_is_long_risk_when_calm_and_runs_to_quality_in_a_crash():
    b = _crash_world()
    model = MODELS.create("flight_to_quality", threshold=0.5)
    s = model.score(b)
    calm, crash = s.iloc[1300:1480], s.iloc[1540:1560]
    assert calm["EQ1"].mean() > 0.6 and (calm["EQ1"] >= 0).all() and calm["BD1"].mean() < 0.1                 # calm: risky assets on (never short), safe assets mostly off
    assert crash["EQ1"].mean() < -0.3 and crash["BD1"].mean() > 0.3                                           # crash: short risky, long safe
    gauge = stress_gauge(b)
    assert gauge["stress"].dropna().between(0.0, 1.0).all()
    assert gauge["stress"].iloc[1540:1560].mean() > 0.6 > gauge["stress"].iloc[1300:1480].mean() + 0.2


def test_flight_to_quality_without_the_short_leg_only_steps_aside_and_treats_gold_as_a_haven():
    b = _crash_world(with_gold=True)
    s = MODELS.create("flight_to_quality", short_risky=False).score(b)
    assert (s[["EQ1", "EQ2", "EQ3"]].dropna() >= 0).all().all()
    assert s["GLD"].iloc[1540:1560].mean() > 0.1 and s["GLD"].iloc[1300:1480].abs().max() < 1e-12             # gold earns a score only in flight, never as a risky asset
    assert "GLD" in HAVENS


def test_flight_to_quality_needs_both_kinds_of_asset_and_a_history_and_rejects_nonsense():
    with pytest.raises(KeyError, match="asset of class"):
        MODELS.create("flight_to_quality").score(_bundle(_world(500, columns=["EQ1", "EQ2"])))
    s = MODELS.create("flight_to_quality").score(_crash_world())
    assert s.iloc[:200].isna().all().all()                                                                   # the gauge needs a year of its own history first
    for bad in ({"threshold": 0.0}, {"threshold": 1.0}, {"high_window": 5}, {"vol_window": 2}, {"corr_window": 5}):
        with pytest.raises(ValueError):
            MODELS.create("flight_to_quality", **bad)


# ------------------------------------------------------------------------------------------------------------------------------- market outlook
def _trend_world(drift: float, n: int = 1500):
    idx = pd.bdate_range("2009-01-05", periods=n)
    rng = np.random.default_rng(4)
    risky = rng.normal(drift, 0.004, (n, 3))
    safe = rng.normal(0.0001, 0.002, (n, 2))
    return _bundle(pd.DataFrame(np.hstack([risky, safe]), index=idx, columns=["EQ1", "EQ2", "EQ3", "BD1", "BD2"]), {"EQ1": "equity", "EQ2": "equity", "EQ3": "equity", "BD1": "rates", "BD2": "rates"})


def test_market_outlook_is_bullish_in_a_steady_rally_and_bearish_in_a_steady_decline():
    up, down = MODELS.create("market_outlook"), MODELS.create("market_outlook")
    bull, bear = _trend_world(0.0015), _trend_world(-0.0015)
    o_up, o_down = up.outlook(bull).iloc[600:], down.outlook(bear).iloc[600:]
    assert o_up.between(-1, 1).all() and o_down.between(-1, 1).all()
    assert o_up.mean() > 0.4 and o_down.mean() < -0.4
    s_up, s_down = up.score(bull).iloc[600:], down.score(bear).iloc[600:]
    assert (s_up["EQ1"] > 0).mean() > 0.95 and (s_up["BD1"] == 0).mean() > 0.95                             # a bullish view is long risk and does not short safety
    assert (s_down["EQ1"] < 0).mean() > 0.95 and (s_down["BD1"] > 0).mean() > 0.95                          # a bearish view shorts risk and moves into safety


def test_market_outlook_bias_and_tilt_move_the_view_and_the_scores():
    b = _trend_world(0.0)
    plain = MODELS.create("market_outlook")
    biased = MODELS.create("market_outlook", bias=0.5, tilt=2.0)
    o, ob = plain.outlook(b), biased.outlook(b)
    assert np.allclose(ob.dropna().to_numpy(), o.dropna().to_numpy())                                         # the bias is applied to the view, not to the readings
    sp, sb = plain.score(b)["EQ1"].dropna(), biased.score(b)["EQ1"].dropna()
    assert np.allclose(sb.to_numpy(), 2.0 * np.clip(o.loc[sb.index] + 0.5, -1, 1).to_numpy())


def test_market_outlook_needs_both_kinds_of_asset_and_rejects_nonsense():
    with pytest.raises(KeyError, match="asset of class"):
        MODELS.create("market_outlook").score(_bundle(_world(500, columns=["EQ1", "EQ2"])))
    for bad in ({"trend": 5}, {"momentum": 5}, {"tilt": 0.0}, {"bias": 2.0}, {"min_periods": 10}):
        with pytest.raises(ValueError):
            MODELS.create("market_outlook", **bad)


def test_the_rebalance_style_models_declare_two_assets_and_a_sleeve_book_where_that_is_true():
    for name in ("rebalancing_flow", "flight_to_quality", "market_outlook"):
        m = MODELS.create(name)
        assert m.required_assets() == 2 and m.book == "sleeves" and m.position_mode == "time_series"
    assert MODELS.create("policy_portfolio").structured
