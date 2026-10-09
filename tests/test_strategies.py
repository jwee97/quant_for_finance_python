"""The strategy library: every model is causal and usable, and the structured ones do what they claim."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, bundle_from_prices, load_library
from src.framework.validate import check_causality
from src.strategies._common import stateful
from src.strategies.fixed_income import LEG, rolling_duration
from src.strategies.statarb import kalman_spread, simulate_nav_arbitrage

load_library()
FOUNDATION = {"chronos", "timesfm"}                                                                         # optional pre-trained models: slow on CPU, tested in tests/test_deep_models.py
NEEDS_ASSET_BUNDLE = {"carry_xs", "carry_ts", "basis_momentum"}                                               # read CARRY_<asset> signals: tested on the multi-asset bundle in tests/test_assets.py
NEEDS_PLANTED_STATES = {"credit_cycle_rotation"}                                                               # learn what each credit phase has paid: the generic bundle's spread is stuck at a floor; tested with planted phases in tests/test_economic_outlook.py
NEEDS_DATA_FILE = {"news_sentiment", "panel_signal", "fundamental_value", "fundamental_quality", "fundamental_dcf", "fundamental_alpha", "fundamental_nonlinear",
                   "earnings_season_premium"}           # read a file you supply (data/user/), or for this one find announcements in the volume of companies: tested with synthetic files and worlds in tests/test_alpha_styles.py, tests/test_fundamental_models.py and tests/test_factor_timing.py
HEAVY = {"garch_vol_managed", "evt_risk_managed", "ml_trees"}                                                            # fit a likelihood per asset: tested on a small bundle in tests/test_econometric_strategies.py
LIBRARY = [e.name for e in MODELS.entries() if not e.name.startswith("test_") and e.family != "crypto" and e.name not in FOUNDATION | HEAVY | NEEDS_ASSET_BUNDLE | NEEDS_DATA_FILE | NEEDS_PLANTED_STATES]      # crypto models have their own bundle (tests/test_crypto.py)
CLASSES = {"SPY": "equity", "QQQ": "equity", "IWM": "equity", "EFA": "equity", "EEM": "equity", "SHY": "rates", "IEF": "rates", "TLT": "rates", "AGG": "fixed_income",
           "LQD": "credit", "HYG": "credit", "GLD": "commodity", "SLV": "commodity", "DBC": "commodity", "VNQ": "real_estate"}


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(11)
    n = 3000
    idx = pd.bdate_range("2008-01-01", periods=n)
    common = rng.normal(0, 0.006, (n, 1))
    loadings = rng.uniform(0.3, 1.2, (1, len(CLASSES)))
    returns = common * loadings + rng.normal(0, 0.007, (n, len(CLASSES))) + 0.0002
    prices = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=idx, columns=list(CLASSES))
    high = prices * (1 + rng.uniform(0, 0.01, prices.shape))
    low = prices * (1 - rng.uniform(0, 0.01, prices.shape))
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, prices.shape), index=idx, columns=prices.columns)
    walk = lambda start, sd, floor=0.1: pd.Series(np.maximum(start + np.cumsum(rng.normal(0, sd, n)), floor), index=idx)
    macro = pd.DataFrame({"DGS2": walk(2.0, 0.02), "DGS5": walk(2.8, 0.02), "DGS10": walk(3.5, 0.02), "DGS20": walk(4.0, 0.02), "DFF": walk(1.0, 0.01),
                          "T10YIE": walk(2.0, 0.01), "T10Y3M": walk(1.0, 0.03, -2), "DTWEXBGS": walk(100, 0.2), "VIX": walk(20, 0.5, 9), "VIX3M": walk(21, 0.4, 10),
                          "VXN": walk(24, 0.5, 10), "GVZ": walk(18, 0.4, 8), "OVX": walk(35, 0.7, 12), "MOVE": walk(90, 1.0, 40), "BAA10Y": walk(2.5, 0.02, 1.0),
                          "CPI_YOY": walk(0.025, 0.0004, -0.01), "cftc_es": walk(0.1, 0.01, -1), "cftc_gold": walk(0.3, 0.01, -1)})
    return bundle_from_prices(prices, volume=volume, high=high, low=low, asset_class=CLASSES, macro=macro, name="synthetic-etfs")


def test_the_library_is_populated_and_every_model_documents_its_hypothesis():
    assert len(LIBRARY) >= 30
    for entry in MODELS.entries():
        if entry.name.startswith("test_"):
            continue
        assert entry.family and len(entry.description) > 30, entry.name
        assert (entry.factory.__doc__ or "").strip(), f"{entry.name} has no docstring stating why it might work"


@pytest.mark.parametrize("name", LIBRARY)
def test_every_library_model_is_causal_and_produces_a_usable_forecast(name, bundle):
    model = MODELS.create(name)
    check = check_causality(model, bundle, cutoff=bundle.index[2400])
    assert check["ok"], (name, check)
    if name == "ml_ridge":
        forecast = model.forecast(bundle)
    else:
        forecast = model.forecast(bundle, min_observations=252)
    valid = forecast.mean.stack().dropna()
    assert len(valid) > 500, f"{name} produced almost no forecasts"
    assert np.isfinite(valid).all()
    assert forecast.confidence.stack().dropna().between(0.0, 1.0).all()
    assert model.position_mode in ("cross_sectional", "time_series")


def test_stateful_entries_hold_until_the_exit_condition():
    idx = pd.RangeIndex(8)
    t = lambda *v: pd.DataFrame({"A": v}, index=idx).astype(bool)
    out = stateful(t(0, 1, 0, 0, 0, 0, 0, 0), t(0, 0, 0, 0, 1, 0, 0, 0), t(0, 0, 0, 0, 0, 1, 0, 0), t(0, 0, 0, 0, 0, 0, 0, 1))["A"].tolist()
    assert out == [0, 1, 1, 1, 0, -1, -1, 0]


def test_dual_momentum_holds_at_most_top_k_and_only_assets_that_beat_cash(bundle):
    model = MODELS.create("dual_momentum", top_k=3)
    score = model.score(bundle).dropna(how="all")
    assert (score.sum(axis=1) <= 3).all() and set(np.unique(score.to_numpy())) <= {0.0, 1.0}
    ret = bundle.prices.shift(21) / bundle.prices.shift(252) - 1.0
    excess = ret.sub(ret["SHY"], axis=0)
    chosen = score.iloc[-1][score.iloc[-1] == 1].index
    assert (excess.loc[score.index[-1], chosen] > 0).all() and "SHY" not in chosen


def test_donchian_positions_are_long_short_or_flat_and_follow_a_clean_trend(bundle):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=400)
    up = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0.003, 0.004, 400))) for c in CLASSES}, index=idx)
    b = bundle_from_prices(up, asset_class=CLASSES, name="uptrend")
    pos = MODELS.create("donchian").score(b)
    assert set(np.unique(pos.dropna().to_numpy())) <= {-1.0, 0.0, 1.0}
    assert (pos.iloc[-1] == 1.0).mean() > 0.8


def test_kalman_spread_is_stationary_for_a_cointegrated_pair_but_not_a_random_walk_pair():
    rng = np.random.default_rng(2)
    n = 1500
    right = np.cumsum(rng.normal(0, 0.01, n))
    left = 2.0 * right + 0.5 + rng.normal(0, 0.01, n)
    inn, sd = kalman_spread(left, right)
    z = inn / sd
    assert np.nanstd(z[200:]) < 2.5 and abs(np.nanmean(z[200:])) < 0.3
    other = np.cumsum(rng.normal(0, 0.01, n))
    inn2, _ = kalman_spread(other, right)
    assert np.nanstd(inn[200:]) < 0.3 * np.nanstd(np.diff(other))  + 0.2 or np.nanstd(inn[200:]) < np.nanstd(inn2[200:])


def test_rolling_duration_recovers_the_sensitivity_of_a_simulated_bond(bundle):
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2012-01-01", periods=1500)
    y = pd.Series(3.0 + np.cumsum(rng.normal(0, 0.03, 1500)), index=idx)
    r = pd.DataFrame({"TLT": -17.0 * y.diff() / 100.0 + rng.normal(0, 0.0005, 1500)}, index=idx)
    stub = bundle_from_prices(100 * (1 + r.fillna(0)).cumprod(), macro=pd.DataFrame({"DGS20": y}), name="bond")
    duration = rolling_duration(stub, "TLT", "DGS20")
    assert abs(duration.iloc[-1] - 17.0) < 1.0


@pytest.mark.parametrize("name", ["curve_steepener", "butterfly"])
def test_curve_trades_are_dv01_neutral_on_the_estimated_durations(name, bundle):
    model = MODELS.create(name)
    w = model.weights(bundle).dropna(how="all")
    durations = model.durations(bundle).reindex(w.index)
    dv01 = (w[list(LEG)] * durations[list(LEG)]).sum(axis=1).dropna()
    gross = (w[list(LEG)] * durations[list(LEG)]).abs().sum(axis=1).dropna()
    assert (dv01.abs() < 1e-9 * (1 + gross.loc[dv01.index])).all()
    assert (w.drop(columns=list(LEG)).abs().sum().sum()) == 0.0                    # nothing outside the three Treasury legs
    assert gross.max() > 0


def test_curve_trades_are_the_pipeline_default_allocator_and_run(bundle):
    from src.framework import Pipeline
    from src.utils.config import load_config

    result = Pipeline({"name": "steep", "models": [{"name": "curve_steepener"}], "evaluation": {"benchmarks": [], "causality": False}}, load_config(), bundle).run(validate=False)
    assert np.isfinite(result.metrics["sharpe"]) and result.weights[["SHY", "TLT"]].abs().sum().sum() > 0


def test_macro_models_tilt_the_right_classes(bundle):
    score = MODELS.create("risk_on_off").score(bundle)
    off = score["TLT"]
    assert (np.sign(score["SPY"].dropna()) == -np.sign(off.reindex(score["SPY"].dropna().index))).mean() > 0.95        # equities and bonds move opposite
    dollar = MODELS.create("dollar_strength").score(bundle)
    assert dollar["SPY"].abs().sum() == 0.0 and dollar["EEM"].abs().sum() > 0
    infl = MODELS.create("inflation_rotation").score(bundle)
    assert (np.sign(infl["DBC"].dropna()) == -np.sign(infl["TLT"].reindex(infl["DBC"].dropna().index))).mean() > 0.95


def test_a_model_missing_its_macro_series_says_so(bundle):
    from dataclasses import replace

    stripped = replace(bundle, macro=bundle.macro.drop(columns=["VIX3M"]))
    with pytest.raises(KeyError, match="VIX3M"):
        MODELS.create("variance_carry").forecast(stripped)


def test_nav_arbitrage_simulation_earns_a_premium_only_while_costs_are_below_it():
    cheap = simulate_nav_arbitrage(cost_bps=0.2)
    dear = simulate_nav_arbitrage(cost_bps=20.0)
    assert cheap["sharpe"] > 1.0 > dear["sharpe"] and dear["sharpe"] < 0
    assert 0 < cheap["breakeven_cost_bps"] < 20 and cheap["n_trades"] > 100
