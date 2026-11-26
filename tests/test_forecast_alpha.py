"""From a forecast to an expected return a portfolio may use: the explicit conversions with answers worked out by hand, the horizon and availability rules, and the proof that nothing from after a decision date
enters an estimate (the data after the date are changed and the estimate must not move)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.alphalab import expected as ex
from src.forecast import alpha as fa
from src.portfolio import lab

D = pd.Timestamp("2022-03-01")


def rec(**kw):
    base = dict(instrument="X", origin=D, available_at=D, horizon_days=21, kind="return", units="log_return", value=0.02)
    return fa.ForecastRecord(**{**base, **kw})


# ------------------------------------------------------------------------------------------------------------------------ conversions
def test_a_volatility_forecast_is_never_an_expected_return():
    with pytest.raises(fa.ForecastConversionError, match="not an expected return"):
        fa.expected_simple_return(rec(kind="volatility", units="annualised_volatility", value=0.25))
    with pytest.raises(fa.ForecastConversionError, match="not an expected return"):
        fa.expected_simple_return(rec(kind="return", units="annualised_volatility", value=0.25))


def test_a_price_forecast_becomes_a_return_with_dividends_and_splits_handled_by_basis():
    adj = fa.expected_simple_return(rec(kind="price", units="price", value=110.0, last_price=100.0, price_basis="adjusted"))
    assert adj.value == pytest.approx(0.10) and adj.statistic == "point"
    raw = fa.expected_simple_return(rec(kind="price", units="price", value=60.0, last_price=100.0, price_basis="raw", split_factor=2.0, dividends=1.0))
    assert raw.value == pytest.approx((60.0 * 2 + 1.0) / 100.0 - 1.0)
    with pytest.raises(fa.ForecastConversionError, match="already include dividends"):
        fa.expected_simple_return(rec(kind="price", units="price", value=110.0, last_price=100.0, price_basis="adjusted", dividends=1.0))
    with pytest.raises(fa.ForecastConversionError, match="already adjusted for splits"):
        fa.expected_simple_return(rec(kind="price", units="price", value=110.0, last_price=100.0, price_basis="adjusted", split_factor=2.0))
    with pytest.raises(fa.ForecastConversionError, match="adjusted or raw"):
        fa.expected_simple_return(rec(kind="price", units="price", value=110.0, last_price=100.0))


def test_a_log_return_is_converted_to_the_mean_when_the_spread_is_known_and_labelled_the_median_when_it_is_not():
    with_sd = fa.expected_simple_return(rec(value=0.02, uncertainty=0.10))
    assert with_sd.value == pytest.approx(math.exp(0.02 + 0.005) - 1) and with_sd.statistic == "mean"
    assert with_sd.uncertainty == pytest.approx((1 + with_sd.value) * 0.10)
    no_sd = fa.expected_simple_return(rec(value=0.02))
    assert no_sd.value == pytest.approx(math.exp(0.02) - 1) and no_sd.statistic == "median" and any("median" in a for a in no_sd.assumptions)
    assert fa.expected_simple_return(rec(units="simple_return", value=0.03)).value == pytest.approx(0.03)


def test_a_distribution_is_converted_with_the_mean_of_the_simple_return_of_its_draws():
    draws = np.array([-0.10, -0.05, 0.0, 0.05, 0.10] * 8)
    out = fa.expected_simple_return(rec(kind="distribution", value=None, samples=tuple(draws)))
    assert out.value == pytest.approx(np.expm1(draws).mean()) and out.uncertainty == pytest.approx(np.expm1(draws).std(ddof=1)) and out.statistic == "mean"
    with pytest.raises(fa.ForecastConversionError, match="at least 30 draws"):
        fa.expected_simple_return(rec(kind="distribution", value=None, samples=(0.1, 0.2)))


def test_a_forecast_in_another_currency_needs_the_exchange_rate_forecast_and_compounds_with_it():
    r = rec(units="simple_return", value=0.05, currency="EUR")
    with pytest.raises(fa.ForecastConversionError, match="exchange rate"):
        fa.expected_simple_return(r, base_currency="USD")
    assert fa.expected_simple_return(r, "USD", fx_forecast=0.02).value == pytest.approx(1.05 * 1.02 - 1)
    assert fa.expected_simple_return(r, "EUR").value == pytest.approx(0.05)


def test_horizons_are_used_as_they_are_scaled_only_when_asked_and_never_more_than_three_times():
    er = fa.expected_simple_return(rec(units="simple_return", value=0.10, uncertainty=0.04, horizon_days=5))
    assert fa.align_horizon(er, 5) is er
    with pytest.raises(fa.ForecastConversionError, match="only at its own horizon"):
        fa.align_horizon(er, 10)
    s = fa.align_horizon(er, 10, allow_scaling=True)
    assert s.value == pytest.approx(1.10 ** 2 - 1) and s.uncertainty == pytest.approx(0.04 * math.sqrt(2)) and any("constant drift" in a for a in s.assumptions)
    with pytest.raises(fa.ForecastConversionError, match="more than 3"):
        fa.align_horizon(er, 21, allow_scaling=True)


def test_annualising_compounds_and_the_optimiser_scaling_is_a_unit_change_not_a_forecast():
    assert fa.annualise(0.01, 21) == pytest.approx(1.01 ** 12 - 1) and fa.annualise(0.01, 21) != pytest.approx(0.12)
    assert fa.to_optimiser_units(0.01, 21) == pytest.approx(0.01 * 252 / 21)


def test_a_forecast_is_usable_strictly_after_it_was_made_and_only_while_it_is_fresh():
    r = rec(origin=D, available_at=D)
    assert not fa.usable(r, D) and fa.usable(r, D + pd.Timedelta(days=1))
    assert fa.usable(r, D + pd.Timedelta(days=10), max_age_days=10) and not fa.usable(r, D + pd.Timedelta(days=11), max_age_days=10)
    later = rec(origin=D, available_at=D + pd.Timedelta(days=3))
    assert not fa.usable(later, D + pd.Timedelta(days=3)) and fa.usable(later, D + pd.Timedelta(days=4))
    with pytest.raises(fa.ForecastConversionError, match="before the day it was made"):
        rec(origin=D, available_at=D - pd.Timedelta(days=1))
    old, new = rec(origin=D - pd.Timedelta(days=5), available_at=D - pd.Timedelta(days=5)), rec(origin=D - pd.Timedelta(days=1), available_at=D - pd.Timedelta(days=1))
    assert fa.latest_usable([old, new], "X", D) is new and fa.latest_usable([old, new], "Y", D) is None


# ------------------------------------------------------------------------------------------------------------------------ generation, reliability and providers use only the past
def _prices(T=900, N=6, seed=1):
    rng = np.random.default_rng(seed)
    drift = np.linspace(-0.0002, 0.0006, N)
    r = rng.standard_normal((T, N)) * 0.01 + drift
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=pd.bdate_range("2018-01-01", periods=T), columns=[f"S{i}" for i in range(N)])


@pytest.fixture(scope="module")
def world():
    px = _prices()
    data = px.pct_change().iloc[1:]
    origins = [data.index[p - 1] for p in range(252, len(data) - 1, 21)]                 # the Lab's own rebalance schedule: the forecast is made on the close before each one
    records = fa.generate(px, "historical", origins, 21, n_paths=200, seed=3)             # the early origins have too little history for the method and make none
    late = [o for o in origins if px.index.get_loc(o) >= 520]
    return px, data, late, records


def test_generated_forecasts_come_with_the_contract_and_the_same_seed_gives_the_same_draws(world):
    px, data, origins, records = world
    assert len(records) >= len(origins) * px.shape[1] and {r.instrument for r in records} == set(px.columns)
    r = records[0]
    assert r.kind == "distribution" and r.units == "log_return" and r.horizon_days == 21 and r.model_id == "historical" and r.model_version == "1" and r.training_fingerprint
    assert r.available_at == r.origin and r.evaluation["status"] == "unevaluated" and r.provenance["history_days"] > 400 and r.to_dict()["origin"] == str(r.origin.date())
    again = fa.generate(px, "historical", origins[:1], 21, n_paths=200, seed=3)
    same = [x for x in records if x.origin == origins[0]]
    assert again[0].samples == same[0].samples and again[0].training_fingerprint == same[0].training_fingerprint
    with pytest.raises(fa.ForecastConversionError, match="choose a forecasting method with a mean"):
        fa.generate(px, "garch", origins[:1], 21)


def test_a_forecast_from_an_origin_is_unchanged_when_every_later_price_is_changed(world):
    px, data, origins, records = world
    o = origins[3]
    tampered = px.copy()
    tampered.loc[tampered.index > o] *= 7.0
    a = [r for r in records if r.origin == o]
    b = fa.generate(tampered, "historical", [o], 21, n_paths=200, seed=3)
    assert len(a) == len(b) and all(x.training_fingerprint == y.training_fingerprint for x, y in zip(a, b)), "the training data are the same"
    for x, y in zip(a, b):
        assert fa.expected_simple_return(x).value == pytest.approx(fa.expected_simple_return(y).value, abs=1e-12)


def test_reliability_counts_only_forecasts_whose_outcome_was_already_known(world):
    px, data, origins, records = world
    as_of = origins[10]
    base = fa.reliability(records, px, as_of)
    assert base["n"] > 0 and base["status"] in ("past-only", "too few resolved forecasts to say")
    tampered = px.copy()
    tampered.loc[tampered.index > as_of] *= 3.0
    assert fa.reliability(records, tampered, as_of) == base, "prices after the as-of day do not enter"
    resolved = [r for r in records if px.index.get_loc(r.origin) + r.horizon_days < len(px) and px.index[px.index.get_loc(r.origin) + r.horizon_days] <= as_of]
    assert base["n"] == len(resolved) or base["n"] < 20


def test_the_forecast_provider_uses_only_forecasts_available_before_the_decision_and_trusts_them_only_as_far_as_their_past_record(world):
    px, data, origins, records = world
    f = ex.forecast_provider(records, px, 21)
    p = data.index.get_loc(origins[10]) + 1
    est = f(p, data)
    assert est.info["forecasts_used"] == px.shape[1] and 0.0 <= est.info["weight"] <= 1.0 and est.info["reliability"]["n"] > 0
    late_records = [r for r in records if r.origin > data.index[p - 1]]
    only_future = ex.forecast_provider(late_records, px, 21)(p, data)
    assert only_future.mu.isna().all() and only_future.info["forecasts_used"] == 0, "forecasts made on or after the last known day are not used"
    stale = ex.forecast_provider(records, px, 21, max_age_days=3, calibrate="none")(p + 10, data)
    assert stale.mu.isna().all() and stale.info["stale"] == px.shape[1], "a record older than the allowed age is stale and not used, and the report counts them"
    uncal = ex.forecast_provider(records, px, 21, calibrate="none")(p, data)
    assert uncal.mu.notna().all() and uncal.info["weight"] == 1.0
    assert all(uncal.mu[a] == pytest.approx(fa.expected_simple_return(fa.latest_usable(records, a, data.index[p])).value) for a in data.columns)


def test_early_in_the_history_the_forecasts_get_no_weight_until_enough_of_them_have_resolved(world):
    px, data, origins, records = world
    p = data.index.get_loc(origins[0]) + 1
    est = ex.forecast_provider(records, px, 21)(p, data)
    assert est.info["weight"] == 0.0 and "too few" in est.info["calibration"] and est.mu.isna().all()


def test_the_factor_provider_does_not_move_when_prices_from_the_decision_day_on_are_changed(world):
    px, data, origins, records = world
    p = 700
    f = ex.factor_provider(px, "momentum", 21)
    a = f(p, data)
    tampered = px.copy()
    tampered.loc[tampered.index >= data.index[p]] *= 5.0
    b = ex.factor_provider(tampered, "momentum", 21)(p, tampered.pct_change().iloc[1:])
    assert a.info["ic_dates"] == b.info["ic_dates"] and a.info["ic"] == pytest.approx(b.info["ic"], abs=1e-12)
    assert np.allclose(a.mu.to_numpy(), b.mu.to_numpy(), atol=1e-12), "the factor's expected returns depend on prices before the decision only"
    assert a.info["ic_dates"] > 0


def test_the_combination_weights_use_only_rebalances_whose_horizon_was_complete(world):
    px, data, origins, records = world
    parts = {"factor": ex.factor_provider(px, "momentum", 21), "forecast": ex.forecast_provider(records, px, 21, calibrate="none")}
    ps = [data.index.get_loc(o) + 1 for o in origins[2:14]]

    def run(frame_px):
        comb = ex.combined_provider({"factor": ex.factor_provider(frame_px, "momentum", 21), "forecast": ex.forecast_provider(records, frame_px, 21, calibrate="none")}, "inverse_error", 21)
        d = frame_px.pct_change().iloc[1:]
        return [comb(p, d) for p in ps]
    base = run(px)
    last = base[-1]
    assert "inverse of past squared error" in last.info["how"] or "equal weights" in last.info["how"]
    assert abs(sum(last.info["weights"].values()) - 1.0) < 1e-12
    tampered = px.copy()
    tampered.loc[tampered.index >= data.index[ps[-1]]] *= 2.0
    again = run(tampered)
    assert again[-1].info["weights"] == pytest.approx(last.info["weights"]), "weights depend on completed outcomes only"
    assert np.allclose(again[-1].mu.to_numpy(), last.mu.to_numpy(), atol=1e-12, equal_nan=True)


# ------------------------------------------------------------------------------------------------------------------------ the optimisers get the same everything but the expected returns
def test_every_source_is_run_with_identical_risk_constraints_costs_and_schedule_and_only_max_sharpe_style_methods_move(world):
    px, data, origins, records = world
    methods = ["equal_weight", "min_variance", "max_sharpe"]
    common = dict(lookback=252, step=21, cost_bps=15.0, max_weight=0.6)
    hist = lab.compare(data, methods, **common)
    fore = lab.compare(data, methods, mu_fn=ex.forecast_provider(records, px, 21, calibrate="none"), mu_label="forecast", horizon=21, **common)
    fac = lab.compare(data, methods, mu_fn=ex.factor_provider(px, "momentum", 21), mu_label="factor", horizon=21, **common)
    for other in (fore, fac):
        assert other["rebalances"] == hist["rebalances"]
        for k in ("lookback", "step", "cost_bps", "cov_method", "max_weight", "risk_aversion", "mu_shrink"):
            assert other["config"][k] == hist["config"][k]
        pd.testing.assert_frame_equal(other["weights"]["min_variance"], hist["weights"]["min_variance"])      # min variance ignores expected returns: identical
        pd.testing.assert_frame_equal(other["weights"]["equal_weight"], hist["weights"]["equal_weight"])
        assert not other["weights"]["max_sharpe"].equals(hist["weights"]["max_sharpe"])
        assert other["table"].loc["min_variance", "sharpe"] == hist["table"].loc["min_variance", "sharpe"]
    assert fore["config"]["mu_source"] == "forecast" and fac["mu_label"] == "factor"
    rep = fore["mu_report"]
    first, last = rep[min(rep)], rep[max(rep)]
    assert first["filled_with_historical"] == first["assets"] and last["assets_with_estimate"] == last["assets"], "early rebalances have no forecast and fall back; later ones use it"
    mu_min = min(fore["mu_used"])
    assert np.allclose(fore["mu_used"][mu_min], hist["mu_used"][mu_min]), "with no forecast the baseline expected returns are used, so the rebalance is the same"


def test_transaction_costs_are_charged_the_same_way_whatever_the_source_of_expected_returns(world):
    px, data, origins, records = world
    cfg = dict(lookback=252, step=21, cost_bps=40.0)
    fore = lab.compare(data, ["max_sharpe"], mu_fn=ex.factor_provider(px, "momentum", 21), mu_label="factor", horizon=21, **cfg)
    zero = lab.compare(data, ["max_sharpe"], mu_fn=ex.factor_provider(px, "momentum", 21), mu_label="factor", horizon=21, lookback=252, step=21, cost_bps=0.0)
    day0 = fore["equity"]["max_sharpe"].index[0]
    assert fore["equity"]["max_sharpe"].iloc[0] - zero["equity"]["max_sharpe"].iloc[0] == pytest.approx(-40e-4, abs=1e-12), "the first day differs by exactly the cost of buying the whole book"
    from src.portfolio import attribution as at
    rp = at.replay(data, fore["weights"]["max_sharpe"], 21, 40.0)
    assert rp["cost"].iloc[0] == pytest.approx(40e-4) and (rp["net"] - (fore["equity"]["max_sharpe"].pct_change().fillna(fore["equity"]["max_sharpe"].iloc[0] - 1))).abs().max() < 1e-14
    assert day0 == rp["net"].index[0]


def test_diagnostics_report_how_the_weights_move_with_errors_in_the_forecast(world):
    px, data, origins, records = world
    res = lab.compare(data, ["min_variance", "max_sharpe"], lookback=252, step=21, mu_fn=ex.factor_provider(px, "momentum", 21), mu_label="factor", horizon=21)
    d = lab.diagnostics(res, data, n_draws=9, frontier_points=6)
    curve = d["methods"]["max_sharpe"]["mu_error_curve"]
    assert set(curve) == {"0.25", "0.5", "1.0", "2.0"} and curve["2.0"] >= curve["0.25"], "bigger errors in the expected returns move the weights more"
    assert d["methods"]["min_variance"]["mu_error_curve"] == {} and d["methods"]["min_variance"]["mu_sensitivity"] is None
