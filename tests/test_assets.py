"""Futures, FX, commodities and fixed income: closed forms against numerics, the continuous-contract identities, recovery of known parameters, and the cross-asset strategies."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.assets import bundles, commodities as com, futures as fut, fx, rates
from src.framework import MODELS, Pipeline, PipelineSpec, bundle_from_prices, load_library
from src.framework.validate import check_causality
from src.utils.config import load_config

warnings.filterwarnings("ignore")
load_library()


# --------------------------------------------------------------------------------------------------------------------- futures
def test_contract_calendars_follow_the_exchange_rules():
    eq = fut.contract_calendar("2021-01-01", "2021-12-31", (3, 6, 9, 12), "third_friday")
    inside = eq[(eq["expiry"] >= "2021-01-01") & (eq["expiry"] <= "2021-12-31")]
    assert len(inside) == 4 and (inside["expiry"].dt.weekday == 4).all() and inside["expiry"].dt.day.between(15, 21).all()
    assert inside["contract"].tolist() == ["H21", "M21", "U21", "Z21"]
    en = fut.contract_calendar("2021-01-01", "2021-06-30", rule="business_day_before_25th")
    assert (en["expiry"].dt.weekday < 5).all() and (en["expiry"].dt.day <= 22).all()
    assert fut.contract_calendar("2021-01-01", "2021-12-31", (1,), "15th")["expiry"].dt.weekday.max() < 5
    assert fut.contract_calendar("2021-01-01", "2021-03-31", rule="last_business_day")["expiry"].dt.weekday.max() < 5
    with pytest.raises(ValueError):
        fut.contract_calendar("2021-01-01", "2021-12-31", rule="nope")


def _two_contract_world():
    """Deterministic prices: contract A expires 2022-03-01, B expires 2022-04-01; B trades 2% above A (contango)."""
    dates = pd.bdate_range("2022-02-01", "2022-03-18")
    rows = []
    spot = 100.0 + np.arange(len(dates)) * 0.5
    for i, d in enumerate(dates):
        rows.append({"date": d, "contract": "A", "expiry": pd.Timestamp("2022-03-01"), "price": spot[i] if d <= pd.Timestamp("2022-03-01") else np.nan})
        rows.append({"date": d, "contract": "B", "expiry": pd.Timestamp("2022-04-01"), "price": spot[i] * 1.02})
    return pd.DataFrame(rows).dropna(subset=["price"]), dates


def test_ratio_adjustment_removes_the_roll_gap_exactly():
    long, dates = _two_contract_world()
    ratio = fut.continuous_series(long, roll_days=5, adjust="ratio")
    raw = fut.continuous_series(long, roll_days=5, adjust="none")
    roll = ratio.roll_dates[0]
    assert roll == pd.Timestamp("2022-02-24")
    i = dates.get_loc(roll)
    s_prev, s_roll = 100.0 + 0.5 * (i - 1), 100.0 + 0.5 * i
    assert raw.returns.loc[roll] == pytest.approx(1.02 * s_roll / s_prev - 1.0)            # the raw splice books the 2% basis as a return
    assert ratio.returns.loc[roll] == pytest.approx(s_roll / s_prev - 1.0)                 # the back-adjusted series shows only the genuine move
    assert ratio.price.iloc[-1] == pytest.approx(raw.price.iloc[-1])                      # the most recent price is untouched by back-adjustment
    diff = fut.continuous_series(long, roll_days=5, adjust="difference")
    assert diff.price.diff().loc[roll] == pytest.approx(0.5)                              # additive shift removes the gap, leaving the day's genuine move of one half point
    with pytest.raises(ValueError):
        fut.continuous_series(long, adjust="bad")


@pytest.fixture(scope="module")
def commodity_world():
    return fut.synthetic_term_structure(700, seed=2)


def test_ratio_adjusted_returns_equal_the_tradable_excess_return(commodity_world):
    long = commodity_world["long"]
    cs = fut.continuous_series(long, 5, "ratio")
    ex = fut.excess_return_series(long, 5)
    both = pd.concat([cs.returns, ex], axis=1).dropna()
    assert np.abs(both.iloc[:, 0] - both.iloc[:, 1]).max() < 1e-12 and len(cs.roll_dates) > 15
    raw_gap = np.abs(fut.continuous_series(long, 5, "none").returns.reindex(both.index) - both.iloc[:, 1]).max()
    assert raw_gap > 0.001                                                                # the unadjusted splice carries roll gaps as fake returns


def test_a_constant_spot_in_contango_pays_the_roll_yield_to_shorts():
    dates = pd.bdate_range("2022-01-03", periods=400)
    cal = fut.contract_calendar(dates[0], dates[-1], rule="15th")
    rows = []
    for d in dates:
        for _, c in cal[cal["expiry"] >= d].head(6).iterrows():
            rows.append({"date": d, "contract": c["contract"], "expiry": c["expiry"], "price": 100.0 * np.exp(0.05 * (c["expiry"] - d).days / 365.0)})
    long = pd.DataFrame(rows)
    ex = fut.excess_return_series(long, 3)
    assert ex.mean() * 252 == pytest.approx(-0.05, abs=0.004)                            # spot never moves: a long loses the 5% contango every year
    ts = fut.term_structure(long, 3)
    assert fut.carry(ts["price"], ts["ttm"]).dropna().mean() == pytest.approx(-0.05, abs=1e-3) and (ts["ttm"].diff(axis=1).iloc[:, 1:].dropna() > 0).all().all()


def test_term_structure_ranks_contracts_by_expiry_and_carry_has_the_right_sign():
    long = pd.DataFrame({"date": pd.to_datetime(["2022-01-03"] * 3), "contract": ["H", "M", "U"], "expiry": pd.to_datetime(["2022-03-15", "2022-06-15", "2022-09-15"]), "price": [102.0, 100.0, 98.0]})
    ts = fut.term_structure(long, 3)
    assert ts["price"].iloc[0].tolist() == [102.0, 100.0, 98.0] and ts["contract"].iloc[0].tolist() == ["H", "M", "U"]
    c = fut.carry(ts["price"], ts["ttm"])
    assert c.iloc[0] == pytest.approx(np.log(102 / 100) / ((pd.Timestamp("2022-06-15") - pd.Timestamp("2022-03-15")).days / 365.0)) and c.iloc[0] > 0      # backwardation: positive carry
    expired = long.assign(date=pd.to_datetime(["2022-04-01"] * 3))
    assert fut.term_structure(expired, 3)["price"].shape[1] == 2                          # the March contract has expired


def test_schwartz_smith_curve_properties():
    p = fut.SchwartzSmithParams()
    assert fut.schwartz_smith_A(0.0, p) == pytest.approx(0.0)
    steep = fut.SchwartzSmithParams(lambda_chi=0.0)
    back = fut.SchwartzSmithParams(lambda_chi=0.4)
    assert fut.schwartz_smith_A(1.0, back) < fut.schwartz_smith_A(1.0, steep)             # a higher risk premium on chi bends the curve toward backwardation
    carries = {}
    for lam in (-0.1, 0.0, 0.2):
        sim = fut.synthetic_term_structure(300, params=fut.SchwartzSmithParams(lambda_chi=lam), seed=1, noise=0.0)
        ts = fut.term_structure(sim["long"], 2)
        carries[lam] = fut.carry(ts["price"], ts["ttm"]).mean()
    assert carries[-0.1] < carries[0.0] < carries[0.2]
    sim = fut.synthetic_term_structure(200, params=fut.SchwartzSmithParams(seasonal_amp=0.1, seasonal_peak_month=1), seed=1, noise=0.0)
    last = sim["long"][sim["long"]["date"] == sim["long"]["date"].max()]
    months = pd.to_datetime(last["expiry"]).dt.month.to_numpy()
    assert last["log_price"].to_numpy()[np.isin(months, [1, 2])].mean() > last["log_price"].to_numpy()[np.isin(months, [7, 8])].mean() - 0.5


def test_kalman_estimation_recovers_the_schwartz_smith_factors_and_parameters():
    truth = fut.SchwartzSmithParams()
    sim = fut.synthetic_term_structure(500, params=truth, seed=4)
    fit = com.fit_schwartz_smith(sim["long"], depth=5, x0=(1.0, 0.3, 0.15, 0.2, 0.0, 0.05, 0.0, 0.004))
    p = fit["params"]
    assert fit["converged"] and p.kappa == pytest.approx(truth.kappa, rel=0.3) and p.sigma_chi == pytest.approx(truth.sigma_chi, rel=0.15) and p.sigma_xi == pytest.approx(truth.sigma_xi, rel=0.15)
    assert fit["sigma_eps"] == pytest.approx(0.002, rel=0.3)
    assert np.corrcoef(fit["chi"], sim["chi"].reindex(fit["chi"].index))[0, 1] > 0.95 and np.corrcoef(fit["xi"], sim["xi"].reindex(fit["xi"].index))[0, 1] > 0.9


def test_basis_momentum_and_curve_factors_and_seasonality(commodity_world):
    long = commodity_world["long"]
    ts = fut.term_structure(long, 3)
    cr = fut.contract_returns(ts["price"], ts["contract"])
    assert cr.iloc[0].isna().all() and cr.iloc[5:].notna().to_numpy().mean() > 0.95            # NaN only on the days a held contract has just expired
    bm = fut.basis_momentum(cr, 126)
    assert np.isfinite(bm.dropna()).all() and bm.dropna().shape[0] > 400
    cf = com.curve_factors(ts["price"])
    back = fut.synthetic_term_structure(100, params=fut.SchwartzSmithParams(lambda_chi=0.5), seed=2, noise=0.0)
    assert com.curve_factors(fut.term_structure(back["long"], 3)["price"])["slope"].mean() < com.curve_factors(fut.term_structure(fut.synthetic_term_structure(100, params=fut.SchwartzSmithParams(lambda_chi=-0.2), seed=2, noise=0.0)["long"], 3)["price"])["slope"].mean()
    assert list(cf.columns) == ["level", "slope", "curvature"]
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000", periods=4000)
    r = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    r[idx.month == 7] += 0.004
    sf = com.seasonal_factors(r)
    assert sf["excess_vs_average"].idxmax() == 7 and abs(sf.loc[7, "t_vs_january"]) > 3 and sf.shape[0] == 12


# --------------------------------------------------------------------------------------------------------------------------- FX
def test_covered_interest_parity_round_trips_and_triangular_arbitrage_is_absent():
    f = fx.forward_rate(1.10, 0.03, 0.01, 0.5)
    assert f == pytest.approx(1.10 * np.exp(0.01)) and fx.forward_points(1.10, 0.03, 0.01, 0.5) == pytest.approx((f - 1.10) / 1e-4)
    assert fx.implied_rate_differential(1.10, f, 0.5) == pytest.approx(0.02)
    idx = pd.bdate_range("2022", periods=50)
    eurusd, usdjpy = pd.Series(np.linspace(1.1, 1.2, 50), index=idx), pd.Series(np.linspace(110, 120, 50), index=idx)
    assert np.abs(fx.triangular_gap(eurusd, usdjpy, eurusd * usdjpy)).max() < 1e-12
    assert fx.triangular_gap(eurusd, usdjpy, eurusd * usdjpy * 1.001).iloc[0] == pytest.approx(np.log(1.001))
    assert (fx.cross_rate(usdjpy, usdjpy) == 1.0).all()


def test_excess_return_includes_carry_with_a_lag_and_fama_regression_recovers_a_planted_slope():
    idx = pd.bdate_range("2020", periods=300)
    spot = pd.Series(np.ones(300), index=idx)
    rb, rq = pd.Series(0.05, index=idx), pd.Series(0.01, index=idx)
    ex = fx.excess_return(spot, rb, rq)
    assert ex.iloc[1:].mean() * 252 == pytest.approx(0.04) and np.isnan(ex.iloc[0])
    rng = np.random.default_rng(1)
    n = 6000
    diff = pd.Series(0.01 * np.sin(np.arange(n) / 150.0) + rng.normal(0, 0.003, n).cumsum() * 0.02, index=pd.bdate_range("2000", periods=n))
    ln_s = pd.Series(np.zeros(n), index=diff.index)
    inc = 0.5 * diff.shift(21).fillna(0.0) * 21 / 252.0 / 21 + rng.normal(0, 0.003, n)
    ln_s = inc.cumsum()
    res = fx.fama_regression(np.exp(ln_s), pd.Series(0.0, index=diff.index), diff, horizon=21)
    assert {"beta", "se", "t_vs_one", "t_vs_zero", "nobs"} <= set(res) and res["se"] > 0


def test_synthetic_fx_carry_earns_the_non_uip_share_of_the_differential_with_crash_risk():
    out = {}
    for beta in (1.0, 0.0):
        rets, skews, diffs = [], [], []
        for seed in range(4):
            m = fx.synthetic_fx_market(4000, 8, uip_beta=beta, seed=seed)
            r = fx.carry_portfolio_returns(m["excess_returns"], m["rates"], m["usd_rate"]).dropna()
            d = m["rates"].sub(m["usd_rate"], axis=0)
            rets.append(r.mean() * 252)
            skews.append(r.skew())
            diffs.append(d.iloc[:, -2:].mean().mean() - d.iloc[:, :2].mean().mean())
        out[beta] = (np.mean(rets), np.mean(skews), np.mean(diffs))
    assert abs(out[1.0][0]) < 0.02 and out[0.0][0] == pytest.approx(out[0.0][2], abs=0.025)
    assert out[0.0][1] < -0.5                                                              # the premium comes with crashes
    m = fx.synthetic_fx_market(500, 4, seed=1)
    assert m["spot"].shape == (500, 4) and (m["spot"] > 0).all().all() and m["volatility_factor"].min() > 0


# --------------------------------------------------------------------------------------------------------------------- rates
def test_bond_analytics_are_internally_consistent():
    assert rates.bond_price(10, 0.05, 0.05) == pytest.approx(100.0)
    p = rates.bond_price(10, 0.05, 0.06)
    assert p < 100 and rates.yield_to_maturity(p, 10, 0.05) == pytest.approx(0.06, abs=1e-10)
    h = 1e-5
    num = -(rates.bond_price(10, 0.05, 0.05 + h) - rates.bond_price(10, 0.05, 0.05 - h)) / (2 * h * 100.0)
    assert rates.duration(10, 0.05, 0.05) == pytest.approx(num, rel=1e-6)
    assert rates.duration(10, 0.05, 0.05, kind="macaulay") == pytest.approx(rates.duration(10, 0.05, 0.05) * (1 + 0.05 / 2))
    conv = (rates.bond_price(10, 0.05, 0.05 + 1e-4) + rates.bond_price(10, 0.05, 0.05 - 1e-4) - 2 * 100.0) / (1e-8 * 100.0)
    assert rates.convexity(10, 0.05, 0.05) == pytest.approx(conv, rel=1e-3)
    assert rates.dv01(10, 0.05, 0.05) == pytest.approx(rates.duration(10, 0.05, 0.05) * 100 * 1e-4, rel=0.02) and rates.dv01(10, 0.05, 0.05) > 0
    assert rates.duration(30, 0.04, 0.04) > rates.duration(5, 0.04, 0.04) > rates.duration(1, 0.04, 0.04) > 0
    assert rates.duration(10, 0.0, 0.05, kind="macaulay") == pytest.approx(10.0)          # a zero-coupon bond's duration is its maturity


def test_zero_curve_bootstrap_reprices_the_par_bonds_and_forwards_are_consistent():
    zc = rates.bootstrap_zero_curve([0.5, 1, 2, 5, 10], [0.02, 0.025, 0.03, 0.035, 0.04])
    f = lambda t: float(np.interp(t, zc.index, zc.values))                                  # noqa: E731
    for m, y in ((1, 0.025), (2, 0.03), (5, 0.035), (10, 0.04)):
        assert rates.bond_price_from_curve(m, y, f) == pytest.approx(100.0, abs=1e-8)
    assert rates.forward_rate(f, 1.0, 2.0) == pytest.approx((f(2.0) * 2 - f(1.0) * 1) / 1.0)
    assert zc.iloc[0] == pytest.approx(np.log(1 + 0.02 / 2) * 2, rel=1e-9)
    krd = rates.key_rate_durations(10, 0.05, f)
    assert krd.sum() == pytest.approx(rates.duration(10, 0.05, 0.05), rel=0.06) and krd.idxmax() == 10.0 and krd[20.0] == pytest.approx(0.0, abs=1e-9)


def test_nelson_siegel_recovers_parameters_and_dynamic_factors():
    tau = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30.0])
    y = rates.nelson_siegel(tau, 0.04, -0.02, -0.01, 0.6)
    fit = rates.fit_nelson_siegel(tau, y, np.linspace(0.2, 1.2, 101))
    assert (fit["beta0"], fit["beta1"], fit["beta2"], fit["lambda"]) == pytest.approx((0.04, -0.02, -0.01, 0.6), abs=1e-6) and fit["rmse"] < 1e-10
    assert rates.nelson_siegel(1e-6, 0.04, -0.02, 0.0, 0.5) == pytest.approx(0.02, abs=1e-5) and rates.nelson_siegel(300.0, 0.04, -0.02, 0.0, 0.5) == pytest.approx(0.04, abs=1e-3)
    assert rates.svensson(tau, 0.04, -0.02, -0.01, 0.0, 0.6, 1.0) == pytest.approx(y)
    panel = rates.synthetic_curve_panel(600, seed=1)
    d = rates.dynamic_nelson_siegel(panel, 0.5)
    assert d["rmse"] < 2e-4 and list(d["factors"].columns) == ["level", "slope", "curvature"]
    fc = rates.forecast_curve(d["factors"], panel.columns, 0.5, 21)
    assert len(fc) == len(panel.columns) and abs(fc.iloc[-1] - panel.iloc[-1, -1]) < 0.01


def test_carry_and_rolldown_depend_on_the_slope_of_the_curve():
    flat = lambda t: 0.03                                                                    # noqa: E731
    out = rates.carry_rolldown(10, 0.03, flat, 0.25)
    assert out["rolldown"] == pytest.approx(0.0, abs=1e-12) and out["carry"] == pytest.approx(0.03 * 0.25 / 1.0, rel=1e-6)
    steep = lambda t: 0.02 + 0.01 * np.log1p(t)                                              # noqa: E731
    up = rates.carry_rolldown(5, steep(5), steep, 0.25)
    assert up["rolldown"] > 0 and up["total"] > up["carry"]
    assert rates.carry_rolldown(5, steep(5), steep, 0.25, financing=0.04)["carry"] < up["carry"]
    inverted = lambda t: 0.05 - 0.01 * np.log1p(t)                                           # noqa: E731
    assert rates.carry_rolldown(5, inverted(5), inverted, 0.25)["rolldown"] < 0


# ------------------------------------------------------------------------------------------------------- bundles and strategies
@pytest.fixture(scope="module")
def world():
    return bundles.multi_asset_demo_bundle(1100, seed=5)


def test_multi_asset_bundle_is_well_formed_and_every_instrument_carries_a_signal(world):
    b, parts = world
    assert b.name == "synthetic-multi-asset" and b.prices.shape[0] == 1100
    assert set(b.asset_class.values()) == {"commodity", "equity", "bond", "fx"}
    for a in b.assets:
        assert f"CARRY_{a}" in b.macro.columns
    assert b.prices.apply(lambda s: s.dropna().iloc[0]).between(85, 115).all()                 # indices start at 100 and have moved by their first return
    assert (b.returns.abs().max() < 1.0).all() and b.investable.iloc[-1].all()
    fb = bundles.futures_bundle({"CRUDE": parts["CRUDE"]["long"]})
    assert fb.assets == ["CRUDE"] and {"CARRY_CRUDE", "SLOPE_CRUDE", "BASISMOM_CRUDE"} <= set(fb.macro.columns)
    ex = fut.excess_return_series(parts["CRUDE"]["long"], 5)
    common = fb.returns["CRUDE"].dropna().index.intersection(ex.index)
    assert len(common) > 600 and np.allclose(fb.returns.loc[common, "CRUDE"].to_numpy(), ex.loc[common].to_numpy())
    f2 = bundles.fx_bundle(parts["fx"])
    assert f2.macro.shape[1] == len(f2.assets) and (f2.macro.iloc[:, 0] == parts["fx"]["rates"].iloc[:, 0] - parts["fx"]["usd_rate"]).all()


@pytest.mark.parametrize("name,kwargs", [("carry_xs", {}), ("carry_ts", {}), ("basis_momentum", {}), ("long_term_reversal", {"lookback": 500, "skip": 60})])
def test_cross_asset_models_are_causal_and_forecast(world, name, kwargs):
    b, _ = world
    model = MODELS.create(name, **kwargs)
    assert check_causality(model, b, cutoff=b.index[800])["ok"]
    score = model.score(b)
    assert score.stack().dropna().shape[0] > 1000 and np.isfinite(score.stack().dropna()).all()
    assert model.forecast(b, min_observations=252).mean.stack().dropna().shape[0] > 500


def test_within_class_standardisation_and_class_restrictions(world):
    b, _ = world
    z = MODELS.create("carry_xs").score(b).dropna(how="all")
    last = z.iloc[-1]
    for cls in ("commodity", "equity", "fx"):
        members = [a for a in b.assets if b.asset_class[a] == cls]
        assert last[members].mean() == pytest.approx(0.0, abs=1e-9)
    bm = MODELS.create("basis_momentum").score(b).dropna(how="all")
    assert set(bm.columns[bm.notna().any()]) == {a for a in b.assets if b.asset_class[a] == "commodity"}
    assert MODELS.create("carry_ts").score(b).dropna(how="all").abs().max().max() <= 1.0


def test_carry_models_make_money_when_carry_is_the_return_and_refuse_bundles_without_signals():
    rng = np.random.default_rng(3)
    n = 1200
    idx = pd.bdate_range("2015", periods=n)
    carry = pd.DataFrame({f"A{i}": 0.04 * np.sin(np.arange(n) / 300.0 + i) for i in range(6)}, index=idx)
    rets = carry.shift(1).fillna(0.0) / 252.0 * 4.0 + rng.normal(0, 0.006, (n, 6))
    prices = 100 * (1 + rets).cumprod()
    b = bundle_from_prices(prices, asset_class={c: "fx" for c in prices.columns}, macro=carry.add_prefix("CARRY_"), name="planted-carry")
    cfg = load_config()
    res = Pipeline(PipelineSpec.from_dict({"name": "carry_ts", "models": [{"name": "carry_ts"}], "allocation": {"allocator": "sleeves"}}), cfg, b).run()
    assert res.metrics["sharpe"] > 0.5
    plain = bundle_from_prices(prices, name="no-signals")
    for name in ("carry_xs", "carry_ts", "basis_momentum"):
        with pytest.raises(KeyError):
            MODELS.create(name).score(plain)
    with pytest.raises(ValueError):
        MODELS.create("carry_xs", clip=0)
    with pytest.raises(ValueError):
        MODELS.create("long_term_reversal", lookback=100, skip=90)


def test_multi_asset_bundle_runs_through_the_pipeline_with_a_combination_of_models(world):
    b, _ = world
    spec = PipelineSpec.from_dict({"name": "carry+trend", "models": [{"name": "carry_ts"}, {"name": "tsmom"}], "combination": {"rule": "equal"}, "allocation": {"allocator": "forecast_stack", "params": {"mode": "time_series"}}})
    res = Pipeline(spec, load_config(), b).run()
    assert np.isfinite(res.net_returns.dropna()).all() and len(res.net_returns.dropna()) > 500 and res.weights.abs().sum(axis=1).max() > 0
