"""Portfolio overlays and liquidation costs: the hedge removes the beta it claims to, the cap caps, the cost-aware optimiser agrees with brute force and trades less, and liquidation arithmetic matches the closed forms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.algo.liquidation import liquidation_horizon, liquidation_profile
from src.framework import ALLOCATORS, Pipeline, PipelineSpec, bundle_from_prices
from src.framework.allocation import Context
from src.framework.allocators_overlay import TcaMeanVariance, inner_required, rolling_beta
from src.framework.forecasting import ForecastModel
from src.framework.types import ForecastPanel
from src.portfolio.constraints import Constraints
from src.portfolio.mean_variance import mean_variance_weights
from src.utils.config import load_config

N, K = 2400, 6
BETAS = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 0.2])
NAMES = [f"M{i}" for i in range(K)]


class Constant(ForecastModel):
    """A model whose score is the same every day: the book the ``sleeves`` allocator builds from it is an equal-weight long book."""

    name, position_mode = "constant", "time_series"

    def score(self, data):
        return pd.DataFrame(1.0, index=data.index, columns=data.assets)


def _returns(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=N)
    factor = rng.normal(0.0003, 0.010, (N, 1))
    return pd.DataFrame(factor * BETAS + rng.normal(0, 0.004, (N, K)), index=idx, columns=NAMES)


def _bundle(returns: pd.DataFrame | None = None, volume: pd.DataFrame | None = None):
    returns = _returns() if returns is None else returns
    prices = 100.0 * (1.0 + returns).cumprod()
    return bundle_from_prices(prices, volume=volume, name="overlay")


def _ctx(bundle, **kwargs) -> Context:
    return Context(bundle=bundle, config=load_config(), models=[Constant()], **kwargs)


SLEEVES = {"allocator": "sleeves"}


# ------------------------------------------------------------------------------------------------------------------------------- beta neutral
def _net_beta(weights: pd.DataFrame, bundle, window: int = 252) -> pd.Series:
    r = bundle.returns.where(bundle.investable)
    return (weights * rolling_beta(r, r.mean(axis=1), window)).sum(axis=1, min_count=1)


def test_beta_neutral_by_projection_removes_the_book_beta_keeps_its_gross_and_goes_long_short():
    b = _bundle()
    inner = ALLOCATORS.create("sleeves").build(_ctx(b))
    w = ALLOCATORS.create("beta_neutral", inner=SLEEVES, window=252).build(_ctx(b))
    ready = w.abs().sum(axis=1) > 0
    assert ready.iloc[300:].all() and not ready.iloc[:100].any()                                             # nothing before the beta window has half filled
    assert _net_beta(inner, b).iloc[400:].mean() > 0.7
    assert _net_beta(w, b).loc[ready].abs().max() < 1e-9
    assert np.allclose(w.abs().sum(axis=1)[ready], inner.abs().sum(axis=1)[ready], atol=1e-9)
    assert (w.loc[ready].min(axis=1) < 0).mean() > 0.9                                                       # the long-only book gained short positions


def test_beta_neutral_book_is_uncorrelated_with_the_market_out_of_sample_while_the_unhedged_book_is_not():
    b = _bundle()
    market = b.returns.mean(axis=1)
    inner = ALLOCATORS.create("sleeves").build(_ctx(b))
    hedged = ALLOCATORS.create("beta_neutral", inner=SLEEVES).build(_ctx(b))
    pnl = lambda w: (w.shift(1) * b.returns).sum(axis=1).iloc[800:]                                         # noqa: E731  (weights set the day before earn the day's return)
    assert pnl(inner).corr(market.iloc[800:]) > 0.9
    assert abs(pnl(hedged).corr(market.iloc[800:])) < 0.2


def test_beta_neutral_with_a_hedge_instrument_trades_only_that_instrument():
    b = _bundle()
    inner = ALLOCATORS.create("sleeves").build(_ctx(b))
    w = ALLOCATORS.create("beta_neutral", inner=SLEEVES, hedge="M2", keep_gross=False).build(_ctx(b))
    ready = w.abs().sum(axis=1) > 0
    others = [c for c in NAMES if c != "M2"]
    assert np.allclose(w.loc[ready, others].to_numpy(), inner.loc[ready, others].to_numpy())
    assert _net_beta(w, b).loc[ready].abs().max() < 1e-9
    assert (w.loc[ready, "M2"] < inner.loc[ready, "M2"]).all()                                               # it sold the hedge to bring the beta down


def test_beta_neutral_target_beta_leaves_the_book_with_that_exposure():
    b = _bundle()
    w = ALLOCATORS.create("beta_neutral", inner=SLEEVES, target_beta=0.5, keep_gross=False).build(_ctx(b))
    ready = w.abs().sum(axis=1) > 0
    assert np.allclose(_net_beta(w, b).loc[ready], 0.5, atol=1e-9)


def test_beta_neutral_uses_no_data_after_the_date_it_stamps():
    b = _bundle()
    cut = b.index[1800]
    noisy = _returns()
    later = noisy.index > cut
    noisy.loc[later] = np.random.default_rng(1).normal(0, 0.03, noisy.loc[later].shape)
    real = ALLOCATORS.create("beta_neutral", inner=SLEEVES).build(_ctx(b)).loc[:cut]
    fake = ALLOCATORS.create("beta_neutral", inner=SLEEVES).build(_ctx(_bundle(noisy))).loc[:cut]
    assert np.allclose(real.to_numpy(), fake.to_numpy(), atol=1e-12)


def test_beta_neutral_declares_two_assets_rejects_nonsense_and_names_a_missing_hedge():
    assert ALLOCATORS.create("beta_neutral", inner=SLEEVES).required_assets() == 2
    assert inner_required({"book": "equal_weight"}) == 1 and inner_required({"book": "hrp"}) == 2 and inner_required(None) == 2
    with pytest.raises(ValueError):
        ALLOCATORS.create("beta_neutral", window=20)
    with pytest.raises(ValueError):
        ALLOCATORS.create("beta_neutral", max_hedge=0.0)
    with pytest.raises(KeyError, match="hedge"):
        ALLOCATORS.create("beta_neutral", inner=SLEEVES, hedge="SPY").build(_ctx(_bundle()))


# ------------------------------------------------------------------------------------------------------------------------------- liquidity cap
def _volume(returns: pd.DataFrame, shares: list[float]) -> pd.DataFrame:
    return pd.DataFrame(np.tile(np.array(shares, float), (len(returns), 1)), index=returns.index, columns=returns.columns)


def test_liquidity_cap_limits_each_position_to_what_can_be_sold_in_the_stated_days_and_keeps_the_excess_as_cash():
    r = _returns()
    b = _bundle(r, _volume(r, [1e3, 1e7, 1e7, 1e7, 1e7, 1e7]))                                               # M0 trades almost nothing
    cap = ALLOCATORS.create("liquidity_cap", inner=SLEEVES, aum=1e8, participation=0.1, days=5.0)
    inner = ALLOCATORS.create("sleeves").build(_ctx(b))
    w = cap.build(_ctx(b))
    limit = cap.limit(b)
    dollars = (b.prices * b.volume).rolling(20, min_periods=10).median()
    assert np.allclose(limit.dropna().to_numpy(), (0.1 * 5.0 * dollars / 1e8).dropna().to_numpy())
    live = limit.notna().all(axis=1) & (inner.abs().min(axis=1) > 0)                                           # the dates on which every asset is investable and in the book
    assert (w.loc[live].abs() <= limit.loc[live] + 1e-12).all().all()
    assert np.allclose(w.loc[live, "M0"], limit.loc[live, "M0"])                                              # capped at its limit ...
    assert np.allclose(w.loc[live, NAMES[1:]].to_numpy(), inner.loc[live, NAMES[1:]].to_numpy())             # ... the liquid ones untouched
    assert (w.loc[live].sum(axis=1) < inner.loc[live].sum(axis=1)).all()                                      # what was clipped stays in cash


def test_liquidity_cap_gives_no_position_where_volume_is_unknown_and_needs_volume():
    r = _returns()
    v = _volume(r, [1e7] * K)
    v.iloc[:, 3] = np.nan
    w = ALLOCATORS.create("liquidity_cap", inner=SLEEVES, aum=1e6).build(_ctx(_bundle(r, v)))
    assert (w["M3"] == 0).all() and (w["M0"].iloc[100:] > 0).all()
    with pytest.raises(KeyError, match="volume"):
        ALLOCATORS.create("liquidity_cap", inner=SLEEVES).build(_ctx(_bundle(r)))
    for bad in ({"aum": 0.0}, {"participation": 0.0}, {"participation": 1.5}, {"days": 0.0}, {"window": 2}):
        with pytest.raises(ValueError):
            ALLOCATORS.create("liquidity_cap", **bad)


# ------------------------------------------------------------------------------------------------------------------------------- optimisation with TCA
def _tca(**kwargs) -> TcaMeanVariance:
    return ALLOCATORS.create("tca_mvo", **kwargs)


def _problem(n: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.1, (n, n))
    cov = a @ a.T + 0.02 * np.eye(n)
    return rng.normal(0.05, 0.05, n), cov


def test_tca_optimiser_agrees_with_brute_force_on_a_two_asset_problem_and_never_does_worse_than_staying_put():
    mu, cov = np.array([0.10, 0.02]), np.array([[0.04, 0.01], [0.01, 0.03]])
    held = np.array([0.5, 0.5])
    sigma, adv = np.array([0.012, 0.015]), np.array([2e8, 1e8])
    model = _tca(risk_aversion=4.0, aum=5e9, spread_bps=6.0)
    cons = Constraints(min_weight=0.0, max_weight=1.0, net_exposure=1.0, long_only=True)
    w = model.optimise(mu, cov, held, sigma, adv, ["A", "B"], cons)
    a, c, d = model.cost_terms(sigma, adv)

    def utility(w1: float) -> float:
        x = np.array([w1, 1.0 - w1])
        trade = np.abs(x - held)
        cost = (a * trade + c * trade ** (1 + model.beta) + d * trade ** 2).sum()
        return float(x @ mu - 2.0 * x @ cov @ x - model.periods * cost)

    grid = np.linspace(0, 1, 100001)
    best = grid[np.argmax([utility(g) for g in grid[::20]]) * 20]
    assert w is not None and w[0] == pytest.approx(best, abs=2e-3)
    assert utility(w[0]) >= utility(0.5) - 1e-12
    assert 0.5 < w[0] < 1.0                                                                                  # it moves toward the better asset, but not all the way (the cost of the trade)


def test_tca_with_no_costs_is_plain_mean_variance():
    mu, cov = _problem(5, 3)
    held = np.full(5, 0.2)
    cons = Constraints(min_weight=0.0, max_weight=0.5, net_exposure=1.0, long_only=True)
    free = _tca(risk_aversion=5.0, spread_bps=0.0, eta=0.0, gamma=0.0).optimise(mu, cov, held, np.full(5, 0.01), np.full(5, 1e8), list("ABCDE"), cons)
    plain = mean_variance_weights(pd.Series(mu, index=list("ABCDE")), pd.DataFrame(cov, index=list("ABCDE"), columns=list("ABCDE")), 5.0, cons)
    assert free is not None and np.allclose(free, plain.weights.to_numpy(), atol=2e-4)


def test_tca_leaves_the_book_alone_when_the_gain_is_smaller_than_the_cost_of_the_first_unit_traded():
    cov = np.diag([0.04, 0.04])
    held = np.array([0.5, 0.5])
    cons = Constraints(min_weight=0.0, max_weight=1.0, net_exposure=1.0, long_only=True)
    model = _tca(risk_aversion=5.0, spread_bps=20.0)                                                         # 10 bps each way, 12 times a year: about 0.0012 a year of slope
    tiny = model.optimise(np.array([0.0006, 0.0]), cov, held, np.full(2, 0.01), np.full(2, 1e9), ["A", "B"], cons)
    assert np.allclose(tiny, held, atol=1e-7)                                                                # the forecast edge is worth less than the spread
    big = model.optimise(np.array([0.05, 0.0]), cov, held, np.full(2, 0.01), np.full(2, 1e9), ["A", "B"], cons)
    free = _tca(risk_aversion=5.0, spread_bps=0.0, eta=0.0, gamma=0.0).optimise(np.array([0.05, 0.0]), cov, held, np.full(2, 0.01), np.full(2, 1e9), ["A", "B"], cons)
    assert free[0] == pytest.approx(0.625, abs=1e-4)                                                         # the cost-free answer: the weight gap is (mu_A - mu_B) / (risk_aversion * variance)
    assert 0.53 < big[0] < free[0] - 0.03                                                                     # a real edge is traded on, but stops short where the saved cost equals the gain


def _panel(bundle, signs: np.ndarray, edge: float = 0.05):
    """A forecast panel whose view flips every month-end: even months favour the first half of the assets, odd months the second."""
    idx = bundle.index
    month = pd.Series(idx.year * 12 + idx.month, index=idx)
    flip = np.where((month.rank(method="dense") % 2 == 0).to_numpy()[:, None], signs[None, :], -signs[None, :])
    mean = pd.DataFrame(edge / 12.0 * flip, index=idx, columns=bundle.assets)                                 # per month, as a 21-day forecast
    std = pd.DataFrame(0.05, index=idx, columns=bundle.assets)
    return ForecastPanel(mean, std, pd.DataFrame(0.5, index=idx, columns=bundle.assets), 21, "flip")


def _turnover(w: pd.DataFrame) -> float:
    marks = w.loc[w.diff().abs().sum(axis=1) > 1e-12]
    return float(marks.diff().abs().sum().sum())


def test_tca_trades_far_less_than_cost_free_optimisation_on_a_forecast_that_flips_every_month():
    r = _returns()
    b = _bundle(r, _volume(r, [2e6] * K))
    panel = _panel(b, np.array([1, 1, 1, -1, -1, -1.0]))
    free = _tca(spread_bps=0.0, eta=0.0, gamma=0.0).build(_ctx(b, forecasts=panel))
    costly = _tca(spread_bps=15.0, aum=2e10).build(_ctx(b, forecasts=panel))
    live = costly.abs().sum(axis=1) > 0
    assert live.sum() > 1500
    for w in (free, costly):
        assert np.allclose(w.loc[live].sum(axis=1), 1.0, atol=1e-6) and (w.loc[live] >= -1e-9).all().all()
        assert (w.loc[live] <= load_config().get("portfolio.constraints.max_weight", 0.25) + 1e-9).all().all()
    assert _turnover(free) > 40 and _turnover(costly) < 0.2 * _turnover(free)                                # a view that flips every month is chased with full trades only when trading is free


def test_tca_needs_forecasts_and_volume_and_rejects_nonsense():
    r = _returns()
    with pytest.raises(ValueError, match="forecasts"):
        _tca().build(_ctx(_bundle(r, _volume(r, [1e6] * K))))
    b = _bundle(r)
    with pytest.raises(KeyError, match="volume"):
        _tca().build(_ctx(b, forecasts=_panel(b, np.ones(K))))
    for bad in ({"risk_aversion": 0.0}, {"aum": 0.0}, {"spread_bps": -1.0}, {"eta": -1.0}, {"beta": 0.0}, {"gamma": -1.0}, {"periods": 0}, {"lookback": 10}):
        with pytest.raises(ValueError):
            _tca(**bad)
    assert _tca().required_assets() == 2


def test_tca_runs_end_to_end_through_the_pipeline():
    r = _returns()
    b = bundle_from_prices(100.0 * (1.0 + r).cumprod(), volume=_volume(r, [3e6] * K), name="tca")
    spec = PipelineSpec.from_dict({"name": "tca", "models": [{"name": "momentum"}], "allocation": {"allocator": "tca_mvo", "params": {"aum": 1e8}}, "evaluation": {"benchmarks": [], "causality": False}})
    result = Pipeline(spec, load_config(), b).run(validate=False)
    held = result.weights.loc[result.weights.abs().sum(axis=1) > 0]
    assert len(held) > 1000 and np.allclose(held.sum(axis=1), 1.0, atol=1e-5)


# ------------------------------------------------------------------------------------------------------------------------------- liquidation costs
def test_liquidation_profile_matches_the_closed_forms():
    values = pd.Series({"A": 5e6, "B": -2e7, "C": 1e5, "D": 0.0})
    adv = pd.Series({"A": 5e7, "B": 4e7, "C": 1e6, "D": 1e7})
    sigma = pd.Series({"A": 0.012, "B": 0.02, "C": 0.03, "D": 0.01})
    p = liquidation_profile(values, adv, sigma, spread=0.0004, participation=0.10)
    assert p.loc["A", "days"] == pytest.approx(1.0) and p.loc["B", "days"] == pytest.approx(5.0) and p.loc["D", "days"] == 0.0
    b = p.loc["B"]                                                                                           # five equal daily slices of 10% of the day's volume
    assert b["spread_bps"] == pytest.approx(2.0)
    assert b["temporary_bps"] == pytest.approx(1e4 * 0.142 * 0.02 * 0.1 ** 0.6)
    assert b["permanent_bps"] == pytest.approx(1e4 * 0.5 * 0.30 * 0.02 * (2e7 / 4e7))
    assert b["cost_bps"] == pytest.approx(b["spread_bps"] + b["temporary_bps"] + b["permanent_bps"])
    assert b["dollars"] == pytest.approx(2e7 * b["cost_bps"] * 1e-4)
    assert b["risk_bps"] == pytest.approx(1e4 * 0.02 * np.sqrt(0.8 ** 2 + 0.6 ** 2 + 0.4 ** 2 + 0.2 ** 2))    # the shares still held at the end of each day
    assert p.loc["A", "risk_bps"] == 0.0                                                                     # done within the day
    assert p.loc["C", "days"] == pytest.approx(1.0) and p.loc["D", "cost_bps"] == 0.0


def test_liquidation_costs_rise_with_size_and_fall_with_a_slower_pace():
    adv, sigma = pd.Series({"X": 1e8}), pd.Series({"X": 0.015})
    small, big = (liquidation_profile(pd.Series({"X": v}), adv, sigma).loc["X"] for v in (1e6, 1e8))
    assert big["cost_bps"] > small["cost_bps"] and big["days"] > small["days"] and big["risk_bps"] > small["risk_bps"]
    fast, slow = (liquidation_profile(pd.Series({"X": 5e7}), adv, sigma, participation=p).loc["X"] for p in (0.25, 0.05))
    assert fast["temporary_bps"] > slow["temporary_bps"] and fast["risk_bps"] < slow["risk_bps"] and fast["days"] < slow["days"]


def test_liquidation_profile_marks_positions_it_cannot_price_as_unliquidatable_and_validates():
    p = liquidation_profile(pd.Series({"A": 1e6, "B": 1e6}), pd.Series({"A": 1e7, "B": 0.0}), pd.Series({"A": 0.01, "B": 0.01}))
    assert np.isfinite(p.loc["A", "cost_bps"]) and np.isinf(p.loc["B", "cost_bps"]) and np.isinf(p.loc["B", "days"])
    with pytest.raises(ValueError):
        liquidation_profile(pd.Series({"A": 1.0}), pd.Series({"A": 1.0}), pd.Series({"A": 0.01}), participation=0.0)


def test_liquidation_horizon_is_the_days_to_raise_each_share_of_the_book_in_parallel():
    values = pd.Series({"A": 5e6, "B": -2e7, "C": 1e5})
    adv = pd.Series({"A": 5e7, "B": 4e7, "C": 1e6})
    h = liquidation_horizon(values, adv, participation=0.1, shares=(0.5, 0.9, 0.99, 1.0))
    # per day (millions): A sells 5, B sells 4, C sells 0.1; A and C are done after one day, B takes five
    assert h[0.5] == pytest.approx((0.5 * 25.1 - 5.1) / 4.0)
    assert h[0.9] == pytest.approx((0.9 * 25.1 - 5.1) / 4.0) and h[1.0] == pytest.approx(5.0)
    stuck = liquidation_horizon(values, adv.where(adv.index != "B", 0.0), shares=(0.1, 0.5))
    assert stuck[0.1] == pytest.approx(0.1 * 25.1 / 5.1, rel=1e-6) and np.isinf(stuck[0.5])                  # B has no volume: only A and C (5.1 a day) can be sold, and B never leaves
    assert liquidation_horizon(values * 0.0, adv) == {0.5: 0.0, 0.9: 0.0, 0.99: 0.0}
    assert all(np.isinf(v) for v in liquidation_horizon(values, adv * 0.0).values())
