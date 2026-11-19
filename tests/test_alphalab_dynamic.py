"""Alpha decay and trading policies, selection/sizing/timing attribution, stops and leverage rules, capacity, crowding and gearing: each against a hand calculation, a closed form, or a simulation of the thing it describes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alphalab import attribution as at
from src.alphalab import capacity as cp
from src.alphalab import dynamic as dy

SIG = 0.01


@pytest.fixture(scope="module")
def world():
    """20 assets, an AR(1) signal (persistence 0.9) with a true IC of 0.1: the expected next-day return is ``0.1 a sigma``."""
    rng = np.random.default_rng(7)
    T, N, PHI, C = 1200, 20, 0.9, 0.10
    dates = pd.bdate_range("2015-01-01", periods=T)
    cols = [f"S{i:02d}" for i in range(N)]
    a = np.zeros((T, N))
    a[0] = rng.standard_normal(N)
    for t in range(1, T):
        a[t] = PHI * a[t - 1] + np.sqrt(1 - PHI ** 2) * rng.standard_normal(N)
    r = np.zeros((T, N))
    r[1:] = C * a[:-1] + rng.standard_normal((T - 1, N))
    return pd.DataFrame(C * a * SIG, index=dates, columns=cols), pd.DataFrame(r * SIG, index=dates, columns=cols), np.eye(N) * SIG ** 2


def test_decay_path_and_aim():
    p = dy.decay_path([1.0, -2.0], tau=4.0, steps=5)
    assert p.shape == (5, 2) and p[0] == pytest.approx([1, -2]) and p[4, 0] == pytest.approx(np.exp(-1.0)) and (dy.decay_path([1.0], np.inf, 3) == 1).all()
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    a = np.array([0.02, 0.03])
    assert dy.aim(a, cov, 2.0) == pytest.approx(np.linalg.solve(2.0 * cov, a))
    long_only = dy.aim(np.array([0.02, -0.03]), cov, 2.0, lb=0.0)
    assert long_only[1] == pytest.approx(0.0, abs=1e-6) and long_only[0] > 0
    wide = dy.aim(a, cov, 2.0, lb=-100.0, ub=100.0)
    assert wide == pytest.approx(np.linalg.solve(2.0 * cov, a), rel=1e-4)


def test_trading_policies_respect_costs_in_the_order_theory_gives(world):
    alpha, R, cov = world
    kw = dict(risk_aversion=100.0, cost_bps=5.0, impact=1e-3)
    t = dy.compare_policies(alpha, R, cov, **kw)
    assert set(t.index) == {"full", "partial", "band", "threshold"}
    assert t.loc["band", "turnover_per_year"] < 0.5 * t.loc["full", "turnover_per_year"] and t.loc["band", "cost_per_year"] < 0.5 * t.loc["full", "cost_per_year"], "a no-trade region trades much less"
    assert t.loc["partial", "turnover_per_year"] < t.loc["full", "turnover_per_year"] and t.loc["partial", "cost_per_year"] < t.loc["full", "cost_per_year"]
    assert t.loc["partial", "sharpe"] > t.loc["full", "sharpe"] - 0.05 and t.loc["band", "sharpe"] > t.loc["full", "sharpe"] - 0.3
    assert (t["gross_sharpe"] > 0).all() and (t["sharpe"] <= t["gross_sharpe"] + 1e-9).all()


def test_with_no_cost_every_policy_that_trades_to_the_aim_is_the_same_book(world):
    alpha, R, cov = world
    kw = dict(risk_aversion=100.0, cost_bps=0.0, impact=0.0)
    a = dy.simulate(alpha, R, cov, "full", **kw)
    b = dy.simulate(alpha, R, cov, "threshold", **{**kw, "cost_bps": 0.0})
    assert a["stats"]["gross_sharpe"] > 3 and abs(a["stats"]["sharpe"] - a["stats"]["gross_sharpe"]) < 1e-9
    assert abs(a["stats"]["sharpe"] - b["stats"]["sharpe"]) < 0.5
    h = a["holdings"].iloc[10].to_numpy()
    assert h == pytest.approx(np.linalg.solve(100.0 * cov, alpha.iloc[10].to_numpy())), "the holdings are the aim"


def test_the_waterfall_loses_at_every_step_and_cost_aware_trading_wins_some_back(world):
    alpha, R, cov = world
    w = dy.implementation_waterfall(alpha, R, cov, risk_aversion=100.0, cost_bps=5.0, lag=1, lb=0.0, impact=1e-3)
    s = w["sharpe"]
    assert list(w.index) == ["paper", "constraints", "costs", "cost-aware trading", "delay"]
    assert s["paper"] > s["constraints"] > s["costs"] and s["cost-aware trading"] > s["costs"] and s["delay"] < s["cost-aware trading"]
    assert w["lost_sharpe"]["paper"] == 0 and w["lost_sharpe"]["constraints"] == pytest.approx(s["constraints"] - s["paper"])


def test_simulation_input_checks(world):
    alpha, R, cov = world
    with pytest.raises(ValueError, match="policy must be"):
        dy.simulate(alpha, R, cov, "teleport")
    with pytest.raises(ValueError, match="20 by 20"):
        dy.simulate(alpha, R, np.eye(3), "full")
    bad = pd.DataFrame(cov, index=["x"] * 20, columns=["x"] * 20)
    with pytest.raises(ValueError, match="no row for"):
        dy.simulate(alpha, R, bad, "full")


def test_attribution_on_a_book_worked_by_hand():
    idx = pd.bdate_range("2020-01-01", periods=30)
    w = pd.DataFrame([[0.6, -0.2, 0.2]] * 30, index=idx, columns=list("ABC"))
    r = pd.DataFrame([[0.01, 0.02, -0.01]] * 30, index=idx, columns=list("ABC"))
    out = at.selection_sizing_timing(w, r, lag=0)
    c = out["components"]
    assert c["selection"].iloc[0] == pytest.approx(-0.02 / 3) and c["total"].iloc[0] == pytest.approx(0.0, abs=1e-15)
    assert c["sizing"].iloc[0] == pytest.approx(0.02 / 3) and c["timing"].abs().max() < 1e-15 and out["hit_rate"] == pytest.approx(1 / 3) and out["max_residual"] < 1e-15


def test_attribution_adds_up_and_isolates_each_source():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.DataFrame(rng.standard_normal((300, 6)) * 0.01, index=idx, columns=list("ABCDEF"))
    side = np.array([1, 1, 1, -1, -1, -1.0])
    equal = pd.DataFrame([side / 6] * 300, index=idx, columns=r.columns)
    out = at.selection_sizing_timing(equal, r, lag=0)["components"]
    assert out["sizing"].abs().max() < 1e-15 and out["timing"].abs().max() < 1e-15, "equal-size, constant-gross: selection is everything"
    sized = pd.DataFrame([side * np.array([0.3, 0.1, 0.05, 0.3, 0.1, 0.05])] * 300, index=idx, columns=r.columns)
    out = at.selection_sizing_timing(sized, r, lag=0)["components"]
    assert out["timing"].abs().max() < 1e-15 and out["sizing"].abs().max() > 0
    timed = equal.mul(1 + 0.5 * np.sin(np.arange(300) / 10), axis=0)
    out = at.selection_sizing_timing(timed, r, lag=0)["components"]
    assert out["sizing"].abs().max() < 1e-15 and out["timing"].abs().max() > 0
    mixed = sized.mul(1 + 0.5 * np.sin(np.arange(300) / 10), axis=0)
    full = at.selection_sizing_timing(mixed, r, lag=1)
    assert full["max_residual"] < 1e-14 and full["components"]["total"].sum() == pytest.approx((mixed.shift(1) * r).sum(axis=1).sum())
    assert full["summary"].loc["total", "sum"] == pytest.approx(full["summary"].loc[["selection", "sizing", "timing"], "sum"].sum())
    with pytest.raises(ValueError, match="at least 20"):
        at.selection_sizing_timing(mixed.iloc[:10], r.iloc[:10])


def test_a_stop_cuts_exposure_for_the_cooloff_and_then_restores_it():
    r = pd.Series([-0.01] * 10 + [0.0] * 50, index=pd.bdate_range("2021-01-01", periods=60))
    out = at.stop_loss(r, limit=0.05, scale=0.0, cooloff=21)
    e = out["exposure"].to_numpy()
    assert out["stops"] == 1 and (e[:6] == 1).all() and (e[6:27] == 0).all() and e[27] == 1, "0.99^6 is the first loss past 5%; flat for 21 days; back to full"
    assert out["returns"].iloc[6:27].abs().sum() == 0 and out["max_drawdown"] > out["unmanaged_max_drawdown"] and out["time_out"] == pytest.approx(21 / 60)
    assert at.stop_loss(r * 0, 0.05)["stops"] == 0


def test_a_stop_never_looks_at_the_return_it_is_applied_to():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.standard_normal(500) * 0.01, index=pd.bdate_range("2021-01-01", periods=500))
    full = at.stop_loss(r, 0.04)
    cut = at.stop_loss(r.iloc[:300], 0.04)
    assert full["exposure"].iloc[:300].equals(cut["exposure"]), "exposure through day 300 is the same whatever happens afterwards"


def test_volatility_targeting_hits_the_target_and_is_causal():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=2000)
    r = pd.Series(rng.standard_normal(2000) * 0.02 * (1 + 0.8 * (np.arange(2000) > 1000)), index=idx)
    out = at.volatility_target(r, target=0.10, lookback=63, max_leverage=5.0)
    assert out["realised_vol"] == pytest.approx(0.10, rel=0.2) and out["leverage"].max() <= 5.0
    assert out["max_drawdown"] > out["unmanaged_max_drawdown"] * 1.5 - 1, "targeting does not make the drawdown worse than the unmanaged one by much"
    early = at.volatility_target(r.iloc[:800], 0.10, 63, 5.0)
    assert out["leverage"].iloc[:800].equals(early["leverage"]), "yesterday's volatility only"


def test_capacity_cost_matches_the_square_root_law_by_hand():
    idx = pd.bdate_range("2020-01-01", periods=100)
    trades = pd.DataFrame({"X": 0.1}, index=idx)
    sigma, adv = pd.DataFrame({"X": 0.01}, index=idx), pd.DataFrame({"X": 1e6}, index=idx)
    cost = cp.impact_cost(trades, 1e6, sigma, adv, c=1.0)
    assert cost.iloc[0] == pytest.approx(0.1 * 0.01 * np.sqrt(0.1 * 1e6 / 1e6)), "|dw| c sigma sqrt(|dw| A / ADV)"
    assert cp.impact_cost(trades, 4e6, sigma, adv).iloc[0] == pytest.approx(2 * cost.iloc[0]), "four times the capital, twice the unit cost"
    lin = cp.impact_cost(trades, 1e6, sigma, adv, c=0.0, linear_bps=10.0)
    assert lin.iloc[0] == pytest.approx(0.1 * 10e-4)


def test_capacity_is_where_the_sharpe_halves():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-01", periods=750)
    gross = pd.Series(rng.standard_normal(750) * 0.004 + 0.0004, index=idx)
    trades = pd.DataFrame({"X": 0.2 * np.ones(750), "Y": 0.2 * np.ones(750)}, index=idx)
    sigma, adv = pd.DataFrame({"X": 0.015, "Y": 0.015}, index=idx), pd.DataFrame({"X": 5e7, "Y": 5e7}, index=idx)
    grid = np.geomspace(1e5, 1e10, 26)
    curve = cp.capacity_curve(gross, trades, grid, sigma, adv)
    assert curve["sharpe"].is_monotonic_decreasing
    cap = cp.capacity(curve)
    assert isinstance(cap["capacity"], float) and grid[0] < cap["capacity"] < grid[-1]
    at_cap = cp.capacity_curve(gross, trades, [cap["capacity"]], sigma, adv)["sharpe"].iloc[0]
    assert at_cap == pytest.approx(cap["level"], rel=0.12), "at the reported capacity the Sharpe ratio is half the starting one"
    assert cp.capacity(curve.iloc[:2])["capacity"] in ("above grid", "below grid") or isinstance(cp.capacity(curve.iloc[:2])["capacity"], float)
    assert cp.capacity(cp.capacity_curve(-gross, trades, grid, sigma, adv))["capacity"] == "n/a"


def test_crowding_sees_the_shared_factor_behind_different_names():
    rng = np.random.default_rng(4)
    assets = [f"S{i}" for i in range(20)]
    B = pd.DataFrame({"value": np.r_[np.ones(10), np.zeros(10)] + 0.05 * rng.standard_normal(20), "size": 0.1 * rng.standard_normal(20)}, index=assets)
    a = pd.Series(np.r_[rng.uniform(0.5, 1, 5), np.zeros(15)], index=assets)                         # names 0-4
    b = pd.Series(np.r_[np.zeros(5), rng.uniform(0.5, 1, 5), np.zeros(10)], index=assets)            # names 5-9: no name in common, same value tilt
    c = pd.Series(np.r_[np.zeros(15), rng.uniform(0.5, 1, 5)], index=assets)                         # names 15-19: growth names
    rep = cp.crowding_report({"a": a, "b": b, "c": c}, B, pd.DataFrame(rng.standard_normal((300, 20)) * 0.01, columns=assets))
    assert rep["positions"].loc["a", "b"] == pytest.approx(0.0) and rep["positions"].loc["a", "a"] == pytest.approx(1.0)
    assert rep["factors"].loc["a", "b"] > 0.9 and rep["factors"].loc["a", "c"] < rep["factors"].loc["a", "b"], "different names, same bet"
    assert 1.0 <= rep["effective_independent_books"] <= 3.0 + 1e-9 and rep["returns"].shape == (3, 3)
    with pytest.raises(ValueError, match="two books"):
        cp.crowding_report({"a": a})


def test_kelly_leverage_maximises_growth_and_half_kelly_keeps_three_quarters_of_it():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0008, 0.01, 5000))
    k = cp.kelly_leverage(r, fraction=1.0)
    mu, var = r.mean(), r.var(ddof=1)
    assert k["full_kelly"] == pytest.approx(mu / var) and k["leverage"] == pytest.approx(mu / var)
    grid = np.linspace(0, 3 * k["full_kelly"], 301)
    g = cp.growth_rate(grid, mu, np.sqrt(var))
    assert grid[np.argmax(g)] == pytest.approx(k["full_kelly"], abs=0.02 * k["full_kelly"] + 1e-9)
    half = cp.kelly_leverage(r, fraction=0.5)
    assert half["growth_at_leverage"] / half["growth_at_full"] == pytest.approx(0.75, abs=1e-9), "half Kelly keeps three quarters of the growth"
    assert cp.kelly_leverage(r, 1.0, cap=2.0)["leverage"] == 2.0 and k["low"] < k["leverage"]


def test_the_drawdown_share_of_time_formula_matches_a_simulation():
    mu, sigma, depth, lev = 0.5, 0.2, 0.1, 1.0
    p = np.exp(-2 * mu * depth / (lev * sigma ** 2))
    assert cp.leverage_for_drawdown(mu, sigma, depth, p) == pytest.approx(lev)
    assert cp.leverage_for_drawdown(mu, sigma, depth, p ** 2) == pytest.approx(lev / 2), "halving the leverage squares the share of time spent that deep"
    rng = np.random.default_rng(6)
    dt, steps, paths = 1 / 252, 2520, 2000
    inc = rng.normal(lev * mu * dt, lev * sigma * np.sqrt(dt), (paths, steps))
    y = np.cumsum(inc, axis=1)
    dd = np.maximum.accumulate(np.maximum(y, 0), axis=1) - y
    share = (dd[:, 300:] >= depth).mean()                                      # after the drawdown has forgotten its start
    corrected = np.exp(-2 * mu * (depth + 0.5826 * sigma * np.sqrt(dt)) / (lev * sigma ** 2))        # a path looked at once a day misses excursions between looks (Broadie, Glasserman and Kou)
    assert share == pytest.approx(corrected, abs=0.008) and share < p < share + 0.03, (share, corrected, p)
    for bad in ((-1, 0.2, 0.1, 0.5), (0.5, 0.2, 0.1, 1.5), (0.5, 0.2, -0.1, 0.5)):
        with pytest.raises(ValueError):
            cp.leverage_for_drawdown(*bad)
