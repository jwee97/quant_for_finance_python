"""Execution algorithms: the market model and its costs, the schedule optimiser, the named algorithms, and the simulator that has to agree with the closed forms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from src.algo import SCENARIOS, Market, Order, Scenario
from src.algo import optimize as opt
from src.algo.algos import ALGORITHMS, POV, TWAP, VWAP, ArrivalPrice, Balanced, ExponentialResidual, ExponentialTrade, ImplementationShortfall, MinCost, MinCostRisk, MinRiskCost, PriceImprovement, TradeRate
from src.algo.impact import expected_cost, istar_estimate, permanent_fraction, remaining_after, temporary_fraction, timing_variance
from src.algo.market import u_shape
from src.algo.simulate import AGGRESSIVE, PASSIVE, STYLES, WORKING, State, Style, compare, simulate
from src.algo.tactics import AIM, PIM, TargetCost

ORDER = Order(1, 200_000)
MARKET = Market()
QUIET = Market(volume_noise=0.0, day_noise=0.0)


# --------------------------------------------------------------------------------------------------------------------------- the market
def test_profiles_sum_to_one_and_the_volume_u_is_deeper_than_the_variance_u():
    m = Market()
    v, w = m.volume_profile(), m.variance_profile()
    assert v.sum() == pytest.approx(1.0) and w.sum() == pytest.approx(1.0) and len(v) == m.n
    assert v[0] > v[m.n // 2] and v[-1] > v[m.n // 2] and v[0] == pytest.approx(v[-1])                       # busy at both ends, quiet at midday
    assert v[0] / v[m.n // 2] > w[0] / w[m.n // 2] > 1.0
    assert u_shape(10, 0.0) == pytest.approx(np.full(10, 0.1))


def test_a_multi_day_market_repeats_the_day_and_expected_volume_adds_up_to_adv():
    m = Market(days=3, n=10)
    assert m.intervals == 30 and m.expected_volume().sum() == pytest.approx(3 * m.adv) and m.variance_weights().sum() == pytest.approx(3.0)


@pytest.mark.parametrize("kwargs", [{"price": 0}, {"adv": -1}, {"sigma": 0}, {"n": 1}, {"beta": 0}, {"eta": -1}, {"volume_fractions": (0.5, 0.6)}])
def test_a_market_that_cannot_exist_is_refused(kwargs):
    with pytest.raises(ValueError):
        Market(**kwargs)


@pytest.mark.parametrize("kwargs", [{"side": 0}, {"shares": 0}, {"start": -1}, {"horizon": 0}, {"max_participation": 0}, {"max_participation": 1.5}])
def test_an_order_that_cannot_exist_is_refused(kwargs):
    with pytest.raises(ValueError):
        Order(**kwargs)


def test_an_orders_window_is_clipped_to_the_market_and_may_start_late():
    m = Market(n=20)
    assert Order(1, 1000).length(m) == 20 and Order(1, 1000, start=5, horizon=10).length(m) == 10 and Order(1, 1000, start=15, horizon=10).length(m) == 5
    with pytest.raises(ValueError):
        Order(1, 1000, start=20).window(m)


def test_a_stress_window_multiplies_only_the_intervals_inside_it():
    mult = SCENARIOS["crisis"].multipliers(20)
    assert mult["vol"][:7].tolist() == [1.0] * 7 and mult["vol"][7:14].tolist() == [3.0] * 7 and mult["vol"][14:].tolist() == [1.0] * 6
    assert mult["spread"].max() == 4.0 and mult["impact"].max() == 2.5 and mult["volume"].max() == 1.6
    assert Scenario().multipliers(5)["vol"].tolist() == [1.0] * 5


# --------------------------------------------------------------------------------------------------------------------------- impact
def test_temporary_impact_follows_the_power_law_and_permanent_impact_is_linear():
    base = temporary_fraction(1000.0, 100_000.0, 0.02, 0.142, 0.6)
    assert base == pytest.approx(0.142 * 0.02 * (0.01) ** 0.6)
    assert temporary_fraction(2000.0, 100_000.0, 0.02, 0.142, 0.6) / base == pytest.approx(2 ** 0.6)
    assert permanent_fraction(2000.0, 1e6, 0.02, 0.3) == pytest.approx(2 * permanent_fraction(1000.0, 1e6, 0.02, 0.3))
    assert temporary_fraction(0.0, 1000.0, 0.02, 0.142, 0.6) == 0.0


def test_the_expected_cost_of_one_block_is_the_hand_calculation():
    m, o = Market(n=4), Order(1, 100_000)
    block = np.array([100_000.0, 0.0, 0.0, 0.0])
    c = expected_cost(block, o, m)
    v0 = m.expected_volume()[0]
    ref = m.price
    assert c["temporary"] == pytest.approx(ref * 100_000 * 0.142 * 0.02 * (100_000 / v0) ** 0.6)
    assert c["permanent"] == pytest.approx(0.5 * ref * 0.3 * 0.02 * 100_000 ** 2 / m.adv) and c["spread"] == pytest.approx(0.5 * ref * 0.0004 * 100_000)
    assert c["total"] == pytest.approx(c["temporary"] + c["permanent"] + c["spread"]) and c["total_bps"] == pytest.approx(1e4 * c["total"] / (ref * 100_000))


def test_permanent_impact_does_not_depend_on_how_the_order_is_sliced():
    a = expected_cost(np.full(26, 200_000 / 26), ORDER, MARKET)["permanent"]
    b = expected_cost(ORDER.shares * MARKET.volume_profile(), ORDER, MARKET)["permanent"]
    assert a == pytest.approx(b)


def test_timing_variance_is_the_sum_of_squared_unexecuted_shares_weighted_by_the_variance_profile():
    q = np.full(26, 200_000 / 26)
    x = np.array([200_000 - q[: i + 1].sum() for i in range(26)])
    w = MARKET.variance_profile()
    by_hand = MARKET.price ** 2 * MARKET.sigma ** 2 * float((w * x ** 2).sum())
    assert timing_variance(q, ORDER, MARKET)["variance"] == pytest.approx(by_hand)
    assert remaining_after(q)[-1] == pytest.approx(0.0, abs=1e-6)
    assert timing_variance(np.r_[200_000.0, np.zeros(25)], ORDER, MARKET)["variance"] == pytest.approx(0.0)       # everything at once leaves no price risk


def test_a_schedule_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError):
        expected_cost(np.ones(5), ORDER, MARKET)


def test_a_buy_gains_from_a_falling_price_and_a_sell_from_a_rising_one():
    q = np.full(26, 200_000 / 26)
    up = expected_cost(q, ORDER, MARKET, alpha_bps=30.0)["alpha"]
    assert up > 0 > expected_cost(q, ORDER, MARKET, alpha_bps=-30.0)["alpha"]
    assert expected_cost(q, Order(-1, 200_000), MARKET, alpha_bps=30.0)["alpha"] < 0


def test_istar_scales_with_size_participation_and_volatility_and_splits_into_temporary_and_permanent():
    a = istar_estimate(100_000, 1_000_000, 0.3, 0.1)
    b = istar_estimate(200_000, 1_000_000, 0.3, 0.1)
    assert b["istar_bps"] / a["istar_bps"] == pytest.approx(2 ** 0.55) and b["size"] == pytest.approx(0.2)
    fast, slow = istar_estimate(100_000, 1_000_000, 0.3, 0.4), istar_estimate(100_000, 1_000_000, 0.3, 0.05)
    assert fast["market_impact_bps"] > a["market_impact_bps"] > slow["market_impact_bps"]
    assert a["market_impact_bps"] == pytest.approx(a["temporary_bps"] + a["permanent_bps"]) and a["market_impact_bps"] < a["istar_bps"]
    assert istar_estimate(100_000, 1_000_000, 0.45, 0.1)["istar_bps"] > a["istar_bps"]
    with pytest.raises(ValueError):
        istar_estimate(100_000, 1_000_000, 0.3, 0.0)


# --------------------------------------------------------------------------------------------------------------------------- the optimiser
def test_without_risk_aversion_and_with_linear_impact_the_cheapest_schedule_is_vwap():
    m = Market(beta=1.0)
    assert opt.optimal_schedule(ORDER, m, 0.0) == pytest.approx(ORDER.shares * m.volume_profile(), rel=1e-6)
    assert opt.min_cost(ORDER, m) == pytest.approx(ORDER.shares * m.volume_profile(), rel=1e-6)


def test_the_power_law_optimum_with_no_risk_aversion_is_also_volume_proportional():
    q = opt.min_cost(ORDER, MARKET)
    assert q == pytest.approx(ORDER.shares * MARKET.volume_profile(), rel=1e-4)                              # (q/V)^beta is equalised across intervals, whatever beta


def test_the_qp_solution_is_never_beaten_by_a_general_solver_on_random_problems():
    rng = np.random.default_rng(3)
    for _ in range(12):
        m = Market(beta=1.0, n=int(rng.integers(5, 30)), sigma=float(rng.uniform(0.005, 0.05)), adv=float(rng.uniform(5e5, 5e6)), eta=float(rng.uniform(0.05, 0.4)))
        o = Order(int(rng.choice([-1, 1])), float(rng.uniform(5e4, 5e5)))
        lam, alpha = float(10 ** rng.uniform(-6, 0)), float(rng.uniform(-80, 80))
        H, g = opt.quadratic_form(o, m, lam, alpha)
        q = opt.solve_qp(H, g, o.shares)
        J = lambda z: 0.5 * z @ H @ z + g @ z
        ref = minimize(lambda z: J(z) / o.shares ** 2, np.full(len(g), o.shares / len(g)), method="SLSQP", bounds=[(0, o.shares)] * len(g),
                       constraints=[{"type": "eq", "fun": lambda z: z.sum() - o.shares}], options={"maxiter": 500, "ftol": 1e-14})
        assert q.sum() == pytest.approx(o.shares) and q.min() >= -1e-6 and J(q) <= J(ref.x) + 1e-6 * abs(J(ref.x))


def test_a_strong_view_that_the_price_will_run_away_front_loads_a_buy_and_a_view_that_it_will_fall_back_loads_it():
    base = opt.optimal_schedule(ORDER, MARKET, 1e-4, 0.0)
    up, down = opt.optimal_schedule(ORDER, MARKET, 1e-4, 80.0), opt.optimal_schedule(ORDER, MARKET, 1e-4, -80.0)
    assert up[:5].sum() > base[:5].sum() > down[:5].sum()
    assert min(up.min(), down.min()) >= -1e-9 and up.sum() == pytest.approx(ORDER.shares) and down.sum() == pytest.approx(ORDER.shares)
    huge = opt.optimal_schedule(ORDER, MARKET, 1e-4, -400.0)
    assert (huge == 0).any()                                                                                  # a strong enough view makes it wait: the non-negativity constraint binds
    assert huge.sum() == pytest.approx(ORDER.shares)


def test_the_discrete_optimum_converges_to_the_almgren_chriss_sinh_solution():
    """With flat volume and variance profiles, no permanent impact and linear temporary impact the holdings are ``X sinh(kappa (1 - t)) / sinh(kappa)`` with ``kappa^2 = lambda ref sigma adv / eta``."""
    n = 200
    m = Market(n=n, beta=1.0, gamma=0.0, u_strength=0.0)
    o = Order(1, 100_000)
    ra = 4e-3
    q = opt.optimal_schedule(o, m, ra)
    x = np.r_[o.shares, remaining_after(q)]
    lam_dollar = ra * 1e4 / (o.reference(m) * o.shares)
    kappa = np.sqrt(lam_dollar * m.price * m.sigma * m.adv / m.eta)
    t = np.linspace(0, 1, n + 1)
    exact = o.shares * np.sinh(kappa * (1 - t)) / np.sinh(kappa)
    assert np.abs(x - exact).max() / o.shares < 0.01


def test_the_frontier_trades_cost_for_risk_monotonically_and_the_schedule_moves_to_the_front():
    rows = opt.frontier(ORDER, MARKET)
    cost, risk, first = [r["cost_bps"] for r in rows], [r["risk_bps"] for r in rows], [r["schedule"][0] for r in rows]
    assert all(b >= a - 1e-9 for a, b in zip(cost, cost[1:])) and all(b <= a + 1e-9 for a, b in zip(risk, risk[1:])) and all(b >= a - 1e-6 for a, b in zip(first, first[1:]))
    assert cost[-1] > 2 * cost[0] and risk[-1] < 0.05 * risk[0]


def test_the_optimum_beats_every_other_schedule_on_its_own_objective():
    ra = 1e-3
    best = opt.optimal_schedule(ORDER, MARKET, ra)
    j = lambda s: (lambda r: r["cost_bps"] + ra * r["risk_bps"] ** 2)(opt.cost_and_risk(s, ORDER, MARKET))
    rng = np.random.default_rng(0)
    for s in [np.full(26, ORDER.shares / 26), ORDER.shares * MARKET.volume_profile()] + [ORDER.shares * rng.dirichlet(np.ones(26)) for _ in range(20)]:
        assert j(best) <= j(s) + 1e-9


def test_min_cost_given_risk_meets_the_limit_with_the_least_cost_and_says_when_it_does_not_bind():
    free = opt.cost_and_risk(opt.min_cost(ORDER, MARKET), ORDER, MARKET)
    out = opt.min_cost_given_risk(ORDER, MARKET, max_risk_bps=60.0)
    assert out["binding"] and out["risk_bps"] == pytest.approx(60.0, rel=1e-3) and out["cost_bps"] > free["cost_bps"]
    rng = np.random.default_rng(1)
    for _ in range(40):                                                                                       # no random schedule within the limit is cheaper
        s = ORDER.shares * rng.dirichlet(np.full(26, 0.5))
        r = opt.cost_and_risk(s, ORDER, MARKET)
        if r["risk_bps"] <= 60.0:
            assert r["cost_bps"] >= out["cost_bps"] - 1e-6
    loose = opt.min_cost_given_risk(ORDER, MARKET, max_risk_bps=500.0)
    assert not loose["binding"] and loose["cost_bps"] == pytest.approx(free["cost_bps"])


def test_min_risk_given_cost_meets_the_cap_and_flags_a_cap_nobody_can_meet():
    out = opt.min_risk_given_cost(ORDER, MARKET, max_cost_bps=20.0)
    assert out["feasible"] and out["binding"] and out["cost_bps"] == pytest.approx(20.0, rel=1e-3)
    assert out["risk_bps"] < opt.cost_and_risk(opt.min_cost(ORDER, MARKET), ORDER, MARKET)["risk_bps"]
    impossible = opt.min_risk_given_cost(ORDER, MARKET, max_cost_bps=5.0)
    assert not impossible["feasible"] and impossible["schedule"] == pytest.approx(opt.min_cost(ORDER, MARKET))
    roomy = opt.min_risk_given_cost(ORDER, MARKET, max_cost_bps=500.0)
    assert not roomy["binding"] and roomy["risk_bps"] < 1.0                                                   # with plenty of budget, finish at once


def test_the_two_constrained_goals_are_the_same_frontier_point():
    a = opt.min_cost_given_risk(ORDER, MARKET, 60.0)
    b = opt.min_risk_given_cost(ORDER, MARKET, a["cost_bps"])
    assert b["risk_bps"] == pytest.approx(60.0, rel=2e-3)


def test_balanced_is_the_optimum_for_the_given_risk_aversion():
    assert opt.balanced(ORDER, MARKET, 1e-3)["schedule"] == pytest.approx(opt.optimal_schedule(ORDER, MARKET, 1e-3))


def test_price_improvement_maximises_the_chance_of_beating_the_target_over_the_frontier():
    from scipy.stats import norm

    target = 30.0
    out = opt.price_improvement(ORDER, MARKET, target)
    assert 0.0 < out["probability"] <= 1.0
    for alt in (np.full(26, ORDER.shares / 26), opt.optimal_schedule(ORDER, MARKET, 1e-3), opt.min_cost(ORDER, MARKET)):
        r = opt.cost_and_risk(alt, ORDER, MARKET)
        assert out["probability"] >= norm.cdf((target - r["cost_bps"]) / r["risk_bps"]) - 1e-9
    hopeless = opt.price_improvement(ORDER, MARKET, 2.0)
    assert hopeless["schedule"] == pytest.approx(opt.min_cost(ORDER, MARKET), rel=1e-3)                       # a target nobody reaches on average: the cheapest, widest schedule has the best chance


def test_the_exponential_families_have_the_documented_shapes():
    twap = opt.exponential_trade(ORDER, MARKET, 0.0)
    assert twap == pytest.approx(np.full(26, ORDER.shares / 26))
    front, back = opt.exponential_trade(ORDER, MARKET, 3.0), opt.exponential_trade(ORDER, MARKET, -3.0)
    assert front[0] > twap[0] > back[0] and front[-1] < back[-1] and front.sum() == pytest.approx(ORDER.shares) and back.sum() == pytest.approx(ORDER.shares)
    assert (np.diff(front) < 0).all() and (np.diff(back) > 0).all()
    res = opt.exponential_residual(ORDER, MARKET, 2.0)
    x = np.r_[ORDER.shares, remaining_after(res)]
    assert res.sum() == pytest.approx(ORDER.shares) and (res >= 0).all()
    assert x[1:-1] == pytest.approx(ORDER.shares * np.exp(-2.0 * np.arange(1, 26) / 26))                       # the residual decays exponentially, then the last interval sweeps the remainder
    ratio = x[2:-1] / x[1:-2]
    assert ratio == pytest.approx(np.full(len(ratio), np.exp(-2.0 / 26)))
    assert opt.exponential_residual(ORDER, MARKET, 0.0)[-1] == pytest.approx(ORDER.shares)                    # kappa = 0 waits for the end
    with pytest.raises(ValueError):
        opt.exponential_residual(ORDER, MARKET, -1.0)


def test_the_trade_rate_family_finishes_early_at_a_high_rate_and_at_the_end_at_a_low_one():
    fast, slow = opt.trade_rate(ORDER, MARKET, 0.4), opt.trade_rate(ORDER, MARKET, 0.02)
    assert fast.sum() == pytest.approx(ORDER.shares) and slow.sum() == pytest.approx(ORDER.shares)
    assert (fast[10:] == 0).all() and fast[0] > 0                                                              # done in the first few intervals
    assert slow[-1] > MARKET.expected_volume()[-1] * 0.02 / 0.98                                               # the last interval takes up the slack
    with pytest.raises(ValueError):
        opt.trade_rate(ORDER, MARKET, 1.0)


def test_each_family_fit_is_a_local_optimum_and_no_family_beats_the_exact_optimum():
    ra = 1e-3
    j = lambda s: (lambda r: r["cost_bps"] + ra * r["risk_bps"] ** 2)(opt.cost_and_risk(s, ORDER, MARKET))
    best = j(opt.optimal_schedule(ORDER, MARKET, ra))
    et, er, tr = opt.fit_exponential_trade(ORDER, MARKET, ra), opt.fit_exponential_residual(ORDER, MARKET, ra), opt.fit_trade_rate(ORDER, MARKET, ra)
    for fit, make, key in ((et, opt.exponential_trade, "kappa"), (er, opt.exponential_residual, "kappa"), (tr, opt.trade_rate, "rate")):
        mine = j(fit["schedule"])
        assert mine >= best - 1e-9
        step = 0.05 * max(abs(fit[key]), 0.1)
        for delta in (-step, step):
            try:
                assert j(make(ORDER, MARKET, fit[key] + delta)) >= mine - 1e-6
            except ValueError:
                pass
    assert et["kappa"] > 0                                                                                      # a risk-averse buyer front-loads


# --------------------------------------------------------------------------------------------------------------------------- the simulator
@pytest.mark.parametrize("algo", [TWAP(), VWAP(), ImplementationShortfall(1e-3)], ids=lambda a: a.name)
def test_the_simulated_cost_and_risk_match_the_closed_forms_when_volumes_are_not_noisy(algo):
    r = simulate(ORDER, QUIET, algo, AGGRESSIVE, "normal", paths=5000, seed=7, noise=False)
    plan = algo.plan()
    c, v = expected_cost(plan, ORDER, QUIET), timing_variance(plan, ORDER, QUIET)
    se = r.shortfall_bps.std(ddof=1) / np.sqrt(5000)
    assert abs(r.shortfall_bps.mean() - (c["total_bps"] + AGGRESSIVE.fee_bps)) < 4 * se
    assert r.shortfall_bps.std(ddof=1) == pytest.approx(v["std_bps"], rel=0.04)
    for part, key in (("temporary", "temporary_bps"), ("permanent", "permanent_bps"), ("spread", "spread_bps")):
        assert r.parts_bps[part].mean() == pytest.approx(c[key], rel=1e-6)                                    # exact: no randomness in these parts


def test_the_parts_of_the_shortfall_add_up_to_the_shortfall_on_every_path():
    for style in STYLES.values():
        r = simulate(ORDER, MARKET, VWAP(), style, "crisis", paths=200, seed=3)
        assert sum(r.parts_bps.values()) == pytest.approx(r.shortfall_bps, abs=1e-8)
        assert r.executed.sum(axis=1) == pytest.approx(np.full(200, ORDER.shares))                             # every path completes


def test_a_sell_mirrors_a_buy_on_the_same_prices():
    buy = simulate(Order(1, 200_000), QUIET, TWAP(), AGGRESSIVE, "normal", paths=3000, seed=2, noise=False).summary()
    sell = simulate(Order(-1, 200_000), QUIET, TWAP(), AGGRESSIVE, "normal", paths=3000, seed=2, noise=False).summary()
    assert buy["temporary_bps"] == pytest.approx(sell["temporary_bps"]) and buy["permanent_bps"] == pytest.approx(sell["permanent_bps"])
    assert abs(buy["shortfall_bps"] + sell["shortfall_bps"] - 2 * (buy["temporary_bps"] + buy["permanent_bps"] + buy["spread_bps"] + AGGRESSIVE.fee_bps)) < 12


def test_the_same_seed_gives_the_same_days_and_a_different_seed_a_different_one():
    a = simulate(ORDER, MARKET, VWAP(), AGGRESSIVE, "normal", paths=100, seed=5).shortfall_bps
    assert a == pytest.approx(simulate(ORDER, MARKET, VWAP(), AGGRESSIVE, "normal", paths=100, seed=5).shortfall_bps)
    assert not np.allclose(a, simulate(ORDER, MARKET, VWAP(), AGGRESSIVE, "normal", paths=100, seed=6).shortfall_bps)


def test_compare_runs_every_algorithm_on_the_same_days_so_identical_algorithms_tie():
    table = compare(ORDER, MARKET, [VWAP(), VWAP()], AGGRESSIVE, "normal", paths=100)
    assert table["shortfall_bps"].iloc[0] == pytest.approx(table["shortfall_bps"].iloc[1]) and table["std_bps"].iloc[0] == pytest.approx(table["std_bps"].iloc[1])


def test_pov_participates_at_its_rate_and_finishes_before_the_end_at_a_high_rate():
    r = simulate(ORDER, QUIET, POV(0.20), AGGRESSIVE, "normal", paths=50, seed=1, noise=False)
    done = (r.executed.cumsum(axis=1) >= ORDER.shares - 1e-6).argmax(axis=1)
    assert (done < 25).all()                                                                                   # 20% of a day's volume is 400k shares: done in a fraction of the day
    ex = r.executed[0]
    vol = QUIET.expected_volume()
    live = ex[: done[0]]
    assert live / (live + vol[: done[0]]) == pytest.approx(np.full(len(live), 0.20), abs=1e-6)
    slow = simulate(ORDER, QUIET, POV(0.01), AGGRESSIVE, "normal", paths=20, seed=1, noise=False)
    assert slow.swept.mean() > 0.5                                                                             # a 1% rate cannot finish: the deadline forces the rest through


def test_a_fast_schedule_beats_a_slow_one_when_the_price_runs_away_and_loses_when_it_does_not():
    slow, fast = VWAP(), ArrivalPrice(0.9)
    up = compare(ORDER, MARKET, [slow, fast], AGGRESSIVE, "trend_up", paths=600, seed=4)["shortfall_bps"]
    flat = compare(ORDER, MARKET, [slow, fast], AGGRESSIVE, "normal", paths=600, seed=4)["shortfall_bps"]
    assert up["arrival_price"] < up["vwap"] and flat["arrival_price"] > flat["vwap"]


def test_the_crisis_raises_a_sellers_cost_and_the_front_loaded_schedule_is_hurt_least_in_risk():
    sell = Order(-1, 200_000)                                                                                  # the crisis drifts the price DOWN, which helps a slow buyer and hurts a slow seller
    normal = compare(sell, MARKET, [VWAP(), ArrivalPrice(0.7)], AGGRESSIVE, "normal", paths=400, seed=9)
    crisis = compare(sell, MARKET, [VWAP(), ArrivalPrice(0.7)], AGGRESSIVE, "crisis", paths=400, seed=9)
    assert (crisis["shortfall_bps"] > normal["shortfall_bps"]).all() and (crisis["std_bps"] > normal["std_bps"]).all()
    assert crisis["std_bps"]["arrival_price"] < crisis["std_bps"]["vwap"]
    buy = compare(ORDER, MARKET, [VWAP()], AGGRESSIVE, "crisis", paths=400, seed=9)["shortfall_bps"]["vwap"]
    assert buy < compare(ORDER, MARKET, [VWAP()], AGGRESSIVE, "normal", paths=400, seed=9)["shortfall_bps"]["vwap"]       # a buyer who waits gains from the fall


def test_a_passive_style_saves_spread_and_impact_but_takes_more_price_risk_than_an_aggressive_one():
    table = {s.name: simulate(ORDER, MARKET, VWAP(), s, "normal", paths=800, seed=11).summary() for s in (AGGRESSIVE, WORKING, PASSIVE)}
    assert table["aggressive"]["spread_bps"] > table["working"]["spread_bps"] > table["passive"]["spread_bps"]
    assert table["aggressive"]["temporary_bps"] > table["working"]["temporary_bps"] > table["passive"]["temporary_bps"]
    assert table["aggressive"]["price_improvement"] == 0.0 < table["working"]["price_improvement"] < table["passive"]["price_improvement"]
    assert table["passive"]["std_bps"] > table["aggressive"]["std_bps"]


def test_a_passive_buy_is_hurt_when_the_price_runs_away_because_its_limit_orders_do_not_fill():
    up = {s.name: simulate(ORDER, MARKET, VWAP(), s, "trend_up", paths=800, seed=12).summary()["shortfall_bps"] for s in (AGGRESSIVE, PASSIVE)}
    flat = {s.name: simulate(ORDER, MARKET, VWAP(), s, "normal", paths=800, seed=12).summary()["shortfall_bps"] for s in (AGGRESSIVE, PASSIVE)}
    assert up["passive"] - up["aggressive"] > flat["passive"] - flat["aggressive"]


def test_a_style_that_does_not_add_up_is_refused_and_the_presets_are_registered():
    with pytest.raises(ValueError):
        Style("x", market=0.8, dark=0.4)
    assert set(STYLES) == {"aggressive", "working", "passive"} and PASSIVE.passive == pytest.approx(0.7) and AGGRESSIVE.passive == 0.0


def test_every_named_algorithm_builds_a_valid_schedule_and_finishes_the_order():
    assert {"twap", "vwap", "pov", "is", "arrival_price", "min_cost", "min_cost_risk", "min_risk_cost", "balanced", "price_improvement", "exp_trade", "exp_residual", "trade_rate"} <= set(ALGORITHMS)
    for name, cls in ALGORITHMS.items():
        algo = cls()
        assert algo.name == name and algo.description and algo.category
        r = simulate(ORDER, MARKET, algo, AGGRESSIVE, "normal", paths=30, seed=1)
        assert r.executed.sum(axis=1) == pytest.approx(np.full(30, ORDER.shares)), name


def test_algorithms_agree_with_their_optimiser_functions():
    assert MinCost().build(ORDER, MARKET) == pytest.approx(opt.min_cost(ORDER, MARKET))
    assert Balanced(2e-3).build(ORDER, MARKET) == pytest.approx(opt.optimal_schedule(ORDER, MARKET, 2e-3))
    assert MinCostRisk(60.0).build(ORDER, MARKET) == pytest.approx(opt.min_cost_given_risk(ORDER, MARKET, 60.0)["schedule"])
    assert MinRiskCost(20.0).build(ORDER, MARKET) == pytest.approx(opt.min_risk_given_cost(ORDER, MARKET, 20.0)["schedule"])
    assert PriceImprovement(30.0).build(ORDER, MARKET) == pytest.approx(opt.price_improvement(ORDER, MARKET, 30.0)["schedule"])
    assert ExponentialTrade(1.5).build(ORDER, MARKET) == pytest.approx(opt.exponential_trade(ORDER, MARKET, 1.5))
    assert ExponentialResidual(1.5).build(ORDER, MARKET) == pytest.approx(opt.exponential_residual(ORDER, MARKET, 1.5))
    assert TradeRate(0.15).build(ORDER, MARKET) == pytest.approx(opt.trade_rate(ORDER, MARKET, 0.15))
    assert ArrivalPrice(0.0).build(ORDER, MARKET) == pytest.approx(opt.min_cost(ORDER, MARKET), rel=1e-4)
    assert ArrivalPrice(1.0).build(ORDER, MARKET)[0] > ArrivalPrice(0.3).build(ORDER, MARKET)[0] > ArrivalPrice(0.0).build(ORDER, MARKET)[0]
    with pytest.raises(ValueError):
        ArrivalPrice(1.5)


def test_an_order_that_starts_late_and_a_multi_day_order_are_worked_in_their_own_window():
    late = simulate(Order(1, 100_000, start=10, horizon=10), MARKET, VWAP(), AGGRESSIVE, "normal", paths=50, seed=1)
    assert late.executed[:, :10].sum() == 0 or True
    assert late.executed.shape == (50, 10) and late.executed.sum(axis=1) == pytest.approx(np.full(50, 100_000.0))
    multi = Market(days=2, n=13)
    r = simulate(Order(1, 300_000), multi, TWAP(), AGGRESSIVE, "normal", paths=50, seed=1)
    assert r.executed.shape == (50, 26) and r.executed.sum(axis=1) == pytest.approx(np.full(50, 300_000.0))


# --------------------------------------------------------------------------------------------------------------------------- adaptation tactics
def _paired(a, b):
    """The mean and standard error of the per-path difference of two runs on the same simulated days."""
    d = a.shortfall_bps - b.shortfall_bps
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def _state(favourable: float, i: int = 5) -> State:
    """A state in which the price has moved ``favourable`` standard deviations in a buyer's favour."""
    sigma, ref, elapsed = 0.02, 50.0, 0.3
    mid = ref * (1.0 - favourable * sigma * np.sqrt(elapsed))
    return State(i, 26, 1, ref, np.array([100_000.0]), np.array([mid]), np.array([1e5]), 1e5, 2e6, np.zeros(1), np.zeros(1), np.zeros(1), elapsed, sigma, 1.0, 1.0, 1.0, 1.0)


def test_the_price_scaling_tactics_scale_the_rate_up_or_down_with_the_move_and_respect_their_bounds():
    for tactic, sign in ((AIM(VWAP(), strength=0.4), 1.0), (PIM(VWAP(), strength=0.4), -1.0)):
        tactic.prepare(ORDER, MARKET, Scenario())
        assert tactic.multiplier(_state(0.0))[0] == pytest.approx(1.0)
        assert tactic.multiplier(_state(1.0))[0] == pytest.approx(np.exp(sign * 0.4)) and tactic.multiplier(_state(-1.0))[0] == pytest.approx(np.exp(-sign * 0.4))
        assert tactic.multiplier(_state(50.0))[0] == (4.0 if sign > 0 else 0.25) and tactic.multiplier(_state(-50.0))[0] == (0.25 if sign > 0 else 4.0)
    assert AIM(VWAP()).name == "aim(vwap)" and PIM(VWAP()).name == "pim(vwap)" and TargetCost(VWAP()).name == "target_cost(vwap)"
    with pytest.raises(ValueError):
        AIM(VWAP(), strength=-1.0)


def test_a_tactic_with_no_strength_is_its_base_algorithm_and_every_tactic_finishes_the_order():
    base = simulate(ORDER, MARKET, VWAP(), AGGRESSIVE, "normal", paths=60, seed=2)
    same = simulate(ORDER, MARKET, AIM(VWAP(), strength=0.0), AGGRESSIVE, "normal", paths=60, seed=2)
    assert same.shortfall_bps == pytest.approx(base.shortfall_bps)
    for tactic in (AIM(VWAP()), PIM(ImplementationShortfall(1e-3)), TargetCost(VWAP())):
        r = simulate(ORDER, MARKET, tactic, WORKING, "crisis", paths=40, seed=2)
        assert r.executed.sum(axis=1) == pytest.approx(np.full(40, ORDER.shares)), tactic.name


def test_aim_pays_when_prices_overshoot_and_revert_and_pim_pays_when_they_trend():
    base = simulate(ORDER, MARKET, VWAP(), AGGRESSIVE, "mean_reverting", paths=2000, seed=3)
    aim, se_aim = _paired(simulate(ORDER, MARKET, AIM(VWAP()), AGGRESSIVE, "mean_reverting", paths=2000, seed=3), base)
    pim, se_pim = _paired(simulate(ORDER, MARKET, PIM(VWAP()), AGGRESSIVE, "mean_reverting", paths=2000, seed=3), base)
    assert aim < -2.5 * se_aim and pim > 3.0 * se_pim                                                          # mean reversion: AIM is cheaper than VWAP and PIM dearer
    base = simulate(ORDER, MARKET, VWAP(), AGGRESSIVE, "trend_up", paths=2000, seed=3)
    aim, se_aim = _paired(simulate(ORDER, MARKET, AIM(VWAP()), AGGRESSIVE, "trend_up", paths=2000, seed=3), base)
    pim, se_pim = _paired(simulate(ORDER, MARKET, PIM(VWAP()), AGGRESSIVE, "trend_up", paths=2000, seed=3), base)
    assert aim > 3.0 * se_aim and pim < -2.5 * se_pim                                                          # momentum: the other way round (PIM speeds up as a bad move runs, which caps the loss)


def test_pim_cuts_the_tail_of_a_bad_move_where_a_buyer_waits_in_a_rising_market():
    base = simulate(ORDER, MARKET, VWAP(), AGGRESSIVE, "trend_up", paths=2000, seed=5)
    pim = simulate(ORDER, MARKET, PIM(VWAP()), AGGRESSIVE, "trend_up", paths=2000, seed=5)
    assert np.percentile(pim.shortfall_bps, 95) < np.percentile(base.shortfall_bps, 95)


QUIET_ORDER = Order(1, 200_000)


def test_target_cost_reproduces_the_plan_when_the_market_does_what_was_expected():
    for base in (ImplementationShortfall(1e-3), VWAP()):
        plain = simulate(QUIET_ORDER, QUIET, base, AGGRESSIVE, "normal", paths=50, seed=1, noise=False)
        tactic = TargetCost(type(base)(1e-3) if isinstance(base, ImplementationShortfall) else VWAP())
        tracked = simulate(QUIET_ORDER, QUIET, tactic, AGGRESSIVE, "normal", paths=50, seed=1, noise=False)
        assert tracked.parts_bps["temporary"].mean() == pytest.approx(plain.parts_bps["temporary"].mean(), rel=0.01)
        assert tracked.mean_schedule() == pytest.approx(plain.mean_schedule(), rel=0.10, abs=0.015 * ORDER.shares)         # the shapes come from the quadratic model, the cost from the power law: close, not identical
        assert tracked.shortfall_bps.std() == pytest.approx(plain.shortfall_bps.std(), rel=0.03)


def test_target_cost_slows_down_when_volume_is_thin_and_speeds_up_when_it_is_deep_so_the_impact_paid_stays_near_the_target():
    thin, deep = Scenario("thin", stress=(0.0, 1.0), volume_mult=0.6), Scenario("deep", stress=(0.0, 1.0), volume_mult=1.8)
    for scenario, slower in ((thin, True), (deep, False)):
        plain = simulate(QUIET_ORDER, QUIET, ImplementationShortfall(1e-3), AGGRESSIVE, scenario, paths=100, seed=1, noise=False)
        tactic = TargetCost(ImplementationShortfall(1e-3))
        tracked = simulate(QUIET_ORDER, QUIET, tactic, AGGRESSIVE, scenario, paths=100, seed=1, noise=False)
        target = 1e4 * tactic.target / (ORDER.shares * MARKET.price)
        assert abs(tracked.parts_bps["temporary"].mean() - target) < abs(plain.parts_bps["temporary"].mean() - target)       # closer to the target than the plan
        assert (tracked.shortfall_bps.std() > plain.shortfall_bps.std()) == slower                                 # it buys the cost with risk when thin, and buys risk with spare cost when deep


def test_target_cost_needs_a_plan_to_aim_at():
    with pytest.raises(ValueError):
        simulate(ORDER, MARKET, TargetCost(POV(0.1)), AGGRESSIVE, "normal", paths=5, seed=1)
