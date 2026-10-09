"""The limit/market order mix: the dynamic programme against a plain-loop reference, exact evaluation against simulation, the benchmark policies, and the comparative statics the theory predicts."""

from __future__ import annotations

import numpy as np
import pytest

from src.algo import limit_orders as lo


def reference_value(model):
    """The same recursion written with plain loops and recursion over fills, for tiny orders: no vectorisation to get wrong."""
    N, T = model.units, model.intervals
    u = model.unit_shares
    h, a = model.half_spread_bps, model.adverse_selection_bps

    def tail(j):                                                                           # P(fill >= j units)
        return model.fill_first * model.fill_decay ** (j - 1)

    def p_fill(s, f):
        if f == 0:
            return 1.0 - tail(1) if s > 0 else 1.0
        if f < s:
            return tail(f) - tail(f + 1)
        return tail(s)

    memo = {}

    def V(t, q):
        if t == T:
            return q * u * (h + model.impact_bps * q * u / model.shares)
        if (t, q) in memo:
            return memo[(t, q)]
        best = np.inf
        for m in range(q + 1):
            now = m * u * (h + model.impact_bps * m * u / model.shares)
            for s in range(q - m + 1):
                cont = 0.0
                for f in range(s + 1):
                    r = q - m - f
                    cont += p_fill(s, f) * (f * u * (a - h) + V(t + 1, r) + model.drift_bps * r * u + model.risk_bps * (r * u) ** 2 / model.shares)
                best = min(best, now + cont)
        memo[(t, q)] = best
        return best

    return V(0, N)


MODELS = [
    lo.LimitOrderModel(shares=10_000, intervals=3, units=5),
    lo.LimitOrderModel(shares=50_000, intervals=3, units=6, drift_bps=3.0, risk_bps=0.5, fill_first=0.6, fill_decay=0.9),
    lo.LimitOrderModel(shares=20_000, intervals=2, units=7, half_spread_bps=5.0, adverse_selection_bps=6.0, fill_first=0.9, fill_decay=0.8, impact_bps=2.0),
]


@pytest.mark.parametrize("model", MODELS, ids=["base", "urgent", "toxic"])
def test_the_dynamic_programme_is_the_plain_recursion(model):
    sol = lo.solve(model)
    assert sol.value[0, model.units] == pytest.approx(reference_value(model), rel=1e-12)
    assert sol.cost_bps == pytest.approx(reference_value(model) / model.shares)
    assert lo.evaluate(model, sol.policy) == pytest.approx(sol.cost_bps, rel=1e-12)                 # played out exactly, the optimal policy costs what the table says


@pytest.mark.parametrize("model", MODELS, ids=["base", "urgent", "toxic"])
def test_no_simple_rule_beats_the_optimum(model):
    best = lo.solve(model).cost_bps
    for rule in (lo.all_market_policy, lo.twap_market_policy, lo.limit_then_market_policy):
        assert lo.evaluate(model, rule(model)) >= best - 1e-12, rule.__name__


def test_simulation_agrees_with_exact_evaluation_for_every_policy():
    model = lo.LimitOrderModel(drift_bps=1.5, risk_bps=0.3, units=20)
    sol = lo.solve(model)
    for name, policy in (("optimal", sol.policy), ("market", lo.all_market_policy(model)), ("twap", lo.twap_market_policy(model)), ("limit then market", lo.limit_then_market_policy(model))):
        sim = lo.simulate(model, policy, paths=20000, seed=3)
        exact = lo.evaluate(model, policy)
        assert abs(sim["cost_bps"] - exact) < 4 * sim["se_bps"] + 1e-9, (name, sim, exact)


def test_the_fill_distribution_has_the_stated_tail():
    model = lo.LimitOrderModel(fill_first=0.6, fill_decay=0.9)
    for s in (0, 1, 2, 7):
        pmf = model.fill_pmf(s)
        assert pmf.sum() == pytest.approx(1.0) and (pmf >= -1e-15).all() and len(pmf) == s + 1
        for j in range(1, s + 1):
            assert pmf[j:].sum() == pytest.approx(0.6 * 0.9 ** (j - 1))                          # P(fill >= j)
    rng_check = lo.simulate(lo.LimitOrderModel(fill_first=0.6, fill_decay=0.9, units=10, intervals=1), lo.limit_then_market_policy(lo.LimitOrderModel(units=10, intervals=1)), paths=40000, seed=1)
    expected_units = sum(0.6 * 0.9 ** (j - 1) for j in range(1, 11))
    assert rng_check["limit_share"] == pytest.approx(expected_units / 10, abs=0.01)                    # the simulated fills have the same mean as the pmf


def test_a_certain_fill_earns_the_spread_and_a_pure_market_order_pays_it():
    sure = lo.LimitOrderModel(fill_first=1.0, fill_decay=1.0, half_spread_bps=3.0, adverse_selection_bps=1.0)
    assert lo.solve(sure).cost_bps == pytest.approx(1.0 - 3.0)                                       # post it all: filled at the bid, less the adverse move after
    never = lo.LimitOrderModel(fill_first=0.0, drift_bps=0.0, impact_bps=8.0, half_spread_bps=3.0, intervals=4, units=20)
    sol = lo.solve(never)
    assert sol.cost_bps == pytest.approx(3.0 + 8.0 / 5, rel=1e-9)                                    # no drift: five equal market slices (four intervals and the cleanup at the end) cut the impact, which is convex
    assert lo.simulate(never, sol.policy, paths=200, seed=0)["limit_share"] == 0.0
    assert lo.evaluate(never, lo.all_market_policy(never)) == pytest.approx(3.0 + 8.0)


def test_more_urgency_means_less_patience():
    shares, last = [], None
    for drift in (0.0, 1.0, 3.0, 8.0, 20.0):
        model = lo.LimitOrderModel(drift_bps=drift, units=40)
        sol = lo.solve(model)
        sim = lo.simulate(model, sol.policy, paths=4000, seed=2)
        shares.append(sim["limit_share"])
        if last is not None:
            assert sol.cost_bps >= last - 1e-12                                                      # a drifting price can only cost more
        last = sol.cost_bps
    assert shares == sorted(shares, reverse=True) and shares[0] > 0.5 > shares[-1]                   # patient when the price stays put, a market order when it runs away


def test_the_mix_beats_both_pure_strategies_when_neither_is_obviously_right():
    model = lo.LimitOrderModel(drift_bps=2.0, risk_bps=0.15, units=50)
    best = lo.solve(model).cost_bps
    market = lo.evaluate(model, lo.all_market_policy(model))
    patient = lo.evaluate(model, lo.limit_then_market_policy(model))
    assert best < market - 0.5 and best < patient - 0.1                                              # not just a tie with one of them
    sim = lo.simulate(model, lo.solve(model).policy, paths=3000, seed=4)
    assert 0.1 < sim["limit_share"] < 0.9


def test_toxic_fills_make_limit_orders_unattractive():
    cheap = lo.LimitOrderModel(adverse_selection_bps=0.5, drift_bps=1.0, units=30)
    toxic = lo.LimitOrderModel(adverse_selection_bps=12.0, drift_bps=1.0, units=30)                   # fills come just before the price moves against you, by more than a market order's cost
    s1 = lo.simulate(cheap, lo.solve(cheap).policy, paths=3000, seed=5)["limit_share"]
    s2 = lo.simulate(toxic, lo.solve(toxic).policy, paths=3000, seed=5)["limit_share"]
    assert s1 > 0.3 and s2 == 0.0


def test_a_bad_model_is_refused():
    for kwargs in ({"shares": 0}, {"intervals": 0}, {"units": 1}, {"half_spread_bps": -1.0}, {"fill_first": 1.5}, {"fill_decay": 0.0}, {"fill_decay": 1.2}, {"risk_bps": -0.1}):
        with pytest.raises(ValueError):
            lo.LimitOrderModel(**kwargs)


def test_the_limit_command_prints_the_costs_and_the_plan_for_the_first_interval(capsys):
    from src import cli

    assert cli.main(["algo", "limit", "--drift-bps", "2", "--paths", "200", "--units", "20", "--shares", "50000"]) == 0
    out = capsys.readouterr().out
    assert "optimal mix" in out and "all market now" in out and "limit orders, market at the end" in out and "75% left" in out and "stylised model" in out
    assert cli.main(["algo", "limit", "--fill-first", "0", "--paths", "100", "--units", "10"]) == 0                       # a limit order that never fills: the optimal mix is all market orders
    assert "0.00" in capsys.readouterr().out
