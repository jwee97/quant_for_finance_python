"""The long-short book optimiser: the named books (130/30, market neutral, dollar neutral, long-only), neutrality constraints, turnover and trading costs, against a general solver."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from src.equity.portfolio import BookSpec, optimise_book, preset


def inputs(n=12, seed=0, k=3):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, k))
    cov = (F @ F.T / k * 0.02 + np.diag(rng.uniform(0.01, 0.04, n))) * 0.1
    alpha = rng.normal(0.0, 0.02, n)
    return alpha, cov, [f"A{i:02d}" for i in range(n)]


def reference(alpha, cov, spec, held=None):
    """The same programme with SLSQP over (w_long, w_short)."""
    n = len(alpha)
    held = np.zeros(n) if held is None else held
    lam = spec.risk_aversion
    penalty = spec.penalty * np.abs(alpha).mean()
    f = lambda x: -(alpha @ (x[:n] - x[n:])) + 0.5 * lam * (x[:n] - x[n:]) @ cov @ (x[:n] - x[n:]) + penalty * x.sum()
    cons = [{"type": "eq", "fun": lambda x: x[:n].sum() - spec.gross_long}]
    if spec.gross_short > 0:
        cons.append({"type": "eq", "fun": lambda x: x[n:].sum() - spec.gross_short})
    bounds = [(0, spec.max_long)] * n + [(0, spec.max_short if spec.gross_short > 0 else 0.0)] * n
    x0 = np.concatenate([np.full(n, spec.gross_long / n), np.full(n, spec.gross_short / n)])
    r = minimize(f, x0, bounds=bounds, constraints=cons, method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000})
    return r.x[:n] - r.x[n:], r.fun


@pytest.mark.parametrize("name", ["long_only", "130_30", "market_neutral", "dollar_neutral"])
def test_the_named_books_match_a_general_solver(name):
    alpha, cov, names = inputs(12, 1)
    spec = preset(name, max_long=0.3, max_short=0.3, risk_aversion=8.0)
    r = optimise_book(alpha, cov, spec, names=names)
    w, fun = reference(alpha, cov, spec)
    assert r.ok and np.abs(r.weights.to_numpy() - w).max() < 1e-4
    assert r.exposures["gross_long"] == pytest.approx(spec.gross_long, abs=1e-7) and r.exposures["gross_short"] == pytest.approx(spec.gross_short, abs=1e-7)
    assert r.exposures["net"] == pytest.approx(spec.net, abs=1e-7)
    assert (r.weights.abs() <= 0.3 + 1e-9).all()


def test_the_book_that_may_short_does_better_than_the_one_that_may_not():
    alpha, cov, names = inputs(20, 2)
    kw = dict(max_long=0.2, max_short=0.2, risk_aversion=5.0)
    utility = lambda w: alpha @ w - 2.5 * w @ cov @ w
    long_only = optimise_book(alpha, cov, preset("long_only", **kw), names=names).weights.to_numpy()
    long_short = optimise_book(alpha, cov, preset("130_30", **kw), names=names).weights.to_numpy()
    assert (long_only >= -1e-9).all() and (long_short < 0).any() and utility(long_short) > utility(long_only)


def test_no_name_is_long_and_short_at_once_and_nothing_is_held_without_a_reason():
    alpha, cov, names = inputs(10, 3)
    flat = optimise_book(np.zeros(10), cov, BookSpec(gross_long=1.0, gross_short=1.0, exact=False, max_long=0.3, max_short=0.3), names=names)
    assert flat.ok and np.abs(flat.weights).max() < 1e-6                                           # no view, a limit rather than a target: hold nothing
    r = optimise_book(alpha, cov, BookSpec(gross_long=1.3, gross_short=0.3, exact=False, max_long=0.3, max_short=0.3), names=names)
    assert r.ok and r.exposures["gross_long"] <= 1.3 + 1e-7 and r.exposures["gross_short"] <= 0.3 + 1e-7


def test_market_dollar_and_beta_neutral_books_have_the_exposures_they_were_asked_for():
    alpha, cov, names = inputs(30, 4)
    rng = np.random.default_rng(4)
    beta = rng.uniform(0.5, 1.5, 30)
    spec = BookSpec(gross_long=1.0, gross_short=1.0, max_long=0.15, max_short=0.15, beta=beta, beta_target=0.0, beta_tolerance=0.0)
    r = optimise_book(alpha, cov, spec, names=names)
    assert r.ok and abs(r.exposures["net"]) < 1e-7 and abs(r.exposures["beta"]) < 1e-7                # dollar neutral and beta neutral
    loose = optimise_book(alpha, cov, BookSpec(gross_long=1.0, gross_short=1.0, max_long=0.15, max_short=0.15), names=names)
    utility = lambda res: alpha @ res.weights.to_numpy() - 2.5 * res.weights.to_numpy() @ cov @ res.weights.to_numpy()
    assert utility(loose) >= utility(r) - 1e-9 and abs(beta @ loose.weights.to_numpy()) > 1e-4         # a constraint can only cost utility; the unconstrained book was not beta neutral
    band = optimise_book(alpha, cov, BookSpec(gross_long=1.0, gross_short=1.0, max_long=0.15, max_short=0.15, beta=beta, beta_tolerance=0.05), names=names)
    assert abs(band.exposures["beta"]) <= 0.05 + 1e-7 and utility(r) <= utility(band) + 1e-9
    biased = optimise_book(alpha, cov, BookSpec(gross_long=1.0, gross_short=1.0, max_long=0.15, max_short=0.15, beta=beta, beta_target=0.2, beta_tolerance=0.0), names=names)
    assert biased.exposures["beta"] == pytest.approx(0.2, abs=1e-7)


def test_a_sector_neutral_book_nets_to_the_benchmark_in_every_group():
    alpha, cov, names = inputs(30, 5)
    groups = {n: f"S{i % 5}" for i, n in enumerate(names)}
    spec = BookSpec(gross_long=1.3, gross_short=0.3, max_long=0.15, max_short=0.15, groups=groups, group_tolerance=0.0)
    r = optimise_book(alpha, cov, spec, names=names)
    assert r.ok and all(abs(v - 1.0 / 5) < 1e-7 for v in r.exposures["group_net"].values())          # each of five equal sectors nets to a fifth of the 100% net exposure
    market = optimise_book(alpha, cov, BookSpec(gross_long=1.0, gross_short=1.0, max_long=0.15, max_short=0.15, groups=groups, group_tolerance=0.0), names=names)
    assert all(abs(v) < 1e-7 for v in market.exposures["group_net"].values())
    custom = optimise_book(alpha, cov, BookSpec(gross_long=1.3, gross_short=0.3, max_long=0.2, max_short=0.2, groups=groups, group_tolerance=0.02,
                                                group_targets={"S0": 0.4, "S1": 0.15, "S2": 0.15, "S3": 0.15, "S4": 0.15}), names=names)
    assert abs(custom.exposures["group_net"]["S0"] - 0.4) <= 0.02 + 1e-7
    free = optimise_book(alpha, cov, BookSpec(gross_long=1.3, gross_short=0.3, max_long=0.15, max_short=0.15), names=names)
    assert alpha @ free.weights.to_numpy() - 2.5 * free.weights.to_numpy() @ cov @ free.weights.to_numpy() >= alpha @ r.weights.to_numpy() - 2.5 * r.weights.to_numpy() @ cov @ r.weights.to_numpy() - 1e-9


def test_other_exposures_can_be_bounded_and_the_bound_binds_when_the_alpha_leans_on_it():
    alpha, cov, names = inputs(40, 6)
    rng = np.random.default_rng(6)
    size = rng.normal(size=40)
    leaning = alpha + 0.03 * size                                                                   # the alpha is partly a bet on size
    free = optimise_book(leaning, cov, BookSpec(gross_long=1.3, gross_short=0.3, max_long=0.1, max_short=0.1), names=names)
    bounded = optimise_book(leaning, cov, BookSpec(gross_long=1.3, gross_short=0.3, max_long=0.1, max_short=0.1, loadings=size[None, :], loading_bounds=np.array([[-0.1, 0.1]])), names=names)
    assert abs(size @ free.weights.to_numpy()) > 0.2 and abs(bounded.exposures["loadings"][0]) <= 0.1 + 1e-7 and abs(bounded.exposures["loadings"][0]) > 0.1 - 1e-6


def test_a_turnover_limit_binds_and_a_trading_cost_creates_a_no_trade_region():
    alpha, cov, names = inputs(20, 7)
    spec = BookSpec(gross_long=1.3, gross_short=0.3, max_long=0.2, max_short=0.2)
    start = optimise_book(alpha, cov, spec, names=names).weights.to_numpy()
    new_alpha = alpha + np.random.default_rng(7).normal(0, 0.02, 20)                                 # the view changes
    free = optimise_book(new_alpha, cov, spec, held=start, names=names)
    assert free.exposures["turnover"] > 0.5
    limited = optimise_book(new_alpha, cov, BookSpec(**{**spec.__dict__, "max_turnover": 0.25}), held=start, names=names)
    assert limited.ok and limited.exposures["turnover"] == pytest.approx(0.25, abs=1e-6)             # the limit binds
    utility = lambda w: new_alpha @ w - 2.5 * w @ cov @ w
    assert utility(free.weights.to_numpy()) >= utility(limited.weights.to_numpy()) >= utility(start) - 1e-9
    costly = optimise_book(new_alpha, cov, BookSpec(**{**spec.__dict__, "trade_cost": 0.02}), held=start, names=names)
    assert costly.exposures["turnover"] < free.exposures["turnover"] * 0.6                            # a cost of 2% a unit leaves most of the book alone
    tiny = optimise_book(alpha + 1e-4, cov, BookSpec(**{**spec.__dict__, "trade_cost": 0.02}), held=start, names=names)
    assert tiny.exposures["turnover"] < 1e-5                                                          # a view that hardly changed: no trade
    same = optimise_book(new_alpha, cov, BookSpec(**{**spec.__dict__, "trade_cost": 0.0}), held=start, names=names)
    assert same.exposures["turnover"] == pytest.approx(free.exposures["turnover"], abs=1e-6)


def test_the_answer_does_not_depend_on_the_units_of_the_inputs():
    alpha, cov, names = inputs(15, 8)
    spec = preset("130_30", max_long=0.2, max_short=0.2, risk_aversion=6.0)
    base = optimise_book(alpha, cov, spec, names=names).weights.to_numpy()
    for k in (0.01, 100.0):
        scaled = optimise_book(k * alpha, k * cov, spec, names=names).weights.to_numpy()
        assert np.abs(scaled - base).max() < 5e-4, k


def test_infeasible_specifications_are_reported_not_returned_as_books():
    alpha, cov, names = inputs(10, 9)
    r = optimise_book(alpha, cov, BookSpec(gross_long=1.3, gross_short=0.3, max_long=0.1, max_short=0.1), names=names)         # ten names capped at 10% cannot hold 130% long
    assert not r.ok and "infeasible" in r.status and (r.weights == 0).all()
    with pytest.raises(ValueError):
        optimise_book(alpha, cov[:5, :5], BookSpec(), names=names)
    with pytest.raises(ValueError):
        optimise_book(alpha, cov, BookSpec(risk_aversion=-1.0), names=names)
    with pytest.raises(ValueError):
        optimise_book(alpha, cov, BookSpec(max_long=-0.1), names=names)
    from src.equity.portfolio import preset as make
    with pytest.raises(KeyError, match="unknown book"):
        make("150_50")


def test_an_equity_sized_sector_and_beta_neutral_book_is_solved_and_satisfies_everything():
    n = 250
    alpha, cov, names = inputs(n, 10, k=5)
    rng = np.random.default_rng(10)
    groups = {nm: f"G{i % 8}" for i, nm in enumerate(names)}
    beta = rng.uniform(0.6, 1.4, n)
    spec = BookSpec(gross_long=1.3, gross_short=0.3, max_long=0.03, max_short=0.03, groups=groups, group_tolerance=0.01, beta=beta, beta_target=1.0, beta_tolerance=0.02, risk_aversion=10.0)
    r = optimise_book(alpha, cov, spec, names=names)
    assert r.ok
    w = r.weights.to_numpy()
    assert (w <= 0.03 + 1e-7).all() and (w >= -0.03 - 1e-7).all() and w.sum() == pytest.approx(1.0, abs=1e-6)
    share = pd.Series(groups).value_counts() / n                                                       # the benchmark: each group's share of the names
    assert all(abs(v - share[g]) <= 0.01 + 1e-6 for g, v in r.exposures["group_net"].items()) and abs(beta @ w - 1.0) <= 0.02 + 1e-6
    assert alpha @ w > 0


def test_a_gross_target_is_met_by_real_one_sided_positions_not_by_holding_a_name_long_and_short():
    alpha, cov, names = inputs(24, 11)
    held = np.zeros(24)
    held[:12], held[12:] = 0.6 / 12, -0.6 / 12                                                       # a market-neutral book with only 60% on each side
    spec = BookSpec(gross_long=1.0, gross_short=1.0, max_long=0.3, max_short=0.3, trade_cost=0.4)   # trading is so dear that the cheapest way to "reach" 100% is to cancel in pairs
    r = optimise_book(alpha * 0.2, cov, spec, held=held, names=names)
    assert r.ok and r.exposures["gross_long"] == pytest.approx(1.0, abs=1e-6) and r.exposures["gross_short"] == pytest.approx(1.0, abs=1e-6) and abs(r.exposures["net"]) < 1e-6
    assert r.status == "solved (one side per name)" or r.status.startswith("solved")


def test_a_book_with_limits_fixes_the_net_exposure_and_may_use_less_gross_than_the_limit():
    alpha, cov, names = inputs(20, 12)
    weak = optimise_book(0.05 * alpha, cov, BookSpec(gross_long=1.0, gross_short=1.0, exact=False, max_long=0.3, max_short=0.3, risk_aversion=20.0), names=names)
    strong = optimise_book(alpha, cov, BookSpec(gross_long=1.0, gross_short=1.0, exact=False, max_long=0.3, max_short=0.3, risk_aversion=20.0), names=names)
    assert weak.ok and abs(weak.exposures["net"]) < 1e-7 and weak.exposures["gross_long"] < 0.7 < 1.0               # no view worth the risk: a smaller book
    assert strong.exposures["gross_long"] > weak.exposures["gross_long"] and strong.exposures["gross_long"] <= 1.0 + 1e-7
    tilted = optimise_book(alpha, cov, BookSpec(gross_long=1.3, gross_short=0.3, exact=False, net_exposure=0.8, max_long=0.3, max_short=0.3), names=names)
    assert tilted.exposures["net"] == pytest.approx(0.8, abs=1e-7) and tilted.exposures["gross_long"] <= 1.3 + 1e-7 and tilted.exposures["gross_short"] <= 0.3 + 1e-7
