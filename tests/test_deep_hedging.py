"""Deep hedging: the network's accounting is the simulator's, its criterion is the certainty equivalent, training finds a no-trade band that beats the delta hedge when trading costs money and stays the delta
hedge when it does not, and the Options tab's hedging report carries it beside the classical rules."""

from __future__ import annotations

import builtins

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.assets import option_models as om, options_analytics as oa           # noqa: E402
from src.derivatives import costs, deep_hedging as dh                          # noqa: E402

torch.set_num_threads(1)
S, K, T, R, Q, SIGMA = 100.0, 100.0, 0.25, 0.03, 0.01, 0.2


def premium(call=True):
    return float(costs.bsm_price(S, K, T, R, Q, SIGMA, call))


def paths(steps, n=4000, seed=0):
    return costs.gbm_hedging_paths(S, R - Q, SIGMA, T, steps, n, np.random.default_rng(seed))


def zero_width():
    policy = dh.HedgePolicy(8)
    torch.nn.init.constant_(policy.head.bias, -60.0)                                       # a band of no width is the delta itself
    return policy


@pytest.mark.parametrize("call", [True, False])
def test_a_band_of_no_width_is_the_delta_hedge_of_the_simulator_to_the_cent(call):
    steps, cost = 21, 0.002
    ref = costs.simulate_hedging(S, K, T, R, Q, SIGMA, cost, steps, 4000, "delta", call, seed=0)
    got = dh.evaluate(zero_width(), paths(steps), K, T, R, Q, SIGMA, cost, call, premium(call))
    assert np.allclose(got["pnl"], ref["pnl"], atol=1e-7)                                  # the same paths, the same trades, the same costs, the same interest and dividends
    assert got["cost"] == pytest.approx(ref["cost"], abs=1e-9) and got["trades"] == pytest.approx(ref["trades"], abs=0.2) and got["premium"] == pytest.approx(ref["premium"])


def test_the_shared_path_generator_gives_the_paths_the_simulator_always_used():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((50, 10))
    dt = T / 10
    old = np.concatenate([np.full((50, 1), S), S * np.exp(np.cumsum((R - Q - 0.5 * SIGMA ** 2) * dt + SIGMA * np.sqrt(dt) * z, axis=1))], axis=1)
    assert np.array_equal(costs.gbm_hedging_paths(S, R - Q, SIGMA, T, 10, 50, np.random.default_rng(3)), old)


def test_the_certainty_equivalent_is_the_exponential_utility_one():
    assert dh.certainty_equivalent(np.full(10, 1.7), 3.0) == pytest.approx(1.7)            # a sure amount is worth itself
    two = np.array([-1.0, 1.0])
    assert dh.certainty_equivalent(two, 2.0) == pytest.approx(-np.log(np.cosh(2.0)) / 2.0)
    x = np.random.default_rng(0).normal(0.3, 0.5, 400_000)
    assert dh.certainty_equivalent(x, 1.5) == pytest.approx(0.3 - 1.5 * 0.25 / 2, abs=0.005)       # for a normal: the mean less half the risk aversion times the variance
    assert dh.certainty_equivalent(x, 4.0) < dh.certainty_equivalent(x, 0.5) < x.mean()
    assert np.isfinite(dh.certainty_equivalent(np.array([-5000.0, 1.0]), 1.0))             # no overflow on a large loss
    assert om._certainty_equivalent(x, 1.5) == pytest.approx(dh.certainty_equivalent(x, 1.5))


def test_the_band_is_ordered_inside_zero_and_one_share_and_a_put_is_held_short():
    torch.manual_seed(0)
    policy = dh.HedgePolicy(8)
    spots = np.linspace(80, 120, 21)
    low, high = dh.band_curve(policy, spots, K, T, T, R, Q, SIGMA, True)
    assert np.all(low <= high) and np.all(low >= 0) and np.all(high <= 1) and np.all(np.diff(low) >= -1e-9)
    width = (high - low)[10]
    assert 0.05 < width < 0.15                                                             # it starts as a narrow band around the delta
    plow, phigh = dh.band_curve(policy, spots, K, T, T, R, Q, SIGMA, False)
    assert np.all(plow <= phigh) and np.all(phigh <= 0) and np.all(plow >= -1)
    held = dh.evaluate(policy, paths(10, 200), K, T, R, Q, SIGMA, 0.001, False, premium(False))
    assert np.isfinite(held["pnl"]).all()


def test_inside_the_band_the_network_does_not_trade_and_outside_it_goes_to_the_edge():
    policy = dh.HedgePolicy(8).double()
    m, t, d = torch.zeros(3, dtype=torch.float64), torch.full((3,), 0.5, dtype=torch.float64), torch.full((3,), 0.5, dtype=torch.float64)
    low, high = (float(v[0]) for v in policy.band(m, t, d))
    held = torch.tensor([0.5, 0.1, 0.95], dtype=torch.float64)                             # inside, below and above the band
    out = policy(m, t, d, held).detach().numpy()
    assert out[0] == 0.5 and out[1] == pytest.approx(low) and out[2] == pytest.approx(high)


def test_training_widens_the_band_and_beats_the_delta_hedge_and_its_own_start_on_paths_it_never_saw():
    steps, cost, call = 42, 0.004, True                                                    # 40 basis points a trade: standing still is worth a lot
    gamma = 1.0 / premium()
    policy, curve = dh.train(S, K, T, R, Q, SIGMA, cost, steps, call, gamma, premium(), iterations=70, batch=1024, seed=1)
    assert len(curve) == 70 and np.mean(curve[-10:]) > np.mean(curve[:5]) + 0.03          # the criterion rose during training
    torch.manual_seed(1)
    start = dh.HedgePolicy(32)
    unseen = paths(steps, 6000, seed=99)
    deep, begun, delta = (dh.evaluate(p, unseen, K, T, R, Q, SIGMA, cost, call, premium()) for p in (policy, start, zero_width()))
    ce = lambda res: dh.certainty_equivalent(res["pnl"], gamma)                           # noqa: E731
    assert ce(deep) > ce(begun) + 0.05 and ce(begun) > ce(delta) + 0.1
    assert deep["cost"] < 0.6 * delta["cost"] and deep["trades"] < 0.6 * delta["trades"]   # by trading less, not by luck
    width = lambda p: float(np.subtract(*dh.band_curve(p, np.array([100.0]), K, T, T, R, Q, SIGMA, call)[::-1])[0])      # noqa: E731
    assert width(policy) > 1.5 * width(start)
    again, _ = dh.train(S, K, T, R, Q, SIGMA, cost, steps, call, gamma, premium(), iterations=5, batch=256, seed=1)
    twin, _ = dh.train(S, K, T, R, Q, SIGMA, cost, steps, call, gamma, premium(), iterations=5, batch=256, seed=1)
    assert all(torch.equal(a, b) for a, b in zip(again.state_dict().values(), twin.state_dict().values()))      # the same seed, the same network


def test_with_free_trading_the_network_closes_its_band_and_is_the_delta_hedge():
    steps, gamma = 12, 1.0 / premium()
    policy, _ = dh.train(S, K, T, R, Q, SIGMA, 0.0, steps, True, gamma, premium(), iterations=80, batch=1024, seed=0)
    low, high = dh.band_curve(policy, np.array([95.0, 100.0, 105.0]), K, T, T, R, Q, SIGMA, True)
    assert np.all(high - low < 0.06)                                                       # narrower than it started: nothing is saved by standing still
    unseen = paths(steps, 6000, seed=5)
    deep, delta = (dh.evaluate(p, unseen, K, T, R, Q, SIGMA, 0.0, True, premium()) for p in (policy, zero_width()))
    assert dh.certainty_equivalent(deep["pnl"], gamma) == pytest.approx(dh.certainty_equivalent(delta["pnl"], gamma), abs=0.02)


def test_bad_training_settings_are_refused():
    for kwargs in ({"iterations": 0}, {"batch": 8}):
        with pytest.raises(ValueError):
            dh.train(S, K, T, R, Q, SIGMA, 0.001, 10, True, 1.0, premium(), **kwargs)
    with pytest.raises(ValueError):
        dh.train(S, K, T, R, Q, SIGMA, 0.001, 10, True, 0.0, premium())


# ------------------------------------------------------------------------------------------------------------------------------ the report
@pytest.fixture(scope="module")
def chain():
    return oa.prepare_chain(oa.build_model_chain(500.0, pd.Timestamp("2026-10-16"), 0.15, 0.18, 0.04, 0.012, 0.15, "TST", 600.0))


def test_the_hedging_report_runs_the_network_on_the_paths_of_the_other_rules_and_scores_every_rule(chain, monkeypatch):
    monkeypatch.setitem(om.DEEP_HEDGE, "iterations", 40)
    monkeypatch.setitem(om.DEEP_HEDGE, "batch", 512)
    spec = om.product_spec(chain, "vanilla", dte=30)
    inputs = om.hedge_inputs(chain, spec, None, 20, 15, 2.0, 2000, 4)
    said = []
    data, warnings, notes = om.hedge_report("TST", spec, inputs, said.append)
    rows = {r["strategy"]: r for r in data["simulation"]}
    assert list(rows) == ["delta", "leland", "band", "deep"] and all(np.isfinite(r["ce"]) for r in rows.values())
    assert rows["deep"]["premium"] == pytest.approx(rows["delta"]["premium"]) and rows["deep"]["n_paths"] == 2000 and rows["deep"]["steps"] == 15
    assert rows["deep"]["ce"] > rows["delta"]["ce"] and rows["deep"]["trades"] < rows["delta"]["trades"]
    assert data["histogram"]["share"]["deep"].sum() == pytest.approx(1.0) and set(data["histogram"]["share"]) == set(rows)
    deep = data["deep"]
    assert deep["iterations"] == 40 and len(deep["training"]) == 40 and len(deep["band_low"]) == len(data["band"]["spot"]) and np.all(np.asarray(deep["band_low"]) <= np.asarray(deep["band_high"]))
    assert any("training the hedging network" in m for m in said) and any("Deep hedging" in n for n in notes) and any("Certainty equivalent" in n for n in notes)
    text = om.format_hedge(data, warnings)
    assert isinstance(text, str) and text


def test_more_hedging_dates_get_fewer_iterations_so_that_the_time_stays_about_the_same(monkeypatch):
    budget = {"iterations": 150, "min_iterations": 40, "batch": 2048, "max_work": 150 * 2048 * 63}
    it = lambda steps: int(min(budget["iterations"], max(budget["min_iterations"], budget["max_work"] // (budget["batch"] * steps))))      # noqa: E731
    assert it(21) == 150 and it(63) == 150 and it(126) == 75 and it(504) == 40


def test_without_pytorch_the_report_keeps_the_three_rules_and_says_why(chain, monkeypatch):
    real = builtins.__import__

    def no_deep(name, globals=None, locals=None, fromlist=(), level=0):
        if fromlist and "deep_hedging" in fromlist:
            raise ImportError("No module named 'torch'")
        return real(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_deep)
    spec = om.product_spec(chain, "vanilla", dte=30)
    data, _, notes = om.hedge_report("TST", spec, om.hedge_inputs(chain, spec, None, 10, 10, 1.0, 1000, 0))
    assert [r["strategy"] for r in data["simulation"]] == ["delta", "leland", "band"] and data["deep"] is None and all(np.isfinite(r["ce"]) for r in data["simulation"])
    assert any("needs PyTorch" in n for n in notes)
