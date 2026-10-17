"""The Ornstein-Uhlenbeck process: exact simulation, the maximum-likelihood fit and what its standard errors say, first-passage times against simulation, and the trading levels that maximise
the profit per unit of time against both the first-order condition and a simulated strategy."""

from __future__ import annotations

import numpy as np
import pytest

from src.probability import ou

THETA, MU, SIGMA = 4.0, 0.0, 1.0


# ------------------------------------------------------------------------------------------------------------------------------ simulation
def test_the_simulation_is_exact_for_any_step_and_stationary_from_the_stationary_law():
    x0, T = 1.5, 0.7
    mean = MU + (x0 - MU) * np.exp(-THETA * T)
    var = SIGMA ** 2 * (1.0 - np.exp(-2 * THETA * T)) / (2 * THETA)
    for steps in (1, 7, 70):
        x = ou.simulate_ou(x0, THETA, MU, SIGMA, T, steps, 80_000, seed=steps)
        assert x.shape == (80_000, steps + 1) and np.all(x[:, 0] == x0)
        assert x[:, -1].mean() == pytest.approx(mean, abs=4.5 * np.sqrt(var / 80_000)) and x[:, -1].var() == pytest.approx(var, rel=0.03), steps
    m, sd = ou.stationary(THETA, MU, SIGMA)
    assert m == MU and sd == pytest.approx(SIGMA / np.sqrt(2 * THETA)) and ou.half_life(THETA) == pytest.approx(np.log(2) / THETA)
    paths = ou.simulate_ou(MU, THETA, MU, SIGMA, 2.0, 40, 100_000, seed=9)
    assert paths[:, 0].std() == 0.0 and paths[:, 20].std() == pytest.approx(paths[:, 40].std(), rel=0.02), "started at the mean it spreads out to the stationary law and stays there"
    assert paths[:, 40].std() == pytest.approx(sd, rel=0.02)
    with pytest.raises(ValueError):
        ou.simulate_ou(0.0, -1.0, MU, SIGMA, 1.0, 10)


# ------------------------------------------------------------------------------------------------------------------------------------ fit
def test_the_fit_recovers_the_parameters_and_its_standard_errors_describe_the_spread_of_the_estimates():
    theta, mu, sigma, dt, n = 40.0, 0.1, 0.5, 1 / 252, 1000
    est, se = [], []
    for seed in range(300):
        x = ou.simulate_ou(mu, theta, mu, sigma, n * dt, n, 1, seed=seed)[0]
        f = ou.fit_ou(x, dt)
        est.append([f.estimates[k] for k in ("theta", "mu", "sigma")])
        se.append([f.std_errors[k] for k in ("theta", "mu", "sigma")])
    est, se = np.array(est), np.array(se)
    assert est[:, 2].mean() == pytest.approx(sigma, rel=0.02) and est[:, 1].mean() == pytest.approx(mu, abs=0.01)
    assert est[:, 0].mean() == pytest.approx(theta, rel=0.25), "the speed is biased upwards in a short sample (the usual small-sample bias of an autoregression), and is the least precise"
    for j in range(3):
        assert est[:, j].std() == pytest.approx(se[:, j].mean(), rel=0.2), j
    truth = np.array([theta, mu, sigma])
    cover = np.mean((est - 1.96 * se <= truth) & (truth <= est + 1.96 * se), axis=0)
    assert cover[1] > 0.9 and cover[2] > 0.9 and cover[0] > 0.85, cover


def test_the_exact_map_differs_from_the_euler_value_when_the_step_is_not_small():
    dt = 0.5                                                          # a step of half a year for a speed of 3: theta dt = 1.5
    x = ou.simulate_ou(0.0, 3.0, 0.0, 1.0, 5000 * dt, 5000, 1, seed=3)[0]
    f = ou.fit_ou(x, dt)
    phi = float(np.polyfit(x[:-1], x[1:], 1)[0])
    euler = (1.0 - phi) / dt
    assert abs(f.estimates["theta"] - 3.0) < 0.15 and abs(euler - 3.0) > 0.9
    ll_truth = ou.ou_loglik((3.0, 0.0, 1.0), x, dt)
    assert f.loglik >= ll_truth - 1e-6 and f.ci["theta"][0] < 3.0 < f.ci["theta"][1] and f.n == 5000


def test_a_random_walk_is_not_distinguishable_from_no_reversion_and_explosive_or_alternating_series_are_refused():
    t_stats = []
    for seed in range(40):
        walk = np.cumsum(np.random.default_rng(seed).standard_normal(1000)) * 0.01
        try:
            f = ou.fit_ou(walk, 1 / 252)
        except ValueError:
            t_stats.append(0.0)
            continue
        t_stats.append(f.estimates["theta"] / f.std_errors["theta"])
    assert np.mean(np.array(t_stats) < 3.0) > 0.8, "the fit does not claim reversion where there is none, by its own standard error"
    explosive = 1.01 ** np.arange(300) * 0.01 + np.random.default_rng(0).normal(0, 1e-4, 300)
    with pytest.raises(ValueError, match="no mean reversion"):
        ou.fit_ou(explosive, 1 / 252)
    alternating = np.array([(-1) ** i * 0.5 for i in range(200)]) + np.random.default_rng(1).normal(0, 0.01, 200)
    with pytest.raises(ValueError, match="persistence"):
        ou.fit_ou(alternating, 1 / 252)
    with pytest.raises(ValueError):
        ou.fit_ou(np.zeros(100), 1 / 252)
    with pytest.raises(ValueError):
        ou.fit_ou(np.arange(10.0), 1 / 252)
    nan_series = ou.simulate_ou(0.0, 30.0, 0.0, 0.3, 4.0, 1000, 1, seed=2)[0]
    nan_series[100] = np.nan
    assert ou.fit_ou(nan_series, 1 / 250).n == 999, "a missing value is dropped (1001 values, one missing: 999 transitions)"


# ------------------------------------------------------------------------------------------------------------------------- hitting times
def _simulated_hitting(x0, level, dt, n, seed, horizon=30.0):
    rng = np.random.default_rng(seed)
    x = np.full(n, x0)
    alive = np.ones(n, bool)
    t_hit = np.full(n, np.nan)
    phi = np.exp(-THETA * dt)
    sd = SIGMA * np.sqrt((1 - phi ** 2) / (2 * THETA))
    up = level > x0
    for i in range(int(horizon / dt)):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        x[idx] = MU + (x[idx] - MU) * phi + sd * rng.standard_normal(len(idx))
        hit = idx[(x[idx] >= level) if up else (x[idx] <= level)]
        t_hit[hit] = (i + 1) * dt
        alive[hit] = False
    return float(np.nanmean(t_hit)), float(np.nanstd(t_hit))


@pytest.mark.parametrize("x0,level", [(0.0, 0.5), (0.5, -0.4)])
def test_the_expected_hitting_time_and_its_spread_agree_with_simulation(x0, level):
    mean, sd = ou.hitting_time_moments(x0, level, THETA, MU, SIGMA)
    assert mean == pytest.approx(ou.hitting_time(x0, level, THETA, MU, SIGMA), rel=1e-4)
    coarse, fine = _simulated_hitting(x0, level, 4e-3, 12_000, 1), _simulated_hitting(x0, level, 1e-3, 12_000, 2)
    extrapolated = [2.0 * f - c for c, f in zip(coarse, fine)]               # monitoring on a grid overstates the time by an amount that goes with sqrt(dt)
    assert coarse[0] > fine[0] > mean * 0.97, "a coarse monitoring misses crossings and finds the time longer"
    assert mean == pytest.approx(extrapolated[0], rel=0.06) and sd == pytest.approx(extrapolated[1], rel=0.08), (mean, sd, extrapolated)


def test_hitting_times_add_up_reflect_and_the_cycle_is_their_sum():
    a, c, b = -0.4, 0.1, 0.6
    assert ou.hitting_time(a, c, THETA, MU, SIGMA) + ou.hitting_time(c, b, THETA, MU, SIGMA) == pytest.approx(ou.hitting_time(a, b, THETA, MU, SIGMA), rel=1e-9), "a path to b passes c"
    assert ou.hitting_time(0.3, 0.9, THETA, 0.2, SIGMA) == pytest.approx(ou.hitting_time(0.1, -0.5, THETA, 0.2, SIGMA), rel=1e-9), "the process is symmetric about its mean"
    assert ou.cycle_time(a, b, THETA, MU, SIGMA) == pytest.approx(ou.hitting_time(a, b, THETA, MU, SIGMA) + ou.hitting_time(b, a, THETA, MU, SIGMA), rel=1e-9)
    assert ou.hitting_time(0.3, 0.3, THETA, MU, SIGMA) == 0.0 and ou.hitting_time_moments(0.3, 0.3, THETA, MU, SIGMA) == (0.0, 0.0)
    assert ou.hitting_time(0.0, 1e-3, THETA, MU, SIGMA) == pytest.approx(np.sqrt(np.pi / THETA) * 1e-3 / SIGMA, rel=3e-3), "a level just above is reached in a time linear in the distance"
    far = [ou.hitting_time(0.0, lv, THETA, MU, SIGMA) for lv in (0.5, 1.0, 1.5, 2.0)]
    assert all(x < y for x, y in zip(far, far[1:])) and far[-1] > 10 * far[0], "the time to reach a far level grows faster than the distance (the pull back grows with it)"
    with pytest.raises(ValueError):
        ou.cycle_time(0.5, -0.5, THETA, MU, SIGMA)
    with pytest.raises(ValueError):
        ou.hitting_time(0.0, 1.0, -1.0, MU, SIGMA)


# ------------------------------------------------------------------------------------------------------------------------------ thresholds
def test_the_optimal_levels_satisfy_the_first_order_condition_beat_every_other_pair_and_move_out_with_the_cost():
    theta, mu, sigma = 3.0, 0.2, 0.5
    cost = 0.05
    best = ou.optimal_thresholds(theta, mu, sigma, cost)
    d = best["half_width"]
    assert best["entry"] == pytest.approx(mu - d) and best["exit"] == pytest.approx(mu + d)
    assert best["rate"] == pytest.approx((2 * d - cost) / ou.cycle_time(mu - d, mu + d, theta, mu, sigma))
    rate = lambda width: (2 * width - cost) / ou.cycle_time(mu - width, mu + width, theta, mu, sigma)           # noqa: E731
    h = 1e-5
    assert abs((rate(d + h) - rate(d - h)) / (2 * h)) < 1e-4 * best["rate"], "the profit rate is flat at the optimum"
    assert rate(d * 0.7) < best["rate"] and rate(d * 1.4) < best["rate"]
    # no asymmetric pair does better
    others = [(m2 - a2 - cost) / ou.cycle_time(a2, m2, theta, mu, sigma) for a2 in np.linspace(mu - 0.6, mu + 0.1, 36) for m2 in np.linspace(mu - 0.1, mu + 0.6, 36) if m2 > a2 + cost]
    assert max(others) <= best["rate"] + 1e-9
    costs = [0.0, 0.01, 0.03, 0.1, 0.3]
    results = [ou.optimal_thresholds(theta, mu, sigma, c) for c in costs]
    assert all(r1["half_width"] < r2["half_width"] for r1, r2 in zip(results, results[1:])) and all(r1["rate"] > r2["rate"] for r1, r2 in zip(results, results[1:]))
    assert results[0]["half_width"] < 1e-3 and results[0]["rate"] > 0, "free trading: trade at the mean, constantly"
    assert best["hold"] == pytest.approx(best["wait"], rel=1e-6) and best["cycle"] == pytest.approx(best["hold"] + best["wait"]) and best["half_width_sd"] == pytest.approx(d / (sigma / np.sqrt(2 * theta)))
    with pytest.raises(ValueError, match="no pair of levels"):
        ou.optimal_thresholds(theta, mu, sigma, 50.0)
    with pytest.raises(ValueError):
        ou.optimal_thresholds(-1.0, mu, sigma, 0.1)


def _run_levels(x, entry, exit_, cost):
    long, trades, profit = False, 0, 0.0
    for v in x:                                                                    # buy at the lower level, sell at the upper one
        if not long and v <= entry:
            long = True
        elif long and v >= exit_:
            long = False
            trades += 1
            profit += (exit_ - entry) - cost
    return trades, profit


def test_a_simulated_strategy_trades_at_the_predicted_frequency_and_earns_the_predicted_rate():
    """Checking the levels on a grid misses crossings between the dates and so trades a little less than the continuous process would; the shortfall goes with the square root of the step,
    so two step sizes give the continuous-time rate by extrapolation."""
    theta, mu, sigma, cost, T = 30.0, 0.0, 0.4, 0.04, 300.0
    levels = ou.optimal_thresholds(theta, mu, sigma, cost)
    rates = []
    for steps in (750_000, 3_000_000):
        x = ou.simulate_ou(mu, theta, mu, sigma, T, steps, 1, seed=7)[0]
        trades, profit = _run_levels(x, levels["entry"], levels["exit"], cost)
        rates.append((trades / T, profit / T))
    coarse, fine = rates
    assert fine[0] > coarse[0], "a finer grid finds more crossings"
    trade_rate, profit_rate = 2.0 * fine[0] - coarse[0], 2.0 * fine[1] - coarse[1]
    assert trade_rate == pytest.approx(1.0 / levels["cycle"], rel=0.15), (trade_rate, 1.0 / levels["cycle"])
    assert profit_rate == pytest.approx(levels["rate"], rel=0.2), (profit_rate, levels["rate"])
