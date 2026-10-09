"""Smart order routing: the marginal allocation against exhaustive search, the Kaplan-Meier estimator against known censored data, and a router that learns."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.algo import routing as rt


def test_a_venue_draws_liquidity_with_the_stated_tail():
    v = rt.Venue("x", 0.6, 0.97)
    rng = np.random.default_rng(0)
    draws = v.draw(rng, 200000)
    tail = v.tail(40)
    for j in (1, 2, 5, 10, 25, 40):
        assert (draws >= j).mean() == pytest.approx(tail[j - 1], abs=0.005)
    assert (draws == 0).mean() == pytest.approx(0.4, abs=0.005) and draws.min() == 0
    for bad in (dict(first=1.2, decay=0.9), dict(first=0.5, decay=0.0), dict(first=0.5, decay=1.1), dict(first=0.5, decay=0.9, value=-1.0)):
        with pytest.raises(ValueError):
            rt.Venue("bad", **bad)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_marginal_allocation_is_the_exhaustive_optimum(seed):
    rng = np.random.default_rng(seed)
    venues = [rt.Venue(f"v{i}", rng.uniform(0.1, 1.0), rng.uniform(0.7, 0.99), rng.uniform(0.5, 1.5)) for i in range(3)]
    S = 12
    tails = np.vstack([v.tail(S) for v in venues])
    got = rt.greedy_allocation(tails, S, [v.value for v in venues])
    best, best_alloc = -np.inf, None
    for a in itertools.product(range(S + 1), repeat=3):
        if sum(a) == S:
            val = rt.expected_fill(venues, a)
            if val > best + 1e-12:
                best, best_alloc = val, a
    assert got.sum() == S and rt.expected_fill(venues, got) == pytest.approx(best, abs=1e-12), (got, best_alloc)


def test_the_allocation_equalises_the_marginal_probabilities_and_ties_split_evenly():
    venues = rt.example_venues()
    tails = np.vstack([v.tail(300) for v in venues])
    a = rt.greedy_allocation(tails, 300)
    assert a.sum() == 300
    last_taken = [tails[i, a[i] - 1] if a[i] > 0 else np.inf for i in range(len(venues))]
    next_offered = [tails[i, a[i]] if a[i] < 300 else -np.inf for i in range(len(venues))]
    assert min(last_taken) >= max(next_offered) - 1e-6                                               # nothing left behind is better than the worst thing taken
    assert rt.greedy_allocation(np.ones((4, 20)), 20).tolist() == [5, 5, 5, 5]                      # unexplored venues are tried alike
    assert a[0] < a[1] and a[3] == 0                                                                 # the venue most likely to fill anything (the first) is not the one that gets the most, and a deep pool is not needed yet
    big = rt.greedy_allocation(np.vstack([v.tail(2000) for v in venues]), 2000)
    assert big[3] > big[0] and big[4] == 0                                                           # an order big enough to exhaust the shallow venues goes to the deep ones
    with pytest.raises(ValueError):
        rt.greedy_allocation(np.ones((2, 5)), 10)


def censored_sample(venue, sizes, rng):
    V = venue.draw(rng, len(sizes))
    return sizes, np.minimum(sizes, V).astype(int)


def test_kaplan_meier_recovers_the_tail_that_a_naive_estimator_underestimates():
    venue = rt.Venue("deep", 0.7, 0.985)
    rng = np.random.default_rng(1)
    sizes = rng.integers(1, 120, 6000)
    sent, filled = censored_sample(venue, sizes, rng)
    km = rt.kaplan_meier_tail(sent, filled, 100)
    truth = venue.tail(100)
    assert np.abs(km - truth).max() < 0.04 and (np.diff(km) <= 1e-12).all()
    naive = np.array([(filled >= j).sum() / len(filled) for j in range(1, 101)])                    # treats "filled everything sent" as if it were the venue's whole liquidity, and ignores sizes
    assert (truth[10:] - naive[10:]).mean() > 0.1 and np.abs(km - truth).mean() < 0.4 * np.abs(naive - truth).mean()


def test_kaplan_meier_on_simple_cases_and_where_there_is_no_data():
    assert rt.kaplan_meier_tail([5, 5, 5], [5, 5, 5], 8).tolist() == [1.0] * 8                       # every send filled in full: nothing says the venue ever runs out
    km = rt.kaplan_meier_tail([10, 10, 10, 10], [0, 2, 4, 10], 6)                                     # three exact observations (0, 2, 4 lots) and one censored (at least 10)
    assert km.tolist() == pytest.approx([0.75, 0.75, 0.5, 0.5, 0.25, 0.25])                          # by hand: hazards 1/4 at 0, 1/3 at 2, 1/2 at 4 (at risk 4, 3, 2)
    values = np.array([i % 10 for i in range(50)])
    no_censoring = rt.kaplan_meier_tail([20] * 50, values.tolist(), 12)                               # with no censoring it is the empirical survival function
    assert no_censoring == pytest.approx([(values >= j).mean() for j in range(1, 13)])
    assert rt.kaplan_meier_tail([], [], 5).tolist() == [1.0] * 5                                      # no data: optimistic about everything
    for bad in (([3], [4]), ([3, 4], [1]), ([3], [-1])):
        with pytest.raises(ValueError):
            rt.kaplan_meier_tail(*bad, width=5)


def test_the_learning_router_closes_most_of_the_gap_to_the_oracle():
    venues = rt.example_venues()
    results = {p: np.mean([rt.simulate_routing(venues, rounds=600, shares=400, policy=p, seed=s)["expected_fill_rate_last_third"] for s in range(4)]) for p in rt.POLICIES}
    uniform, first_unit, learned, oracle = (results[p] for p in rt.POLICIES)
    assert oracle >= learned >= 0.95 * oracle                                                       # within five percent of knowing the venues after 600 rounds
    assert learned > uniform + 0.08 and learned > first_unit + 0.25                                   # clearly better than splitting evenly or than following the highest probability of any fill


def test_without_optimism_a_venue_that_disappoints_early_is_never_tried_again():
    venues = rt.example_venues()
    blind = np.mean([rt.simulate_routing(venues, rounds=600, shares=400, seed=s, optimism=0.0)["expected_fill_rate_last_third"] for s in range(4)])
    curious = np.mean([rt.simulate_routing(venues, rounds=600, shares=400, seed=s, optimism=2.0)["expected_fill_rate_last_third"] for s in range(4)])
    assert curious > blind + 0.05                                                                    # the plain Kaplan-Meier router gets stuck on the first venues that looked fine


def test_it_improves_with_experience_and_a_zero_value_venue_is_ignored():
    venues = rt.example_venues()
    early_late = [rt.simulate_routing(venues, rounds=300, shares=400, policy="learned (Kaplan-Meier, optimistic)", seed=s) for s in range(4)]
    first_third = np.mean([r["expected_by_round"][:100].mean() for r in early_late])
    last_third = np.mean([r["expected_by_round"][-100:].mean() for r in early_late])
    assert last_third > first_third
    worthless = [rt.Venue("fees eat it", 0.99, 0.999, value=0.0)] + venues[:2]
    out = rt.simulate_routing(worthless, rounds=100, shares=200, policy="oracle (true distributions)", seed=0)
    assert out["allocation"][0] == 0 and out["allocation"].sum() == 200                               # a venue whose fills are worth nothing gets nothing


def test_the_policies_are_named_and_a_bad_one_is_refused():
    assert len(rt.POLICIES) == 4
    with pytest.raises(ValueError):
        rt.simulate_routing(rt.example_venues(), rounds=3, shares=10, policy="clairvoyant")
    out = rt.simulate_routing(rt.example_venues(), rounds=30, shares=50, policy="uniform", seed=1)
    assert out["allocation"].sum() == 50 and 0.0 <= out["fill_rate"] <= 1.0


def test_the_route_command_compares_the_four_routers(capsys):
    from src import cli

    assert cli.main(["algo", "route", "--rounds", "60", "--lots", "100"]) == 0
    out = capsys.readouterr().out
    assert all(p in out for p in rt.POLICIES) and "lit exchange" in out and "expected fill rate" in out
