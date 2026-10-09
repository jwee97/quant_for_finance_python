"""Black-box and high-frequency research simulators: pair trading, ETF arbitrage, auto market making and rebate / liquidity trading."""

from __future__ import annotations

import numpy as np
import pytest

from src.algo.blackbox import latency_table, simulate_etf_arbitrage, simulate_pair_trading
from src.algo.hft import auto_market_making, pressure_table, simulate_rebate_trading


# --------------------------------------------------------------------------------------------------------------------------- pair trading
@pytest.mark.parametrize("kwargs", [{"days": 1}, {"bars": 5}, {"window": 3}, {"half_life": 0}, {"entry": 0.2, "exit": 0.3}, {"stop": 1.0}, {"latency": -1}, {"cost_bps": -1}])
def test_a_pair_trade_that_cannot_be_defined_is_refused(kwargs):
    with pytest.raises(ValueError):
        simulate_pair_trading(**kwargs)


def test_the_pair_rule_earns_the_reversion_when_it_is_free_to_trade_and_not_when_the_spread_has_no_memory():
    free = simulate_pair_trading(cost_bps=0.0, seed=1)
    assert free["mean_daily_bps"] > 6 * free["se_daily_bps"] and free["win_rate"] > 0.7 and free["sharpe"] > 5
    control = simulate_pair_trading(cost_bps=0.0, mean_reverting=False, seed=1)
    assert abs(control["mean_daily_bps"]) < 3.5 * control["se_daily_bps"]                                       # a random-walk spread: the same rule earns nothing


def test_costs_come_straight_off_the_profit_and_can_erase_it():
    free = simulate_pair_trading(cost_bps=0.0, seed=2)
    paid = simulate_pair_trading(cost_bps=1.0, seed=2)
    assert paid["gross_bps"] == pytest.approx(free["gross_bps"]) and paid["mean_daily_bps"] == pytest.approx(paid["gross_bps"] - paid["cost_bps_total"])
    assert paid["cost_bps_total"] > 0 and paid["mean_daily_bps"] < 0.4 * free["mean_daily_bps"]
    dear = simulate_pair_trading(cost_bps=3.0, seed=2)
    assert dear["mean_daily_bps"] < 0 and simulate_pair_trading(cost_bps=3.0, mean_reverting=False, seed=2)["mean_daily_bps"] < dear["mean_daily_bps"] + 1e-9


def test_a_late_order_loses_the_edge_and_the_book_is_flat_at_the_close():
    fast, slow = simulate_pair_trading(cost_bps=0.0, latency=0, seed=3), simulate_pair_trading(cost_bps=0.0, latency=10, seed=3)
    assert fast["gross_bps"] > slow["gross_bps"] + 1e-9
    for r in (fast, slow):
        assert set(np.unique(r["position"])) <= {-1.0, 0.0, 1.0} and (r["position"][:, -1] == 0).all()


def test_a_faster_reverting_spread_is_held_for_less_time_and_the_run_is_reproducible():
    quick, lazy = simulate_pair_trading(half_life=15.0, seed=4), simulate_pair_trading(half_life=90.0, seed=4)
    assert quick["average_hold_bars"] < lazy["average_hold_bars"]
    again = simulate_pair_trading(half_life=15.0, seed=4)
    assert again["daily_bps"] == pytest.approx(quick["daily_bps"]) and not np.allclose(simulate_pair_trading(half_life=15.0, seed=5)["daily_bps"], quick["daily_bps"])


# --------------------------------------------------------------------------------------------------------------------------- ETF arbitrage
@pytest.mark.parametrize("kwargs", [{"days": 1}, {"half_life": 0}, {"entry_bps": 0.2, "exit_bps": 0.5}, {"latency": -1}, {"premium_sd_bps": 0}])
def test_an_arbitrage_that_cannot_be_defined_is_refused(kwargs):
    with pytest.raises(ValueError):
        simulate_etf_arbitrage(**kwargs)


def test_the_arbitrage_edge_decays_with_the_delay_and_reverses_once_the_premium_has_gone():
    table = latency_table(latencies=(0, 1, 2, 4, 8), seed=1)
    assert (np.diff(table["gross_bps"].to_numpy()) < 0).all()                                                    # the premium is half gone after one half-life: so is the edge
    assert (np.diff(table["mean_daily_bps"].to_numpy()) < 0).all()
    assert table["mean_daily_bps"].iloc[0] > 0 > table["mean_daily_bps"].iloc[-1]
    assert table["win_rate"].iloc[0] > 0.9 > table["win_rate"].iloc[-1]


def test_the_edge_per_trade_follows_the_premiums_decay_with_the_delay():
    free = {L: simulate_etf_arbitrage(latency=L, cost_bps=0.0, seed=2) for L in (0, 3)}
    ratio = free[3]["average_trip_bps"] / free[0]["average_trip_bps"]
    assert 0.3 < ratio < 0.9                                                                                      # three bars of a four-bar half-life leaves a bit over half of it, less what the exit rule gives back


def test_a_higher_entry_threshold_trades_less_and_earns_more_per_trade():
    low, high = simulate_etf_arbitrage(entry_bps=3.0, seed=3), simulate_etf_arbitrage(entry_bps=8.0, seed=3)
    assert high["round_trips"] < low["round_trips"] and high["average_trip_bps"] > low["average_trip_bps"]
    assert (simulate_etf_arbitrage(seed=3)["position"][:, -1] == 0).all()


# --------------------------------------------------------------------------------------------------------------------------- auto market making
def test_inventory_shaded_quotes_carry_less_inventory_for_a_similar_profit_than_symmetric_quotes():
    table = auto_market_making(n_paths=400, steps=300, seed=1)
    assert list(table.index) == ["as", "symmetric"]
    assert table.loc["as", "mean_squared_inventory"] < 0.5 * table.loc["symmetric", "mean_squared_inventory"]
    assert table.loc["as", "sharpe"] > table.loc["symmetric", "sharpe"] and table.loc["as", "std_wealth"] < table.loc["symmetric", "std_wealth"]
    assert table.loc["as", "mean_wealth"] > 0.8 * table.loc["symmetric", "mean_wealth"]


# --------------------------------------------------------------------------------------------------------------------------- rebate / liquidity trading
@pytest.mark.parametrize("kwargs", [{"strategy": "x"}, {"persistence": 1.0}, {"latency": -1}, {"paths": 1}, {"steps": 5}])
def test_a_rebate_simulation_that_cannot_run_is_refused(kwargs):
    with pytest.raises(ValueError):
        simulate_rebate_trading(**kwargs)


def _paired(a, b):
    d = a["wealth"] - b["wealth"]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def test_the_pressure_aware_maker_beats_the_naive_one_that_pays_the_adverse_selection_tax():
    naive, aware = simulate_rebate_trading("naive", seed=1, paths=300, steps=3000), simulate_rebate_trading("aware", seed=1, paths=300, steps=3000)
    diff, se = _paired(aware, naive)
    assert diff > 5 * se and aware["price_pnl"] > naive["price_pnl"] and naive["price_pnl"] < 0 < aware["price_pnl"]
    assert aware["fills"] < naive["fills"]                                                                       # it earns less rebate and spread: it skips the fills that lose
    assert aware["rebate"] < naive["rebate"] and aware["spread"] < naive["spread"]


def test_the_profit_parts_add_up_and_the_rebate_is_the_rate_times_the_fills():
    r = simulate_rebate_trading("naive", seed=2, paths=100, steps=1500, rebate=0.25)
    assert r["rebate"] == pytest.approx(0.25 * r["fills"]) and r["spread"] == pytest.approx(1.0 * r["fills"])
    assert r["mean_wealth"] == pytest.approx(r["rebate"] + r["spread"] + r["adverse_selection"] + r["inventory_pnl"])
    assert r["price_pnl"] == pytest.approx(r["adverse_selection"] + r["inventory_pnl"])


def test_the_awareness_edge_fades_with_the_delay_and_vanishes_when_the_flow_is_stale():
    table = pressure_table(latencies=(0, 2, 8, 24), seed=3, paths=250, steps=2500)
    aware = table["mean_wealth"].iloc[1:].to_numpy()
    assert (np.diff(aware[:3]) < 0).all()                                                                        # a staler view of the flow is worth less, until there is nothing left to see (after that the curve is flat, up to noise)
    assert table["mean_wealth"].iloc[1] > table["mean_wealth"]["naive"] + 5 * table["se_wealth"]["naive"]
    assert abs(aware[-1] - table["mean_wealth"]["naive"]) < 0.03 * abs(table["mean_wealth"]["naive"])             # after 24 steps of a persistence-0.85 flow nothing is left: it quotes like the naive maker


def test_when_flow_does_not_move_prices_there_is_nothing_to_avoid_and_the_aware_maker_just_trades_less():
    naive = simulate_rebate_trading("naive", impact=0.0, seed=4, paths=250, steps=2500)
    aware = simulate_rebate_trading("aware", impact=0.0, seed=4, paths=250, steps=2500)
    assert aware["mean_wealth"] < naive["mean_wealth"] and aware["fills"] < naive["fills"]


def test_inventory_limits_are_respected_and_runs_are_reproducible():
    r = simulate_rebate_trading("naive", seed=5, paths=60, steps=2000, max_inventory=5.0)
    assert r["mean_abs_inventory"] <= 6.0
    a = simulate_rebate_trading("aware", seed=6, paths=50, steps=1000)
    assert a["wealth"] == pytest.approx(simulate_rebate_trading("aware", seed=6, paths=50, steps=1000)["wealth"])
