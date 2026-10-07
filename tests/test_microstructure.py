"""Microstructure: estimators against simulations with known parameters, order-book invariants, the Avellaneda-Stoikov formulas and their effect, and the Almgren-Chriss closed forms."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.microstructure import execution as ex, lob, market_making as mm, order_flow as of

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------------------------ order flow
def test_trade_classification_rules():
    p = pd.Series([100.0, 100.5, 100.5, 100.2, 100.2, 100.2, 100.4])
    ticks = of.tick_rule(p)
    assert np.isnan(ticks.iloc[0]) and ticks.tolist()[1:] == [1.0, 1.0, -1.0, -1.0, -1.0, 1.0]       # an unchanged price repeats the previous sign
    bid, ask = pd.Series([99.9, 100.4, 100.1, 100.2]), pd.Series([100.1, 100.6, 100.3, 100.4])
    sign = of.lee_ready(pd.Series([100.1, 100.5, 100.0, 100.3]), bid, ask)
    assert sign.tolist() == [1.0, 1.0, -1.0, 1.0]                                                    # above the midpoint buy, below sell; the last trade is exactly at the 100.3 midpoint and the tick rule says up (100.0 -> 100.3)
    mid_trade = of.lee_ready(pd.Series([100.0, 100.3, 100.3]), pd.Series([99.9, 100.2, 100.2]), pd.Series([100.1, 100.4, 100.4]))
    assert mid_trade.iloc[1] == 1.0 and mid_trade.iloc[2] == 1.0                                     # midpoint trades fall back to the tick rule, which repeats the sign when the price is unchanged


def test_ofi_event_formula_on_hand_computed_cases():
    pb, qb = pd.Series([100.0, 100.0, 101.0, 100.0]), pd.Series([10.0, 15.0, 7.0, 4.0])
    pa, qa = pd.Series([101.0, 101.0, 102.0, 102.0]), pd.Series([8.0, 8.0, 9.0, 5.0])
    e = of.ofi(pb, qb, pa, qa)
    # n=1: bid unchanged (+15 - 10), ask unchanged (-8 + 8) = +5; n=2: bid up (+7), ask up (+qa_{n-1} = 8) = +15; n=3: bid down (-qb_{n-1} = -7), ask same (-5 + 9) = -3
    assert e.tolist() == [0.0, 5.0, 15.0, -3.0]


def test_ofi_price_impact_recovers_a_planted_slope():
    rng = np.random.default_rng(1)
    n = 40000
    flow = pd.Series(rng.normal(0, 5, n))
    mid = (0.002 * flow + rng.normal(0, 0.003, n)).cumsum()
    r = of.ofi_price_impact(flow, mid, 100, pd.Series(np.full(n, 1.0)))
    assert r["beta"] == pytest.approx(0.002, rel=0.15) and r["t"] > 10 and 0 < r["r2"] < 1 and r["n_blocks"] == 400


def test_kyle_lambda_recovers_planted_impact():
    rng = np.random.default_rng(2)
    n = 20000
    sv = pd.Series(rng.normal(0, 100, n))
    dp = 0.01 * sv + pd.Series(rng.normal(0, 0.5, n))
    out = of.kyle_lambda(dp, sv)
    assert out["lambda"] == pytest.approx(0.01, abs=0.0015) and out["t"] > 20


def test_amihud_roll_and_corwin_schultz_estimators():
    rng = np.random.default_rng(0)
    r, dv = pd.Series([0.01, -0.02, 0.0, 0.03]), pd.Series([1e6, 2e6, 1e6, 3e6])
    assert of.amihud(r, dv) == pytest.approx(np.mean([0.01 / 1e6, 0.02 / 2e6, np.nan, 0.03 / 3e6][:2] + [0.0, 0.01 / 1e6]))
    assert of.amihud(r, dv, window=2).iloc[-1] == pytest.approx(np.mean([0.0, 0.01 / 1e6]))
    n = 60000
    eff = np.cumsum(rng.normal(0, 0.001, n))
    q = rng.choice([-1, 1], n)
    for s in (0.0, 0.01):
        prices = pd.Series(100 * np.exp(eff) + s * 100 / 2 * q)
        assert of.roll_spread(prices) / 100 == pytest.approx(s, abs=0.0006)
    path = np.exp(np.cumsum(rng.normal(0, 0.015 / np.sqrt(200), (4000, 200)), axis=1))
    est = {}
    for s in (0.0, 0.004, 0.01):
        est[s] = of.corwin_schultz(pd.Series(path.max(axis=1) * (1 + s / 2)), pd.Series(path.min(axis=1) * (1 - s / 2))).mean()
    assert est[0.0] < est[0.004] < est[0.01] and est[0.01] - est[0.0] > 0.003


def test_effective_realized_spread_separate_adverse_selection():
    rng = np.random.default_rng(3)
    n = 40000
    half, impact = 0.05, 0.03
    sign = pd.Series(rng.choice([-1.0, 1.0], n))
    eps = rng.normal(0, 0.01, n)
    mid = pd.Series(100.0 + np.cumsum(eps + impact * np.r_[0.0, sign.to_numpy()[:-1]]))        # each trade moves the mid permanently by `impact` in its own direction
    trade = mid + half * sign
    out = of.realized_spread(trade, mid, sign, horizon=1)
    assert out["effective"] == pytest.approx(2 * half, abs=0.003)
    assert out["realized"] == pytest.approx(2 * (half - impact), abs=0.003) and out["adverse_selection"] == pytest.approx(2 * impact, abs=0.003)
    assert out["adverse_selection"] == pytest.approx(out["effective"] - out["realized"], abs=1e-12)
    assert of.effective_spread(trade, mid, sign, relative=False).mean() == pytest.approx(2 * half)
    assert of.effective_spread(trade, mid, sign).mean() == pytest.approx(float((2 * half / mid).mean()))                     # relative to the contemporaneous midpoint
    quiet = of.realized_spread(mid + half * sign, pd.Series(100.0 + np.cumsum(eps)), sign, horizon=1)
    assert quiet["adverse_selection"] == pytest.approx(0.0, abs=0.003)                          # no information in the flow: the maker keeps the whole spread


def test_vpin_is_low_for_balanced_flow_and_high_for_one_sided_flow():
    rng = np.random.default_rng(4)
    n = 5000
    idx = pd.date_range("2022-01-03", periods=n, freq="min")
    vol = pd.Series(rng.uniform(80, 120, n), index=idx)
    balanced = pd.Series(100 + rng.normal(0, 0.05, n).cumsum() * 0, index=idx) + pd.Series(rng.normal(0, 0.05, n), index=idx).cumsum() * 0
    noise_price = pd.Series(100 + rng.normal(0, 0.05, n), index=idx)                  # mean-reverting noise: up and down moves are equally likely and uncorrelated with direction
    low = of.vpin(noise_price, vol, bucket_volume=2000, n_buckets=20).dropna()
    trending = pd.Series(100 + np.arange(n) * 0.05 + rng.normal(0, 0.005, n), index=idx)     # every bar rises: bulk classification calls it all buying
    high = of.vpin(trending, vol, bucket_volume=2000, n_buckets=20).dropna()
    assert high.mean() > 0.8 and low.mean() < 0.5 and high.mean() > low.mean() + 0.3
    assert len(of.vpin(noise_price, vol, bucket_volume=2000, n_buckets=1).dropna()) == int(vol.sum() // 2000) and balanced is not None


# ----------------------------------------------------------------------------------------------------------------------- LOB
@pytest.fixture(scope="module")
def book():
    return lob.simulate_lob(30000, seed=1)


def test_lob_invariants_and_reproducibility(book):
    assert (book["ask"] > book["bid"]).all() and (book["spread"] >= 1).all() and (book["bid_size"] > 0).all() and (book["ask_size"] > 0).all()
    assert book["time"].is_monotonic_increasing and set(book["event"]) <= {"limit_buy", "limit_sell", "cancel", "market_buy", "market_sell", "none"}
    again = lob.simulate_lob(2000, seed=1)
    pd.testing.assert_frame_equal(again, book.iloc[:2000].reset_index(drop=True))
    assert not lob.simulate_lob(2000, seed=2)["mid"].equals(again["mid"])
    trades = book[book["sign"] != 0]
    prev = book.shift(1).loc[trades.index]
    assert (trades.loc[trades["sign"] == 1, "trade_price"] == prev.loc[trades["sign"] == 1, "ask"]).mean() > 0.95        # a market buy executes at the standing best ask
    assert (trades.loc[trades["sign"] == -1, "trade_price"] == prev.loc[trades["sign"] == -1, "bid"]).mean() > 0.95


def test_lob_event_mix_and_stationary_depth(book):
    freq = book["event"].value_counts(normalize=True)
    assert freq["limit_buy"] == pytest.approx(freq["limit_sell"], rel=0.1) and freq["market_buy"] == pytest.approx(freq["market_sell"], rel=0.2)
    assert freq["cancel"] > freq["market_buy"] and book["spread"].mean() < 2.5
    half = len(book) // 2
    assert book["bid_size"].iloc[:half].mean() == pytest.approx(book["bid_size"].iloc[half:].mean(), rel=0.3)


def test_order_flow_imbalance_explains_price_changes_in_the_simulated_book(book):
    o = of.ofi(book["bid"], book["bid_size"], book["ask"], book["ask_size"])
    res = of.ofi_price_impact(o, book["mid"], 200)
    assert res["beta"] > 0 and res["t"] > 2
    signed = book["sign"] * 1.0
    pooled = of.kyle_lambda(book["mid"].diff().fillna(0.0), signed)
    assert np.isfinite(pooled["lambda"])


# --------------------------------------------------------------------------------------------------------------- market making
def test_avellaneda_stoikov_formulas():
    gamma, sigma, k = 0.1, 2.0, 1.5
    assert mm.as_reservation_price(100.0, 0, gamma, sigma, 1.0) == 100.0
    assert mm.as_reservation_price(100.0, 3, gamma, sigma, 1.0) == pytest.approx(100.0 - 3 * gamma * sigma ** 2)
    assert mm.as_reservation_price(100.0, -3, gamma, sigma, 1.0) > 100.0                      # short: quote higher to buy back
    assert mm.as_optimal_spread(gamma, sigma, k, 1.0) == pytest.approx(gamma * sigma ** 2 + 2 / gamma * np.log(1 + gamma / k))
    assert mm.as_optimal_spread(1e-6, sigma, k, 0.0) == pytest.approx(2.0 / k, rel=1e-4)       # gamma -> 0: the spread is 2/k
    assert mm.as_optimal_spread(gamma, 3.0, k, 1.0) > mm.as_optimal_spread(gamma, 2.0, k, 1.0) and mm.as_optimal_spread(gamma, sigma, k, 1.0) > mm.as_optimal_spread(gamma, sigma, k, 0.1)
    bid, ask = mm.as_quotes(100.0, 3, gamma, sigma, k, 1.0)
    assert bid < ask and 0.5 * (bid + ask) == pytest.approx(mm.as_reservation_price(100.0, 3, gamma, sigma, 1.0)) and ask - bid == pytest.approx(mm.as_optimal_spread(gamma, sigma, k, 1.0))


def test_inventory_shading_cuts_risk_for_similar_profit():
    table = mm.compare_strategies(n_paths=3000, seed=1)
    assert table.loc["as", "std_wealth"] < 0.7 * table.loc["symmetric", "std_wealth"]
    assert table.loc["as", "std_inventory"] < 0.5 * table.loc["symmetric", "std_inventory"] and table.loc["as", "mean_squared_inventory"] < table.loc["symmetric", "mean_squared_inventory"]
    assert table.loc["as", "mean_wealth"] > 0.8 * table.loc["symmetric", "mean_wealth"] and table.loc["as", "sharpe"] > table.loc["symmetric", "sharpe"]


def test_pnl_decomposition_adverse_selection_and_inventory_limits():
    clean = mm.simulate_market_making("symmetric", n_paths=3000, informed_fraction=0.0, seed=2)
    toxic = mm.simulate_market_making("symmetric", n_paths=3000, informed_fraction=0.3, seed=2)
    for r in (clean, toxic):
        assert np.allclose(r["wealth"], r["spread_capture"] + r["adverse_selection"] + r["inventory_pnl"])
    assert clean["adverse_selection"].sum() == 0 and toxic["adverse_selection"].mean() < -5 and toxic["mean_wealth"] < clean["mean_wealth"] - 5
    assert toxic["spread_capture"].mean() == pytest.approx(clean["spread_capture"].mean(), rel=0.01)       # the same fills, the same spread: informed flow hurts after the fill
    capped = mm.simulate_market_making("symmetric", n_paths=1000, max_inventory=3, seed=3)
    assert np.abs(capped["inventory"]).max() <= 3
    again = mm.simulate_market_making("as", n_paths=200, seed=7)
    assert np.array_equal(again["wealth"], mm.simulate_market_making("as", n_paths=200, seed=7)["wealth"])
    with pytest.raises(ValueError):
        mm.simulate_market_making("x")


# -------------------------------------------------------------------------------------------------------------------- execution
def test_almgren_chriss_shapes_cost_and_risk():
    X, T, n, sigma, eta = 1e6, 1.0, 20, 0.3, 2.5e-6
    twap = ex.almgren_chriss_schedule(X, T, n, sigma, eta, 0.0)
    assert np.allclose(twap["holdings"].diff().dropna(), -X / n) and twap["holdings"].iloc[0] == X and twap["holdings"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    urgent = ex.almgren_chriss_schedule(X, T, n, sigma, eta, 1e-3)
    assert urgent["holdings"].iloc[10] < twap["holdings"].iloc[10] and urgent["holdings"].is_monotonic_decreasing and urgent["trade"].iloc[1] > urgent["trade"].iloc[-1]
    assert urgent["trade"].sum() == pytest.approx(X)
    f = ex.efficient_frontier(X, T, n, sigma, eta)
    assert f["expected_cost"].is_monotonic_increasing and f["std_cost"].is_monotonic_decreasing
    cost_twap = ex.expected_cost_and_variance(twap, sigma, eta)
    assert cost_twap["expected_cost"] == pytest.approx(eta * X ** 2 / T)                    # eta sum n_k^2 / tau with n_k = X/N: eta X^2 / T
    perm = ex.expected_cost_and_variance(twap, sigma, eta, gamma=1e-7)
    assert perm["expected_cost"] == pytest.approx(cost_twap["expected_cost"] + 0.5 * 1e-7 * X ** 2)


def test_simulated_shortfall_matches_the_closed_form_and_vwap_follows_volume():
    X, T, n, sigma, eta = 1e5, 1.0, 10, 0.3, 1e-5
    sched = ex.almgren_chriss_schedule(X, T, n, sigma, eta, 0.0)
    out = ex.implementation_shortfall(sched["trade"].to_numpy()[1:], sigma, eta, 0.0, T / n, n_paths=20000, seed=1)
    assert out["mean"] == pytest.approx(ex.expected_cost_and_variance(sched, sigma, eta)["expected_cost"], rel=0.05)
    assert out["std"] > 0
    prof = np.array([5.0, 1.0, 1.0, 1.0, 2.0])
    v = ex.vwap_schedule(1000.0, prof)
    assert v.sum() == pytest.approx(1000.0) and v[0] == pytest.approx(500.0) and ex.twap_schedule(1000.0, 4).tolist() == [250.0] * 4
