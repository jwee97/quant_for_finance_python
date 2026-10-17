"""The Ornstein-Uhlenbeck threshold pairs rule on pairs whose truth is known: a cointegrated pair it should trade at about the frequency theory predicts and profitably, two random walks it should
mostly leave alone, a broken relationship it should walk away from, and the properties every strategy here must have (causal, hedged, registered, declared as needing two tickers)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, bundle_from_prices, load_library
from src.probability import ou

load_library()
N = 3000
IDX = pd.bdate_range("2012-01-02", periods=N)
THETA, SIGMA, BETA = 40.0, 0.12, 1.3                       # half-life about 4.4 days; the spread's stationary sd is 1.3%


def _bundle(seed=0, break_at=None, n=N, theta=THETA, sigma=SIGMA):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.01, n))
    spread = ou.simulate_ou(0.0, theta, 0.0, sigma, n / 252, n, 1, seed=seed + 1)[0][1:]
    if break_at is not None:
        spread = spread + np.where(np.arange(n) >= break_at, 0.20, 0.0)                    # the relationship breaks: the spread jumps by about fifteen standard deviations
    other = np.cumsum(rng.normal(0, 0.01, n))
    prices = pd.DataFrame({"L": np.exp(3.0 + BETA * common + spread), "R": np.exp(4.0 + common), "X": np.exp(4.0 + other)}, index=IDX[:n])
    return bundle_from_prices(prices, asset_class={"L": "equity", "R": "equity", "X": "equity"}, name="synthetic")


def _model(**kw):
    return MODELS.create("ou_threshold_pairs", **{"pairs": (("L", "R"),), **kw})


def _spread_pnl(pos, hedge, prices, cost_bps=0.0):
    sp = np.log(prices["L"].to_numpy()) - np.nanmean(hedge) * np.log(prices["R"].to_numpy())
    gross = pos[:-1] * np.diff(sp)
    trades = np.abs(np.diff(pos))
    return gross - trades * 2.0 * (1 + abs(np.nanmean(hedge))) * cost_bps / 1e4 / 2.0


# ----------------------------------------------------------------------------------------------------------------------- the cointegrated pair
def test_it_trades_a_cointegrated_pair_with_about_the_right_hedge_ratio_and_profits():
    bundle = _bundle()
    m = _model()
    prices = bundle.prices
    pos, hedge = m.pair_positions(np.log(prices["L"].to_numpy()), np.log(prices["R"].to_numpy()))
    assert np.nanmean(hedge) == pytest.approx(BETA, abs=0.03) and np.all(pos[:252] == 0), "nothing is traded until a window of data exists"
    assert np.mean(pos[252:] != 0) > 0.9, "buying at one level and holding to the other keeps the rule in the market nearly all the time"
    years = (N - 252) / 252
    pnl = _spread_pnl(pos, hedge, prices, cost_bps=m.cost_bps)[252:]
    assert pnl.sum() / years > 0.1 and pnl.mean() / pnl.std() * np.sqrt(252) > 1.0, "after costs it earns a good Sharpe ratio on a pair that really reverts"


def test_the_trading_frequency_is_a_known_fraction_of_the_continuous_time_optimum_because_a_daily_close_misses_crossings():
    """Bertram's levels are for a spread watched at every instant. A daily close misses crossings between closes, so the rule trades only 60 to 80 per cent as often as theory says (the
    shortfall shrinks only with the square root of the monitoring interval), and a rolling estimate of the reversion speed costs a little more. The number is worth knowing: it is the
    difference between the trades a backtest on closes shows and those the continuous theory promises."""
    cost_bps, years = 5.0, (N - 252) / 252
    ratios = []
    for theta, sigma in ((40.0, 0.12), (8.0, 0.05)):
        m = _model(cost_bps=cost_bps, max_half_life=120.0)
        got = []
        for seed in (1, 2, 3):
            prices = _bundle(seed=seed, theta=theta, sigma=sigma).prices
            pos, _ = m.pair_positions(np.log(prices["L"].to_numpy()), np.log(prices["R"].to_numpy()))
            got.append(np.sum(np.abs(np.diff(pos[252:])) > 0) / years)
        levels = ou.optimal_thresholds(theta, 0.0, sigma, 2.0 * (1 + BETA) * cost_bps / 1e4)
        ratios.append(float(np.mean(got)) / (2.0 / levels["cycle"]))
    assert all(0.45 < r < 0.85 for r in ratios), ratios


def test_it_leaves_two_independent_random_walks_almost_alone():
    bundle = _bundle(seed=3)
    m = MODELS.create("ou_threshold_pairs", pairs=(("L", "X"),))
    pos, _ = m.pair_positions(np.log(bundle.prices["L"].to_numpy()), np.log(bundle.prices["X"].to_numpy()))
    real, _ = _model().pair_positions(np.log(bundle.prices["L"].to_numpy()), np.log(bundle.prices["R"].to_numpy()))
    assert np.mean(pos != 0) < 0.15 < 0.6 < np.mean(real != 0), "the Dickey-Fuller screen stops it trading the spurious reversion of a regression of one random walk on another"
    relaxed = MODELS.create("ou_threshold_pairs", pairs=(("L", "X"),), df_critical=-0.5)
    spurious, _ = relaxed.pair_positions(np.log(bundle.prices["L"].to_numpy()), np.log(bundle.prices["X"].to_numpy()))
    assert np.mean(spurious != 0) > 2 * np.mean(pos != 0), "without it the rule would trade noise"


def test_the_exit_level_trades_off_holding_time_against_the_number_of_trades():
    prices = _bundle().prices
    y, x = np.log(prices["L"].to_numpy()), np.log(prices["R"].to_numpy())
    flip = _model(exit_scale=1.0).pair_positions(y, x)[0]
    mean_exit = _model(exit_scale=0.0).pair_positions(y, x)[0]
    early = _model(exit_scale=-0.5).pair_positions(y, x)[0]
    share = lambda p: np.mean(p[252:] != 0)                          # noqa: E731
    changes = lambda p: np.sum(np.abs(np.diff(p[252:])) > 0)         # noqa: E731
    assert share(flip) > share(mean_exit) > share(early), "exiting earlier means less time in the market"
    assert changes(flip) < changes(mean_exit) < changes(early), "and more trades"
    assert set(np.unique(flip)) <= {-1.0, 0.0, 1.0}


def test_a_broken_relationship_is_closed_out_and_not_re_entered_until_the_spread_comes_back_inside():
    bundle = _bundle(seed=2, break_at=2000)
    prices = bundle.prices
    y, x = np.log(prices["L"].to_numpy()), np.log(prices["R"].to_numpy())
    pos, _ = _model(stop_sd=4.0).pair_positions(y, x)
    # the jump lands in a refit window that has not seen it: the position must be closed within a few days and the pair stay out until the next refit absorbs the new level
    assert np.all(pos[2003:2021] == 0), "closed by the stop within days, and flat for the rest of the window"
    assert np.mean(pos[2300:] != 0) > 0.8, "once the windows have seen the new level the pair trades again"
    no_stop, _ = _model(stop_sd=50.0).pair_positions(y, x)
    assert np.sum(no_stop[2000:2021] != 0) >= np.sum(pos[2000:2021] != 0), "without the stop the rule stays in a position that is losing"


# ------------------------------------------------------------------------------------------------------------------------ the framework
def test_the_weights_are_a_hedged_pair_trade_scaled_to_unit_gross_exposure_and_causal():
    bundle = _bundle(seed=4)
    m = _model()
    w = m.weights(bundle)
    assert list(w.columns) == ["L", "R", "X"] and (w["X"] == 0).all()
    held = w[(w["L"] != 0)]
    assert len(held) > 1000
    assert np.allclose(held["L"].abs() + held["R"].abs(), 1.0), "one pair: unit gross exposure while it is held"
    ratio = -(held["R"] / held["L"])
    assert ratio.mean() == pytest.approx(BETA, abs=0.05) and (np.sign(held["L"]) == -np.sign(held["R"])).all(), "long one leg, short the other, in the ratio of the hedge"
    cut = 1500
    truncated = bundle_from_prices(bundle.prices.iloc[:cut], asset_class={"L": "equity", "R": "equity", "X": "equity"}, name="cut")
    wt = _model().weights(truncated)
    assert np.allclose(w.iloc[:cut].to_numpy(), wt.to_numpy()), "the weights through a date do not depend on anything after it"
    sc = m.score(bundle)
    assert sc.shape == w.shape


def test_it_is_registered_declared_as_a_two_ticker_strategy_and_checks_its_arguments():
    entry = next(e for e in MODELS.entries() if e.name == "ou_threshold_pairs")
    assert entry.family == "stat-arb"
    m = MODELS.create("ou_threshold_pairs")
    assert m.required_assets() == 2 and m.structured and m.position_mode == "time_series" and m.params()["cost_bps"] == 5.0
    one = bundle_from_prices(pd.DataFrame({"ONE": np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 800)))}, index=IDX[:800]), asset_class={"ONE": "equity"}, name="one")
    assert (m.weights(one) == 0).all().all(), "no pair is available on one ticker: an empty book, not an error"
    for bad in ({"window": 30}, {"cost_bps": -1.0}, {"min_half_life": 50.0}, {"exit_scale": 2.0}, {"stop_sd": 0.5}, {"df_critical": 0.5}, {"refit_every": 0}):
        with pytest.raises(ValueError):
            MODELS.create("ou_threshold_pairs", **bad)


def test_a_gap_in_one_price_does_not_break_the_rule():
    bundle = _bundle(seed=5)
    prices = bundle.prices.copy()
    prices.iloc[1000:1004, 0] = np.nan
    gapped = bundle_from_prices(prices, asset_class={"L": "equity", "R": "equity", "X": "equity"}, name="gap")
    w = _model().weights(gapped)
    assert np.isfinite(w.to_numpy()).all()
