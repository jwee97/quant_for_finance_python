"""What the option quotes imply, against numbers worked out independently: the implied distribution of a flat-volatility chain is the lognormal, the model-free variance follows the Cboe formula strike by
strike, the straddle prices the move the Black-Scholes formula says it does, max pain and the gamma exposure follow their definitions, and the cone is the percentiles of rolling realised volatility."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.assets import options_analytics as oa
from src.derivatives.pricing import bsm_price

AS_OF = pd.Timestamp("2026-10-01")


def flat_chain(spot=100.0, sigma=0.20, rate=0.03, dividend=0.01, days=(20, 40), step=0.5, width=0.6, half_spread=0.01, open_interest=100.0) -> pd.DataFrame:
    """Calls and puts at every strike on a grid, priced by Black-Scholes at ONE volatility, with a bid and ask a fixed share (``half_spread``) either side of the mid, so that the mid is the exact price
    even for an option worth almost nothing."""
    rows = []
    for d in days:
        T = d / 365.0
        for strike in np.arange(spot * (1.0 - width), spot * (1.0 + width) + 1e-9, step):
            for right in ("C", "P"):
                mid = float(bsm_price(spot, strike, T, rate, dividend, sigma, right == "C"))
                rows.append({"date": AS_OF, "expiry": AS_OF + pd.Timedelta(days=d), "strike": float(strike), "right": right, "bid": mid * (1.0 - half_spread), "ask": mid * (1.0 + half_spread), "underlying": spot,
                             "rate": rate, "dividend": dividend, "volume": 10.0, "open_interest": open_interest, "iv": sigma})
    return oa.prepare_chain(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def flat():
    return flat_chain()


# ------------------------------------------------------------------------------------------------------------------ the implied distribution
def test_the_distribution_of_a_flat_smile_is_the_lognormal(flat):
    d = oa.implied_distribution(flat, flat["expiry"].min())
    F, sigma, T = d["forward"], 0.20, d["dte"] / 365.0
    price, density = np.array(d["price"]), np.array(d["density"])
    exact = norm.pdf(np.log(price / F), loc=-0.5 * sigma ** 2 * T, scale=sigma * math.sqrt(T)) / price
    centre = np.abs(np.log(price / F)) < 2.5 * sigma * math.sqrt(T)
    assert np.max(np.abs(density[centre] / exact[centre] - 1.0)) < 2e-3
    assert d["area"] > 0.999 and d["moments"]["mean_price_over_forward"] == pytest.approx(1.0, abs=1e-3)
    assert d["moments"]["sd_log_return"] == pytest.approx(sigma * math.sqrt(T), rel=2e-3)
    assert d["moments"]["skewness"] == pytest.approx(0.0, abs=0.02) and d["moments"]["excess_kurtosis"] == pytest.approx(0.0, abs=0.05)
    assert d["fit"]["arbitrage_free"] and d["fit"]["negative_share"] == 0.0 and d["fit"]["max_error_vol"] < 5e-4


def test_quantiles_and_tail_probabilities_of_a_flat_smile_are_the_lognormal_ones(flat):
    d = oa.implied_distribution(flat, flat["expiry"].min())
    F, sigma, T = d["forward"], 0.20, d["dte"] / 365.0
    sd = sigma * math.sqrt(T)
    for p, value in d["quantiles"].items():
        exact = F * math.exp(-0.5 * sigma ** 2 * T + sd * norm.ppf(float(p)))
        assert value == pytest.approx(exact, rel=2e-3), p
    cdf = np.array(d["cdf"])
    assert (np.diff(cdf) >= -1e-12).all() and cdf[0] < 0.001 < 0.999 < cdf[-1] + 1e-9
    spot = d["spot"]
    for t in d["tails"]:
        k = math.log(t["level"] / F)
        below = norm.cdf((k + 0.5 * sigma ** 2 * T) / sd)
        exact = below if t["side"] == "fall" else 1.0 - below
        assert t["lognormal"] == pytest.approx(exact, abs=1e-12) and t["implied"] == pytest.approx(exact, abs=1.5e-3), (t["side"], t["move"])
        assert t["level"] == pytest.approx(spot * (1.0 - t["move"] if t["side"] == "fall" else 1.0 + t["move"]))


def test_the_straddle_prices_the_move_black_scholes_says_it_does(flat):
    d = oa.implied_distribution(flat, flat["expiry"].min())
    T, F, r = d["dte"] / 365.0, d["forward"], 0.03
    strike = float(flat.loc[(flat["expiry"] == flat["expiry"].min()) & ((flat["strike"] - F).abs() == (flat["strike"] - F).abs().min()), "strike"].iloc[0])
    exact = bsm_price(100.0, strike, T, r, 0.01, 0.20, True) + bsm_price(100.0, strike, T, r, 0.01, 0.20, False)
    assert d["expected_move"]["straddle"] == pytest.approx(float(exact), abs=1e-9) and d["expected_move"]["straddle_pct"] == pytest.approx(float(exact) / 100.0)
    assert d["expected_move"]["straddle"] / d["spot"] == pytest.approx(math.sqrt(2.0 / math.pi) * 0.20 * math.sqrt(T), rel=0.03)      # a straddle is worth about 0.8 standard deviations of the move
    rows = oa.expected_moves(flat)
    assert [r["dte"] for r in rows] == [20, 40] and rows[0]["straddle"] == pytest.approx(float(exact), abs=1e-9) and rows[1]["one_sd_pct"] == pytest.approx(0.20 * math.sqrt(40 / 365.0))
    assert oa.expected_moves(flat, min_dte=30)[0]["dte"] == 40


def test_a_skewed_smile_puts_more_probability_on_a_fall_than_the_lognormal_does():
    chain = oa.prepare_chain(oa.build_model_chain(500.0, AS_OF, 0.15, 0.18, 0.04, 0.013, 0.15, "SPY"))
    d = oa.implied_distribution(chain, oa.nearest_expiry_to(chain, 30))
    assert d["moments"]["skewness"] < -0.5 and d["moments"]["excess_kurtosis"] > 0.5
    fall = next(t for t in d["tails"] if t["side"] == "fall" and t["move"] == 0.10)
    rise = next(t for t in d["tails"] if t["side"] == "rise" and t["move"] == 0.10)
    assert fall["implied"] > 2.0 * fall["lognormal"] and rise["implied"] < rise["lognormal"]
    assert d["fit"]["max_error_vol"] < 2e-3 and d["fit"]["quotes"] >= 40 and d["moments"]["mean_price_over_forward"] == pytest.approx(1.0, abs=3e-3)
    assert d["expected_move"]["band_low"] < d["spot"] < d["expected_move"]["band_high"]


def test_too_few_quotes_for_a_distribution_is_refused(flat):
    few = flat[(flat["expiry"] == flat["expiry"].min()) & (flat["strike"].between(99.0, 101.0))]
    with pytest.raises(ValueError, match="at least 6"):
        oa.implied_distribution(few, few["expiry"].iloc[0])
    with pytest.raises(ValueError, match="not on the chain"):
        oa.implied_distribution(flat, pd.Timestamp("2030-01-01"))


# ------------------------------------------------------------------------------------------------------------------ model-free variance
def toy_expiry(put_bids=None, F=100.0, T=0.5, r=0.0):
    """Five strikes, 5 apart, with chosen mids (a half-point spread on each), forward ``F``: for working the Cboe formula by hand."""
    mid = {("P", 90.0): 0.5, ("P", 95.0): 1.5, ("P", 100.0): 4.0, ("C", 100.0): 4.0, ("C", 105.0): 1.8, ("C", 110.0): 0.6, ("C", 95.0): 6.0, ("C", 90.0): 11.0, ("P", 105.0): 7.0, ("P", 110.0): 12.0}
    rows = []
    for (right, strike), m in mid.items():
        rows.append({"date": AS_OF, "expiry": AS_OF + pd.Timedelta(days=round(T * 365)), "strike": strike, "right": right, "bid": m - 0.1, "ask": m + 0.1, "underlying": F, "rate": r, "dividend": 0.0,
                     "volume": 1.0, "open_interest": 1.0, "iv": 0.2, "mid": m, "T": round(T * 365) / 365.0, "F": F, "k": math.log(strike / F)})
    chain = pd.DataFrame(rows)
    for (right, strike), bid in (put_bids or {}).items():
        chain.loc[(chain["right"] == right) & (chain["strike"] == strike), "bid"] = bid
    return chain


def cboe_by_hand(strikes, prices, F, K0, T, r):
    """The Cboe formula as printed, with an explicit loop: dK is half the gap between the neighbours, the whole gap at the ends."""
    total = 0.0
    for i, (k, q) in enumerate(zip(strikes, prices)):
        dk = (strikes[1] - strikes[0]) if i == 0 else (strikes[-1] - strikes[-2]) if i == len(strikes) - 1 else (strikes[i + 1] - strikes[i - 1]) / 2.0
        total += dk / k ** 2 * math.exp(r * T) * q
    return 2.0 / T * total - (F / K0 - 1.0) ** 2 / T


def test_model_free_variance_is_the_cboe_formula_worked_by_hand():
    chain = toy_expiry()
    got = oa.model_free_variance(chain, chain["expiry"].iloc[0])
    T = chain["T"].iloc[0]
    expected = cboe_by_hand([90.0, 95.0, 100.0, 105.0, 110.0], [0.5, 1.5, 4.0, 1.8, 0.6], 100.0, 100.0, T, 0.0)          # puts below K0, the average of call and put at K0, calls above
    assert got["variance"] == pytest.approx(expected, rel=1e-12) and got["vol"] == pytest.approx(math.sqrt(expected)) and got["k0"] == 100.0 and got["strikes"] == 5
    assert (got["lowest"], got["highest"]) == (90.0, 110.0)


def test_the_forward_between_strikes_adds_the_correction_term():
    chain = toy_expiry(F=101.0, r=0.04)
    got = oa.model_free_variance(chain, chain["expiry"].iloc[0])
    T = chain["T"].iloc[0]
    expected = cboe_by_hand([90.0, 95.0, 100.0, 105.0, 110.0], [0.5, 1.5, 4.0, 1.8, 0.6], 101.0, 100.0, T, 0.04)
    assert got["variance"] == pytest.approx(expected, rel=1e-12) and got["k0"] == 100.0


def test_a_strike_without_a_bid_is_skipped_and_two_in_a_row_end_the_wing():
    chain = toy_expiry()
    extra = []
    for strike, bid in ((85.0, 0.0), (80.0, 0.0), (75.0, 0.9)):              # two strikes in a row with no bid, then one with a bid: it must not be used
        extra.append({**chain[(chain["right"] == "P") & (chain["strike"] == 90.0)].iloc[0].to_dict(), "strike": strike, "bid": bid, "ask": 0.3, "mid": 0.15})
    chain = pd.concat([chain, pd.DataFrame(extra)], ignore_index=True)
    got = oa.model_free_variance(chain, chain["expiry"].iloc[0])
    assert got["lowest"] == 90.0 and got["strikes"] == 5
    one_gap = toy_expiry(put_bids={("P", 95.0): 0.0})                          # a single strike without a bid is passed over and the walk goes on
    got = oa.model_free_variance(one_gap, one_gap["expiry"].iloc[0])
    assert got["strikes"] == 4 and got["lowest"] == 90.0
    T = one_gap["T"].iloc[0]
    assert got["variance"] == pytest.approx(cboe_by_hand([90.0, 100.0, 105.0, 110.0], [0.5, 4.0, 1.8, 0.6], 100.0, 100.0, T, 0.0), rel=1e-12)       # dK is half the gap between the strikes USED


def test_the_model_free_variance_of_a_flat_smile_is_its_variance(flat):
    for expiry in flat["expiry"].unique():
        got = oa.model_free_variance(flat, expiry)
        assert got["variance"] == pytest.approx(0.20 ** 2, rel=2e-3) and got["vol"] == pytest.approx(0.20, rel=1e-3)
    term = oa.model_free_term(flat, 30.0)
    assert [r["dte"] for r in term["rows"]] == [20, 40] and term["iv_30d"] == pytest.approx(0.20, rel=1e-3) and term["atm_30d"] == pytest.approx(0.20, rel=1e-3)
    assert all(abs(r["difference"]) < 5e-4 for r in term["rows"])
    assert oa.model_free_term(flat, 5.0)["iv_30d"] == pytest.approx(term["rows"][0]["vol"]) and oa.model_free_term(flat, 90.0)["iv_30d"] == pytest.approx(term["rows"][-1]["vol"])


def test_the_constant_maturity_figure_interpolates_total_variance_in_time():
    near, far = toy_expiry(T=20 / 365.0), toy_expiry(T=40 / 365.0)
    chain = pd.concat([near, far], ignore_index=True)
    rows = {r["dte"]: r for r in oa.model_free_term(chain, 30.0)["rows"]}
    w = lambda dte: rows[dte]["variance"] * dte / 365.0                                                          # noqa: E731
    expected = math.sqrt((0.5 * w(20) + 0.5 * w(40)) / (30.0 / 365.0))
    assert oa.model_free_term(chain, 30.0)["iv_30d"] == pytest.approx(expected, rel=1e-12)


def test_the_skew_makes_the_model_free_volatility_exceed_the_at_the_money_one():
    chain = oa.prepare_chain(oa.build_model_chain(500.0, AS_OF, 0.15, 0.18, 0.04, 0.013, 0.15, "SPY"))
    term = oa.model_free_term(chain, 30.0)
    assert len(term["rows"]) >= 8 and all(r["difference"] > 0 for r in term["rows"] if r["dte"] >= 7)
    assert term["iv_30d"] > term["atm_30d"] > 0.1


# ------------------------------------------------------------------------------------------------------------------ the cone
def test_the_cone_is_the_percentiles_of_rolling_realised_volatility():
    a = 0.01
    returns = np.tile([a, -a], 400)                                                              # alternating returns: the standard deviation of an even window of h is a sqrt(h / (h - 1))
    prices = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=pd.bdate_range("2020-01-01", periods=len(returns)))
    cone = oa.volatility_cone(prices, windows=(10, 21, 63))
    first = cone["rows"][0]
    assert first["days"] == 10 and first["now"] == pytest.approx(a * math.sqrt(10 / 9) * math.sqrt(252.0), rel=1e-9) and first["calendar_days"] == 14
    assert first["min"] == pytest.approx(first["max"], rel=1e-9) and first["median"] == pytest.approx(first["now"], rel=1e-9) and first["percentile"] == 1.0         # every window is the same
    assert [r["days"] for r in cone["rows"]] == [10, 21, 63] and cone["history_days"] == len(returns) - 1
    rng = np.random.default_rng(3)
    noisy = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 900))), index=pd.bdate_range("2020-01-01", periods=900))
    row = oa.volatility_cone(noisy, windows=(21,))["rows"][0]
    vols = (np.log(noisy).diff().rolling(21).std(ddof=1) * math.sqrt(252.0)).dropna()
    assert row["median"] == pytest.approx(vols.median()) and row["percentile"] == pytest.approx((vols <= vols.iloc[-1]).mean()) and row["p10"] <= row["p25"] <= row["median"] <= row["p75"] <= row["p90"]
    assert oa.volatility_cone(noisy.iloc[:50], windows=(21,))["rows"] == []                       # under 60 windows of history: left out


# ------------------------------------------------------------------------------------------------------------------ positioning
def oi_chain(contracts, spot=100.0, sigma=0.2, days=30):
    """A chain of the given ``(right, strike, open_interest)`` contracts at one expiry."""
    T = days / 365.0
    return pd.DataFrame([{"date": AS_OF, "expiry": AS_OF + pd.Timedelta(days=days), "strike": k, "right": right, "bid": 1.0, "ask": 1.1, "mid": 1.05, "underlying": spot, "rate": 0.0, "dividend": 0.0, "volume": 5.0,
                          "open_interest": oi, "iv": sigma, "T": T, "F": spot, "k": math.log(k / spot)} for right, k, oi in contracts])


def test_max_pain_is_the_settlement_that_pays_option_holders_least():
    calls = pd.Series({95.0: 300.0, 100.0: 500.0, 105.0: 900.0, 110.0: 200.0})
    puts = pd.Series({90.0: 400.0, 95.0: 700.0, 100.0: 600.0, 105.0: 100.0})
    pain = oa.max_pain(calls, puts)
    strikes = sorted(set(calls.index) | set(puts.index))
    payout = {s: sum(calls.get(k, 0) * max(s - k, 0) for k in strikes) + sum(puts.get(k, 0) * max(k - s, 0) for k in strikes) for s in np.arange(85.0, 115.01, 0.25)}
    assert pain == min(payout, key=payout.get) and pain in strikes
    assert oa.max_pain(pd.Series({100.0: 10.0}), pd.Series({100.0: 10.0})) == 100.0
    assert oa.max_pain(pd.Series({100.0: 10.0, 120.0: 0.0}), pd.Series(dtype=float)) == 100.0           # only calls: the lowest strike pays them nothing


def test_the_walls_and_totals_of_the_open_interest():
    chain = oi_chain([("C", 105.0, 500.0), ("C", 110.0, 800.0), ("C", 95.0, 9999.0), ("P", 95.0, 700.0), ("P", 90.0, 300.0), ("P", 105.0, 9999.0)])
    out = oa.positioning(chain, chain["expiry"].iloc[0])
    assert out["call_wall"] == 110.0 and out["put_wall"] == 95.0                                      # only calls above the price and puts below it count
    assert out["call_oi"] == 500 + 800 + 9999 and out["put_oi"] == 700 + 300 + 9999 and out["put_call_oi"] == pytest.approx((700 + 300 + 9999) / (500 + 800 + 9999))
    assert [r["strike"] for r in out["rows"]] == [90.0, 95.0, 105.0, 110.0] and out["max_pain"] in (90.0, 95.0, 105.0, 110.0)
    assert oa.positioning(chain, None)["max_pain"] is None and oa.positioning(chain, None)["scope"] == "all expiries"


def test_gamma_exposure_is_gamma_times_open_interest_times_the_square_of_the_price_per_percent():
    spot, sigma, days, K, oi = 100.0, 0.25, 30, 105.0, 400.0
    T = days / 365.0
    d1 = (math.log(spot / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    gamma = norm.pdf(d1) / (spot * sigma * math.sqrt(T))
    chain = oi_chain([("C", K, oi), ("P", 95.0, oi / 2)], spot, sigma, days)
    out = oa.positioning(chain, chain["expiry"].iloc[0])
    call = gamma * oi * 100.0 * spot ** 2 / 100.0
    d1p = (math.log(spot / 95.0) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    put = -norm.pdf(d1p) / (spot * sigma * math.sqrt(T)) * (oi / 2) * 100.0 * spot ** 2 / 100.0
    assert out["gex"]["calls"] == pytest.approx(call, rel=1e-9) and out["gex"]["puts"] == pytest.approx(put, rel=1e-9) and out["gex"]["net"] == pytest.approx(call + put, rel=1e-9)
    rows = {r["strike"]: r for r in out["rows"]}
    assert rows[105.0]["call_gex"] == pytest.approx(call, rel=1e-9) and rows[95.0]["put_gex"] == pytest.approx(put, rel=1e-9) and rows[105.0]["put_gex"] == 0.0


def test_the_gamma_flip_is_where_the_call_and_put_gammas_cancel():
    sigma, days = 0.2, 30
    h = 0.5 * sigma ** 2 * days / 365.0
    chain = oi_chain([("C", 120.0, 1000.0), ("P", 80.0, 1000.0)], 100.0, sigma, days)
    out = oa.positioning(chain, chain["expiry"].iloc[0], grid_points=121, span=0.3)
    exact = math.sqrt(120.0 * 80.0) * math.exp(-h)                                                   # the two gammas are equal where |d1| is, by symmetry in log price
    assert out["gex"]["flip"] == pytest.approx(exact, abs=0.05)
    curve = out["gex"]["curve"]
    assert len(curve["spot"]) == 121 and curve["net"][0] < 0 < curve["net"][-1] and out["gex"]["net"] > 0                       # the price (100) is above the flip, where the call gamma is the larger
    calls_only = oa.positioning(oi_chain([("C", 105.0, 10.0)]), AS_OF + pd.Timedelta(days=30))
    assert calls_only["gex"]["flip"] is None and calls_only["gex"]["net"] > 0 and min(calls_only["gex"]["curve"]["net"]) > 0
    puts_only = oa.positioning(oi_chain([("P", 95.0, 10.0)]), AS_OF + pd.Timedelta(days=30))
    assert puts_only["gex"]["flip"] is None and puts_only["gex"]["net"] < 0


# ------------------------------------------------------------------------------------------------------------------ a real chain
SPX_SNAPSHOT = "tests/fixtures/spx_chain_20261029.csv.gz"
PUBLISHED_VIX = 17.20                                                          # Cboe's delayed quote for the VIX at 16:15 ET on 2026-10-29, read a minute after the chain below


def spx_chain() -> pd.DataFrame:
    """The SPX options of 29 October 2026 (Cboe delayed quotes, as the Free data card downloads them) on the two expiries around 30 days, 29 and 36 days out: every strike with a live bid within half to one and a half times the index."""
    from src.utils.config import load_config

    return oa.prepare_chain(pd.read_csv(load_config().root / SPX_SNAPSHOT, parse_dates=["date", "expiry"]))


def test_the_model_free_volatility_of_the_real_spx_chain_is_the_published_vix():
    chain = spx_chain()
    term = oa.model_free_term(chain, 30.0)
    assert [r["dte"] for r in term["rows"]] == [29, 36] and [r["strikes"] for r in term["rows"]] == [215, 186]
    assert term["iv_30d"] * 100 == pytest.approx(17.1392, abs=0.002)                                       # pinned: what the code gives for this snapshot
    assert abs(term["iv_30d"] * 100 - PUBLISHED_VIX) < 0.15                                                # and the VIX the exchange published: the Cboe formula, on the same options, to within a tenth of a point
    assert term["atm_30d"] * 100 < PUBLISHED_VIX - 3.0                                                     # the at-the-money volatility alone is far lower: the skew is most of the VIX


def test_the_real_chain_gives_a_fat_left_tail_and_a_fit_that_keeps_to_the_quotes():
    chain = spx_chain()
    d = oa.implied_distribution(chain, pd.Timestamp("2026-11-27"))
    assert d["moments"]["skewness"] == pytest.approx(-1.024, abs=0.02) and d["moments"]["excess_kurtosis"] > 3.0
    assert d["fit"]["arbitrage_free"] and d["fit"]["negative_share"] == 0.0 and d["fit"]["quotes"] == 215 and d["fit"]["max_error_vol"] < 0.012
    fall = next(t for t in d["tails"] if t["side"] == "fall" and t["move"] == 0.10)
    rise = next(t for t in d["tails"] if t["side"] == "rise" and t["move"] == 0.10)
    assert fall["implied"] > 5.0 * fall["lognormal"] and fall["implied"] > 5.0 * rise["implied"]           # a 10% fall is priced several times the lognormal's chance, and far above a 10% rise
    assert d["moments"]["mean_price_over_forward"] == pytest.approx(1.0, abs=5e-3) and 0.99 < d["area"] <= 1.0001
    em = d["expected_move"]
    assert em["straddle_pct"] < em["one_sd_pct"] and em["straddle_pct"] == pytest.approx(0.8 * em["one_sd_pct"], rel=0.12)
