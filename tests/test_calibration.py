"""Calibrating a model to option quotes: the fit recovers the parameters that generated the quotes (exactly, then with noise, where its standard errors have to describe the miss), it
says so when the model is the wrong one or strains against a bound, it holds fixed parameters and refuses a start outside the valid region, and it picks usable out-of-the-money quotes out of
a chain table."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.derivatives import calibration as cal
from src.derivatives import fourier, levy
from src.derivatives.iv import implied_vol
from src.derivatives.pricing import bsm_price

S, R, Q = 100.0, 0.02, 0.01
TRUTH = {
    "merton": levy.Merton(sigma=0.14, lam=1.2, mu_j=-0.09, sigma_j=0.11),
    "kou": levy.Kou(sigma=0.13, lam=2.5, p=0.3, eta1=14.0, eta2=9.0),
    "vg": levy.VarianceGamma(sigma=0.17, theta=-0.14, nu=0.3),
    "nig": levy.NIG(sigma=0.17, theta=-0.14, kappa=0.3),
    "heston": levy.Heston(v0=0.045, kappa=1.8, theta=0.05, xi=0.55, rho=-0.7),
}


def market(model, expiries=(0.08, 0.25, 0.75), n_strikes=9, noise=0.0, seed=0) -> cal.Quotes:
    """The out-of-the-money implied volatilities that ``model`` implies (plus normal noise, in volatility), over a range of strikes that widens with the expiry."""
    rng = np.random.default_rng(seed)
    T, K, iv = [], [], []
    for t in expiries:
        F = S * np.exp((R - Q) * t)
        strikes = F * np.exp(np.linspace(-0.22, 0.22, n_strikes) * (t / 0.25) ** 0.25)
        v = fourier.implied_vols(model, S, strikes, t, R, Q)
        T += [t] * n_strikes
        K += list(strikes)
        iv += list(v + noise * rng.standard_normal(n_strikes))
    return cal.Quotes(S, R, Q, np.array(T), np.array(K), np.array(iv))


# ------------------------------------------------------------------------------------------------------------------------------- recovery
@pytest.mark.parametrize("name", list(TRUTH))
def test_the_fit_recovers_the_parameters_that_generated_the_quotes(name):
    truth, quotes = TRUTH[name], market(TRUTH[name])
    fit = cal.calibrate(name, quotes)
    assert fit.success and fit.rmse < 1e-4 and fit.max_error < 5e-4 and fit.n_failed == 0
    for k, v in truth.params.items():
        assert fit.model.params[k] == pytest.approx(v, rel=0.02), (name, k)
    assert fit.n_quotes == 27 and fit.n_expiries == 3 and [b["n"] for b in fit.by_expiry] == [9, 9, 9] and max(b["rmse"] for b in fit.by_expiry) < 1e-4
    assert fit.at_bounds == [] and fit.seconds < 30
    assert cal.flat_smile_rmse(quotes) > 100 * fit.rmse, "the smile is large next to the fit's error: this is a real test"
    assert json.loads(json.dumps(fit.summary()))["model"] == name, "the summary goes straight to the dashboard"


def test_noisy_quotes_are_fitted_to_the_noise_level_and_the_standard_errors_describe_the_miss():
    truth = TRUTH["vg"]
    fit = cal.calibrate("vg", market(truth, noise=0.003, seed=4))
    assert 0.0024 < fit.rmse < 0.0036, "the residual is the noise that went in (about 0.003, a little less for the parameters fitted)"
    for k, v in truth.params.items():
        se = fit.std_errors[k]
        assert 0 < se < 0.2 * abs(v) and abs(fit.model.params[k] - v) < 4 * se, (k, fit.model.params[k], v, se)
    assert fit.condition < 100, "three parameters, three expiries: the problem is well conditioned"


# ----------------------------------------------------------------------------------------------------------------------------- wrong model
def test_the_wrong_model_is_ranked_below_the_right_one_and_a_strained_fit_is_flagged():
    quotes = market(TRUTH["vg"])
    ranked = cal.compare_models(quotes)
    assert ranked[0].model.name == "vg" and ranked[0].rmse < 1e-4
    assert [c.rmse for c in ranked] == sorted(c.rmse for c in ranked)
    heston = next(c for c in ranked if c.model.name == "heston")
    assert heston.rmse > 100 * ranked[0].rmse and {"kappa", "xi"} <= set(heston.at_bounds), "a stochastic-volatility model cannot make a short-dated smile this steep and says so"
    assert heston.condition > 1000, "and its parameters are not pinned down by the quotes"
    flat = cal.flat_smile_rmse(quotes)
    bs = cal.calibrate("bs", quotes)
    assert bs.rmse > flat > 0.03, "one volatility for all the expiries cannot even match one volatility per expiry"
    assert all(c.rmse < flat / 5 for c in ranked), "every model with a smile does much better than a flat line"


def test_comparing_models_skips_one_whose_search_fails_and_rejects_unknown_names():
    quotes = market(TRUTH["vg"])
    assert cal.compare_models(quotes, models=("vg",), fixed={"sigma": 0.17, "theta": -0.14, "nu": 0.3}) == []
    with pytest.raises(KeyError):
        cal.compare_models(quotes, models=("sabr",))


# ------------------------------------------------------------------------------------------------------------------------- fixed, invalid
def test_fixed_parameters_stay_where_they_are_and_cost_some_fit():
    quotes = market(TRUTH["heston"])
    free = cal.calibrate("heston", quotes)
    held = cal.calibrate("heston", quotes, fixed={"rho": -0.4})
    assert held.model.params["rho"] == -0.4 and "rho" not in held.std_errors and held.rmse > 50 * free.rmse
    with pytest.raises(ValueError, match="fixed"):
        cal.calibrate("vg", quotes, fixed={"sigma": 0.2, "theta": -0.1, "nu": 0.3})


def test_a_start_outside_the_valid_region_is_refused_not_returned_as_a_fit():
    quotes = market(TRUTH["vg"])
    start = {"sigma": 0.5, "theta": 0.9, "nu": 1.5}                      # 1 - theta nu - sigma^2 nu / 2 < 0: the price has no mean
    assert not levy.VarianceGamma(**start).valid()
    with pytest.raises(ValueError, match="valid region"):
        cal.calibrate("vg", quotes, start=start)


# --------------------------------------------------------------------------------------------------------------------------------- quotes
def test_quotes_describe_themselves_and_refuse_nonsense():
    q = market(TRUTH["nig"])
    assert q.forward == pytest.approx(S * np.exp((R - Q) * q.T)) and list(q.maturities) == [0.08, 0.25, 0.75]
    assert np.array_equal(q.is_call, q.K >= q.forward) and q.is_call.any() and (~q.is_call).any()
    px = q.market_prices()
    assert np.allclose(px[q.is_call], bsm_price(S, q.K[q.is_call], q.T[q.is_call], R, Q, q.iv[q.is_call], True))
    assert np.allclose(px[~q.is_call], bsm_price(S, q.K[~q.is_call], q.T[~q.is_call], R, Q, q.iv[~q.is_call], False))
    assert np.all(q.vegas() > 0) and 0.15 < q.atm_iv() < 0.25
    with pytest.raises(ValueError, match="at least three"):
        cal.Quotes(S, R, Q, [0.5, 0.5], [90.0, 110.0], [0.2, 0.2])
    with pytest.raises(ValueError, match="positive"):
        cal.Quotes(S, R, Q, [0.5, 0.5, 0.5], [90.0, 100.0, 110.0], [0.2, -0.2, 0.2])
    with pytest.raises(ValueError, match="positive"):
        cal.Quotes(S, R, Q, [0.5, 0.5, 0.0], [90.0, 100.0, 110.0], [0.2, 0.2, 0.2])


@pytest.mark.parametrize("name", list(TRUTH))
def test_starting_points_lie_inside_the_bounds_and_in_the_valid_region_of_every_model(name):
    cls = levy.MODELS[name]
    for start in cal.starting_points(cls, market(TRUTH[name])):
        assert set(start) == set(cls.param_names)
        assert all(cls.bounds[k][0] <= v <= cls.bounds[k][1] for k, v in start.items())
        m = cls(**start)
        assert getattr(m, "valid", lambda: True)() and m.exponent_limit() > 1.05


def test_the_fitted_models_prices_and_volatilities_are_what_the_calibration_reports():
    quotes = market(TRUTH["kou"])
    fit = cal.calibrate("kou", quotes)
    assert np.allclose(cal.model_ivs(fit.model, quotes), fit.model_iv, atol=1e-12)
    px = cal.model_otm_prices(fit.model, quotes)
    for T in quotes.maturities:
        idx = quotes.T == T
        calls, puts = fourier.price(fit.model, S, quotes.K[idx], T, R, Q, True), fourier.price(fit.model, S, quotes.K[idx], T, R, Q, False)
        assert np.allclose(px[idx], np.where(quotes.is_call[idx], calls, puts), atol=1e-8)


# ----------------------------------------------------------------------------------------------------------------------------------- chain
def chain_from(model, spot=S, rate=R, dividend=Q, dtes=(3, 10, 20, 35, 60, 100, 180, 400), step=2.5, half_spread=0.02) -> pd.DataFrame:
    """A chain table in the platform's schema: both rights at every strike for each expiry, priced by ``model`` with a bid-ask spread, plus the junk a real chain has."""
    date = pd.Timestamp("2026-03-02")
    rows = []
    for dte in dtes:
        T = dte / 365.0
        strikes = np.arange(60.0, 140.0 + step, step)
        F = spot * np.exp((rate - dividend) * T)
        calls = fourier.price(model, spot, strikes, T, rate, dividend, True)
        for right, px in (("C", calls), ("P", calls - spot * np.exp(-dividend * T) + strikes * np.exp(-rate * T))):
            mid = np.maximum(px, 0.0)
            df = pd.DataFrame({"date": date, "expiry": date + pd.Timedelta(days=dte), "strike": strikes, "right": right, "bid": np.maximum(mid - half_spread, 0.0), "ask": mid + half_spread,
                               "underlying": spot, "rate": rate, "dividend": dividend})
            df["mid"], df["T"], df["F"] = mid, T, F
            iv = np.full(len(df), np.nan)
            ok = mid > 1e-4
            iv[ok] = implied_vol(mid[ok], spot, strikes[ok], T, rate, dividend, right == "C")
            df["iv"] = iv
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_the_quotes_taken_from_a_chain_are_liquid_out_of_the_money_and_spread_along_the_curve():
    chain = chain_from(TRUTH["vg"])
    q = cal.quotes_from_chain(chain, max_expiries=4, min_dte=7, window=0.25, min_points=6, max_strikes=10, min_mid=0.05)
    assert len(q.maturities) == 4 and len(q.expiries) == 4 and q.maturities.min() * 365 >= 7 and q.maturities.max() * 365 == pytest.approx(400)
    assert q.maturities[1] < q.maturities[2] < q.maturities[3] and q.expiries == sorted(q.expiries)
    assert all((q.T == T).sum() <= 10 and (q.T == T).sum() >= 6 for T in q.maturities)
    assert np.all(np.abs(np.log(q.K / q.forward)) <= 0.25 + 1e-12)
    assert q.S == S and q.r == pytest.approx(R) and q.q == pytest.approx(Q)
    for T in q.maturities:                                               # the volatilities are the model's: a calibration to them recovers it
        idx = q.T == T
        assert np.allclose(q.iv[idx], fourier.implied_vols(TRUTH["vg"], S, q.K[idx], T, R, Q), atol=2e-3)


def test_a_chain_without_volatilities_or_mids_gives_the_same_quotes_and_a_thin_chain_is_refused():
    chain = chain_from(TRUTH["nig"])
    full = cal.quotes_from_chain(chain, max_expiries=3)
    got = cal.quotes_from_chain(chain.drop(columns=["iv", "mid", "T", "F"]), max_expiries=3)
    assert np.allclose(got.K, full.K) and np.allclose(got.T, full.T, atol=1e-12) and np.allclose(got.iv, full.iv, atol=1e-7)
    assert cal.quotes_from_chain(chain.drop(columns=["dividend"]), max_expiries=3).q == 0.0
    with pytest.raises(ValueError, match="too thin"):
        cal.quotes_from_chain(chain, min_points=500)
    with pytest.raises(ValueError, match="too thin"):
        cal.quotes_from_chain(chain, min_dte=5000)


def test_quotes_with_no_bid_a_tiny_mid_or_no_volatility_are_left_out():
    chain = chain_from(TRUTH["vg"], dtes=(60, 120))
    junk = chain.index[(chain["expiry"] == chain["expiry"].min()) & (chain["strike"].between(80, 85))]
    chain.loc[junk, "bid"] = 0.0
    clean = cal.quotes_from_chain(chain, max_expiries=2, min_points=3)
    base = cal.quotes_from_chain(chain_from(TRUTH["vg"], dtes=(60, 120)), max_expiries=2, min_points=3)
    first = base.T == base.T.min()
    dropped = [k for k in base.K[first] if 80 <= k <= 85 and k < base.forward[first][0]]
    assert dropped and not any(np.isclose(clean.K[clean.T == clean.T.min()], d).any() for d in dropped)
    assert (clean.T == clean.T.max()).sum() == (base.T == base.T.max()).sum(), "the other expiry is untouched"
