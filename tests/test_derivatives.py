"""Derivatives: pricing against closed forms, Monte Carlo and each other; Greeks against finite differences and the Black-Scholes PDE; surfaces against known parameters;
the option backtester against hand-computed cash flows and its own accounting identities."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.derivatives import backtest as bt
from src.derivatives import greeks as gk
from src.derivatives import iv as ivmod
from src.derivatives import pricing as pr
from src.derivatives import schema, strategies as st, surface as sf, synthetic as syn, volatility as vol
from src.probability.montecarlo import simulate_merton_jump

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------------------------------------------------- pricing
def test_bsm_matches_textbook_values_and_parity():
    assert pr.bsm_price(100, 100, 1, 0.05, 0.0, 0.2, True) == pytest.approx(10.450583572185565)
    assert pr.bsm_price(100, 100, 1, 0.05, 0.0, 0.2, False) == pytest.approx(5.573526022256971)
    assert pr.bsm_price(42, 40, 0.5, 0.10, 0.0, 0.20, True) == pytest.approx(4.7594, abs=1e-3)               # Hull, example 15.6
    rng = np.random.default_rng(0)
    S, K, T, r, q, s = (rng.uniform(50, 150, 200), rng.uniform(50, 150, 200), rng.uniform(0.05, 3, 200), rng.uniform(0, 0.06, 200), rng.uniform(0, 0.03, 200),
                        rng.uniform(0.05, 0.9, 200))
    c, p = pr.bsm_price(S, K, T, r, q, s, True), pr.bsm_price(S, K, T, r, q, s, False)
    assert np.abs(pr.put_call_parity_gap(c, p, S, K, T, r, q)).max() < 1e-10
    assert (c >= np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0) - 1e-12).all() and (c <= S * np.exp(-q * T) + 1e-12).all()


def test_bsm_degenerate_cases_and_monotonicity():
    assert pr.bsm_price(110, 100, 0.0, 0.0, 0.0, 0.2, True) == pytest.approx(10.0)
    assert pr.bsm_price(90, 100, 1.0, 0.0, 0.0, 0.0, False) == pytest.approx(10.0)
    sig = np.linspace(0.05, 1.0, 50)
    assert (np.diff(pr.bsm_price(100, 100, 1, 0.02, 0, sig, True)) > 0).all()
    assert (np.diff(pr.bsm_price(100, np.linspace(60, 140, 50), 1, 0.02, 0, 0.2, True)) < 0).all()
    assert pr.intrinsic(110, 100, True) == 10 and pr.intrinsic(110, 100, False) == 0


def test_black76_and_bachelier_consistency():
    F, K, T, r, s = 100.0, 95.0, 0.5, 0.03, 0.3
    assert pr.black76_price(F, K, T, r, s, True) == pytest.approx(np.exp(-r * T) * pr.bsm_price(F, K, T, 0.0, 0.0, s, True))
    c, p = pr.black76_price(F, K, T, r, s, True), pr.black76_price(F, K, T, r, s, False)
    assert c - p == pytest.approx(np.exp(-r * T) * (F - K))
    assert pr.bachelier_price(100, 100, 1, 0.0, 20.0, True) == pytest.approx(20.0 / np.sqrt(2 * np.pi))
    assert pr.bachelier_price(-5.0, 0.0, 1, 0.0, 10.0, True) > 0                                        # normal model prices negative forwards
    # a small normal vol is approximately the lognormal vol times the forward
    assert pr.bachelier_price(100, 100, 0.25, 0.0, 20.0, True) == pytest.approx(pr.black76_price(100, 100, 0.25, 0.0, 0.2, True), rel=0.01)


def test_binomial_converges_to_bsm_and_american_put_has_an_early_exercise_premium():
    bs = pr.bsm_price(100, 100, 1, 0.05, 0.0, 0.2, True)
    errs = [abs(pr.binomial_price(100, 100, 1, 0.05, 0.0, 0.2, True, False, n) - bs) for n in (50, 200, 800)]
    assert errs[2] < errs[1] < errs[0] + 1e-9 and errs[2] < 0.01
    eu_put, am_put = pr.binomial_price(100, 100, 1, 0.05, 0.0, 0.2, False, False, 500), pr.binomial_price(100, 100, 1, 0.05, 0.0, 0.2, False, True, 500)
    assert am_put > eu_put + 0.3 and eu_put == pytest.approx(pr.bsm_price(100, 100, 1, 0.05, 0.0, 0.2, False), abs=0.01)
    assert pr.binomial_price(100, 100, 1, 0.05, 0.0, 0.2, True, True, 500) == pytest.approx(pr.binomial_price(100, 100, 1, 0.05, 0.0, 0.2, True, False, 500), abs=1e-9)   # no dividend: never exercise a call early
    assert pr.binomial_price(100, 100, 1, 0.05, 0.06, 0.2, True, True, 500) > pr.binomial_price(100, 100, 1, 0.05, 0.06, 0.2, True, False, 500) + 0.01    # with a high yield, it pays


def test_longstaff_schwartz_agrees_with_the_tree():
    tree = pr.binomial_price(100, 100, 1, 0.05, 0.0, 0.2, False, True, 800)
    out = pr.american_lsmc(100, 100, 1, 0.05, 0.0, 0.2, False, n_paths=60000, steps=50, seed=1)
    assert abs(out["price"] - tree) < 4 * out["se"] + 0.05 and out["early_exercise_premium"] > 0.2


def test_heston_reduces_to_bsm_satisfies_parity_and_matches_monte_carlo():
    K = np.array([85.0, 100.0, 115.0])
    assert pr.heston_price(100, K, 1, 0.02, 0.01, 0.04, 2.0, 0.04, 1e-4, 0.0) == pytest.approx(pr.bsm_price(100, K, 1, 0.02, 0.01, 0.2, True), abs=2e-4)
    c = pr.heston_price(100, 100, 1, 0.02, 0.01, 0.04, 1.5, 0.05, 0.5, -0.7, True)
    p = pr.heston_price(100, 100, 1, 0.02, 0.01, 0.04, 1.5, 0.05, 0.5, -0.7, False)
    assert c - p == pytest.approx(100 * np.exp(-0.01) - 100 * np.exp(-0.02), abs=1e-6)
    s, _ = pr.heston_simulate(100, 1, 0.02, 0.01, 0.04, 1.5, 0.05, 0.5, -0.7, 252, 60000, 1)
    mc = np.exp(-0.02) * np.maximum(s[:, -1] - 100, 0)
    assert abs(c - mc.mean()) < 4 * mc.std() / np.sqrt(len(mc)) + 0.02


def test_heston_negative_correlation_produces_a_negative_skew():
    k = np.array([80.0, 100.0, 120.0])
    px = pr.heston_price(100, k, 1, 0.0, 0.0, 0.04, 2.0, 0.04, 0.6, -0.8, True)
    iv_ = ivmod.implied_vol(px, 100, k, 1, 0.0, 0.0, True)
    assert iv_[0] > iv_[1] > iv_[2]


def test_heston_calibration_recovers_a_smile():
    truth = dict(v0=0.04, kappa=2.0, theta=0.05, xi=0.4, rho=-0.6)
    strikes, expiries = np.array([85.0, 95.0, 100.0, 105.0, 115.0]), np.array([0.25, 1.0])
    mkt = np.vstack([ivmod.implied_vol(pr.heston_price(100, strikes, T, 0.01, 0.0, **truth), 100, strikes, T, 0.01, 0.0, True) for T in expiries])
    fit = pr.heston_calibrate(100, strikes, expiries, mkt, 0.01, 0.0, x0=(0.05, 1.5, 0.04, 0.5, -0.5))
    assert fit["rmse_vol"] < 5e-4 and abs(fit["rho"] - truth["rho"]) < 0.15 and abs(fit["v0"] - truth["v0"]) < 0.01


def test_merton_jump_diffusion_matches_monte_carlo_and_reduces_to_bsm():
    assert pr.merton_price(100, 100, 1, 0.05, 0.0, 0.2, 0.0, -0.05, 0.1, True) == pytest.approx(pr.bsm_price(100, 100, 1, 0.05, 0.0, 0.2, True), abs=1e-9)
    price = pr.merton_price(100, 100, 1, 0.05, 0.0, 0.15, 0.5, -0.05, 0.1, True)
    paths = simulate_merton_jump(100, 0.05, 0.15, 0.5, -0.05, 0.1, 1.0, 50, 200000, seed=2)
    mc = np.exp(-0.05) * np.maximum(paths[:, -1] - 100, 0)
    assert abs(price - mc.mean()) < 4 * mc.std() / np.sqrt(len(mc))
    assert pr.merton_price(100, 100, 1, 0.05, 0.0, 0.15, 0.5, -0.05, 0.1, True) > pr.bsm_price(100, 100, 1, 0.05, 0.0, 0.15, True)


def test_sabr_atm_limit_smile_shape_and_calibration():
    F, T, a, b, rho, nu = 100.0, 1.0, 0.2, 0.5, -0.3, 0.4
    atm = pr.sabr_vol(F, F, T, a, b, rho, nu)
    assert atm == pytest.approx(pr.sabr_vol(F, F * (1 + 1e-7), T, a, b, rho, nu), rel=1e-5)
    k = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
    smile = pr.sabr_vol(F, k, T, a, b, rho, nu)
    assert smile[0] > smile[-1] and smile.min() > 0
    fit = pr.sabr_calibrate(F, k, T, smile, beta=0.5)
    assert fit["rmse"] < 1e-8 and fit["rho"] == pytest.approx(rho, abs=1e-3) and fit["nu"] == pytest.approx(nu, abs=1e-3) and fit["alpha"] == pytest.approx(a, abs=1e-4)


# ------------------------------------------------------------------------------------------------------------- implied volatility
@pytest.mark.parametrize("call", [True, False])
def test_implied_vol_round_trip_on_a_wide_random_grid(call):
    rng = np.random.default_rng(0)
    n = 30000
    S, K, T, r, q, s = (np.full(n, 100.0), rng.uniform(40, 200, n), rng.uniform(0.01, 3, n), rng.uniform(0, 0.06, n), rng.uniform(0, 0.03, n), rng.uniform(0.05, 1.5, n))
    px = pr.bsm_price(S, K, T, r, q, s, call)
    iv_ = ivmod.implied_vol(px, S, K, T, r, q, call)
    ok = np.isfinite(iv_)
    assert ok.mean() > 0.95 and np.abs(iv_ - s)[ok].max() < 5e-5
    assert (np.abs(iv_ - s)[ok] < 1e-8).mean() > 0.99


def test_implied_vol_rejects_prices_outside_the_no_arbitrage_bounds():
    lower = 100 - 90 * np.exp(-0.05)
    out = ivmod.implied_vol(np.array([lower - 0.5, 101.0, lower + 3.0]), 100, 90, 1, 0.05, 0.0, True)
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isfinite(out[2])
    assert np.isnan(ivmod.implied_vol(5.0, 100, 100, 0.0, 0.05, 0.0, True))


def test_implied_vol_for_black76_bachelier_and_american():
    assert ivmod.implied_vol_black76(pr.black76_price(100, 95, 0.5, 0.03, 0.3, True), 100, 95, 0.5, 0.03, True) == pytest.approx(0.3, abs=1e-8)
    assert ivmod.implied_vol_bachelier(pr.bachelier_price(100, 95, 0.5, 0.03, 15.0, True), 100, 95, 0.5, 0.03, True) == pytest.approx(15.0, abs=1e-6)
    am = pr.binomial_price(100, 100, 1, 0.05, 0.0, 0.25, False, True, 200)
    assert ivmod.implied_vol_american(am, 100, 100, 1, 0.05, 0.0, False) == pytest.approx(0.25, abs=1e-4)
    assert ivmod.implied_vol(am, 100, 100, 1, 0.05, 0.0, False) > 0.25            # treating an American put as European overstates the volatility


# ----------------------------------------------------------------------------------------------------------------------- Greeks
@pytest.mark.parametrize("call", [True, False])
def test_analytic_greeks_match_finite_differences(call):
    S, K, T, r, q, s = 100.0, 105.0, 0.75, 0.03, 0.01, 0.25
    g = gk.bsm_greeks(S, K, T, r, q, s, call)
    n = gk.numerical_greeks(lambda S, sigma, T, r: pr.bsm_price(S, K, T, r, q, sigma, call), S, s, T, r)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert float(g[name]) == pytest.approx(n[name], rel=2e-4, abs=1e-6), name
    h = 1e-4
    up, dn = gk.bsm_greeks(S, K, T, r, q, s + h, call), gk.bsm_greeks(S, K, T, r, q, s - h, call)
    assert float(g["vanna"]) == pytest.approx((float(up["delta"]) - float(dn["delta"])) / (2 * h), rel=1e-4)
    assert float(g["volga"]) == pytest.approx((float(up["vega"]) - float(dn["vega"])) / (2 * h), rel=1e-4)
    tu, td = gk.bsm_greeks(S, K, T + h, r, q, s, call), gk.bsm_greeks(S, K, T - h, r, q, s, call)
    assert float(g["charm"]) == pytest.approx(-(float(tu["delta"]) - float(td["delta"])) / (2 * h), rel=1e-4)
    su, sd = gk.bsm_greeks(S + 0.01, K, T, r, q, s, call), gk.bsm_greeks(S - 0.01, K, T, r, q, s, call)
    assert float(g["speed"]) == pytest.approx((float(su["gamma"]) - float(sd["gamma"])) / 0.02, rel=1e-3)


@pytest.mark.parametrize("call", [True, False])
def test_greeks_satisfy_the_black_scholes_pde_and_parity_relations(call):
    S, K, T, r, q, s = 100.0, 95.0, 0.5, 0.04, 0.02, 0.3
    g = gk.bsm_greeks(S, K, T, r, q, s, call)
    price = float(pr.bsm_price(S, K, T, r, q, s, call))
    assert float(g["theta"]) + 0.5 * s ** 2 * S ** 2 * float(g["gamma"]) + (r - q) * S * float(g["delta"]) - r * price == pytest.approx(0.0, abs=1e-9)
    c, p = gk.bsm_greeks(S, K, T, r, q, s, True), gk.bsm_greeks(S, K, T, r, q, s, False)
    assert float(c["delta"]) - float(p["delta"]) == pytest.approx(np.exp(-q * T)) and float(c["gamma"]) == pytest.approx(float(p["gamma"])) and float(c["vega"]) == pytest.approx(float(p["vega"]))


def test_black76_greeks_and_portfolio_aggregation():
    g = gk.black76_greeks(100, 100, 0.5, 0.02, 0.2, True)
    h = 1e-3
    num = (pr.black76_price(100 + h, 100, 0.5, 0.02, 0.2, True) - pr.black76_price(100 - h, 100, 0.5, 0.02, 0.2, True)) / (2 * h)
    assert float(g["delta"]) == pytest.approx(float(num), rel=1e-5)
    pos = pd.DataFrame({"quantity": [-10, 10], "delta": [0.5, 0.3], "gamma": [0.02, 0.01], "vega": [20.0, 15.0], "theta": [-8.0, -5.0]})
    agg = gk.portfolio_greeks(pos, 100.0)
    assert agg["delta_shares"] == pytest.approx(100.0 * (-10 * 0.5 + 10 * 0.3)) and agg["hedge_units"] == pytest.approx(-agg["delta_shares"])
    assert agg["dollar_gamma_1pct"] == pytest.approx(0.5 * 100.0 * (-10 * 0.02 + 10 * 0.01) * 1.0)
    ex = gk.pnl_explain(0.5, 0.02, 20.0, -8.0, 2.0, 0.01, 1.0)
    assert ex["total_explained"] == pytest.approx(0.5 * 2 + 0.5 * 0.02 * 4 + 20 * 0.01 - 8 / 365)


def test_delta_hedged_option_earns_gamma_theta_balance():
    S0, K, T, r, s_imp = 100.0, 100.0, 30 / 365, 0.0, 0.2
    rng = np.random.default_rng(1)
    pnls = []
    for realised in (0.1, 0.2, 0.4):
        total = []
        for _ in range(300):
            S, t, pnl = S0, T, 0.0
            for _ in range(30):
                g = gk.bsm_greeks(S, K, t, r, 0.0, s_imp, True)
                dS = S * realised * np.sqrt(1 / 365) * rng.standard_normal()
                v0 = float(pr.bsm_price(S, K, t, r, 0.0, s_imp, True))
                v1 = float(pr.bsm_price(S + dS, K, t - 1 / 365, r, 0.0, s_imp, True))
                pnl += (v1 - v0) - float(g["delta"]) * dS
                S, t = S + dS, t - 1 / 365
            total.append(pnl)
        pnls.append(np.mean(total))
    assert pnls[0] < pnls[1] + 0.05 < pnls[2] + 0.05 and pnls[0] < -0.3 and pnls[2] > 0.5 and abs(pnls[1]) < 0.3     # long gamma wins when realised > implied, loses when below


# ---------------------------------------------------------------------------------------------------------------------- surfaces
def test_svi_fit_recovers_parameters_and_flags_arbitrage():
    true = dict(a=0.02, b=0.15, rho=-0.6, m=0.02, s=0.15)
    T = 0.5
    k = np.linspace(-0.5, 0.4, 25)
    fit = sf.fit_svi_slice(k, np.sqrt(sf.svi_total_variance(k, **true) / T), T)
    assert fit.params == pytest.approx(tuple(true.values()), abs=1e-5) and fit.rmse < 1e-8 and fit.arbitrage_free()
    bad = sf.SVISlice(0.0, 2.0, 0.9, 0.0, 0.01, T)               # a very steep right wing violates the butterfly condition
    assert not bad.arbitrage_free()
    assert (fit.density_g(np.linspace(-1, 1, 50)) > 0).all()


def test_ssvi_recovery_and_arbitrage_conditions():
    surf = sf.SSVISurface([0.1, 0.25, 0.5, 1.0], [0.01, 0.025, 0.05, 0.1], -0.6, 0.8, 0.5)
    assert surf.arbitrage_free()["free"]
    rows = [{"T": T, "k": k, "iv": float(surf.iv(k, T))} for T in (0.1, 0.25, 0.5, 1.0) for k in np.linspace(-0.4, 0.4, 17)]
    fit = sf.fit_ssvi(pd.DataFrame(rows))
    assert (fit.rho, fit.eta, fit.gamma) == pytest.approx((-0.6, 0.8, 0.5), abs=1e-3) and fit.arbitrage_free()["free"]
    assert not sf.ssvi_arbitrage_free([0.5, 1.0], -0.9, 6.0, 0.5)["butterfly"]
    assert not sf.ssvi_arbitrage_free([0.1, 0.05], -0.5, 0.5, 0.5)["calendar"]
    w = np.array([[float(surf.w(0.0, T)) for T in (0.1, 0.25, 0.5, 1.0)], [float(surf.w(0.2, T)) for T in (0.1, 0.25, 0.5, 1.0)]])
    assert (np.diff(w, axis=1) > 0).all()                          # total variance increases with expiry: no calendar arbitrage


def test_local_vol_density_variance_swap_and_vix_on_known_surfaces():
    const = lambda k, T: 0.04 * np.asarray(T) * np.ones_like(np.asarray(k, float))
    assert sf.local_volatility(const, np.array([-0.2, 0.0, 0.2]), 0.5) == pytest.approx(0.2, abs=1e-6)
    surf = sf.SSVISurface([0.1, 0.25, 0.5, 1.0], [0.01, 0.025, 0.05, 0.1], -0.6, 0.8, 0.5)
    lv = sf.local_volatility(surf.w, np.array([-0.3, 0.0, 0.3]), 0.5)
    assert np.isfinite(lv).all() and lv[0] > lv[1] > lv[2]         # negative skew: local vol is higher on the downside
    rnd = sf.risk_neutral_density(surf.w, 0.5)
    assert np.trapezoid(rnd.values, rnd.index) == pytest.approx(1.0, abs=0.01) and np.trapezoid(rnd.values * np.exp(rnd.index), rnd.index) == pytest.approx(1.0, abs=0.01)
    F, T, r = 100.0, 0.25, 0.02
    K = np.arange(40, 200, 0.5)
    px = np.where(K <= F, pr.bsm_price(F * np.exp(-r * T), K, T, r, 0, 0.2, False), pr.bsm_price(F * np.exp(-r * T), K, T, r, 0, 0.2, True))
    assert sf.variance_swap_strike(F, T, r, K, px) == pytest.approx(0.04, rel=2e-3)
    assert sf.price_from_surface(100, 100, 0.5, 0.0, 0.0, const) == pytest.approx(pr.bsm_price(100, 100, 0.5, 0.0, 0.0, 0.2, True))
    sm = sf.smile_metrics(surf.w, 0.5)
    assert sm["rr_25d"] < 0 and sm["skew_slope"] < 0 and sm["atm_vol"] == pytest.approx(float(surf.iv(0.0, 0.5)))


def _bs_chain(sigma=0.2, S=100.0, r=0.02, q=0.0, expiries=(20, 45), date="2022-01-03"):
    rows = []
    d = pd.Timestamp(date)
    for days in expiries:
        T = days / 365.0
        for K in np.arange(60, 141, 1.0):
            for right in ("C", "P"):
                mid = float(pr.bsm_price(S, K, T, r, q, sigma, right == "C"))
                rows.append({"date": d, "expiry": d + pd.Timedelta(days=days), "strike": K, "right": right, "bid": max(mid - 0.005, 0.0), "ask": mid + 0.005, "underlying": S, "rate": r, "dividend": q,
                             "iv": sigma})
    return schema.normalise_chain(pd.DataFrame(rows))


def test_vix_style_index_on_a_flat_bsm_chain_equals_the_volatility():
    ch = _bs_chain(0.2)
    front, nxt = ch[ch["expiry"] == ch["expiry"].min()], ch[ch["expiry"] == ch["expiry"].max()]
    assert sf.vix_style_index(front, nxt, 30.0) == pytest.approx(20.0, abs=0.35)
    assert vol.vix_series({pd.Timestamp("2022-01-03"): ch}).iloc[0] == pytest.approx(20.0, abs=0.35)


# ------------------------------------------------------------------------------------------------------------------------ schema
def test_validation_flags_every_kind_of_bad_quote():
    ch = _bs_chain().head(20).copy()
    ch.loc[0, "bid"], ch.loc[0, "ask"] = 2.0, 1.0
    ch.loc[1, "strike"] = -5.0
    ch.loc[2, "expiry"] = ch.loc[2, "date"]
    ch.loc[3, "right"] = "X"
    ch = pd.concat([ch, ch.iloc[[10]]], ignore_index=True)
    problems = schema.validate_chain(ch)
    assert {"bid > ask (crossed quote)", "non-positive strike", "expiry on or before the quote date", "right not C or P"} <= set(problems["check"])
    assert any("duplicate" in c for c in problems["check"])
    assert schema.validate_chain(_bs_chain()).empty
    with pytest.raises(ValueError):
        schema.normalise_chain(pd.DataFrame({"date": [1]}))


def test_arbitrage_report_is_clean_for_bsm_and_catches_planted_violations():
    ch = _bs_chain()
    rep = schema.arbitrage_report(ch).set_index("check")
    assert (rep["violations"] == 0).all()
    bad = ch.copy()
    mask = (bad["expiry"] == bad["expiry"].min()) & (bad["right"] == "C") & (bad["strike"] == 100.0)
    bad.loc[mask, ["bid", "ask", "mid"]] = [0.0, 0.01, 0.005]              # a call at the money worth almost nothing: breaks monotonicity and convexity
    bad_rep = schema.arbitrage_report(bad).set_index("check")
    assert bad_rep.loc["vertical spread (price monotone in strike)", "violations"] > 0 and bad_rep.loc["butterfly (price convex in strike)", "violations"] > 0
    crossed = ch.copy()
    crossed.loc[(crossed["right"] == "C") & (crossed["strike"] == 100.0) & (crossed["expiry"] == crossed["expiry"].min()), ["bid", "ask", "mid"]] += 3.0
    assert schema.arbitrage_report(crossed).set_index("check").loc["put-call parity within the spread", "violations"] > 0
    calendar = ch.copy()
    calendar.loc[calendar["expiry"] == calendar["expiry"].max(), "iv"] = 0.1
    assert schema.arbitrage_report(calendar).set_index("check").loc["calendar spread (total variance rises with expiry)", "violations"] > 0


def test_parity_forward_recovers_the_forward_and_rate():
    ch = _bs_chain(0.25, S=100.0, r=0.03, q=0.01)
    e = ch[ch["expiry"] == ch["expiry"].min()]
    out = schema.parity_forward(e)
    assert out["forward"] == pytest.approx(float(e["F"].iloc[0]), rel=1e-4) and out["rate"] == pytest.approx(0.03, abs=2e-3)
    assert np.isnan(schema.parity_forward(e.head(2))["forward"])
    otm = schema.otm_quotes(e)
    assert ((otm["right"] == "P") == (otm["strike"] <= otm["F"])).all()


def test_vendor_csv_loader_maps_names_scales_strikes_and_complains_about_missing_columns(tmp_path):
    raw = pd.DataFrame({"Quote_Date": ["2022-01-03"] * 2, "Expiration": ["2022-02-18"] * 2, "Strike_Price": [100000, 100000], "CP_Flag": ["call", "put"], "Best_Bid": [3.0, 2.5], "Best_Offer": [3.2, 2.7],
                        "Underlying_Last": [100.0, 100.0]})
    path = tmp_path / "chain.csv"
    raw.to_csv(path, index=False)
    ch = syn.load_option_csv(path, strike_scale=0.001, default_rate=0.01)
    assert ch["strike"].tolist() == [100.0, 100.0] and set(ch["right"]) == {"C", "P"} and ch["T"].iloc[0] == pytest.approx(46 / 365) and ch["rate"].iloc[0] == 0.01
    raw.drop(columns=["Best_Bid"]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        syn.load_option_csv(path)
    assert syn.load_option_csv(path, column_map={"bid": "Best_Offer"}).shape[0] == 2


# --------------------------------------------------------------------------------------------------------------- synthetic market
@pytest.fixture(scope="module")
def market():
    return syn.generate_market(syn.SyntheticMarketSpec(), n_days=330, seed=7, warm=250)


def test_synthetic_market_is_reproducible_and_well_formed(market):
    again = syn.generate_market(syn.SyntheticMarketSpec(), n_days=40, seed=7, warm=250)
    first = market.chain(market.dates()[0])
    pd.testing.assert_frame_equal(again.chain(again.dates()[0]).reset_index(drop=True), first.reset_index(drop=True))
    c = market.chain(market.dates()[100])
    assert schema.validate_chain(c).empty and (c["ask"] >= c["bid"]).all() and (c["bid"] >= 0).all()
    assert c["expiry"].nunique() >= 8 and set(c["right"]) == {"C", "P"} and (c["T"] > 0).all()
    assert market.underlying["var"].min() > 0 and market.underlying["spot"].min() > 0


def test_synthetic_listings_persist_until_expiry(market):
    dates = market.dates()
    for d0, d1 in zip(dates[:-1:15], dates[1::15]):
        a = {(r.expiry, r.strike, r.right) for r in market.chain(d0).itertuples() if r.expiry > d1}
        b = {(r.expiry, r.strike, r.right) for r in market.chain(d1).itertuples()}
        assert a <= b


def test_synthetic_quotes_carry_the_known_surface_and_it_is_arbitrage_free(market):
    d = market.dates()[120]
    c = market.chain(d)
    e = c[(c["expiry"] == sorted(c["expiry"].unique())[4]) & (c["right"] == "C") & (c["bid"] > 0.2)]
    truth = market.true_iv(d, e["strike"].to_numpy(), e["expiry"].iloc[0])
    assert np.abs(e["iv"].to_numpy() / truth - 1).max() < 0.03
    sur = syn.surface_from_spec(market.spec, float(market.underlying.loc[d, "var"]), [0.05, 0.1, 0.25, 0.5, 1.0])
    assert sur.arbitrage_free()["free"]
    rep = schema.arbitrage_report(c).set_index("check")
    assert rep.loc["put-call parity within the spread", "violations"] / max(rep.loc["put-call parity within the spread", "tested"], 1) < 0.05
    assert rep.loc["calendar spread (total variance rises with expiry)", "violations"] == 0


def test_surface_fit_to_synthetic_quotes_recovers_the_generating_skew(market):
    d = market.dates()[150]
    c = market.chain(d)
    otm = schema.otm_quotes(c)
    otm = otm[(otm["bid"] > 0.05) & (otm["T"] > 0.05) & (otm["k"].abs() < 0.35)]
    fit = sf.fit_ssvi(otm)
    assert fit.rho == pytest.approx(market.spec.ssvi_rho, abs=0.15) and fit.arbitrage_free()["free"]
    e = sorted(c["expiry"].unique())[5]
    g = otm[otm["expiry"] == e]
    assert np.abs(np.sqrt(fit.w(g["k"].to_numpy(), float(g["T"].iloc[0])) / float(g["T"].iloc[0])) - g["iv"].to_numpy()).mean() < 0.01


def test_synthetic_market_has_a_variance_risk_premium_in_expectation():
    spec = syn.SyntheticMarketSpec()
    und = syn.simulate_underlying(spec, 30000, seed=3)
    realised = float((np.log(und["spot"]).diff() ** 2).mean() * 252)
    implied_30d = float(np.mean(syn._atm_total_variance(spec, und["var"].to_numpy(), 30 / 365) / (30 / 365)))
    assert implied_30d > realised * 1.1                                    # implied variance exceeds what is realised on average


# --------------------------------------------------------------------------------------------------------------------- backtester
def _tiny_world():
    """Three dates, one expiry, hand-set quotes: lets every cash flow be computed by hand."""
    d = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"])
    expiry = pd.Timestamp("2022-01-05")
    spots = [100.0, 102.0, 105.0]
    quotes = {0: (4.0, 4.4), 1: (5.0, 5.4), 2: (5.0, 5.0)}
    chains = {}
    for i, day in enumerate(d):
        rows = [{"date": day, "expiry": expiry, "strike": 100.0, "right": "C", "bid": quotes[i][0], "ask": quotes[i][1], "underlying": spots[i], "rate": 0.0, "dividend": 0.0, "iv": 0.2}]
        chains[day] = schema.normalise_chain(pd.DataFrame(rows)) if i < 2 else schema.normalise_chain(pd.DataFrame(rows).assign(expiry=expiry + pd.Timedelta(days=1)))
    return chains, pd.Series(spots, index=d), expiry


class _Once(bt.OptionStrategy):
    def __init__(self, orders):
        self.orders, self.done = orders, False

    def on_date(self, ctx):
        if self.done:
            return []
        self.done = True
        return list(self.orders)


def test_orders_fill_on_the_next_date_at_that_dates_quotes_and_cross_the_spread():
    chains, spot, expiry = _tiny_world()
    b = bt.OptionBacktester(chains, spot, capital=1000.0, commission=1.0, execution_lag=1, fill_fraction=1.0)
    res = b.run(_Once([bt.Order(expiry, 100.0, "C", 1.0)]))
    trade = res.trades.iloc[0]
    assert trade["date"] == pd.Timestamp("2022-01-04") and trade["price"] == pytest.approx(5.4) and trade["cost"] == 1.0           # day-1 ask, not day-0 ask
    assert res.equity.iloc[0] == 0.0 and res.equity.iloc[1] == pytest.approx(100 * (5.2 - 5.4) - 1.0)                             # marked at the mid 5.2 after paying the ask 5.4
    assert res.trades.shape[0] == 1 and res.skipped_orders == 0
    mid = bt.OptionBacktester(chains, spot, capital=1000.0, commission=0.0, fill_fraction=0.0).run(_Once([bt.Order(expiry, 100.0, "C", 1.0)]))
    assert mid.trades.iloc[0]["price"] == pytest.approx(5.2) and mid.equity.iloc[1] == pytest.approx(0.0, abs=1e-9)
    same_day = bt.OptionBacktester(chains, spot, capital=1000.0, commission=0.0, execution_lag=0).run(_Once([bt.Order(expiry, 100.0, "C", 1.0)]))
    assert same_day.trades.iloc[0]["date"] == pd.Timestamp("2022-01-03") and same_day.trades.iloc[0]["price"] == pytest.approx(4.4)


def test_expiry_settles_at_intrinsic_value_and_unquoted_orders_are_skipped():
    chains, spot, expiry = _tiny_world()
    res = bt.OptionBacktester(chains, spot, capital=1000.0, commission=0.0, execution_lag=0).run(_Once([bt.Order(expiry, 100.0, "C", 1.0)]))
    # bought at the day-0 ask 4.4; on day 2 (expiry) settled at intrinsic 105 - 100 = 5: P&L = 100 * (5 - 4.4)
    assert res.equity.iloc[-1] == pytest.approx(100 * (5.0 - 4.4)) and res.positions["open_contracts"].iloc[-1] == 0
    short = bt.OptionBacktester(chains, spot, capital=1000.0, commission=0.0, execution_lag=0, fill_fraction=1.0).run(_Once([bt.Order(expiry, 100.0, "C", -1.0)]))
    assert short.equity.iloc[-1] == pytest.approx(100 * (4.0 - 5.0))
    missing = bt.OptionBacktester(chains, spot, capital=1000.0).run(_Once([bt.Order(expiry, 999.0, "C", 1.0)]))
    assert missing.skipped_orders == 1 and missing.trades.empty


def test_hedge_orders_trade_the_underlying_and_pay_the_hedge_cost():
    chains, spot, expiry = _tiny_world()
    res = bt.OptionBacktester(chains, spot, capital=1000.0, commission=0.0, hedge_cost_bps=10.0, execution_lag=0).run(_Once([bt.Hedge(10.0)]))
    assert res.trades.iloc[0]["kind"] == "hedge" and res.costs.iloc[0] == pytest.approx(10 * 100.0 * 10 / 1e4)
    assert res.equity.iloc[-1] == pytest.approx(10 * (105.0 - 100.0) - 1.0)                       # 10 shares gained 5 each, less the one-off cost
    assert res.positions["underlying_units"].iloc[-1] == 10.0


def test_backtester_validates_arguments():
    chains, spot, _ = _tiny_world()
    for kwargs in ({"capital": 0}, {"capital": 1, "fill_fraction": 1.5}, {"capital": 1, "execution_lag": -1}, {"capital": 1, "multiplier": 0}):
        with pytest.raises(ValueError):
            bt.OptionBacktester(chains, spot, **kwargs)


@pytest.fixture(scope="module")
def backtester(market):
    return bt.OptionBacktester({d: market.chain(d) for d in market.dates()}, market.underlying["spot"], capital=10000.0)


@pytest.mark.parametrize("name", sorted(st.STRATEGIES))
def test_every_strategy_runs_without_skipped_orders_and_the_accounting_identity_holds(backtester, name):
    res = backtester.run(st.make_strategy(name))
    assert res.skipped_orders == 0 and len(res.trades) > 0 and np.isfinite(res.equity).all()
    attr = res.attribution.drop(columns=[c for c in res.attribution if c == "costs"])
    total = res.attribution.sum(axis=1)
    assert total.to_numpy() == pytest.approx(res.pnl.reindex(total.index).to_numpy(), abs=1e-6)     # delta+gamma+vega+theta+hedge+unexplained+slippage+costs = P&L, every day
    assert attr.shape[0] == len(res.pnl) - 1
    s = res.summary()
    assert np.isfinite(s["sharpe"]) and s["costs_total"] > 0 and s["n_trades"] == len(res.trades)


def test_greek_attribution_explains_most_of_a_hedged_straddles_pnl(backtester):
    res = backtester.run(st.make_strategy("delta_hedged_straddle", side=-1))
    a = res.attribution
    assert abs(a["unexplained"]).sum() < 0.35 * (abs(a["gamma"]) + abs(a["theta"]) + abs(a["vega"])).sum()
    held = res.positions["open_contracts"].shift(1).reindex(a.index) > 0
    assert a.loc[held, "gamma"].sum() < 0 < a.loc[held, "theta"].sum()                         # short gamma pays theta; the loss from moving is the price
    assert res.greeks["delta"].abs().iloc[5:].median() < 15.0                                      # hedged daily with a one-day execution lag: net delta stays in the tens of shares, not hundreds


def test_strategy_positions_have_the_documented_shape(backtester):
    put = backtester.run(st.make_strategy("short_put", size=2))
    assert (put.trades[put.trades["kind"] == "option"]["right"] == "P").all() and put.positions["open_contracts"].max() == 2
    cc = backtester.run(st.make_strategy("covered_call"))
    assert cc.positions["underlying_units"].max() == 100.0 and cc.greeks["delta"].max() < 100.0 and cc.greeks["delta"].iloc[5:].min() > 0
    ic = backtester.run(st.make_strategy("iron_condor"))
    assert ic.positions["open_contracts"].max() == 4
    strangle = backtester.run(st.make_strategy("short_strangle"))
    assert ic.summary()["worst_day"] >= strangle.summary()["worst_day"] - 1e-9                    # the long wings cap the loss on a large move
    ds = backtester.run(st.make_strategy("delta_hedged_straddle", side=1))
    held = ds.positions["open_contracts"] > 0
    assert (ds.greeks.loc[held, "gamma"] > 0).all() and (ds.greeks.loc[held, "vega"] > 0).all()   # long volatility: long gamma and long vega while the straddle is open
    with pytest.raises(ValueError):
        st.make_strategy("delta_hedged_straddle", side=0)
    with pytest.raises(KeyError):
        st.make_strategy("nope")
    with pytest.raises(ValueError):
        st.make_strategy("short_put", dte=1)


def test_vrp_timing_trades_a_subset_of_the_time(backtester):
    inner = st.make_strategy("short_strangle")
    timed = st.VRPTimed(inner, window=21, threshold=0.02)
    res_inner, res_timed = backtester.run(st.make_strategy("short_strangle")), backtester.run(timed)
    assert res_timed.positions["open_contracts"].gt(0).mean() <= res_inner.positions["open_contracts"].gt(0).mean() + 1e-12
    assert timed.name == "vrp_timed_short_strangle"


def test_long_and_short_straddles_are_mirror_images_up_to_costs():
    spec = syn.SyntheticMarketSpec(jump_intensity=0.0)
    calm = syn.generate_market(spec, n_days=160, seed=11, warm=250)
    chains = {d: calm.chain(d) for d in calm.dates()}
    short = bt.OptionBacktester(chains, calm.underlying["spot"], capital=10000).run(st.make_strategy("delta_hedged_straddle", side=-1))
    long = bt.OptionBacktester(chains, calm.underlying["spot"], capital=10000).run(st.make_strategy("delta_hedged_straddle", side=1))
    drag = sum(r.attribution[["costs", "slippage"]].sum().sum() for r in (short, long))
    assert short.pnl.sum() + long.pnl.sum() == pytest.approx(drag, abs=1e-6) and drag < 0           # positions are exact opposites, so only the spread and fees do not cancel
    assert np.allclose(short.attribution["gamma"].to_numpy(), -long.attribution["gamma"].to_numpy()) and np.allclose(short.attribution["theta"].to_numpy(), -long.attribution["theta"].to_numpy())


# ------------------------------------------------------------------------------------------------------- volatility analytics
def test_variance_risk_premium_frame_and_term_structure(market):
    v = vol.vix_series(market)
    assert len(v) > 250 and v.between(5, 90).all()
    vr = vol.variance_risk_premium(v, market.underlying["spot"])
    assert {"implied_var", "trailing_rv", "forward_rv", "vrp_trailing", "vrp_forward"} <= set(vr)
    probe = vr.dropna().index[10]
    assert vr.loc[probe, "vrp_trailing"] == pytest.approx(vr.loc[probe, "implied_var"] - vol.realised_variance(market.underlying["spot"]).loc[probe])
    assert vol.realised_variance(market.underlying["spot"], 21, forward=True).iloc[-1] != vol.realised_variance(market.underlying["spot"], 21, forward=True).iloc[-1]    # NaN: no future
    ts = vol.term_structure(market.chain(market.dates()[100]))
    assert ts.index.is_monotonic_increasing and ts.iloc[-1] > 0.05
    ss = vol.smile_summary(market.chain(market.dates()[100]), sorted(market.chain(market.dates()[100])["expiry"].unique())[3])
    assert ss["skew_slope"] < 0 and ss["svi_rmse"] < 0.02


def test_dispersion_trade_earns_the_implied_correlation_premium():
    assert vol.implied_correlation(0.3 * np.sqrt(0.1 + 0.9 * 0.5), [0.3] * 10, np.full(10, 0.1)) == pytest.approx(0.5)
    means = {}
    for premium in (0.0, 0.10):
        w = vol.simulate_dispersion_world(implied_corr_premium=premium, n_days=6000, seed=3)
        r = vol.dispersion_backtest(w, cost_vol_points=0.0)
        means[premium] = (r["pnl"].mean(), r["pnl"].std() / np.sqrt(len(r)))
        assert r["implied_corr"].mean() - r["realised_corr"].mean() == pytest.approx(premium, abs=0.05)
    assert abs(means[0.0][0]) < 3 * means[0.0][1] and means[0.10][0] > 0.01 and means[0.10][0] > means[0.0][0] + 5 * means[0.10][1]
    w = vol.simulate_dispersion_world(implied_corr_premium=0.10, n_days=3000, seed=3)
    assert vol.dispersion_backtest(w, cost_vol_points=1.0)["pnl"].mean() < vol.dispersion_backtest(w, cost_vol_points=0.0)["pnl"].mean()
    assert vol.dispersion_backtest(w, ignore_premium=True, cost_vol_points=0.0)["pnl"].mean() < vol.dispersion_backtest(w, cost_vol_points=0.0)["pnl"].mean()
