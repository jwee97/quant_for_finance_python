"""Independent checks of the options mathematics: parity, finite-difference Greeks and their units, implied-volatility round trips, tree convergence, brute-force payoffs, Monte Carlo odds, margin by hand."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.assets import options_analytics as oa
from src.derivatives.iv import implied_vol
from src.derivatives.pricing import binomial_price, bsm_price

S, K, T, R, Q, SIG = 100.0, 105.0, 0.5, 0.04, 0.015, 0.25
AS_OF = pd.Timestamp("2025-01-02")


def test_put_call_parity_and_the_chains_forward_and_log_moneyness():
    c, p = float(bsm_price(S, K, T, R, Q, SIG, True)), float(bsm_price(S, K, T, R, Q, SIG, False))
    assert c - p == pytest.approx(S * math.exp(-Q * T) - K * math.exp(-R * T), abs=1e-10)
    chain = oa.prepare_chain(oa.build_model_chain(100.0, AS_OF, 0.2, rate=0.04, dividend=0.015))
    assert np.allclose(chain["F"], chain["underlying"] * np.exp((chain["rate"] - chain["dividend"]) * chain["T"]), rtol=1e-9)
    assert np.allclose(chain["k"], np.log(chain["strike"] / chain["F"]), atol=1e-12)


def test_greeks_match_finite_differences_in_their_stated_units():
    h = 1e-4
    price = lambda s=S, v=SIG, t=T: float(bsm_price(s, K, t, R, Q, v, True))                                                       # noqa: E731
    pos = [{"type": "option", "right": "C", "strike": K, "dte": T * 365, "quantity": 1.0, "iv": SIG}]
    g = oa.position_values(pos, S, R, Q, AS_OF).iloc[0]
    assert g["delta"] == pytest.approx(100 * (price(S + 0.01) - price(S - 0.01)) / 0.02, rel=1e-5)                                  # shares-equivalent of one contract
    assert g["gamma"] == pytest.approx(100 * (price(S + 0.01) - 2 * price() + price(S - 0.01)) / 1e-4, rel=1e-3)
    assert g["vega"] == pytest.approx(100 * (price(v=SIG + h) - price(v=SIG - h)) / (2 * h) / 100.0, rel=1e-5)                       # dollars per volatility POINT
    assert g["theta"] == pytest.approx(100 * (price(t=T - 1e-5) - price(t=T + 1e-5)) / 2e-5 / 365.0, rel=1e-4)                       # dollars per CALENDAR DAY, negative for a long option
    one_day = oa.position_values(pos, S, R, Q, AS_OF, days=1.0).iloc[0]["value"] - g["value"]
    assert one_day == pytest.approx(g["theta"], rel=0.02)


@pytest.mark.parametrize("call", [True, False])
def test_implied_volatility_inverts_the_price_and_rejects_prices_outside_the_bounds(call):
    for sig in (0.05, 0.2, 0.9, 2.0):
        px = float(bsm_price(S, K, T, R, Q, sig, call))
        assert float(implied_vol(px, S, K, T, R, Q, call)) == pytest.approx(sig, abs=1e-7)
    assert math.isnan(float(implied_vol(0.0, S, K, T, R, Q, call)))
    assert math.isnan(float(implied_vol(S * 2, S, K, T, R, Q, call))) if call else True


def test_the_tree_converges_to_black_scholes_and_early_exercise_only_adds_value():
    eu = float(binomial_price(S, K, T, R, Q, SIG, False, american=False, steps=1500))
    assert eu == pytest.approx(float(bsm_price(S, K, T, R, Q, SIG, False)), abs=5e-3)
    am = float(binomial_price(S, K, T, R, Q, SIG, False, american=True, steps=1500))
    assert am >= eu - 1e-9 and am - eu > 0.01, "an in-the-money put with positive rates has early-exercise value"
    amc = float(binomial_price(S, K, T, R, 0.0, SIG, True, american=True, steps=800))
    assert amc == pytest.approx(float(bsm_price(S, K, T, R, 0.0, SIG, True)), abs=1e-2), "no dividend: an American call is worth its European value"


def _grid_pnl(legs, prices, cost):
    pay = np.zeros_like(prices)
    for l in legs:
        k, qty = l["strike"], l["quantity"]
        pay += qty * 100 * (np.maximum(prices - k, 0) if l["right"] == "C" else np.maximum(k - prices, 0))
    return pay - cost


def test_a_condors_extremes_breakevens_and_odds_match_brute_force_and_monte_carlo():
    chain = oa.prepare_chain(oa.build_model_chain(100.0, AS_OF, 0.2))
    legs, meta = oa.named_structure(chain, 100.0, "iron_condor", dte=30)
    a = oa.strategy_analysis(legs, 100.0, meta["forward"], meta["atm_iv"], horizon_days=min(l["dte"] for l in legs if l["type"] == "option"))
    prices = np.linspace(0.01, 300.0, 60001)
    pnl = _grid_pnl([l for l in legs if l["type"] == "option"], prices, a["cost"])
    assert a["max_profit"] == pytest.approx(pnl.max(), abs=0.6) and a["max_loss"] == pytest.approx(pnl.min(), abs=0.6)
    bes = prices[np.flatnonzero(np.diff(np.sign(pnl)) != 0)]
    assert len(a["breakevens"]) == len(bes) and np.allclose(sorted(a["breakevens"]), sorted(bes), atol=0.2)
    rng = np.random.default_rng(0)
    H = min(l["dte"] for l in legs) / 365.0
    z = rng.standard_normal(400_000)
    end = 100.0 * 0 + meta["forward"] * 0 + 100.0 * np.exp((a["view_drift"] - 0.5 * a["view_vol"] ** 2) * H + a["view_vol"] * math.sqrt(H) * z)
    mc = float((_grid_pnl([l for l in legs if l["type"] == "option"], end, a["cost"]) > 0).mean())
    assert a["prob_profit"] == pytest.approx(mc, abs=0.015), "the probability of profit is the lognormal mass where the expiry curve is positive"


def test_margin_of_a_naked_put_follows_the_regt_style_formula_by_hand():
    legs = [{"type": "option", "right": "P", "strike": 95.0, "quantity": -2.0}]
    spot = 100.0
    otm = spot - 95.0
    expected = 2 * 100 * max(0.2 * spot - otm, 0.1 * 95.0)
    assert oa.margin_estimate(legs, spot, 0.0)["naked_requirement"] == pytest.approx(expected)
    assert oa.margin_estimate(legs, spot, 0.0, cash_secured=True)["naked_requirement"] == pytest.approx(2 * 100 * 95.0)
