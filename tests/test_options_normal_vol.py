"""The Bachelier (normal) model in the Options tab's smile: the normal implied volatility of each quote is the volatility in price units that reproduces its mid, it recovers a known normal volatility
exactly, it is close to the lognormal volatility times the forward at the money, and the smile of an expiry carries both so the page can show either."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.assets import options_analytics as oa
from src.derivatives.pricing import bachelier_price


def chain_from_bachelier(sigma_n: float = 20.0, F: float = 100.0, T: float = 0.25, r: float = 0.03) -> pd.DataFrame:
    """A one-expiry chain whose mids are Bachelier prices with one normal volatility."""
    strikes = np.arange(80.0, 121.0, 2.5)
    rows = []
    for K in strikes:
        right = "P" if K <= F else "C"
        price = float(bachelier_price(F, K, T, r, sigma_n, call=right == "C"))
        rows.append({"strike": K, "right": right, "mid": price, "bid": price * 0.99, "ask": price * 1.01, "F": F, "T": T, "rate": r, "underlying": F * np.exp(-r * T), "k": np.log(K / F)})
    return pd.DataFrame(rows)


def test_a_known_normal_volatility_is_recovered_at_every_strike():
    chain = chain_from_bachelier(20.0)
    got = oa.normal_vols(chain)
    assert np.allclose(got, 20.0, atol=1e-6)
    assert oa._spread(got) == pytest.approx(0.0, abs=1e-9)


def test_a_quote_with_no_time_value_has_no_normal_volatility():
    chain = chain_from_bachelier(20.0).iloc[:2].copy()
    disc = np.exp(-0.03 * 0.25)
    chain["mid"] = [disc * max(K - 100.0, 0.0) for K in chain["strike"]]                              # a put's intrinsic value at a strike below the forward: nothing
    assert np.isnan(oa.normal_vols(chain)).all()


def chain_from_model(vol: float = 0.18) -> pd.DataFrame:
    return oa.prepare_chain(oa.build_model_chain(450.0, pd.Timestamp("2026-10-01"), vol))


def test_the_smile_carries_the_normal_volatility_of_every_point_and_is_plain_json():
    chain = chain_from_model()
    expiry = oa.expiry_list(chain)[3]["expiry"]
    sm = oa.smile(chain, expiry)
    assert all("normal_iv" in p for p in sm["points"]) and sum(p["normal_iv"] is not None for p in sm["points"]) >= 10
    assert set(sm["normal_vs_lognormal"]) == {"lognormal_spread", "normal_spread"}
    json.dumps(sm, allow_nan=True)
    assert all(p["normal_iv"] is None or p["normal_iv"] > 0 for p in sm["points"])


def test_at_the_money_the_normal_volatility_is_about_the_lognormal_one_times_the_forward():
    chain = chain_from_model()
    for entry in oa.expiry_list(chain)[1:5]:
        sm = oa.smile(chain, entry["expiry"])
        if np.isfinite(sm["atm_normal_iv"]) and np.isfinite(sm["atm_iv"]):
            assert sm["atm_normal_iv"] == pytest.approx(sm["atm_iv"] * sm["forward"], rel=0.02)
            assert sm["atm_normal_check"] == pytest.approx(sm["atm_iv"] * sm["forward"])


def test_a_flat_normal_market_has_a_lognormal_smile_and_the_reverse_is_not_true():
    from src.derivatives.iv import implied_vol

    chain = chain_from_bachelier(20.0)
    ln = np.array([float(implied_vol(r.mid, r.underlying, r.strike, r.T, r.rate, 0.0, r.right == "C")) for r in chain.itertuples()])
    assert oa._spread(ln) > 0.05 and oa._spread(oa.normal_vols(chain)) < 1e-6                         # the same prices: a sloped lognormal smile and a flat normal one
