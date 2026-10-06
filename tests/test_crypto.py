"""The crypto branch: data alignment, the carry asset's arithmetic, and the three models' causality (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.crypto import build_crypto_frames
from src.framework import MODELS, bundle_from_prices, load_library
from src.framework.validate import check_causality

load_library()


def _write(directory, days=400, seed=3):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2020-01-01", periods=days * 24, freq="h", tz="UTC")
    for asset, base in (("BTC", 7000.0), ("ETH", 200.0)):
        index = base * np.exp(np.cumsum(rng.normal(0, 0.002, len(hours))))
        pd.DataFrame({"time": hours, "index_price": index, "interest_1h": 1e-6, "interest_8h": 8e-6}).to_csv(directory / f"{asset}_funding.csv.gz", index=False, compression="gzip")
        days_idx = pd.date_range("2020-01-01", periods=days, freq="D", tz="UTC")
        # a perpetual that trades exactly 10 bps above the index at the 08:00 settlement ending each candle
        settle = pd.Series(index, index=hours)
        close = [settle.loc[d + pd.Timedelta(days=1, hours=8)] * 1.001 if d + pd.Timedelta(days=1, hours=8) in settle.index else np.nan for d in days_idx]
        pd.DataFrame({"time": days_idx, "close": close}).to_csv(directory / f"{asset}_perp_daily.csv", index=False)
    pd.DataFrame({"time": pd.date_range("2020-01-01", periods=days, freq="D", tz="UTC"), "supply_usd": 1e9 * (1 + 0.001 * np.arange(days))}).to_csv(directory / "stablecoin_supply.csv", index=False)


def test_frames_align_the_spot_index_with_the_perpetual_candle_and_sum_hourly_funding(tmp_path):
    _write(tmp_path)
    frames = build_crypto_frames(tmp_path)
    spot, perp, funding = frames["spot"], frames["perp"], frames["funding"]
    both = spot.index.tz_localize(None) if spot.index.tz is not None else spot.index
    day = pd.Timestamp("2020-03-10", tz="UTC")
    assert (perp.loc[day, "BTC"] / spot.loc[day, "BTC"] - 1) == pytest.approx(0.001, abs=1e-9)     # aligned: the planted basis is recovered exactly
    assert funding.loc[day, "BTC"] == pytest.approx(24e-6)


@pytest.fixture(scope="module")
def crypto_bundle():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2019-06-03", periods=1500)
    spot = 100 * np.cumprod(1 + rng.normal(0.001, 0.03, (1500, 2)), axis=0)
    carry = 100 * np.cumprod(1 + rng.normal(0.0002, 0.001, (1500, 2)), axis=0)
    prices = pd.DataFrame({"BTC": spot[:, 0], "ETH": spot[:, 1], "BTC_CARRY": carry[:, 0], "ETH_CARRY": carry[:, 1]}, index=idx)
    macro = pd.DataFrame({"FUNDING_BTC": 0.05 + 0.1 * np.sin(np.arange(1500) / 80), "FUNDING_ETH": 0.06 + 0.1 * np.cos(np.arange(1500) / 70),
                          "BASIS_BTC": rng.normal(0.001, 0.001, 1500), "BASIS_ETH": rng.normal(0.001, 0.001, 1500),
                          "STABLE_SUPPLY": 1e9 * np.exp(np.cumsum(rng.normal(0.0008, 0.002, 1500)))}, index=idx)
    return bundle_from_prices(prices, asset_class={"BTC": "crypto", "ETH": "crypto", "BTC_CARRY": "crypto_carry", "ETH_CARRY": "crypto_carry"}, macro=macro, name="crypto-test")


@pytest.mark.parametrize("name", ["funding_carry", "basis_reversion", "stablecoin_flow"])
def test_crypto_models_are_causal_and_only_score_their_own_assets(name, crypto_bundle):
    model = MODELS.create(name)
    assert check_causality(model, crypto_bundle, cutoff=crypto_bundle.index[1100])["ok"]
    score = model.score(crypto_bundle).dropna(how="all")
    assert score.abs().sum().sum() > 0
    carry_cols, spot_cols = ["BTC_CARRY", "ETH_CARRY"], ["BTC", "ETH"]
    own = carry_cols if name != "stablecoin_flow" else spot_cols
    other = [c for c in score.columns if c not in own]
    assert score[other].abs().sum().sum() == 0
