"""The Crypto tab's views of a coin's price against its own history, its trend rules and its signals board, against plain loops written from their definitions; the Mayer multiple rule; and the settings
the strategies accept."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.assets import crypto_analytics as ca
from src.framework import MODELS, bundle_from_prices, load_library
from src.framework.validate import check_causality
from src.risk import performance
from src.utils.config import load_config
from src.webapp.asset_api import AssetApi

load_library()
CONFIG = load_config()
HAS_CRYPTO = (CONFIG.root / "data" / "raw" / "crypto" / "BTC_funding.csv.gz").exists()
needs_crypto = pytest.mark.skipif(not HAS_CRYPTO, reason="the Deribit crypto data is not in this checkout")


def coin(symbol: str = "X", n: int = 900, seed: int = 5, drift: float = 0.0006, vol: float = 0.03) -> ca.CryptoSeries:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    spot = pd.Series(100.0 * np.exp(np.cumsum(drift + rng.normal(0.0, vol, n))), index=idx)
    return ca.CryptoSeries(symbol, spot, None, None, "simulated")


# ------------------------------------------------------------------------------------------------------------------ the Mayer multiple rule
def test_the_mayer_multiple_holds_between_its_floor_and_cap_and_sits_out_otherwise():
    c = coin(n=600)
    prices = pd.DataFrame({"A": c.spot, "B": coin(seed=9, drift=-0.001, n=600).spot})
    b = bundle_from_prices(prices, asset_class={"A": "crypto", "B": "crypto"}, name="mayer")
    for settings in ({}, {"window": 50, "cap": 1.3, "floor": 0.9}):
        window, cap, floor = settings.get("window", 200), settings.get("cap", 2.4), settings.get("floor", 0.0)
        model = MODELS.create("mayer_multiple", **settings)
        got, multiple = model.score(b), model.multiple(b)
        for asset in prices.columns:
            p = prices[asset].to_numpy()
            for t in range(len(p)):
                if t < window - 1:
                    assert math.isnan(got[asset].iloc[t]) and math.isnan(multiple[asset].iloc[t])
                    continue
                m = p[t] / np.mean(p[t - window + 1:t + 1])
                assert multiple[asset].iloc[t] == pytest.approx(m, rel=1e-12)
                if b.investable[asset].iloc[t]:
                    assert got[asset].iloc[t] == (1.0 if floor <= m < cap else 0.0)
    assert check_causality(MODELS.create("mayer_multiple"), b, cutoff=b.index[400])["ok"]
    for bad in ({"window": 5}, {"floor": -1}, {"cap": 0.5, "floor": 0.9}, {"cap": 1.0, "floor": 1.0}):
        with pytest.raises(ValueError):
            MODELS.create("mayer_multiple", **bad)


# ------------------------------------------------------------------------------------------------------------------ the cone of volatility
def test_the_volatility_cone_is_each_windows_current_volatility_against_every_window_of_the_history():
    c = coin(n=500)
    out = ca.volatility_cone(c.spot)
    logret = np.log(c.spot.to_numpy())[1:] - np.log(c.spot.to_numpy())[:-1]
    assert [r["days"] for r in out["rows"]] == [7, 14, 30, 60, 90] and out["history_days"] == len(logret)
    for r in out["rows"]:
        h = r["days"]
        vols = np.array([np.std(logret[i - h:i], ddof=1) * math.sqrt(365.0) for i in range(h, len(logret) + 1)])
        assert r["now"] == pytest.approx(vols[-1], rel=1e-12) and r["min"] == pytest.approx(vols.min(), rel=1e-12) and r["max"] == pytest.approx(vols.max(), rel=1e-12)
        for key, q in (("p10", 0.1), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p90", 0.9)):
            assert r[key] == pytest.approx(float(np.quantile(vols, q)), rel=1e-12)
        assert r["percentile"] == pytest.approx(float((vols <= vols[-1]).mean()), rel=1e-12) and r["windows"] == len(vols)
        assert r["min"] <= r["p10"] <= r["p25"] <= r["median"] <= r["p75"] <= r["p90"] <= r["max"]
    assert [r["days"] for r in ca.volatility_cone(coin(n=100).spot)["rows"]] == [7, 14, 30]            # a short history leaves out the windows with fewer than sixty observations


# ------------------------------------------------------------------------------------------------------------------ the long average, the pair and the drawdowns
def test_the_long_average_view_is_the_multiple_of_the_price_over_its_average_with_where_today_stands():
    c = coin(n=700)
    out = ca.long_average_view(c.spot, window=200, days=300)
    p = c.spot.to_numpy()
    multiple = np.array([p[t] / np.mean(p[t - 199:t + 1]) for t in range(199, len(p))])
    assert out["multiple"] == pytest.approx([round(v, 4) for v in multiple[-300:]], abs=1e-12) and out["dates"][-1] == str(c.spot.index[-1].date()) and len(out["dates"]) == 300
    assert out["now"] == pytest.approx(multiple[-1], rel=1e-12) and out["percentile"] == pytest.approx(float((multiple <= multiple[-1]).mean()), abs=1e-12)
    assert out["share_above_cap"] == pytest.approx(float((multiple >= 2.4).mean()), abs=1e-12) and out["share_below_one"] == pytest.approx(float((multiple < 1.0).mean()), abs=1e-12)
    assert out["average"][-1] == pytest.approx(float(np.mean(p[-200:])), rel=1e-12)
    assert ca.long_average_view(coin(n=150).spot) is None


def test_the_relative_view_is_the_ratios_zscore_and_the_rolling_correlation_of_returns():
    a, b = coin("A", n=400, seed=1), coin("B", n=400, seed=2)
    out = ca.relative_view(a, b, window=60, corr_window=90, days=200)
    pa, pb = a.spot.to_numpy(), b.spot.to_numpy()
    lr = np.log(pa / pb)
    z = (lr[-1] - lr[-60:].mean()) / np.std(lr[-60:], ddof=1)
    ra, rb = np.diff(np.log(pa)), np.diff(np.log(pb))
    assert out["pair"] == "A/B" and len(out["dates"]) == 200 and out["z"][-1] == pytest.approx(round(float(z), 4), abs=1e-12) and out["z_now"] == pytest.approx(float(z), rel=1e-12)
    assert out["correlation_now"] == pytest.approx(float(np.corrcoef(ra[-90:], rb[-90:])[0, 1]), rel=1e-10) and out["ratio"][-1] == pytest.approx(round(float(pa[-1] / pb[-1]), 6), abs=1e-12)
    assert out["correlation_mean"] == pytest.approx(float(np.mean([np.corrcoef(ra[i - 90:i], rb[i - 90:i])[0, 1] for i in range(90, len(ra) + 1)])), rel=1e-9)
    assert ca.relative_view(coin("A", n=80), coin("B", n=80, seed=3)) is None


def test_the_drawdown_view_is_the_depth_below_the_previous_high_and_the_deepest_falls():
    c = coin(n=500, seed=8)
    out = ca.drawdown_view(c.spot, days=400, limit=3)
    p = c.spot.to_numpy()
    depth = p / np.maximum.accumulate(p) - 1.0
    assert out["depth"] == pytest.approx([round(v, 4) for v in depth[-400:]], abs=1e-12) and out["now"] == pytest.approx(depth[-1], rel=1e-12)
    assert out["episodes"] == performance.drawdown_episodes(c.spot.pct_change().dropna(), 3) and out["episodes"][0]["depth"] <= out["episodes"][-1]["depth"] <= 0.0


def test_the_technical_view_has_the_four_parts_and_leaves_the_pair_out_without_another_coin():
    c = coin(n=700)
    both = ca.technical_view(c, coin("Y", seed=4, n=700))
    assert set(both) == {"volatility_cone", "long_average", "drawdown", "relative"} and both["relative"]["pair"] == "X/Y"
    assert ca.technical_view(c)["relative"] is None


# ------------------------------------------------------------------------------------------------------------------ the strategies and the board
def test_every_crypto_strategy_has_a_group_its_settings_with_its_own_defaults_and_its_sources():
    infos = {k: AssetApi._crypto_strategy_info(k, v) for k, v in ca.CRYPTO_STRATEGIES.items()}
    assert len(infos) == 16 and {i["group"] for i in infos.values()} == {"Trend", "Carry and basis", "On-chain", "Risk management"}
    assert next(p for p in infos["spot_momentum"]["params"] if p["name"] == "lookback")["default"] == 90 and next(p for p in infos["vol_breakout"]["params"] if p["name"] == "k")["default"] == 2.0
    assert next(p for p in infos["mayer"]["params"] if p["name"] == "cap")["default"] == 2.4 and infos["mayer"]["references"] and infos["ewmac"]["references"]
    assert infos["exchange_flow"]["available"] is False and infos["exchange_flow"]["params"] == [] and all(i["available"] for k, i in infos.items() if k != "exchange_flow")


@needs_crypto
def test_a_crypto_strategys_own_settings_reach_the_pipeline_and_are_checked():
    out = ca.run_crypto_strategy(CONFIG, "mayer", ["BTC"], "weekly", None, None, 5.0, 2.0, None, {"cap": 3.0, "floor": 1.0})
    assert out["spec"]["models"][0]["params"] == {"cap": 3.0, "floor": 1.0} and out["label"] == "Mayer multiple" and out["risk"]["sortino"] is not None and out["correlation"] is None
    both = ca.run_crypto_strategy(CONFIG, "spot_momentum", ["BTC", "ETH"], "weekly", 60, None, 5.0, 2.0, None, {"lookback": 120})
    assert both["spec"]["models"][0]["params"] == {"lookback": 120} and both["correlation"]["assets"] == ["BTC", "ETH"]
    with pytest.raises(ValueError, match="Mayer multiple.*cap"):
        ca.run_crypto_strategy(CONFIG, "mayer", ["BTC"], None, None, None, 5.0, 2.0, None, {"cap": 0.5, "floor": 0.9})
    with pytest.raises(ValueError, match="Mayer multiple"):
        ca.run_crypto_strategy(CONFIG, "mayer", ["BTC"], None, None, None, 5.0, 2.0, None, {"nonsense": 1})


@needs_crypto
def test_the_crypto_board_reads_the_carry_rules_on_the_coins_carry_index_and_shows_them_under_the_coin():
    out = ca.run_crypto_signals(CONFIG)
    assert out["assets"] == ["BTC", "ETH"] and out["source"] == "platform" and [r["id"] for r in out["signals"]] == [name for _, name, _ in ca.CRYPTO_SIGNAL_BOARD]
    from src.framework.crypto_bundle import load_crypto_bundle

    bundle = load_crypto_bundle(CONFIG)
    funding = next(r for r in out["signals"] if r["id"] == "funding_carry")
    score = MODELS.create("funding_carry").score(bundle).where(bundle.investable)
    live = score.dropna(how="all").iloc[-1]
    assert funding["score"]["BTC"] == pytest.approx(float(live["BTC_CARRY"]), rel=1e-12) and funding["score"]["ETH"] == pytest.approx(float(live["ETH_CARRY"]), rel=1e-12)
    trend = next(r for r in out["signals"] if r["id"] == "tsmom")
    spot = MODELS.create("tsmom").score(bundle).where(bundle.investable).dropna(how="all").iloc[-1]
    assert trend["score"]["BTC"] == pytest.approx(float(spot["BTC"]), rel=1e-12)
    shown = [r for r in out["signals"] if "strength" in r]
    for coin_name, counts in out["agreement"].items():
        assert counts["long"] + counts["short"] + counts["flat"] == len(shown) and counts["long"] == sum(1 for r in shown if (r["strength"][coin_name] or 0.0) > 0.25)
