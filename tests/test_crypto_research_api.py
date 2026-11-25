"""The crypto research routes and the observed-only data mode of the options and crypto endpoints (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets.free_data.store import DownloadStore
from src.webapp.api import ApiError
from tests.test_crypto_research import snapshot
from tests.test_webapp_assets_ml import _app, _safe, finished, wait
from tests.test_webapp_labs import _rich


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    root = tmp_path_factory.mktemp("cr")
    d = root / "data" / "user" / "crypto"
    d.mkdir(parents=True)
    n = 700
    t = pd.date_range("2023-01-01", periods=n, freq="D")
    for sym, seed in (("BTC", 1), ("ETH", 2)):
        spot = 100 * np.cumprod(1 + np.random.default_rng(seed).normal(0.0004, 0.02, n))
        pd.DataFrame({"time": t, "spot": spot, "perp": spot * (1.001 + 0.0005 * np.sin(np.arange(n) / 9)), "funding_daily": 0.0002 + 0.0001 * np.sin(np.arange(n) / 40)}).to_csv(d / f"{sym}.csv", index=False)
    DownloadStore(root).save_json("crypto", "venues_BTC", snapshot(), {"source": "test", "as_of": "2026-01-01"})
    return _app(root, _rich, root=root)


def get(app, path, **q):
    return _safe(app.extra_get[path]({"data_mode": "observed", **q}))


def test_the_market_overview_gives_percentiles_and_says_what_is_not_stored(app):
    out = get(app, "/api/crypto/market", symbol="BTC", days="365")
    tiles = {t["id"]: t for t in out["tiles"]}
    assert tiles["spot"]["state"] == "ok" and 0 <= tiles["realized_vol"]["percentile"] <= 1 and tiles["volume"]["state"] == "unavailable" and tiles["open_interest"]["state"] == "unavailable"
    assert out["regime"]["volatility"] in ("high", "normal", "low") and out["sources"]["volume"] == "unavailable"
    cal = get(app, "/api/crypto/calendar", symbol="BTC")["audit"]
    assert cal["is_24_7"] and cal["periods_per_year"] == 365


def test_derivatives_intelligence_shows_observed_venues_only_and_lists_the_missing_feeds(app):
    out = get(app, "/api/crypto/derivatives", symbols="BTC,ETH")
    assert {r["venue"] for r in out["rows"]} == {"DERIBIT"} or all(r["source"] == "user_file" for r in out["rows"])
    assert out["open_interest"]["state"] == "unavailable" and out["liquidations"]["state"] == "unavailable" and out["unavailable_rows"]
    row = out["rows"][0]
    assert row["annualised"] == pytest.approx(row["rate_per_interval"] * 3 * 365 / 1.0, rel=1e-6) or row["interval_hours"] in (1, 4, 8)


def test_carry_executable_and_execution_routes(app):
    carry = get(app, "/api/crypto/carry", symbol="BTC", leverage="3", notional="1000000")
    p = carry["pnl"]
    assert p["total"] == pytest.approx(sum(p[k] for k in ("funding_received", "basis_change", "unhedged_delta", "trading_costs", "financing", "collateral_revaluation", "residual")))
    ex = get(app, "/api/crypto/executable", symbol="BTC", notional="100")
    assert ex["pairs"][0]["verdict"] in ("edge_at_snapshot_not_confirmed", "stale", "no_edge_after_costs") and "confirmed arbitrage" in ex["reading"]
    none = get(app, "/api/crypto/executable", symbol="ETH")
    assert none["state"] == "unavailable"
    sim = get(app, "/api/crypto/execution", symbol="BTC", venue="KRAKEN", side="buy", notional="150", order_type="twap", slices="2", replenish="0.5")
    assert sim["result"]["fill_fraction"] > 0 and sim["venues"] == ["KRAKEN", "COINBASE"] and sim["assumptions"]
    with pytest.raises(ApiError, match="venue must be"):
        get(app, "/api/crypto/execution", symbol="BTC", venue="NOPE")


def test_validation_job_runs_on_observed_files_and_refuses_unknown_settings(app):
    out = finished(app, app.extra_post["/api/crypto/validate"]({"data_mode": "observed", "strategy": "tsmom", "assets": ["BTC", "ETH"], "params": {"lookback": 60}}))
    assert out["annualisation"] == 365 and out["params"]["lookback"] == 60 and out["walk_forward"]["folds"] and out["calendar"]["is_24_7"]
    with pytest.raises(ApiError, match="no setting"):
        app.extra_post["/api/crypto/validate"]({"strategy": "tsmom", "params": {"bogus": 1}})
    refused = wait(app, app.extra_post["/api/crypto/validate"]({"data_mode": "observed", "strategy": "xs_momentum", "assets": ["BTC", "ETH"]}))
    assert refused["status"] == "error" and "three or more" in refused["error"]


def test_liquidation_route_uses_measured_betas(app):
    out = _safe(app.extra_post["/api/crypto/liquidation"]({"collateral": 1e5, "positions": [{"symbol": "ETH", "venue": "OKX", "side": "long", "notional": 3e5}]}))
    assert "ETH" in out["betas"] and abs(out["betas"]["ETH"]) < 0.5 and out["gross_leverage"] == pytest.approx(3.0), "the two test coins are independent, so the measured beta is near zero"
    forced = _safe(app.extra_post["/api/crypto/liquidation"]({"collateral": 1e5, "betas": {"ETH": 1.0}, "positions": [{"symbol": "ETH", "venue": "OKX", "side": "long", "notional": 3e5}]}))
    assert forced["betas"]["ETH"] == 1.0 and forced["liquidation_shock_down"] is not None
    with pytest.raises(ApiError, match="collateral"):
        app.extra_post["/api/crypto/liquidation"]({"positions": [{"symbol": "ETH", "venue": "OKX", "side": "long", "notional": 1}]})


def test_observed_only_refuses_simulated_chains_curves_coins_and_synthetic_validation(app):
    for path, q in (("/api/options/chain", {"underlying": "AAA"}), ("/api/futures/curve", {"root": "CL"}), ("/api/crypto/overview", {"symbol": "SOL"})):
        with pytest.raises(ApiError, match="observed-only"):
            app.extra_get[path]({"data_mode": "observed", **q})
    with pytest.raises(ApiError, match="observed-only"):
        app.extra_post["/api/options/strategy/backtest"]({"data_mode": "observed", "strategy": "covered_call"})
    chain = _safe(app.extra_get["/api/options/chain"]({"data_mode": "model", "underlying": "AAA"}))
    assert chain["sources"]["chain"] == "simulated"
    with pytest.raises(ApiError, match="data_mode"):
        app.extra_get["/api/options/chain"]({"data_mode": "yes"})
    funding = get(app, "/api/crypto/funding", symbols="BTC")
    assert all(r["source"] != "simulated" for r in funding["rows"])
    cross = get(app, "/api/crypto/cross-exchange", symbol="ETH")
    assert cross["state"] == "unavailable" and cross["pairs"] == []


def test_liquidation_route_takes_custom_tiers_and_haircuts(app):
    pos = [{"symbol": "ETH", "venue": "OKX", "side": "long", "notional": 3e5}]
    flat = _safe(app.extra_post["/api/crypto/liquidation"]({"collateral": 1e5, "betas": {"ETH": 1.0}, "positions": pos}))
    tiered = _safe(app.extra_post["/api/crypto/liquidation"]({"collateral": 1e5, "betas": {"ETH": 1.0}, "positions": pos, "tiers": {"OKX": [[None, 0.05, 10]]}, "collateral_haircut": 0.1}))
    assert tiered["liquidation_shock_down"] > flat["liquidation_shock_down"] and tiered["approximate"] and "haircut 10%" in tiered["margin_model"]
    with pytest.raises(ApiError, match="tiers are missing"):
        app.extra_post["/api/crypto/liquidation"]({"collateral": 1e5, "positions": pos, "tiers": {"BINANCE": [[None, 0.05, 10]]}})
