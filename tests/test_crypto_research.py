"""Crypto research: 24/7 calendar, observed-only gating, carry decomposition, depth-aware execution, funding intelligence, liquidation risk, strategy validation (no network)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.assets import crypto_research as cr
from src.assets import crypto_validation as cv
from src.assets.crypto_analytics import CryptoSeries


def series(n=400, seed=1, source="downloaded", venue="OKX", drift=0.0003, basis=0.001, funding=0.0003, sym="BTC"):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    spot = pd.Series(100 * np.cumprod(1 + np.random.default_rng(seed).normal(drift, 0.02, n)), index=idx)
    wobble = 0.0005 * np.sin(np.arange(n) / 9.0)
    return CryptoSeries(sym, spot, spot * (1 + basis + wobble), pd.Series(funding, index=idx), source, funding_venue=venue)


def flat(n=60, funding=0.0003, venue="OKX", source="downloaded"):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return CryptoSeries("BTC", pd.Series(100.0, index=idx), pd.Series(101.0, index=idx), pd.Series(funding, index=idx), source, funding_venue=venue)


def test_the_calendar_audit_tells_a_weekday_index_from_a_24_7_one_and_annualises_by_365():
    daily = cr.calendar_audit(pd.date_range("2024-01-01", periods=200, freq="D"))
    week = cr.calendar_audit(pd.bdate_range("2024-01-01", periods=200))
    assert daily["is_24_7"] and daily["periods_per_year"] == 365 and daily["missing_days"] == 0
    assert not week["is_24_7"] and week["weekend_marks"] == 0 and week["coverage"] == pytest.approx(5 / 7, abs=0.01) and week["weekday_pipeline"]["periods_per_year"] == 252
    r = pd.Series(0.001 + 0.01 * np.sin(np.arange(100)))
    a = cr.annualise(r)
    assert a["ann_return"] == pytest.approx(r.mean() * 365) and a["ann_vol"] == pytest.approx(r.std(ddof=1) * 365 ** 0.5)


def test_a_flat_market_carry_earns_exactly_its_funding_less_costs_and_the_components_add_up():
    out = cr.carry_pnl(flat(60), notional=1e6, leverage=4.0, fee_bps=5, slippage_bps=2, financing_apr=0.05)
    p = out["pnl"]
    n = out["days"]
    assert n == 59 and p["funding_received"] == pytest.approx(1e6 * 1.0 * 0.0003 * n * (101 / 101), rel=1e-9)   # qp * P * F per day = N * F
    assert p["basis_change"] == pytest.approx(0.0, abs=1e-6) and p["unhedged_delta"] == pytest.approx(0.0, abs=1e-6)
    assert p["trading_costs"] == pytest.approx(-(7e-4) * 4 * 1e6 + 0.0, rel=0.02)                                 # four fills of about the notional
    assert p["financing"] == pytest.approx(-1.25e6 * 0.05 * n / 365)
    assert p["total"] == pytest.approx(sum(v for k, v in p.items() if k in ("funding_received", "basis_change", "unhedged_delta", "trading_costs", "financing", "collateral_revaluation", "residual")))
    assert abs(p["residual"]) < 1e-6 and out["margin"]["margin_calls"] == 0 and out["funding"]["negative_days"] == 0
    assert out["funding"]["breakeven_days"] == pytest.approx(-p["trading_costs"] / (1e6 * 0.0003), rel=1e-6)


def test_carry_margin_calls_and_negative_funding_runs_are_counted_and_simulated_data_is_refused():
    idx = pd.date_range("2023-01-01", periods=40, freq="D")
    perp = pd.Series(100.0, index=idx)
    perp.iloc[20:] = 140.0                                                                                       # the short loses 40% of its notional on a 3x margin
    f = pd.Series(0.0003, index=idx)
    f.iloc[5:9] = -0.001
    s = CryptoSeries("BTC", perp.copy(), perp, f, "downloaded", funding_venue="OKX")
    out = cr.carry_pnl(s, leverage=3.0)
    assert out["margin"]["margin_calls"] > 0 and out["margin"]["first_breach_day"] == str(idx[20].date()) and out["margin"]["min_ratio"] < 1
    assert out["funding"]["longest_negative_run"] == 4 and out["funding"]["negative_days"] == 4
    assert out["margin"]["liquidation_move_from_entry"] == pytest.approx((1 / 3 + 1.0) / 1.004 - 1.0)
    sim = cr.carry_pnl(CryptoSeries("SOL", s.spot, s.perp, s.funding, "simulated"))
    assert sim["state"] == "unavailable" and "simulated" in sim["reason"]


def test_an_inverse_perpetual_shows_the_collateral_revaluation_and_its_own_residual():
    s = series(300, venue="DERIBIT")
    out = cr.carry_pnl(s, leverage=3.0)
    p = out["pnl"]
    assert out["perp_type"] == "inverse" and p["collateral_revaluation"] == pytest.approx(1e6 / 3.0 / s.spot.iloc[0] * (s.spot.iloc[-1] - s.spot.iloc[0]), rel=1e-9)
    assert "collateral revaluation" in out["residual_note"] and p["total"] == pytest.approx(sum(p[k] for k in ("funding_received", "basis_change", "unhedged_delta", "trading_costs", "financing", "collateral_revaluation", "residual")))


def test_walking_a_book_is_exact_and_reports_what_stays_unfilled():
    asks = [[101.0, 1.0], [102.0, 2.0], [103.0, 5.0]]
    w = cr.walk_book(asks, notional=101 + 102 * 2)
    assert w["coins"] == pytest.approx(3.0) and w["vwap"] == pytest.approx((101 + 204) / 3) and w["levels_used"] == 2 and w["unfilled_dollars"] == pytest.approx(0)
    short = cr.walk_book(asks[:2], notional=10_000)
    assert short["unfilled_dollars"] == pytest.approx(10_000 - 305)
    q = cr.walk_book(asks, qty=2.5)
    assert q["coins"] == pytest.approx(2.5) and q["dollars"] == pytest.approx(101 + 1.5 * 102)
    with pytest.raises(ValueError):
        cr.walk_book(asks)


def snapshot(age_s=2, ask_b=100.0, bid_b=101.5):
    t = (datetime.now(timezone.utc)).replace(microsecond=0)
    stamp = (t - pd.Timedelta(seconds=age_s)).isoformat().replace("+00:00", "Z")
    a = {"venue": "KRAKEN", "label": "Kraken", "kind": "spot", "pair": "BTC/USD", "quote_asset": "USD", "taker_bps": 10.0, "bids": [[ask_b - 0.1, 5.0]], "asks": [[ask_b, 1.0], [ask_b + 1.0, 4.0]]}
    b = {"venue": "COINBASE", "label": "Coinbase", "kind": "spot", "pair": "BTC/USD", "quote_asset": "USD", "taker_bps": 20.0, "bids": [[bid_b, 1.0], [bid_b - 2.0, 4.0]], "asks": [[bid_b + 0.1, 5.0]]}
    return {"coin": "BTC", "fetched_at": stamp, "venues": [a, b]}


def test_the_executable_edge_walks_both_books_and_never_calls_a_spread_confirmed():
    out = cr.executable_cross(snapshot(), notional=100.0)
    best = out["pairs"][0]
    assert (best["buy"], best["sell"]) == ("KRAKEN", "COINBASE") and best["quoted_bps"] == pytest.approx((101.5 / 100.0 - 1) * 1e4)
    assert best["fill_fraction"] == pytest.approx(1.0) and best["fees_bps"] == 30.0
    coins = 1.0                                                                                                   # $100 buys exactly the first level's one coin
    assert best["net_usd"] == pytest.approx(coins * 101.5 - 100.0 - (100.0 * 10 / 1e4 + 101.5 * 20 / 1e4))
    assert best["verdict"] == "edge_at_snapshot_not_confirmed" and "confirmed arbitrage" in out["reading"]
    big = cr.executable_cross(snapshot(), notional=2000.0)["pairs"][0]
    assert big["price_impact_bps"] > 0 and big["executable_bps"] < big["quoted_bps"] and big["fill_fraction"] < 1.0 and any("partial" in n for n in big["notes"])
    assert cr.executable_cross(snapshot(age_s=60), notional=100.0)["pairs"][0]["verdict"] == "stale"
    flat_book = cr.executable_cross(snapshot(bid_b=100.0), notional=100.0)["pairs"][0]
    assert flat_book["verdict"] == "no_edge_after_costs"
    assert cr.executable_cross(None)["state"] == "unavailable"
    with pytest.raises(ValueError):
        cr.executable_cross(snapshot(), fee_mode="x")


def test_the_execution_simulator_matches_the_book_and_labels_the_assumed_part():
    book = {"venue": "X", "taker_bps": 5.0, "bids": [[100, 1], [99, 2], [98, 5]], "asks": [[101, 1], [102, 2], [103, 5]]}
    m = cr.simulate_execution(book, "buy", 305.0)
    assert m["result"]["vwap"] == pytest.approx(305 / 3) and m["result"]["slippage_bps"] == pytest.approx((305 / 3 / 100.5 - 1) * 1e4) and m["result"]["fee_usd"] == pytest.approx(305 * 5e-4)
    lim = cr.simulate_execution(book, "buy", 400.0, "limit", limit_bps=100)
    assert lim["result"]["fill_fraction"] < 1.0 and lim["result"]["unfilled_usd"] > 0 and lim["limit_price"] == pytest.approx(100.5 * 1.01)
    never = cr.simulate_execution(book, "buy", 400.0, "twap", slices=4, replenish=0.0)
    full = cr.simulate_execution(book, "buy", 400.0, "twap", slices=4, replenish=1.0)
    assert full["result"]["slippage_bps"] < never["result"]["slippage_bps"] and any("assumption" in a for a in full["assumptions"])
    one = cr.simulate_execution(book, "sell", 150.0, "twap", slices=1)
    assert one["result"]["vwap"] == pytest.approx(cr.simulate_execution(book, "sell", 150.0)["result"]["vwap"])
    assert cr.simulate_execution(book, "buy", 200.0, "vwap", slices=3, profile="u_shape", vol_annual=0.6, horizon_hours=6)["result"]["timing_risk_bps_1sd"] == pytest.approx(0.6 * (6 / (24 * 365)) ** 0.5 * 1e4)
    for bad in (("hold", 100.0, "market"), ("buy", -1.0, "market"), ("buy", 100.0, "iceberg")):
        with pytest.raises(ValueError):
            cr.simulate_execution(book, *bad)


def test_funding_intelligence_hides_simulated_venues_in_observed_only_mode_and_never_invents_open_interest():
    s = series(500)
    obs = cr.funding_intelligence({"BTC": s}, observed_only=True)
    assert {r["venue"] for r in obs["rows"]} == {"OKX"} and all(r["source"] == "downloaded" for r in obs["rows"]) and {u["venue"] for u in obs["unavailable_rows"]} == {"DERIBIT", "BINANCE", "BYBIT"}
    row = obs["rows"][0]
    assert row["annualised"] == pytest.approx(0.0003 * 365) and row["rate_per_interval"] == pytest.approx(0.0003 * 8 / 24) and row["calendar"]["settlement_hours_utc"] == [0, 8, 16]
    for k in ("open_interest", "liquidations", "dated_futures"):
        assert obs[k]["state"] == "unavailable" and obs[k]["needs"]
    sim = cr.funding_intelligence({"BTC": s}, observed_only=False)
    assert {r["source"] for r in sim["rows"]} == {"downloaded", "simulated"} and sim["dispersion"]["BTC"]["dispersion_annualised"] is not None
    assert cr.funding_regimes(pd.Series(0.0003, index=pd.date_range("2024-01-01", periods=100)))["current"] == "crowded"


def test_liquidation_risk_of_a_levered_long_matches_the_closed_form():
    out = cr.portfolio_liquidation([{"symbol": "BTC", "venue": "BINANCE", "side": "long", "notional": 2e5}], 1e5)
    mmr = 0.004
    assert out["gross_leverage"] == pytest.approx(2.0) and out["liquidation_shock_down"] == pytest.approx((mmr * 2e5 - 1e5) / (2e5 * (1 - mmr)), abs=2e-3)
    assert out["liquidation_shock_up"] is None and out["positions"][0]["isolated"]["liquidation_move"] == pytest.approx((mmr - 0.5) / (1 - mmr))
    short = cr.portfolio_liquidation([{"symbol": "BTC", "venue": "BINANCE", "side": "short", "notional": 2e5}], 1e5)
    assert short["liquidation_shock_up"] == pytest.approx((0.5 - mmr) / (1 + mmr), abs=2e-3) and short["liquidation_shock_down"] is None
    hedged = cr.portfolio_liquidation([{"symbol": "BTC", "venue": "BINANCE", "side": "long", "notional": 2e5}, {"symbol": "BTC", "venue": "OKX", "side": "short", "notional": 2e5}], 1e5)
    assert hedged["net_leverage"] == pytest.approx(0.0) and hedged["liquidation_shock_down"] is None
    for bad in ([{"symbol": "X", "venue": "NOPE", "side": "long", "notional": 1}], [{"symbol": "X", "venue": "OKX", "side": "flat", "notional": 1}], [{"symbol": "X", "venue": "OKX", "side": "long", "notional": 0}]):
        with pytest.raises(ValueError):
            cr.portfolio_liquidation(bad, 100.0)


def panel_series():
    return {"BTC": series(700, seed=1), "ETH": series(700, seed=2, sym="ETH", drift=0.0004)}


def test_strategies_trade_only_the_next_day_and_annualise_by_365():
    ser = panel_series()
    out = cv.run(ser, "tsmom", walk=False)
    net = out["net"]
    assert out["annualisation"] == 365 and out["calendar"]["is_24_7"]
    assert net["sharpe"] == pytest.approx(net["ann_return"] / net["ann_vol"], rel=1e-9)
    changed = {k: CryptoSeries(v.symbol, v.spot.copy(), v.perp, v.funding, v.source, funding_venue=v.funding_venue) for k, v in ser.items()}
    changed["BTC"].spot.iloc[-1] *= 2.0                                                                          # a jump on the last day must not touch any earlier return
    a, b = cv.run(ser, "tsmom", walk=False), cv.run(changed, "tsmom", walk=False)
    assert a["series"]["equity"][:-2] == b["series"]["equity"][:-2]


def test_validation_refuses_simulated_coins_and_says_what_a_strategy_needs():
    sim = {"BTC": series(300, source="simulated")}
    assert cv.run(sim, "tsmom")["state"] == "unavailable"
    ok = cv.run({**sim, **{"BTC": series(300)}}, "pair_spread")
    assert ok["state"] == "unavailable" and "BTC and ETH" in ok["reason"]
    assert cv.run({"BTC": series(300)}, "xs_momentum")["state"] == "unavailable"
    assert cv.run({"BTC": series(300)}, "funding_dispersion")["state"] == "unavailable"
    with pytest.raises(ValueError):
        cv.run({"BTC": series(300)}, "nope")


def test_every_family_runs_with_costs_benchmarks_and_a_walk_forward():
    ser = {**panel_series(), "SOL": series(700, seed=3, sym="SOL")}
    vf = {"A": ser["BTC"].funding, "B": ser["BTC"].funding * 0.2 + pd.Series(0.0001 * np.cos(np.arange(700) / 20), index=ser["BTC"].funding.index)}
    for name in cv.FAMILIES:
        out = cv.run(ser, name, venue_funding=vf, bootstrap_n=60, walk=(name != "xs_momentum"))
        assert out["state"] == "ok", (name, out)
        assert out["net"]["n_days"] > 300 and "hold BTC" in out["benchmarks"] and out["costs"]["cost_drag_per_year"] >= 0 and out["verdict"]["reading"]
        assert out["gross"]["total_return"] >= out["net"]["total_return"] - 1e-9
    wf = cv.run(ser, "tsmom", bootstrap_n=60)["walk_forward"]
    assert wf["folds"] and wf["oos_days"] > 0 and wf["n_settings"] == 5
    capped = cv.run(ser, "tsmom", target_vol=0.2, max_leverage=1.0, walk=False, bootstrap_n=30)
    assert capped["constraints"]["target_vol"] == 0.2


def test_funding_of_different_settlement_intervals_is_normalised_to_one_daily_and_annual_basis():
    s = series(120, funding=0.0003)
    out = cr.funding_intelligence({"BTC": s}, observed_only=False)
    rows = {r["venue"]: r for r in out["rows"] if r["symbol"] == "BTC"}
    assert rows["OKX"]["annualised"] == pytest.approx(0.0003 * 365)
    for v, r in rows.items():
        assert r["rate_per_interval"] == pytest.approx(r["annualised"] / 365 * r["interval_hours"] / 24)
    assert cr.PERIODS == 365.0 and "365" in out["normalisation"]


def test_risk_tiers_raise_the_maintenance_requirement_with_size_and_haircuts_cut_the_collateral():
    pos = [{"symbol": "BTC", "venue": "BINANCE", "side": "long", "notional": 2e5}]
    big = [{"symbol": "BTC", "venue": "BINANCE", "side": "long", "notional": 6e5}]
    flat = cr.portfolio_liquidation(big, 3e5)
    tiered = cr.portfolio_liquidation(big, 3e5, use_tiers=True)
    assert cr.tier_rate("BINANCE", 2e5) == 0.005 and cr.tier_rate("BINANCE", 3e7) == 0.10
    assert tiered["liquidation_shock_down"] > flat["liquidation_shock_down"] and tiered["approximate"] and "illustrative" in tiered["margin_model"]
    cut = cr.portfolio_liquidation(pos, 1e5, collateral_haircut=0.2)
    assert cut["liquidation_shock_down"] == pytest.approx((0.004 * 2e5 - 0.8e5) / (2e5 * (1 - 0.004)), abs=2e-3)
    custom = cr.portfolio_liquidation(pos, 1e5, tiers={"BINANCE": [(float("inf"), 0.02, 20)]})
    assert custom["liquidation_shock_down"] == pytest.approx((0.02 * 2e5 - 1e5) / (2e5 * 0.98), abs=2e-3)


def test_the_dated_futures_curve_annualises_the_basis_per_contract():
    from datetime import timedelta

    now = datetime.now(timezone.utc).replace(microsecond=0)
    exp = (now + timedelta(days=90)).date()
    payload = {"venue": "DERIBIT", "fetched_at": now.isoformat().replace("+00:00", "Z"), "contracts": [{"instrument": "BTC-X", "expiry": exp.isoformat(), "mark_price": 51_000.0, "underlying_price": 50_000.0, "open_interest": 10.0, "volume_usd": 1.0}]}
    out = cr.futures_term_structure(payload, now)
    row = out["contracts"][0]
    assert row["basis"] == pytest.approx(0.02) and row["annualised_basis"] == pytest.approx(0.02 * 365 / row["days"]) and 89 < row["days"] < 91
    assert cr.futures_term_structure(None)["state"] == "unavailable"
